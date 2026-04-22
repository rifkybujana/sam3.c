/*
 * src/model/feature_cache_persist.c - .sam3cache file I/O
 *
 * Binary serializer/deserializer for sam3_image_bundle and
 * sam3_text_bundle. File format: 48-byte header (magic + version +
 * type + model signature) followed by a type-specific entry header
 * and a sequence of tensor records. Tensors with NULL pointers are
 * written as "not present" and restored to NULL. All integers are
 * little-endian on disk (the implementation relies on the host being
 * LE; macOS arm64 and Linux x86_64 both qualify).
 *
 * Tensor bodies are stored raw — no compression. Earlier versions
 * deflated each body with zlib; the compression throughput (~12 MB/s
 * at level 6, ~50 MB/s at level 1) made saves and in-memory spills
 * unacceptably slow on 235 MiB image bundles, so version 3 drops it
 * entirely. Bundle files are ~40% larger as a result, which is the
 * right trade for a cache whose working set is local SSD. Older
 * version-1/2 files are rejected with SAM3_EMODEL (users re-precache).
 *
 * Key types:  sam3_cache_persist_sig
 * Depends on: feature_cache_persist.h, core/tensor.h, core/alloc.h,
 *             model/graph_helpers.h, util/log.h
 * Used by:    src/sam3.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "feature_cache_persist.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "core/tensor.h"
#include "core/alloc.h"
#include "graph_helpers.h"
#include "util/log.h"

static const char SAM3_CACHE_MAGIC[8] = {'S','A','M','3','C','A','C','H'};
#define SAM3_CACHE_VERSION 3u
#define SAM3_CACHE_TYPE_IMAGE 0u
#define SAM3_CACHE_TYPE_TEXT  1u

/* --- file-header write/read --- */

struct file_hdr {
	char     magic[8];
	uint32_t version;
	uint32_t type;
	struct sam3_cache_persist_sig sig;
};

static enum sam3_error write_file_hdr(FILE *f, uint32_t type,
				      const struct sam3_cache_persist_sig *sig)
{
	struct file_hdr h;
	memcpy(h.magic, SAM3_CACHE_MAGIC, 8);
	h.version = SAM3_CACHE_VERSION;
	h.type    = type;
	h.sig     = *sig;
	if (fwrite(&h, sizeof(h), 1, f) != 1)
		return SAM3_EIO;
	return SAM3_OK;
}

static enum sam3_error read_file_hdr(FILE *f, uint32_t expected_type,
	const struct sam3_cache_persist_sig *expected_sig)
{
	struct file_hdr h;
	if (fread(&h, sizeof(h), 1, f) != 1) {
		sam3_log_error("cache_load: short read on header");
		return SAM3_EIO;
	}
	if (memcmp(h.magic, SAM3_CACHE_MAGIC, 8) != 0) {
		sam3_log_error("cache_load: bad magic");
		return SAM3_EMODEL;
	}
	if (h.version != SAM3_CACHE_VERSION) {
		sam3_log_error("cache_load: unsupported version %u",
			       h.version);
		return SAM3_EMODEL;
	}
	if (h.type != expected_type) {
		sam3_log_error("cache_load: type mismatch (%u vs %u)",
			       h.type, expected_type);
		return SAM3_EMODEL;
	}
	if (memcmp(&h.sig, expected_sig, sizeof(h.sig)) != 0) {
		sam3_log_error("cache_load: model signature mismatch — "
			       "file was saved from a different model");
		return SAM3_EMODEL;
	}
	return SAM3_OK;
}

/* --- tensor write/read --- */

struct tensor_hdr {
	int32_t  present;   /* 0 = null pointer, 1 = real tensor */
	int32_t  dtype;
	int32_t  n_dims;
	int32_t  dims[SAM3_MAX_DIMS];
	uint64_t nbytes;
};

static enum sam3_error write_tensor(FILE *f, const struct sam3_tensor *t)
{
	struct tensor_hdr h;
	memset(&h, 0, sizeof(h));
	if (!t) {
		h.present = 0;
		if (fwrite(&h, sizeof(h), 1, f) != 1)
			return SAM3_EIO;
		return SAM3_OK;
	}
	h.present = 1;
	h.dtype   = (int32_t)t->dtype;
	h.n_dims  = t->n_dims;
	for (int i = 0; i < SAM3_MAX_DIMS; i++)
		h.dims[i] = (i < t->n_dims) ? t->dims[i] : 0;
	h.nbytes  = t->nbytes;
	if (fwrite(&h, sizeof(h), 1, f) != 1)
		return SAM3_EIO;
	if (t->nbytes == 0 || !t->data)
		return SAM3_OK;
	if (fwrite(t->data, 1, t->nbytes, f) != t->nbytes)
		return SAM3_EIO;
	return SAM3_OK;
}

static enum sam3_error read_tensor(FILE *f, struct sam3_arena *arena,
				   struct sam3_tensor **out)
{
	struct tensor_hdr h;
	if (fread(&h, sizeof(h), 1, f) != 1) {
		sam3_log_error("cache_load: short read on tensor header");
		return SAM3_EIO;
	}
	if (h.present == 0) {
		*out = NULL;
		return SAM3_OK;
	}
	if (h.n_dims < 1 || h.n_dims > SAM3_MAX_DIMS) {
		sam3_log_error("cache_load: bad n_dims %d", h.n_dims);
		return SAM3_EIO;
	}
	int dims[SAM3_MAX_DIMS];
	for (int i = 0; i < h.n_dims; i++)
		dims[i] = h.dims[i];

	struct sam3_tensor *t = gh_alloc_tensor(arena,
		(enum sam3_dtype)h.dtype, h.n_dims, dims);
	if (!t) {
		sam3_log_error("cache_load: alloc_tensor failed");
		return SAM3_ENOMEM;
	}
	if (t->nbytes != h.nbytes) {
		sam3_log_error("cache_load: nbytes mismatch "
			       "(alloc=%zu, file=%llu)",
			       t->nbytes,
			       (unsigned long long)h.nbytes);
		return SAM3_EIO;
	}
	if (h.nbytes > 0 &&
	    fread(t->data, 1, h.nbytes, f) != h.nbytes) {
		sam3_log_error("cache_load: short read on tensor body");
		return SAM3_EIO;
	}
	*out = t;
	return SAM3_OK;
}

/* --- image bundle --- */

struct image_entry_hdr {
	uint64_t hash;
	int32_t  prompt_w;
	int32_t  prompt_h;
	int32_t  width;
	int32_t  height;
	uint32_t n_tensors;
	uint32_t prefix_len;
	uint8_t  prefix[SAM3_CACHE_PREFIX_BYTES];
};

enum sam3_error sam3_image_bundle_save(const char *path,
	const struct sam3_cache_persist_sig *sig,
	uint64_t hash, const uint8_t *prefix, size_t prefix_len,
	const struct sam3_image_bundle *bundle)
{
	if (!path || !sig || !bundle)
		return SAM3_EINVAL;
	if (prefix_len > SAM3_CACHE_PREFIX_BYTES)
		prefix_len = SAM3_CACHE_PREFIX_BYTES;

	FILE *f = fopen(path, "wb");
	if (!f) {
		sam3_log_error("cache_save: open %s failed", path);
		return SAM3_EIO;
	}

	enum sam3_error err = write_file_hdr(f, SAM3_CACHE_TYPE_IMAGE, sig);
	if (err != SAM3_OK)
		goto done;

	struct image_entry_hdr eh;
	memset(&eh, 0, sizeof(eh));
	eh.hash      = hash;
	eh.prompt_w  = bundle->prompt_w;
	eh.prompt_h  = bundle->prompt_h;
	eh.width     = bundle->width;
	eh.height    = bundle->height;
	eh.n_tensors = 8;
	eh.prefix_len = (uint32_t)prefix_len;
	if (prefix && prefix_len > 0)
		memcpy(eh.prefix, prefix, prefix_len);
	if (fwrite(&eh, sizeof(eh), 1, f) != 1) {
		err = SAM3_EIO;
		goto done;
	}

	const struct sam3_tensor *ts[8] = {
		bundle->image_features,
		bundle->feat_s0_nhwc,
		bundle->feat_s1_nhwc,
		bundle->feat_4x_nhwc,
		bundle->sam2_05x_nhwc,
		bundle->sam2_1x_nhwc,
		bundle->sam2_2x_nhwc,
		bundle->sam2_4x_nhwc,
	};
	for (int i = 0; i < 8; i++) {
		err = write_tensor(f, ts[i]);
		if (err != SAM3_OK)
			goto done;
	}
	err = SAM3_OK;

done:
	fclose(f);
	return err;
}

enum sam3_error sam3_image_bundle_load(const char *path,
	const struct sam3_cache_persist_sig *expected_sig,
	struct sam3_arena *arena,
	uint64_t *out_hash,
	uint8_t *out_prefix, size_t *out_prefix_len,
	struct sam3_image_bundle *out_bundle)
{
	if (!path || !expected_sig || !arena || !out_hash ||
	    !out_prefix || !out_prefix_len || !out_bundle)
		return SAM3_EINVAL;

	FILE *f = fopen(path, "rb");
	if (!f) {
		sam3_log_error("cache_load: open %s failed", path);
		return SAM3_EIO;
	}

	enum sam3_error err = read_file_hdr(f, SAM3_CACHE_TYPE_IMAGE,
					    expected_sig);
	if (err != SAM3_OK)
		goto done;

	struct image_entry_hdr eh;
	if (fread(&eh, sizeof(eh), 1, f) != 1) {
		sam3_log_error("cache_load: short read on image entry");
		err = SAM3_EIO;
		goto done;
	}
	if (eh.n_tensors != 8) {
		sam3_log_error("cache_load: image n_tensors %u != 8",
			       eh.n_tensors);
		err = SAM3_EIO;
		goto done;
	}
	if (eh.prefix_len > SAM3_CACHE_PREFIX_BYTES) {
		err = SAM3_EIO;
		goto done;
	}

	*out_hash = eh.hash;
	memcpy(out_prefix, eh.prefix, eh.prefix_len);
	*out_prefix_len = eh.prefix_len;

	memset(out_bundle, 0, sizeof(*out_bundle));
	out_bundle->prompt_w = eh.prompt_w;
	out_bundle->prompt_h = eh.prompt_h;
	out_bundle->width    = eh.width;
	out_bundle->height   = eh.height;

	struct sam3_tensor **slots[8] = {
		&out_bundle->image_features,
		&out_bundle->feat_s0_nhwc,
		&out_bundle->feat_s1_nhwc,
		&out_bundle->feat_4x_nhwc,
		&out_bundle->sam2_05x_nhwc,
		&out_bundle->sam2_1x_nhwc,
		&out_bundle->sam2_2x_nhwc,
		&out_bundle->sam2_4x_nhwc,
	};
	for (int i = 0; i < 8; i++) {
		err = read_tensor(f, arena, slots[i]);
		if (err != SAM3_OK)
			goto done;
	}
	err = SAM3_OK;

done:
	fclose(f);
	return err;
}

/* --- text bundle --- */

#define SAM3_CACHE_TEXT_PREFIX_TOKENS (SAM3_CACHE_PREFIX_BYTES / 4)

struct text_entry_hdr {
	uint64_t hash;
	int32_t  n_tokens;
	uint32_t n_tensors;
	int32_t  prefix_len;
	int32_t  prefix_tokens[SAM3_CACHE_TEXT_PREFIX_TOKENS];
};

enum sam3_error sam3_text_bundle_save(const char *path,
	const struct sam3_cache_persist_sig *sig,
	uint64_t hash,
	const int32_t *prefix_tokens, int prefix_len,
	const struct sam3_text_bundle *bundle)
{
	if (!path || !sig || !bundle)
		return SAM3_EINVAL;
	if (prefix_len < 0)
		prefix_len = 0;
	if (prefix_len > SAM3_CACHE_TEXT_PREFIX_TOKENS)
		prefix_len = SAM3_CACHE_TEXT_PREFIX_TOKENS;

	FILE *f = fopen(path, "wb");
	if (!f) {
		sam3_log_error("cache_save: open %s failed", path);
		return SAM3_EIO;
	}
	enum sam3_error err = write_file_hdr(f, SAM3_CACHE_TYPE_TEXT, sig);
	if (err != SAM3_OK)
		goto done;

	struct text_entry_hdr eh;
	memset(&eh, 0, sizeof(eh));
	eh.hash      = hash;
	eh.n_tokens  = bundle->n_tokens;
	eh.n_tensors = 1;
	eh.prefix_len = prefix_len;
	if (prefix_tokens && prefix_len > 0)
		memcpy(eh.prefix_tokens, prefix_tokens,
		       (size_t)prefix_len * sizeof(int32_t));
	if (fwrite(&eh, sizeof(eh), 1, f) != 1) {
		err = SAM3_EIO;
		goto done;
	}
	err = write_tensor(f, bundle->features);

done:
	fclose(f);
	return err;
}

enum sam3_error sam3_text_bundle_load(const char *path,
	const struct sam3_cache_persist_sig *expected_sig,
	struct sam3_arena *arena,
	uint64_t *out_hash,
	int32_t *out_prefix_tokens, int *out_prefix_len,
	struct sam3_text_bundle *out_bundle)
{
	if (!path || !expected_sig || !arena || !out_hash ||
	    !out_prefix_tokens || !out_prefix_len || !out_bundle)
		return SAM3_EINVAL;

	FILE *f = fopen(path, "rb");
	if (!f) {
		sam3_log_error("cache_load: open %s failed", path);
		return SAM3_EIO;
	}
	enum sam3_error err = read_file_hdr(f, SAM3_CACHE_TYPE_TEXT,
					    expected_sig);
	if (err != SAM3_OK)
		goto done;

	struct text_entry_hdr eh;
	if (fread(&eh, sizeof(eh), 1, f) != 1) {
		sam3_log_error("cache_load: short read on text entry");
		err = SAM3_EIO;
		goto done;
	}
	if (eh.n_tensors != 1 || eh.prefix_len < 0 ||
	    eh.prefix_len > SAM3_CACHE_TEXT_PREFIX_TOKENS) {
		err = SAM3_EIO;
		goto done;
	}

	*out_hash = eh.hash;
	memcpy(out_prefix_tokens, eh.prefix_tokens,
	       (size_t)eh.prefix_len * sizeof(int32_t));
	*out_prefix_len = eh.prefix_len;

	memset(out_bundle, 0, sizeof(*out_bundle));
	out_bundle->n_tokens = eh.n_tokens;
	err = read_tensor(f, arena, &out_bundle->features);

done:
	fclose(f);
	return err;
}

/* --- disk-backed spill (no signature/magic) --- */

/*
 * In-process spill file layout. No magic / version / signature —
 * these files are only valid within the writing process's lifetime
 * (auto-deleted on promote or cache_destroy). The tensor records
 * reuse the same on-disk layout as the portable .sam3cache format
 * (see write_tensor/read_tensor above) but without the file header
 * or model signature; we just need geometry fields plus the bundle's
 * tensors.
 */
struct spill_hdr {
	int32_t  prompt_w;
	int32_t  prompt_h;
	int32_t  width;
	int32_t  height;
	uint32_t n_tensors;
};

enum sam3_error sam3_image_bundle_write_uncompressed(
	const char *path,
	const struct sam3_image_bundle *bundle)
{
	if (!path || !bundle)
		return SAM3_EINVAL;
	FILE *f = fopen(path, "wb");
	if (!f) {
		sam3_log_error("spill_write: open %s failed", path);
		return SAM3_EIO;
	}
	struct spill_hdr h = {0};
	h.prompt_w  = bundle->prompt_w;
	h.prompt_h  = bundle->prompt_h;
	h.width     = bundle->width;
	h.height    = bundle->height;
	h.n_tensors = 8;

	enum sam3_error err = SAM3_OK;
	if (fwrite(&h, sizeof(h), 1, f) != 1) {
		err = SAM3_EIO;
		goto done;
	}
	const struct sam3_tensor *ts[8] = {
		bundle->image_features,
		bundle->feat_s0_nhwc,
		bundle->feat_s1_nhwc,
		bundle->feat_4x_nhwc,
		bundle->sam2_05x_nhwc,
		bundle->sam2_1x_nhwc,
		bundle->sam2_2x_nhwc,
		bundle->sam2_4x_nhwc,
	};
	for (int i = 0; i < 8; i++) {
		err = write_tensor(f, ts[i]);
		if (err != SAM3_OK)
			goto done;
	}
done:
	fclose(f);
	if (err != SAM3_OK)
		remove(path);
	return err;
}

enum sam3_error sam3_image_bundle_read_uncompressed(
	const char *path,
	struct sam3_arena *arena,
	struct sam3_image_bundle *out_bundle)
{
	if (!path || !arena || !out_bundle)
		return SAM3_EINVAL;
	FILE *f = fopen(path, "rb");
	if (!f) {
		sam3_log_error("spill_read: open %s failed", path);
		return SAM3_EIO;
	}
	struct spill_hdr h;
	enum sam3_error err = SAM3_OK;
	if (fread(&h, sizeof(h), 1, f) != 1) {
		err = SAM3_EIO;
		goto done;
	}
	if (h.n_tensors != 8) {
		err = SAM3_EIO;
		goto done;
	}
	memset(out_bundle, 0, sizeof(*out_bundle));
	out_bundle->prompt_w = h.prompt_w;
	out_bundle->prompt_h = h.prompt_h;
	out_bundle->width    = h.width;
	out_bundle->height   = h.height;
	struct sam3_tensor **slots[8] = {
		&out_bundle->image_features,
		&out_bundle->feat_s0_nhwc,
		&out_bundle->feat_s1_nhwc,
		&out_bundle->feat_4x_nhwc,
		&out_bundle->sam2_05x_nhwc,
		&out_bundle->sam2_1x_nhwc,
		&out_bundle->sam2_2x_nhwc,
		&out_bundle->sam2_4x_nhwc,
	};
	for (int i = 0; i < 8; i++) {
		err = read_tensor(f, arena, slots[i]);
		if (err != SAM3_OK)
			goto done;
	}
done:
	fclose(f);
	return err;
}
