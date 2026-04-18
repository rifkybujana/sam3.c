/*
 * src/core/weight.c - Native .sam3 weight file loader and writer
 *
 * Implements mmap-based loading of .sam3 weight files with FNV-1a hash
 * table for O(1) tensor lookup. Also provides the writer that serializes
 * tensors from any weight_reader into the .sam3 binary format.
 *
 * Key types:  sam3_weight_file
 * Depends on: core/weight.h, util/log.h
 * Used by:    sam3.c, tools/sam3_convert.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "weight.h"
#include "util/log.h"

/* ── Async prefetch ────────────────────────────────────────────────── */

struct prefetch_args {
	void   *addr;
	size_t  len;
};

static void *prefetch_worker(void *arg)
{
	struct prefetch_args *a = arg;

	madvise(a->addr, a->len, MADV_WILLNEED);
	free(a);
	return NULL;
}

/* ── Helpers ───────────────────────────────────────────────────────── */

static size_t align_up(size_t val, size_t align)
{
	return (val + align - 1) & ~(align - 1);
}

static uint32_t fnv1a(const char *s)
{
	uint32_t h = 0x811c9dc5;

	while (*s) {
		h ^= (uint8_t)*s++;
		h *= 0x01000193;
	}
	return h;
}

/* ── Writer ────────────────────────────────────────────────────────── */

enum sam3_error sam3_weight_write(const char *output_path,
				  const struct sam3_model_config *config,
				  struct weight_reader *reader)
{
	FILE *fp = NULL;
	struct sam3_weight_tensor_desc *descs = NULL;
	void *buf = NULL;
	enum sam3_error err = SAM3_OK;
	int n;

	if (!output_path || !config || !reader) {
		sam3_log_error("weight_write: NULL argument");
		return SAM3_EINVAL;
	}

	n = reader->ops->n_tensors(reader);
	if (n <= 0) {
		sam3_log_error("weight_write: no tensors in reader");
		return SAM3_EINVAL;
	}

	/* Build header */
	struct sam3_weight_header hdr;
	memset(&hdr, 0, sizeof(hdr));
	hdr.magic            = SAM3_WEIGHT_MAGIC;
	hdr.version          = SAM3_WEIGHT_VERSION;
	hdr.flags            = 0;
	hdr.n_tensors        = (uint32_t)n;
	hdr.image_size       = config->image_size;
	hdr.encoder_dim      = config->encoder_dim;
	hdr.decoder_dim      = config->decoder_dim;
	hdr.n_encoder_layers = config->n_encoder_layers;
	hdr.n_decoder_layers = config->n_decoder_layers;
	hdr.reserved[0]      = (uint32_t)config->backbone_type;
	hdr.reserved[1]      = (uint32_t)config->variant;
	hdr.reserved[2]      = (uint32_t)config->n_fpn_scales;

	/* Build tensor descriptors */
	descs = calloc((size_t)n, sizeof(*descs));
	if (!descs) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	size_t data_offset = 0;

	for (int i = 0; i < n; i++) {
		struct weight_tensor_info info;
		err = reader->ops->get_tensor_info(reader, i, &info);
		if (err != SAM3_OK)
			goto cleanup;

		/* Copy name, truncating if necessary */
		size_t name_len = strlen(info.name);
		if (name_len >= SAM3_WEIGHT_NAME_MAX) {
			sam3_log_warn("tensor name truncated: %s", info.name);
			name_len = SAM3_WEIGHT_NAME_MAX - 1;
		}
		memcpy(descs[i].name, info.name, name_len);
		descs[i].name[name_len] = '\0';

		descs[i].dtype       = (uint32_t)info.dtype;
		descs[i].n_dims      = (uint32_t)info.n_dims;
		for (int d = 0; d < SAM3_MAX_DIMS; d++)
			descs[i].dims[d] = info.dims[d];
		descs[i].data_offset = data_offset;
		descs[i].data_size   = info.nbytes;
		descs[i].reserved    = (uint64_t)fnv1a(descs[i].name);

		sam3_log_debug("tensor[%d] \"%s\": %zu bytes @ offset %zu",
			       i, descs[i].name,
			       (size_t)descs[i].data_size,
			       (size_t)descs[i].data_offset);

		data_offset = align_up(data_offset + info.nbytes,
				       SAM3_WEIGHT_DATA_ALIGN);
	}

	/* Open output file */
	fp = fopen(output_path, "wb");
	if (!fp) {
		sam3_log_error("weight_write: cannot open %s", output_path);
		err = SAM3_EIO;
		goto cleanup;
	}

	/* Write header */
	if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
		err = SAM3_EIO;
		goto cleanup;
	}

	/* Write tensor descriptors */
	if (fwrite(descs, sizeof(*descs), (size_t)n, fp) != (size_t)n) {
		err = SAM3_EIO;
		goto cleanup;
	}

	/* Pad to page boundary before data blob */
	size_t table_end = sizeof(hdr) + (size_t)n * sizeof(*descs);
	size_t data_start = align_up(table_end, SAM3_WEIGHT_PAGE_ALIGN);
	size_t pad_bytes = data_start - table_end;

	if (pad_bytes > 0) {
		void *zeros = calloc(1, pad_bytes);
		if (!zeros) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		size_t written = fwrite(zeros, 1, pad_bytes, fp);
		free(zeros);
		if (written != pad_bytes) {
			err = SAM3_EIO;
			goto cleanup;
		}
	}

	/* Write tensor data with 64-byte alignment padding */
	for (int i = 0; i < n; i++) {
		size_t nbytes = (size_t)descs[i].data_size;

		/* Allocate buffer for this tensor's data */
		buf = malloc(nbytes);
		if (!buf) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}

		err = reader->ops->read_tensor_data(reader, i, buf, nbytes);
		if (err != SAM3_OK) {
			free(buf);
			buf = NULL;
			goto cleanup;
		}

		if (fwrite(buf, 1, nbytes, fp) != nbytes) {
			free(buf);
			buf = NULL;
			err = SAM3_EIO;
			goto cleanup;
		}

		free(buf);
		buf = NULL;

		/* Pad to 64-byte alignment (except after the last tensor) */
		if (i < n - 1) {
			size_t aligned = align_up(nbytes, SAM3_WEIGHT_DATA_ALIGN);
			size_t tensor_pad = aligned - nbytes;
			if (tensor_pad > 0) {
				uint8_t pad[SAM3_WEIGHT_DATA_ALIGN];
				memset(pad, 0, sizeof(pad));
				if (fwrite(pad, 1, tensor_pad, fp) != tensor_pad) {
					err = SAM3_EIO;
					goto cleanup;
				}
			}
		}
	}

	sam3_log_info("wrote %d tensors to %s", n, output_path);

cleanup:
	if (fp)
		fclose(fp);
	free(descs);
	if (err != SAM3_OK && output_path)
		remove(output_path);
	return err;
}

/* ── Loader ────────────────────────────────────────────────────────── */

static uint32_t next_pow2(uint32_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

static int build_hash_table(struct sam3_weight_file *wf)
{
	uint32_t n = wf->header->n_tensors;

	/*
	 * Size table at 2x the tensor count to keep load factor ≤ 50%.
	 * At 50% occupancy, average probe length with linear probing is
	 * ~1.5 — well below the O(n) degradation at >75%.
	 */
	uint32_t min_cap = n * 2;

	if (min_cap < 8)
		min_cap = 8;
	uint32_t cap = next_pow2(min_cap);

	wf->hash_table = malloc(cap * sizeof(uint32_t));
	if (!wf->hash_table)
		return -1;

	wf->hash_capacity = cap;

	for (uint32_t i = 0; i < cap; i++)
		wf->hash_table[i] = SAM3_WEIGHT_HASH_EMPTY;

	uint32_t mask = cap - 1;
	uint32_t max_probe = 0;

	for (uint32_t i = 0; i < n; i++) {
		uint32_t hash;
		if (wf->tensors[i].reserved != 0)
			hash = (uint32_t)wf->tensors[i].reserved;
		else
			hash = fnv1a(wf->tensors[i].name);
		uint32_t slot = hash & mask;
		uint32_t probe = 0;
		while (wf->hash_table[slot] != SAM3_WEIGHT_HASH_EMPTY) {
			slot = (slot + 1) & mask;
			probe++;
		}
		wf->hash_table[slot] = i;
		if (probe > max_probe)
			max_probe = probe;
	}

	if (max_probe > 8)
		sam3_log_warn("weight hash: max probe length %u "
			      "(%u tensors, %u slots)",
			      max_probe, n, cap);

	return 0;
}

enum sam3_error sam3_weight_open(struct sam3_weight_file *wf,
				 const char *path)
{
	int fd = -1;
	void *mapped = MAP_FAILED;
	size_t file_size = 0;

	if (!wf || !path) {
		sam3_log_error("weight_open: NULL argument");
		return SAM3_EINVAL;
	}

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		sam3_log_error("weight_open: cannot open %s", path);
		return SAM3_EIO;
	}

	struct stat st;
	if (fstat(fd, &st) < 0) {
		sam3_log_error("weight_open: fstat failed for %s", path);
		close(fd);
		return SAM3_EIO;
	}

	file_size = (size_t)st.st_size;
	if (file_size < sizeof(struct sam3_weight_header)) {
		sam3_log_error("weight_open: file too small: %s", path);
		close(fd);
		return SAM3_EMODEL;
	}

	mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);

	if (mapped == MAP_FAILED) {
		sam3_log_error("weight_open: mmap failed for %s", path);
		return SAM3_EIO;
	}

	/* Hint sequential access for initial weight loading pass */
	madvise(mapped, file_size, MADV_SEQUENTIAL);

	/* Validate header */
	const struct sam3_weight_header *hdr = mapped;

	if (hdr->magic != SAM3_WEIGHT_MAGIC) {
		sam3_log_error("weight_open: bad magic 0x%08x in %s",
			       hdr->magic, path);
		munmap(mapped, file_size);
		memset(wf, 0, sizeof(*wf));
		return SAM3_EMODEL;
	}

	if (hdr->version != SAM3_WEIGHT_VERSION) {
		sam3_log_error("weight_open: unsupported version %u in %s "
			       "(expected %u; regenerate via sam3_convert)",
			       hdr->version, path, SAM3_WEIGHT_VERSION);
		munmap(mapped, file_size);
		memset(wf, 0, sizeof(*wf));
		return SAM3_EMODEL;
	}

	/* Validate tensor table fits in file */
	size_t table_size = (size_t)hdr->n_tensors *
			    sizeof(struct sam3_weight_tensor_desc);
	size_t table_end = sizeof(struct sam3_weight_header) + table_size;

	if (table_end > file_size) {
		sam3_log_error("weight_open: tensor table exceeds file: %s",
			       path);
		munmap(mapped, file_size);
		memset(wf, 0, sizeof(*wf));
		return SAM3_EMODEL;
	}

	/* Populate the handle */
	enum sam3_error err;
	size_t data_start = align_up(table_end, SAM3_WEIGHT_PAGE_ALIGN);

	if (data_start > file_size) {
		sam3_log_error("data section start (%zu) exceeds file size (%zu)",
			       data_start, file_size);
		err = SAM3_EMODEL;
		goto fail;
	}

	wf->mapped      = mapped;
	wf->mapped_size = file_size;
	wf->header      = hdr;
	wf->tensors     = (const struct sam3_weight_tensor_desc *)
			  ((const char *)mapped +
			   sizeof(struct sam3_weight_header));
	wf->data_base   = (const char *)mapped + data_start;

	/* Validate all tensor data ranges fit in file */
	uint32_t n = hdr->n_tensors;
	size_t data_section_size = file_size - data_start;

	for (uint32_t i = 0; i < n; i++) {
		uint64_t end = wf->tensors[i].data_offset +
			       wf->tensors[i].data_size;
		if (end > data_section_size) {
			sam3_log_error("tensor %u data extends past file end",
				       i);
			err = SAM3_EMODEL;
			goto fail;
		}
	}

	if (build_hash_table(wf) < 0) {
		sam3_log_error("weight_open: hash table alloc failed");
		err = SAM3_ENOMEM;
		goto fail;
	}

	/*
	 * The initial scan above touches the header, the tensor table,
	 * and (via build_hash_table) every tensor name — strictly
	 * sequential, so MADV_SEQUENTIAL was the right hint. Inference,
	 * however, accesses the data blob in a random-but-repeated
	 * pattern (the same ~50 tensors per ViT block, per inference).
	 * MADV_SEQUENTIAL would let the kernel drop those pages after
	 * the first read, forcing re-faults on every subsequent block.
	 *
	 * Switch to MADV_RANDOM to discourage pre-eviction, then spawn
	 * a background thread to issue MADV_WILLNEED so the caller can
	 * overlap processor/module init with the prefetch I/O.
	 */
	madvise(mapped, file_size, MADV_RANDOM);

	struct prefetch_args *pa = malloc(sizeof(*pa));
	if (pa) {
		pa->addr = (void *)wf->data_base;
		pa->len  = file_size - data_start;
		if (pthread_create(&wf->prefetch_thread, NULL,
				   prefetch_worker, pa) == 0) {
			wf->prefetch_active = 1;
		} else {
			/* Thread creation failed — prefetch synchronously */
			madvise(pa->addr, pa->len, MADV_WILLNEED);
			free(pa);
		}
	} else {
		/* Alloc failed — prefetch synchronously */
		madvise((void *)wf->data_base, file_size - data_start,
			MADV_WILLNEED);
	}

	sam3_log_info("opened %s: %u tensors", path, hdr->n_tensors);
	return SAM3_OK;

fail:
	munmap(mapped, file_size);
	memset(wf, 0, sizeof(*wf));
	return err;
}

void sam3_weight_prefetch_wait(struct sam3_weight_file *wf)
{
	if (!wf || !wf->prefetch_active)
		return;

	pthread_join(wf->prefetch_thread, NULL);
	wf->prefetch_active = 0;
}

void sam3_weight_close(struct sam3_weight_file *wf)
{
	if (!wf)
		return;

	sam3_weight_prefetch_wait(wf);

	if (wf->mapped && wf->mapped_size > 0)
		munmap(wf->mapped, wf->mapped_size);

	free(wf->hash_table);
	memset(wf, 0, sizeof(*wf));
}

void sam3_weight_madvise(struct sam3_weight_file *wf, int advice)
{
	if (wf && wf->mapped && wf->mapped_size > 0)
		madvise(wf->mapped, wf->mapped_size, advice);
}

const struct sam3_weight_tensor_desc *sam3_weight_find(
	const struct sam3_weight_file *wf, const char *name)
{
	if (!wf || !name || !wf->hash_table)
		return NULL;

	uint32_t mask = wf->hash_capacity - 1;
	uint32_t slot = fnv1a(name) & mask;

	while (wf->hash_table[slot] != SAM3_WEIGHT_HASH_EMPTY) {
		uint32_t idx = wf->hash_table[slot];
		if (strcmp(wf->tensors[idx].name, name) == 0)
			return &wf->tensors[idx];
		slot = (slot + 1) & mask;
	}

	return NULL;
}

const void *sam3_weight_tensor_data(
	const struct sam3_weight_file *wf,
	const struct sam3_weight_tensor_desc *desc)
{
	if (!wf || !desc)
		return NULL;

	return (const char *)wf->data_base + desc->data_offset;
}

enum sam3_error sam3_weight_to_tensor(
	const struct sam3_weight_file *wf,
	const struct sam3_weight_tensor_desc *desc,
	struct sam3_tensor *out)
{
	if (!wf || !desc || !out)
		return SAM3_EINVAL;

	out->dtype  = (enum sam3_dtype)desc->dtype;
	out->n_dims = (int)desc->n_dims;

	for (int i = 0; i < SAM3_MAX_DIMS; i++)
		out->dims[i] = desc->dims[i];

	out->data   = (void *)sam3_weight_tensor_data(wf, desc);
	out->nbytes = (size_t)desc->data_size;

	/* Q8_0 is block-quantized; per-element strides are meaningless */
	if (out->dtype != SAM3_DTYPE_Q8_0)
		sam3_tensor_compute_strides(out);
	else
		memset(out->strides, 0, sizeof(out->strides));

	return SAM3_OK;
}
