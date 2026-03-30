/*
 * src/core/weight_safetensors.c - SafeTensors format reader
 *
 * Reads SafeTensors files (.safetensors) through the weight_reader
 * vtable interface. The file is memory-mapped and the JSON header is
 * parsed with cJSON. Tensor data is read directly from the mapped
 * region. Used by sam3_convert to produce .sam3 files.
 *
 * Key types:  weight_reader (via vtable)
 * Depends on: core/weight.h, core/json/cJSON.h, util/log.h
 * Used by:    tools/sam3_convert.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "weight.h"
#include "json/cJSON.h"
#include "util/log.h"

/* ── Internal types ────────────────────────────────────────────────── */

struct st_tensor_entry {
	char            name[SAM3_WEIGHT_NAME_MAX];
	enum sam3_dtype dtype;
	int             n_dims;
	int             dims[SAM3_MAX_DIMS];
	size_t          data_start;  /* offset into data section */
	size_t          data_end;
	size_t          nbytes;
};

struct st_reader_state {
	void                   *mapped;
	size_t                  mapped_size;
	const void             *data_section;
	struct st_tensor_entry *entries;
	int                     n_entries;
};

/* ── Helpers ───────────────────────────────────────────────────────── */

static enum sam3_dtype parse_dtype(const char *s)
{
	if (strcmp(s, "F32") == 0)  return SAM3_DTYPE_F32;
	if (strcmp(s, "F16") == 0)  return SAM3_DTYPE_F16;
	if (strcmp(s, "BF16") == 0) return SAM3_DTYPE_BF16;
	if (strcmp(s, "I32") == 0)  return SAM3_DTYPE_I32;
	if (strcmp(s, "I8") == 0)   return SAM3_DTYPE_I8;
	return (enum sam3_dtype)-1;
}

/* ── Vtable implementations ────────────────────────────────────────── */

static enum sam3_error st_open(struct weight_reader *r, const char *path)
{
	int fd = -1;
	void *mapped = MAP_FAILED;
	cJSON *root = NULL;
	struct st_reader_state *s = NULL;
	struct st_tensor_entry *entries = NULL;
	size_t file_size = 0;

	if (!r || !path) {
		sam3_log_error("st_open: NULL argument");
		return SAM3_EINVAL;
	}

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		sam3_log_error("st_open: cannot open %s", path);
		return SAM3_EIO;
	}

	struct stat st;
	if (fstat(fd, &st) < 0) {
		sam3_log_error("st_open: fstat failed for %s", path);
		close(fd);
		return SAM3_EIO;
	}

	file_size = (size_t)st.st_size;
	if (file_size < 8) {
		sam3_log_error("st_open: file too small: %s", path);
		close(fd);
		return SAM3_EIO;
	}

	mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);

	if (mapped == MAP_FAILED) {
		sam3_log_error("st_open: mmap failed for %s", path);
		return SAM3_EIO;
	}

	/* Read 8-byte header_size (little-endian uint64_t) */
	uint64_t header_size;
	memcpy(&header_size, mapped, sizeof(header_size));

	if (8 + header_size > file_size) {
		sam3_log_error("st_open: header_size exceeds file: %s",
			       path);
		goto fail;
	}

	/* Parse JSON header (need null-terminated copy) */
	char *json_str = malloc((size_t)header_size + 1);
	if (!json_str) {
		sam3_log_error("st_open: malloc failed for JSON buffer");
		goto fail;
	}
	memcpy(json_str, (const char *)mapped + 8, (size_t)header_size);
	json_str[header_size] = '\0';

	root = cJSON_Parse(json_str);
	free(json_str);

	if (!root) {
		sam3_log_error("st_open: JSON parse failed for %s", path);
		goto fail;
	}

	/* Count tensors (skip __metadata__) */
	int n_entries = 0;
	cJSON *item = NULL;
	cJSON_ArrayForEach(item, root) {
		if (strcmp(item->string, "__metadata__") == 0)
			continue;
		n_entries++;
	}

	if (n_entries == 0) {
		sam3_log_error("st_open: no tensors in %s", path);
		goto fail;
	}

	entries = calloc((size_t)n_entries, sizeof(*entries));
	if (!entries) {
		sam3_log_error("st_open: calloc failed for entries");
		goto fail;
	}

	/* Parse each tensor entry */
	int idx = 0;
	cJSON_ArrayForEach(item, root) {
		if (strcmp(item->string, "__metadata__") == 0)
			continue;

		struct st_tensor_entry *e = &entries[idx];

		/* Name */
		size_t name_len = strlen(item->string);
		if (name_len >= SAM3_WEIGHT_NAME_MAX) {
			sam3_log_warn("st_open: name truncated: %s",
				      item->string);
			name_len = SAM3_WEIGHT_NAME_MAX - 1;
		}
		memcpy(e->name, item->string, name_len);
		e->name[name_len] = '\0';

		/* dtype */
		cJSON *dtype_obj = cJSON_GetObjectItemCaseSensitive(
			item, "dtype");
		if (!cJSON_IsString(dtype_obj)) {
			sam3_log_error("st_open: missing dtype for %s",
				       item->string);
			goto fail;
		}
		e->dtype = parse_dtype(dtype_obj->valuestring);
		if ((int)e->dtype == -1) {
			sam3_log_error("st_open: unknown dtype '%s' for %s",
				       dtype_obj->valuestring,
				       item->string);
			goto fail;
		}

		/* shape */
		cJSON *shape = cJSON_GetObjectItemCaseSensitive(
			item, "shape");
		if (!cJSON_IsArray(shape)) {
			sam3_log_error("st_open: missing shape for %s",
				       item->string);
			goto fail;
		}
		e->n_dims = cJSON_GetArraySize(shape);
		if (e->n_dims > SAM3_MAX_DIMS) {
			sam3_log_error("st_open: too many dims for %s",
				       item->string);
			goto fail;
		}
		for (int d = 0; d < e->n_dims; d++) {
			cJSON *dim = cJSON_GetArrayItem(shape, d);
			if (!cJSON_IsNumber(dim)) {
				sam3_log_error("st_open: bad shape[%d] "
					       "for %s", d,
					       item->string);
				goto fail;
			}
			e->dims[d] = (int)dim->valuedouble;
		}

		/* data_offsets */
		cJSON *offsets = cJSON_GetObjectItemCaseSensitive(
			item, "data_offsets");
		if (!cJSON_IsArray(offsets) ||
		    cJSON_GetArraySize(offsets) != 2) {
			sam3_log_error("st_open: bad data_offsets for %s",
				       item->string);
			goto fail;
		}
		cJSON *off_start = cJSON_GetArrayItem(offsets, 0);
		cJSON *off_end = cJSON_GetArrayItem(offsets, 1);
		if (!cJSON_IsNumber(off_start) ||
		    !cJSON_IsNumber(off_end)) {
			sam3_log_error("st_open: non-numeric offsets "
				       "for %s", item->string);
			goto fail;
		}
		e->data_start = (size_t)off_start->valuedouble;
		e->data_end   = (size_t)off_end->valuedouble;
		e->nbytes     = e->data_end - e->data_start;

		sam3_log_debug("st tensor[%d] \"%s\": %zu bytes "
			       "[%zu, %zu)", idx, e->name, e->nbytes,
			       e->data_start, e->data_end);
		idx++;
	}

	/* Build reader state */
	s = calloc(1, sizeof(*s));
	if (!s) {
		sam3_log_error("st_open: calloc failed for state");
		goto fail;
	}

	s->mapped       = mapped;
	s->mapped_size  = file_size;
	s->data_section = (const char *)mapped + 8 + (size_t)header_size;
	s->entries      = entries;
	s->n_entries    = n_entries;

	r->impl = s;

	cJSON_Delete(root);
	sam3_log_info("st_open: loaded %d tensors from %s",
		      n_entries, path);
	return SAM3_OK;

fail:
	if (root)
		cJSON_Delete(root);
	free(entries);
	if (mapped != MAP_FAILED)
		munmap(mapped, file_size);
	return SAM3_EIO;
}

static int st_n_tensors(struct weight_reader *r)
{
	struct st_reader_state *s = r->impl;
	return s->n_entries;
}

static enum sam3_error st_get_tensor_info(struct weight_reader *r, int idx,
					  struct weight_tensor_info *info)
{
	struct st_reader_state *s = r->impl;

	if (idx < 0 || idx >= s->n_entries)
		return SAM3_EINVAL;

	struct st_tensor_entry *e = &s->entries[idx];
	info->name   = e->name;
	info->dtype  = e->dtype;
	info->n_dims = e->n_dims;
	memset(info->dims, 0, sizeof(info->dims));
	for (int d = 0; d < e->n_dims; d++)
		info->dims[d] = e->dims[d];
	info->nbytes = e->nbytes;

	return SAM3_OK;
}

static enum sam3_error st_read_tensor_data(struct weight_reader *r, int idx,
					   void *dst, size_t dst_size)
{
	struct st_reader_state *s = r->impl;

	if (idx < 0 || idx >= s->n_entries)
		return SAM3_EINVAL;

	struct st_tensor_entry *e = &s->entries[idx];

	if (dst_size < e->nbytes) {
		sam3_log_error("st_read_tensor_data: dst_size %zu < %zu",
			       dst_size, e->nbytes);
		return SAM3_EINVAL;
	}

	memcpy(dst, (const char *)s->data_section + e->data_start,
	       e->nbytes);
	return SAM3_OK;
}

static void st_close(struct weight_reader *r)
{
	struct st_reader_state *s = r->impl;

	if (!s)
		return;

	if (s->mapped && s->mapped_size > 0)
		munmap(s->mapped, s->mapped_size);

	free(s->entries);
	free(s);
	r->impl = NULL;
}

/* ── Public vtable ─────────────────────────────────────────────────── */

static const struct weight_reader_ops st_reader_ops = {
	.open             = st_open,
	.n_tensors        = st_n_tensors,
	.get_tensor_info  = st_get_tensor_info,
	.read_tensor_data = st_read_tensor_data,
	.close            = st_close,
};

void weight_reader_safetensors_init(struct weight_reader *r)
{
	r->ops  = &st_reader_ops;
	r->impl = NULL;
}
