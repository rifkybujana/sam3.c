/*
 * tests/test_npy.h - Minimal numpy .npy reader for fixture loading
 *
 * Supports F32 and I32 arrays of rank 1-2 in little-endian NPY v1.0/v2.0
 * format. Used by MobileCLIP parity tests to load reference activations.
 * Returns 0 on success, -1 on error. Caller provides a buffer sized to
 * accommodate the array; on success n_dims and dims[] are populated.
 *
 * Key types:  (header-only helpers)
 * Depends on: <stdio.h>, <stdlib.h>, <stdint.h>, <string.h>
 * Used by:    tests/test_mobileclip_text.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TEST_NPY_H
#define SAM3_TEST_NPY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/*
 * test_npy_parse_header - Parse the NPY header and leave fp at data start.
 *
 * @fp:          Open file positioned at byte 0.
 * @out_dtype:   Set to 'f' for float32, 'i' for int32/int64.
 * @out_n_dims:  Number of dimensions parsed from shape tuple.
 * @out_dims:    Caller-provided array of at least 4 ints; dims filled in.
 *
 * Returns 0 on success, -1 on any parse or format error.
 */
static inline int
test_npy_parse_header(FILE *fp, int *out_dtype, int *out_n_dims, int *out_dims)
{
	char     magic[6];
	uint8_t  v_major, v_minor;
	uint32_t hdr_len = 0;
	char    *hdr;
	const char *desc, *shape, *open_paren, *p;
	int      n;

	if (fread(magic, 1, 6, fp) != 6) return -1;
	if (memcmp(magic, "\x93NUMPY", 6) != 0) return -1;
	if (fread(&v_major, 1, 1, fp) != 1) return -1;
	if (fread(&v_minor, 1, 1, fp) != 1) return -1;
	(void)v_minor;

	if (v_major == 1) {
		uint16_t hl16;
		if (fread(&hl16, 2, 1, fp) != 1) return -1;
		hdr_len = hl16;
	} else if (v_major == 2 || v_major == 3) {
		if (fread(&hdr_len, 4, 1, fp) != 1) return -1;
	} else {
		return -1;
	}

	hdr = (char *)malloc(hdr_len + 1);
	if (!hdr) return -1;
	if (fread(hdr, 1, hdr_len, fp) != hdr_len) { free(hdr); return -1; }
	hdr[hdr_len] = '\0';

	desc  = strstr(hdr, "'descr':");
	shape = strstr(hdr, "'shape':");
	if (!desc || !shape) { free(hdr); return -1; }

	if (strstr(desc, "<f4"))
		*out_dtype = 'f';
	else if (strstr(desc, "<i4") || strstr(desc, "<i8"))
		*out_dtype = 'i';
	else { free(hdr); return -1; }

	open_paren = strchr(shape, '(');
	if (!open_paren) { free(hdr); return -1; }

	p = open_paren + 1;
	n = 0;
	while (*p && *p != ')') {
		while (*p == ' ' || *p == ',') p++;
		if (*p == ')') break;
		out_dims[n++] = atoi(p);
		while (*p && *p != ',' && *p != ')') p++;
	}
	*out_n_dims = n;

	free(hdr);
	return 0;
}

/*
 * test_npy_load_f32 - Load an F32 .npy file into a caller-supplied buffer.
 *
 * @path:   File path to the .npy file.
 * @out:    Caller-provided float buffer, must hold at least n_elems floats.
 * @dims:   Caller-provided int[4]; filled with the array dimensions.
 * @n_dims: Set to the number of dimensions.
 *
 * Returns 0 on success, -1 on error.
 */
static inline int
test_npy_load_f32(const char *path, float *out, int *dims, int *n_dims)
{
	FILE   *fp;
	int     dtype;
	size_t  total, i;

	fp = fopen(path, "rb");
	if (!fp) return -1;

	if (test_npy_parse_header(fp, &dtype, n_dims, dims) != 0 ||
	    dtype != 'f') {
		fclose(fp);
		return -1;
	}

	total = 1;
	for (i = 0; i < (size_t)*n_dims; i++)
		total *= (size_t)dims[i];

	if (fread(out, sizeof(float), total, fp) != total) {
		fclose(fp);
		return -1;
	}

	fclose(fp);
	return 0;
}

/*
 * test_npy_load_i32 - Load an I32 (or I64-cast-to-I32) .npy file.
 *
 * Handles both '<i4' and '<i8' descriptors. For '<i8', each 8-byte value
 * is truncated to int32. Caller-provided buffer must hold at least
 * max_tokens elements.
 *
 * @path:       File path to the .npy file.
 * @out:        Caller-provided int32_t buffer.
 * @n_tokens:   Set to the total number of elements read.
 *
 * Returns 0 on success, -1 on error.
 */
static inline int
test_npy_load_i32(const char *path, int32_t *out, int *n_tokens)
{
	FILE    *fp;
	int      dtype, n_dims, dims[4];
	size_t   total, i;

	fp = fopen(path, "rb");
	if (!fp) return -1;

	if (test_npy_parse_header(fp, &dtype, &n_dims, dims) != 0 ||
	    dtype != 'i') {
		fclose(fp);
		return -1;
	}

	total = 1;
	for (i = 0; i < (size_t)n_dims; i++)
		total *= (size_t)dims[i];

	/* Re-open to peek at descriptor for width determination: we already
	 * consumed the header. Use file position heuristic: remaining bytes /
	 * total tells us element width. */
	{
		long data_start = ftell(fp);
		long file_end;
		size_t elem_bytes;

		fseek(fp, 0, SEEK_END);
		file_end = ftell(fp);
		fseek(fp, data_start, SEEK_SET);

		if (total == 0) { fclose(fp); *n_tokens = 0; return 0; }
		elem_bytes = (size_t)(file_end - data_start) / total;

		if (elem_bytes == 4) {
			if (fread(out, sizeof(int32_t), total, fp) != total) {
				fclose(fp);
				return -1;
			}
		} else if (elem_bytes == 8) {
			/* int64 — cast each down to int32 */
			for (i = 0; i < total; i++) {
				int64_t v;
				if (fread(&v, sizeof(int64_t), 1, fp) != 1) {
					fclose(fp);
					return -1;
				}
				out[i] = (int32_t)v;
			}
		} else {
			fclose(fp);
			return -1;
		}
	}

	fclose(fp);
	*n_tokens = (int)total;
	return 0;
}

#endif /* SAM3_TEST_NPY_H */
