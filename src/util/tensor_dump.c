/*
 * src/util/tensor_dump.c - Binary tensor dump for validation.
 *
 * Implements sam3_tensor_dump: writes tensor shape + float32 data to a
 * binary file for comparison against Python reference outputs. No
 * allocations — writes directly from the tensor's data pointer.
 *
 * Key types:  none
 * Depends on: sam3/internal/tensor_dump.h, core/tensor.h
 * Used by:    tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdint.h>
#include "sam3/internal/tensor_dump.h"
#include "core/tensor.h"

int sam3_tensor_dump(const char *path, const struct sam3_tensor *tensor)
{
	FILE *f;
	int32_t hdr[1 + SAM3_MAX_DIMS];
	int n_elems = 1;

	if (!path || !tensor || !tensor->data)
		return -1;

	if (tensor->dtype != SAM3_DTYPE_F32)
		return -1;

	f = fopen(path, "wb");
	if (!f)
		return -1;

	hdr[0] = (int32_t)tensor->n_dims;
	for (int i = 0; i < tensor->n_dims; i++) {
		hdr[1 + i] = (int32_t)tensor->dims[i];
		n_elems *= tensor->dims[i];
	}

	if (fwrite(hdr, sizeof(int32_t),
		   (size_t)(1 + tensor->n_dims), f) !=
	    (size_t)(1 + tensor->n_dims)) {
		fclose(f);
		return -1;
	}

	if (fwrite(tensor->data, sizeof(float),
		   (size_t)n_elems, f) != (size_t)n_elems) {
		fclose(f);
		return -1;
	}

	fclose(f);
	return 0;
}
