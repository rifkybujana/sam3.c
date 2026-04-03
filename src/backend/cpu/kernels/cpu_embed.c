/*
 * src/backend/cpu/kernels/cpu_embed.c - Embedding table lookup kernel
 *
 * Looks up rows from an embedding table by index.  inputs[0] is the
 * embedding table of shape [vocab_size, embed_dim] (F32), inputs[1]
 * is the index tensor of shape [seq_len] (I32), and the output is
 * [seq_len, embed_dim] (F32).  Each output row is a memcpy of the
 * corresponding table row.  No thread pool is used since the copy
 * per token is trivially fast.
 *
 * Key types:  sam3_node, sam3_tensor
 * Depends on: cpu_kernels.h, core/tensor.h, util/log.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_kernels.h"
#include "core/tensor.h"
#include "util/log.h"

#include <string.h>

enum sam3_error cpu_kernel_embed(const struct sam3_node *node,
				 struct sam3_threadpool *pool)
{
	(void)pool;

	if (!node->inputs[0] || !node->inputs[1] || !node->output) {
		sam3_log_error("embed: NULL tensor");
		return SAM3_EINVAL;
	}

	const struct sam3_tensor *table = node->inputs[0];
	const struct sam3_tensor *indices = node->inputs[1];
	struct sam3_tensor *output = node->output;

	if (table->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("embed: table must be F32");
		return SAM3_EINVAL;
	}

	if (indices->dtype != SAM3_DTYPE_I32) {
		sam3_log_error("embed: indices must be I32");
		return SAM3_EINVAL;
	}

	int vocab_size = table->dims[0];
	int embed_dim = table->dims[1];
	int seq_len = sam3_tensor_nelems(indices);

	const float *tbl = (const float *)table->data;
	const int32_t *idx = (const int32_t *)indices->data;
	float *out = (float *)output->data;

	for (int i = 0; i < seq_len; i++) {
		int32_t tok = idx[i];
		if (tok < 0 || tok >= vocab_size) {
			sam3_log_error("embed: index %d out of range "
				       "[0, %d)", tok, vocab_size);
			return SAM3_EINVAL;
		}
		memcpy(out + i * embed_dim,
		       tbl + tok * embed_dim,
		       (size_t)embed_dim * sizeof(float));
	}

	return SAM3_OK;
}
