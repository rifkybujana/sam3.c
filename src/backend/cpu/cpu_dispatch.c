/*
 * src/backend/cpu/cpu_dispatch.c - 2D dispatch table for CPU kernels
 *
 * Implements a static [op][dtype] table of kernel function pointers.
 * Each entry is a thin wrapper that adapts the uniform kernel signature
 * to the actual kernel API (some kernels omit scratch or pool). NULL
 * entries mean the (op, dtype) combination is not yet implemented and
 * will return SAM3_EDTYPE. Only F32 entries are populated here; fp16
 * and bf16 entries are filled in later tasks.
 *
 * Key types:  sam3_kernel_fn
 * Depends on: cpu_dispatch.h, kernels/cpu_kernels.h, core/tensor.h,
 *             core/trace.h, util/log.h, util/threadpool.h
 * Used by:    cpu_backend.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "cpu_dispatch.h"
#include "kernels/cpu_kernels.h"
#include "core/tensor.h"
#include "core/trace.h"
#include "util/log.h"
#include "util/threadpool.h"

/*
 * Uniform kernel function pointer type. All entries in the dispatch
 * table share this signature; wrappers below adapt each real kernel.
 */
typedef enum sam3_error (*sam3_kernel_fn)(const struct sam3_node *,
					  struct sam3_arena *,
					  struct sam3_threadpool *);

/* ── Wrapper functions ─────────────────────────────────────────────── */

static enum sam3_error
wrap_matmul(const struct sam3_node *node, struct sam3_arena *scratch,
	    struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_matmul(node, pool);
}

static enum sam3_error
wrap_add(const struct sam3_node *node, struct sam3_arena *scratch,
	 struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_add(node, pool);
}

static enum sam3_error
wrap_mul(const struct sam3_node *node, struct sam3_arena *scratch,
	 struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_mul(node, pool);
}

static enum sam3_error
wrap_softmax(const struct sam3_node *node, struct sam3_arena *scratch,
	     struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_softmax(node, pool);
}

static enum sam3_error
wrap_relu(const struct sam3_node *node, struct sam3_arena *scratch,
	  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_relu(node, pool);
}

static enum sam3_error
wrap_gelu(const struct sam3_node *node, struct sam3_arena *scratch,
	  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_gelu(node, pool);
}

static enum sam3_error
wrap_layernorm(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_layernorm(node, pool);
}

static enum sam3_error
wrap_conv2d(const struct sam3_node *node, struct sam3_arena *scratch,
	    struct sam3_threadpool *pool)
{
	return cpu_kernel_conv2d(node, scratch, pool);
}

static enum sam3_error
wrap_reshape(const struct sam3_node *node, struct sam3_arena *scratch,
	     struct sam3_threadpool *pool)
{
	(void)scratch;
	(void)pool;
	return cpu_kernel_reshape(node);
}

static enum sam3_error
wrap_transpose(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_transpose(node, pool);
}

static enum sam3_error
wrap_add_f16(const struct sam3_node *node, struct sam3_arena *scratch,
	     struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_add_f16(node, pool);
}

static enum sam3_error
wrap_mul_f16(const struct sam3_node *node, struct sam3_arena *scratch,
	     struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_mul_f16(node, pool);
}

static enum sam3_error
wrap_relu_f16(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_relu_f16(node, pool);
}

static enum sam3_error
wrap_gelu_f16(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_gelu_f16(node, pool);
}

static enum sam3_error
wrap_matmul_f16(const struct sam3_node *node, struct sam3_arena *scratch,
		struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_matmul_f16(node, pool);
}

static enum sam3_error
wrap_softmax_f16(const struct sam3_node *node, struct sam3_arena *scratch,
		 struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_softmax_f16(node, pool);
}

static enum sam3_error
wrap_layernorm_f16(const struct sam3_node *node, struct sam3_arena *scratch,
		   struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_layernorm_f16(node, pool);
}

static enum sam3_error
wrap_conv2d_f16(const struct sam3_node *node, struct sam3_arena *scratch,
		struct sam3_threadpool *pool)
{
	/* Conv2d requires two inputs (input tensor + weight tensor). */
	if (node->n_inputs < 2 || !node->inputs[1])
		return SAM3_EDTYPE;
	return cpu_kernel_conv2d_f16(node, scratch, pool);
}

static enum sam3_error
wrap_add_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_add_bf16(node, pool);
}

static enum sam3_error
wrap_mul_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_mul_bf16(node, pool);
}

static enum sam3_error
wrap_relu_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_relu_bf16(node, pool);
}

static enum sam3_error
wrap_gelu_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_gelu_bf16(node, pool);
}

/* ── Dispatch table ────────────────────────────────────────────────── */

/*
 * cpu_dispatch_table[op][dtype] — NULL means not implemented.
 * Reshape is registered for all dtypes because it is dtype-agnostic.
 * All other ops are F32-only until later tasks add fp16/bf16 kernels.
 */
static const sam3_kernel_fn
cpu_dispatch_table[SAM3_OP_COUNT][SAM3_DTYPE_COUNT] = {
	[SAM3_OP_MATMUL] = {
		[SAM3_DTYPE_F32]  = wrap_matmul,
		[SAM3_DTYPE_F16]  = wrap_matmul_f16,
	},
	[SAM3_OP_ADD] = {
		[SAM3_DTYPE_F32]  = wrap_add,
		[SAM3_DTYPE_F16]  = wrap_add_f16,
		[SAM3_DTYPE_BF16] = wrap_add_bf16,
	},
	[SAM3_OP_MUL] = {
		[SAM3_DTYPE_F32]  = wrap_mul,
		[SAM3_DTYPE_F16]  = wrap_mul_f16,
		[SAM3_DTYPE_BF16] = wrap_mul_bf16,
	},
	[SAM3_OP_SOFTMAX] = {
		[SAM3_DTYPE_F32]  = wrap_softmax,
		[SAM3_DTYPE_F16]  = wrap_softmax_f16,
	},
	[SAM3_OP_RELU] = {
		[SAM3_DTYPE_F32]  = wrap_relu,
		[SAM3_DTYPE_F16]  = wrap_relu_f16,
		[SAM3_DTYPE_BF16] = wrap_relu_bf16,
	},
	[SAM3_OP_GELU] = {
		[SAM3_DTYPE_F32]  = wrap_gelu,
		[SAM3_DTYPE_F16]  = wrap_gelu_f16,
		[SAM3_DTYPE_BF16] = wrap_gelu_bf16,
	},
	[SAM3_OP_LAYERNORM] = {
		[SAM3_DTYPE_F32]  = wrap_layernorm,
		[SAM3_DTYPE_F16]  = wrap_layernorm_f16,
	},
	[SAM3_OP_CONV2D] = {
		[SAM3_DTYPE_F32]  = wrap_conv2d,
		[SAM3_DTYPE_F16]  = wrap_conv2d_f16,
	},
	[SAM3_OP_RESHAPE] = {
		[SAM3_DTYPE_F32]  = wrap_reshape,
		[SAM3_DTYPE_F16]  = wrap_reshape,
		[SAM3_DTYPE_BF16] = wrap_reshape,
		[SAM3_DTYPE_I32]  = wrap_reshape,
		[SAM3_DTYPE_I8]   = wrap_reshape,
	},
	[SAM3_OP_TRANSPOSE] = {
		[SAM3_DTYPE_F32]  = wrap_transpose,
		[SAM3_DTYPE_F16]  = wrap_transpose,
		[SAM3_DTYPE_BF16] = wrap_transpose,
		[SAM3_DTYPE_I32]  = wrap_transpose,
		[SAM3_DTYPE_I8]   = wrap_transpose,
	},
};

/* ── Dispatch function ─────────────────────────────────────────────── */

/*
 * cpu_dispatch_node - Dispatch a graph node to the correct dtype-specific kernel.
 *
 * @node:    Node to execute. node->inputs[0] determines the dtype.
 * @scratch: Scratch arena passed through to kernels that need temp buffers.
 * @pool:    Thread pool passed through to kernels that parallelise.
 *
 * Returns SAM3_OK on success, SAM3_EDTYPE for dtype mismatches or when
 * the (op, dtype) pair has no registered kernel.
 */
enum sam3_error
cpu_dispatch_node(const struct sam3_node *node,
		  struct sam3_arena *scratch,
		  struct sam3_threadpool *pool)
{
	enum sam3_dtype dtype;
	sam3_kernel_fn fn;
	enum sam3_error err;

	/* SAM3_OP_NONE is a no-op; nothing to dispatch. */
	if (node->op == SAM3_OP_NONE)
		return SAM3_OK;

	/* Require at least one input to determine dtype. */
	if (node->n_inputs < 1 || !node->inputs[0]) {
		sam3_log_error("cpu_dispatch: node op=%d has no inputs",
			       node->op);
		return SAM3_EINVAL;
	}

	dtype = node->inputs[0]->dtype;

	/* All inputs must share the same dtype. */
	for (int i = 1; i < node->n_inputs; i++) {
		if (!node->inputs[i])
			continue;
		if (node->inputs[i]->dtype != dtype) {
			sam3_log_error(
				"cpu_dispatch: dtype mismatch input[0]=%d "
				"input[%d]=%d",
				dtype, i, node->inputs[i]->dtype);
			return SAM3_EDTYPE;
		}
	}

	/* Range-check op and dtype indices. */
	if ((unsigned)node->op >= (unsigned)SAM3_OP_COUNT ||
	    (unsigned)dtype >= (unsigned)SAM3_DTYPE_COUNT) {
		sam3_log_error("cpu_dispatch: out-of-range op=%d dtype=%d",
			       node->op, dtype);
		return SAM3_EDTYPE;
	}

	fn = cpu_dispatch_table[node->op][dtype];
	if (!fn) {
		sam3_log_error(
			"cpu_dispatch: no kernel for op=%d dtype=%d",
			node->op, dtype);
		return SAM3_EDTYPE;
	}

	SAM3_TRACE_KERNEL(sam3_op_str(node->op),
			  dtype,
			  node->output ? node->output->dtype : dtype,
			  "cpu");

	err = fn(node, scratch, pool);

	if (err == SAM3_OK && node->output) {
		SAM3_TRACE_NUMERIC(sam3_op_str(node->op), node->output);
	}

	return err;
}
