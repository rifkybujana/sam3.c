/*
 * src/backend/cpu/cpu_dispatch.c - 2D dispatch table for CPU kernels
 *
 * Implements a static [op][dtype] table of kernel function pointers.
 * Each entry is a thin wrapper that adapts the uniform kernel signature
 * to the actual kernel API (some kernels omit scratch or pool). NULL
 * entries mean the (op, dtype) combination is not yet implemented and
 * will return SAM3_EDTYPE. F32, fp16, and bf16 entries are populated
 * for all compute ops; reshape and transpose are dtype-agnostic.
 *
 * Key types:  sam3_kernel_fn
 * Depends on: cpu_dispatch.h, kernels/cpu_kernels.h, core/tensor.h,
 *             core/trace.h, util/log.h, util/threadpool.h
 * Used by:    cpu_backend.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
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

static enum sam3_error
wrap_matmul_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
		 struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_matmul_bf16(node, pool);
}

static enum sam3_error
wrap_matmul_q8(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_matmul_q8(node, pool);
}

static enum sam3_error
wrap_softmax_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
		  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_softmax_bf16(node, pool);
}

static enum sam3_error
wrap_layernorm_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
		    struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_layernorm_bf16(node, pool);
}

static enum sam3_error
wrap_conv2d_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
		 struct sam3_threadpool *pool)
{
	if (node->n_inputs < 2 || !node->inputs[1])
		return SAM3_EDTYPE;
	return cpu_kernel_conv2d_bf16(node, scratch, pool);
}

static enum sam3_error
wrap_cast(const struct sam3_node *node, struct sam3_arena *scratch,
	  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_cast(node, pool);
}

static enum sam3_error
wrap_sigmoid(const struct sam3_node *node, struct sam3_arena *scratch,
	     struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_sigmoid(node, pool);
}

static enum sam3_error
wrap_sigmoid_f16(const struct sam3_node *node, struct sam3_arena *scratch,
		 struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_sigmoid_f16(node, pool);
}

static enum sam3_error
wrap_sigmoid_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
		  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_sigmoid_bf16(node, pool);
}

static enum sam3_error
wrap_silu(const struct sam3_node *node, struct sam3_arena *scratch,
	  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_silu(node, pool);
}

static enum sam3_error
wrap_silu_f16(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_silu_f16(node, pool);
}

static enum sam3_error
wrap_silu_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_silu_bf16(node, pool);
}

static enum sam3_error
wrap_embed(const struct sam3_node *node, struct sam3_arena *scratch,
	   struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_embed(node, pool);
}

static enum sam3_error
wrap_concat(const struct sam3_node *node, struct sam3_arena *scratch,
	    struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_concat(node, pool);
}

static enum sam3_error
wrap_slice(const struct sam3_node *node, struct sam3_arena *scratch,
	   struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_slice(node, pool);
}

static enum sam3_error
wrap_upsample(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_upsample(node, pool);
}

static enum sam3_error
wrap_rope(const struct sam3_node *node, struct sam3_arena *scratch,
	  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_rope(node, pool);
}

static enum sam3_error
wrap_rope_f16(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_rope_f16(node, pool);
}

static enum sam3_error
wrap_rope_bf16(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_rope_bf16(node, pool);
}

static enum sam3_error
wrap_conv_transpose2d(const struct sam3_node *node, struct sam3_arena *scratch,
		      struct sam3_threadpool *pool)
{
	return cpu_kernel_conv_transpose2d(node, scratch, pool);
}

static enum sam3_error
wrap_maxpool2d(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	if (node->params[2])
		return cpu_maxpool2d_nhwc_wrap(node, scratch, pool);
	(void)scratch;
	return cpu_kernel_maxpool2d(node, pool);
}

static enum sam3_error
wrap_sdpa(const struct sam3_node *node, struct sam3_arena *scratch,
	  struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_sdpa(node, pool);
}

static enum sam3_error
wrap_groupnorm(const struct sam3_node *node, struct sam3_arena *scratch,
	       struct sam3_threadpool *pool)
{
	(void)scratch;
	return cpu_kernel_groupnorm(node, pool);
}

/*
 * wrap_bias_add - Bias add with layout switch.
 *
 * params[2] == 0: NCHW [N,C,H,W] + bias[C] broadcast over HW.
 * params[2] == 1: NHWC [N,H,W,C] + bias[C] broadcast over pixels.
 *
 * The NCHW path walks the bias once per (n,c) plane and sums across
 * a contiguous HW block; the NHWC path walks the bias across the
 * innermost dimension so each pixel just adds the per-channel value.
 */
static enum sam3_error
wrap_bias_add(const struct sam3_node *node, struct sam3_arena *scratch,
	      struct sam3_threadpool *pool)
{
	(void)scratch;
	(void)pool;

	const struct sam3_tensor *x = node->inputs[0];
	const struct sam3_tensor *bias = node->inputs[1];
	struct sam3_tensor *out = node->output;
	const float *xd = (const float *)x->data;
	const float *bd = (const float *)bias->data;
	float *od = (float *)out->data;

	if (node->params[2]) {
		/* NHWC: dims = [N, H, W, C] */
		int N = x->dims[0];
		int H = x->dims[1];
		int W = x->dims[2];
		int C = x->dims[3];
		size_t pixels = (size_t)N * H * W;

		for (size_t p = 0; p < pixels; p++) {
			size_t off = p * (size_t)C;
			for (int c = 0; c < C; c++)
				od[off + c] = xd[off + c] + bd[c];
		}
		return SAM3_OK;
	}

	int N = x->dims[0];
	int C = x->dims[1];
	int H = x->dims[2];
	int W = x->dims[3];
	int spatial = H * W;

	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			float b = bd[c];
			int off = (n * C + c) * spatial;
			for (int i = 0; i < spatial; i++)
				od[off + i] = xd[off + i] + b;
		}
	}

	return SAM3_OK;
}

/* ── Dispatch table ────────────────────────────────────────────────── */

/*
 * cpu_dispatch_table[op][dtype] — NULL means not implemented.
 * Reshape and transpose are registered for all dtypes (dtype-agnostic).
 * Cast is registered for all floating-point source dtypes.
 */
static const sam3_kernel_fn
cpu_dispatch_table[SAM3_OP_COUNT][SAM3_DTYPE_COUNT] = {
	[SAM3_OP_MATMUL] = {
		[SAM3_DTYPE_F32]  = wrap_matmul,
		[SAM3_DTYPE_F16]  = wrap_matmul_f16,
		[SAM3_DTYPE_BF16] = wrap_matmul_bf16,
		[SAM3_DTYPE_Q8_0] = wrap_matmul_q8,
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
		[SAM3_DTYPE_BF16] = wrap_softmax_bf16,
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
		[SAM3_DTYPE_BF16] = wrap_layernorm_bf16,
	},
	[SAM3_OP_CONV2D] = {
		[SAM3_DTYPE_F32]  = wrap_conv2d,
		[SAM3_DTYPE_F16]  = wrap_conv2d_f16,
		[SAM3_DTYPE_BF16] = wrap_conv2d_bf16,
	},
	[SAM3_OP_RESHAPE] = {
		[SAM3_DTYPE_F32]  = wrap_reshape,
		[SAM3_DTYPE_F16]  = wrap_reshape,
		[SAM3_DTYPE_BF16] = wrap_reshape,
		[SAM3_DTYPE_I32]  = wrap_reshape,
		[SAM3_DTYPE_I8]   = wrap_reshape,
		[SAM3_DTYPE_Q8_0] = wrap_reshape,
	},
	[SAM3_OP_TRANSPOSE] = {
		[SAM3_DTYPE_F32]  = wrap_transpose,
		[SAM3_DTYPE_F16]  = wrap_transpose,
		[SAM3_DTYPE_BF16] = wrap_transpose,
		[SAM3_DTYPE_I32]  = wrap_transpose,
		[SAM3_DTYPE_I8]   = wrap_transpose,
		[SAM3_DTYPE_Q8_0] = wrap_transpose,
	},
	[SAM3_OP_CAST] = {
		[SAM3_DTYPE_F32]  = wrap_cast,
		[SAM3_DTYPE_F16]  = wrap_cast,
		[SAM3_DTYPE_BF16] = wrap_cast,
	},
	[SAM3_OP_SIGMOID] = {
		[SAM3_DTYPE_F32]  = wrap_sigmoid,
		[SAM3_DTYPE_F16]  = wrap_sigmoid_f16,
		[SAM3_DTYPE_BF16] = wrap_sigmoid_bf16,
	},
	[SAM3_OP_SILU] = {
		[SAM3_DTYPE_F32]  = wrap_silu,
		[SAM3_DTYPE_F16]  = wrap_silu_f16,
		[SAM3_DTYPE_BF16] = wrap_silu_bf16,
	},
	[SAM3_OP_EMBED] = {
		[SAM3_DTYPE_F32]  = wrap_embed,
	},
	[SAM3_OP_CONCAT] = {
		[SAM3_DTYPE_F32]  = wrap_concat,
		[SAM3_DTYPE_F16]  = wrap_concat,
		[SAM3_DTYPE_BF16] = wrap_concat,
	},
	[SAM3_OP_SLICE] = {
		[SAM3_DTYPE_F32]  = wrap_slice,
		[SAM3_DTYPE_F16]  = wrap_slice,
		[SAM3_DTYPE_BF16] = wrap_slice,
	},
	[SAM3_OP_UPSAMPLE] = {
		[SAM3_DTYPE_F32]  = wrap_upsample,
		[SAM3_DTYPE_F16]  = wrap_upsample,
		[SAM3_DTYPE_BF16] = wrap_upsample,
	},
	[SAM3_OP_ROPE] = {
		[SAM3_DTYPE_F32]  = wrap_rope,
		[SAM3_DTYPE_F16]  = wrap_rope_f16,
		[SAM3_DTYPE_BF16] = wrap_rope_bf16,
	},
	[SAM3_OP_CONV_TRANSPOSE2D] = {
		[SAM3_DTYPE_F32]  = wrap_conv_transpose2d,
	},
	[SAM3_OP_MAXPOOL2D] = {
		[SAM3_DTYPE_F32]  = wrap_maxpool2d,
	},
	[SAM3_OP_SDPA] = {
		[SAM3_DTYPE_F32]  = wrap_sdpa,
	},
	[SAM3_OP_BIAS_ADD] = {
		[SAM3_DTYPE_F32]  = wrap_bias_add,
	},
	[SAM3_OP_GROUPNORM] = {
		[SAM3_DTYPE_F32]  = wrap_groupnorm,
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

	/*
	 * All inputs must share the same dtype, EXCEPT:
	 * - MATMUL with Q8_0: input[0]=F32 (activations), input[1]=Q8_0 (weights)
	 *   Dispatch on input[1]'s dtype (Q8_0) to reach the mixed kernel.
	 * - EMBED: input[0]=F32 (table), input[1]=I32 (indices)
	 *   Dispatch on input[0]'s dtype (F32); kernel validates I32 internally.
	 * - ROPE with F16/BF16: input[0]=F16/BF16, inputs[1..2]=F32 (cos/sin)
	 *   Dispatch on input[0]'s dtype; kernel validates F32 cos/sin internally.
	 */
	int mixed_q8 = (node->op == SAM3_OP_MATMUL &&
			node->n_inputs >= 2 &&
			node->inputs[1] &&
			node->inputs[1]->dtype == SAM3_DTYPE_Q8_0 &&
			dtype == SAM3_DTYPE_F32);

	int mixed_embed = (node->op == SAM3_OP_EMBED &&
			   node->n_inputs >= 2 &&
			   node->inputs[1] &&
			   node->inputs[1]->dtype == SAM3_DTYPE_I32 &&
			   dtype == SAM3_DTYPE_F32);

	int mixed_rope = (node->op == SAM3_OP_ROPE &&
			  node->n_inputs >= 3 &&
			  node->inputs[1] && node->inputs[2] &&
			  node->inputs[1]->dtype == SAM3_DTYPE_F32 &&
			  node->inputs[2]->dtype == SAM3_DTYPE_F32 &&
			  (dtype == SAM3_DTYPE_F16 ||
			   dtype == SAM3_DTYPE_BF16));

	if (mixed_q8) {
		dtype = SAM3_DTYPE_Q8_0;
	} else if (!mixed_embed && !mixed_rope) {
		for (int i = 1; i < node->n_inputs; i++) {
			if (!node->inputs[i])
				continue;
			if (node->inputs[i]->dtype != dtype) {
				sam3_log_error(
					"cpu_dispatch: dtype mismatch "
					"input[0]=%d input[%d]=%d",
					dtype, i,
					node->inputs[i]->dtype);
				return SAM3_EDTYPE;
			}
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
