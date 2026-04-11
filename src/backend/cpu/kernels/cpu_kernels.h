/*
 * src/backend/cpu/kernels/cpu_kernels.h - CPU kernel declarations
 *
 * Declares all CPU compute kernels for f32, fp16, and bf16 dtypes.
 * Each kernel validates its inputs, selects scalar or SIMD path, and
 * operates on the node's tensor data. fp16 kernels use native NEON
 * fp16 arithmetic (ARMv8.2-A+) or scalar fallback; bf16 kernels always
 * upcast to f32 for computation. Cast kernel converts between dtypes.
 *
 * Key types:  (function declarations only)
 * Depends on: core/graph.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    cpu_dispatch.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_KERNELS_H
#define SAM3_CPU_KERNELS_H

#include "core/graph.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

struct sam3_threadpool;

/* Matrix multiply: inputs[0] @ inputs[1] -> output */
enum sam3_error cpu_kernel_matmul(const struct sam3_node *node,
				  struct sam3_threadpool *pool);

/* Element-wise add with broadcasting: inputs[0] + inputs[1] -> output */
enum sam3_error cpu_kernel_add(const struct sam3_node *node,
			       struct sam3_threadpool *pool);

/* Element-wise multiply with broadcasting: inputs[0] * inputs[1] -> output */
enum sam3_error cpu_kernel_mul(const struct sam3_node *node,
			       struct sam3_threadpool *pool);

/* Row-wise softmax along last dimension */
enum sam3_error cpu_kernel_softmax(const struct sam3_node *node,
				   struct sam3_threadpool *pool);

/* Element-wise ReLU: max(0, x) */
enum sam3_error cpu_kernel_relu(const struct sam3_node *node,
				struct sam3_threadpool *pool);

/* Element-wise GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
enum sam3_error cpu_kernel_gelu(const struct sam3_node *node,
				struct sam3_threadpool *pool);

/* Layer normalization with optional gamma/beta */
enum sam3_error cpu_kernel_layernorm(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* Conv2D via im2col + matmul. scratch arena for temp buffers. */
enum sam3_error cpu_kernel_conv2d(const struct sam3_node *node,
				  struct sam3_arena *scratch,
				  struct sam3_threadpool *pool);

/*
 * Uniform kernel signature used by the NHWC wrappers below. Matches
 * the real cpu_kernel_conv2d* and cpu_kernel_conv_transpose2d
 * signatures so the wrappers can recursively re-enter the NCHW path
 * with params[2] cleared.
 */
typedef enum sam3_error (*cpu_conv_kernel_fn)(const struct sam3_node *,
					      struct sam3_arena *,
					      struct sam3_threadpool *);

/*
 * cpu_conv2d_nhwc_wrap - Run a CONV2D kernel on NHWC inputs by
 * transposing to NCHW into scratch, calling the legacy NCHW path,
 * and transposing the result back to NHWC.
 *
 * @kernel:  The dtype-specific conv2d kernel to invoke on the NCHW
 *           scratch copies. Must take the uniform signature above.
 * @node:    Original node with NHWC input [N,H,W,C], OHWI weight
 *           [OC,KH,KW,IC], NHWC output [N,OH,OW,OC], and
 *           params[2]==1.
 * @scratch: Scratch arena for the NCHW intermediate buffers; offset
 *           is saved/restored on every return path.
 * @pool:    Thread pool forwarded to the inner NCHW kernel.
 *
 * The wrapper does not touch the output tensor's dims; it only
 * writes bytes into output->data. The CPU conv path is test-only,
 * so the transposes are plain scalar loops over dtype_size bytes.
 */
enum sam3_error cpu_conv2d_nhwc_wrap(cpu_conv_kernel_fn kernel,
				     const struct sam3_node *node,
				     struct sam3_arena *scratch,
				     struct sam3_threadpool *pool);

/*
 * cpu_conv_transpose2d_nhwc_wrap - Same as cpu_conv2d_nhwc_wrap but
 * for transposed convolution, whose weight permutation differs.
 *
 * Input is NHWC [N,H,W,C_in]; weight is OHWI [C_out,KH,KW,C_in] and
 * must be repacked to IOHW [C_in,C_out,KH,KW] for the legacy kernel.
 * Output is NHWC [N,OH,OW,C_out].
 */
enum sam3_error
cpu_conv_transpose2d_nhwc_wrap(cpu_conv_kernel_fn kernel,
			       const struct sam3_node *node,
			       struct sam3_arena *scratch,
			       struct sam3_threadpool *pool);

/* Reshape: zero-copy, sets output->data = input->data */
enum sam3_error cpu_kernel_reshape(const struct sam3_node *node);

/* Transpose: 2D copy with optional SIMD block transpose */
enum sam3_error cpu_kernel_transpose(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* FP16 elementwise kernels */
enum sam3_error cpu_kernel_add_f16(const struct sam3_node *node,
				   struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_mul_f16(const struct sam3_node *node,
				   struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_relu_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_gelu_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);

/* FP16 matmul kernel */
enum sam3_error cpu_kernel_matmul_f16(const struct sam3_node *node,
				      struct sam3_threadpool *pool);

/* FP16 softmax and layernorm */
enum sam3_error cpu_kernel_softmax_f16(const struct sam3_node *node,
				       struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_layernorm_f16(const struct sam3_node *node,
					 struct sam3_threadpool *pool);

/* FP16 conv2d kernel */
enum sam3_error cpu_kernel_conv2d_f16(const struct sam3_node *node,
				      struct sam3_arena *scratch,
				      struct sam3_threadpool *pool);

/* BF16 elementwise kernels */
enum sam3_error cpu_kernel_add_bf16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_mul_bf16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_relu_bf16(const struct sam3_node *node,
				     struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_gelu_bf16(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* BF16 matmul kernel */
enum sam3_error cpu_kernel_matmul_bf16(const struct sam3_node *node,
				       struct sam3_threadpool *pool);

/* BF16 softmax and layernorm */
enum sam3_error cpu_kernel_softmax_bf16(const struct sam3_node *node,
					struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_layernorm_bf16(const struct sam3_node *node,
					  struct sam3_threadpool *pool);

/* BF16 conv2d kernel */
enum sam3_error cpu_kernel_conv2d_bf16(const struct sam3_node *node,
				       struct sam3_arena *scratch,
				       struct sam3_threadpool *pool);

/* Cast between dtypes: node->params[0] = target dtype */
enum sam3_error cpu_kernel_cast(const struct sam3_node *node,
				struct sam3_threadpool *pool);

/* Q8_0 mixed-dtype matmul: A[F32] @ B[Q8_0] -> C[F32] */
enum sam3_error cpu_kernel_matmul_q8(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* Element-wise sigmoid: 1/(1+exp(-x)) */
enum sam3_error cpu_kernel_sigmoid(const struct sam3_node *node,
				   struct sam3_threadpool *pool);

/* Element-wise SiLU (Swish): x * sigmoid(x) */
enum sam3_error cpu_kernel_silu(const struct sam3_node *node,
				struct sam3_threadpool *pool);

/* FP16 sigmoid and silu */
enum sam3_error cpu_kernel_sigmoid_f16(const struct sam3_node *node,
				       struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_sigmoid_bf16(const struct sam3_node *node,
					struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_silu_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_silu_bf16(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* Embedding table lookup: table[indices[i], :] -> output[i, :] */
enum sam3_error cpu_kernel_embed(const struct sam3_node *node,
				 struct sam3_threadpool *pool);

/* Concatenate tensors along axis: params[0]=axis */
enum sam3_error cpu_kernel_concat(const struct sam3_node *node,
				  struct sam3_threadpool *pool);

/* Slice sub-tensor: params[0]=axis, [1]=start, [2]=end */
enum sam3_error cpu_kernel_slice(const struct sam3_node *node,
				 struct sam3_threadpool *pool);

/* Nearest-neighbor upsample 4D [N,C,H,W]: params[0]=scale */
enum sam3_error cpu_kernel_upsample(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* Rotary position embedding: rotate pairs by precomputed frequencies */
enum sam3_error cpu_kernel_rope(const struct sam3_node *node,
				struct sam3_threadpool *pool);

/* FP16/BF16 rope (mixed-dtype: input F16/BF16, cos/sin F32) */
enum sam3_error cpu_kernel_rope_f16(const struct sam3_node *node,
				    struct sam3_threadpool *pool);
enum sam3_error cpu_kernel_rope_bf16(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/* Transposed conv2d: scatter-accumulate. scratch arena for temp buffers. */
enum sam3_error cpu_kernel_conv_transpose2d(const struct sam3_node *node,
					    struct sam3_arena *scratch,
					    struct sam3_threadpool *pool);

/* Max pooling 2D: sliding-window max. params[0]=kernel, [1]=stride.
 * NCHW path only; the NHWC wrapper below handles params[2]==1. */
enum sam3_error cpu_kernel_maxpool2d(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

/*
 * cpu_maxpool2d_nhwc_wrap - Run the NCHW maxpool kernel on NHWC input
 * by transposing to NCHW into scratch, calling the legacy path, and
 * transposing the result back.
 *
 * @node:    Original node with NHWC input [N,H,W,C], NHWC output
 *           [N,OH,OW,C], and params[2]==1.
 * @scratch: Scratch arena for the NCHW intermediate buffers; offset
 *           is saved/restored on every return path.
 * @pool:    Thread pool forwarded to the inner NCHW kernel.
 *
 * Used by the FPN neck's scale=0.5 stage on the CPU backend. The
 * production path runs on Metal (NHWC-native); this shim keeps the
 * CPU test surface green without forking a second maxpool kernel.
 */
enum sam3_error cpu_maxpool2d_nhwc_wrap(const struct sam3_node *node,
					struct sam3_arena *scratch,
					struct sam3_threadpool *pool);

/*
 * Tiled scaled dot-product attention.
 * inputs: Q[seq,hd], K[seq,hd], V[seq,hd], mask[seq,seq]?
 * params[0]=head_dim. Uses online softmax, never materializes [seq,seq].
 */
enum sam3_error cpu_kernel_sdpa(const struct sam3_node *node,
				struct sam3_threadpool *pool);

/* Group normalization on NCHW input.
 * inputs[0]=x[N,C,H,W], inputs[1]=gamma[C], inputs[2]=beta[C].
 * params[0]=num_groups. eps=1e-5.
 */
enum sam3_error cpu_kernel_groupnorm(const struct sam3_node *node,
				     struct sam3_threadpool *pool);

#endif /* SAM3_CPU_KERNELS_H */
