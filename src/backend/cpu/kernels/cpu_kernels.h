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

#endif /* SAM3_CPU_KERNELS_H */
