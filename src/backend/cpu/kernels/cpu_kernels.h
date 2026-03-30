/*
 * src/backend/cpu/kernels/cpu_kernels.h - CPU kernel declarations
 *
 * Declares all CPU compute kernels. Each kernel validates its inputs,
 * selects scalar or SIMD path, and operates on the node's tensor data.
 * All kernels expect F32 tensors; other dtypes return SAM3_EINVAL.
 *
 * Key types:  (function declarations only)
 * Depends on: core/graph.h, core/alloc.h, sam3/sam3_types.h
 * Used by:    cpu_backend.c (dispatch switch)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CPU_KERNELS_H
#define SAM3_CPU_KERNELS_H

#include "core/graph.h"
#include "core/alloc.h"
#include "sam3/sam3_types.h"

/* Matrix multiply: inputs[0] @ inputs[1] -> output */
enum sam3_error cpu_kernel_matmul(const struct sam3_node *node);

/* Element-wise add with broadcasting: inputs[0] + inputs[1] -> output */
enum sam3_error cpu_kernel_add(const struct sam3_node *node);

/* Element-wise multiply with broadcasting: inputs[0] * inputs[1] -> output */
enum sam3_error cpu_kernel_mul(const struct sam3_node *node);

/* Row-wise softmax along last dimension */
enum sam3_error cpu_kernel_softmax(const struct sam3_node *node);

/* Element-wise ReLU: max(0, x) */
enum sam3_error cpu_kernel_relu(const struct sam3_node *node);

/* Element-wise GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
enum sam3_error cpu_kernel_gelu(const struct sam3_node *node);

/* Layer normalization with optional gamma/beta */
enum sam3_error cpu_kernel_layernorm(const struct sam3_node *node);

/* Conv2D via im2col + matmul. scratch arena for temp buffers. */
enum sam3_error cpu_kernel_conv2d(const struct sam3_node *node,
				  struct sam3_arena *scratch);

/* Reshape: zero-copy, sets output->data = input->data */
enum sam3_error cpu_kernel_reshape(const struct sam3_node *node);

/* Transpose: 2D copy with optional SIMD block transpose */
enum sam3_error cpu_kernel_transpose(const struct sam3_node *node);

#endif /* SAM3_CPU_KERNELS_H */
