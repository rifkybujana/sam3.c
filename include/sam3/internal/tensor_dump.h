/*
 * include/sam3/internal/tensor_dump.h - Binary tensor dump for validation.
 *
 * Writes sam3_tensor data to a simple binary format for cross-checking
 * against the Python reference. Format: int32 n_dims, int32[n_dims] shape,
 * then raw float32 data. All values little-endian.
 *
 * Key types:  none
 * Depends on: core/tensor.h
 * Used by:    tools/sam3_main.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_INTERNAL_TENSOR_DUMP_H
#define SAM3_INTERNAL_TENSOR_DUMP_H

struct sam3_tensor;

/*
 * sam3_tensor_dump - Write a tensor to a binary file.
 *
 * @path:   Output file path
 * @tensor: Tensor to dump (must be F32 dtype, contiguous)
 *
 * File format: int32 n_dims | int32[n_dims] dims | float32[...] data
 *
 * Returns 0 on success, -1 on error (NULL args, non-F32, I/O failure).
 */
int sam3_tensor_dump(const char *path, const struct sam3_tensor *tensor);

#endif /* SAM3_INTERNAL_TENSOR_DUMP_H */
