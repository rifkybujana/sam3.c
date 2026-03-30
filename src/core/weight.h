/*
 * src/core/weight.h - Weight file format and loader
 *
 * Defines the .sam3 binary weight format (header, tensor descriptors),
 * the mmap-based native loader, and the reader vtable interface for
 * converting external formats (SafeTensors, GGUF). The native loader
 * maps the file and provides O(1) tensor lookup via FNV-1a hashing.
 *
 * Key types:  sam3_weight_file, sam3_weight_header, sam3_weight_tensor_desc,
 *             weight_reader, weight_reader_ops
 * Depends on: sam3/sam3_types.h, core/tensor.h
 * Used by:    sam3.c, weight.c, weight_safetensors.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_WEIGHT_H
#define SAM3_CORE_WEIGHT_H

#include <stdint.h>
#include "sam3/sam3_types.h"
#include "core/tensor.h"

/* .sam3 file magic: ASCII "SAM3" = 0x53414D33 (little-endian: 0x334D4153) */
#define SAM3_WEIGHT_MAGIC   0x334D4153
#define SAM3_WEIGHT_VERSION 1

#define SAM3_WEIGHT_NAME_MAX   64
#define SAM3_WEIGHT_DATA_ALIGN 64   /* per-tensor alignment in data blob */
#define SAM3_WEIGHT_PAGE_ALIGN 4096 /* data blob start alignment */

/* ── On-disk structures (little-endian) ─────────────────────────────── */

/*
 * File header: 48 bytes.
 * Sits at offset 0 in the .sam3 file.
 */
struct sam3_weight_header {
	uint32_t magic;
	uint32_t version;
	uint32_t flags;
	uint32_t n_tensors;
	int32_t  image_size;
	int32_t  encoder_dim;
	int32_t  decoder_dim;
	int32_t  n_encoder_layers;
	int32_t  n_decoder_layers;
	uint32_t reserved[3];
};

/*
 * Tensor descriptor: 112 bytes.
 * Array of n_tensors descriptors follows the header.
 */
struct sam3_weight_tensor_desc {
	char     name[SAM3_WEIGHT_NAME_MAX];
	uint32_t dtype;
	uint32_t n_dims;
	int32_t  dims[SAM3_MAX_DIMS];
	uint64_t data_offset;  /* offset from start of data blob */
	uint64_t data_size;    /* total bytes */
	uint64_t reserved;
};

_Static_assert(sizeof(struct sam3_weight_header) == 48,
	       "header must be exactly 48 bytes on disk");
_Static_assert(sizeof(struct sam3_weight_tensor_desc) == 112,
	       "tensor desc must be exactly 112 bytes on disk");

/* ── Runtime loader ─────────────────────────────────────────────────── */

/*
 * Handle to a memory-mapped .sam3 weight file.
 * Tensor data pointers are valid until sam3_weight_close().
 */
struct sam3_weight_file {
	void                                *mapped;
	size_t                               mapped_size;
	const struct sam3_weight_header      *header;
	const struct sam3_weight_tensor_desc *tensors;
	const void                           *data_base;
	uint32_t                             *hash_table;
	uint32_t                              hash_capacity;
};

#define SAM3_WEIGHT_HASH_EMPTY UINT32_MAX

/*
 * sam3_weight_open - Open and validate a .sam3 weight file via mmap.
 *
 * @wf:   Weight file handle (caller-allocated, zeroed)
 * @path: Path to .sam3 file
 *
 * Memory-maps the file, validates the header, and builds a hash table
 * for O(1) tensor lookup. Returns SAM3_OK on success.
 */
enum sam3_error sam3_weight_open(struct sam3_weight_file *wf,
				 const char *path);

/*
 * sam3_weight_close - Unmap the weight file and free the hash table.
 *
 * @wf: Weight file handle (may be zeroed / never opened).
 */
void sam3_weight_close(struct sam3_weight_file *wf);

/*
 * sam3_weight_find - Look up a tensor descriptor by name.
 *
 * @wf:   Open weight file
 * @name: Null-terminated tensor name
 *
 * Returns pointer to the descriptor within the mapped file,
 * or NULL if not found.
 */
const struct sam3_weight_tensor_desc *sam3_weight_find(
	const struct sam3_weight_file *wf, const char *name);

/*
 * sam3_weight_tensor_data - Get a pointer to a tensor's raw data.
 *
 * @wf:   Open weight file
 * @desc: Tensor descriptor (from sam3_weight_find)
 *
 * Returns pointer into the mapped region. Valid until sam3_weight_close().
 */
const void *sam3_weight_tensor_data(
	const struct sam3_weight_file *wf,
	const struct sam3_weight_tensor_desc *desc);

/*
 * sam3_weight_to_tensor - Populate a sam3_tensor from a weight descriptor.
 *
 * @wf:   Open weight file
 * @desc: Tensor descriptor
 * @out:  Tensor to populate (dims, dtype, strides, data, nbytes)
 *
 * The tensor's data pointer points into the mmap region (read-only).
 * Returns SAM3_OK on success, SAM3_EINVAL on bad descriptor.
 */
enum sam3_error sam3_weight_to_tensor(
	const struct sam3_weight_file *wf,
	const struct sam3_weight_tensor_desc *desc,
	struct sam3_tensor *out);

/* ── Reader vtable (for conversion from external formats) ───────────── */

struct weight_reader;

struct weight_tensor_info {
	const char     *name;
	enum sam3_dtype dtype;
	int             n_dims;
	int             dims[SAM3_MAX_DIMS];
	size_t          nbytes;
};

struct weight_reader_ops {
	enum sam3_error (*open)(struct weight_reader *r, const char *path);
	int             (*n_tensors)(struct weight_reader *r);
	enum sam3_error (*get_tensor_info)(struct weight_reader *r, int idx,
					   struct weight_tensor_info *info);
	enum sam3_error (*read_tensor_data)(struct weight_reader *r, int idx,
					    void *dst, size_t dst_size);
	void            (*close)(struct weight_reader *r);
};

struct weight_reader {
	const struct weight_reader_ops *ops;
	void                           *impl;
};

/* Format-specific reader constructors */
void weight_reader_safetensors_init(struct weight_reader *r);

/* ── Writer (produces .sam3 from any reader) ────────────────────────── */

/*
 * sam3_weight_write - Write a .sam3 file from a weight reader.
 *
 * @output_path: Destination file path
 * @config:      Model configuration to embed in header
 * @reader:      Opened weight reader (any format)
 *
 * Iterates all tensors from the reader, writes header + tensor table +
 * aligned data blob. Returns SAM3_OK on success.
 */
enum sam3_error sam3_weight_write(const char *output_path,
				  const struct sam3_model_config *config,
				  struct weight_reader *reader);

#endif /* SAM3_CORE_WEIGHT_H */
