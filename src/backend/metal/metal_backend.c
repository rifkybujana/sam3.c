/*
 * src/backend/metal/metal_backend.c - Metal backend implementation (MLX-C)
 *
 * Implements the Metal compute backend by translating SAM3 compute graphs
 * into MLX-C lazy operations. Each graph_eval builds an MLX op graph,
 * calls mlx_eval() once for a single GPU dispatch, then copies results
 * back to SAM3 tensors. Q8_0 tensors are dequantized to F16 on GPU
 * via a custom Metal kernel.
 *
 * Key types:  sam3_metal_backend
 * Depends on: metal_backend.h, core/tensor.h, core/quant.h,
 *             util/log.h
 * Used by:    backend.h (registered at init)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "metal_backend.h"
#include "core/tensor.h"
#include "util/log.h"

#ifdef SAM3_HAS_METAL

#include "core/quant.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* --- Dtype mapping ─ --- */

/*
 * metal_map_dtype - Convert sam3_dtype to mlx_dtype.
 *
 * Returns MLX_FLOAT16 for Q8_0 (cache validation only; wrap uploads raw bytes).
 * Returns -1 for unknown dtypes.
 */
static int metal_map_dtype(enum sam3_dtype dt, mlx_dtype *out)
{
	switch (dt) {
	case SAM3_DTYPE_F32:  *out = MLX_FLOAT32;  return 0;
	case SAM3_DTYPE_F16:  *out = MLX_FLOAT16;  return 0;
	case SAM3_DTYPE_BF16: *out = MLX_BFLOAT16; return 0;
	case SAM3_DTYPE_I32:  *out = MLX_INT32;    return 0;
	case SAM3_DTYPE_I8:   *out = MLX_INT8;     return 0;
	case SAM3_DTYPE_Q8_0: *out = MLX_FLOAT16;  return 0;
	default: return -1;
	}
}

/* --- Tensor-to-mlx_array lookup table  --- */

/*
 * Tombstone sentinel for deleted slots in the open-addressing hash
 * table. Required so that linear probing does not stop early when a
 * slot in the middle of a probe chain is evicted.
 */
#define METAL_MAP_TOMBSTONE ((const struct sam3_tensor *)(uintptr_t)1)

static int metal_map_init(struct sam3_metal_backend *mtl)
{
	int cap = SAM3_METAL_MAP_INIT_CAP;

	mtl->map_keys = calloc((size_t)cap, sizeof(*mtl->map_keys));
	mtl->map_vals = calloc((size_t)cap, sizeof(*mtl->map_vals));
	if (!mtl->map_keys || !mtl->map_vals) {
		free(mtl->map_keys);
		free(mtl->map_vals);
		mtl->map_keys = NULL;
		mtl->map_vals = NULL;
		return -1;
	}
	mtl->map_count = 0;
	mtl->map_capacity = cap;
	return 0;
}

static void metal_map_free(struct sam3_metal_backend *mtl)
{
	for (int i = 0; i < mtl->map_capacity; i++) {
		if (mtl->map_keys[i] &&
		    mtl->map_keys[i] != METAL_MAP_TOMBSTONE)
			mlx_array_free(mtl->map_vals[i]);
	}
	free(mtl->map_keys);
	free(mtl->map_vals);
	mtl->map_keys = NULL;
	mtl->map_vals = NULL;
	mtl->map_count = 0;
	mtl->map_capacity = 0;
}

static unsigned metal_map_slot(const struct sam3_tensor *ptr, int capacity)
{
	uintptr_t v = (uintptr_t)ptr;
	v = (v >> 4) ^ (v >> 16);
	return (unsigned)(v & (unsigned)(capacity - 1));
}

/*
 * metal_map_ensure_capacity - Pre-size the map if a known entry count is coming.
 *
 * @hint: expected number of entries that will be inserted.
 *
 * Doubles the map until capacity can hold @hint entries at 75% load.
 * Returns 0 on success, -1 on allocation failure.
 */
static int metal_map_ensure_capacity(struct sam3_metal_backend *mtl, int hint);

/*
 * metal_map_rehash - Double the map capacity and reinsert all entries.
 *
 * Tombstones are dropped during rehash, compacting the table.
 * Returns 0 on success, -1 on allocation failure.
 */
static int metal_map_rehash(struct sam3_metal_backend *mtl)
{
	int old_cap = mtl->map_capacity;
	int new_cap = old_cap * 2;
	unsigned mask = (unsigned)(new_cap - 1);

	const struct sam3_tensor **new_keys =
		calloc((size_t)new_cap, sizeof(*new_keys));
	mlx_array *new_vals =
		calloc((size_t)new_cap, sizeof(*new_vals));

	if (!new_keys || !new_vals) {
		free(new_keys);
		free(new_vals);
		return -1;
	}

	for (int i = 0; i < old_cap; i++) {
		if (!mtl->map_keys[i] ||
		    mtl->map_keys[i] == METAL_MAP_TOMBSTONE)
			continue;
		unsigned slot = metal_map_slot(mtl->map_keys[i], new_cap);
		while (new_keys[slot])
			slot = (slot + 1) & mask;
		new_keys[slot] = mtl->map_keys[i];
		new_vals[slot] = mtl->map_vals[i];
	}

	free(mtl->map_keys);
	free(mtl->map_vals);
	mtl->map_keys = new_keys;
	mtl->map_vals = new_vals;
	mtl->map_capacity = new_cap;

	sam3_log_debug("metal: map rehashed to %d slots (%d entries)",
		       new_cap, mtl->map_count);
	return 0;
}

static int metal_map_ensure_capacity(struct sam3_metal_backend *mtl, int hint)
{
	int needed = (hint + mtl->map_count) * 4 / 3;
	while (mtl->map_capacity < needed) {
		if (metal_map_rehash(mtl) < 0)
			return -1;
	}
	return 0;
}

static mlx_array *metal_map_get(struct sam3_metal_backend *mtl,
				const struct sam3_tensor *key)
{
	int cap = mtl->map_capacity;
	unsigned mask = (unsigned)(cap - 1);
	unsigned idx = metal_map_slot(key, cap);

	for (int i = 0; i < cap; i++) {
		unsigned slot = (idx + (unsigned)i) & mask;
		if (mtl->map_keys[slot] == key)
			return &mtl->map_vals[slot];
		if (!mtl->map_keys[slot])
			return NULL;
		/* Skip tombstones — keep probing */
	}
	return NULL;
}

static int metal_map_put(struct sam3_metal_backend *mtl,
			 const struct sam3_tensor *key, mlx_array val)
{
	/* Rehash at 75% load factor to keep probe chains short */
	if (mtl->map_count >= mtl->map_capacity * 3 / 4) {
		if (metal_map_rehash(mtl) < 0) {
			sam3_log_error("metal: map rehash failed at %d/%d",
				       mtl->map_count, mtl->map_capacity);
			return -1;
		}
	}

	int cap = mtl->map_capacity;
	unsigned mask = (unsigned)(cap - 1);
	unsigned idx = metal_map_slot(key, cap);
	int first_empty = -1;

	for (int i = 0; i < cap; i++) {
		unsigned slot = (idx + (unsigned)i) & mask;
		if (mtl->map_keys[slot] == key) {
			/* Update existing entry in-place */
			mlx_array_free(mtl->map_vals[slot]);
			mtl->map_vals[slot] = val;
			return 0;
		}
		if (!mtl->map_keys[slot]) {
			if (first_empty < 0)
				first_empty = (int)slot;
			break;
		}
		if (mtl->map_keys[slot] == METAL_MAP_TOMBSTONE
		    && first_empty < 0) {
			first_empty = (int)slot;
		}
	}

	if (first_empty >= 0) {
		mtl->map_keys[first_empty] = key;
		mtl->map_vals[first_empty] = val;
		mtl->map_count++;
		return 0;
	}
	sam3_log_error("metal: tensor map full (%d entries)",
		       mtl->map_count);
	return -1;
}

static void metal_map_evict(struct sam3_metal_backend *mtl,
			    const struct sam3_tensor *key)
{
	int cap = mtl->map_capacity;
	unsigned mask = (unsigned)(cap - 1);
	unsigned idx = metal_map_slot(key, cap);

	for (int i = 0; i < cap; i++) {
		unsigned slot = (idx + (unsigned)i) & mask;
		if (mtl->map_keys[slot] == key) {
			mlx_array_free(mtl->map_vals[slot]);
			mtl->map_keys[slot] = METAL_MAP_TOMBSTONE;
			mtl->map_count--;
			return;
		}
		if (!mtl->map_keys[slot])
			return;
		/* Skip tombstones — keep probing */
	}
}

/*
 * metal_cache_invalidate_range - Evict cached entries in a memory range.
 *
 * Scans the tensor-to-mlx_array map and evicts any entry whose tensor
 * pointer falls within [start, start + len). Called when arena memory
 * is recycled so that subsequent wrap calls create fresh mlx_arrays
 * from the new host data instead of returning stale GPU-resident arrays.
 */
static void metal_cache_invalidate_range(struct sam3_metal_backend *mtl,
					  const void *start, size_t len)
{
	if (!mtl->map_keys || len == 0)
		return;

	const char *lo = (const char *)start;
	const char *hi = lo + len;
	int evicted = 0;

	for (int i = 0; i < mtl->map_capacity; i++) {
		const struct sam3_tensor *key = mtl->map_keys[i];
		if (!key || key == METAL_MAP_TOMBSTONE)
			continue;
		const char *addr = (const char *)key;
		if (addr >= lo && addr < hi) {
			mlx_array_free(mtl->map_vals[i]);
			mtl->map_keys[i] = METAL_MAP_TOMBSTONE;
			mtl->map_count--;
			evicted++;
		}
	}

	if (evicted > 0)
		sam3_log_debug("metal: cache_invalidate evicted %d "
			       "entries in range [%p, +%zu)",
			       evicted, start, len);
}

static void metal_cache_invalidate(struct sam3_backend *be,
				    const void *start, size_t len)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;
	metal_cache_invalidate_range(mtl, start, len);
}

/* --- Tensor wrapping ─ --- */

/*
 * metal_wrap_tensor - Get or create an mlx_array for a sam3_tensor.
 *
 * If the tensor is already in the map, returns the existing handle.
 * Otherwise creates a new mlx_array from the tensor's host data,
 * handling Q8_0 dequantization to F16 if needed.
 */
static mlx_array metal_wrap_tensor(struct sam3_metal_backend *mtl,
				   const struct sam3_tensor *t)
{
	mlx_array *existing = metal_map_get(mtl, t);
	if (existing) {
		/* Validate cached array matches current tensor shape + dtype */
		mlx_dtype expected_mtype;
		metal_map_dtype(t->dtype, &expected_mtype);
		mlx_dtype actual_mtype = mlx_array_dtype(*existing);
		bool valid = ((int)mlx_array_ndim(*existing) == t->n_dims);
		if (valid) {
			/*
			 * With F16 compute, intermediate results are F16
			 * even though the sam3 tensor says F32. Accept
			 * either F16 or F32 when use_f16 is active.
			 */
			if (actual_mtype == expected_mtype)
				; /* exact match */
			else if (mtl->use_f16
				 && expected_mtype == MLX_FLOAT32
				 && actual_mtype == MLX_FLOAT16)
				; /* F16 intermediate from F16 compute */
			else
				valid = false;
		}
		for (int i = 0; valid && i < t->n_dims; i++) {
			if (mlx_array_dim(*existing, i) != t->dims[i])
				valid = false;
		}
		if (valid)
			return *existing;
		/* Stale entry (pointer reuse) — evict and re-wrap */
		metal_map_evict(mtl, t);
	}

	if (!t->data) {
		sam3_log_error("metal: wrap_tensor: no data and not in map "
			       "(tensor %p, ndims=%d, dims=[%d,%d,%d,%d], "
			       "dtype=%d)",
			       (const void *)t, t->n_dims,
			       t->n_dims > 0 ? t->dims[0] : 0,
			       t->n_dims > 1 ? t->dims[1] : 0,
			       t->n_dims > 2 ? t->dims[2] : 0,
			       t->n_dims > 3 ? t->dims[3] : 0,
			       t->dtype);
		return mlx_array_new();
	}

	mlx_dtype mtype;
	if (metal_map_dtype(t->dtype, &mtype) < 0) {
		sam3_log_error("metal: unsupported dtype %d", t->dtype);
		return mlx_array_new();
	}

	const void *data = t->data;

	if (t->dtype == SAM3_DTYPE_Q8_0) {
		int nelems = sam3_tensor_nelems(t);
		int nblocks = (nelems + SAM3_Q8_BLOCK_SIZE - 1)
			      / SAM3_Q8_BLOCK_SIZE;
		int raw_bytes = nblocks
				* (int)sizeof(struct sam3_q8_block);

		/* Upload raw Q8 blocks as uint8 byte buffer */
		int raw_shape[] = {raw_bytes};
		mlx_array raw = mlx_array_new_data(
			t->data, raw_shape, 1, MLX_UINT8);

		mlx_vector_array in_vec =
			mlx_vector_array_new_data(&raw, 1);
		mlx_fast_metal_kernel_config cfg =
			mlx_fast_metal_kernel_config_new();

		int out_shape[] = {nelems};
		mlx_fast_metal_kernel_config_add_output_arg(
			cfg, out_shape, 1, MLX_FLOAT16);

		int tpg = nelems < 256 ? nelems : 256;
		mlx_fast_metal_kernel_config_set_grid(
			cfg, nelems, 1, 1);
		mlx_fast_metal_kernel_config_set_thread_group(
			cfg, tpg, 1, 1);

		mlx_vector_array out_vec = mlx_vector_array_new();
		int rc = mlx_fast_metal_kernel_apply(
			&out_vec, mtl->dequant_q8_kernel,
			in_vec, cfg, mtl->stream);

		mlx_array flat_f16 = mlx_array_new();
		if (!rc)
			mlx_vector_array_get(&flat_f16, out_vec, 0);

		mlx_vector_array_free(out_vec);
		mlx_fast_metal_kernel_config_free(cfg);
		mlx_vector_array_free(in_vec);
		mlx_array_free(raw);

		if (rc) {
			mlx_array_free(flat_f16);
			sam3_log_error("metal: Q8 dequant kernel failed");
			return mlx_array_new();
		}

		/* Reshape flat [nelems] to original tensor dims */
		mlx_array arr = mlx_array_new();
		mlx_reshape(&arr, flat_f16, t->dims, t->n_dims,
			    mtl->stream);
		mlx_array_free(flat_f16);

		if (metal_map_put(mtl, t, arr) < 0)
			sam3_log_warn("metal: wrap_tensor: map full, "
				      "array not cached");
		return arr;
	}

	mlx_array arr = mlx_array_new_data(data, t->dims, t->n_dims, mtype);

	/* Cast F32 -> F16 for reduced Metal memory bandwidth */
	if (mtl->use_f16 && t->dtype == SAM3_DTYPE_F32) {
		mlx_array f16 = mlx_array_new();
		int cast_rc = mlx_astype(&f16, arr, MLX_FLOAT16,
					  mtl->stream);
		if (cast_rc == 0) {
			mlx_array_free(arr);
			arr = f16;
		} else {
			mlx_array_free(f16);
			/* Fall through with F32 on cast failure */
		}
	}

	if (metal_map_put(mtl, t, arr) < 0)
		sam3_log_warn("metal: wrap_tensor: map full, array not cached");
	return arr;
}

/* --- Op dispatch  --- */

/* Ensure cached GELU constants (0.5, 1/sqrt(2), 1.0) exist for the dtype. */
static void metal_ensure_gelu_consts(struct sam3_metal_backend *mtl,
				     mlx_dtype dt)
{
	if (mtl->gelu_half[dt].ctx)
		return;

	int shape[] = {1};
	float half_val = 0.5f;
	float rsqrt2_val = 0.7071067811865475f; /* 1/sqrt(2) */
	float one_val = 1.0f;

	mlx_array h = mlx_array_new_data(&half_val, shape, 1, MLX_FLOAT32);
	mlx_array r = mlx_array_new_data(&rsqrt2_val, shape, 1, MLX_FLOAT32);
	mlx_array o = mlx_array_new_data(&one_val, shape, 1, MLX_FLOAT32);

	mlx_astype(&mtl->gelu_half[dt], h, dt, mtl->stream);
	mlx_astype(&mtl->gelu_rsqrt2[dt], r, dt, mtl->stream);
	mlx_astype(&mtl->gelu_one[dt], o, dt, mtl->stream);

	mlx_array_free(o);
	mlx_array_free(r);
	mlx_array_free(h);
}

/* Get or create a cached scalar zero of the given mlx_dtype for ReLU. */
static mlx_array metal_get_relu_zero(struct sam3_metal_backend *mtl,
				     mlx_dtype dt)
{
	if (!mtl->relu_zeros[dt].ctx) {
		int shape[] = {1};
		float zero = 0.0f;
		mlx_array f32_zero = mlx_array_new_data(
			&zero, shape, 1, MLX_FLOAT32);
		mlx_astype(&mtl->relu_zeros[dt], f32_zero, dt,
			   mtl->stream);
		mlx_array_free(f32_zero);
	}
	return mtl->relu_zeros[dt];
}

/*
 * Stack-local cache for reshaped SDPA masks.
 *
 * The text encoder passes the same causal mask through 24 layers.
 * Each SDPA dispatch reshapes it from [seq_q, seq_kv] to
 * [1, 1, seq_q, seq_kv]. This cache avoids 23 redundant reshapes.
 */
#define METAL_MASK_CACHE_SLOTS 4

struct metal_mask_cache_entry {
	const struct sam3_tensor *key;
	mlx_array val;
};

/*
 * metal_dispatch_node - Translate one sam3_node into MLX-C lazy ops.
 *
 * @mtl:        Metal backend state.
 * @node:       Node to dispatch.
 * @mask_cache: Per-graph-eval cache of reshaped SDPA masks. Caller
 *              owns the array; entries are freed by metal_graph_eval
 *              after the dispatch loop. Must point to an array of
 *              METAL_MASK_CACHE_SLOTS zero-initialized entries.
 *
 * The result mlx_array is stored in the map keyed by node->output.
 * Returns SAM3_OK on success.
 */
static enum sam3_error metal_dispatch_node(struct sam3_metal_backend *mtl,
					   const struct sam3_node *node,
					   struct metal_mask_cache_entry *mask_cache)
{
	mlx_array result = mlx_array_new();
	mlx_array inputs[SAM3_NODE_MAX_INPUTS];
	mlx_stream stream = mtl->stream;
	int rc;

	for (int i = 0; i < node->n_inputs; i++) {
		const struct sam3_tensor *inp = node->inputs[i];
		if (!inp->data && !metal_map_get(mtl, inp)) {
			sam3_log_error("metal: dispatch op=%d input[%d]: "
				       "tensor %p ndims=%d data=%p "
				       "(will fail wrap)",
				       node->op, i, (const void *)inp,
				       inp->n_dims, (const void *)inp->data);
		}
		inputs[i] = metal_wrap_tensor(mtl, inp);
	}

	switch (node->op) {
	case SAM3_OP_MATMUL:
		/*
		 * MLX's steel GEMM accumulates in F32 regardless of operand
		 * dtype (AccumType defaults to float in gemm.h/transforms.h).
		 * So F16 operands already get F32 accumulation for free —
		 * no manual astype round-trip needed. For fully-F32 compute,
		 * set SAM3_METAL_F32=1 which keeps tensors F32 end-to-end.
		 */
		rc = mlx_matmul(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_ADD:
		rc = mlx_add(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_MUL:
		rc = mlx_multiply(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_DIV:
		rc = mlx_divide(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_SOFTMAX:
		rc = mlx_softmax_axis(&result, inputs[0], -1, true, stream);
		break;

	case SAM3_OP_RELU: {
		mlx_dtype in_dtype = mlx_array_dtype(inputs[0]);
		mlx_array zero = metal_get_relu_zero(mtl, in_dtype);
		rc = mlx_maximum(&result, inputs[0], zero, stream);
		break;
	}

	case SAM3_OP_GELU: {
		/*
		 * Exact GELU matching PyTorch nn.GELU(approximate='none'):
		 *   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
		 *
		 * Scalar constants (0.5, 1/sqrt(2), 1.0) are cached per
		 * dtype to avoid 6 temporary arrays per dispatch.
		 */
		mlx_dtype in_dtype = mlx_array_dtype(inputs[0]);
		metal_ensure_gelu_consts(mtl, in_dtype);

		/* x / sqrt(2) */
		mlx_array x_scaled = mlx_array_new();
		mlx_multiply(&x_scaled, inputs[0],
			     mtl->gelu_rsqrt2[in_dtype], stream);

		/* erf(x / sqrt(2)) */
		mlx_array erf_val = mlx_array_new();
		mlx_erf(&erf_val, x_scaled, stream);

		/* 1 + erf(...) */
		mlx_array one_plus_erf = mlx_array_new();
		mlx_add(&one_plus_erf, mtl->gelu_one[in_dtype],
			erf_val, stream);

		/* 0.5 * x */
		mlx_array half_x = mlx_array_new();
		mlx_multiply(&half_x, inputs[0],
			     mtl->gelu_half[in_dtype], stream);

		/* 0.5 * x * (1 + erf(...)) */
		rc = mlx_multiply(&result, half_x, one_plus_erf, stream);

		mlx_array_free(half_x);
		mlx_array_free(one_plus_erf);
		mlx_array_free(erf_val);
		mlx_array_free(x_scaled);
		break;
	}

	case SAM3_OP_LAYERNORM: {
		mlx_array weight = (node->n_inputs > 1)
			? inputs[1] : (mlx_array){0};
		mlx_array bias = (node->n_inputs > 2)
			? inputs[2] : (mlx_array){0};
		float eps = 1e-5f;
		rc = mlx_fast_layer_norm(&result, inputs[0],
					 weight, bias, eps, stream);
		break;
	}

	case SAM3_OP_CONV2D: {
		/*
		 * MLX conv2d consumes NHWC inputs and OHWI weights.
		 * The SAM3 graph is NHWC end-to-end, so we forward
		 * to MLX with no transpose wrapping.
		 *
		 * params[0] = stride (same for H and W)
		 * params[1] = padding (same for H and W)
		 * params[2] = groups (0 means 1)
		 */
		int stride = node->params[0] ? node->params[0] : 1;
		int pad_h = node->params[1];
		int groups = node->params[2] > 0 ? node->params[2] : 1;
		/* params[3]=pad_w: non-zero enables asymmetric H/W padding.
		 * Zero means symmetric (pad_w = pad_h). */
		int pad_w = node->params[3] ? node->params[3] : pad_h;

		rc = mlx_conv2d(&result, inputs[0], inputs[1],
				stride, stride, pad_h, pad_w,
				1, 1,  /* dilation */
				groups,
				stream);
		break;
	}

	case SAM3_OP_RESHAPE: {
		const struct sam3_tensor *out = node->output;
		rc = mlx_reshape(&result, inputs[0],
				 out->dims, out->n_dims, stream);
		break;
	}

	case SAM3_OP_TRANSPOSE: {
		int ndim = (int)mlx_array_ndim(inputs[0]);
		int axes[SAM3_MAX_DIMS];
		int has_perm = 0;
		for (int i = 0; i < ndim; i++) {
			axes[i] = node->params[i];
			if (axes[i] != 0)
				has_perm = 1;
		}
		if (!has_perm) {
			for (int i = 0; i < ndim; i++)
				axes[i] = ndim - 1 - i;
		}
		rc = mlx_transpose_axes(&result, inputs[0],
					axes, (size_t)ndim, stream);
		break;
	}

	case SAM3_OP_CAST: {
		mlx_dtype target;
		if (metal_map_dtype(node->output->dtype, &target) < 0) {
			sam3_log_error("metal: cast to unsupported dtype %d",
				       node->output->dtype);
			mlx_array_free(result);
			return SAM3_EDTYPE;
		}
		rc = mlx_astype(&result, inputs[0], target, stream);
		break;
	}

	case SAM3_OP_SIGMOID:
		rc = mlx_sigmoid(&result, inputs[0], stream);
		break;

	case SAM3_OP_SILU: {
		/* Fused SiLU via custom Metal kernel: x / (1 + exp(-x)) */
		mlx_vector_array in_vec = mlx_vector_array_new_data(
			inputs, 1);
		mlx_fast_metal_kernel_config cfg =
			mlx_fast_metal_kernel_config_new();

		int ndims_in = mlx_array_ndim(inputs[0]);
		int total = 1;
		for (int d = 0; d < ndims_in; d++)
			total *= mlx_array_dim(inputs[0], d);

		int shape[SAM3_MAX_DIMS];
		for (int d = 0; d < ndims_in; d++)
			shape[d] = mlx_array_dim(inputs[0], d);
		mlx_fast_metal_kernel_config_add_output_arg(
			cfg, shape, (size_t)ndims_in,
			mlx_array_dtype(inputs[0]));

		int tpg = total < 256 ? total : 256;
		mlx_fast_metal_kernel_config_set_grid(cfg, total, 1, 1);
		mlx_fast_metal_kernel_config_set_thread_group(
			cfg, tpg, 1, 1);
		mlx_fast_metal_kernel_config_add_template_arg_dtype(
			cfg, "T", mlx_array_dtype(inputs[0]));

		mlx_vector_array out_vec = mlx_vector_array_new();
		rc = mlx_fast_metal_kernel_apply(
			&out_vec, mtl->silu_kernel, in_vec, cfg, stream);
		if (!rc) {
			mlx_array tmp = mlx_array_new();
			mlx_vector_array_get(&tmp, out_vec, 0);
			mlx_array_set(&result, tmp);
			mlx_array_free(tmp);
		}

		mlx_vector_array_free(out_vec);
		mlx_fast_metal_kernel_config_free(cfg);
		mlx_vector_array_free(in_vec);
		break;
	}

	case SAM3_OP_HSWISH: {
		/* Fused Hard Swish via custom Metal kernel:
		 * x * clamp(x+3, 0, 6) / 6 */
		mlx_vector_array in_vec = mlx_vector_array_new_data(
			inputs, 1);
		mlx_fast_metal_kernel_config cfg =
			mlx_fast_metal_kernel_config_new();

		int ndims_in = mlx_array_ndim(inputs[0]);
		int total = 1;
		for (int d = 0; d < ndims_in; d++)
			total *= mlx_array_dim(inputs[0], d);

		int shape[SAM3_MAX_DIMS];
		for (int d = 0; d < ndims_in; d++)
			shape[d] = mlx_array_dim(inputs[0], d);
		mlx_fast_metal_kernel_config_add_output_arg(
			cfg, shape, (size_t)ndims_in,
			mlx_array_dtype(inputs[0]));

		int tpg = total < 256 ? total : 256;
		mlx_fast_metal_kernel_config_set_grid(cfg, total, 1, 1);
		mlx_fast_metal_kernel_config_set_thread_group(
			cfg, tpg, 1, 1);
		mlx_fast_metal_kernel_config_add_template_arg_dtype(
			cfg, "T", mlx_array_dtype(inputs[0]));

		mlx_vector_array out_vec = mlx_vector_array_new();
		rc = mlx_fast_metal_kernel_apply(
			&out_vec, mtl->hswish_kernel, in_vec, cfg, stream);
		if (!rc) {
			mlx_array tmp = mlx_array_new();
			mlx_vector_array_get(&tmp, out_vec, 0);
			mlx_array_set(&result, tmp);
			mlx_array_free(tmp);
		}

		mlx_vector_array_free(out_vec);
		mlx_fast_metal_kernel_config_free(cfg);
		mlx_vector_array_free(in_vec);
		break;
	}

	case SAM3_OP_CONCAT: {
		int axis = node->params[0];
		mlx_vector_array arr = mlx_vector_array_new_data(
			inputs, (size_t)node->n_inputs);
		rc = mlx_concatenate_axis(&result, arr, axis, stream);
		mlx_vector_array_free(arr);
		break;
	}

	case SAM3_OP_SLICE: {
		/*
		 * params: [0]=axis, [1]=start, [2]=end
		 * mlx_slice wants per-dimension start/stop/stride arrays.
		 */
		int axis = node->params[0];
		int s_start = node->params[1];
		int s_end = node->params[2];
		int ndim = (int)mlx_array_ndim(inputs[0]);

		int start_arr[SAM3_MAX_DIMS];
		int stop_arr[SAM3_MAX_DIMS];
		int stride_arr[SAM3_MAX_DIMS];

		for (int d = 0; d < ndim; d++) {
			start_arr[d] = 0;
			stop_arr[d] = mlx_array_dim(inputs[0], d);
			stride_arr[d] = 1;
		}
		start_arr[axis] = s_start;
		stop_arr[axis] = s_end;

		rc = mlx_slice(&result, inputs[0],
			       start_arr, (size_t)ndim,
			       stop_arr, (size_t)ndim,
			       stride_arr, (size_t)ndim,
			       stream);
		break;
	}

	case SAM3_OP_EMBED:
		/* Embedding lookup = gather rows: take(table, indices, axis=0) */
		rc = mlx_take_axis(&result, inputs[0], inputs[1], 0, stream);
		break;

	case SAM3_OP_UPSAMPLE: {
		/*
		 * Nearest-neighbor upsample by integer scale (NHWC).
		 * Reshape [N,H,W,C] to [N,H,1,W,1,C], broadcast to
		 * [N,H,s,W,s,C], reshape to [N,H*s,W*s,C]. The channel
		 * axis at position 5 is not replicated.
		 *
		 * params[0] = scale.
		 */
		int scale = node->params[0];
		int ndim = (int)mlx_array_ndim(inputs[0]);
		if (ndim != 4) {
			sam3_log_error("metal: upsample requires 4D input");
			mlx_array_free(result);
			return SAM3_EINVAL;
		}
		int d0 = mlx_array_dim(inputs[0], 0);
		int d1 = mlx_array_dim(inputs[0], 1);
		int d2 = mlx_array_dim(inputs[0], 2);
		int d3 = mlx_array_dim(inputs[0], 3);

		int shape6[6] = {d0, d1, 1, d2, 1, d3};
		int bcast[6] = {d0, d1, scale, d2, scale, d3};
		int shape4[4] = {d0, d1 * scale, d2 * scale, d3};

		mlx_array reshaped = mlx_array_new();
		rc = mlx_reshape(&reshaped, inputs[0], shape6, 6, stream);
		if (rc) { mlx_array_free(reshaped); break; }

		mlx_array expanded = mlx_array_new();
		rc = mlx_broadcast_to(&expanded, reshaped, bcast, 6, stream);
		mlx_array_free(reshaped);
		if (rc) { mlx_array_free(expanded); break; }

		rc = mlx_reshape(&result, expanded, shape4, 4, stream);
		mlx_array_free(expanded);
		break;
	}

	case SAM3_OP_ROPE: {
		/*
		 * Rotary position embedding.
		 * inputs[0] = x [batch, seq, heads, head_dim]
		 * inputs[1] = cos [seq, head_dim/2]
		 * inputs[2] = sin [seq, head_dim/2]
		 * params[0] = head_dim
		 * params[1] = grid_w (>0 = fast axial path)
		 * params[2] = position scale (float bits)
		 */
		int ndim = (int)mlx_array_ndim(inputs[0]);
		if (ndim != 4) {
			sam3_log_error("metal: rope requires 4D input");
			mlx_array_free(result);
			return SAM3_EINVAL;
		}

		int batch = mlx_array_dim(inputs[0], 0);
		int seq = mlx_array_dim(inputs[0], 1);
		int heads = mlx_array_dim(inputs[0], 2);
		int head_dim = mlx_array_dim(inputs[0], 3);
		int half_dim = head_dim / 2;
		int grid_w = node->params[1];

		if (grid_w > 0) {
			/*
			 * Fast 2D axial RoPE via mlx_fast_rope_dynamic.
			 *
			 * Split head_dim into x-half and y-half, apply
			 * fused RoPE to each with per-token position
			 * offsets from grid geometry. Reshape to
			 * [B*S, H, 1, HD/2] so each token is its own
			 * batch element with T=1; offset[b] gives the
			 * grid position for that token.
			 */
			float pos_scale;
			memcpy(&pos_scale, &node->params[2],
			       sizeof(float));
			int total = batch * seq;

			/* Build x/y offset arrays (one alloc) */
			int *off_buf = (int *)malloc(
				2 * (size_t)total * sizeof(int));
			if (!off_buf) {
				mlx_array_free(result);
				return SAM3_ENOMEM;
			}
			int *ox = off_buf;
			int *oy = off_buf + total;
			for (int s = 0; s < seq; s++) {
				ox[s] = s % grid_w;
				oy[s] = s / grid_w;
			}
			for (int b = 1; b < batch; b++) {
				memcpy(ox + b * seq, ox,
				       (size_t)seq * sizeof(int));
				memcpy(oy + b * seq, oy,
				       (size_t)seq * sizeof(int));
			}

			int off_shape[] = {total};
			mlx_array offset_x = mlx_array_new_data(
				ox, off_shape, 1, MLX_INT32);
			mlx_array offset_y = mlx_array_new_data(
				oy, off_shape, 1, MLX_INT32);
			free(off_buf);

			/* Slice x-half and y-half along last axis */
			int sx0[] = {0, 0, 0, 0};
			int sx1[] = {batch, seq, heads, half_dim};
			int sy0[] = {0, 0, 0, half_dim};
			int sy1[] = {batch, seq, heads, head_dim};
			int ss[]  = {1, 1, 1, 1};

			mlx_array x_half = mlx_array_new();
			rc = mlx_slice(&x_half, inputs[0],
				       sx0, 4, sx1, 4, ss, 4, stream);
			if (rc) {
				mlx_array_free(x_half);
				mlx_array_free(offset_x);
				mlx_array_free(offset_y);
				break;
			}

			mlx_array y_half = mlx_array_new();
			rc = mlx_slice(&y_half, inputs[0],
				       sy0, 4, sy1, 4, ss, 4, stream);
			if (rc) {
				mlx_array_free(x_half);
				mlx_array_free(y_half);
				mlx_array_free(offset_x);
				mlx_array_free(offset_y);
				break;
			}

			/* Reshape: [B,S,H,HD/2] -> [B*S,H,1,HD/2] */
			int rope_shape[] = {total, heads, 1, half_dim};
			mlx_array xr = mlx_array_new();
			mlx_reshape(&xr, x_half, rope_shape, 4, stream);
			mlx_array_free(x_half);

			mlx_array yr = mlx_array_new();
			mlx_reshape(&yr, y_half, rope_shape, 4, stream);
			mlx_array_free(y_half);

			/* Fused RoPE on each half */
			mlx_optional_float base = {10000.0f, true};
			mlx_array no_freqs = (mlx_array){0};

			mlx_array x_rot = mlx_array_new();
			rc = mlx_fast_rope_dynamic(
				&x_rot, xr, half_dim, true,
				base, pos_scale, offset_x,
				no_freqs, stream);
			mlx_array_free(xr);
			mlx_array_free(offset_x);
			if (rc) {
				mlx_array_free(x_rot);
				mlx_array_free(yr);
				mlx_array_free(offset_y);
				break;
			}

			mlx_array y_rot = mlx_array_new();
			rc = mlx_fast_rope_dynamic(
				&y_rot, yr, half_dim, true,
				base, pos_scale, offset_y,
				no_freqs, stream);
			mlx_array_free(yr);
			mlx_array_free(offset_y);
			if (rc) {
				mlx_array_free(x_rot);
				mlx_array_free(y_rot);
				break;
			}

			/* Reshape back: [B*S,H,1,HD/2] -> [B,S,H,HD/2] */
			int half_out[] = {batch, seq, heads, half_dim};
			mlx_array xb = mlx_array_new();
			mlx_reshape(&xb, x_rot, half_out, 4, stream);
			mlx_array_free(x_rot);

			mlx_array yb = mlx_array_new();
			mlx_reshape(&yb, y_rot, half_out, 4, stream);
			mlx_array_free(y_rot);

			/* Concatenate -> [B, S, H, HD] */
			mlx_vector_array halves = mlx_vector_array_new();
			mlx_vector_array_append_value(halves, xb);
			mlx_vector_array_append_value(halves, yb);

			rc = mlx_concatenate_axis(
				&result, halves, -1, stream);
			mlx_vector_array_free(halves);
			mlx_array_free(xb);
			mlx_array_free(yb);
			break;
		}

		/*
		 * Legacy manual RoPE (grid_w == 0).
		 *
		 * Interleaved pairs: x[...,2d], x[...,2d+1]
		 *   out_even = x_even * cos - x_odd * sin
		 *   out_odd  = x_even * sin + x_odd * cos
		 *
		 * Strategy: reshape x to [..., head_dim/2, 2], split,
		 * apply rotation, interleave back.
		 */

		/* Reshape x to [batch, seq, heads, half_dim, 2] */
		int shape5[] = {batch, seq, heads, half_dim, 2};
		mlx_array x5 = mlx_array_new();
		rc = mlx_reshape(&x5, inputs[0], shape5, 5, stream);
		if (rc) { mlx_array_free(x5); break; }

		/* Slice even: x5[..., 0:1] and odd: x5[..., 1:2] */
		int s_start_e[] = {0, 0, 0, 0, 0};
		int s_stop_e[]  = {batch, seq, heads, half_dim, 1};
		int s_stride[]  = {1, 1, 1, 1, 1};
		int s_start_o[] = {0, 0, 0, 0, 1};
		int s_stop_o[]  = {batch, seq, heads, half_dim, 2};

		mlx_array x_even5 = mlx_array_new();
		rc = mlx_slice(&x_even5, x5,
			       s_start_e, 5, s_stop_e, 5, s_stride, 5,
			       stream);
		if (rc) {
			mlx_array_free(x5);
			mlx_array_free(x_even5);
			break;
		}

		mlx_array x_odd5 = mlx_array_new();
		rc = mlx_slice(&x_odd5, x5,
			       s_start_o, 5, s_stop_o, 5, s_stride, 5,
			       stream);
		mlx_array_free(x5);
		if (rc) {
			mlx_array_free(x_even5);
			mlx_array_free(x_odd5);
			break;
		}

		/* Squeeze trailing dim: [batch, seq, heads, half_dim] */
		mlx_array x_even = mlx_array_new();
		mlx_squeeze_axis(&x_even, x_even5, -1, stream);
		mlx_array_free(x_even5);

		mlx_array x_odd = mlx_array_new();
		mlx_squeeze_axis(&x_odd, x_odd5, -1, stream);
		mlx_array_free(x_odd5);

		/*
		 * cos/sin are [seq, half_dim].
		 * Reshape to [1, seq, 1, half_dim] for broadcasting.
		 */
		int cs_shape[] = {1, seq, 1, half_dim};
		mlx_array cos_r = mlx_array_new();
		mlx_reshape(&cos_r, inputs[1], cs_shape, 4, stream);
		mlx_array sin_r = mlx_array_new();
		mlx_reshape(&sin_r, inputs[2], cs_shape, 4, stream);

		/* out_even = x_even * cos - x_odd * sin */
		mlx_array ec = mlx_array_new();
		mlx_multiply(&ec, x_even, cos_r, stream);
		mlx_array os = mlx_array_new();
		mlx_multiply(&os, x_odd, sin_r, stream);
		mlx_array r_even = mlx_array_new();
		mlx_subtract(&r_even, ec, os, stream);
		mlx_array_free(ec);
		mlx_array_free(os);

		/* out_odd = x_even * sin + x_odd * cos */
		mlx_array es = mlx_array_new();
		mlx_multiply(&es, x_even, sin_r, stream);
		mlx_array oc = mlx_array_new();
		mlx_multiply(&oc, x_odd, cos_r, stream);
		mlx_array r_odd = mlx_array_new();
		mlx_add(&r_odd, es, oc, stream);
		mlx_array_free(es);
		mlx_array_free(oc);

		mlx_array_free(x_even);
		mlx_array_free(x_odd);
		mlx_array_free(cos_r);
		mlx_array_free(sin_r);

		/*
		 * Stack even/odd along a new last dim, then reshape
		 * to interleave back into [batch, seq, heads, head_dim].
		 *
		 * Expand both to [B,S,H,half,1], stack => [B,S,H,half,2]
		 */
		mlx_array re_exp = mlx_array_new();
		mlx_expand_dims(&re_exp, r_even, -1, stream);
		mlx_array_free(r_even);

		mlx_array ro_exp = mlx_array_new();
		mlx_expand_dims(&ro_exp, r_odd, -1, stream);
		mlx_array_free(r_odd);

		mlx_vector_array pair = mlx_vector_array_new();
		mlx_vector_array_append_value(pair, re_exp);
		mlx_vector_array_append_value(pair, ro_exp);

		mlx_array interleaved = mlx_array_new();
		rc = mlx_concatenate_axis(&interleaved, pair, -1, stream);
		mlx_vector_array_free(pair);
		mlx_array_free(re_exp);
		mlx_array_free(ro_exp);
		if (rc) {
			mlx_array_free(interleaved);
			break;
		}

		/* Final reshape: [batch, seq, heads, head_dim] */
		int out_shape[] = {batch, seq, heads, head_dim};
		rc = mlx_reshape(&result, interleaved, out_shape, 4, stream);
		mlx_array_free(interleaved);
		break;
	}

	case SAM3_OP_SDPA: {
		/*
		 * Fused scaled dot-product attention via MLX.
		 *
		 * 4D batched: Q[B, H, seq_q, hd] - pass directly.
		 * 2D legacy:  Q[seq_q, hd] - reshape to [1, 1, seq_q, hd].
		 */
		int hd = node->params[0];
		float scale = 1.0f / sqrtf((float)hd);
		int ndims = node->inputs[0]->n_dims;

		mlx_array q_in, k_in, v_in;

		if (ndims == 4) {
			/* Batched: already [B, H, seq, hd] */
			q_in = inputs[0];
			k_in = inputs[1];
			v_in = inputs[2];
		} else {
			/* Legacy 2D: [seq, hd] -> [1, 1, seq, hd] */
			int seq_q = node->inputs[0]->dims[0];
			int seq_kv = node->inputs[1]->dims[0];
			int q_shape[] = {1, 1, seq_q, hd};
			int kv_shape[] = {1, 1, seq_kv, hd};

			q_in = mlx_array_new();
			k_in = mlx_array_new();
			v_in = mlx_array_new();
			rc = mlx_reshape(&q_in, inputs[0], q_shape, 4,
					  stream);
			if (!rc)
				rc = mlx_reshape(&k_in, inputs[1],
						  kv_shape, 4, stream);
			if (!rc)
				rc = mlx_reshape(&v_in, inputs[2],
						  kv_shape, 4, stream);
			if (rc) {
				mlx_array_free(q_in);
				mlx_array_free(k_in);
				mlx_array_free(v_in);
				break;
			}
		}

		/* Prepare mask if present */
		mlx_array mask4d = (mlx_array){0};
		bool mask_cached = false;
		const char *mask_mode = "";
		if (node->n_inputs > 3 && node->inputs[3]) {
			/* Check mask cache first */
			const struct sam3_tensor *mask_key = node->inputs[3];
			mlx_array cached = (mlx_array){0};
			for (int mc = 0; mc < METAL_MASK_CACHE_SLOTS; mc++) {
				if (mask_cache[mc].key == mask_key) {
					cached = mask_cache[mc].val;
					break;
				}
			}

			if (cached.ctx) {
				mask4d = cached;
				mask_cached = true;
			} else {
				mask4d = mlx_array_new();
				int seq_q = (ndims == 4)
					? node->inputs[0]->dims[2]
					: node->inputs[0]->dims[0];
				int seq_kv = (ndims == 4)
					? node->inputs[1]->dims[2]
					: node->inputs[1]->dims[0];
				int mshape[] = {1, 1, seq_q, seq_kv};
				rc = mlx_reshape(&mask4d, inputs[3],
						  mshape, 4, stream);
				if (rc) {
					if (ndims != 4) {
						mlx_array_free(q_in);
						mlx_array_free(k_in);
						mlx_array_free(v_in);
					}
					mlx_array_free(mask4d);
					break;
				}

				/* Store in first free cache slot */
				for (int mc = 0; mc < METAL_MASK_CACHE_SLOTS; mc++) {
					if (!mask_cache[mc].key) {
						mask_cache[mc].key = mask_key;
						mask_cache[mc].val = mask4d;
						mask_cached = true;
						break;
					}
				}
			}
			mask_mode = "array";
		}

		mlx_array sdpa_out = mlx_array_new();
		mlx_array no_sinks = (mlx_array){0};
		rc = mlx_fast_scaled_dot_product_attention(
			&sdpa_out, q_in, k_in, v_in, scale,
			mask_mode, mask4d, no_sinks, stream);

		if (ndims != 4) {
			mlx_array_free(q_in);
			mlx_array_free(k_in);
			mlx_array_free(v_in);
		}
		if (mask4d.ctx && !mask_cached)
			mlx_array_free(mask4d);

		if (rc) {
			mlx_array_free(sdpa_out);
			break;
		}

		if (ndims == 4) {
			/* Batched: output [B, H, seq_q, hd] as-is */
			result = sdpa_out;
		} else {
			/* Legacy: [1, 1, seq_q, hd] -> [seq_q, hd] */
			int seq_q = node->inputs[0]->dims[0];
			int shape2d[] = {seq_q, hd};
			rc = mlx_reshape(&result, sdpa_out, shape2d, 2,
					  stream);
			mlx_array_free(sdpa_out);
		}
		break;
	}

	case SAM3_OP_CONV_TRANSPOSE2D: {
		/*
		 * MLX conv_transpose2d consumes NHWC inputs and OHWI
		 * weights [C_out, KH, KW, C_in]. The SAM3 graph is
		 * NHWC end-to-end, so we forward to MLX with no
		 * transpose wrapping.
		 *
		 * params[0] = stride, params[1] = padding.
		 */
		int stride = node->params[0] ? node->params[0] : 1;
		int pad = node->params[1];

		rc = mlx_conv_transpose2d(&result,
					   inputs[0], inputs[1],
					   stride, stride,
					   pad, pad,
					   1, 1,  /* dilation */
					   0, 0,  /* output_padding */
					   1,     /* groups */
					   stream);
		break;
	}

	case SAM3_OP_MAXPOOL2D: {
		/*
		 * MaxPool2d (NHWC) for the non-overlapping case
		 * (stride == kernel_size). [N, H, W, C] -> reshape
		 * to [N, H/k, k, W/k, k, C] -> max over axis 4 then
		 * axis 2 -> [N, H/k, W/k, C].
		 *
		 * params[0] = kernel_size, params[1] = stride.
		 */
		int mp_k = node->params[0];
		int mp_s = node->params[1];
		const struct sam3_tensor *mp_in = node->inputs[0];
		int mp_N = mp_in->dims[0];
		int mp_H = mp_in->dims[1];
		int mp_W = mp_in->dims[2];
		int mp_C = mp_in->dims[3];

		if (mp_s != mp_k || mp_H % mp_k != 0 || mp_W % mp_k != 0) {
			sam3_log_error("metal: maxpool2d only supports "
				       "stride==kernel, even dims");
			mlx_array_free(result);
			return SAM3_EINVAL;
		}

		int mp_shape[6] = {
			mp_N,
			mp_H / mp_k, mp_k,
			mp_W / mp_k, mp_k,
			mp_C
		};

		mlx_array mp_reshaped = mlx_array_new();
		rc = mlx_reshape(&mp_reshaped, inputs[0],
				 mp_shape, 6, stream);
		if (rc) { mlx_array_free(mp_reshaped); break; }

		/* Max over inner k-axis (axis 4) */
		mlx_array mp_tmp = mlx_array_new();
		rc = mlx_max_axis(&mp_tmp, mp_reshaped, 4, false, stream);
		mlx_array_free(mp_reshaped);
		if (rc) { mlx_array_free(mp_tmp); break; }

		/* Max over remaining k-axis (shifted to axis 2) */
		rc = mlx_max_axis(&result, mp_tmp, 2, false, stream);
		mlx_array_free(mp_tmp);
		break;
	}

	case SAM3_OP_GROUPNORM: {
		/*
		 * GroupNorm(num_groups) on NHWC input [N, H, W, C].
		 * inputs[0]=x, inputs[1]=gamma[C], inputs[2]=beta[C].
		 * params[0] = num_groups.
		 *
		 * Strategy: transpose [N,H,W,C]->[N,C,H,W] so the
		 * C-axis becomes contiguous, reshape to
		 * [N, G, C/G * H * W], run layer_norm (no affine) over
		 * the last dim, reshape back to [N,C,H,W], apply
		 * per-channel affine, then transpose back to
		 * [N,H,W,C]. GroupNorm runs only a couple of times per
		 * inference, so the extra transposes are negligible.
		 */
		int gn_groups = node->params[0];
		const struct sam3_tensor *gn_in = node->inputs[0];
		int gn_N = gn_in->dims[0];
		int gn_H = gn_in->dims[1];
		int gn_W = gn_in->dims[2];
		int gn_C = gn_in->dims[3];
		int gn_cpg = gn_C / gn_groups;
		int gn_hw = gn_H * gn_W;

		/* Transpose [N,H,W,C] -> [N,C,H,W] for the body. */
		int gn_in_perm[] = {0, 3, 1, 2};
		mlx_array gn_body_in = mlx_array_new();
		rc = mlx_transpose_axes(&gn_body_in, inputs[0],
					gn_in_perm, 4, stream);
		if (rc) {
			mlx_array_free(gn_body_in);
			break;
		}

		/* Reshape to [N, G, C/G * H * W] for layer_norm over last dim */
		int gn_shape3[] = {gn_N, gn_groups, gn_cpg * gn_hw};
		mlx_array gn_x3 = mlx_array_new();
		rc = mlx_reshape(&gn_x3, gn_body_in, gn_shape3, 3, stream);
		mlx_array_free(gn_body_in);
		if (rc) { mlx_array_free(gn_x3); break; }

		/* Layer norm over last dim (no affine) */
		mlx_array gn_no_w = (mlx_array){0};
		mlx_array gn_no_b = (mlx_array){0};
		mlx_array gn_normed3 = mlx_array_new();
		rc = mlx_fast_layer_norm(&gn_normed3, gn_x3,
					 gn_no_w, gn_no_b, 1e-5f, stream);
		mlx_array_free(gn_x3);
		if (rc) { mlx_array_free(gn_normed3); break; }

		/* Reshape back to [N, C, H, W] */
		int gn_shape4[] = {gn_N, gn_C, gn_H, gn_W};
		mlx_array gn_normed4 = mlx_array_new();
		rc = mlx_reshape(&gn_normed4, gn_normed3,
				 gn_shape4, 4, stream);
		mlx_array_free(gn_normed3);
		if (rc) { mlx_array_free(gn_normed4); break; }

		/* Apply per-channel gamma and beta via [1,C,1,1] broadcast */
		if (node->n_inputs > 1 && node->inputs[1]) {
			int gn_aff_shape[] = {1, gn_C, 1, 1};
			mlx_array gn_gamma4 = mlx_array_new();
			mlx_reshape(&gn_gamma4, inputs[1],
				    gn_aff_shape, 4, stream);
			mlx_array gn_scaled = mlx_array_new();
			rc = mlx_multiply(&gn_scaled, gn_normed4,
					  gn_gamma4, stream);
			mlx_array_free(gn_gamma4);
			mlx_array_free(gn_normed4);
			if (rc) { mlx_array_free(gn_scaled); break; }
			gn_normed4 = gn_scaled;
		}

		if (node->n_inputs > 2 && node->inputs[2]) {
			int gn_aff_shape[] = {1, gn_C, 1, 1};
			mlx_array gn_beta4 = mlx_array_new();
			mlx_reshape(&gn_beta4, inputs[2],
				    gn_aff_shape, 4, stream);
			mlx_array gn_biased = mlx_array_new();
			rc = mlx_add(&gn_biased, gn_normed4,
				     gn_beta4, stream);
			mlx_array_free(gn_beta4);
			mlx_array_free(gn_normed4);
			if (rc) { mlx_array_free(gn_biased); break; }
			gn_normed4 = gn_biased;
		}

		/* Transpose [N,C,H,W] -> [N,H,W,C] for the output. */
		int gn_out_perm[] = {0, 2, 3, 1};
		rc = mlx_transpose_axes(&result, gn_normed4,
					gn_out_perm, 4, stream);
		mlx_array_free(gn_normed4);
		break;
	}

	case SAM3_OP_BATCHNORM: {
		/*
		 * BatchNorm (eval) on NHWC input [N, H, W, C].
		 * inputs[0]=x, inputs[1]=gamma[C], inputs[2]=beta[C],
		 * inputs[3]=running_mean[C], inputs[4]=running_var[C].
		 *
		 * out = (x - mean) / sqrt(var + eps) * gamma + beta
		 *     = (x - mean) * rsqrt(var + eps) * gamma + beta
		 *
		 * Channel dim is last, so [C] broadcasts over [N,H,W,C].
		 */
		float bn_eps = 1e-5f;
		int bn_shape[] = {1};
		mlx_array bn_eps_arr = mlx_array_new_data(
			&bn_eps, bn_shape, 1, MLX_FLOAT32);

		/* var + eps */
		mlx_array bn_var_eps = mlx_array_new();
		rc = mlx_add(&bn_var_eps, inputs[4], bn_eps_arr, stream);
		mlx_array_free(bn_eps_arr);
		if (rc) { mlx_array_free(bn_var_eps); break; }

		/* rsqrt(var + eps) */
		mlx_array bn_inv_std = mlx_array_new();
		rc = mlx_rsqrt(&bn_inv_std, bn_var_eps, stream);
		mlx_array_free(bn_var_eps);
		if (rc) { mlx_array_free(bn_inv_std); break; }

		/* x - mean */
		mlx_array bn_centered = mlx_array_new();
		rc = mlx_subtract(&bn_centered, inputs[0], inputs[3],
				  stream);
		if (rc) {
			mlx_array_free(bn_inv_std);
			mlx_array_free(bn_centered);
			break;
		}

		/* (x - mean) * rsqrt(var + eps) */
		mlx_array bn_normed = mlx_array_new();
		rc = mlx_multiply(&bn_normed, bn_centered, bn_inv_std,
				  stream);
		mlx_array_free(bn_centered);
		mlx_array_free(bn_inv_std);
		if (rc) { mlx_array_free(bn_normed); break; }

		/* * gamma */
		if (node->n_inputs > 1 && node->inputs[1]) {
			mlx_array bn_scaled = mlx_array_new();
			rc = mlx_multiply(&bn_scaled, bn_normed,
					  inputs[1], stream);
			mlx_array_free(bn_normed);
			if (rc) { mlx_array_free(bn_scaled); break; }
			bn_normed = bn_scaled;
		}

		/* + beta */
		if (node->n_inputs > 2 && node->inputs[2]) {
			rc = mlx_add(&result, bn_normed, inputs[2],
				     stream);
			mlx_array_free(bn_normed);
		} else {
			result = bn_normed;
		}
		break;
	}

	case SAM3_OP_BIAS_ADD: {
		/*
		 * Bias add on NHWC x[N,H,W,C] + bias[C]. Broadcast by
		 * reshaping bias to [1, 1, 1, C] so the channel axis
		 * aligns with the last dim of the input.
		 */
		int ba_C = node->inputs[1]->dims[0];
		int ba_shape[] = {1, 1, 1, ba_C};
		mlx_array ba_bias4d = mlx_array_new();
		rc = mlx_reshape(&ba_bias4d, inputs[1],
				 ba_shape, 4, stream);
		if (rc) { mlx_array_free(ba_bias4d); break; }

		rc = mlx_add(&result, inputs[0], ba_bias4d, stream);
		mlx_array_free(ba_bias4d);
		break;
	}

	default:
		sam3_log_error("metal: unsupported op %d", node->op);
		mlx_array_free(result);
		return SAM3_EINVAL;
	}

	if (rc != 0) {
		sam3_log_error("metal: MLX op %d failed", node->op);
		mlx_array_free(result);
		return SAM3_EBACKEND;
	}

	if (metal_map_put(mtl, node->output, result) < 0) {
		mlx_array_free(result);
		return SAM3_ENOMEM;
	}
	return SAM3_OK;
}

/* --- Backend vtable implementation  --- */

static enum sam3_error metal_init(struct sam3_backend *be)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;
	size_t capacity = mtl->arena_capacity;
	enum sam3_error err;

	if (capacity == 0)
		capacity = SAM3_METAL_ARENA_DEFAULT_CAPACITY;

	bool has_metal = false;
	mlx_metal_is_available(&has_metal);
	if (!has_metal) {
		sam3_log_error("Metal backend: Metal not available");
		return SAM3_EBACKEND;
	}

	err = sam3_arena_init(&mtl->arena, capacity);
	if (err != SAM3_OK) {
		sam3_log_error("Metal backend: arena init failed (%zu bytes)",
			       capacity);
		return err;
	}

	err = sam3_arena_init(&mtl->scratch,
			      SAM3_METAL_SCRATCH_DEFAULT_CAPACITY);
	if (err != SAM3_OK) {
		sam3_arena_free(&mtl->arena);
		sam3_log_error("Metal backend: scratch arena init failed");
		return err;
	}

	/*
	 * Default to F16 compute for reduced memory bandwidth on Metal.
	 * Set SAM3_METAL_F32=1 to force full F32 precision (useful for
	 * fixture comparison against CPU-generated Python references).
	 */
	{
		const char *f32_env = getenv("SAM3_METAL_F32");
		if (f32_env && f32_env[0] == '1')
			mtl->use_f16 = false;
		else
			mtl->use_f16 = true;
	}

	mtl->device = mlx_device_new_type(MLX_GPU, 0);
	mtl->stream = mlx_default_gpu_stream_new();
	if (metal_map_init(mtl) < 0) {
		sam3_log_error("metal: map alloc failed");
		return SAM3_ENOMEM;
	}

	size_t mem_limit = 0;
	mlx_set_memory_limit(&mem_limit,
			     (size_t)(0.75 * 16ULL * 1024 * 1024 * 1024));

	/* Build fused SiLU Metal kernel (created once, reused every dispatch) */
	{
		static const char *in_names[] = {"x"};
		static const char *out_names[] = {"out"};
		mlx_vector_string ins = mlx_vector_string_new_data(
			in_names, 1);
		mlx_vector_string outs = mlx_vector_string_new_data(
			out_names, 1);

		static const char silu_src[] =
			"uint i = thread_position_in_grid.x;\n"
			"T v = x[i];\n"
			"out[i] = v / (T(1) + metal::exp(-v));\n";

		mtl->silu_kernel = mlx_fast_metal_kernel_new(
			"silu", ins, outs, silu_src,
			/* header= */ "",
			/* ensure_row_contiguous= */ true,
			/* atomic_outputs= */ false);

		mlx_vector_string_free(ins);
		mlx_vector_string_free(outs);
	}

	/* Build fused Hard Swish kernel (created once, reused every dispatch) */
	{
		static const char *in_names[] = {"x"};
		static const char *out_names[] = {"out"};
		mlx_vector_string ins = mlx_vector_string_new_data(
			in_names, 1);
		mlx_vector_string outs = mlx_vector_string_new_data(
			out_names, 1);

		static const char hswish_src[] =
			"uint i = thread_position_in_grid.x;\n"
			"T v = x[i];\n"
			"T t = v + T(3);\n"
			"t = metal::clamp(t, T(0), T(6));\n"
			"out[i] = v * t / T(6);\n";

		mtl->hswish_kernel = mlx_fast_metal_kernel_new(
			"hswish", ins, outs, hswish_src,
			/* header= */ "",
			/* ensure_row_contiguous= */ true,
			/* atomic_outputs= */ false);

		mlx_vector_string_free(ins);
		mlx_vector_string_free(outs);
	}

	/* Build Q8_0→F16 dequant kernel (created once, reused per wrap) */
	{
		static const char *in_names[] = {"src"};
		static const char *out_names[] = {"out"};
		mlx_vector_string ins = mlx_vector_string_new_data(
			in_names, 1);
		mlx_vector_string outs = mlx_vector_string_new_data(
			out_names, 1);

		static const char dequant_header[] =
			"struct q8_block {\n"
			"    float scale;\n"
			"    int8_t data[32];\n"
			"};\n";

		static const char dequant_src[] =
			"uint tid = thread_position_in_grid.x;\n"
			"uint block_idx = tid / 32;\n"
			"uint elem_idx  = tid % 32;\n"
			"device const q8_block *blocks = "
				"(device const q8_block *)src;\n"
			"out[tid] = half(blocks[block_idx].data[elem_idx]) "
				"* half(blocks[block_idx].scale);\n";

		mtl->dequant_q8_kernel = mlx_fast_metal_kernel_new(
			"dequant_q8", ins, outs, dequant_src,
			dequant_header,
			/* ensure_row_contiguous= */ true,
			/* atomic_outputs= */ false);

		mlx_vector_string_free(ins);
		mlx_vector_string_free(outs);

		if (!mtl->dequant_q8_kernel.ctx) {
			sam3_log_error("metal: dequant_q8 kernel create failed");
			return SAM3_EBACKEND;
		}
	}

	sam3_log_info("Metal backend initialized (MLX-C, %s, arena: %zu bytes)",
		      mtl->use_f16 ? "F16" : "F32", capacity);
	return SAM3_OK;
}

static void metal_free(struct sam3_backend *be)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;

	metal_map_free(mtl);
	for (int i = 0; i < 13; i++) {
		if (mtl->relu_zeros[i].ctx)
			mlx_array_free(mtl->relu_zeros[i]);
		if (mtl->gelu_half[i].ctx)
			mlx_array_free(mtl->gelu_half[i]);
		if (mtl->gelu_rsqrt2[i].ctx)
			mlx_array_free(mtl->gelu_rsqrt2[i]);
		if (mtl->gelu_one[i].ctx)
			mlx_array_free(mtl->gelu_one[i]);
	}
	if (mtl->silu_kernel.ctx)
		mlx_fast_metal_kernel_free(mtl->silu_kernel);
	if (mtl->hswish_kernel.ctx)
		mlx_fast_metal_kernel_free(mtl->hswish_kernel);
	if (mtl->dequant_q8_kernel.ctx)
		mlx_fast_metal_kernel_free(mtl->dequant_q8_kernel);
	mlx_stream_free(mtl->stream);
	mlx_device_free(mtl->device);
	mlx_clear_cache();
	sam3_arena_free(&mtl->scratch);
	sam3_arena_free(&mtl->arena);
	sam3_log_debug("Metal backend freed");
}

static enum sam3_error metal_alloc_tensor(struct sam3_backend *be,
					  struct sam3_tensor *t)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;
	size_t elem_size;
	size_t nbytes;

	if (!t) {
		sam3_log_error("Metal alloc_tensor: NULL tensor");
		return SAM3_EINVAL;
	}

	if (t->n_dims < 1 || t->n_dims > SAM3_MAX_DIMS) {
		sam3_log_error("Metal alloc_tensor: invalid n_dims=%d",
			       t->n_dims);
		return SAM3_EINVAL;
	}

	elem_size = sam3_dtype_size(t->dtype);
	if (elem_size == 0) {
		sam3_log_error("Metal alloc_tensor: unknown dtype=%d",
			       t->dtype);
		return SAM3_EINVAL;
	}

	nbytes = (size_t)sam3_tensor_nelems(t) * elem_size;
	t->data = sam3_arena_alloc(&mtl->arena, nbytes);
	if (!t->data) {
		sam3_log_warn("Metal alloc_tensor: OOM (%zu bytes)", nbytes);
		return SAM3_ENOMEM;
	}

	t->nbytes = nbytes;
	sam3_tensor_compute_strides(t);

	sam3_log_debug("Metal alloc_tensor: %zu bytes @ %p", nbytes, t->data);
	return SAM3_OK;
}

/* Hash set entry for intermediate output detection in graph_eval. */
struct metal_imap_entry {
	const struct sam3_tensor *key;
	int idx;
};

static enum sam3_error metal_graph_eval(struct sam3_backend *be,
					struct sam3_graph *g)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;
	enum sam3_error err;

	sam3_arena_reset(&mtl->scratch);

	/* Pre-size tensor map: each node wraps up to ~3 input tensors */
	if (metal_map_ensure_capacity(mtl, g->n_nodes * 3) < 0) {
		sam3_log_error("metal_graph_eval: map pre-size failed");
		return SAM3_ENOMEM;
	}

	/* Mask reshape cache: avoids redundant reshape for shared masks */
	struct metal_mask_cache_entry mask_cache[METAL_MASK_CACHE_SLOTS];
	memset(mask_cache, 0, sizeof(mask_cache));

	/* Phase 1: translate all nodes to MLX lazy ops */
	for (int i = 0; i < g->n_nodes; i++) {
		err = metal_dispatch_node(mtl, &g->nodes[i], mask_cache);
		if (err != SAM3_OK) {
			sam3_log_error("metal_graph_eval: node %d failed", i);
			for (int mc = 0; mc < METAL_MASK_CACHE_SLOTS; mc++) {
				if (mask_cache[mc].val.ctx)
					mlx_array_free(mask_cache[mc].val);
			}
			return err;
		}
	}

	/* Free cached mask arrays — no longer needed after dispatch */
	for (int mc = 0; mc < METAL_MASK_CACHE_SLOTS; mc++) {
		if (mask_cache[mc].val.ctx)
			mlx_array_free(mask_cache[mc].val);
	}

	/*
	 * Phase 1.5: detect intermediate outputs in O(n * max_inputs).
	 *
	 * Build a pointer→node-index hash set, then scan all inputs
	 * to mark producing nodes as intermediate. Only final outputs
	 * are materialized by mlx_eval, letting MLX optimize its graph.
	 *
	 * Scratch reset is safe here: mlx_array_new_data() copies data
	 * eagerly, so Q8 dequant buffers are no longer referenced.
	 */
	bool is_intermediate[SAM3_GRAPH_MAX_NODES];
	memset(is_intermediate, 0, (size_t)g->n_nodes * sizeof(bool));

	{
		sam3_arena_reset(&mtl->scratch);

		int imap_cap = 1;
		while (imap_cap < g->n_nodes * 2)
			imap_cap <<= 1;
		unsigned imap_mask = (unsigned)(imap_cap - 1);

		struct metal_imap_entry *imap = sam3_arena_alloc(
			&mtl->scratch,
			(size_t)imap_cap * sizeof(struct metal_imap_entry));
		if (!imap) {
			sam3_log_error("metal_graph_eval: scratch OOM");
			return SAM3_ENOMEM;
		}
		memset(imap, 0,
		       (size_t)imap_cap * sizeof(struct metal_imap_entry));

		/* Insert all node output pointers */
		for (int i = 0; i < g->n_nodes; i++) {
			uintptr_t v = (uintptr_t)g->nodes[i].output;
			unsigned s = (unsigned)((v >> 4) ^ (v >> 16))
				& imap_mask;
			while (imap[s].key)
				s = (s + 1) & imap_mask;
			imap[s].key = g->nodes[i].output;
			imap[s].idx = i;
		}

		/* Mark outputs consumed as inputs to other nodes */
		for (int i = 0; i < g->n_nodes; i++) {
			for (int j = 0; j < g->nodes[i].n_inputs; j++) {
				const struct sam3_tensor *inp =
					g->nodes[i].inputs[j];
				uintptr_t v = (uintptr_t)inp;
				unsigned s = (unsigned)((v >> 4) ^ (v >> 16))
					& imap_mask;
				while (imap[s].key) {
					if (imap[s].key == inp
					    && imap[s].idx < i) {
						is_intermediate[imap[s].idx]
							= true;
						break;
					}
					s = (s + 1) & imap_mask;
				}
			}
		}
	}

	/* Phase 2: collect only final outputs and eval in one batch */
	mlx_vector_array outputs = mlx_vector_array_new();
	{
		int n_inter = 0, n_final = 0, n_nomap = 0;
		for (int i = 0; i < g->n_nodes; i++) {
			if (is_intermediate[i]) {
				n_inter++;
				continue;
			}
			mlx_array *out = metal_map_get(mtl, g->nodes[i].output);
			if (out) {
				mlx_vector_array_append_value(outputs, *out);
				n_final++;
			} else {
				n_nomap++;
			}
		}
		sam3_log_debug("metal_eval: %d nodes, %d intermediate, "
			"%d final, %d nomap",
			g->n_nodes, n_inter, n_final, n_nomap);
	}

	int rc = mlx_eval(outputs);
	mlx_vector_array_free(outputs);

	if (rc != 0) {
		sam3_log_error("metal_graph_eval: mlx_eval failed");
		return SAM3_EBACKEND;
	}

	/*
	 * Phase 3: readback or GPU-resident forwarding.
	 *
	 * When no_readback is set, skip host readback entirely.
	 * Evict within-graph intermediates and data-less outputs
	 * from the tensor map, but keep final outputs with host
	 * buffers resident for the next graph_eval to consume
	 * via metal_wrap_tensor.
	 */
	if (g->no_readback) {
		int n_kept = 0;
		for (int i = 0; i < g->n_nodes; i++) {
			struct sam3_tensor *out_t = g->nodes[i].output;
			if (is_intermediate[i] || !out_t->data)
				metal_map_evict(mtl, out_t);
			else
				n_kept++;
		}
		sam3_log_debug("metal_eval: no_readback, kept %d "
			"final outputs GPU-resident", n_kept);
	} else {
		int n_copied = 0, n_skip_inter = 0, n_skip_nodata = 0;

#ifdef SAM3_DEBUG_DUMP
		/* Debug: un-mark intermediates if the tensor was flagged
		 * for force-readback. Allocate a host buffer if the caller
		 * didn't supply one (sam3_dbg_pix_* pointers point at arena
		 * tensors that never had ->data set for graph intermediates). */
		for (int i = 0; i < g->n_nodes; i++) {
			struct sam3_tensor *t = g->nodes[i].output;
			if (!t->dbg_force_readback || !is_intermediate[i])
				continue;
			is_intermediate[i] = false;
			if (!t->data) {
				t->data = sam3_arena_alloc(
					&mtl->scratch,
					t->nbytes ? t->nbytes :
					(size_t)sam3_tensor_nelems(t) *
					sam3_dtype_size(t->dtype));
				if (t->data && !t->nbytes)
					t->nbytes =
						(size_t)sam3_tensor_nelems(t) *
						sam3_dtype_size(t->dtype);
			}
		}
#endif

		/* Count final outputs for scratch allocation */
		int n_final = 0;
		for (int i = 0; i < g->n_nodes; i++) {
			if (is_intermediate[i])
				continue;
			mlx_array *out_a = metal_map_get(
				mtl, g->nodes[i].output);
			if (out_a && g->nodes[i].output->data)
				n_final++;
		}

		mlx_array *readback = NULL;
		int *rb_idx = NULL;

		if (n_final > 0) {
			readback = sam3_arena_alloc(
				&mtl->scratch,
				(size_t)n_final * sizeof(mlx_array));
			rb_idx = sam3_arena_alloc(
				&mtl->scratch,
				(size_t)n_final * sizeof(int));
			if (!readback || !rb_idx) {
				sam3_log_error("metal: scratch OOM for "
					"readback batch");
				return SAM3_ENOMEM;
			}
		}

		/*
		 * Pass 1: build lazy contiguous arrays with optional
		 * F16->F32 cast. Evict intermediate and no-data nodes.
		 */
		int rb_count = 0;
		bool build_ok = true;
		for (int i = 0; i < g->n_nodes; i++) {
			struct sam3_tensor *out_t = g->nodes[i].output;
			mlx_array *out_a = metal_map_get(mtl, out_t);
			if (!out_a || !out_t->data) {
				n_skip_nodata++;
				goto evict_p3;
			}

			if (is_intermediate[i]) {
				n_skip_inter++;
				goto evict_p3;
			}

			{
				mlx_array contig = mlx_array_new();
				int crc = mlx_contiguous(
					&contig, *out_a,
					false, /* allow_col_major */
					mtl->stream);
				if (crc) {
					mlx_array_free(contig);
					sam3_log_error(
						"metal: contiguous failed");
					build_ok = false;
					break;
				}

				/*
				 * If Metal computed in F16 but SAM3
				 * tensor expects F32, add lazy cast.
				 */
				if (mlx_array_dtype(contig) == MLX_FLOAT16
				    && out_t->dtype == SAM3_DTYPE_F32) {
					mlx_array f32 = mlx_array_new();
					int cast_rc = mlx_astype(
						&f32, contig,
						MLX_FLOAT32,
						mtl->stream);
					if (cast_rc == 0) {
						mlx_array_free(contig);
						contig = f32;
					} else {
						mlx_array_free(f32);
					}
				}

				readback[rb_count] = contig;
				rb_idx[rb_count] = i;
				rb_count++;
			}
			continue;

		evict_p3:
			metal_map_evict(mtl, out_t);
		}

		/* Single batched eval for all readback arrays */
		if (build_ok && rb_count > 0) {
			mlx_vector_array va =
				mlx_vector_array_new();
			for (int j = 0; j < rb_count; j++)
				mlx_vector_array_append_value(
					va, readback[j]);
			int eval_rc = mlx_eval(va);
			mlx_vector_array_free(va);
			if (eval_rc != 0) {
				sam3_log_error(
					"metal: readback eval failed");
				build_ok = false;
			}
		}

		/* Pass 2: copy data to host and clean up */
		for (int j = 0; j < rb_count; j++) {
			struct sam3_tensor *out_t =
				g->nodes[rb_idx[j]].output;
			const void *src = NULL;

			if (build_ok) {
				mlx_dtype mtype =
					mlx_array_dtype(readback[j]);
				switch (mtype) {
				case MLX_FLOAT32:
					src = mlx_array_data_float32(
						readback[j]);
					break;
				case MLX_FLOAT16:
					src = mlx_array_data_float16(
						readback[j]);
					break;
				case MLX_BFLOAT16:
					src = mlx_array_data_bfloat16(
						readback[j]);
					break;
				case MLX_INT32:
					src = mlx_array_data_int32(
						readback[j]);
					break;
				case MLX_INT8:
					src = mlx_array_data_int8(
						readback[j]);
					break;
				default:
					sam3_log_error("metal: unsupported "
						"output dtype");
					break;
				}
			}

			if (src) {
				memcpy(out_t->data, src,
				       out_t->nbytes);
				n_copied++;
			}
			mlx_array_free(readback[j]);
			metal_map_evict(mtl, out_t);
		}

		sam3_log_debug("metal_eval: phase3 copied=%d "
			"skip_inter=%d skip_nodata=%d",
				n_copied, n_skip_inter, n_skip_nodata);
	}

	/*
	 * Phase 4: evict ephemeral input tensors.
	 *
	 * Tensors created by gh_tensor_wrap live in the caller's
	 * scratch arena, which gets reset between graph_eval calls.
	 * Subsequent allocations reuse the same header address but
	 * may point at different external data. Without eviction,
	 * metal_wrap_tensor would return a stale mlx_array with
	 * data copied from the previous graph_eval.
	 */
	for (int i = 0; i < g->n_nodes; i++) {
		for (int j = 0; j < g->nodes[i].n_inputs; j++) {
			const struct sam3_tensor *inp =
				g->nodes[i].inputs[j];
			if (inp && inp->ephemeral)
				metal_map_evict(mtl, inp);
		}
	}

	return SAM3_OK;
}

static const struct sam3_backend_ops metal_ops = {
	.init              = metal_init,
	.free              = metal_free,
	.alloc_tensor      = metal_alloc_tensor,
	.graph_eval        = metal_graph_eval,
	.arena_reset       = NULL,  /* MLX manages memory automatically */
	.cache_invalidate  = metal_cache_invalidate,
};

const struct sam3_backend_ops *sam3_metal_backend_ops(void)
{
	return &metal_ops;
}

#else /* !SAM3_HAS_METAL */

const struct sam3_backend_ops *sam3_metal_backend_ops(void)
{
	return NULL;
}

#endif /* SAM3_HAS_METAL */
