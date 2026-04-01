/*
 * src/backend/metal/metal_backend.c - Metal backend implementation (MLX-C)
 *
 * Implements the Metal compute backend by translating SAM3 compute graphs
 * into MLX-C lazy operations. Each graph_eval builds an MLX op graph,
 * calls mlx_eval() once for a single GPU dispatch, then copies results
 * back to SAM3 tensors. Q8_0 tensors are dequantized to F16 on host.
 *
 * Key types:  sam3_metal_backend
 * Depends on: metal_backend.h, core/tensor.h, core/quant.h, core/half.h,
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
#include "core/half.h"
#include <string.h>

/* ── Dtype mapping ─────────────────────────────────────────────────── */

/*
 * metal_map_dtype - Convert sam3_dtype to mlx_dtype.
 *
 * Returns MLX_FLOAT16 for Q8_0 (caller must dequantize first).
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

/* ── Q8_0 dequantization to F16 ───────────────────────────────────── */

/*
 * metal_dequant_q8_to_f16 - Dequantize Q8_0 blocks to F16 buffer.
 *
 * @src:     Q8 block array
 * @dst:     Destination F16 buffer (caller-allocated, nelems * 2 bytes)
 * @nelems:  Number of elements
 * @scratch: Scratch arena for F32 temporary buffer
 */
static void metal_dequant_q8_to_f16(const struct sam3_q8_block *src,
				     uint16_t *dst, int nelems,
				     struct sam3_arena *scratch)
{
	float *tmp = sam3_arena_alloc(scratch, (size_t)nelems * sizeof(float));
	if (!tmp) {
		sam3_log_error("metal: scratch OOM for Q8 dequant (%d elems)",
			       nelems);
		return;
	}

	sam3_q8_dequantize(src, tmp, nelems);

	for (int i = 0; i < nelems; i++)
		dst[i] = f32_to_fp16(tmp[i]);
}

/* ── Tensor-to-mlx_array lookup table ─────────────────────────────── */

#define METAL_MAP_SIZE 8192  /* Must be power of 2 */

struct metal_tensor_map {
	const struct sam3_tensor *keys[METAL_MAP_SIZE];
	mlx_array                vals[METAL_MAP_SIZE];
	int                      count;
};

static void metal_map_init(struct metal_tensor_map *m)
{
	memset(m->keys, 0, sizeof(m->keys));
	m->count = 0;
}

static void metal_map_free(struct metal_tensor_map *m)
{
	for (int i = 0; i < METAL_MAP_SIZE; i++) {
		if (m->keys[i])
			mlx_array_free(m->vals[i]);
	}
	m->count = 0;
}

static unsigned metal_map_hash(const struct sam3_tensor *ptr)
{
	uintptr_t v = (uintptr_t)ptr;
	v = (v >> 4) ^ (v >> 16);
	return (unsigned)(v & (METAL_MAP_SIZE - 1));
}

static mlx_array *metal_map_get(struct metal_tensor_map *m,
				const struct sam3_tensor *key)
{
	unsigned idx = metal_map_hash(key);
	for (unsigned i = 0; i < METAL_MAP_SIZE; i++) {
		unsigned slot = (idx + i) & (METAL_MAP_SIZE - 1);
		if (m->keys[slot] == key)
			return &m->vals[slot];
		if (!m->keys[slot])
			return NULL;
	}
	return NULL;
}

static void metal_map_put(struct metal_tensor_map *m,
			   const struct sam3_tensor *key, mlx_array val)
{
	unsigned idx = metal_map_hash(key);
	for (unsigned i = 0; i < METAL_MAP_SIZE; i++) {
		unsigned slot = (idx + i) & (METAL_MAP_SIZE - 1);
		if (!m->keys[slot]) {
			m->keys[slot] = key;
			m->vals[slot] = val;
			m->count++;
			return;
		}
	}
	sam3_log_error("metal: tensor map full");
}

/* ── Tensor wrapping ──────────────────────────────────────────────── */

/*
 * metal_wrap_tensor - Get or create an mlx_array for a sam3_tensor.
 *
 * If the tensor is already in the map, returns the existing handle.
 * Otherwise creates a new mlx_array from the tensor's host data,
 * handling Q8_0 dequantization to F16 if needed.
 */
static mlx_array metal_wrap_tensor(struct metal_tensor_map *map,
				   const struct sam3_tensor *t,
				   struct sam3_arena *scratch,
				   mlx_stream stream)
{
	mlx_array *existing = metal_map_get(map, t);
	if (existing)
		return *existing;

	mlx_dtype mtype;
	if (metal_map_dtype(t->dtype, &mtype) < 0) {
		sam3_log_error("metal: unsupported dtype %d", t->dtype);
		return mlx_array_new();
	}

	const void *data = t->data;

	if (t->dtype == SAM3_DTYPE_Q8_0) {
		int nelems = sam3_tensor_nelems(t);
		uint16_t *fp16_buf = sam3_arena_alloc(scratch,
					(size_t)nelems * sizeof(uint16_t));
		if (!fp16_buf) {
			sam3_log_error("metal: scratch OOM for Q8 wrap");
			return mlx_array_new();
		}
		metal_dequant_q8_to_f16(
			(const struct sam3_q8_block *)t->data,
			fp16_buf, nelems, scratch);
		data = fp16_buf;
	}

	mlx_array arr = mlx_array_new_data(data, t->dims, t->n_dims, mtype);
	metal_map_put(map, t, arr);
	return arr;
}

/* ── Op dispatch ──────────────────────────────────────────────────── */

/*
 * metal_dispatch_node - Translate one sam3_node into MLX-C lazy ops.
 *
 * The result mlx_array is stored in the map keyed by node->output.
 * Returns SAM3_OK on success.
 */
static enum sam3_error metal_dispatch_node(const struct sam3_node *node,
					   struct metal_tensor_map *map,
					   struct sam3_arena *scratch,
					   mlx_stream stream)
{
	mlx_array result = mlx_array_new();
	mlx_array inputs[SAM3_NODE_MAX_INPUTS];
	int rc;

	for (int i = 0; i < node->n_inputs; i++)
		inputs[i] = metal_wrap_tensor(map, node->inputs[i],
					      scratch, stream);

	switch (node->op) {
	case SAM3_OP_MATMUL:
		rc = mlx_matmul(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_ADD:
		rc = mlx_add(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_MUL:
		rc = mlx_multiply(&result, inputs[0], inputs[1], stream);
		break;

	case SAM3_OP_SOFTMAX:
		rc = mlx_softmax(&result, inputs[0], true, stream);
		break;

	case SAM3_OP_RELU: {
		int scalar_shape[] = {1};
		float zero = 0.0f;
		mlx_array zero_arr = mlx_array_new_data(
			&zero, scalar_shape, 1, MLX_FLOAT32);
		mlx_dtype in_dtype;
		mlx_array_dtype(&in_dtype, inputs[0]);
		mlx_array zero_cast = mlx_array_new();
		mlx_astype(&zero_cast, zero_arr, in_dtype, stream);
		rc = mlx_maximum(&result, inputs[0], zero_cast, stream);
		mlx_array_free(zero_cast);
		mlx_array_free(zero_arr);
		break;
	}

	case SAM3_OP_GELU: {
		/* gelu(x) ~ x * sigmoid(1.702 * x) */
		int scalar_shape[] = {1};
		float coeff = 1.702f;
		mlx_dtype in_dtype;
		mlx_array_dtype(&in_dtype, inputs[0]);
		mlx_array coeff_arr = mlx_array_new_data(
			&coeff, scalar_shape, 1, MLX_FLOAT32);
		mlx_array coeff_cast = mlx_array_new();
		mlx_astype(&coeff_cast, coeff_arr, in_dtype, stream);

		mlx_array scaled = mlx_array_new();
		mlx_multiply(&scaled, inputs[0], coeff_cast, stream);

		mlx_array sig = mlx_array_new();
		mlx_sigmoid(&sig, scaled, stream);

		rc = mlx_multiply(&result, inputs[0], sig, stream);

		mlx_array_free(sig);
		mlx_array_free(scaled);
		mlx_array_free(coeff_cast);
		mlx_array_free(coeff_arr);
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
		int sh = node->params[0] ? node->params[0] : 1;
		int sw = node->params[1] ? node->params[1] : 1;
		int ph = node->params[2];
		int pw = node->params[3];
		rc = mlx_conv2d(&result, inputs[0], inputs[1],
				sh, sw, ph, pw,
				1, 1,  /* dilation */
				1,     /* groups */
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
		int ndim;
		mlx_array_ndim(&ndim, inputs[0]);
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

	metal_map_put(map, node->output, result);
	return SAM3_OK;
}

/* ── Backend vtable implementation ────────────────────────────────── */

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

	mtl->device = mlx_device_new_type(MLX_GPU, 0);
	mtl->stream = mlx_default_gpu_stream_new();

	size_t mem_limit = 0;
	mlx_set_memory_limit(&mem_limit,
			     (size_t)(0.75 * 16ULL * 1024 * 1024 * 1024));

	sam3_log_info("Metal backend initialized (MLX-C, arena: %zu bytes)",
		      capacity);
	return SAM3_OK;
}

static void metal_free(struct sam3_backend *be)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;

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

static enum sam3_error metal_graph_eval(struct sam3_backend *be,
					struct sam3_graph *g)
{
	struct sam3_metal_backend *mtl = (struct sam3_metal_backend *)be;
	struct metal_tensor_map map;
	enum sam3_error err;

	metal_map_init(&map);
	sam3_arena_reset(&mtl->scratch);

	/* Phase 1: translate all nodes to MLX lazy ops */
	for (int i = 0; i < g->n_nodes; i++) {
		err = metal_dispatch_node(&g->nodes[i], &map,
					  &mtl->scratch, mtl->stream);
		if (err != SAM3_OK) {
			sam3_log_error("metal_graph_eval: node %d failed", i);
			metal_map_free(&map);
			return err;
		}
	}

	/* Phase 2: collect outputs and eval in one batch */
	mlx_vector_array outputs = mlx_vector_array_new();
	for (int i = 0; i < g->n_nodes; i++) {
		mlx_array *out = metal_map_get(&map, g->nodes[i].output);
		if (out)
			mlx_vector_array_append(outputs, *out);
	}

	int rc = mlx_eval(outputs);
	mlx_vector_array_free(outputs);

	if (rc != 0) {
		sam3_log_error("metal_graph_eval: mlx_eval failed");
		metal_map_free(&map);
		return SAM3_EBACKEND;
	}

	/* Phase 3: copy results back to sam3 tensors */
	for (int i = 0; i < g->n_nodes; i++) {
		struct sam3_tensor *out_t = g->nodes[i].output;
		mlx_array *out_a = metal_map_get(&map, out_t);
		if (!out_a || !out_t->data)
			continue;

		const void *src = NULL;
		mlx_dtype mtype;
		mlx_array_dtype(&mtype, *out_a);

		switch (mtype) {
		case MLX_FLOAT32:
			src = mlx_array_data_float32(*out_a);
			break;
		case MLX_FLOAT16:
			src = mlx_array_data_float16(*out_a);
			break;
		case MLX_BFLOAT16:
			src = mlx_array_data_bfloat16(*out_a);
			break;
		case MLX_INT32:
			src = mlx_array_data_int32(*out_a);
			break;
		case MLX_INT8:
			src = mlx_array_data_int8(*out_a);
			break;
		default:
			sam3_log_error("metal: unsupported output dtype");
			break;
		}

		if (src)
			memcpy(out_t->data, src, out_t->nbytes);
	}

	metal_map_free(&map);
	return SAM3_OK;
}

static const struct sam3_backend_ops metal_ops = {
	.init         = metal_init,
	.free         = metal_free,
	.alloc_tensor = metal_alloc_tensor,
	.graph_eval   = metal_graph_eval,
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
