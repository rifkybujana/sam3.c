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
#include <math.h>
#include <stdbool.h>
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
 * metal_dequant_q8_to_f16 - Dequantize Q8_0 blocks directly to F16.
 *
 * Single-pass: converts Q8 blocks to fp16 without an intermediate
 * f32 buffer. NEON-accelerated on aarch64, scalar fallback otherwise.
 *
 * @src:    Q8 block array
 * @dst:    Destination F16 buffer (caller-allocated, nelems * 2 bytes)
 * @nelems: Number of elements
 */
static void metal_dequant_q8_to_f16(const struct sam3_q8_block *src,
				     uint16_t *dst, int nelems)
{
	int nblocks = nelems / SAM3_Q8_BLOCK_SIZE;
	int tail = nelems % SAM3_Q8_BLOCK_SIZE;

#if defined(SAM3_HAS_NEON) || \
	(defined(__aarch64__) && defined(__ARM_NEON))
	for (int b = 0; b < nblocks; b++) {
		const struct sam3_q8_block *blk = &src[b];
		uint16_t *out = &dst[b * SAM3_Q8_BLOCK_SIZE];
		float32x4_t vscale = vdupq_n_f32(blk->scale);

		for (int i = 0; i < SAM3_Q8_BLOCK_SIZE; i += 8) {
			int8x8_t d8 = vld1_s8(&blk->data[i]);
			int16x8_t d16 = vmovl_s8(d8);
			float32x4_t flo = vmulq_f32(
				vcvtq_f32_s32(
					vmovl_s16(vget_low_s16(d16))),
				vscale);
			float32x4_t fhi = vmulq_f32(
				vcvtq_f32_s32(
					vmovl_s16(vget_high_s16(d16))),
				vscale);
			f32x4x2_to_fp16x8(&out[i], flo, fhi);
		}
	}
#else
	for (int b = 0; b < nblocks; b++) {
		const struct sam3_q8_block *blk = &src[b];
		uint16_t *out = &dst[b * SAM3_Q8_BLOCK_SIZE];
		float scale = blk->scale;

		for (int i = 0; i < SAM3_Q8_BLOCK_SIZE; i++)
			out[i] = f32_to_fp16((float)blk->data[i] * scale);
	}
#endif

	if (tail > 0) {
		const struct sam3_q8_block *blk = &src[nblocks];
		uint16_t *out = &dst[nblocks * SAM3_Q8_BLOCK_SIZE];
		float scale = blk->scale;
		for (int i = 0; i < tail; i++)
			out[i] = f32_to_fp16((float)blk->data[i] * scale);
	}
}

/* ── Tensor-to-mlx_array lookup table ─────────────────────────────── */

/*
 * Tombstone sentinel for deleted slots in the open-addressing hash
 * table. Required so that linear probing does not stop early when a
 * slot in the middle of a probe chain is evicted.
 */
#define METAL_MAP_TOMBSTONE ((const struct sam3_tensor *)(uintptr_t)1)

static void metal_map_init(struct sam3_metal_backend *mtl)
{
	memset(mtl->map_keys, 0, sizeof(mtl->map_keys));
	mtl->map_count = 0;
}

static void metal_map_free(struct sam3_metal_backend *mtl)
{
	for (int i = 0; i < SAM3_METAL_MAP_SIZE; i++) {
		if (mtl->map_keys[i] &&
		    mtl->map_keys[i] != METAL_MAP_TOMBSTONE)
			mlx_array_free(mtl->map_vals[i]);
	}
	memset(mtl->map_keys, 0, sizeof(mtl->map_keys));
	mtl->map_count = 0;
}

static unsigned metal_map_hash(const struct sam3_tensor *ptr)
{
	uintptr_t v = (uintptr_t)ptr;
	v = (v >> 4) ^ (v >> 16);
	return (unsigned)(v & (SAM3_METAL_MAP_SIZE - 1));
}

static mlx_array *metal_map_get(struct sam3_metal_backend *mtl,
				const struct sam3_tensor *key)
{
	unsigned idx = metal_map_hash(key);
	for (unsigned i = 0; i < SAM3_METAL_MAP_SIZE; i++) {
		unsigned slot = (idx + i) & (SAM3_METAL_MAP_SIZE - 1);
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
	unsigned idx = metal_map_hash(key);
	for (unsigned i = 0; i < SAM3_METAL_MAP_SIZE; i++) {
		unsigned slot = (idx + i) & (SAM3_METAL_MAP_SIZE - 1);
		if (!mtl->map_keys[slot] ||
		    mtl->map_keys[slot] == METAL_MAP_TOMBSTONE) {
			mtl->map_keys[slot] = key;
			mtl->map_vals[slot] = val;
			mtl->map_count++;
			return 0;
		}
	}
	sam3_log_error("metal: tensor map full (%d entries)",
		       mtl->map_count);
	return -1;
}

static void metal_map_evict(struct sam3_metal_backend *mtl,
			    const struct sam3_tensor *key)
{
	unsigned idx = metal_map_hash(key);
	for (unsigned i = 0; i < SAM3_METAL_MAP_SIZE; i++) {
		unsigned slot = (idx + i) & (SAM3_METAL_MAP_SIZE - 1);
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

/* ── Tensor wrapping ──────────────────────────────────────────────── */

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
		bool valid = ((int)mlx_array_ndim(*existing) == t->n_dims);
		if (valid)
			valid = (mlx_array_dtype(*existing) == expected_mtype);
		for (int i = 0; valid && i < t->n_dims; i++) {
			if (mlx_array_dim(*existing, i) != t->dims[i])
				valid = false;
		}
		if (valid)
			return *existing;
		/* Stale entry (pointer reuse) — evict and re-wrap */
		metal_map_evict(mtl, t);
	}

	mlx_dtype mtype;
	if (metal_map_dtype(t->dtype, &mtype) < 0) {
		sam3_log_error("metal: unsupported dtype %d", t->dtype);
		return mlx_array_new();
	}

	const void *data = t->data;

	if (t->dtype == SAM3_DTYPE_Q8_0) {
		int nelems = sam3_tensor_nelems(t);
		uint16_t *fp16_buf = sam3_arena_alloc(&mtl->scratch,
					(size_t)nelems * sizeof(uint16_t));
		if (!fp16_buf) {
			sam3_log_error("metal: scratch OOM for Q8 wrap");
			return mlx_array_new();
		}
		metal_dequant_q8_to_f16(
			(const struct sam3_q8_block *)t->data,
			fp16_buf, nelems);
		data = fp16_buf;
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

/* ── Op dispatch ──────────────────────────────────────────────────── */

/*
 * metal_dispatch_node - Translate one sam3_node into MLX-C lazy ops.
 *
 * The result mlx_array is stored in the map keyed by node->output.
 * Returns SAM3_OK on success.
 */
static enum sam3_error metal_dispatch_node(struct sam3_metal_backend *mtl,
					   const struct sam3_node *node)
{
	mlx_array result = mlx_array_new();
	mlx_array inputs[SAM3_NODE_MAX_INPUTS];
	mlx_stream stream = mtl->stream;
	int rc;

	for (int i = 0; i < node->n_inputs; i++)
		inputs[i] = metal_wrap_tensor(mtl, node->inputs[i]);

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
		rc = mlx_softmax_axis(&result, inputs[0], -1, true, stream);
		break;

	case SAM3_OP_RELU: {
		int scalar_shape[] = {1};
		float zero = 0.0f;
		mlx_array zero_arr = mlx_array_new_data(
			&zero, scalar_shape, 1, MLX_FLOAT32);
		mlx_dtype in_dtype = mlx_array_dtype(inputs[0]);
		mlx_array zero_cast = mlx_array_new();
		mlx_astype(&zero_cast, zero_arr, in_dtype, stream);
		rc = mlx_maximum(&result, inputs[0], zero_cast, stream);
		mlx_array_free(zero_cast);
		mlx_array_free(zero_arr);
		break;
	}

	case SAM3_OP_GELU: {
		/*
		 * Exact GELU matching PyTorch nn.GELU(approximate='none'):
		 *   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
		 */
		int scalar_shape[] = {1};
		mlx_dtype in_dtype = mlx_array_dtype(inputs[0]);

		float half_val = 0.5f;
		float rsqrt2_val = 0.7071067811865475f; /* 1/sqrt(2) */
		float one_val = 1.0f;

		mlx_array half_arr = mlx_array_new_data(
			&half_val, scalar_shape, 1, MLX_FLOAT32);
		mlx_array rsqrt2_arr = mlx_array_new_data(
			&rsqrt2_val, scalar_shape, 1, MLX_FLOAT32);
		mlx_array one_arr = mlx_array_new_data(
			&one_val, scalar_shape, 1, MLX_FLOAT32);

		mlx_array half_cast = mlx_array_new();
		mlx_array rsqrt2_cast = mlx_array_new();
		mlx_array one_cast = mlx_array_new();
		mlx_astype(&half_cast, half_arr, in_dtype, stream);
		mlx_astype(&rsqrt2_cast, rsqrt2_arr, in_dtype, stream);
		mlx_astype(&one_cast, one_arr, in_dtype, stream);

		/* x / sqrt(2) */
		mlx_array x_scaled = mlx_array_new();
		mlx_multiply(&x_scaled, inputs[0], rsqrt2_cast, stream);

		/* erf(x / sqrt(2)) */
		mlx_array erf_val = mlx_array_new();
		mlx_erf(&erf_val, x_scaled, stream);

		/* 1 + erf(...) */
		mlx_array one_plus_erf = mlx_array_new();
		mlx_add(&one_plus_erf, one_cast, erf_val, stream);

		/* 0.5 * x */
		mlx_array half_x = mlx_array_new();
		mlx_multiply(&half_x, inputs[0], half_cast, stream);

		/* 0.5 * x * (1 + erf(...)) */
		rc = mlx_multiply(&result, half_x, one_plus_erf, stream);

		mlx_array_free(half_x);
		mlx_array_free(one_plus_erf);
		mlx_array_free(erf_val);
		mlx_array_free(x_scaled);
		mlx_array_free(one_cast);
		mlx_array_free(rsqrt2_cast);
		mlx_array_free(half_cast);
		mlx_array_free(one_arr);
		mlx_array_free(rsqrt2_arr);
		mlx_array_free(half_arr);
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
		 * SAM3 uses NCHW (input) and OIHW (weight).
		 * MLX conv2d expects NHWC (input) and OHWI (weight).
		 * Transpose before conv, transpose result back.
		 *
		 * params[0] = stride (same for H and W)
		 * params[1] = padding (same for H and W)
		 */
		int stride = node->params[0] ? node->params[0] : 1;
		int pad = node->params[1];

		/* Input NCHW -> NHWC: perm [0,2,3,1] */
		int in_perm[] = {0, 2, 3, 1};
		mlx_array in_nhwc = mlx_array_new();
		rc = mlx_transpose_axes(&in_nhwc, inputs[0],
					in_perm, 4, stream);
		if (rc) { mlx_array_free(in_nhwc); break; }

		/* Weight OIHW -> OHWI: perm [0,2,3,1] */
		int wt_perm[] = {0, 2, 3, 1};
		mlx_array wt_ohwi = mlx_array_new();
		rc = mlx_transpose_axes(&wt_ohwi, inputs[1],
					wt_perm, 4, stream);
		if (rc) {
			mlx_array_free(in_nhwc);
			mlx_array_free(wt_ohwi);
			break;
		}

		/* Conv2d in NHWC layout */
		mlx_array conv_out = mlx_array_new();
		rc = mlx_conv2d(&conv_out, in_nhwc, wt_ohwi,
				stride, stride, pad, pad,
				1, 1,  /* dilation */
				1,     /* groups */
				stream);
		mlx_array_free(in_nhwc);
		mlx_array_free(wt_ohwi);
		if (rc) { mlx_array_free(conv_out); break; }

		/* Result NHWC -> NCHW: perm [0,3,1,2] */
		int out_perm[] = {0, 3, 1, 2};
		rc = mlx_transpose_axes(&result, conv_out,
					out_perm, 4, stream);
		mlx_array_free(conv_out);
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
		/* silu(x) = x * sigmoid(x) */
		mlx_array sig = mlx_array_new();
		rc = mlx_sigmoid(&sig, inputs[0], stream);
		if (!rc)
			rc = mlx_multiply(&result, inputs[0], sig, stream);
		mlx_array_free(sig);
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
		 * Nearest-neighbor upsample of [N,C,H,W] by integer scale.
		 * Strategy: reshape to [N,C,H,1,W,1], broadcast to
		 * [N,C,H,s,W,s], then reshape to [N,C,H*s,W*s].
		 */
		int scale = node->params[0];
		int ndim = (int)mlx_array_ndim(inputs[0]);
		if (ndim != 4) {
			sam3_log_error("metal: upsample requires 4D input");
			mlx_array_free(result);
			return SAM3_EINVAL;
		}
		int n = mlx_array_dim(inputs[0], 0);
		int c = mlx_array_dim(inputs[0], 1);
		int h = mlx_array_dim(inputs[0], 2);
		int w = mlx_array_dim(inputs[0], 3);

		/* Reshape: [N,C,H,1,W,1] */
		int shape6[] = {n, c, h, 1, w, 1};
		mlx_array reshaped = mlx_array_new();
		rc = mlx_reshape(&reshaped, inputs[0], shape6, 6, stream);
		if (rc) { mlx_array_free(reshaped); break; }

		/* Broadcast: [N,C,H,s,W,s] */
		int bcast[] = {n, c, h, scale, w, scale};
		mlx_array expanded = mlx_array_new();
		rc = mlx_broadcast_to(&expanded, reshaped, bcast, 6, stream);
		mlx_array_free(reshaped);
		if (rc) { mlx_array_free(expanded); break; }

		/* Reshape: [N,C,H*s,W*s] */
		int shape4[] = {n, c, h * scale, w * scale};
		rc = mlx_reshape(&result, expanded, shape4, 4, stream);
		mlx_array_free(expanded);
		break;
	}

	case SAM3_OP_ROPE: {
		/*
		 * Rotary position embedding from primitives.
		 * inputs[0] = x [batch, seq, heads, head_dim]
		 * inputs[1] = cos [seq, head_dim/2]
		 * inputs[2] = sin [seq, head_dim/2]
		 * params[0] = head_dim
		 *
		 * Interleaved pairs: x[...,2d], x[...,2d+1]
		 *   out_even = x_even * cos - x_odd * sin
		 *   out_odd  = x_even * sin + x_odd * cos
		 *
		 * Strategy: reshape x to [..., head_dim/2, 2], split,
		 * apply rotation, interleave back.
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
		const char *mask_mode = "";
		if (node->n_inputs > 3 && node->inputs[3]) {
			mask4d = mlx_array_new();
			int seq_q = (ndims == 4)
				? node->inputs[0]->dims[2]
				: node->inputs[0]->dims[0];
			int seq_kv = (ndims == 4)
				? node->inputs[1]->dims[2]
				: node->inputs[1]->dims[0];
			int mshape[] = {1, 1, seq_q, seq_kv};
			rc = mlx_reshape(&mask4d, inputs[3], mshape, 4,
					  stream);
			if (rc) {
				if (ndims != 4) {
					mlx_array_free(q_in);
					mlx_array_free(k_in);
					mlx_array_free(v_in);
				}
				mlx_array_free(mask4d);
				break;
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
		if (mask4d.ctx)
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
		 * SAM3: NCHW input, IOHW weight [C_in, C_out, KH, KW].
		 * MLX:  NHWC input, OHWI weight [C_out, KH, KW, C_in].
		 */
		int stride = node->params[0] ? node->params[0] : 1;
		int pad = node->params[1];

		/* Input NCHW -> NHWC: perm [0,2,3,1] */
		int ct_in_perm[] = {0, 2, 3, 1};
		mlx_array ct_in_nhwc = mlx_array_new();
		rc = mlx_transpose_axes(&ct_in_nhwc, inputs[0],
					ct_in_perm, 4, stream);
		if (rc) { mlx_array_free(ct_in_nhwc); break; }

		/* Weight IOHW -> OHWI: perm [1,2,3,0] */
		int ct_wt_perm[] = {1, 2, 3, 0};
		mlx_array ct_wt_ohwi = mlx_array_new();
		rc = mlx_transpose_axes(&ct_wt_ohwi, inputs[1],
					ct_wt_perm, 4, stream);
		if (rc) {
			mlx_array_free(ct_in_nhwc);
			mlx_array_free(ct_wt_ohwi);
			break;
		}

		/* Debug: print shapes going into conv_transpose2d */
		{
			size_t in_ndim = mlx_array_ndim(ct_in_nhwc);
			size_t wt_ndim = mlx_array_ndim(ct_wt_ohwi);
			const int *in_sh = mlx_array_shape(ct_in_nhwc);
			const int *wt_sh = mlx_array_shape(ct_wt_ohwi);
			sam3_log_debug("convT2d: in_nhwc [%d,%d,%d,%d] "
				"wt_ohwi [%d,%d,%d,%d] s=%d p=%d",
				in_sh[0], in_sh[1], in_sh[2], in_sh[3],
				wt_sh[0], wt_sh[1], wt_sh[2], wt_sh[3],
				stride, pad);
		}

		/* ConvTranspose2d in NHWC layout */
		mlx_array ct_conv_out = mlx_array_new();
		rc = mlx_conv_transpose2d(&ct_conv_out, ct_in_nhwc,
					   ct_wt_ohwi,
					   stride, stride,
					   pad, pad,
					   1, 1,  /* dilation */
					   0, 0,  /* output_padding */
					   1,     /* groups */
					   stream);
		mlx_array_free(ct_in_nhwc);
		mlx_array_free(ct_wt_ohwi);
		if (rc) { mlx_array_free(ct_conv_out); break; }

		/* Debug: print output shape before transpose */
		{
			const int *out_sh = mlx_array_shape(ct_conv_out);
			sam3_log_debug("convT2d: out_nhwc [%d,%d,%d,%d]",
				out_sh[0], out_sh[1], out_sh[2], out_sh[3]);
		}

		/* Result NHWC -> NCHW: perm [0,3,1,2] */
		int ct_out_perm[] = {0, 3, 1, 2};
		rc = mlx_transpose_axes(&result, ct_conv_out,
					ct_out_perm, 4, stream);
		mlx_array_free(ct_conv_out);
		break;
	}

	case SAM3_OP_MAXPOOL2D: {
		/*
		 * MaxPool2d for non-overlapping case (stride == kernel_size).
		 * NCHW [N, C, H, W] -> reshape [N, C, H/k, k, W/k, k]
		 * then max over k-dims (axes 3, 5) -> [N, C, H/k, W/k].
		 */
		int mp_k = node->params[0];
		int mp_s = node->params[1];
		const struct sam3_tensor *mp_in = node->inputs[0];
		int mp_N = mp_in->dims[0];
		int mp_C = mp_in->dims[1];
		int mp_H = mp_in->dims[2];
		int mp_W = mp_in->dims[3];

		if (mp_s != mp_k || mp_H % mp_k != 0 || mp_W % mp_k != 0) {
			sam3_log_error("metal: maxpool2d only supports "
				       "stride==kernel, even dims");
			mlx_array_free(result);
			return SAM3_EINVAL;
		}

		/* Reshape to [N, C, H/k, k, W/k, k] */
		int mp_shape[] = {mp_N, mp_C, mp_H / mp_k, mp_k,
				  mp_W / mp_k, mp_k};
		mlx_array mp_reshaped = mlx_array_new();
		rc = mlx_reshape(&mp_reshaped, inputs[0],
				 mp_shape, 6, stream);
		if (rc) { mlx_array_free(mp_reshaped); break; }

		/* Max over axis 5 (last k-dim) */
		mlx_array mp_tmp = mlx_array_new();
		rc = mlx_max_axis(&mp_tmp, mp_reshaped, 5, false, stream);
		mlx_array_free(mp_reshaped);
		if (rc) { mlx_array_free(mp_tmp); break; }

		/* Max over axis 3 (remaining k-dim, now shifted) */
		rc = mlx_max_axis(&result, mp_tmp, 3, false, stream);
		mlx_array_free(mp_tmp);
		break;
	}

	case SAM3_OP_GROUPNORM: {
		/*
		 * GroupNorm(num_groups) on NCHW input.
		 * inputs[0]=x[N,C,H,W], inputs[1]=gamma[C], inputs[2]=beta[C].
		 * params[0]=num_groups.
		 *
		 * Strategy: reshape [N,C,H,W] -> [N,G,C/G,H*W], normalize
		 * over last 2 dims via layer_norm (no affine), reshape back,
		 * then apply per-channel gamma/beta.
		 */
		int gn_groups = node->params[0];
		const struct sam3_tensor *gn_in = node->inputs[0];
		int gn_N = gn_in->dims[0];
		int gn_C = gn_in->dims[1];
		int gn_H = gn_in->dims[2];
		int gn_W = gn_in->dims[3];
		int gn_cpg = gn_C / gn_groups;
		int gn_hw = gn_H * gn_W;

		/* Reshape to [N, G, C/G * H * W] for layer_norm over last dim */
		int gn_shape3[] = {gn_N, gn_groups, gn_cpg * gn_hw};
		mlx_array gn_x3 = mlx_array_new();
		rc = mlx_reshape(&gn_x3, inputs[0], gn_shape3, 3, stream);
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

		/* Apply per-channel gamma and beta: x * gamma[1,C,1,1] + beta[1,C,1,1] */
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

		result = gn_normed4;
		rc = 0;
		break;
	}

	case SAM3_OP_BIAS_ADD: {
		/*
		 * NCHW bias add: x[N,C,H,W] + bias[C].
		 * Reshape bias to [1,C,1,1] for MLX broadcast.
		 */
		int ba_C = node->inputs[1]->dims[0];
		int ba_shape[] = {1, ba_C, 1, 1};
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

	mtl->use_f16 = true;

	mtl->device = mlx_device_new_type(MLX_GPU, 0);
	mtl->stream = mlx_default_gpu_stream_new();
	metal_map_init(mtl);

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

	metal_map_free(mtl);
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

	/* Phase 1: translate all nodes to MLX lazy ops */
	for (int i = 0; i < g->n_nodes; i++) {
		err = metal_dispatch_node(mtl, &g->nodes[i]);
		if (err != SAM3_OK) {
			sam3_log_error("metal_graph_eval: node %d failed", i);
			return err;
		}
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
					if (imap[s].key == inp) {
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

	/* Phase 3: copy only final results back to host tensors */
	{
		int n_copied = 0, n_skip_inter = 0, n_skip_nodata = 0;
		for (int i = 0; i < g->n_nodes; i++) {
			struct sam3_tensor *out_t = g->nodes[i].output;
			mlx_array *out_a = metal_map_get(mtl, out_t);
			if (!out_a || !out_t->data) {
				n_skip_nodata++;
				goto evict;
			}

			if (is_intermediate[i]) {
				n_skip_inter++;
				goto evict;
			}

			{
				/*
				 * MLX transpose/reshape create views with
				 * non-standard strides. Force row-major
				 * contiguous layout before reading data.
				 */
				mlx_array contig = mlx_array_new();
				int crc = mlx_contiguous(
					&contig, *out_a,
					false, /* allow_col_major */
					mtl->stream);
				if (crc) {
					mlx_array_free(contig);
					sam3_log_error("metal: contiguous failed");
					break;
				}
				{
					mlx_vector_array va =
						mlx_vector_array_new();
					mlx_vector_array_append_value(va,
								       contig);
					mlx_eval(va);
					mlx_vector_array_free(va);
				}

				const void *src = NULL;
				mlx_dtype mtype = mlx_array_dtype(contig);

				/*
				 * If Metal computed in F16 but SAM3 tensor
				 * expects F32, cast back for host readback.
				 */
				if (mtype == MLX_FLOAT16
				    && out_t->dtype == SAM3_DTYPE_F32) {
					mlx_array f32 = mlx_array_new();
					int cast_rc = mlx_astype(
						&f32, contig, MLX_FLOAT32,
						mtl->stream);
					if (cast_rc == 0) {
						mlx_array_free(contig);
						contig = f32;
						mtype = MLX_FLOAT32;
						/* Eval the F16->F32 cast */
						mlx_vector_array va2 =
							mlx_vector_array_new();
						mlx_vector_array_append_value(
							va2, contig);
						mlx_eval(va2);
						mlx_vector_array_free(va2);
					} else {
						mlx_array_free(f32);
					}
				}

				switch (mtype) {
				case MLX_FLOAT32:
					src = mlx_array_data_float32(contig);
					break;
				case MLX_FLOAT16:
					src = mlx_array_data_float16(contig);
					break;
				case MLX_BFLOAT16:
					src = mlx_array_data_bfloat16(contig);
					break;
				case MLX_INT32:
					src = mlx_array_data_int32(contig);
					break;
				case MLX_INT8:
					src = mlx_array_data_int8(contig);
					break;
				default:
					sam3_log_error("metal: unsupported output dtype");
					break;
				}

				if (src) {
					memcpy(out_t->data, src, out_t->nbytes);
					n_copied++;
				}
				mlx_array_free(contig);
			}

		evict:
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
