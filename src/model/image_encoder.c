/*
 * src/model/image_encoder.c - ViT image encoder graph construction
 *
 * Implements initialization, weight loading, and compute graph
 * construction for the Vision Transformer (ViT) backbone used in
 * SAM3. The encoder takes a normalized RGB image, applies patch
 * embedding via conv2d, adds tiled absolute positional embeddings,
 * applies ln_pre, then runs through 32 transformer blocks with
 * multi-head self-attention (using 2D axial RoPE) and GELU MLP,
 * producing per-patch feature embeddings.
 *
 * Window blocks (28 of 32) use a small 576-position local RoPE
 * (24x24, scale 1.0) and unmasked SDPA on partitioned
 * [n_windows, ws*ws, e] tensors. Global blocks (7, 15, 23, 31) use
 * the 5184-position RoPE (72x72, scale 1/3) with full attention.
 *
 * Key types:  sam3_vit
 * Depends on: image_encoder.h, graph_helpers.h
 * Used by:    sam3.c (top-level API)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "image_encoder.h"
#include "graph_helpers.h"
#include "util/log.h"
#include "util/profile.h"

/* Global attention block indices */
static const int global_blocks[] = {7, 15, 23, 31};

/*
 * precompute_rope_table - Precompute 2D axial RoPE cosine and sine tables.
 *
 * Matches Python compute_axial_cis():
 *   freqs = 1/(theta^(arange(0, dim, 4)[:dim//4] / dim))
 *   For each position (y, x) with optional scale:
 *     first dim//4 entries: y * scale * freq
 *     last  dim//4 entries: x * scale * freq
 *
 * @arena:     Arena to allocate from
 * @n_pos:     Number of positions (grid_w * grid_h)
 * @grid_w:    Grid width
 * @head_dim:  Per-head dimension (embed_dim / n_heads)
 * @scale:     Position scale factor (1.0 for window-local, <1 for global)
 * @theta:     RoPE base frequency (10000.0)
 * @out_cos:   Output cosine table [n_pos, head_dim/2]
 * @out_sin:   Output sine table [n_pos, head_dim/2]
 */
static enum sam3_error precompute_rope_table(
	struct sam3_arena *arena,
	int n_pos, int grid_w, int head_dim,
	float scale, float theta,
	struct sam3_tensor **out_cos,
	struct sam3_tensor **out_sin)
{
	int half = head_dim / 2;	/* 32 */
	int quarter = head_dim / 4;	/* 16 frequencies per axis */

	int dims[] = {n_pos, half};
	*out_cos = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	if (!*out_cos)
		return SAM3_ENOMEM;

	*out_sin = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	if (!*out_sin)
		return SAM3_ENOMEM;

	float *cos_data = (float *)(*out_cos)->data;
	float *sin_data = (float *)(*out_sin)->data;

	/*
	 * Precompute frequency table matching Python:
	 *   freqs = 1/(theta^(arange(0, dim, 4)[:dim//4] / dim))
	 * where dim = head_dim = 64, giving 16 frequencies.
	 */
	float freqs[16];
	for (int i = 0; i < quarter; i++)
		freqs[i] = 1.0f / powf(theta,
				(float)(i * 4) / (float)head_dim);

	int grid_h = n_pos / grid_w;
	for (int py = 0; py < grid_h; py++) {
		for (int px = 0; px < grid_w; px++) {
			int pos = py * grid_w + px;
			float sy = (float)py * scale;
			float sx = (float)px * scale;

			/*
			 * First quarter: x-axis (column) frequencies.
			 * Python compute_axial_cis concatenates
			 * [freqs_cis_x, freqs_cis_y], so x (column)
			 * goes first, matching init_t_xy's t_x = t % end_x.
			 */
			for (int i = 0; i < quarter; i++) {
				float angle = sx * freqs[i];
				cos_data[pos * half + i] = cosf(angle);
				sin_data[pos * half + i] = sinf(angle);
			}
			/* Second quarter: y-axis (row) frequencies */
			for (int i = 0; i < quarter; i++) {
				float angle = sy * freqs[i];
				cos_data[pos * half + quarter + i] = cosf(angle);
				sin_data[pos * half + quarter + i] = sinf(angle);
			}
		}
	}

	return SAM3_OK;
}

enum sam3_error sam3_vit_init(struct sam3_vit *vit,
			       int img_size, int patch_size,
			       int embed_dim, int depth, int n_heads,
			       int window_size, int mlp_dim,
			       struct sam3_arena *arena)
{
	memset(vit, 0, sizeof(*vit));

	vit->img_size = img_size;
	vit->patch_size = patch_size;
	vit->embed_dim = embed_dim;
	vit->depth = depth;
	vit->n_heads = n_heads;
	vit->window_size = window_size;
	vit->mlp_dim = mlp_dim;
	vit->grid_size = img_size / patch_size;
	vit->n_patches = vit->grid_size * vit->grid_size;

	/* Mark global attention blocks */
	for (int i = 0; i < depth; i++) {
		vit->layers[i].is_global = 0;
		for (int j = 0; j < SAM3_VIT_N_GLOBAL_BLOCKS; j++) {
			if (i == global_blocks[j]) {
				vit->layers[i].is_global = 1;
				break;
			}
		}
	}

	vit->model_arena = arena;

	return SAM3_OK;
}

/*
 * Weight name prefix for ViT backbone weights in the .sam3 file.
 * Original PyTorch: detector_model.vision_encoder.backbone.*
 */
#define VIT_P "detector_model.vision_encoder.backbone."

/*
 * tile_pos_embed - Tile pretrain positional embedding to inference resolution.
 *
 * The checkpoint stores pos_embed as [1, num_pretrain_patches + 1, embed_dim]
 * where the +1 is a CLS token. We strip the CLS token, reshape from
 * pretrain grid (24x24) to [24, 24, C], tile 3x3 to [72, 72, C], and
 * flatten to [5184, embed_dim].
 *
 * Matches Python get_abs_pos() with tiling=True.
 */
static struct sam3_tensor *tile_pos_embed(
	struct sam3_arena *arena,
	const struct sam3_tensor *raw_pos,
	int pretrain_grid, int target_grid, int embed_dim)
{
	/*
	 * raw_pos is [1, pretrain_patches+1, embed_dim] or
	 * [pretrain_patches+1, embed_dim]. Strip CLS (first row).
	 */
	int n_pretrain = pretrain_grid * pretrain_grid;
	int n_target = target_grid * target_grid;
	int tile_factor = target_grid / pretrain_grid;
	size_t row_bytes = (size_t)embed_dim * sizeof(float);

	int out_dims[] = {n_target, embed_dim};
	struct sam3_tensor *out = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						  2, out_dims);
	if (!out)
		return NULL;

	/*
	 * Source data starts after CLS token (row 0).
	 * raw_pos may be [1, N+1, C] (3D) or [N+1, C] (2D).
	 */
	const float *src;
	if (raw_pos->n_dims == 3)
		src = (const float *)raw_pos->data + embed_dim;
	else
		src = (const float *)raw_pos->data + embed_dim;

	float *dst = (float *)out->data;

	/*
	 * Tile: for each target position (ty, tx), find source
	 * position (ty % pretrain_grid, tx % pretrain_grid).
	 */
	for (int ty = 0; ty < target_grid; ty++) {
		int sy = ty % pretrain_grid;
		for (int tx = 0; tx < target_grid; tx++) {
			int sx = tx % pretrain_grid;
			int src_idx = sy * pretrain_grid + sx;
			int dst_idx = ty * target_grid + tx;
			memcpy(dst + dst_idx * embed_dim,
			       src + src_idx * embed_dim,
			       row_bytes);
		}
	}

	(void)tile_factor;
	(void)n_pretrain;
	return out;
}

/*
 * fuse_3 - Load 3 separate [d, d_in] weights and fuse into [3*d, d_in].
 */
static struct sam3_tensor *fuse_3(const struct sam3_weight_file *wf,
				   const char *name_a,
				   const char *name_b,
				   const char *name_c,
				   struct sam3_arena *arena,
				   int d, int n_dims, const int *part_dims)
{
	struct sam3_tensor *a, *b, *c, *out;
	int fused_dims[2];

	a = gh_load_mmap(wf, name_a, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	b = gh_load_mmap(wf, name_b, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	c = gh_load_mmap(wf, name_c, arena, SAM3_DTYPE_F32,
			      n_dims, part_dims);
	if (!a || !b || !c)
		return NULL;

	if (n_dims == 2) {
		fused_dims[0] = 3 * d;
		fused_dims[1] = part_dims[1];
	} else {
		fused_dims[0] = 3 * d;
	}

	out = gh_alloc_tensor(arena, SAM3_DTYPE_F32, n_dims, fused_dims);
	if (!out)
		return NULL;

	memcpy(out->data, a->data, a->nbytes);
	memcpy((char *)out->data + a->nbytes, b->data, b->nbytes);
	memcpy((char *)out->data + a->nbytes + b->nbytes,
	       c->data, c->nbytes);

	return out;
}

enum sam3_error sam3_vit_load(struct sam3_vit *vit,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	int e = vit->embed_dim;
	int m = vit->mlp_dim;
	int ps = vit->patch_size;
	char name[128];

	/*
	 * Patch embedding Conv2d weight. sam3_convert permutes the
	 * checkpoint OIHW [embed_dim, 3, ps, ps] tensor to OHWI
	 * [embed_dim, ps, ps, 3] before writing, so the load path
	 * maps it directly for the NHWC conv dispatch.
	 */
	int pe_w_dims[] = {e, ps, ps, 3};
	vit->patch_embed_w = gh_load_mmap(wf,
		VIT_P "embeddings.patch_embeddings.projection.weight",
		arena, SAM3_DTYPE_F32, 4, pe_w_dims);
	if (!vit->patch_embed_w)
		return SAM3_ENOMEM;

	/* Patch embedding bias [embed_dim] — optional; absent when the
	 * upstream ViT was built with bias_patch_embed=False (silent
	 * zero-fill rather than a noisy warning). */
	int pe_b_dims[] = {e};
	vit->patch_embed_b = gh_load_mmap_optional(wf,
		VIT_P "embeddings.patch_embeddings.projection.bias",
		arena, SAM3_DTYPE_F32, 1, pe_b_dims);
	if (!vit->patch_embed_b) {
		vit->patch_embed_b = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						     1, pe_b_dims);
		if (!vit->patch_embed_b)
			return SAM3_ENOMEM;
	}

	/*
	 * Absolute positional embedding.
	 * Checkpoint has [1, 577, 1024] (576 patches + CLS, from pretrain
	 * 336/14=24 grid). We tile to 72x72 target grid.
	 */
	int pretrain_grid = 24; /* pretrain_img_size=336, patch_size=14 */
	int n_pretrain_with_cls = pretrain_grid * pretrain_grid + 1;
	int pos_dims[] = {1, n_pretrain_with_cls, e};
	struct sam3_tensor *raw_pos = gh_load_mmap(wf,
		VIT_P "embeddings.position_embedding.weight",
		arena, SAM3_DTYPE_F32, 3, pos_dims);
	if (!raw_pos)
		return SAM3_ENOMEM;

	vit->raw_pos_embed = raw_pos;

	/* ln_pre (pre-block layer norm) */
	int e_dims[] = {e};
	vit->ln_pre_w = gh_load_mmap(wf,
		VIT_P "layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, e_dims);
	if (!vit->ln_pre_w)
		return SAM3_ENOMEM;

	vit->ln_pre_b = gh_load_mmap(wf,
		VIT_P "layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, e_dims);
	if (!vit->ln_pre_b)
		return SAM3_ENOMEM;

	/* Per-layer weights */
	int single_w_dims[] = {e, e};
	int single_b_dims[] = {e};
	int proj_w_dims[] = {e, e};
	int fc1_w_dims[] = {m, e};
	int fc1_b_dims[] = {m};
	int fc2_w_dims[] = {e, m};

	for (int i = 0; i < vit->depth; i++) {
		/* Layer norm 1 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm1.weight", i);
		vit->layers[i].ln1_w = gh_load_mmap(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm1.bias", i);
		vit->layers[i].ln1_b = gh_load_mmap(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln1_b)
			return SAM3_ENOMEM;

		/* Attention QKV: fuse Q/K/V into [3*e, e] / [3*e] */
		{
			char q_name[128], k_name[128], v_name[128];
			snprintf(q_name, sizeof(q_name),
				 VIT_P "layers.%d.attention.q_proj.weight", i);
			snprintf(k_name, sizeof(k_name),
				 VIT_P "layers.%d.attention.k_proj.weight", i);
			snprintf(v_name, sizeof(v_name),
				 VIT_P "layers.%d.attention.v_proj.weight", i);
			vit->layers[i].qkv_w = fuse_3(wf, q_name, k_name,
						       v_name, arena, e,
						       2, single_w_dims);
			if (!vit->layers[i].qkv_w)
				return SAM3_ENOMEM;

			snprintf(q_name, sizeof(q_name),
				 VIT_P "layers.%d.attention.q_proj.bias", i);
			snprintf(k_name, sizeof(k_name),
				 VIT_P "layers.%d.attention.k_proj.bias", i);
			snprintf(v_name, sizeof(v_name),
				 VIT_P "layers.%d.attention.v_proj.bias", i);
			vit->layers[i].qkv_b = fuse_3(wf, q_name, k_name,
						       v_name, arena, e,
						       1, single_b_dims);
			if (!vit->layers[i].qkv_b)
				return SAM3_ENOMEM;
		}

		/* Attention output projection */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.attention.o_proj.weight", i);
		vit->layers[i].proj_w = gh_load_mmap(wf, name, arena,
						       SAM3_DTYPE_F32,
						       2, proj_w_dims);
		if (!vit->layers[i].proj_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.attention.o_proj.bias", i);
		vit->layers[i].proj_b = gh_load_mmap(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, e_dims);
		if (!vit->layers[i].proj_b)
			return SAM3_ENOMEM;

		/* Layer norm 2 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm2.weight", i);
		vit->layers[i].ln2_w = gh_load_mmap(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.layer_norm2.bias", i);
		vit->layers[i].ln2_b = gh_load_mmap(wf, name, arena,
						      SAM3_DTYPE_F32,
						      1, e_dims);
		if (!vit->layers[i].ln2_b)
			return SAM3_ENOMEM;

		/* MLP fc1 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc1.weight", i);
		vit->layers[i].mlp_fc1_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc1_w_dims);
		if (!vit->layers[i].mlp_fc1_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc1.bias", i);
		vit->layers[i].mlp_fc1_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
		if (!vit->layers[i].mlp_fc1_b)
			return SAM3_ENOMEM;

		/* MLP fc2 */
		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc2.weight", i);
		vit->layers[i].mlp_fc2_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc2_w_dims);
		if (!vit->layers[i].mlp_fc2_w)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 VIT_P "layers.%d.mlp.fc2.bias", i);
		vit->layers[i].mlp_fc2_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, e_dims);
		if (!vit->layers[i].mlp_fc2_b)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

/*
 * vit_lazy_precompute - Compute RoPE tables and pos embed tiling.
 *
 * Deferred from init/load to first sam3_vit_build() call so that
 * model loading does not pay the cost of these CPU-bound computations.
 */
static enum sam3_error vit_lazy_precompute(struct sam3_vit *vit)
{
	struct sam3_arena *arena = vit->model_arena;
	int head_dim = vit->embed_dim / vit->n_heads;
	float theta = 10000.0f;
	enum sam3_error err;

	/* Global RoPE */
	float global_scale = (float)vit->window_size /
			     (float)vit->grid_size;
	err = precompute_rope_table(arena, vit->n_patches, vit->grid_size,
				    head_dim, global_scale, theta,
				    &vit->rope_glo_cos,
				    &vit->rope_glo_sin);
	if (err != SAM3_OK)
		return err;

	/* Window-local RoPE (used by mask-free windowed attention) */
	int ws = vit->window_size;
	err = precompute_rope_table(arena, ws * ws, ws, head_dim,
				    1.0f, theta,
				    &vit->rope_win_local_cos,
				    &vit->rope_win_local_sin);
	if (err != SAM3_OK)
		return err;

	/* Tile position embedding if raw is available */
	if (vit->raw_pos_embed && !vit->pos_embed) {
		vit->pos_embed = tile_pos_embed(arena,
						vit->raw_pos_embed,
						24, vit->grid_size,
						vit->embed_dim);
		if (!vit->pos_embed)
			return SAM3_ENOMEM;
		sam3_log_info("vit: pos_embed tiled to %dx%d (lazy)",
			      vit->grid_size, vit->grid_size);
	}

	vit->precomputed = 1;
	return SAM3_OK;
}

enum sam3_error sam3_vit_precompute(struct sam3_vit *vit)
{
	if (vit->precomputed)
		return SAM3_OK;
	return vit_lazy_precompute(vit);
}

struct sam3_tensor *sam3_vit_build(struct sam3_vit *vit,
				    struct sam3_backend *be,
				    struct sam3_tensor *image,
				    struct sam3_arena *scratch,
				    struct sam3_arena *persist,
				    struct sam3_profiler *profiler)
{
	int gs = vit->grid_size;
	int e = vit->embed_dim;
	int np = vit->n_patches;
	size_t x_bytes = (size_t)np * e * sam3_dtype_size(SAM3_DTYPE_F32);
	struct sam3_graph g;
	enum sam3_error err;

	/* Lazy-init RoPE tables and tiled pos_embed on first build */
	if (!vit->precomputed) {
		SAM3_PROF_BEGIN(profiler, "vit_precompute");
		err = vit_lazy_precompute(vit);
		SAM3_PROF_END(profiler, "vit_precompute");
		if (err != SAM3_OK)
			return NULL;
	}

	/*
	 * Allocate persistent buffer for the block output that
	 * survives arena resets between blocks.
	 */
	void *x_buf = sam3_arena_alloc(persist, x_bytes);
	if (!x_buf)
		return NULL;

	/*
	 * Step 1: Patch embedding + pos_embed + ln_pre.
	 *
	 * Build a single graph: conv2d (NHWC) -> reshape -> add bias
	 * -> add pos_embed -> layernorm. The conv input is reshaped
	 * from [3, img, img] to NHWC [1, img, img, 3] and the weight
	 * was permuted to OHWI at load time (see sam3_vit_load). The
	 * conv output [1, gs, gs, e] row-major reshapes directly to
	 * [np, e] without the transpose the NCHW path used to need.
	 */
	SAM3_PROF_BEGIN(profiler, "vit_patch_embed");
	sam3_graph_init(&g);

	/*
	 * The caller delivers the image as planar CHW [3, img, img]
	 * (see sam3_normalize_rgb_chw). We reshape to NCHW
	 * [1, 3, img, img] and then permute to NHWC [1, img, img, 3]
	 * so the NHWC conv can consume it directly. The permute runs
	 * as a graph op — it is a real data movement, not a view.
	 */
	int img_nchw_dims[] = {1, 3, vit->img_size, vit->img_size};
	struct sam3_tensor *image_nchw;
	image_nchw = gh_reshape(&g, scratch, image, 4, img_nchw_dims);
	if (!image_nchw)
		return NULL;

	int chw_to_hwc[] = {0, 2, 3, 1};
	struct sam3_tensor *image_nhwc;
	image_nhwc = gh_permute(&g, scratch, image_nchw, chw_to_hwc);
	if (!image_nhwc)
		return NULL;

	struct sam3_tensor *conv_out;
	conv_out = gh_conv2d(&g, scratch, image_nhwc,
				  vit->patch_embed_w, NULL,
				  vit->patch_size, 0, 1);
	if (!conv_out)
		return NULL;

	/*
	 * conv_out is NHWC [1, gs, gs, e]. Row-major, this is exactly
	 * [gs*gs, e] = [np, e] — the element order in memory already
	 * has the embedding dim as the last, fastest-varying axis. A
	 * single reshape replaces the old reshape+transpose pair.
	 */
	int flat_dims[] = {np, e};
	struct sam3_tensor *x;
	x = gh_reshape(&g, scratch, conv_out, 2, flat_dims);
	if (!x)
		return NULL;

	x = gh_add(&g, scratch, x, vit->patch_embed_b);
	if (!x)
		return NULL;

	x = gh_add(&g, scratch, x, vit->pos_embed);
	if (!x)
		return NULL;

	x = gh_layernorm(&g, scratch, x, vit->ln_pre_w, vit->ln_pre_b);
	if (!x)
		return NULL;

	err = be->ops->graph_eval(be, &g);
	if (err != SAM3_OK)
		return NULL;

	/* Copy result to persistent buffer */
	memcpy(x_buf, x->data, x_bytes);
#ifdef SAM3_DEBUG_DUMP
	{
		FILE *fp = fopen("/tmp/dbg_vit_patch.bin", "wb");
		if (fp) {
			int n_elt = 1;
			for (int i = 0; i < x->n_dims; i++)
				n_elt *= x->dims[i];
			fwrite(x->data, sizeof(float), (size_t)n_elt, fp);
			fclose(fp);
			sam3_log_info("dump: wrote /tmp/dbg_vit_patch.bin "
				      "(%d floats)", n_elt);
		}
	}
#endif

	sam3_log_info("vit: patch embedding + pos_embed + ln_pre evaluated "
		      "(%d patches)", np);
	sam3_log_info("vit: scratch arena: %zu / %zu bytes used",
		       scratch->offset, scratch->size);
	SAM3_PROF_END(profiler, "vit_patch_embed");

	/*
	 * Step 2: Per-block transformer evaluation.
	 *
	 * With skip_data=1 on Metal, intermediate tensors get no host
	 * buffer (MLX manages GPU memory), so arena usage is ~160 KB
	 * per batch instead of ~1.2 GB per block. This allows larger
	 * batches (fewer GPU dispatches) without exhausting the arena.
	 *
	 * CPU batch size depends on per-block scratch cost: each block
	 * allocates ~(np * mlp_dim * 4) bytes for the MLP expansion
	 * plus smaller attention tensors. Hiera (5184×4736) needs
	 * ~294 MiB/block → batch=2 fits in 1 GiB scratch. Smaller
	 * backbones (EfficientViT, TinyViT) use batch=4.
	 */
	SAM3_PROF_BEGIN(profiler, "vit_blocks");
	{
		int skip_data = (be->type == SAM3_BACKEND_METAL);
		size_t block_bytes = (size_t)np * vit->mlp_dim * 4 * 3;
		int batch = (skip_data || block_bytes < 256UL * 1024 * 1024)
			  ? 4 : 2;
#ifdef SAM3_DEBUG_DUMP
		/* Force batch=1 + host-side readback so we can dump after
		 * every ViT block. Slow but diagnostic. */
		batch = 1;
		skip_data = 0;
#endif

		/*
		 * GPU-resident forwarding tensor. Allocated in
		 * persist arena so it survives scratch resets.
		 * On non-last batches, the output mlx_array stays
		 * in the tensor map; the next batch finds it via
		 * metal_wrap_tensor without data transfer.
		 */
		struct sam3_tensor *x_fwd = NULL;
		if (skip_data) {
			int fwd_dims[] = {np, e};
			persist->skip_data = 1;
			x_fwd = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						 2, fwd_dims);
			persist->skip_data = 0;
			if (!x_fwd)
				return NULL;
			x_fwd->data = x_buf;
		}

		for (int base = 0; base < vit->depth; base += batch) {
			int end = base + batch;
			if (end > vit->depth)
				end = vit->depth;

			sam3_graph_init(&g);
			sam3_arena_reset(scratch);
			if (skip_data)
				scratch->skip_data = 1;

			if (skip_data && base > 0) {
				/* GPU-resident: x_fwd in tensor map */
				x = x_fwd;
			} else {
				int x_dims[] = {np, e};
				x = gh_tensor_wrap(scratch,
						    SAM3_DTYPE_F32,
						    2, x_dims, x_buf);
				if (!x)
					return NULL;
			}

			for (int i = base; i < end; i++) {
				/* Pre-norm for attention */
				struct sam3_tensor *x_norm;
				x_norm = gh_layernorm(&g, scratch, x,
						       vit->layers[i].ln1_w,
						       vit->layers[i].ln1_b);
				if (!x_norm)
					return NULL;

				/* Reshape to 3D for MHA */
				int attn_dims[] = {1, np, e};
				struct sam3_tensor *x3d;
				x3d = gh_reshape(&g, scratch, x_norm,
						  3, attn_dims);
				if (!x3d)
					return NULL;

				/* Self-attention with RoPE */
				int is_global = vit->layers[i].is_global;
				struct sam3_tensor *attn;

				if (is_global) {
					attn = gh_multihead_attention_rope(
						&g, scratch,
						x3d, NULL, NULL,
						vit->layers[i].qkv_w,
						vit->layers[i].qkv_b,
						vit->layers[i].proj_w,
						vit->layers[i].proj_b,
						vit->n_heads,
						vit->rope_glo_cos,
						vit->rope_glo_sin,
						NULL,
						vit->grid_size,
						(float)vit->window_size /
						(float)vit->grid_size);
					if (!attn) {
						sam3_log_error("vit: "
							"block %d "
							"attention OOM "
							"(scratch %zu / "
							"%zu)",
							i, scratch->offset,
							scratch->size);
						return NULL;
					}
				} else {
					struct sam3_tensor *x_win;
					x_win = gh_window_partition(
						&g, scratch, x_norm,
						vit->window_size,
						vit->grid_size);
					if (!x_win)
						return NULL;

					struct sam3_tensor *attn_win;
					attn_win = gh_multihead_attention_rope(
						&g, scratch,
						x_win, NULL, NULL,
						vit->layers[i].qkv_w,
						vit->layers[i].qkv_b,
						vit->layers[i].proj_w,
						vit->layers[i].proj_b,
						vit->n_heads,
						vit->rope_win_local_cos,
						vit->rope_win_local_sin,
						NULL,
						vit->window_size,
						1.0f);
					if (!attn_win) {
						sam3_log_error("vit: "
							"block %d "
							"windowed mha OOM "
							"(scratch %zu / "
							"%zu)",
							i, scratch->offset,
							scratch->size);
						return NULL;
					}

					int nw = vit->grid_size /
						 vit->window_size;
					int win_3d[] = {
						nw * nw,
						vit->window_size *
						vit->window_size,
						e
					};
					attn_win = gh_reshape(&g, scratch,
							      attn_win, 3,
							      win_3d);
					if (!attn_win)
						return NULL;

					attn = gh_window_unpartition(
						&g, scratch, attn_win,
						vit->window_size,
						vit->grid_size);
					if (!attn) {
						sam3_log_error("vit: "
							"block %d "
							"unpartition fail",
							i);
						return NULL;
					}
				}

				/* Residual: x + attn */
				x = gh_add(&g, scratch, x, attn);
				if (!x)
					return NULL;

				/* Pre-norm for MLP */
				x_norm = gh_layernorm(&g, scratch, x,
						       vit->layers[i].ln2_w,
						       vit->layers[i].ln2_b);
				if (!x_norm)
					return NULL;

				/* MLP: fc1 -> GELU -> fc2 */
				struct sam3_tensor *ff;
				ff = gh_mlp(&g, scratch, x_norm,
					     vit->layers[i].mlp_fc1_w,
					     vit->layers[i].mlp_fc1_b,
					     vit->layers[i].mlp_fc2_w,
					     vit->layers[i].mlp_fc2_b,
					     SAM3_OP_GELU);
				if (!ff) {
					sam3_log_error("vit: block %d "
						"MLP OOM "
						"(scratch %zu / %zu)",
						i, scratch->offset,
						scratch->size);
					return NULL;
				}

				/* Residual: x + ff */
				x = gh_add(&g, scratch, x, ff);
				if (!x)
					return NULL;

				sam3_log_debug("vit: block %d/%d built "
					"(arena %zu/%zu)",
					i + 1, vit->depth,
					scratch->offset, scratch->size);
			}

			if (skip_data) {
				bool last = (end >= vit->depth);
				/*
				 * Redirect last node output to the
				 * persistent forwarding tensor so its
				 * mlx_array key survives scratch reset.
				 */
				g.nodes[g.n_nodes - 1].output = x_fwd;
				x = x_fwd;
				g.no_readback = !last;
			}

			err = be->ops->graph_eval(be, &g);
			scratch->skip_data = 0;
			if (err != SAM3_OK)
				return NULL;

			if (!skip_data)
				memcpy(x_buf, x->data, x_bytes);
#ifdef SAM3_DEBUG_DUMP
			/* Dump selected blocks for Python parity diff.
			 * `end-1` is the block index we just finished. */
			{
				int done = end - 1;
				if (done == 0 || done == 3 || done == 7 ||
				    done == 11 || done == 13 || done == 14 ||
				    done == 15 || done == 16 || done == 17 ||
				    done == 19 || done == 23 || done == 27 ||
				    done == 31) {
					char p[128];
					snprintf(p, sizeof(p),
						 "/tmp/dbg_vit_block%02d.bin",
						 done);
					FILE *fp = fopen(p, "wb");
					if (fp) {
						int n = np * e;
						fwrite(x_buf, sizeof(float),
						       (size_t)n, fp);
						fclose(fp);
						sam3_log_info("dump: wrote %s "
							      "(%d floats)",
							      p, n);
					}
				}
			}
#endif
		}
	}
	SAM3_PROF_END(profiler, "vit_blocks");

	sam3_log_info("vit: all %d blocks evaluated (no post-norm, "
		      "ln_post=Identity)", vit->depth);

	/*
	 * No final layer norm — Python ViT has ln_post=False (Identity).
	 * The only ViT-level norm is ln_pre, applied before blocks above.
	 */

	/* Return a tensor in persist arena wrapping the final output */
	int out_dims[] = {np, e};
	return gh_tensor_wrap(persist, SAM3_DTYPE_F32, 2, out_dims, x_buf);
}
