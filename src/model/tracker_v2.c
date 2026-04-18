/*
 * src/model/tracker_v2.c - SAM 3.1 multiplex tracker loader (phase 2.1)
 *
 * Implements sam3_tracker_v2_init and sam3_tracker_v2_load. This commit
 * only handles the "small" sub-modules: maskmem backbone (38 tensors),
 * object-pointer MLPs and linear layers (12 tensors), and the singleton
 * embeddings (6 tensors) — 56 total. The memory-attention transformer
 * and mask decoders land in phases 2.3-2.5.
 *
 * Key types:  sam3_tracker_v2
 * Depends on: tracker_v2.h, graph_helpers.h, util/log.h
 * Used by:    sam3_video.c (phase 2.5)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "tracker_v2.h"
#include "graph_helpers.h"
#include "util/log.h"

/*
 * Helper: load a Conv2d weight from OHWI mmap with an explicit
 * [O, H, W, I] shape and abort cleanly on any failure.
 */
static enum sam3_error
load_tensor_req(struct sam3_tensor **out,
		const struct sam3_weight_file *wf,
		const char *name, struct sam3_arena *arena,
		enum sam3_dtype dtype, int n_dims, const int *dims)
{
	*out = gh_load_mmap_optional(wf, name, arena, dtype, n_dims, dims);
	if (!*out) {
		sam3_log_error("tracker_v2: required tensor absent: %s", name);
		return SAM3_EMODEL;
	}
	return SAM3_OK;
}

#define LOAD(dst, dtype_, ndims_, ...) \
	do { \
		int _dims[] = { __VA_ARGS__ }; \
		enum sam3_error _e = load_tensor_req(&(dst), wf, _name, \
				arena, (dtype_), (ndims_), _dims); \
		if (_e != SAM3_OK) return _e; \
	} while (0)

/* ── Sub-module loaders ─────────────────────────────────────────────── */

static enum sam3_error load_mask_downsampler(
		struct sam3_v2_mask_downsampler *ms,
		const struct sam3_weight_file *wf,
		struct sam3_arena *arena)
{
	/*
	 * encoder.{0,3,6,9}  — 4 Conv2d stages (k=3, s=2, p=1)
	 * encoder.{1,4,7,10} — 4 LayerNorm2d
	 * encoder.12         — final 1x1 projection
	 *
	 * conv_perm in the converter writes conv weights in OHWI order.
	 * Channel progression: 32 -> 16 -> 64 -> 256 -> 1024 -> 256.
	 */
	static const int chans[5] = {32, 16, 64, 256, 1024};
	char buf[SAM3_WEIGHT_NAME_MAX];
	for (int s = 0; s < 4; s++) {
		int conv_idx = s * 3;        /* 0, 3, 6, 9 */
		int norm_idx = conv_idx + 1; /* 1, 4, 7, 10 */
		int out_c = chans[s + 1];
		int in_c  = chans[s];

		snprintf(buf, sizeof(buf),
			 "tracker_v2.maskmem_backbone."
			 "mask_downsampler.encoder.%d.weight", conv_idx);
		{
			const char *_name = buf;
			LOAD(ms->conv_w[s], SAM3_DTYPE_F32, 4,
			     out_c, 3, 3, in_c);
		}
		snprintf(buf, sizeof(buf),
			 "tracker_v2.maskmem_backbone."
			 "mask_downsampler.encoder.%d.bias", conv_idx);
		{
			const char *_name = buf;
			LOAD(ms->conv_b[s], SAM3_DTYPE_F32, 1, out_c);
		}
		snprintf(buf, sizeof(buf),
			 "tracker_v2.maskmem_backbone."
			 "mask_downsampler.encoder.%d.weight", norm_idx);
		{
			const char *_name = buf;
			LOAD(ms->norm_w[s], SAM3_DTYPE_F32, 1, out_c);
		}
		snprintf(buf, sizeof(buf),
			 "tracker_v2.maskmem_backbone."
			 "mask_downsampler.encoder.%d.bias", norm_idx);
		{
			const char *_name = buf;
			LOAD(ms->norm_b[s], SAM3_DTYPE_F32, 1, out_c);
		}
	}

	{
		const char *_name =
			"tracker_v2.maskmem_backbone."
			"mask_downsampler.encoder.12.weight";
		LOAD(ms->proj_w, SAM3_DTYPE_F32, 4, 256, 1, 1, 1024);
	}
	{
		const char *_name =
			"tracker_v2.maskmem_backbone."
			"mask_downsampler.encoder.12.bias";
		LOAD(ms->proj_b, SAM3_DTYPE_F32, 1, 256);
	}
	return SAM3_OK;
}

static enum sam3_error load_cxblock(struct sam3_v2_cxblock *blk,
				    int layer_idx,
				    const struct sam3_weight_file *wf,
				    struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const int d = SAM3_V2_HIDDEN_DIM;
	const int d4 = d * 4;

#define LX(dst, field, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_v2.maskmem_backbone.fuser.layers.%d." field, \
			 layer_idx); \
		const char *_name = buf; \
		LOAD(dst, SAM3_DTYPE_F32, (ndims_), __VA_ARGS__); \
	} while (0)

	LX(blk->dwconv_w, "dwconv.weight", 4, d, 7, 7, 1);
	LX(blk->dwconv_b, "dwconv.bias",   1, d);
	LX(blk->norm_w,   "norm.weight",   1, d);
	LX(blk->norm_b,   "norm.bias",     1, d);
	LX(blk->pwconv1_w, "pwconv1.weight", 2, d4, d);
	LX(blk->pwconv1_b, "pwconv1.bias",   1, d4);
	LX(blk->pwconv2_w, "pwconv2.weight", 2, d, d4);
	LX(blk->pwconv2_b, "pwconv2.bias",   1, d);
	LX(blk->gamma,    "gamma",          1, d);
#undef LX
	return SAM3_OK;
}

static enum sam3_error load_maskmem(struct sam3_v2_maskmem *mm,
				    const struct sam3_weight_file *wf,
				    struct sam3_arena *arena)
{
	enum sam3_error err;
	err = load_mask_downsampler(&mm->mask_downsampler, wf, arena);
	if (err != SAM3_OK) return err;

	{
		const char *_name =
			"tracker_v2.maskmem_backbone.pix_feat_proj.weight";
		LOAD(mm->pix_feat_proj_w, SAM3_DTYPE_F32, 4, 256, 1, 1, 256);
	}
	{
		const char *_name =
			"tracker_v2.maskmem_backbone.pix_feat_proj.bias";
		LOAD(mm->pix_feat_proj_b, SAM3_DTYPE_F32, 1, 256);
	}

	for (int i = 0; i < 2; i++) {
		err = load_cxblock(&mm->fuser[i], i, wf, arena);
		if (err != SAM3_OK) return err;
	}
	return SAM3_OK;
}

static enum sam3_error load_mlp3(struct sam3_v2_mlp3 *mlp,
				 const char *prefix,
				 const struct sam3_weight_file *wf,
				 struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	for (int i = 0; i < 3; i++) {
		snprintf(buf, sizeof(buf), "%s.layers.%d.weight", prefix, i);
		{
			const char *_name = buf;
			LOAD(mlp->fc_w[i], SAM3_DTYPE_F32, 2, 256, 256);
		}
		snprintf(buf, sizeof(buf), "%s.layers.%d.bias", prefix, i);
		{
			const char *_name = buf;
			LOAD(mlp->fc_b[i], SAM3_DTYPE_F32, 1, 256);
		}
	}
	return SAM3_OK;
}

static enum sam3_error load_singletons(struct sam3_tracker_v2 *trk,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena)
{
	{
		const char *_name =
			"tracker_v2.image_pe_layer."
			"positional_encoding_gaussian_matrix";
		LOAD(trk->image_pe_gauss, SAM3_DTYPE_F32, 2, 2, 128);
	}
	{
		const char *_name = "tracker_v2.maskmem_tpos_enc";
		LOAD(trk->maskmem_tpos_enc, SAM3_DTYPE_F32, 4,
		     SAM3_V2_NUM_MASKMEM, 1, 1, SAM3_V2_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_v2.no_obj_embed_spatial";
		LOAD(trk->no_obj_embed_spatial, SAM3_DTYPE_F32, 2,
		     SAM3_V2_MULTIPLEX_COUNT, SAM3_V2_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_v2.output_valid_embed";
		LOAD(trk->output_valid_embed, SAM3_DTYPE_F32, 2,
		     SAM3_V2_MULTIPLEX_COUNT, SAM3_V2_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_v2.output_invalid_embed";
		LOAD(trk->output_invalid_embed, SAM3_DTYPE_F32, 2,
		     SAM3_V2_MULTIPLEX_COUNT, SAM3_V2_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_v2.interactivity_no_mem_embed";
		LOAD(trk->interactivity_no_mem_embed, SAM3_DTYPE_F32, 3,
		     1, 1, SAM3_V2_HIDDEN_DIM);
	}
	return SAM3_OK;
}

/* ── Public API ─────────────────────────────────────────────────────── */

enum sam3_error sam3_tracker_v2_init(struct sam3_tracker_v2 *trk)
{
	if (!trk)
		return SAM3_EINVAL;
	memset(trk, 0, sizeof(*trk));
	return SAM3_OK;
}

enum sam3_error sam3_tracker_v2_load(struct sam3_tracker_v2 *trk,
				     const struct sam3_weight_file *wf,
				     struct sam3_arena *arena)
{
	enum sam3_error err;

	if (!trk || !wf || !arena)
		return SAM3_EINVAL;

	err = load_maskmem(&trk->maskmem, wf, arena);
	if (err != SAM3_OK) return err;

	err = load_mlp3(&trk->obj_ptr_proj, "tracker_v2.obj_ptr_proj",
			wf, arena);
	if (err != SAM3_OK) return err;

	{
		const char *_name = "tracker_v2.obj_ptr_tpos_proj.weight";
		LOAD(trk->obj_ptr_tpos_proj_w, SAM3_DTYPE_F32, 2, 256, 256);
	}
	{
		const char *_name = "tracker_v2.obj_ptr_tpos_proj.bias";
		LOAD(trk->obj_ptr_tpos_proj_b, SAM3_DTYPE_F32, 1, 256);
	}
	{
		const char *_name = "tracker_v2.no_obj_ptr_linear.weight";
		LOAD(trk->no_obj_ptr_linear_w, SAM3_DTYPE_F32, 2, 256, 256);
	}
	{
		const char *_name = "tracker_v2.no_obj_ptr_linear.bias";
		LOAD(trk->no_obj_ptr_linear_b, SAM3_DTYPE_F32, 1, 256);
	}

	err = load_singletons(trk, wf, arena);
	if (err != SAM3_OK) return err;

	sam3_log_info("tracker_v2: loaded phase-2.1 weights "
		      "(maskmem + obj_ptr + singletons)");
	return SAM3_OK;
}

#undef LOAD
