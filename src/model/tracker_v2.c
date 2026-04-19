/*
 * src/model/tracker_v2.c - SAM 3.1 multiplex tracker (loader + forwards)
 *
 * Loader (sam3_tracker_v2_init / sam3_tracker_v2_load) populates the
 * tracker_v2 struct from a SAM 3.1 .sam3 file: maskmem backbone
 * (38 tensors), memory-attention transformer (122 tensors), SAM mask
 * decoder (125 tensors), obj_ptr MLPs + linears (12 tensors), and
 * singleton embeddings (6 tensors) — 303 tensors end-to-end.
 *
 * Forwards:
 *   - sam3_v2_maskmem_forward (phase 2.2): pix_feat + masks -> memory
 *     tokens via the multiplex-aware SimpleMaskEncoder + CXBlock fuser.
 *   - sam3_v2_memory_attn_forward (phase 2.3b): 4-layer decoupled
 *     RoPE transformer (DecoupledTransformerDecoderLayerv2) with the
 *     SAM 3.1 multiplex config baked in.
 *
 * Key types:  sam3_tracker_v2, sam3_v2_memory_attn, sam3_v2_maskmem
 * Depends on: tracker_v2.h, graph_helpers.h, util/log.h
 * Used by:    sam3_video.c (variant dispatch, phase 2.5),
 *             tests/test_tracker_v2_load.c,
 *             tests/test_maskmem_v2_forward.c,
 *             tests/test_memory_attn_v2_forward.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tracker_v2.h"
#include "graph_helpers.h"
#include "util/log.h"
#include "core/graph.h"

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

static enum sam3_error load_memory_attn(struct sam3_v2_memory_attn *ma,
					const struct sam3_weight_file *wf,
					struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const int d = SAM3_V2_HIDDEN_DIM;
	const int d_ffn = 2048;

#define LX(dst, field_suffix, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_v2.transformer.encoder.layers.%d." \
			 field_suffix, i); \
		const char *_name = buf; \
		LOAD(dst, SAM3_DTYPE_F32, (ndims_), __VA_ARGS__); \
	} while (0)

	for (int i = 0; i < 4; i++) {
		struct sam3_v2_memory_attn_layer *L = &ma->layers[i];

		/* self_attn */
		LX(L->self_q_w, "self_attn_q_proj.weight", 2, d, d);
		LX(L->self_q_b, "self_attn_q_proj.bias",   1, d);
		LX(L->self_k_w, "self_attn_k_proj.weight", 2, d, d);
		LX(L->self_k_b, "self_attn_k_proj.bias",   1, d);
		LX(L->self_v_w, "self_attn_v_proj.weight", 2, d, d);
		LX(L->self_v_b, "self_attn_v_proj.bias",   1, d);
		LX(L->self_out_w, "self_attn_out_proj.weight", 2, d, d);
		LX(L->self_out_b, "self_attn_out_proj.bias",   1, d);

		/* cross_attn */
		LX(L->cross_q_w, "cross_attn_q_proj.weight", 2, d, d);
		LX(L->cross_q_b, "cross_attn_q_proj.bias",   1, d);
		LX(L->cross_k_w, "cross_attn_k_proj.weight", 2, d, d);
		LX(L->cross_k_b, "cross_attn_k_proj.bias",   1, d);
		LX(L->cross_v_w, "cross_attn_v_proj.weight", 2, d, d);
		LX(L->cross_v_b, "cross_attn_v_proj.bias",   1, d);
		LX(L->cross_out_w, "cross_attn_out_proj.weight", 2, d, d);
		LX(L->cross_out_b, "cross_attn_out_proj.bias",   1, d);

		/* image_cross_attn — q and k only (v/out shared with cross) */
		LX(L->img_q_w, "image_cross_attn_q_proj.weight", 2, d, d);
		LX(L->img_q_b, "image_cross_attn_q_proj.bias",   1, d);
		LX(L->img_k_w, "image_cross_attn_k_proj.weight", 2, d, d);
		LX(L->img_k_b, "image_cross_attn_k_proj.bias",   1, d);

		/* FFN */
		LX(L->lin1_w, "linear1.weight", 2, d_ffn, d);
		LX(L->lin1_b, "linear1.bias",   1, d_ffn);
		LX(L->lin2_w, "linear2.weight", 2, d, d_ffn);
		LX(L->lin2_b, "linear2.bias",   1, d);

		/* 3 LayerNorms */
		LX(L->norm1_w, "norm1.weight", 1, d);
		LX(L->norm1_b, "norm1.bias",   1, d);
		LX(L->norm2_w, "norm2.weight", 1, d);
		LX(L->norm2_b, "norm2.bias",   1, d);
		LX(L->norm3_w, "norm3.weight", 1, d);
		LX(L->norm3_b, "norm3.bias",   1, d);
	}
#undef LX

	{
		const char *_name = "tracker_v2.transformer.encoder.norm.weight";
		LOAD(ma->final_norm_w, SAM3_DTYPE_F32, 1, d);
	}
	{
		const char *_name = "tracker_v2.transformer.encoder.norm.bias";
		LOAD(ma->final_norm_b, SAM3_DTYPE_F32, 1, d);
	}
	return SAM3_OK;
}

static enum sam3_error load_mask_decoder_layer(
		struct sam3_v2_mask_decoder_layer *L, int layer_idx,
		const struct sam3_weight_file *wf, struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
#define LM(dst, field, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_v2.sam_mask_decoder.transformer." \
			 "layers.%d." field, layer_idx); \
		const char *_name = buf; \
		LOAD(dst, SAM3_DTYPE_F32, (ndims_), __VA_ARGS__); \
	} while (0)

	/* self_attn (full 256-dim) */
	LM(L->self_q_w, "self_attn.q_proj.weight", 2, 256, 256);
	LM(L->self_q_b, "self_attn.q_proj.bias",   1, 256);
	LM(L->self_k_w, "self_attn.k_proj.weight", 2, 256, 256);
	LM(L->self_k_b, "self_attn.k_proj.bias",   1, 256);
	LM(L->self_v_w, "self_attn.v_proj.weight", 2, 256, 256);
	LM(L->self_v_b, "self_attn.v_proj.bias",   1, 256);
	LM(L->self_out_w, "self_attn.out_proj.weight", 2, 256, 256);
	LM(L->self_out_b, "self_attn.out_proj.bias",   1, 256);

	/* cross_attn_token_to_image (downsample-2: 256 -> 128 -> 256) */
	LM(L->ct2i_q_w, "cross_attn_token_to_image.q_proj.weight", 2, 128, 256);
	LM(L->ct2i_q_b, "cross_attn_token_to_image.q_proj.bias",   1, 128);
	LM(L->ct2i_k_w, "cross_attn_token_to_image.k_proj.weight", 2, 128, 256);
	LM(L->ct2i_k_b, "cross_attn_token_to_image.k_proj.bias",   1, 128);
	LM(L->ct2i_v_w, "cross_attn_token_to_image.v_proj.weight", 2, 128, 256);
	LM(L->ct2i_v_b, "cross_attn_token_to_image.v_proj.bias",   1, 128);
	LM(L->ct2i_out_w, "cross_attn_token_to_image.out_proj.weight", 2, 256, 128);
	LM(L->ct2i_out_b, "cross_attn_token_to_image.out_proj.bias",   1, 256);

	/* cross_attn_image_to_token (same shapes) */
	LM(L->ci2t_q_w, "cross_attn_image_to_token.q_proj.weight", 2, 128, 256);
	LM(L->ci2t_q_b, "cross_attn_image_to_token.q_proj.bias",   1, 128);
	LM(L->ci2t_k_w, "cross_attn_image_to_token.k_proj.weight", 2, 128, 256);
	LM(L->ci2t_k_b, "cross_attn_image_to_token.k_proj.bias",   1, 128);
	LM(L->ci2t_v_w, "cross_attn_image_to_token.v_proj.weight", 2, 128, 256);
	LM(L->ci2t_v_b, "cross_attn_image_to_token.v_proj.bias",   1, 128);
	LM(L->ci2t_out_w, "cross_attn_image_to_token.out_proj.weight", 2, 256, 128);
	LM(L->ci2t_out_b, "cross_attn_image_to_token.out_proj.bias",   1, 256);

	/* MLP (256 -> 2048 -> 256) */
	LM(L->mlp_lin1_w, "mlp.lin1.weight", 2, 2048, 256);
	LM(L->mlp_lin1_b, "mlp.lin1.bias",   1, 2048);
	LM(L->mlp_lin2_w, "mlp.lin2.weight", 2, 256, 2048);
	LM(L->mlp_lin2_b, "mlp.lin2.bias",   1, 256);

	/* 4 LayerNorms */
	LM(L->norm1_w, "norm1.weight", 1, 256);
	LM(L->norm1_b, "norm1.bias",   1, 256);
	LM(L->norm2_w, "norm2.weight", 1, 256);
	LM(L->norm2_b, "norm2.bias",   1, 256);
	LM(L->norm3_w, "norm3.weight", 1, 256);
	LM(L->norm3_b, "norm3.bias",   1, 256);
	LM(L->norm4_w, "norm4.weight", 1, 256);
	LM(L->norm4_b, "norm4.bias",   1, 256);
#undef LM
	return SAM3_OK;
}

static enum sam3_error load_sam_mask_decoder(
		struct sam3_v2_mask_decoder *md,
		const struct sam3_weight_file *wf, struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	enum sam3_error err;

	/* 2-layer transformer */
	for (int i = 0; i < 2; i++) {
		err = load_mask_decoder_layer(&md->layers[i], i, wf, arena);
		if (err != SAM3_OK) return err;
	}

#define LD(dst, path, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_v2.sam_mask_decoder." path); \
		const char *_name = buf; \
		LOAD(dst, SAM3_DTYPE_F32, (ndims_), __VA_ARGS__); \
	} while (0)

	/* final_attn_token_to_image (downsample-2) */
	LD(md->final_q_w, "transformer.final_attn_token_to_image.q_proj.weight", 2, 128, 256);
	LD(md->final_q_b, "transformer.final_attn_token_to_image.q_proj.bias",   1, 128);
	LD(md->final_k_w, "transformer.final_attn_token_to_image.k_proj.weight", 2, 128, 256);
	LD(md->final_k_b, "transformer.final_attn_token_to_image.k_proj.bias",   1, 128);
	LD(md->final_v_w, "transformer.final_attn_token_to_image.v_proj.weight", 2, 128, 256);
	LD(md->final_v_b, "transformer.final_attn_token_to_image.v_proj.bias",   1, 128);
	LD(md->final_out_w, "transformer.final_attn_token_to_image.out_proj.weight", 2, 256, 128);
	LD(md->final_out_b, "transformer.final_attn_token_to_image.out_proj.bias",   1, 256);
	LD(md->norm_final_w, "transformer.norm_final_attn.weight", 1, 256);
	LD(md->norm_final_b, "transformer.norm_final_attn.bias",   1, 256);

	/* output_upscaling: conv_transpose2d (OHWI), LN2d, conv_transpose2d */
	LD(md->up0_w, "output_upscaling.0.weight", 4, 256, 2, 2, 64);
	LD(md->up0_b, "output_upscaling.0.bias",   1, 64);
	LD(md->up1_w, "output_upscaling.1.weight", 1, 64);
	LD(md->up1_b, "output_upscaling.1.bias",   1, 64);
	LD(md->up3_w, "output_upscaling.3.weight", 4, 64, 2, 2, 32);
	LD(md->up3_b, "output_upscaling.3.bias",   1, 32);

	/* output_hypernetworks_mlps[0..2]: 3-layer MLPs (256 -> 256 -> 256 -> 32) */
	for (int m = 0; m < 3; m++) {
		char mbuf[SAM3_WEIGHT_NAME_MAX];
		int out_dims[3] = {256, 256, 32};
		int in_dims[3]  = {256, 256, 256};
		for (int L = 0; L < 3; L++) {
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_v2.sam_mask_decoder."
				 "output_hypernetworks_mlps.%d.layers.%d.weight",
				 m, L);
			{
				const char *_name = mbuf;
				LOAD(md->hn_w[m][L], SAM3_DTYPE_F32, 2,
				     out_dims[L], in_dims[L]);
			}
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_v2.sam_mask_decoder."
				 "output_hypernetworks_mlps.%d.layers.%d.bias",
				 m, L);
			{
				const char *_name = mbuf;
				LOAD(md->hn_b[m][L], SAM3_DTYPE_F32, 1,
				     out_dims[L]);
			}
		}
	}

	/* iou_prediction_head: 3-layer MLP (256 -> 256 -> 256 -> 3) */
	{
		int out_dims[3] = {256, 256, 3};
		int in_dims[3]  = {256, 256, 256};
		char mbuf[SAM3_WEIGHT_NAME_MAX];
		for (int L = 0; L < 3; L++) {
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_v2.sam_mask_decoder."
				 "iou_prediction_head.layers.%d.weight", L);
			{
				const char *_name = mbuf;
				LOAD(md->iou_head_w[L], SAM3_DTYPE_F32, 2,
				     out_dims[L], in_dims[L]);
			}
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_v2.sam_mask_decoder."
				 "iou_prediction_head.layers.%d.bias", L);
			{
				const char *_name = mbuf;
				LOAD(md->iou_head_b[L], SAM3_DTYPE_F32, 1,
				     out_dims[L]);
			}
		}
	}

	/* pred_obj_score_head: 3-layer MLP (256 -> 256 -> 256 -> 1) */
	{
		int out_dims[3] = {256, 256, 1};
		int in_dims[3]  = {256, 256, 256};
		char mbuf[SAM3_WEIGHT_NAME_MAX];
		for (int L = 0; L < 3; L++) {
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_v2.sam_mask_decoder."
				 "pred_obj_score_head.layers.%d.weight", L);
			{
				const char *_name = mbuf;
				LOAD(md->score_head_w[L], SAM3_DTYPE_F32, 2,
				     out_dims[L], in_dims[L]);
			}
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_v2.sam_mask_decoder."
				 "pred_obj_score_head.layers.%d.bias", L);
			{
				const char *_name = mbuf;
				LOAD(md->score_head_b[L], SAM3_DTYPE_F32, 1,
				     out_dims[L]);
			}
		}
	}

	/* High-res feature convs (OHWI) */
	LD(md->conv_s0_w, "conv_s0.weight", 4, 32, 1, 1, 256);
	LD(md->conv_s0_b, "conv_s0.bias",   1, 32);
	LD(md->conv_s1_w, "conv_s1.weight", 4, 64, 1, 1, 256);
	LD(md->conv_s1_b, "conv_s1.bias",   1, 64);

	/* Learned output tokens (multiplex-sized) */
	LD(md->iou_token,        "iou_token.weight",        2, 16, 256);
	LD(md->mask_tokens,      "mask_tokens.weight",      2, 48, 256);
	LD(md->obj_score_token,  "obj_score_token.weight",  2, 16, 256);
#undef LD
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

	err = load_memory_attn(&trk->transformer, wf, arena);
	if (err != SAM3_OK) return err;

	err = load_sam_mask_decoder(&trk->sam_mask_decoder, wf, arena);
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

	sam3_log_info("tracker_v2: loaded phase-2.4a weights "
		      "(maskmem + transformer + mask_decoder + obj_ptr + singletons)");
	return SAM3_OK;
}

#undef LOAD

/* ── Forward graph builders (phase 2.2) ─────────────────────────────── */

/*
 * Apply one CXBlock (ConvNeXt) in channels-last:
 *   y = dwconv(x)
 *   y = layernorm(y)
 *   y = linear(pwconv1, gelu)
 *   y = linear(pwconv2)
 *   y = y * gamma
 *   return x + y
 */
static struct sam3_tensor *cxblock_forward(
		struct sam3_graph *g, struct sam3_arena *a,
		const struct sam3_v2_cxblock *blk,
		struct sam3_tensor *x)
{
	int dim = x->dims[3];
	struct sam3_tensor *residual = x;

	/* Depthwise conv: groups == channels. Weight is OHWI [C, 7, 7, 1]. */
	struct sam3_tensor *y = gh_conv2d(g, a, x,
		blk->dwconv_w, blk->dwconv_b, 1, 3, dim);
	if (!y) return NULL;

	y = gh_layernorm(g, a, y, blk->norm_w, blk->norm_b);
	if (!y) return NULL;

	y = gh_linear(g, a, y, blk->pwconv1_w, blk->pwconv1_b);
	if (!y) return NULL;

	y = gh_gelu(g, a, y);
	if (!y) return NULL;

	y = gh_linear(g, a, y, blk->pwconv2_w, blk->pwconv2_b);
	if (!y) return NULL;

	y = gh_mul(g, a, y, blk->gamma);
	if (!y) return NULL;

	return gh_add(g, a, residual, y);
}

struct sam3_tensor *sam3_v2_maskmem_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_v2_maskmem *mm,
		struct sam3_tensor *pix_feat,
		struct sam3_tensor *masks,
		int skip_mask_sigmoid)
{
	if (!g || !arena || !mm || !pix_feat || !masks)
		return NULL;
	if (pix_feat->n_dims != 4 || masks->n_dims != 4)
		return NULL;

	struct sam3_tensor *x = masks;

	/* 1. Sigmoid the mask logits (unless caller already applied). */
	if (!skip_mask_sigmoid) {
		x = gh_sigmoid(g, arena, x);
		if (!x) return NULL;
	}

	/* 2. Four-stage downsampler: Conv2d(k=3, s=2, p=1) + LN2d + GELU. */
	for (int s = 0; s < 4; s++) {
		x = gh_conv2d(g, arena, x,
			       mm->mask_downsampler.conv_w[s],
			       mm->mask_downsampler.conv_b[s],
			       2, 1, 1);
		if (!x) return NULL;

		/* LayerNorm2d in channels-first is equivalent to LayerNorm
		 * over the last dim in NHWC — our layout. */
		x = gh_layernorm(g, arena, x,
				  mm->mask_downsampler.norm_w[s],
				  mm->mask_downsampler.norm_b[s]);
		if (!x) return NULL;

		x = gh_gelu(g, arena, x);
		if (!x) return NULL;
	}

	/* 3. Final 1x1 projection to hidden_dim (256). */
	x = gh_conv2d(g, arena, x,
		       mm->mask_downsampler.proj_w,
		       mm->mask_downsampler.proj_b,
		       1, 0, 1);
	if (!x) return NULL;

	/* 4. Project pix_feat through a 1x1 conv (pix_feat_proj). */
	struct sam3_tensor *pix = gh_conv2d(g, arena, pix_feat,
		mm->pix_feat_proj_w, mm->pix_feat_proj_b, 1, 0, 1);
	if (!pix) return NULL;

	/* 5. Element-wise sum of projected pix_feat and downsampled masks. */
	x = gh_add(g, arena, pix, x);
	if (!x) return NULL;

	/* 6. Two CXBlocks in the fuser. */
	for (int i = 0; i < 2; i++) {
		x = cxblock_forward(g, arena, &mm->fuser[i], x);
		if (!x) return NULL;
	}

	return x;
}

/*
 * ── Memory-attention forward (phase 2.3b, full implementation) ────
 *
 * 8-head decoupled RoPE transformer encoder, 4 layers. See the header
 * doc for the full semantics of each layer. The config is hard-coded
 * because no other SAM 3.1 variant ships today.
 */

/* Fixed config baked into the .sam3 weights. */
#define V2_ATTN_HEADS           8
#define V2_ATTN_HEAD_DIM        (SAM3_V2_HIDDEN_DIM / V2_ATTN_HEADS)  /* 32 */
#define V2_ATTN_HALF            (V2_ATTN_HEAD_DIM / 2)                /* 16 */
#define V2_ATTN_QUARTER         (V2_ATTN_HEAD_DIM / 4)                /* 8 */
#define V2_ROPE_THETA           10000.0f
/* pos_enc_at_input=True scales src_pos by 0.1 before the residual add.
 * Matches TransformerEncoderDecoupledCrossAttention.forward in
 * reference/sam3/sam3/model/decoder.py. */
#define V2_POS_ENC_INPUT_SCALE  0.1f

/*
 * build_rope_2d_axial - Fill [n_pos, head_dim/2] cos/sin tables for a
 * grid_w × grid_w 2D axial RoPE basis.
 *
 * Matches Python compute_axial_cis(dim=head_dim, end_x=grid_w,
 * end_y=grid_w, theta=10000). First head_dim/4 entries encode the x
 * (column) axis, last head_dim/4 entries encode the y (row) axis —
 * exactly the convention used by the ViT encoder's
 * precompute_rope_table in src/model/image_encoder.c.
 */
static enum sam3_error build_rope_2d_axial(
		struct sam3_arena *arena, int grid_w,
		struct sam3_tensor **out_cos, struct sam3_tensor **out_sin)
{
	int n_pos = grid_w * grid_w;
	int dims[] = {n_pos, V2_ATTN_HALF};

	*out_cos = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	*out_sin = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	if (!*out_cos || !*out_sin)
		return SAM3_ENOMEM;

	float freqs[V2_ATTN_QUARTER];
	for (int i = 0; i < V2_ATTN_QUARTER; i++)
		freqs[i] = 1.0f / powf(V2_ROPE_THETA,
				       (float)(i * 4) / (float)V2_ATTN_HEAD_DIM);

	float *cos_d = (float *)(*out_cos)->data;
	float *sin_d = (float *)(*out_sin)->data;
	for (int py = 0; py < grid_w; py++) {
		for (int px = 0; px < grid_w; px++) {
			int pos = py * grid_w + px;
			float *cr = cos_d + pos * V2_ATTN_HALF;
			float *sr = sin_d + pos * V2_ATTN_HALF;
			for (int i = 0; i < V2_ATTN_QUARTER; i++) {
				float ax = (float)px * freqs[i];
				cr[i] = cosf(ax);
				sr[i] = sinf(ax);
			}
			for (int i = 0; i < V2_ATTN_QUARTER; i++) {
				float ay = (float)py * freqs[i];
				cr[V2_ATTN_QUARTER + i] = cosf(ay);
				sr[V2_ATTN_QUARTER + i] = sinf(ay);
			}
		}
	}
	return SAM3_OK;
}

/*
 * tile_rope_table - Repeat an [n, half] table `r` times along axis 0
 * to produce an [r*n, half] table. Matches repeat_freqs_k semantics
 * in apply_rotary_enc: the same cis values are re-applied for each
 * memory-frame slot so tokens at the same spatial position share
 * the same rotation regardless of which slot they live in.
 *
 * When r == 1 returns @base unchanged (shared pointer). The returned
 * table must be treated as read-only — gh_rope consumes it as a
 * constant graph input and never writes back.
 */
static struct sam3_tensor *tile_rope_table(
		struct sam3_arena *arena,
		struct sam3_tensor *base, int r)
{
	if (r <= 0)
		return NULL;
	if (r == 1)
		return base;

	int n = base->dims[0];
	int half = base->dims[1];
	int dims[] = {n * r, half};

	struct sam3_tensor *out = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						   2, dims);
	if (!out)
		return NULL;

	const float *src = (const float *)base->data;
	float *dst = (float *)out->data;
	size_t row = (size_t)half * sizeof(float);
	for (int i = 0; i < r; i++)
		memcpy(dst + (size_t)i * n * half, src, row * (size_t)n);
	return out;
}

/*
 * mha_sdpa_with_rope - Shared attention body: project Q/K/V → RoPE on Q
 * and (part of) K → fused SDPA → output projection.
 *
 * All of q_in/k_in/v_in are expected in [B, N, d_model] layout. They
 * have already been projected by the caller (memory attention's
 * decoupled queries and keys are summed from two sources, which does
 * not fit the "single input, internal projection" helper).
 *
 * @cos_q/sin_q:       [N_q, head_dim/2] RoPE tables for Q.
 * @cos_k/sin_k:       [N_k - num_k_exclude, head_dim/2] RoPE tables
 *                     for the first N_k - num_k_exclude rows of K,
 *                     or NULL to skip K-side RoPE entirely.
 * @num_k_exclude:     trailing K rows that skip RoPE.
 * @out_proj_w/_b:     linear projection applied after SDPA.
 *
 * Returns tensor [B, N_q, d_model] on success.
 */
static struct sam3_tensor *mha_sdpa_with_rope(
		struct sam3_graph *g, struct sam3_arena *a,
		struct sam3_tensor *q, struct sam3_tensor *k,
		struct sam3_tensor *v,
		struct sam3_tensor *cos_q, struct sam3_tensor *sin_q,
		struct sam3_tensor *cos_k, struct sam3_tensor *sin_k,
		int num_k_exclude,
		struct sam3_tensor *out_w, struct sam3_tensor *out_b)
{
	int B = q->dims[0];
	int Nq = q->dims[1];
	int D = q->dims[2];
	int Nk = k->dims[1];
	int H = V2_ATTN_HEADS;
	int HD = V2_ATTN_HEAD_DIM;

	int r4q[] = {B, Nq, H, HD};
	int r4k[] = {B, Nk, H, HD};
	int perm[] = {0, 2, 1, 3};
	int r3[]  = {B, Nq, D};

	/* Reshape to [B, N, H, HD] for RoPE. */
	q = gh_reshape(g, a, q, 4, r4q);
	k = gh_reshape(g, a, k, 4, r4k);
	v = gh_reshape(g, a, v, 4, r4k);
	if (!q || !k || !v)
		return NULL;

	/* RoPE on Q (full length) and K (first Nk - num_k_exclude). */
	q = gh_rope(g, a, q, cos_q, sin_q, 0, 1.0f);
	if (!q)
		return NULL;

	/* cos_k == NULL is the "skip K-side RoPE entirely" contract. In
	 * the current call path that only happens when the outer forward
	 * sees k_rope_len <= 0 (all memory tokens excluded from RoPE —
	 * e.g. Nm == num_k_exclude_rope, which would mean the memory
	 * bank is pure obj_ptr tokens). Self-attention always passes
	 * cos_q for both Q and K. */
	if (cos_k && sin_k) {
		int k_rope_len = Nk - num_k_exclude;
		if (k_rope_len < 0)
			return NULL;
		if (num_k_exclude > 0) {
			struct sam3_tensor *k_rope =
				gh_slice(g, a, k, 1, 0, k_rope_len);
			struct sam3_tensor *k_no =
				gh_slice(g, a, k, 1, k_rope_len, Nk);
			if (!k_rope || !k_no)
				return NULL;
			k_rope = gh_rope(g, a, k_rope, cos_k, sin_k,
					 0, 1.0f);
			if (!k_rope)
				return NULL;
			struct sam3_tensor *pair[2] = {k_rope, k_no};
			k = gh_concat(g, a, pair, 2, 1);
			if (!k)
				return NULL;
		} else {
			k = gh_rope(g, a, k, cos_k, sin_k, 0, 1.0f);
			if (!k)
				return NULL;
		}
	}

	/* Permute to [B, H, N, HD] for SDPA. */
	q = gh_permute(g, a, q, perm);
	k = gh_permute(g, a, k, perm);
	v = gh_permute(g, a, v, perm);
	if (!q || !k || !v)
		return NULL;

	struct sam3_tensor *attn = gh_sdpa(g, a, q, k, v, NULL, HD);
	if (!attn)
		return NULL;

	/* Back to [B, Nq, H, HD] then [B, Nq, D]. */
	attn = gh_permute(g, a, attn, perm);
	if (!attn)
		return NULL;
	attn = gh_reshape(g, a, attn, 3, r3);
	if (!attn)
		return NULL;

	return gh_linear(g, a, attn, out_w, out_b);
}

/*
 * memory_attn_layer - Build one DecoupledTransformerDecoderLayerv2
 * forward in pre-norm, cross-attention-second mode.
 *
 * Residual accumulator `output` is [B, Nq, 256]. Cross-attention
 * draws from memory / memory_image of length Nm. All positional
 * behaviours (pos_enc_at_attn=False, pos_enc_at_cross_attn_queries=
 * False, pos_enc_at_cross_attn_keys=True) are baked in.
 */
static struct sam3_tensor *memory_attn_layer(
		struct sam3_graph *g, struct sam3_arena *a,
		const struct sam3_v2_memory_attn_layer *L,
		struct sam3_tensor *output,
		struct sam3_tensor *image,
		struct sam3_tensor *memory,
		struct sam3_tensor *memory_image,
		struct sam3_tensor *memory_image_pos,
		struct sam3_tensor *cos_q, struct sam3_tensor *sin_q,
		struct sam3_tensor *cos_k, struct sam3_tensor *sin_k,
		int num_k_exclude)
{
	/* ── Self-attention ───────────────────────────────────────── */
	struct sam3_tensor *tgt2 = gh_layernorm(g, a, output,
						L->norm1_w, L->norm1_b);
	if (!tgt2)
		return NULL;

	struct sam3_tensor *q = gh_linear(g, a, tgt2, L->self_q_w, L->self_q_b);
	struct sam3_tensor *k = gh_linear(g, a, tgt2, L->self_k_w, L->self_k_b);
	struct sam3_tensor *v = gh_linear(g, a, tgt2, L->self_v_w, L->self_v_b);
	if (!q || !k || !v)
		return NULL;

	struct sam3_tensor *attn = mha_sdpa_with_rope(
		g, a, q, k, v,
		cos_q, sin_q, cos_q, sin_q, 0,
		L->self_out_w, L->self_out_b);
	if (!attn)
		return NULL;
	output = gh_add(g, a, output, attn);
	if (!output)
		return NULL;

	/* ── Decoupled cross-attention ────────────────────────────── */
	tgt2 = gh_layernorm(g, a, output, L->norm2_w, L->norm2_b);
	if (!tgt2)
		return NULL;

	/* q = image_cross_attn_q_proj(image) + cross_attn_q_proj(tgt2) */
	struct sam3_tensor *q_tgt = gh_linear(g, a, tgt2,
					       L->cross_q_w, L->cross_q_b);
	struct sam3_tensor *q_img = gh_linear(g, a, image,
					       L->img_q_w, L->img_q_b);
	if (!q_tgt || !q_img)
		return NULL;
	q = gh_add(g, a, q_img, q_tgt);
	if (!q)
		return NULL;

	/* k = image_cross_attn_k_proj(memory_image)
	 *     + cross_attn_k_proj(memory) + memory_image_pos */
	struct sam3_tensor *k_mem = gh_linear(g, a, memory,
					       L->cross_k_w, L->cross_k_b);
	struct sam3_tensor *k_img = gh_linear(g, a, memory_image,
					       L->img_k_w, L->img_k_b);
	if (!k_mem || !k_img)
		return NULL;
	k = gh_add(g, a, k_img, k_mem);
	if (!k)
		return NULL;
	if (memory_image_pos) {
		k = gh_add(g, a, k, memory_image_pos);
		if (!k)
			return NULL;
	}

	v = gh_linear(g, a, memory, L->cross_v_w, L->cross_v_b);
	if (!v)
		return NULL;

	attn = mha_sdpa_with_rope(
		g, a, q, k, v,
		cos_q, sin_q, cos_k, sin_k, num_k_exclude,
		L->cross_out_w, L->cross_out_b);
	if (!attn)
		return NULL;
	output = gh_add(g, a, output, attn);
	if (!output)
		return NULL;

	/* ── FFN ──────────────────────────────────────────────────── */
	tgt2 = gh_layernorm(g, a, output, L->norm3_w, L->norm3_b);
	if (!tgt2)
		return NULL;
	tgt2 = gh_linear(g, a, tgt2, L->lin1_w, L->lin1_b);
	if (!tgt2)
		return NULL;
	tgt2 = gh_gelu(g, a, tgt2);
	if (!tgt2)
		return NULL;
	tgt2 = gh_linear(g, a, tgt2, L->lin2_w, L->lin2_b);
	if (!tgt2)
		return NULL;
	return gh_add(g, a, output, tgt2);
}

struct sam3_tensor *sam3_v2_memory_attn_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_v2_memory_attn *ma,
		struct sam3_tensor *tgt,
		struct sam3_tensor *tgt_pos,
		struct sam3_tensor *image,
		struct sam3_tensor *memory,
		struct sam3_tensor *memory_image,
		struct sam3_tensor *memory_image_pos,
		int grid_w,
		int num_k_exclude_rope)
{
	if (!g || !arena || !ma || !tgt || !image
	    || !memory || !memory_image)
		return NULL;
	if (tgt->n_dims != 3 || image->n_dims != 3
	    || memory->n_dims != 3 || memory_image->n_dims != 3) {
		sam3_log_error("memory_attn: inputs must be 3D [B,N,D]");
		return NULL;
	}

	int B  = tgt->dims[0];
	int Nq = tgt->dims[1];
	int D  = tgt->dims[2];
	int Nm = memory->dims[1];

	if (D != SAM3_V2_HIDDEN_DIM) {
		sam3_log_error("memory_attn: d_model %d != %d",
			       D, SAM3_V2_HIDDEN_DIM);
		return NULL;
	}
	if (grid_w <= 0) {
		sam3_log_error("memory_attn: grid_w must be positive, got %d",
			       grid_w);
		return NULL;
	}
	if (Nq != grid_w * grid_w) {
		sam3_log_error("memory_attn: Nq %d != grid_w*grid_w (%d)",
			       Nq, grid_w * grid_w);
		return NULL;
	}
	if (image->dims[0] != B || image->dims[1] != Nq
	    || image->dims[2] != D
	    || memory_image->dims[0] != B || memory_image->dims[1] != Nm
	    || memory_image->dims[2] != D) {
		sam3_log_error("memory_attn: image/memory_image shape mismatch");
		return NULL;
	}
	if (num_k_exclude_rope < 0 || num_k_exclude_rope > Nm) {
		sam3_log_error("memory_attn: num_k_exclude_rope %d out of "
			       "range [0, %d]", num_k_exclude_rope, Nm);
		return NULL;
	}

	/* ── Build 2D axial RoPE tables ───────────────────────────── */
	struct sam3_tensor *cos_q = NULL, *sin_q = NULL;
	if (build_rope_2d_axial(arena, grid_w, &cos_q, &sin_q) != SAM3_OK)
		return NULL;

	struct sam3_tensor *cos_k = NULL, *sin_k = NULL;
	int k_rope_len = Nm - num_k_exclude_rope;
	if (k_rope_len > 0) {
		if (k_rope_len % Nq != 0) {
			sam3_log_error("memory_attn: K rope len %d not a "
				       "multiple of Nq %d",
				       k_rope_len, Nq);
			return NULL;
		}
		int r = k_rope_len / Nq;
		cos_k = tile_rope_table(arena, cos_q, r);
		sin_k = tile_rope_table(arena, sin_q, r);
		if (!cos_k || !sin_k)
			return NULL;
	}

	/* ── pos_enc_at_input=True: output = tgt + 0.1 * tgt_pos ──── */
	struct sam3_tensor *output = tgt;
	if (tgt_pos) {
		int one[] = {1};
		struct sam3_tensor *scale = gh_alloc_tensor(arena,
			SAM3_DTYPE_F32, 1, one);
		if (!scale)
			return NULL;
		((float *)scale->data)[0] = V2_POS_ENC_INPUT_SCALE;
		struct sam3_tensor *scaled = gh_mul(g, arena, tgt_pos, scale);
		if (!scaled)
			return NULL;
		output = gh_add(g, arena, tgt, scaled);
		if (!output)
			return NULL;
	}

	/* ── 4 layers ─────────────────────────────────────────────── */
	for (int i = 0; i < 4; i++) {
		output = memory_attn_layer(g, arena, &ma->layers[i],
			output, image, memory, memory_image,
			memory_image_pos, cos_q, sin_q, cos_k, sin_k,
			num_k_exclude_rope);
		if (!output)
			return NULL;
	}

	/* ── Final encoder.norm (use_image_in_output=False) ───────── */
	return gh_layernorm(g, arena, output,
			    ma->final_norm_w, ma->final_norm_b);
}
