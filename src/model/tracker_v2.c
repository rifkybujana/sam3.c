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
 * PHASE 2.3B STUB — the decoupled 8-head RoPE memory attention.
 *
 * Needs three sub-pieces that don't exist as graph helpers today:
 *   (a) Multi-head attention with pre-projected Q/K/V and optional
 *       RoPE on a configurable per-head stride. Existing
 *       gh_multihead_attention_rope_sep always projects internally
 *       from a single input and assumes one position basis for both
 *       Q and K.
 *   (b) RoPE tables for the 72x72 image feature grid that match the
 *       `SimpleRoPEAttention(rope_theta=10000, feat_sizes=[72,72])`
 *       config (2D axial RoPE, not the 1D sequence RoPE used in the
 *       text encoder).
 *   (c) A `num_k_exclude_rope` knob so obj_ptr memory tokens bypass
 *       the rotation while maskmem tokens don't.
 *
 * Returning NULL here instead of silently producing garbage keeps
 * sam3_video_start_ex's SAM-3.1 reject honest: tracking fails cleanly
 * until this implementation lands.
 */
struct sam3_tensor *sam3_v2_memory_attn_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_v2_memory_attn *ma,
		struct sam3_tensor *tgt,
		struct sam3_tensor *tgt_pos,
		struct sam3_tensor *image,
		struct sam3_tensor *image_pos,
		struct sam3_tensor *memory,
		struct sam3_tensor *memory_pos,
		int num_k_exclude_rope)
{
	(void)g; (void)arena; (void)ma; (void)tgt; (void)tgt_pos;
	(void)image; (void)image_pos; (void)memory; (void)memory_pos;
	(void)num_k_exclude_rope;
	sam3_log_error("sam3_v2_memory_attn_forward: not yet implemented "
		       "(phase 2.3b)");
	return NULL;
}
