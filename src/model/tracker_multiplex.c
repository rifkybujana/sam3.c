/*
 * src/model/tracker_multiplex.c - SAM 3.1 multiplex tracker (loader + forwards)
 *
 * Loader (sam3_tracker_multiplex_init / sam3_tracker_multiplex_load) populates the
 * tracker_multiplex struct from a SAM 3.1 .sam3 file: maskmem backbone
 * (38 tensors), memory-attention transformer (122 tensors), SAM mask
 * decoder (125 tensors), obj_ptr MLPs + linears (12 tensors), and
 * singleton embeddings (6 tensors) — 303 tensors end-to-end.
 *
 * Forwards:
 *   - sam3_multiplex_maskmem_forward (phase 2.2): pix_feat + masks -> memory
 *     tokens via the multiplex-aware SimpleMaskEncoder + CXBlock fuser.
 *   - sam3_multiplex_memory_attn_forward (phase 2.3b): 4-layer decoupled
 *     RoPE transformer (DecoupledTransformerDecoderLayerv2) with the
 *     SAM 3.1 multiplex config baked in.
 *   - sam3_multiplex_mask_decoder_forward (phase 2.4b): 2-layer two-way
 *     transformer + output upscaling + hypernetwork + heads, matching
 *     MultiplexMaskDecoder.predict_masks.
 *
 * Key types:  sam3_tracker_multiplex, sam3_multiplex_memory_attn, sam3_multiplex_maskmem,
 *             sam3_multiplex_mask_decoder
 * Depends on: tracker_multiplex.h, graph_helpers.h, util/log.h
 * Used by:    sam3_video.c (variant dispatch, phase 2.5),
 *             tests/test_tracker_multiplex_load.c,
 *             tests/test_maskmem_multiplex_forward.c,
 *             tests/test_memory_attn_multiplex_forward.c,
 *             tests/test_mask_decoder_multiplex_forward.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tracker_multiplex.h"
#include "tracker_multiplex_internal.h"
#include "graph_helpers.h"
#include "memory_bank.h"
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
		sam3_log_error("tracker_multiplex: required tensor absent: %s", name);
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

/* --- Sub-module loaders --- */

static enum sam3_error load_mask_downsampler(
		struct sam3_multiplex_mask_downsampler *ms,
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
			 "tracker_multiplex.maskmem_backbone."
			 "mask_downsampler.encoder.%d.weight", conv_idx);
		{
			const char *_name = buf;
			LOAD(ms->conv_w[s], SAM3_DTYPE_F32, 4,
			     out_c, 3, 3, in_c);
		}
		snprintf(buf, sizeof(buf),
			 "tracker_multiplex.maskmem_backbone."
			 "mask_downsampler.encoder.%d.bias", conv_idx);
		{
			const char *_name = buf;
			LOAD(ms->conv_b[s], SAM3_DTYPE_F32, 1, out_c);
		}
		snprintf(buf, sizeof(buf),
			 "tracker_multiplex.maskmem_backbone."
			 "mask_downsampler.encoder.%d.weight", norm_idx);
		{
			const char *_name = buf;
			LOAD(ms->norm_w[s], SAM3_DTYPE_F32, 1, out_c);
		}
		snprintf(buf, sizeof(buf),
			 "tracker_multiplex.maskmem_backbone."
			 "mask_downsampler.encoder.%d.bias", norm_idx);
		{
			const char *_name = buf;
			LOAD(ms->norm_b[s], SAM3_DTYPE_F32, 1, out_c);
		}
	}

	{
		const char *_name =
			"tracker_multiplex.maskmem_backbone."
			"mask_downsampler.encoder.12.weight";
		LOAD(ms->proj_w, SAM3_DTYPE_F32, 4, 256, 1, 1, 1024);
	}
	{
		const char *_name =
			"tracker_multiplex.maskmem_backbone."
			"mask_downsampler.encoder.12.bias";
		LOAD(ms->proj_b, SAM3_DTYPE_F32, 1, 256);
	}
	return SAM3_OK;
}

static enum sam3_error load_cxblock(struct sam3_multiplex_cxblock *blk,
				    int layer_idx,
				    const struct sam3_weight_file *wf,
				    struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const int d = SAM3_MULTIPLEX_HIDDEN_DIM;
	const int d4 = d * 4;

#define LX(dst, field, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_multiplex.maskmem_backbone.fuser.layers.%d." field, \
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

static enum sam3_error load_maskmem(struct sam3_multiplex_maskmem *mm,
				    const struct sam3_weight_file *wf,
				    struct sam3_arena *arena)
{
	enum sam3_error err;
	err = load_mask_downsampler(&mm->mask_downsampler, wf, arena);
	if (err != SAM3_OK) return err;

	{
		const char *_name =
			"tracker_multiplex.maskmem_backbone.pix_feat_proj.weight";
		LOAD(mm->pix_feat_proj_w, SAM3_DTYPE_F32, 4, 256, 1, 1, 256);
	}
	{
		const char *_name =
			"tracker_multiplex.maskmem_backbone.pix_feat_proj.bias";
		LOAD(mm->pix_feat_proj_b, SAM3_DTYPE_F32, 1, 256);
	}

	for (int i = 0; i < 2; i++) {
		err = load_cxblock(&mm->fuser[i], i, wf, arena);
		if (err != SAM3_OK) return err;
	}
	return SAM3_OK;
}

static enum sam3_error load_mlp3(struct sam3_multiplex_mlp3 *mlp,
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

static enum sam3_error load_memory_attn(struct sam3_multiplex_memory_attn *ma,
					const struct sam3_weight_file *wf,
					struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const int d = SAM3_MULTIPLEX_HIDDEN_DIM;
	const int d_ffn = 2048;

#define LX(dst, field_suffix, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_multiplex.transformer.encoder.layers.%d." \
			 field_suffix, i); \
		const char *_name = buf; \
		LOAD(dst, SAM3_DTYPE_F32, (ndims_), __VA_ARGS__); \
	} while (0)

	for (int i = 0; i < 4; i++) {
		struct sam3_multiplex_memory_attn_layer *L = &ma->layers[i];

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
		const char *_name = "tracker_multiplex.transformer.encoder.norm.weight";
		LOAD(ma->final_norm_w, SAM3_DTYPE_F32, 1, d);
	}
	{
		const char *_name = "tracker_multiplex.transformer.encoder.norm.bias";
		LOAD(ma->final_norm_b, SAM3_DTYPE_F32, 1, d);
	}
	return SAM3_OK;
}

static enum sam3_error load_mask_decoder_layer(
		struct sam3_multiplex_mask_decoder_layer *L, int layer_idx,
		const struct sam3_weight_file *wf, struct sam3_arena *arena)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
#define LM(dst, field, ndims_, ...) \
	do { \
		snprintf(buf, sizeof(buf), \
			 "tracker_multiplex.sam_mask_decoder.transformer." \
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
		struct sam3_multiplex_mask_decoder *md,
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
			 "tracker_multiplex.sam_mask_decoder." path); \
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

	/* output_upscaling: conv_transpose2d (OHWI), LN2d, conv_transpose2d.
	 * Raw PyTorch ConvTranspose2d weight is IOHW [IC, OC, kH, kW]; the
	 * conv_perm pass in tools/weight_conv_perm.c permutes it to OHWI
	 * [OC, kH, kW, IC] so this loader matches gh_conv_transpose2d's
	 * expected layout. */
	LD(md->up0_w, "output_upscaling.0.weight", 4, 64, 2, 2, 256);
	LD(md->up0_b, "output_upscaling.0.bias",   1, 64);
	LD(md->up1_w, "output_upscaling.1.weight", 1, 64);
	LD(md->up1_b, "output_upscaling.1.bias",   1, 64);
	LD(md->up3_w, "output_upscaling.3.weight", 4, 32, 2, 2, 64);
	LD(md->up3_b, "output_upscaling.3.bias",   1, 32);

	/* output_hypernetworks_mlps[0..2]: 3-layer MLPs (256 -> 256 -> 256 -> 32) */
	for (int m = 0; m < 3; m++) {
		char mbuf[SAM3_WEIGHT_NAME_MAX];
		int out_dims[3] = {256, 256, 32};
		int in_dims[3]  = {256, 256, 256};
		for (int L = 0; L < 3; L++) {
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_multiplex.sam_mask_decoder."
				 "output_hypernetworks_mlps.%d.layers.%d.weight",
				 m, L);
			{
				const char *_name = mbuf;
				LOAD(md->hn_w[m][L], SAM3_DTYPE_F32, 2,
				     out_dims[L], in_dims[L]);
			}
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_multiplex.sam_mask_decoder."
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
				 "tracker_multiplex.sam_mask_decoder."
				 "iou_prediction_head.layers.%d.weight", L);
			{
				const char *_name = mbuf;
				LOAD(md->iou_head_w[L], SAM3_DTYPE_F32, 2,
				     out_dims[L], in_dims[L]);
			}
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_multiplex.sam_mask_decoder."
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
				 "tracker_multiplex.sam_mask_decoder."
				 "pred_obj_score_head.layers.%d.weight", L);
			{
				const char *_name = mbuf;
				LOAD(md->score_head_w[L], SAM3_DTYPE_F32, 2,
				     out_dims[L], in_dims[L]);
			}
			snprintf(mbuf, sizeof(mbuf),
				 "tracker_multiplex.sam_mask_decoder."
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

static enum sam3_error load_singletons(struct sam3_tracker_multiplex *trk,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena)
{
	{
		const char *_name =
			"tracker_multiplex.image_pe_layer."
			"positional_encoding_gaussian_matrix";
		LOAD(trk->image_pe_gauss, SAM3_DTYPE_F32, 2, 2, 128);
	}
	{
		const char *_name = "tracker_multiplex.maskmem_tpos_enc";
		LOAD(trk->maskmem_tpos_enc, SAM3_DTYPE_F32, 4,
		     SAM3_MULTIPLEX_NUM_MASKMEM, 1, 1, SAM3_MULTIPLEX_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_multiplex.no_obj_embed_spatial";
		LOAD(trk->no_obj_embed_spatial, SAM3_DTYPE_F32, 2,
		     SAM3_MULTIPLEX_COUNT, SAM3_MULTIPLEX_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_multiplex.output_valid_embed";
		LOAD(trk->output_valid_embed, SAM3_DTYPE_F32, 2,
		     SAM3_MULTIPLEX_COUNT, SAM3_MULTIPLEX_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_multiplex.output_invalid_embed";
		LOAD(trk->output_invalid_embed, SAM3_DTYPE_F32, 2,
		     SAM3_MULTIPLEX_COUNT, SAM3_MULTIPLEX_HIDDEN_DIM);
	}
	{
		const char *_name = "tracker_multiplex.interactivity_no_mem_embed";
		LOAD(trk->interactivity_no_mem_embed, SAM3_DTYPE_F32, 3,
		     1, 1, SAM3_MULTIPLEX_HIDDEN_DIM);
	}
	return SAM3_OK;
}

/* --- Public API  --- */

enum sam3_error sam3_tracker_multiplex_init(struct sam3_tracker_multiplex *trk)
{
	if (!trk)
		return SAM3_EINVAL;
	memset(trk, 0, sizeof(*trk));
	return SAM3_OK;
}

enum sam3_error sam3_tracker_multiplex_load(struct sam3_tracker_multiplex *trk,
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

	err = load_mlp3(&trk->obj_ptr_proj, "tracker_multiplex.obj_ptr_proj",
			wf, arena);
	if (err != SAM3_OK) return err;

	{
		const char *_name = "tracker_multiplex.obj_ptr_tpos_proj.weight";
		LOAD(trk->obj_ptr_tpos_proj_w, SAM3_DTYPE_F32, 2, 256, 256);
	}
	{
		const char *_name = "tracker_multiplex.obj_ptr_tpos_proj.bias";
		LOAD(trk->obj_ptr_tpos_proj_b, SAM3_DTYPE_F32, 1, 256);
	}
	{
		const char *_name = "tracker_multiplex.no_obj_ptr_linear.weight";
		LOAD(trk->no_obj_ptr_linear_w, SAM3_DTYPE_F32, 2, 256, 256);
	}
	{
		const char *_name = "tracker_multiplex.no_obj_ptr_linear.bias";
		LOAD(trk->no_obj_ptr_linear_b, SAM3_DTYPE_F32, 1, 256);
	}

	err = load_singletons(trk, wf, arena);
	if (err != SAM3_OK) return err;

	sam3_log_info("tracker_multiplex: loaded phase-2.4a weights "
		      "(maskmem + transformer + mask_decoder + obj_ptr + singletons)");
	return SAM3_OK;
}

#undef LOAD

/* --- Forward graph builders (phase 2.2)  --- */

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
		const struct sam3_multiplex_cxblock *blk,
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

struct sam3_tensor *sam3_multiplex_maskmem_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_multiplex_maskmem *mm,
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
 *  Memory-attention forward (phase 2.3b, full implementation) ────
 *
 * 8-head decoupled RoPE transformer encoder, 4 layers. See the header
 * doc for the full semantics of each layer. The config is hard-coded
 * because no other SAM 3.1 variant ships today.
 */

/* Fixed config baked into the .sam3 weights. */
#define MUX_ATTN_HEADS           8
#define MUX_ATTN_HEAD_DIM        (SAM3_MULTIPLEX_HIDDEN_DIM / MUX_ATTN_HEADS)  /* 32 */
#define MUX_ATTN_HALF            (MUX_ATTN_HEAD_DIM / 2)                /* 16 */
#define MUX_ATTN_QUARTER         (MUX_ATTN_HEAD_DIM / 4)                /* 8 */
#define MUX_ROPE_THETA           10000.0f
/* pos_enc_at_input=True scales src_pos by 0.1 before the residual add.
 * Matches TransformerEncoderDecoupledCrossAttention.forward in
 * reference/sam3/sam3/model/decoder.py. */
#define MUX_POS_ENC_INPUT_SCALE  0.1f

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
	int dims[] = {n_pos, MUX_ATTN_HALF};

	*out_cos = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	*out_sin = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	if (!*out_cos || !*out_sin)
		return SAM3_ENOMEM;

	float freqs[MUX_ATTN_QUARTER];
	for (int i = 0; i < MUX_ATTN_QUARTER; i++)
		freqs[i] = 1.0f / powf(MUX_ROPE_THETA,
				       (float)(i * 4) / (float)MUX_ATTN_HEAD_DIM);

	float *cos_d = (float *)(*out_cos)->data;
	float *sin_d = (float *)(*out_sin)->data;
	for (int py = 0; py < grid_w; py++) {
		for (int px = 0; px < grid_w; px++) {
			int pos = py * grid_w + px;
			float *cr = cos_d + pos * MUX_ATTN_HALF;
			float *sr = sin_d + pos * MUX_ATTN_HALF;
			for (int i = 0; i < MUX_ATTN_QUARTER; i++) {
				float ax = (float)px * freqs[i];
				cr[i] = cosf(ax);
				sr[i] = sinf(ax);
			}
			for (int i = 0; i < MUX_ATTN_QUARTER; i++) {
				float ay = (float)py * freqs[i];
				cr[MUX_ATTN_QUARTER + i] = cosf(ay);
				sr[MUX_ATTN_QUARTER + i] = sinf(ay);
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
	int H = MUX_ATTN_HEADS;
	int HD = MUX_ATTN_HEAD_DIM;

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
		const struct sam3_multiplex_memory_attn_layer *L,
		struct sam3_tensor *output,
		struct sam3_tensor *image,
		struct sam3_tensor *memory,
		struct sam3_tensor *memory_image,
		struct sam3_tensor *memory_image_pos,
		struct sam3_tensor *cos_q, struct sam3_tensor *sin_q,
		struct sam3_tensor *cos_k, struct sam3_tensor *sin_k,
		int num_k_exclude)
{
	/* --- Self-attention  --- */
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

	/* --- Decoupled cross-attention ─ --- */
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

	/* --- FFN  --- */
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

struct sam3_tensor *sam3_multiplex_memory_attn_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_multiplex_memory_attn *ma,
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

	if (D != SAM3_MULTIPLEX_HIDDEN_DIM) {
		sam3_log_error("memory_attn: d_model %d != %d",
			       D, SAM3_MULTIPLEX_HIDDEN_DIM);
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

	/* --- Build 2D axial RoPE tables --- */
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

	/* --- pos_enc_at_input=True: output = tgt + 0.1 * tgt_pos --- */
	struct sam3_tensor *output = tgt;
	if (tgt_pos) {
		int one[] = {1};
		struct sam3_tensor *scale = gh_alloc_tensor(arena,
			SAM3_DTYPE_F32, 1, one);
		if (!scale)
			return NULL;
		((float *)scale->data)[0] = MUX_POS_ENC_INPUT_SCALE;
		struct sam3_tensor *scaled = gh_mul(g, arena, tgt_pos, scale);
		if (!scaled)
			return NULL;
		output = gh_add(g, arena, tgt, scaled);
		if (!output)
			return NULL;
	}

	/* --- 4 layers --- */
	for (int i = 0; i < 4; i++) {
		output = memory_attn_layer(g, arena, &ma->layers[i],
			output, image, memory, memory_image,
			memory_image_pos, cos_q, sin_q, cos_k, sin_k,
			num_k_exclude_rope);
		if (!output)
			return NULL;
	}

	/* --- Final encoder.norm (use_image_in_output=False) --- */
	return gh_layernorm(g, arena, output,
			    ma->final_norm_w, ma->final_norm_b);
}

/*
 *  SAM 3.1 mask decoder forward (phase 2.4b) ────────────────────────
 *
 * MultiplexMaskDecoder.predict_masks + forward wrapper (Python
 * reference/sam3/sam3/model/multiplex_mask_decoder.py). The SAM 3.1
 * multiplex config is baked in: multiplex_count=16, num_multimask=3,
 * use_high_res_features=True, pred_obj_scores=True, ReLU inside the
 * two-way transformer MLP (TwoWayAttentionBlock default), GELU inside
 * the output upscaling stack, ReLU inside the MLP heads. We operate on
 * the canonical B=1 case (per-object processing); the multiplex-joint
 * forward in sub-project 4 will generalize to B>1 by looping.
 */

#define MUX_DEC_HIDDEN        256
#define MUX_DEC_N_HEADS_SELF  8
#define MUX_DEC_HEAD_DIM_SELF 32        /* 256 / 8 */
#define MUX_DEC_N_HEADS_XA    8
#define MUX_DEC_HEAD_DIM_XA   16        /* 128 / 8 */
#define MUX_DEC_N_TOKENS      80        /* 16 obj_score + 16 iou + 48 mask */
#define MUX_DEC_N_MASK_TOKENS 48        /* 16 * 3 */
#define MUX_DEC_N_MULTIMASK   3
#define MUX_DEC_UPSCALE_1C    64
#define MUX_DEC_UPSCALE_2C    32

/*
 * mha_sdpa_basic - 8-head SDPA with separate Q/K/V projections.
 *
 * Handles both the 256-dim self-attention (d_internal=256, head_dim=32)
 * and the 128-dim cross-attention (d_internal=128, head_dim=16) paths —
 * d_internal is inferred from the projection weight shape.
 *
 * q_src: [N_q, d_model]; k_src / v_src: [N_kv, d_model].
 * Returns [N_q, d_model].
 */
static struct sam3_tensor *mha_sdpa_basic(
		struct sam3_graph *g, struct sam3_arena *a,
		struct sam3_tensor *q_src, struct sam3_tensor *k_src,
		struct sam3_tensor *v_src,
		struct sam3_tensor *q_w, struct sam3_tensor *q_b,
		struct sam3_tensor *k_w, struct sam3_tensor *k_b,
		struct sam3_tensor *v_w, struct sam3_tensor *v_b,
		struct sam3_tensor *out_w, struct sam3_tensor *out_b,
		int n_heads)
{
	struct sam3_tensor *q = gh_linear(g, a, q_src, q_w, q_b);
	struct sam3_tensor *k = gh_linear(g, a, k_src, k_w, k_b);
	struct sam3_tensor *v = gh_linear(g, a, v_src, v_w, v_b);
	if (!q || !k || !v)
		return NULL;

	int N_q = q->dims[0];
	int N_k = k->dims[0];
	int D_int = q->dims[1];
	if (n_heads <= 0 || D_int % n_heads != 0) {
		sam3_log_error("mha_sdpa_basic: d_internal %d not divisible "
			       "by n_heads %d", D_int, n_heads);
		return NULL;
	}
	int head_dim = D_int / n_heads;

	int q4[] = {1, N_q, n_heads, head_dim};
	int k4[] = {1, N_k, n_heads, head_dim};
	int perm[] = {0, 2, 1, 3};
	int flat_q[] = {N_q, D_int};

	q = gh_reshape(g, a, q, 4, q4);
	k = gh_reshape(g, a, k, 4, k4);
	v = gh_reshape(g, a, v, 4, k4);
	if (!q || !k || !v)
		return NULL;

	q = gh_permute(g, a, q, perm);
	k = gh_permute(g, a, k, perm);
	v = gh_permute(g, a, v, perm);
	if (!q || !k || !v)
		return NULL;

	struct sam3_tensor *attn = gh_sdpa(g, a, q, k, v, NULL, head_dim);
	if (!attn)
		return NULL;

	attn = gh_permute(g, a, attn, perm);
	if (!attn)
		return NULL;
	attn = gh_reshape(g, a, attn, 2, flat_q);
	if (!attn)
		return NULL;

	return gh_linear(g, a, attn, out_w, out_b);
}

/*
 * mlp3_relu - Apply a 3-layer MLP with ReLU between layers (no activation
 * after the last layer). Matches the `MLP` class in
 * reference/sam3/sam3/model/multiplex_mask_decoder.py.
 *
 * @w/@b arrays must be 3 entries each. Output shape follows the last
 * linear's out_features.
 */
static struct sam3_tensor *mlp3_relu(
		struct sam3_graph *g, struct sam3_arena *a,
		struct sam3_tensor *x,
		struct sam3_tensor *w[3], struct sam3_tensor *b[3])
{
	struct sam3_tensor *h;
	h = gh_linear(g, a, x, w[0], b[0]);
	if (!h) return NULL;
	h = gh_relu(g, a, h);
	if (!h) return NULL;
	h = gh_linear(g, a, h, w[1], b[1]);
	if (!h) return NULL;
	h = gh_relu(g, a, h);
	if (!h) return NULL;
	return gh_linear(g, a, h, w[2], b[2]);
}

/*
 * materialize_mask_tokens_plus_extra - Pre-add extra_per_object to
 * mask_tokens at graph-build time.
 *
 * mask_tokens is [48, 256] = [16 slots × 3 multimask, 256]. extra is
 * [16, 256]. The Python reference broadcast-adds extra (unsqueezed to
 * [16, 1, 256]) over the 3 multimask tokens. Our gh_add only handles
 * [M, N] + [N] bias-style broadcast, not this 3-way duplicate, so we
 * pre-expand on the CPU side. This assumes extra_per_object has host
 * data available (the per-object multiplex controller feeds a static
 * tensor; when sub-project 4 lands, this can grow a graph-time path).
 */
static struct sam3_tensor *materialize_mask_tokens_plus_extra(
		struct sam3_arena *arena,
		struct sam3_tensor *mask_tokens,
		struct sam3_tensor *extra_per_object)
{
	if (extra_per_object->n_dims != 2 ||
	    extra_per_object->dims[0] != SAM3_MULTIPLEX_COUNT ||
	    extra_per_object->dims[1] != MUX_DEC_HIDDEN) {
		sam3_log_error("mask_decoder_multiplex: extra_per_object shape "
			       "%dD must be [16, 256]",
			       extra_per_object->n_dims);
		return NULL;
	}

	int dims[] = {MUX_DEC_N_MASK_TOKENS, MUX_DEC_HIDDEN};
	struct sam3_tensor *out = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						   2, dims);
	if (!out)
		return NULL;

	const float *mt = (const float *)mask_tokens->data;
	const float *ex = (const float *)extra_per_object->data;
	float *dst = (float *)out->data;
	for (int m = 0; m < SAM3_MULTIPLEX_COUNT; m++) {
		for (int k = 0; k < MUX_DEC_N_MULTIMASK; k++) {
			int row = m * MUX_DEC_N_MULTIMASK + k;
			for (int c = 0; c < MUX_DEC_HIDDEN; c++) {
				dst[row * MUX_DEC_HIDDEN + c] =
					mt[row * MUX_DEC_HIDDEN + c] +
					ex[m * MUX_DEC_HIDDEN + c];
			}
		}
	}
	return out;
}

/*
 * two_way_block - One TwoWayAttentionBlock forward pass (Python
 * reference/sam3/sam3/sam/transformer.py:110-183).
 *
 * Updates @queries (by value through the returned pointer) and @keys (by
 * the p_keys out-param). query_pe / key_pe are the static positional
 * encodings added to Q/K before each attention. skip_pe=1 for layer 0
 * (no PE and no residual in self-attn).
 */
static struct sam3_tensor *two_way_block(
		struct sam3_graph *g, struct sam3_arena *a,
		const struct sam3_multiplex_mask_decoder_layer *L,
		struct sam3_tensor *queries,
		struct sam3_tensor **p_keys,
		struct sam3_tensor *query_pe,
		struct sam3_tensor *key_pe,
		int skip_pe)
{
	struct sam3_tensor *keys = *p_keys;
	struct sam3_tensor *q, *k, *attn, *mlp;

	/* --- Self-attention on tokens  --- */
	if (skip_pe) {
		attn = mha_sdpa_basic(g, a, queries, queries, queries,
			L->self_q_w, L->self_q_b, L->self_k_w, L->self_k_b,
			L->self_v_w, L->self_v_b,
			L->self_out_w, L->self_out_b, MUX_DEC_N_HEADS_SELF);
		if (!attn) return NULL;
		queries = attn;   /* no residual */
	} else {
		q = gh_add(g, a, queries, query_pe);
		if (!q) return NULL;
		attn = mha_sdpa_basic(g, a, q, q, queries,
			L->self_q_w, L->self_q_b, L->self_k_w, L->self_k_b,
			L->self_v_w, L->self_v_b,
			L->self_out_w, L->self_out_b, MUX_DEC_N_HEADS_SELF);
		if (!attn) return NULL;
		queries = gh_add(g, a, queries, attn);
		if (!queries) return NULL;
	}
	queries = gh_layernorm(g, a, queries, L->norm1_w, L->norm1_b);
	if (!queries) return NULL;

	/* --- Cross-attention: tokens attend to image --- */
	q = gh_add(g, a, queries, query_pe);
	k = gh_add(g, a, keys, key_pe);
	if (!q || !k) return NULL;
	attn = mha_sdpa_basic(g, a, q, k, keys,
		L->ct2i_q_w, L->ct2i_q_b, L->ct2i_k_w, L->ct2i_k_b,
		L->ct2i_v_w, L->ct2i_v_b,
		L->ct2i_out_w, L->ct2i_out_b, MUX_DEC_N_HEADS_XA);
	if (!attn) return NULL;
	queries = gh_add(g, a, queries, attn);
	if (!queries) return NULL;
	queries = gh_layernorm(g, a, queries, L->norm2_w, L->norm2_b);
	if (!queries) return NULL;

	/* --- MLP on tokens (lin1 → ReLU → lin2) --- */
	mlp = gh_linear(g, a, queries, L->mlp_lin1_w, L->mlp_lin1_b);
	if (!mlp) return NULL;
	mlp = gh_relu(g, a, mlp);
	if (!mlp) return NULL;
	mlp = gh_linear(g, a, mlp, L->mlp_lin2_w, L->mlp_lin2_b);
	if (!mlp) return NULL;
	queries = gh_add(g, a, queries, mlp);
	if (!queries) return NULL;
	queries = gh_layernorm(g, a, queries, L->norm3_w, L->norm3_b);
	if (!queries) return NULL;

	/* --- Cross-attention: image attends to tokens  *
	 * Python calls self.cross_attn_image_to_token(q=k, k=q, v=queries)
	 * where `q` and `k` are local Python variables (q=queries+pe,
	 * k=keys+pe). That swap means the actual Q input is image+PE and
	 * actual K input is tokens+PE, actual V is tokens. */
	q = gh_add(g, a, queries, query_pe);  /* tokens + pe */
	k = gh_add(g, a, keys, key_pe);        /* image + pe */
	if (!q || !k) return NULL;
	attn = mha_sdpa_basic(g, a, k, q, queries,
		L->ci2t_q_w, L->ci2t_q_b, L->ci2t_k_w, L->ci2t_k_b,
		L->ci2t_v_w, L->ci2t_v_b,
		L->ci2t_out_w, L->ci2t_out_b, MUX_DEC_N_HEADS_XA);
	if (!attn) return NULL;
	keys = gh_add(g, a, keys, attn);
	if (!keys) return NULL;
	keys = gh_layernorm(g, a, keys, L->norm4_w, L->norm4_b);
	if (!keys) return NULL;

	*p_keys = keys;
	return queries;
}

/*
 * Error-code convention for the graph-build paths below. `gh_*` helpers
 * return NULL on any failure — arena exhaustion, graph-capacity overflow,
 * or an internal shape-validation failure — and always log the specific
 * cause via `sam3_log_error`. We collapse all of these to SAM3_ENOMEM at
 * the caller boundary because (a) caller-side input shapes are validated
 * at the top of the forward (so a shape mismatch downstream implies an
 * internal bug rather than a user error), and (b) the common production
 * failure mode is genuine resource exhaustion. Consult the error log for
 * diagnosis.
 */
enum sam3_error sam3_multiplex_mask_decoder_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_multiplex_mask_decoder *md,
		struct sam3_tensor *image_embeddings,
		struct sam3_tensor *image_pe,
		struct sam3_tensor *feat_s1,
		struct sam3_tensor *feat_s0,
		struct sam3_tensor *extra_per_object,
		struct sam3_tensor **out_masks,
		struct sam3_tensor **out_iou_scores,
		struct sam3_tensor **out_obj_score_logits,
		struct sam3_tensor **out_sam_tokens)
{
	if (!g || !arena || !md || !image_embeddings || !image_pe
	    || !feat_s1 || !feat_s0
	    || !out_masks || !out_iou_scores || !out_obj_score_logits
	    || !out_sam_tokens) {
		sam3_log_error("mask_decoder_multiplex: NULL argument");
		return SAM3_EINVAL;
	}
	if (image_embeddings->n_dims != 4 ||
	    image_embeddings->dims[0] != 1 ||
	    image_embeddings->dims[3] != MUX_DEC_HIDDEN) {
		sam3_log_error("mask_decoder_multiplex: image_embeddings must be "
			       "[1, H, W, 256] NHWC");
		return SAM3_EINVAL;
	}

	int H = image_embeddings->dims[1];
	int W = image_embeddings->dims[2];
	int HW = H * W;

	if (image_pe->n_dims != 2 ||
	    image_pe->dims[0] != HW ||
	    image_pe->dims[1] != MUX_DEC_HIDDEN) {
		sam3_log_error("mask_decoder_multiplex: image_pe must be [H*W, 256]");
		return SAM3_EINVAL;
	}
	if (feat_s1->n_dims != 4 ||
	    feat_s1->dims[0] != 1 || feat_s1->dims[1] != 2 * H ||
	    feat_s1->dims[2] != 2 * W ||
	    feat_s1->dims[3] != MUX_DEC_HIDDEN) {
		sam3_log_error("mask_decoder_multiplex: feat_s1 must be "
			       "[1, 2H, 2W, 256] NHWC");
		return SAM3_EINVAL;
	}
	if (feat_s0->n_dims != 4 ||
	    feat_s0->dims[0] != 1 || feat_s0->dims[1] != 4 * H ||
	    feat_s0->dims[2] != 4 * W ||
	    feat_s0->dims[3] != MUX_DEC_HIDDEN) {
		sam3_log_error("mask_decoder_multiplex: feat_s0 must be "
			       "[1, 4H, 4W, 256] NHWC");
		return SAM3_EINVAL;
	}

	/* --- 1. Prepare tokens  --- */
	struct sam3_tensor *mask_tokens = md->mask_tokens;
	if (extra_per_object) {
		mask_tokens = materialize_mask_tokens_plus_extra(
				arena, md->mask_tokens, extra_per_object);
		if (!mask_tokens)
			return SAM3_ENOMEM;
	}

	struct sam3_tensor *parts[3] = {
		md->obj_score_token,   /* [16, 256] */
		md->iou_token,         /* [16, 256] */
		mask_tokens,           /* [48, 256] */
	};
	struct sam3_tensor *tokens = gh_concat(g, arena, parts, 3, 0);
	if (!tokens) return SAM3_ENOMEM;

	/* --- 2. Flatten image embeddings into [H*W, 256] as keys --- */
	int keys_dims[] = {HW, MUX_DEC_HIDDEN};
	struct sam3_tensor *keys = gh_reshape(g, arena,
			image_embeddings, 2, keys_dims);
	if (!keys) return SAM3_ENOMEM;

	struct sam3_tensor *queries = tokens;
	struct sam3_tensor *query_pe = tokens;
	struct sam3_tensor *key_pe = image_pe;

	/* --- 3. Two-way transformer: 2 blocks --- */
	for (int i = 0; i < 2; i++) {
		queries = two_way_block(g, arena, &md->layers[i],
					queries, &keys, query_pe, key_pe,
					/*skip_pe=*/(i == 0));
		if (!queries) return SAM3_ENOMEM;
	}

	/* --- 4. Final token→image attention + norm --- */
	{
		struct sam3_tensor *q = gh_add(g, arena, queries, query_pe);
		struct sam3_tensor *k = gh_add(g, arena, keys, key_pe);
		if (!q || !k) return SAM3_ENOMEM;
		struct sam3_tensor *fa = mha_sdpa_basic(g, arena, q, k, keys,
			md->final_q_w, md->final_q_b,
			md->final_k_w, md->final_k_b,
			md->final_v_w, md->final_v_b,
			md->final_out_w, md->final_out_b,
			MUX_DEC_N_HEADS_XA);
		if (!fa) return SAM3_ENOMEM;
		queries = gh_add(g, arena, queries, fa);
		if (!queries) return SAM3_ENOMEM;
		queries = gh_layernorm(g, arena, queries,
				md->norm_final_w, md->norm_final_b);
		if (!queries) return SAM3_ENOMEM;
	}

	/* --- 5. Slice output tokens  --- */
	struct sam3_tensor *obj_score_tok = gh_slice(g, arena, queries, 0,
			0, SAM3_MULTIPLEX_COUNT);
	struct sam3_tensor *iou_tok = gh_slice(g, arena, queries, 0,
			SAM3_MULTIPLEX_COUNT,
			2 * SAM3_MULTIPLEX_COUNT);
	struct sam3_tensor *mask_tok_out = gh_slice(g, arena, queries, 0,
			2 * SAM3_MULTIPLEX_COUNT,
			MUX_DEC_N_TOKENS);
	if (!obj_score_tok || !iou_tok || !mask_tok_out)
		return SAM3_ENOMEM;

	/* --- 6. Upscaling: dc1(src)+s1 → ln → gelu → dc2(up)+s0 → gelu --- */
	int src_4d[] = {1, H, W, MUX_DEC_HIDDEN};
	struct sam3_tensor *src = gh_reshape(g, arena, keys, 4, src_4d);
	if (!src) return SAM3_ENOMEM;

	struct sam3_tensor *s1_proj = gh_conv2d(g, arena, feat_s1,
			md->conv_s1_w, md->conv_s1_b, 1, 0, 1);
	struct sam3_tensor *s0_proj = gh_conv2d(g, arena, feat_s0,
			md->conv_s0_w, md->conv_s0_b, 1, 0, 1);
	if (!s1_proj || !s0_proj) return SAM3_ENOMEM;

	struct sam3_tensor *up = gh_conv_transpose2d(g, arena, src,
			md->up0_w, md->up0_b, 2, 0);
	if (!up) return SAM3_ENOMEM;
	up = gh_add(g, arena, up, s1_proj);
	if (!up) return SAM3_ENOMEM;

	/* LayerNorm2d in NHWC is LN over the last (C) axis: reshape to 2D
	 * flat, run gh_layernorm, reshape back to NHWC. */
	int H2 = 2 * H, W2 = 2 * W;
	int up1_flat[] = {H2 * W2, MUX_DEC_UPSCALE_1C};
	int up1_nhwc[] = {1, H2, W2, MUX_DEC_UPSCALE_1C};
	up = gh_reshape(g, arena, up, 2, up1_flat);
	if (!up) return SAM3_ENOMEM;
	up = gh_layernorm(g, arena, up, md->up1_w, md->up1_b);
	if (!up) return SAM3_ENOMEM;
	up = gh_reshape(g, arena, up, 4, up1_nhwc);
	if (!up) return SAM3_ENOMEM;
	up = gh_gelu(g, arena, up);
	if (!up) return SAM3_ENOMEM;

	up = gh_conv_transpose2d(g, arena, up, md->up3_w, md->up3_b, 2, 0);
	if (!up) return SAM3_ENOMEM;
	up = gh_add(g, arena, up, s0_proj);
	if (!up) return SAM3_ENOMEM;
	up = gh_gelu(g, arena, up);
	if (!up) return SAM3_ENOMEM;
	/* up shape: [1, 4H, 4W, 32] */

	/* --- 7. Hypernetwork MLPs  --- */
	int mt_3d[] = {SAM3_MULTIPLEX_COUNT, MUX_DEC_N_MULTIMASK,
		       MUX_DEC_HIDDEN};
	mask_tok_out = gh_reshape(g, arena, mask_tok_out, 3, mt_3d);
	if (!mask_tok_out) return SAM3_ENOMEM;

	struct sam3_tensor *hyper_parts[MUX_DEC_N_MULTIMASK];
	for (int i = 0; i < MUX_DEC_N_MULTIMASK; i++) {
		struct sam3_tensor *slc = gh_slice(g, arena, mask_tok_out,
				1, i, i + 1);
		if (!slc) return SAM3_ENOMEM;
		int slc_2d[] = {SAM3_MULTIPLEX_COUNT, MUX_DEC_HIDDEN};
		slc = gh_reshape(g, arena, slc, 2, slc_2d);
		if (!slc) return SAM3_ENOMEM;

		struct sam3_tensor *hn_w[3] = {
			md->hn_w[i][0], md->hn_w[i][1], md->hn_w[i][2]
		};
		struct sam3_tensor *hn_b[3] = {
			md->hn_b[i][0], md->hn_b[i][1], md->hn_b[i][2]
		};
		struct sam3_tensor *h = mlp3_relu(g, arena, slc, hn_w, hn_b);
		if (!h) return SAM3_ENOMEM;
		/* h: [16, 32] → reshape to [16, 1, 32] for later concat */
		int r3[] = {SAM3_MULTIPLEX_COUNT, 1, MUX_DEC_UPSCALE_2C};
		hyper_parts[i] = gh_reshape(g, arena, h, 3, r3);
		if (!hyper_parts[i]) return SAM3_ENOMEM;
	}

	struct sam3_tensor *hyper_in = gh_concat(g, arena, hyper_parts,
						  MUX_DEC_N_MULTIMASK, 1);
	if (!hyper_in) return SAM3_ENOMEM;
	/* hyper_in: [16, 3, 32] → flatten to [48, 32] for matmul */
	int hi_2d[] = {MUX_DEC_N_MASK_TOKENS, MUX_DEC_UPSCALE_2C};
	hyper_in = gh_reshape(g, arena, hyper_in, 2, hi_2d);
	if (!hyper_in) return SAM3_ENOMEM;

	/* --- 8. Mask logits: hyper_in @ upscaled^T --- */
	int H4 = 4 * H, W4 = 4 * W;
	int up_2d[] = {H4 * W4, MUX_DEC_UPSCALE_2C};
	up = gh_reshape(g, arena, up, 2, up_2d);
	if (!up) return SAM3_ENOMEM;
	struct sam3_tensor *up_t = gh_transpose(g, arena, up);
	if (!up_t) return SAM3_ENOMEM;

	struct sam3_tensor *masks_flat = gh_matmul(g, arena, hyper_in, up_t);
	if (!masks_flat) return SAM3_ENOMEM;
	int mask_4d[] = {SAM3_MULTIPLEX_COUNT, MUX_DEC_N_MULTIMASK,
			 H4, W4};
	struct sam3_tensor *masks = gh_reshape(g, arena, masks_flat,
						4, mask_4d);
	if (!masks) return SAM3_ENOMEM;

	/* --- 9. IoU head (256→256→256→3, ReLU between) --- */
	struct sam3_tensor *iou_w[3] = {md->iou_head_w[0],
					md->iou_head_w[1], md->iou_head_w[2]};
	struct sam3_tensor *iou_b[3] = {md->iou_head_b[0],
					md->iou_head_b[1], md->iou_head_b[2]};
	struct sam3_tensor *iou = mlp3_relu(g, arena, iou_tok, iou_w, iou_b);
	if (!iou) return SAM3_ENOMEM;

	/* --- 10. Object score head (256→256→256→1, ReLU between) --- */
	struct sam3_tensor *score_w[3] = {md->score_head_w[0],
					  md->score_head_w[1],
					  md->score_head_w[2]};
	struct sam3_tensor *score_b[3] = {md->score_head_b[0],
					  md->score_head_b[1],
					  md->score_head_b[2]};
	struct sam3_tensor *obj_logits = mlp3_relu(g, arena, obj_score_tok,
						   score_w, score_b);
	if (!obj_logits) return SAM3_ENOMEM;

	*out_masks = masks;
	*out_iou_scores = iou;
	*out_obj_score_logits = obj_logits;
	*out_sam_tokens = mask_tok_out;  /* [16, 3, 256] */
	return SAM3_OK;
}

/*
 * sam3_multiplex_image_pe_layer - Host-side implementation of the Gaussian PE
 *                          basis lookup.
 *
 * Pure CPU compute: basis is a small [2, 128] weight (mmap-resident f32)
 * and the output is only [H*W, 256] f32 (~640 KiB at H=W=72), so there's
 * no benefit to putting sin/cos + matmul into the graph. The result
 * tensor has its data buffer populated directly and can then feed
 * downstream graph operations as an input.
 */
struct sam3_tensor *sam3_multiplex_image_pe_layer(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		struct sam3_tensor *basis,
		int grid_h,
		int grid_w)
{
	(void)g;  /* pure host compute, no graph ops */

	if (!arena || !basis || grid_h <= 0 || grid_w <= 0) {
		sam3_log_error("image_pe_layer: bad args");
		return NULL;
	}
	if (basis->dtype != SAM3_DTYPE_F32 || !basis->data) {
		sam3_log_error("image_pe_layer: basis dtype/data missing");
		return NULL;
	}
	if (basis->n_dims != 2 || basis->dims[0] != 2 ||
	    basis->dims[1] != 128) {
		sam3_log_error("image_pe_layer: basis must be [2, 128]");
		return NULL;
	}

	const int HW = grid_h * grid_w;
	const int D  = 128;        /* half of 256 */
	int out_dims[2] = {HW, 2 * D};
	struct sam3_tensor *pe = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						 2, out_dims);
	if (!pe)
		return NULL;

	const float *B = (const float *)basis->data;  /* [2, 128] row-major */
	float *dst = (float *)pe->data;
	const float two_pi = 6.28318530717958647692f;

	for (int y = 0; y < grid_h; y++) {
		float ny = ((float)y + 0.5f) / (float)grid_h;
		float sy = 2.0f * ny - 1.0f;
		for (int x = 0; x < grid_w; x++) {
			float nx = ((float)x + 0.5f) / (float)grid_w;
			float sx = 2.0f * nx - 1.0f;
			float *row = dst + (size_t)(y * grid_w + x) * 2 * D;
			for (int k = 0; k < D; k++) {
				/* proj_k = sx * B[0, k] + sy * B[1, k] */
				float v = sx * B[0 * D + k]
					+ sy * B[1 * D + k];
				v *= two_pi;
				row[k]       = sinf(v);
				row[D + k]   = cosf(v);
			}
		}
	}

	return pe;
}

/*
 * Cap on the number of most-recent bank entries fed into memory
 * attention. Each entry contributes HW=5184 spatial tokens at the
 * production 72×72 grid, so capping keeps the memory-attention K/V
 * tensors manageable (2 × 5184 = 10368 tokens fit comfortably in a
 * 3 GiB scratch arena).
 */
#define MUX_MAX_MEM_ENTRIES_IN_ATTN 2

/*
 * multiplex_apply_linear_256 - Host-side y = x @ W^T + b for single-vector linear.
 *
 * Broadcasts over N contiguous 256-D rows of @src → @dst. Used for the
 * obj_ptr temporal positional projection where we want to write the
 * result straight into a specific row of memory_image_pos without a
 * graph round-trip.
 */
void multiplex_apply_linear_256(float *dst, const float *src, int n_rows,
			 const float *W, const float *b)
{
	const int D = SAM3_MULTIPLEX_HIDDEN_DIM;
	for (int r = 0; r < n_rows; r++) {
		const float *x = src + (size_t)r * D;
		float *y = dst + (size_t)r * D;
		for (int i = 0; i < D; i++) {
			float acc = b ? b[i] : 0.0f;
			const float *wrow = W + (size_t)i * D;
			for (int j = 0; j < D; j++)
				acc += x[j] * wrow[j];
			y[i] = acc;
		}
	}
}

/*
 * multiplex_sine_tpos_256 - Build a 1D sine/cos temporal positional encoding
 * at `norm_pos` into a 256-D row.
 *
 * Matches get_1d_sine_pe (reference/sam3/sam3/model/sam3_tracker_utils.py):
 *   first half: sin(norm_pos / temperature^(2*(j/2)/128))
 *   second half: cos(norm_pos / temperature^(2*(j/2)/128))
 */
void multiplex_sine_tpos_256(float *row, float norm_pos)
{
	const int pe_dim = SAM3_MULTIPLEX_HIDDEN_DIM / 2;  /* 128 */
	const float temperature = 10000.0f;
	for (int j = 0; j < pe_dim; j++) {
		float exponent = (float)(2 * (j / 2)) / (float)pe_dim;
		float dim_t = powf(temperature, exponent);
		float v = norm_pos / dim_t;
		row[j]          = sinf(v);
		row[pe_dim + j] = cosf(v);
	}
}

/*
 * multiplex_maskmem_tpos_slot - Python's use_maskmem_tpos_v2 rule:
 *   idx = num_maskmem - t_rel - 1                    if 0 < t_rel < num_maskmem
 *   idx = num_maskmem - 1                            otherwise (out of range)
 *
 * `t_rel` is the absolute frame distance |current - entry.frame_idx|.
 * This is a simplification of Python's sign-aware t_pos but matches the
 * v2 formula whenever current > entry (the common tracking-forward case)
 * and degrades gracefully to the out-of-range slot otherwise.
 *
 * TODO(sub-project 4 / reverse tracking): Python uses
 *   t_pos = (frame_idx - prev_frame_idx) * tpos_sign_mul
 * where tpos_sign_mul = -1 when track_in_reverse=True. With SAM3_PROPAGATE_
 * REVERSE or SAM3_PROPAGATE_BOTH and prev_frame_idx > current_frame_idx,
 * our abs() collapses what Python would keep signed, so we degrade to the
 * out-of-range slot (num_maskmem - 1) for all backward entries instead of
 * using maskmem_tpos_enc[num_maskmem - t_pos - 1] with negative t_pos. Fix
 * when reverse-direction parity becomes load-bearing.
 */
int multiplex_maskmem_tpos_slot(int t_rel)
{
	const int N = SAM3_MULTIPLEX_NUM_MASKMEM;  /* 7 */
	if (t_rel <= 0 || t_rel >= N)
		return N - 1;
	return N - t_rel - 1;
}

/*
 * multiplex_build_memory_from_bank - Materialise [1, Nm, 256] memory, memory_image
 * and memory_image_pos tensors from the N most-recent entries of a per-
 * object bank.
 *
 * Layout:
 *   rows [0 .. total_spatial):
 *     memory[r]           = entry.spatial_features (maskmem output)
 *     memory_image[r]     = entry.image_features   (raw backbone 1x feats)
 *     memory_image_pos[r] = image_pe[r_in_frame] + maskmem_tpos_enc[slot]
 *   rows [total_spatial .. Nm):
 *     memory[r]           = entry.obj_pointer
 *     memory_image[r]     = 0   (image K side gets zero contribution here)
 *     memory_image_pos[r] = obj_ptr_tpos_proj(sine_pe(t_diff))
 *
 * Mirrors Python video_tracking_multiplex._prepare_memory_conditioned_features
 * + decoder.TransformerEncoderDecoupledCrossAttention.forward: memory_image_pos
 * is padded to match memory by copying the obj_ptr tail from memory_pos so a
 * single K = k_img + k_mem + memory_image_pos is correct.
 *
 * Returns num_k_exclude_rope (= number of obj_ptr rows) on success, or
 * -1 on failure.
 */
static int multiplex_build_memory_from_bank(
		struct sam3_arena *arena,
		const struct sam3_tracker_multiplex *trk,
		const struct sam3_memory_bank *bank,
		int current_frame_idx,
		const struct sam3_tensor *image_pe,
		struct sam3_tensor **out_memory,
		struct sam3_tensor **out_memory_image,
		struct sam3_tensor **out_memory_image_pos,
		int *out_Nm)
{
	const struct sam3_memory_entry *picks[MUX_MAX_MEM_ENTRIES_IN_ATTN];
	int n_pick = 0;

	/* Walk newest-first across non_cond (end of ring) then cond. */
	for (int i = bank->n_non_cond - 1;
	     i >= 0 && n_pick < MUX_MAX_MEM_ENTRIES_IN_ATTN; i--) {
		if (bank->non_cond[i].spatial_features)
			picks[n_pick++] = &bank->non_cond[i];
	}
	for (int i = bank->n_cond - 1;
	     i >= 0 && n_pick < MUX_MAX_MEM_ENTRIES_IN_ATTN; i--) {
		if (bank->cond[i].spatial_features)
			picks[n_pick++] = &bank->cond[i];
	}
	if (n_pick == 0) {
		*out_memory = NULL;
		*out_memory_image = NULL;
		*out_memory_image_pos = NULL;
		*out_Nm = 0;
		return 0;
	}

	const int D  = SAM3_MULTIPLEX_HIDDEN_DIM;
	const int HW = image_pe ? image_pe->dims[0] : -1;
	int total_spatial = 0;
	int total_obj_ptrs = 0;
	for (int i = 0; i < n_pick; i++) {
		if (picks[i]->spatial_features->dims[1] != D)
			return -1;
		if (HW > 0 && picks[i]->spatial_features->dims[0] != HW)
			return -1;
		total_spatial += picks[i]->spatial_features->dims[0];
		if (picks[i]->obj_pointer)
			total_obj_ptrs++;
	}
	int Nm = total_spatial + total_obj_ptrs;

	int dims[3] = {1, Nm, D};
	struct sam3_tensor *memory = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						     3, dims);
	struct sam3_tensor *memory_image = gh_alloc_tensor(arena,
							   SAM3_DTYPE_F32,
							   3, dims);
	struct sam3_tensor *memory_image_pos = gh_alloc_tensor(
		arena, SAM3_DTYPE_F32, 3, dims);
	if (!memory || !memory_image || !memory_image_pos)
		return -1;
	memset(memory->data, 0, memory->nbytes);
	memset(memory_image->data, 0, memory_image->nbytes);
	memset(memory_image_pos->data, 0, memory_image_pos->nbytes);

	float *mp  = (float *)memory->data;
	float *mip = (float *)memory_image->data;
	float *mpp = (float *)memory_image_pos->data;
	const float *pe = image_pe ? (const float *)image_pe->data : NULL;
	const float *tpos_base = trk->maskmem_tpos_enc
		? (const float *)trk->maskmem_tpos_enc->data
		: NULL;

	size_t row = 0;

	/* One-shot warn across the whole function call when the fallback
	 * fires. The fallback is reachable in principle (a SAM 3 memory
	 * entry reused by a SAM 3.1 session) but shouldn't hit in
	 * production — sam3_video.c clones cf.feat_s1 into every SAM 3.1
	 * bank entry at commit time. If this warning prints, a stale
	 * entry leaked in. */
	int warned_missing_image_feat = 0;

	/* Pass 1: spatial rows for each picked frame. */
	for (int i = 0; i < n_pick; i++) {
		const struct sam3_tensor *sf = picks[i]->spatial_features;
		const struct sam3_tensor *ife = picks[i]->image_features;
		int n_rows = sf->dims[0];
		size_t base = row * D;
		memcpy(mp + base, sf->data,
		       (size_t)n_rows * D * sizeof(float));
		if (ife && ife->data && ife->dims[0] == n_rows
		    && ife->dims[1] == D) {
			memcpy(mip + base, ife->data,
			       (size_t)n_rows * D * sizeof(float));
		} else {
			/* Older SAM 3 entries may lack image_features; fall
			 * back to spatial_features so the K-side addition is
			 * at least coherent. This should not hit on SAM 3.1
			 * in practice because sam3_video clones feat_s1 at
			 * commit time. */
			if (!warned_missing_image_feat) {
				sam3_log_warn("multiplex_build_memory_from_bank: "
					      "entry frame %d missing "
					      "image_features; falling back "
					      "to maskmem spatial features "
					      "(stale SAM 3 entry?)",
					      picks[i]->frame_idx);
				warned_missing_image_feat = 1;
			}
			memcpy(mip + base, sf->data,
			       (size_t)n_rows * D * sizeof(float));
		}

		/* memory_image_pos: image_pe + maskmem_tpos_enc[slot]. */
		int t_rel = current_frame_idx - picks[i]->frame_idx;
		if (t_rel < 0)
			t_rel = -t_rel;
		int slot = multiplex_maskmem_tpos_slot(t_rel);
		const float *tpos = tpos_base
			? tpos_base + (size_t)slot * D
			: NULL;
		for (int r = 0; r < n_rows; r++) {
			float *dst = mpp + base + (size_t)r * D;
			const float *pe_row = pe
				? pe + (size_t)r * D
				: NULL;
			for (int c = 0; c < D; c++) {
				float v = pe_row ? pe_row[c] : 0.0f;
				if (tpos) v += tpos[c];
				dst[c] = v;
			}
		}
		row += n_rows;
	}

	/* Pass 2: obj_ptr rows — memory gets pointer, image zeros,
	 * memory_image_pos gets obj_ptr_tpos_proj(sine_pe(t_diff)). */
	const float *W = trk->obj_ptr_tpos_proj_w
		? (const float *)trk->obj_ptr_tpos_proj_w->data : NULL;
	const float *B = trk->obj_ptr_tpos_proj_b
		? (const float *)trk->obj_ptr_tpos_proj_b->data : NULL;
	/* Normalization denominator from Python: max_obj_ptrs_in_encoder - 1.
	 * SAM 3.1 uses 16 obj_ptrs in encoder (model_builder default). */
	const float max_obj_ptrs_in_enc = 16.0f;
	const float t_diff_max = max_obj_ptrs_in_enc - 1.0f;
	float sine_row[SAM3_MULTIPLEX_HIDDEN_DIM];
	for (int i = 0; i < n_pick; i++) {
		if (!picks[i]->obj_pointer)
			continue;
		const struct sam3_tensor *op = picks[i]->obj_pointer;
		int n_elems = sam3_tensor_nelems(op);
		if (n_elems < D)
			continue;
		size_t base = row * D;
		memcpy(mp + base, op->data, D * sizeof(float));
		/* memory_image row already zero from memset. */

		if (W && B) {
			int t_rel = current_frame_idx - picks[i]->frame_idx;
			float norm_pos = (t_rel < 0 ? -(float)t_rel : (float)t_rel)
				/ t_diff_max;
			multiplex_sine_tpos_256(sine_row, norm_pos);
			multiplex_apply_linear_256(mpp + base, sine_row, 1, W, B);
		}
		row++;
	}

	*out_memory = memory;
	*out_memory_image = memory_image;
	*out_memory_image_pos = memory_image_pos;
	*out_Nm = Nm;
	return total_obj_ptrs;
}

enum sam3_error sam3_tracker_multiplex_track_frame(
		struct sam3_tracker_multiplex *trk,
		struct sam3_graph *g,
		const struct sam3_memory_bank *bank,
		struct sam3_tensor *image_embed,
		struct sam3_tensor *feat_s1,
		struct sam3_tensor *feat_s0,
		int frame_idx,
		int is_cond,
		struct sam3_arena *arena,
		struct sam3_tensor **out_masks,
		struct sam3_tensor **out_iou,
		struct sam3_tensor **out_obj_ptrs,
		struct sam3_tensor **out_score)
{
	if (!trk || !g || !image_embed || !feat_s1 || !feat_s0 || !arena
	    || !out_masks || !out_iou || !out_obj_ptrs || !out_score) {
		sam3_log_error("tracker_multiplex_track_frame: NULL arg");
		return SAM3_EINVAL;
	}
	if (image_embed->n_dims != 4 || image_embed->dims[0] != 1 ||
	    image_embed->dims[3] != SAM3_MULTIPLEX_HIDDEN_DIM ||
	    image_embed->dtype != SAM3_DTYPE_F32) {
		sam3_log_error("tracker_multiplex_track_frame: image_embed must be "
			       "[1,H,W,256] F32");
		return SAM3_EINVAL;
	}
	if (!trk->interactivity_no_mem_embed ||
	    !trk->image_pe_gauss) {
		sam3_log_error("tracker_multiplex_track_frame: tracker weights missing");
		return SAM3_EINVAL;
	}

	const int H = image_embed->dims[1];
	const int W = image_embed->dims[2];
	const int HW = H * W;
	const int D = SAM3_MULTIPLEX_HIDDEN_DIM;

	/*
	 * High-res features must match the mask decoder's contract:
	 * feat_s1 is the 2x scale, feat_s0 is the 4x scale. Catch swaps at
	 * the API boundary instead of deep inside sam3_multiplex_mask_decoder_forward
	 * — the convention inverts SAM 3's naming and is easy to miswire.
	 */
	if (feat_s1->n_dims != 4 || feat_s1->dims[0] != 1 ||
	    feat_s1->dims[1] != 2 * H || feat_s1->dims[2] != 2 * W ||
	    feat_s1->dims[3] != D) {
		sam3_log_error("tracker_multiplex_track_frame: feat_s1 must be "
			       "[1,%d,%d,%d] NHWC (2x scale)",
			       2 * H, 2 * W, D);
		return SAM3_EINVAL;
	}
	if (feat_s0->n_dims != 4 || feat_s0->dims[0] != 1 ||
	    feat_s0->dims[1] != 4 * H || feat_s0->dims[2] != 4 * W ||
	    feat_s0->dims[3] != D) {
		sam3_log_error("tracker_multiplex_track_frame: feat_s0 must be "
			       "[1,%d,%d,%d] NHWC (4x scale)",
			       4 * H, 4 * W, D);
		return SAM3_EINVAL;
	}

	sam3_log_debug("tracker_multiplex_track_frame: frame=%d is_cond=%d HW=%d "
		       "bank_entries=%d",
		       frame_idx, is_cond, HW,
		       bank ? sam3_memory_bank_total(bank) : 0);

	struct sam3_tensor *pix_feat_with_mem = NULL;

	/*
	 * Dense positional encoding is deterministic in (H, W, basis) and
	 * is consumed by both the memory attention (as K-side memory_image_pos
	 * spatial rows) and the mask decoder. Compute it once up front and
	 * share.
	 */
	struct sam3_tensor *image_pe = sam3_multiplex_image_pe_layer(
		g, arena, trk->image_pe_gauss, H, W);
	if (!image_pe)
		return SAM3_ENOMEM;

	int total_entries = bank ? sam3_memory_bank_total(bank) : 0;
	int use_memory_attn = (bank != NULL) && (total_entries > 0);

	if (!use_memory_attn) {
		/*
		 * No memory: compute pix_feat + interactivity_no_mem_embed
		 * on the host side. image_embed is materialised (came from
		 * the frame cache) so this is a straight CPU pass.
		 */
		int dims[4] = {1, H, W, D};
		pix_feat_with_mem = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						    4, dims);
		if (!pix_feat_with_mem)
			return SAM3_ENOMEM;
		const float *src = (const float *)image_embed->data;
		const float *add = (const float *)
			trk->interactivity_no_mem_embed->data;  /* [1,1,256] */
		float *dst = (float *)pix_feat_with_mem->data;
		for (int i = 0; i < HW; i++) {
			for (int c = 0; c < D; c++)
				dst[i * D + c] = src[i * D + c] + add[c];
		}
	} else {
		/*
		 * Memory path: flatten image_embed to [1, HW, 256], build
		 * memory inputs from the bank (with per-entry maskmem_tpos_enc
		 * and obj_ptr_tpos_proj position encodings on memory_image_pos),
		 * run the 4-layer decoupled transformer, reshape back to
		 * [1, H, W, 256]. Mirrors Python:
		 *   video_tracking_multiplex._prepare_memory_conditioned_features
		 */
		int tgt_dims[3] = {1, HW, D};
		struct sam3_tensor *tgt = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
							  3, tgt_dims);
		if (!tgt)
			return SAM3_ENOMEM;
		memcpy(tgt->data, image_embed->data,
		       (size_t)HW * D * sizeof(float));

		struct sam3_tensor *memory = NULL;
		struct sam3_tensor *memory_image = NULL;
		struct sam3_tensor *memory_image_pos = NULL;
		int Nm = 0;
		int num_k_exclude = multiplex_build_memory_from_bank(
			arena, trk, bank, frame_idx, image_pe,
			&memory, &memory_image, &memory_image_pos, &Nm);
		if (num_k_exclude < 0 || !memory || !memory_image || Nm == 0) {
			sam3_log_warn("tracker_multiplex_track_frame: memory build "
				      "failed, falling back to no_mem path");
			int dims[4] = {1, H, W, D};
			pix_feat_with_mem = gh_alloc_tensor(arena,
							    SAM3_DTYPE_F32,
							    4, dims);
			if (!pix_feat_with_mem)
				return SAM3_ENOMEM;
			const float *src = (const float *)image_embed->data;
			const float *add = (const float *)
				trk->interactivity_no_mem_embed->data;
			float *dst = (float *)pix_feat_with_mem->data;
			for (int i = 0; i < HW; i++)
				for (int c = 0; c < D; c++)
					dst[i * D + c] = src[i * D + c] + add[c];
		} else {
			struct sam3_tensor *cond_2d =
				sam3_multiplex_memory_attn_forward(
					g, arena, &trk->transformer,
					tgt, NULL, tgt,
					memory, memory_image,
					memory_image_pos,
					W, num_k_exclude);
			if (!cond_2d)
				return SAM3_ENOMEM;
			int dims_4d[4] = {1, H, W, D};
			pix_feat_with_mem = gh_reshape(g, arena, cond_2d,
						       4, dims_4d);
			if (!pix_feat_with_mem)
				return SAM3_ENOMEM;
		}
	}

	/* Build extra_per_object [16, 256] from output_valid/invalid_embed
	 * (add_output_suppression_embeddings=True). For this single-object
	 * path slot 0 is the only active slot; slots 1..15 use the invalid
	 * embed. Python: video_tracking_multiplex.py:795-801. */
	struct sam3_tensor *extra_per_object = NULL;
	if (trk->output_valid_embed && trk->output_invalid_embed) {
		int ex_dims[2] = {SAM3_MULTIPLEX_COUNT, SAM3_MULTIPLEX_HIDDEN_DIM};
		extra_per_object = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						   2, ex_dims);
		if (!extra_per_object)
			return SAM3_ENOMEM;
		const float *valid = (const float *)trk->output_valid_embed->data;
		const float *invalid = (const float *)trk->output_invalid_embed->data;
		float *dst = (float *)extra_per_object->data;
		for (int slot = 0; slot < SAM3_MULTIPLEX_COUNT; slot++) {
			const float *src = (slot == 0) ? valid : invalid;
			memcpy(dst + slot * SAM3_MULTIPLEX_HIDDEN_DIM,
			       src + slot * SAM3_MULTIPLEX_HIDDEN_DIM,
			       SAM3_MULTIPLEX_HIDDEN_DIM * sizeof(float));
		}
	}

	struct sam3_tensor *all_masks = NULL;
	struct sam3_tensor *all_iou   = NULL;
	struct sam3_tensor *all_score = NULL;
	struct sam3_tensor *all_sam   = NULL;
	enum sam3_error err = sam3_multiplex_mask_decoder_forward(
		g, arena, &trk->sam_mask_decoder,
		pix_feat_with_mem, image_pe, feat_s1, feat_s0,
		extra_per_object,
		&all_masks, &all_iou, &all_score, &all_sam);
	if (err != SAM3_OK)
		return err;

	/* Slice slot 0 from the 16-slot multiplex outputs, then collapse
	 * the leading singleton dim for caller convenience.
	 *
	 * masks : [16, 3, 4H, 4W] → [1, 3, 4H, 4W] → [3, 4H, 4W]
	 * iou   : [16, 3]         → [1, 3]         → [3]
	 * score : [16, 1]         → [1, 1]         → [1]
	 * sam   : [16, 3, 256]    → [1, 3, 256]    → [3, 256]
	 */
	struct sam3_tensor *s_masks = gh_slice(g, arena, all_masks, 0, 0, 1);
	if (!s_masks)
		return SAM3_ENOMEM;
	int masks3_dims[3] = {3, 4 * H, 4 * W};
	struct sam3_tensor *slot0_masks = gh_reshape(g, arena, s_masks,
						     3, masks3_dims);
	if (!slot0_masks)
		return SAM3_ENOMEM;

	struct sam3_tensor *s_iou = gh_slice(g, arena, all_iou, 0, 0, 1);
	if (!s_iou)
		return SAM3_ENOMEM;
	int iou_dims[1] = {3};
	struct sam3_tensor *slot0_iou = gh_reshape(g, arena, s_iou,
						   1, iou_dims);
	if (!slot0_iou)
		return SAM3_ENOMEM;

	struct sam3_tensor *s_score = gh_slice(g, arena, all_score, 0, 0, 1);
	if (!s_score)
		return SAM3_ENOMEM;
	int score_dims[1] = {1};
	struct sam3_tensor *slot0_score = gh_reshape(g, arena, s_score,
						     1, score_dims);
	if (!slot0_score)
		return SAM3_ENOMEM;

	struct sam3_tensor *s_sam = gh_slice(g, arena, all_sam, 0, 0, 1);
	if (!s_sam)
		return SAM3_ENOMEM;
	int sam2_dims[2] = {3, D};
	struct sam3_tensor *slot0_sam = gh_reshape(g, arena, s_sam,
						   2, sam2_dims);
	if (!slot0_sam)
		return SAM3_ENOMEM;

	/* Project all 3 sam_tokens through obj_ptr_proj (256→256→256→256).
	 * Caller picks the row matching the best-IoU mask after eval. */
	struct sam3_tensor *obj_w[3] = {
		trk->obj_ptr_proj.fc_w[0],
		trk->obj_ptr_proj.fc_w[1],
		trk->obj_ptr_proj.fc_w[2],
	};
	struct sam3_tensor *obj_b[3] = {
		trk->obj_ptr_proj.fc_b[0],
		trk->obj_ptr_proj.fc_b[1],
		trk->obj_ptr_proj.fc_b[2],
	};
	struct sam3_tensor *obj_ptrs = mlp3_relu(g, arena, slot0_sam,
						 obj_w, obj_b);
	if (!obj_ptrs)
		return SAM3_ENOMEM;

	*out_masks    = slot0_masks;
	*out_iou      = slot0_iou;
	*out_obj_ptrs = obj_ptrs;
	*out_score    = slot0_score;
	return SAM3_OK;
}
