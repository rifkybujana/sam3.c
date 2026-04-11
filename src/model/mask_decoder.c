/*
 * src/model/mask_decoder.c - SAM3 mask decoder implementation
 *
 * Implements the two-way transformer mask decoder matching SAM3's
 * tracker_model.mask_decoder architecture. The decoder processes 6
 * learned tokens (4 mask + 1 IoU + 1 obj_score) plus optional prompt
 * tokens through a 2-layer two-way transformer that bidirectionally
 * cross-attends between tokens and image features. Output mask tokens
 * are converted to 32-dim embeddings via hypernetwork MLPs, which are
 * then dotted with pixel-decoded image features for final mask logits.
 *
 * Key types:  sam3_mask_decoder
 * Depends on: mask_decoder.h, graph_helpers.h, util/log.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "mask_decoder.h"
#include "graph_helpers.h"
#include "util/log.h"

#define P "tracker_model.mask_decoder."
#define TP "tracker_model.mask_decoder.transformer."

/* ── Initialization ──────────────────────────────────────────────── */

enum sam3_error sam3_mask_decoder_init(struct sam3_mask_decoder *dec)
{
	memset(dec, 0, sizeof(*dec));
	dec->d_model = 256;
	dec->d_inner = 128;
	dec->d_pixel = 32;
	dec->n_heads = 8;
	dec->n_masks = 4;
	return SAM3_OK;
}

/* ── Weight loading helpers ──────────────────────────────────────── */

/*
 * fuse_3 - Load 3 separate [d, d_in] weights and fuse into [3*d, d_in].
 */
static struct sam3_tensor *fuse_3(const struct sam3_weight_file *wf,
				   const char *a, const char *b,
				   const char *c, struct sam3_arena *arena,
				   int d, int n_dims, const int *part_dims)
{
	struct sam3_tensor *ta, *tb, *tc, *out;
	int fused_dims[2];

	ta = gh_load_mmap(wf, a, arena, SAM3_DTYPE_F32,
			       n_dims, part_dims);
	tb = gh_load_mmap(wf, b, arena, SAM3_DTYPE_F32,
			       n_dims, part_dims);
	tc = gh_load_mmap(wf, c, arena, SAM3_DTYPE_F32,
			       n_dims, part_dims);
	if (!ta || !tb || !tc)
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

	memcpy(out->data, ta->data, ta->nbytes);
	memcpy((char *)out->data + ta->nbytes, tb->data, tb->nbytes);
	memcpy((char *)out->data + ta->nbytes + tb->nbytes,
	       tc->data, tc->nbytes);
	return out;
}

/*
 * load_ca_128 - Load cross-attention weights with 128-dim projections.
 *
 * Stores separate Q/K/V weights (not fused) since d_inner != d_model.
 */
static int load_ca_128(const struct sam3_weight_file *wf,
			struct sam3_arena *arena,
			const char *prefix,
			struct sam3_tensor **q_w, struct sam3_tensor **q_b,
			struct sam3_tensor **k_w, struct sam3_tensor **k_b,
			struct sam3_tensor **v_w, struct sam3_tensor **v_b,
			struct sam3_tensor **out_w, struct sam3_tensor **out_b)
{
	char name[128];
	int di = SAM3_MASK_DEC_D_INNER;
	int dm = 256;
	int proj_dims[] = {di, dm};
	int out_dims[] = {dm, di};
	int di_dims[] = {di};
	int dm_dims[] = {dm};

	snprintf(name, sizeof(name), "%sq_proj.weight", prefix);
	*q_w = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				 2, proj_dims);
	if (!*q_w) return -1;
	snprintf(name, sizeof(name), "%sq_proj.bias", prefix);
	*q_b = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				 1, di_dims);
	if (!*q_b) return -1;

	snprintf(name, sizeof(name), "%sk_proj.weight", prefix);
	*k_w = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				 2, proj_dims);
	if (!*k_w) return -1;
	snprintf(name, sizeof(name), "%sk_proj.bias", prefix);
	*k_b = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				 1, di_dims);
	if (!*k_b) return -1;

	snprintf(name, sizeof(name), "%sv_proj.weight", prefix);
	*v_w = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				 2, proj_dims);
	if (!*v_w) return -1;
	snprintf(name, sizeof(name), "%sv_proj.bias", prefix);
	*v_b = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				 1, di_dims);
	if (!*v_b) return -1;

	snprintf(name, sizeof(name), "%so_proj.weight", prefix);
	*out_w = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				   2, out_dims);
	if (!*out_w) return -1;
	snprintf(name, sizeof(name), "%so_proj.bias", prefix);
	*out_b = gh_load_mmap(wf, name, arena, SAM3_DTYPE_F32,
				   1, dm_dims);
	if (!*out_b) return -1;

	return 0;
}

/* ── Weight loading ──────────────────────────────────────────────── */

enum sam3_error sam3_mask_decoder_load(struct sam3_mask_decoder *dec,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena)
{
	int d = dec->d_model;
	int ff = 2048;
	char name[128], q_name[128], k_name[128], v_name[128];
	char ca_prefix[128];

	/* ── Learned tokens ─────────────────────────────────────── */
	int mt_dims[] = {dec->n_masks, d};
	dec->mask_tokens = gh_load_mmap(wf, P "mask_tokens.weight",
					     arena, SAM3_DTYPE_F32,
					     2, mt_dims);
	if (!dec->mask_tokens)
		return SAM3_ENOMEM;

	int tok1_dims[] = {1, d};
	dec->iou_token = gh_load_mmap(wf, P "iou_token.weight",
					   arena, SAM3_DTYPE_F32,
					   2, tok1_dims);
	if (!dec->iou_token)
		return SAM3_ENOMEM;

	dec->obj_score_token = gh_load_mmap(wf,
						 P "obj_score_token.weight",
						 arena, SAM3_DTYPE_F32,
						 2, tok1_dims);
	if (!dec->obj_score_token)
		return SAM3_ENOMEM;

	/* ── Transformer layers ─────────────────────────────────── */
	int d_dims[] = {d};
	int proj_dims[] = {d, d};
	int fc1_dims[] = {ff, d};
	int fc1_b_dims[] = {ff};
	int fc2_dims[] = {d, ff};

	for (int i = 0; i < SAM3_MASK_DEC_LAYERS; i++) {
		/* Self-attention: fuse Q/K/V into [768, 256] */
		snprintf(q_name, sizeof(q_name),
			 TP "layers.%d.self_attn.q_proj.weight", i);
		snprintf(k_name, sizeof(k_name),
			 TP "layers.%d.self_attn.k_proj.weight", i);
		snprintf(v_name, sizeof(v_name),
			 TP "layers.%d.self_attn.v_proj.weight", i);
		dec->layers[i].sa_qkv_w = fuse_3(wf, q_name, k_name,
						   v_name, arena, d,
						   2, proj_dims);
		if (!dec->layers[i].sa_qkv_w)
			return SAM3_ENOMEM;

		snprintf(q_name, sizeof(q_name),
			 TP "layers.%d.self_attn.q_proj.bias", i);
		snprintf(k_name, sizeof(k_name),
			 TP "layers.%d.self_attn.k_proj.bias", i);
		snprintf(v_name, sizeof(v_name),
			 TP "layers.%d.self_attn.v_proj.bias", i);
		dec->layers[i].sa_qkv_b = fuse_3(wf, q_name, k_name,
						   v_name, arena, d,
						   1, d_dims);
		if (!dec->layers[i].sa_qkv_b)
			return SAM3_ENOMEM;

		/* Self-attention output projection */
		snprintf(name, sizeof(name),
			 TP "layers.%d.self_attn.o_proj.weight", i);
		dec->layers[i].sa_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, proj_dims);
		if (!dec->layers[i].sa_out_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.self_attn.o_proj.bias", i);
		dec->layers[i].sa_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].sa_out_b)
			return SAM3_ENOMEM;

		/* Layer norm 1 (after self-attention) */
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm1.weight", i);
		dec->layers[i].ln1_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln1_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm1.bias", i);
		dec->layers[i].ln1_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln1_b)
			return SAM3_ENOMEM;

		/* Cross-attention: token -> image (128-dim) */
		snprintf(ca_prefix, sizeof(ca_prefix),
			 TP "layers.%d.cross_attn_token_to_image.",
			 i);
		if (load_ca_128(wf, arena, ca_prefix,
				&dec->layers[i].ca_ti_q_w,
				&dec->layers[i].ca_ti_q_b,
				&dec->layers[i].ca_ti_k_w,
				&dec->layers[i].ca_ti_k_b,
				&dec->layers[i].ca_ti_v_w,
				&dec->layers[i].ca_ti_v_b,
				&dec->layers[i].ca_ti_out_w,
				&dec->layers[i].ca_ti_out_b))
			return SAM3_ENOMEM;

		/* Layer norm 2 (after token->image CA) */
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm2.weight", i);
		dec->layers[i].ln2_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln2_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm2.bias", i);
		dec->layers[i].ln2_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln2_b)
			return SAM3_ENOMEM;

		/* MLP */
		snprintf(name, sizeof(name),
			 TP "layers.%d.mlp.proj_in.weight", i);
		dec->layers[i].mlp_fc1_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc1_dims);
		if (!dec->layers[i].mlp_fc1_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.mlp.proj_in.bias", i);
		dec->layers[i].mlp_fc1_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, fc1_b_dims);
		if (!dec->layers[i].mlp_fc1_b)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.mlp.proj_out.weight", i);
		dec->layers[i].mlp_fc2_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, fc2_dims);
		if (!dec->layers[i].mlp_fc2_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.mlp.proj_out.bias", i);
		dec->layers[i].mlp_fc2_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].mlp_fc2_b)
			return SAM3_ENOMEM;

		/* Layer norm 3 (after MLP) */
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm3.weight", i);
		dec->layers[i].ln3_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln3_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm3.bias", i);
		dec->layers[i].ln3_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln3_b)
			return SAM3_ENOMEM;

		/* Cross-attention: image -> token (128-dim) */
		snprintf(ca_prefix, sizeof(ca_prefix),
			 TP "layers.%d.cross_attn_image_to_token.",
			 i);
		if (load_ca_128(wf, arena, ca_prefix,
				&dec->layers[i].ca_it_q_w,
				&dec->layers[i].ca_it_q_b,
				&dec->layers[i].ca_it_k_w,
				&dec->layers[i].ca_it_k_b,
				&dec->layers[i].ca_it_v_w,
				&dec->layers[i].ca_it_v_b,
				&dec->layers[i].ca_it_out_w,
				&dec->layers[i].ca_it_out_b))
			return SAM3_ENOMEM;

		/* Layer norm 4 (after image->token CA) */
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm4.weight", i);
		dec->layers[i].ln4_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln4_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 TP "layers.%d.layer_norm4.bias", i);
		dec->layers[i].ln4_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->layers[i].ln4_b)
			return SAM3_ENOMEM;
	}

	/* ── Final cross-attention ──────────────────────────────── */
	if (load_ca_128(wf, arena,
			TP "final_attn_token_to_image.",
			&dec->final_q_w, &dec->final_q_b,
			&dec->final_k_w, &dec->final_k_b,
			&dec->final_v_w, &dec->final_v_b,
			&dec->final_out_w, &dec->final_out_b))
		return SAM3_ENOMEM;

	dec->final_ln_w = gh_load_mmap(wf,
		TP "layer_norm_final_attn.weight",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->final_ln_w)
		return SAM3_ENOMEM;
	dec->final_ln_b = gh_load_mmap(wf,
		TP "layer_norm_final_attn.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->final_ln_b)
		return SAM3_ENOMEM;

	/* ── Pixel decoder ──────────────────────────────────────── */
	/*
	 * Pixel decoder conv weights ship in OHWI [OC, KH, KW, IC]
	 * after Task 12's permute in sam3_convert. Source checkpoint
	 * was IOHW for ConvTranspose2d (upscale_conv*) and OIHW for
	 * Conv2d (conv_s*); both converge on the same OHWI layout.
	 *
	 * upscale_conv1: IOHW [256, 64, 2, 2] -> OHWI [64, 2, 2, 256]
	 * upscale_conv2: IOHW [64, 32, 2, 2]  -> OHWI [32, 2, 2, 64]
	 */
	int conv1_w_dims[] = {64, 2, 2, 256};
	int conv1_b_dims[] = {64};
	int ln64_dims[] = {64};
	int conv2_w_dims[] = {32, 2, 2, 64};
	int conv2_b_dims[] = {32};

	dec->up_conv1_w = gh_load_mmap(wf,
		P "upscale_conv1.weight",
		arena, SAM3_DTYPE_F32, 4, conv1_w_dims);
	if (!dec->up_conv1_w)
		return SAM3_ENOMEM;
	dec->up_conv1_b = gh_load_mmap(wf,
		P "upscale_conv1.bias",
		arena, SAM3_DTYPE_F32, 1, conv1_b_dims);
	if (!dec->up_conv1_b)
		return SAM3_ENOMEM;

	dec->up_ln_w = gh_load_mmap(wf,
		P "upscale_layer_norm.weight",
		arena, SAM3_DTYPE_F32, 1, ln64_dims);
	if (!dec->up_ln_w)
		return SAM3_ENOMEM;
	dec->up_ln_b = gh_load_mmap(wf,
		P "upscale_layer_norm.bias",
		arena, SAM3_DTYPE_F32, 1, ln64_dims);
	if (!dec->up_ln_b)
		return SAM3_ENOMEM;

	dec->up_conv2_w = gh_load_mmap(wf,
		P "upscale_conv2.weight",
		arena, SAM3_DTYPE_F32, 4, conv2_w_dims);
	if (!dec->up_conv2_w)
		return SAM3_ENOMEM;
	dec->up_conv2_b = gh_load_mmap(wf,
		P "upscale_conv2.bias",
		arena, SAM3_DTYPE_F32, 1, conv2_b_dims);
	if (!dec->up_conv2_b)
		return SAM3_ENOMEM;

	/* ── Hypernetwork MLPs ──────────────────────────────────── */
	for (int i = 0; i < dec->n_masks; i++) {
		int hd_dims[] = {d, d};
		int ho_w_dims[] = {dec->d_pixel, d};
		int ho_b_dims[] = {dec->d_pixel};

		snprintf(name, sizeof(name),
			 P "output_hypernetworks_mlps.%d.proj_in.weight",
			 i);
		dec->hyper[i].proj_in_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, hd_dims);
		if (!dec->hyper[i].proj_in_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 P "output_hypernetworks_mlps.%d.proj_in.bias",
			 i);
		dec->hyper[i].proj_in_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->hyper[i].proj_in_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 P "output_hypernetworks_mlps.%d.layers.0.weight",
			 i);
		dec->hyper[i].hidden_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 2, hd_dims);
		if (!dec->hyper[i].hidden_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 P "output_hypernetworks_mlps.%d.layers.0.bias",
			 i);
		dec->hyper[i].hidden_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32, 1, d_dims);
		if (!dec->hyper[i].hidden_b)
			return SAM3_ENOMEM;

		snprintf(name, sizeof(name),
			 P "output_hypernetworks_mlps.%d.proj_out.weight",
			 i);
		dec->hyper[i].proj_out_w = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			2, ho_w_dims);
		if (!dec->hyper[i].proj_out_w)
			return SAM3_ENOMEM;
		snprintf(name, sizeof(name),
			 P "output_hypernetworks_mlps.%d.proj_out.bias",
			 i);
		dec->hyper[i].proj_out_b = gh_load_mmap(
			wf, name, arena, SAM3_DTYPE_F32,
			1, ho_b_dims);
		if (!dec->hyper[i].proj_out_b)
			return SAM3_ENOMEM;
	}

	/* ── IoU prediction MLP ─────────────────────────────────── */
	int iou_d_dims[] = {d, d};
	int iou_out_w_dims[] = {dec->n_masks, d};
	int iou_out_b_dims[] = {dec->n_masks};

	dec->iou_proj_in_w = gh_load_mmap(wf,
		P "iou_prediction_head.proj_in.weight",
		arena, SAM3_DTYPE_F32, 2, iou_d_dims);
	if (!dec->iou_proj_in_w)
		return SAM3_ENOMEM;
	dec->iou_proj_in_b = gh_load_mmap(wf,
		P "iou_prediction_head.proj_in.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->iou_proj_in_b)
		return SAM3_ENOMEM;

	dec->iou_hidden_w = gh_load_mmap(wf,
		P "iou_prediction_head.layers.0.weight",
		arena, SAM3_DTYPE_F32, 2, iou_d_dims);
	if (!dec->iou_hidden_w)
		return SAM3_ENOMEM;
	dec->iou_hidden_b = gh_load_mmap(wf,
		P "iou_prediction_head.layers.0.bias",
		arena, SAM3_DTYPE_F32, 1, d_dims);
	if (!dec->iou_hidden_b)
		return SAM3_ENOMEM;

	dec->iou_proj_out_w = gh_load_mmap(wf,
		P "iou_prediction_head.proj_out.weight",
		arena, SAM3_DTYPE_F32, 2, iou_out_w_dims);
	if (!dec->iou_proj_out_w)
		return SAM3_ENOMEM;
	dec->iou_proj_out_b = gh_load_mmap(wf,
		P "iou_prediction_head.proj_out.bias",
		arena, SAM3_DTYPE_F32, 1, iou_out_b_dims);
	if (!dec->iou_proj_out_b)
		return SAM3_ENOMEM;

	/* ── Multi-scale skip connection convolutions ──────────── */
	{
		/* Conv2d weights ship in OHWI after Task 12's permute:
		 * conv_s0: OIHW [32, 256, 1, 1] -> OHWI [32, 1, 1, 256]
		 * conv_s1: OIHW [64, 256, 1, 1] -> OHWI [64, 1, 1, 256]
		 */
		int s0w[] = {32, 1, 1, 256}, s0b[] = {32};
		int s1w[] = {64, 1, 1, 256}, s1b[] = {64};

		dec->conv_s0_w = gh_load_mmap(wf,
			P "conv_s0.weight", arena,
			SAM3_DTYPE_F32, 4, s0w);
		if (!dec->conv_s0_w) return SAM3_ENOMEM;
		dec->conv_s0_b = gh_load_mmap(wf,
			P "conv_s0.bias", arena,
			SAM3_DTYPE_F32, 1, s0b);
		if (!dec->conv_s0_b) return SAM3_ENOMEM;

		dec->conv_s1_w = gh_load_mmap(wf,
			P "conv_s1.weight", arena,
			SAM3_DTYPE_F32, 4, s1w);
		if (!dec->conv_s1_w) return SAM3_ENOMEM;
		dec->conv_s1_b = gh_load_mmap(wf,
			P "conv_s1.bias", arena,
			SAM3_DTYPE_F32, 1, s1b);
		if (!dec->conv_s1_b) return SAM3_ENOMEM;
	}

	/* ── no_mask_embed & PE gaussian ──────────────────────── */
	{
		int nm_dims[] = {1, d};
		int pe_dims[] = {2, 128};

		dec->no_mask_embed = gh_load_mmap(wf,
			"tracker_model.prompt_encoder.no_mask_embed.weight",
			arena, SAM3_DTYPE_F32, 2, nm_dims);
		if (!dec->no_mask_embed)
			return SAM3_ENOMEM;

		dec->pe_gaussian = gh_load_mmap(wf,
			"tracker_model.prompt_encoder.shared_embedding.positional_embedding",
			arena, SAM3_DTYPE_F32, 2, pe_dims);
		if (!dec->pe_gaussian)
			return SAM3_ENOMEM;
	}

	/* Diagnostic: verify critical weights are non-zero */
	{
		const float *g = (const float *)dec->pe_gaussian->data;
		float gmin = g[0], gmax = g[0];
		for (int i = 1; i < 256; i++) {
			if (g[i] < gmin) gmin = g[i];
			if (g[i] > gmax) gmax = g[i];
		}
		const float *nm = (const float *)dec->no_mask_embed->data;
		float nmmin = nm[0], nmmax = nm[0];
		for (int i = 1; i < 256; i++) {
			if (nm[i] < nmmin) nmmin = nm[i];
			if (nm[i] > nmmax) nmmax = nm[i];
		}
		const float *u1 = (const float *)dec->up_conv1_w->data;
		float u1min = u1[0], u1max = u1[0];
		for (int i = 1; i < 256 * 64 * 4; i++) {
			if (u1[i] < u1min) u1min = u1[i];
			if (u1[i] > u1max) u1max = u1[i];
		}
		const float *h0 = (const float *)dec->hyper[0].proj_in_w->data;
		float h0min = h0[0], h0max = h0[0];
		for (int i = 1; i < 256 * 256; i++) {
			if (h0[i] < h0min) h0min = h0[i];
			if (h0[i] > h0max) h0max = h0[i];
		}
		sam3_log_debug("mask_dec load: pe_gauss [%.4f, %.4f] "
			       "no_mask [%.4f, %.4f] "
			       "up_conv1 [%.4f, %.4f] "
			       "hyper0_in [%.4f, %.4f]",
			       gmin, gmax, nmmin, nmmax,
			       u1min, u1max, h0min, h0max);
	}

	return SAM3_OK;
}

/* ── Graph building helpers ──────────────────────────────────────── */

/*
 * cross_attn_128 - Cross-attention with 128-dim internal projections.
 *
 * Q projected from q_src [n_q, 256] -> [n_q, 128]
 * K projected from k_src [n_kv, 256] -> [n_kv, 128]
 * V projected from v_src [n_kv, 256] -> [n_kv, 128]
 * Per-head SDPA, then output projection [128] -> [256].
 *
 * k_src and v_src may differ (e.g., K has PE, V does not).
 *
 * Returns [n_q, 256], or NULL on error.
 */
static struct sam3_tensor *cross_attn_128(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *q_src,
	struct sam3_tensor *k_src,
	struct sam3_tensor *v_src,
	struct sam3_tensor *q_w, struct sam3_tensor *q_b,
	struct sam3_tensor *k_w, struct sam3_tensor *k_b,
	struct sam3_tensor *v_w, struct sam3_tensor *v_b,
	struct sam3_tensor *out_w, struct sam3_tensor *out_b,
	int n_heads)
{
	int d_inner = SAM3_MASK_DEC_D_INNER;
	int head_dim = d_inner / n_heads;

	/* Project Q, K, V to 128-dim */
	struct sam3_tensor *q = gh_linear(g, arena, q_src, q_w, q_b);
	struct sam3_tensor *k = gh_linear(g, arena, k_src, k_w, k_b);
	struct sam3_tensor *v = gh_linear(g, arena, v_src, v_w, v_b);
	if (!q || !k || !v)
		return NULL;

	/* Per-head SDPA */
	struct sam3_tensor *head_outs[64];
	for (int h = 0; h < n_heads; h++) {
		int s = h * head_dim;
		int e = s + head_dim;

		struct sam3_tensor *hq = gh_slice(g, arena, q, 1, s, e);
		struct sam3_tensor *hk = gh_slice(g, arena, k, 1, s, e);
		struct sam3_tensor *hv = gh_slice(g, arena, v, 1, s, e);
		if (!hq || !hk || !hv)
			return NULL;

		head_outs[h] = gh_sdpa(g, arena, hq, hk, hv,
					NULL, head_dim);
		if (!head_outs[h])
			return NULL;
	}

	/* Concatenate heads: [n_q, 128] */
	struct sam3_tensor *merged;
	if (n_heads == 1) {
		merged = head_outs[0];
	} else {
		merged = gh_concat(g, arena, head_outs, n_heads, 1);
		if (!merged)
			return NULL;
	}

	/* Output projection: [n_q, 128] -> [n_q, 256] */
	return gh_linear(g, arena, merged, out_w, out_b);
}

/*
 * hypernetwork_mlp - 3-layer MLP: proj_in -> relu -> hidden -> relu -> proj_out.
 *
 * Input: [d_model], output: [d_pixel].
 */
static struct sam3_tensor *hypernetwork_mlp(
	struct sam3_graph *g, struct sam3_arena *arena,
	struct sam3_tensor *input,
	struct sam3_tensor *pi_w, struct sam3_tensor *pi_b,
	struct sam3_tensor *hid_w, struct sam3_tensor *hid_b,
	struct sam3_tensor *po_w, struct sam3_tensor *po_b)
{
	struct sam3_tensor *h;

	h = gh_linear(g, arena, input, pi_w, pi_b);
	if (!h) return NULL;
	h = gh_relu(g, arena, h);
	if (!h) return NULL;
	h = gh_linear(g, arena, h, hid_w, hid_b);
	if (!h) return NULL;
	h = gh_relu(g, arena, h);
	if (!h) return NULL;
	h = gh_linear(g, arena, h, po_w, po_b);
	if (!h) return NULL;

	return h;
}

/*
 * compute_image_pe - Precompute positional encoding for image grid.
 *
 * Uses PositionEmbeddingRandom: normalized coords → Gaussian projection
 * → 2π scaling → sin/cos → [H*W, 256].
 */
static struct sam3_tensor *compute_image_pe(struct sam3_arena *arena,
					     struct sam3_tensor *gaussian,
					     int grid_h, int grid_w)
{
	int hw = grid_h * grid_w;
	int dims[] = {hw, 256};
	struct sam3_tensor *pe;
	const float *gauss;
	float *out;
	int i, j, c;

	pe = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, dims);
	if (!pe)
		return NULL;

	gauss = (const float *)gaussian->data;
	out = (float *)pe->data;

	for (i = 0; i < grid_h; i++) {
		for (j = 0; j < grid_w; j++) {
			float y = 2.0f * ((i + 0.5f) / grid_h) - 1.0f;
			float x = 2.0f * ((j + 0.5f) / grid_w) - 1.0f;
			int idx = i * grid_w + j;
			for (c = 0; c < 128; c++) {
				float v = x * gauss[c] +
					  y * gauss[128 + c];
				v *= 2.0f * 3.14159265f;
				out[idx * 256 + c] = sinf(v);
				out[idx * 256 + 128 + c] = cosf(v);
			}
		}
	}
	return pe;
}

/*
 * broadcast_add_1d - Add [1, d] bias to every row of [n, d] tensor.
 *
 * Materializes result in arena (no graph op needed).
 */
static struct sam3_tensor *broadcast_add_1d(struct sam3_arena *arena,
					     struct sam3_tensor *src,
					     struct sam3_tensor *bias)
{
	int n = src->dims[0];
	int d = src->dims[1];
	struct sam3_tensor *out;
	const float *s, *b;
	float *o;
	int i, j;

	out = gh_alloc_tensor(arena, src->dtype, src->n_dims, src->dims);
	if (!out)
		return NULL;

	s = (const float *)src->data;
	b = (const float *)bias->data;
	o = (float *)out->data;

	for (i = 0; i < n; i++)
		for (j = 0; j < d; j++)
			o[i * d + j] = s[i * d + j] + b[j];
	return out;
}

/* ── Main graph build ────────────────────────────────────────────── */

enum sam3_error sam3_mask_decoder_build(
	struct sam3_mask_decoder *dec,
	struct sam3_graph *g,
	struct sam3_tensor *img_feat,
	int grid_h, int grid_w,
	struct sam3_tensor *prompt,
	struct sam3_tensor *feat_s0,
	struct sam3_tensor *feat_s1,
	struct sam3_arena *arena,
	struct sam3_tensor **out_masks,
	struct sam3_tensor **out_iou)
{
	int d = dec->d_model;
	int nm = dec->n_masks;
	int nh = dec->n_heads;
	int dp = dec->d_pixel;
	int n_pix = grid_h * grid_w;

	/*
	 * Step 1: Prepare tokens.
	 * Concatenate [iou_token(1), mask_tokens(4), obj_score_token(1)]
	 * and optionally prompt tokens.
	 * Total: 6 + n_prompt tokens.
	 */
	struct sam3_tensor *parts[4];
	int n_parts = 0;

	parts[n_parts++] = dec->iou_token;	/* [1, 256] */
	parts[n_parts++] = dec->mask_tokens;	/* [4, 256] */
	parts[n_parts++] = dec->obj_score_token; /* [1, 256] */
	if (prompt)
		parts[n_parts++] = prompt;

	struct sam3_tensor *tokens;
	tokens = gh_concat(g, arena, parts, n_parts, 0);
	if (!tokens)
		return SAM3_ENOMEM;

	int n_tokens = tokens->dims[0];

	sam3_log_debug("mask_dec: %d tokens, %d image pixels",
		       n_tokens, n_pix);

	/*
	 * Step 2: Precompute image positional encoding and add
	 * no_mask_embed (dense prompt embedding) to image features.
	 *
	 * SAM2 reference: src = image_embeddings + dense_prompt_embeddings
	 * where dense_prompt_embeddings = no_mask_embed when no mask prompt.
	 * PE is added to K (not V) in cross-attention projections.
	 */
	struct sam3_tensor *image_pe;
	image_pe = compute_image_pe(arena, dec->pe_gaussian,
				     grid_h, grid_w);
	if (!image_pe)
		return SAM3_ENOMEM;

	/* Add no_mask_embed [1, 256] to every pixel of image features */
	img_feat = broadcast_add_1d(arena, img_feat, dec->no_mask_embed);
	if (!img_feat)
		return SAM3_ENOMEM;

	struct sam3_tensor *queries = tokens;	/* evolving tokens */
	struct sam3_tensor *keys = img_feat;	/* evolving image feat */
	struct sam3_tensor *query_pe = tokens;	/* fixed token PE */

	for (int i = 0; i < SAM3_MASK_DEC_LAYERS; i++) {
		/* Precompute keys + image_pe for cross-attention.
		 * SAM2 adds PE to K before projection (not to V). */
		struct sam3_tensor *keys_pe;
		keys_pe = gh_add(g, arena, keys, image_pe);
		if (!keys_pe)
			return SAM3_ENOMEM;

		/*
		 * (a) Self-attention on tokens.
		 * Layer 0: skip_first_layer_pe — no PE, no residual.
		 * Layer 1+: Q/K = queries+query_pe, V = queries,
		 *   with residual connection.
		 *
		 * Since packed QKV projects a single input to Q/K/V,
		 * layer 1 passes queries+query_pe (V gets PE too —
		 * small approximation vs separate projections).
		 */
		int attn_dims[] = {1, n_tokens, d};
		struct sam3_tensor *sa_input;
		struct sam3_tensor *q3d;

		if (i == 0) {
			sa_input = queries;
		} else {
			sa_input = gh_add(g, arena, queries, query_pe);
			if (!sa_input)
				return SAM3_ENOMEM;
		}

		q3d = gh_reshape(g, arena, sa_input, 3, attn_dims);
		if (!q3d)
			return SAM3_ENOMEM;

		struct sam3_tensor *sa_out;
		sa_out = gh_multihead_attention(
			g, arena, q3d, q3d, q3d,
			dec->layers[i].sa_qkv_w,
			dec->layers[i].sa_qkv_b,
			dec->layers[i].sa_out_w,
			dec->layers[i].sa_out_b,
			nh);
		if (!sa_out)
			return SAM3_ENOMEM;

		/* Layer 0: replace (no residual).
		 * Layer 1+: residual connection. */
		if (i == 0) {
			queries = sa_out;
		} else {
			queries = gh_add(g, arena, queries, sa_out);
			if (!queries)
				return SAM3_ENOMEM;
		}

		queries = gh_layernorm(g, arena, queries,
					dec->layers[i].ln1_w,
					dec->layers[i].ln1_b);
		if (!queries)
			return SAM3_ENOMEM;

		/*
		 * (b) Cross-attention: tokens attend to image.
		 * Q = tokens+query_pe, K = image+PE, V = image.
		 */
		struct sam3_tensor *q_with_pe;
		q_with_pe = gh_add(g, arena, queries, query_pe);
		if (!q_with_pe)
			return SAM3_ENOMEM;

		struct sam3_tensor *ca_ti;
		ca_ti = cross_attn_128(g, arena,
			q_with_pe, keys_pe, keys,
			dec->layers[i].ca_ti_q_w,
			dec->layers[i].ca_ti_q_b,
			dec->layers[i].ca_ti_k_w,
			dec->layers[i].ca_ti_k_b,
			dec->layers[i].ca_ti_v_w,
			dec->layers[i].ca_ti_v_b,
			dec->layers[i].ca_ti_out_w,
			dec->layers[i].ca_ti_out_b,
			nh);
		if (!ca_ti)
			return SAM3_ENOMEM;

		queries = gh_add(g, arena, queries, ca_ti);
		if (!queries)
			return SAM3_ENOMEM;

		queries = gh_layernorm(g, arena, queries,
					dec->layers[i].ln2_w,
					dec->layers[i].ln2_b);
		if (!queries)
			return SAM3_ENOMEM;

		/*
		 * (c) MLP on tokens with residual.
		 */
		struct sam3_tensor *mlp_out;
		mlp_out = gh_mlp(g, arena, queries,
				  dec->layers[i].mlp_fc1_w,
				  dec->layers[i].mlp_fc1_b,
				  dec->layers[i].mlp_fc2_w,
				  dec->layers[i].mlp_fc2_b,
				  SAM3_OP_RELU);
		if (!mlp_out)
			return SAM3_ENOMEM;

		queries = gh_add(g, arena, queries, mlp_out);
		if (!queries)
			return SAM3_ENOMEM;

		queries = gh_layernorm(g, arena, queries,
					dec->layers[i].ln3_w,
					dec->layers[i].ln3_b);
		if (!queries)
			return SAM3_ENOMEM;

		/*
		 * (d) Cross-attention: image attends to tokens.
		 * Q = image+PE, K = tokens+query_pe, V = tokens.
		 */
		q_with_pe = gh_add(g, arena, queries, query_pe);
		if (!q_with_pe)
			return SAM3_ENOMEM;

		struct sam3_tensor *ca_it;
		ca_it = cross_attn_128(g, arena,
			keys_pe, q_with_pe, queries,
			dec->layers[i].ca_it_q_w,
			dec->layers[i].ca_it_q_b,
			dec->layers[i].ca_it_k_w,
			dec->layers[i].ca_it_k_b,
			dec->layers[i].ca_it_v_w,
			dec->layers[i].ca_it_v_b,
			dec->layers[i].ca_it_out_w,
			dec->layers[i].ca_it_out_b,
			nh);
		if (!ca_it)
			return SAM3_ENOMEM;

		keys = gh_add(g, arena, keys, ca_it);
		if (!keys)
			return SAM3_ENOMEM;

		keys = gh_layernorm(g, arena, keys,
				     dec->layers[i].ln4_w,
				     dec->layers[i].ln4_b);
		if (!keys)
			return SAM3_ENOMEM;
	}

	/*
	 * Step 3: Final cross-attention (token -> image) + layer norm.
	 * Q = queries+query_pe, K = keys+image_pe, V = keys.
	 */
	struct sam3_tensor *final_q_pe;
	final_q_pe = gh_add(g, arena, queries, query_pe);
	if (!final_q_pe)
		return SAM3_ENOMEM;

	struct sam3_tensor *final_keys_pe;
	final_keys_pe = gh_add(g, arena, keys, image_pe);
	if (!final_keys_pe)
		return SAM3_ENOMEM;

	struct sam3_tensor *final_ca;
	final_ca = cross_attn_128(g, arena,
		final_q_pe, final_keys_pe, keys,
		dec->final_q_w, dec->final_q_b,
		dec->final_k_w, dec->final_k_b,
		dec->final_v_w, dec->final_v_b,
		dec->final_out_w, dec->final_out_b,
		nh);
	if (!final_ca)
		return SAM3_ENOMEM;

	queries = gh_add(g, arena, queries, final_ca);
	if (!queries)
		return SAM3_ENOMEM;

	queries = gh_layernorm(g, arena, queries,
				dec->final_ln_w,
				dec->final_ln_b);
	if (!queries)
		return SAM3_ENOMEM;

	/*
	 * Step 4: Extract mask token outputs [4, d_model].
	 * Token layout: [iou(1), mask(4), obj_score(1), prompt(...)].
	 * Mask tokens are at indices 1..4.
	 */
	struct sam3_tensor *mask_tokens_out;
	mask_tokens_out = gh_slice(g, arena, queries, 0, 1, 1 + nm);
	if (!mask_tokens_out)
		return SAM3_ENOMEM;

	/* IoU token is at index 0 */
	struct sam3_tensor *iou_token_out;
	iou_token_out = gh_slice(g, arena, queries, 0, 0, 1);
	if (!iou_token_out)
		return SAM3_ENOMEM;

	/*
	 * Step 5: Pixel decoder — upsample image features.
	 * Reshape [n_pix, d_model] -> [1, H, W, d_model] (NHWC),
	 * then 2x transposed conv + layer norm + GELU pipeline.
	 *
	 * keys is row-major [H*W, d] with element order
	 * (h*W + w, c), which matches NHWC byte order exactly, so
	 * the reshape is a pure view change with no data movement.
	 */
	struct sam3_tensor *px;
	{
		int nhwc_dims[] = {1, grid_h, grid_w, d};
		px = gh_reshape(g, arena, keys, 4, nhwc_dims);
		if (!px)
			return SAM3_ENOMEM;
	}

	/* Conv transpose 1: [1, H, W, 256] -> [1, 2H, 2W, 64] */
	px = gh_conv_transpose2d(g, arena, px,
				       dec->up_conv1_w, dec->up_conv1_b,
				       2, 0);
	if (!px)
		return SAM3_ENOMEM;

	/*
	 * Multi-scale skip: add conv_s1(feat_s1) at 2x resolution.
	 * feat_s1 is [1, 2H, 2W, 256] from 1x FPN scale (NHWC),
	 * conv_s1 projects 256→64 channels via 1x1 conv.
	 */
	if (feat_s1) {
		struct sam3_tensor *skip1;
		skip1 = gh_conv2d(g, arena, feat_s1,
					dec->conv_s1_w, dec->conv_s1_b,
					1, 0);
		if (!skip1)
			return SAM3_ENOMEM;
		px = gh_add(g, arena, px, skip1);
		if (!px)
			return SAM3_ENOMEM;
	}

	/* Layer norm on 64-dim channels.
	 * In NHWC [1, 2H, 2W, 64], channels are the innermost axis,
	 * so a reshape to [4HW, 64] is a pure view. LN normalizes
	 * across the last axis, then reshape back to NHWC.
	 */
	{
		int h2 = grid_h * 2;
		int w2 = grid_w * 2;
		int hw2 = h2 * w2;
		int c1 = dec->up_ln_w->dims[0];
		int flat_dims[] = {hw2, c1};
		int nhwc2_dims[] = {1, h2, w2, c1};

		px = gh_reshape(g, arena, px, 2, flat_dims);
		if (!px) return SAM3_ENOMEM;

		px = gh_layernorm(g, arena, px,
				   dec->up_ln_w, dec->up_ln_b);
		if (!px) return SAM3_ENOMEM;

		px = gh_reshape(g, arena, px, 4, nhwc2_dims);
		if (!px) return SAM3_ENOMEM;
	}

	/* GELU after LayerNorm (act1 in SAM2 output_upscaling) */
	px = gh_gelu(g, arena, px);
	if (!px)
		return SAM3_ENOMEM;

	/* Conv transpose 2: [1, 2H, 2W, 64] -> [1, 4H, 4W, 32]
	 * SAM2: act2(dc2(act1(ln1(dc1(src) + feat_s1))) + feat_s0) */
	px = gh_conv_transpose2d(g, arena, px,
				       dec->up_conv2_w, dec->up_conv2_b,
				       2, 0);
	if (!px)
		return SAM3_ENOMEM;

	/*
	 * Multi-scale skip: add conv_s0(feat_s0) at 4x resolution.
	 * feat_s0 is [1, 4H, 4W, 256] from 2x FPN scale (NHWC),
	 * conv_s0 projects 256→32 channels via 1x1 conv.
	 */
	if (feat_s0) {
		struct sam3_tensor *skip0;
		skip0 = gh_conv2d(g, arena, feat_s0,
					dec->conv_s0_w, dec->conv_s0_b,
					1, 0);
		if (!skip0)
			return SAM3_ENOMEM;
		px = gh_add(g, arena, px, skip0);
		if (!px)
			return SAM3_ENOMEM;
	}

	/* GELU #2 (after dc2 + feat_s0 skip) */
	px = gh_gelu(g, arena, px);
	if (!px)
		return SAM3_ENOMEM;

	/*
	 * Flatten pixel features for dot product.
	 * px is NHWC [1, 4H, 4W, 32]; reshape to [4H*4W, 32] is a
	 * pure view (channels are innermost), then transpose once to
	 * [32, 4H*4W] so the downstream matmul can do
	 * hyper_in [4, 32] @ px [32, final_hw] -> [4, final_hw].
	 */
	int final_h = grid_h * 4;
	int final_w = grid_w * 4;
	int final_hw = final_h * final_w;
	{
		int flat_dims[] = {final_hw, dp};
		px = gh_reshape(g, arena, px, 2, flat_dims);
		if (!px) return SAM3_ENOMEM;
	}
	/* px is [final_hw, 32] */

	/*
	 * Step 6: Hypernetwork MLPs.
	 * Each mask token -> 3-layer MLP -> [1, 32].
	 * Stack all 4 -> [4, 32].
	 */
	struct sam3_tensor *hyper_parts[SAM3_MASK_DEC_MASKS];
	for (int i = 0; i < nm; i++) {
		/* Slice one mask token: [1, d_model] */
		struct sam3_tensor *mt;
		mt = gh_slice(g, arena, mask_tokens_out, 0, i, i + 1);
		if (!mt)
			return SAM3_ENOMEM;

		hyper_parts[i] = hypernetwork_mlp(g, arena, mt,
			dec->hyper[i].proj_in_w, dec->hyper[i].proj_in_b,
			dec->hyper[i].hidden_w, dec->hyper[i].hidden_b,
			dec->hyper[i].proj_out_w, dec->hyper[i].proj_out_b);
		if (!hyper_parts[i])
			return SAM3_ENOMEM;
	}

	/* Concatenate: [4, 32] */
	struct sam3_tensor *hyper_in;
	hyper_in = gh_concat(g, arena, hyper_parts, nm, 0);
	if (!hyper_in)
		return SAM3_ENOMEM;

	/*
	 * Step 7: Mask logits = hyper_in @ px^T.
	 * hyper_in [4, 32] @ px^T [32, final_hw] -> [4, final_hw].
	 */
	struct sam3_tensor *px_t = gh_transpose(g, arena, px);
	if (!px_t)
		return SAM3_ENOMEM;

	struct sam3_tensor *masks = gh_matmul(g, arena, hyper_in, px_t);
	if (!masks)
		return SAM3_ENOMEM;

	/* Reshape to [4, final_h, final_w] */
	{
		int mask_dims[] = {nm, final_h, final_w};
		masks = gh_reshape(g, arena, masks, 3, mask_dims);
		if (!masks)
			return SAM3_ENOMEM;
	}

	/* Output raw logits — positive = inside mask, negative = outside.
	 * Consumer applies threshold at 0.0 or sigmoid + threshold at 0.5. */
	*out_masks = masks;

	/* Store debug intermediates for post-eval diagnostics */
	dec->_debug_px = px;
	dec->_debug_hyper = hyper_in;
	dec->_debug_keys = keys;
	dec->_debug_queries = queries;

	sam3_log_debug("mask_dec: masks [%d, %d, %d]",
		       nm, final_h, final_w);

	/*
	 * Step 8: IoU prediction from IoU token.
	 * iou_token_out [1, 256] -> 3-layer MLP -> [1, 4].
	 */
	if (out_iou) {
		struct sam3_tensor *iou;
		iou = hypernetwork_mlp(g, arena, iou_token_out,
					dec->iou_proj_in_w,
					dec->iou_proj_in_b,
					dec->iou_hidden_w,
					dec->iou_hidden_b,
					dec->iou_proj_out_w,
					dec->iou_proj_out_b);
		if (!iou)
			return SAM3_ENOMEM;

		/* Sigmoid to get 0-1 range */
		iou = gh_sigmoid(g, arena, iou);
		if (!iou)
			return SAM3_ENOMEM;

		*out_iou = iou;
	}

	return SAM3_OK;
}
