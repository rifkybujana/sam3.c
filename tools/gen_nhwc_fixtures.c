/*
 * tools/gen_nhwc_fixtures.c - Capture pre-migration NCHW fixture tensors
 *
 * Generates deterministic reference tensors for the NHWC layout migration.
 * Builds small synthetic neck, seg_head, and mask_decoder pixel-decoder
 * instances on the CPU backend, runs them, and dumps each input and
 * output to a .bin file. The subsequent NHWC implementations compare
 * against these files; any drift beyond 1e-4 is a migration bug.
 *
 * Key types:  sam3_ctx, sam3_graph, sam3_tensor
 * Depends on: sam3/sam3.h, model/{necks,segmentation,mask_decoder}.h
 * Used by:    ci script (one-shot generator), tests/test_*_nhwc.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "backend/backend.h"
#include "backend/cpu/cpu_backend.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "core/tensor.h"
#include "model/graph_helpers.h"
#include "model/mask_decoder.h"
#include "model/necks.h"
#include "model/segmentation.h"
#include "util/log.h"

/* Fixed PRNG seed for byte-identical fixture regeneration. */
#define GEN_NHWC_SEED 0x5A3ABCDEu

/* Neck fixture dimensions (mirrors tests/test_necks.c except grid_size). */
#define NECK_D_MODEL      16
#define NECK_BACKBONE_DIM 32
#define NECK_GRID_SIZE    8
#define NECK_N_SCALES     4
#define NECK_N_PATCHES    (NECK_GRID_SIZE * NECK_GRID_SIZE)

/* Seg head fixture dimensions. */
#define SEG_D_MODEL 32
#define SEG_N_HEADS 4
#define SEG_ENC_H   8
#define SEG_ENC_W   8
#define SEG_SEQ     (SEG_ENC_H * SEG_ENC_W)
#define SEG_FEAT2_H (SEG_ENC_H * 2)
#define SEG_FEAT2_W (SEG_ENC_W * 2)
#define SEG_FEAT4_H (SEG_ENC_H * 4)
#define SEG_FEAT4_W (SEG_ENC_W * 4)
#define SEG_N_QUERIES 4

/*
 * Mask decoder pixel-decoder fixture dimensions.
 *
 * d_model, d_pixel, n_heads and n_masks are all hardcoded inside
 * sam3_mask_decoder_init (256 / 32 / 8 / 4); only the spatial grid
 * and the optional prompt token count are tunable at build time.
 */
#define MD_GRID_H    4
#define MD_GRID_W    4
#define MD_N_PIX     (MD_GRID_H * MD_GRID_W)
#define MD_D_MODEL   256
#define MD_FEAT_S1_H (MD_GRID_H * 2)
#define MD_FEAT_S1_W (MD_GRID_W * 2)
#define MD_FEAT_S0_H (MD_GRID_H * 4)
#define MD_FEAT_S0_W (MD_GRID_W * 4)

static const float neck_scales[NECK_N_SCALES] = {4.0f, 2.0f, 1.0f, 0.5f};

/*
 * prng_next - Simple LCG producing deterministic floats in [-0.5, 0.5].
 *
 * Used to fill synthetic inputs and weight tensors. Uses Numerical
 * Recipes constants so that the sequence is reproducible on any host
 * that runs this tool.
 */
static float prng_next(uint32_t *state)
{
	*state = (*state) * 1664525u + 1013904223u;
	/* Map the top 24 bits into [-0.5, 0.5]. */
	uint32_t bits = (*state) >> 8;
	float unit = (float)bits / (float)(1u << 24);
	return unit - 0.5f;
}

/*
 * fill_prng - Write n floats from the PRNG into dst, scaled by amp.
 */
static void fill_prng(float *dst, int n, float amp, uint32_t *state)
{
	for (int i = 0; i < n; i++)
		dst[i] = amp * prng_next(state);
}

/*
 * write_bin - Write [u32 n_elems][float payload] to out_dir/name.
 *
 * Returns SAM3_OK on success or SAM3_EIO on any I/O error. Logs the
 * failing path so callers do not need to repeat it.
 */
static enum sam3_error write_bin(const char *out_dir, const char *name,
				 const float *data, int n_elems)
{
	char path[512];
	FILE *fp = NULL;
	uint32_t hdr = (uint32_t)n_elems;
	size_t wrote;
	enum sam3_error err = SAM3_OK;

	snprintf(path, sizeof(path), "%s/%s", out_dir, name);

	fp = fopen(path, "wb");
	if (!fp) {
		sam3_log_error("open %s failed: %s", path, strerror(errno));
		return SAM3_EIO;
	}

	wrote = fwrite(&hdr, sizeof(hdr), 1, fp);
	if (wrote != 1) {
		sam3_log_error("header write %s failed", path);
		err = SAM3_EIO;
		goto cleanup;
	}

	wrote = fwrite(data, sizeof(float), (size_t)n_elems, fp);
	if (wrote != (size_t)n_elems) {
		sam3_log_error("payload write %s failed (%zu/%d)",
			       path, wrote, n_elems);
		err = SAM3_EIO;
		goto cleanup;
	}

cleanup:
	if (fclose(fp) != 0) {
		sam3_log_error("close %s failed: %s",
			       path, strerror(errno));
		err = SAM3_EIO;
	}
	return err;
}

/*
 * dump_tensor - Convenience wrapper: flatten a tensor and write as .bin.
 */
static enum sam3_error dump_tensor(const char *out_dir, const char *name,
				   const struct sam3_tensor *t)
{
	int n = sam3_tensor_nelems((struct sam3_tensor *)t);
	return write_bin(out_dir, name, (const float *)t->data, n);
}

/*
 * fill_weight_tensors_neck - Populate every conv weight/bias with PRNG.
 *
 * The neck loader leaves weights zeroed when wf is NULL; this routine
 * replaces the zeros with small deterministic values so the pipeline
 * produces finite nonzero outputs.
 */
static void fill_weight_tensors_neck(struct sam3_neck *neck,
				     uint32_t *state)
{
	for (int s = 0; s < neck->n_scales; s++) {
		for (int j = 0; j < neck->stages[s].n_convs; j++) {
			struct sam3_tensor *w = neck->stages[s].conv_w[j];
			struct sam3_tensor *b = neck->stages[s].conv_b[j];
			if (w)
				fill_prng((float *)w->data,
					  sam3_tensor_nelems(w),
					  0.1f, state);
			if (b)
				fill_prng((float *)b->data,
					  sam3_tensor_nelems(b),
					  0.05f, state);
		}
	}
}

/*
 * dump_neck_fixture - Build and evaluate the small synthetic neck.
 *
 * Captures the raw ViT input and all 4 scale outputs as reference
 * tensors. Output feature maps are NCHW [1, d, H, W] (pre-migration).
 */
static enum sam3_error dump_neck_fixture(const char *out_dir,
					 struct sam3_cpu_backend *cpu)
{
	uint32_t rng = GEN_NHWC_SEED;
	struct sam3_neck neck;
	struct sam3_graph graph;
	struct sam3_tensor *vit_out;
	struct sam3_tensor *features[SAM3_NECK_MAX_SCALES];
	enum sam3_error err;

	err = sam3_neck_init(&neck, NECK_D_MODEL, NECK_BACKBONE_DIM,
			     NECK_GRID_SIZE, NECK_N_SCALES, neck_scales);
	if (err != SAM3_OK) {
		sam3_log_error("neck_init failed (%d)", err);
		return err;
	}

	err = sam3_neck_load(&neck, NULL, &cpu->arena);
	if (err != SAM3_OK) {
		sam3_log_error("neck_load failed (%d)", err);
		return err;
	}

	fill_weight_tensors_neck(&neck, &rng);

	int vit_dims[] = {NECK_N_PATCHES, NECK_BACKBONE_DIM};
	vit_out = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32,
				  2, vit_dims);
	if (!vit_out) {
		sam3_log_error("neck vit_out alloc failed");
		return SAM3_ENOMEM;
	}

	fill_prng((float *)vit_out->data,
		  sam3_tensor_nelems(vit_out), 0.5f, &rng);

	err = dump_tensor(out_dir, "neck_input.bin", vit_out);
	if (err != SAM3_OK)
		return err;

	sam3_graph_init(&graph);

	err = sam3_neck_build(&neck, &graph, vit_out, features,
			      &cpu->arena);
	if (err != SAM3_OK) {
		sam3_log_error("neck_build failed (%d)", err);
		return err;
	}

	err = cpu->base.ops->graph_eval(&cpu->base, &graph);
	if (err != SAM3_OK) {
		sam3_log_error("neck graph_eval failed (%d)", err);
		return err;
	}

	static const char *names[NECK_N_SCALES] = {
		"neck_features_s0.bin",
		"neck_features_s1.bin",
		"neck_features_s2.bin",
		"neck_features_s3.bin",
	};

	for (int s = 0; s < NECK_N_SCALES; s++) {
		if (!features[s]) {
			sam3_log_error("neck feature %d is NULL", s);
			return SAM3_EINVAL;
		}
		err = dump_tensor(out_dir, names[s], features[s]);
		if (err != SAM3_OK)
			return err;
	}

	sam3_log_info("neck fixture written (4 scales, grid %d)",
		      NECK_GRID_SIZE);
	return SAM3_OK;
}

/*
 * fill_weight_tensors_seg - Populate seg head weights with small PRNG.
 */
static void fill_weight_tensors_seg(struct sam3_seg_head *head,
				    uint32_t *state)
{
	for (int i = 0; i < SAM3_SEG_FPN_STAGES; i++) {
		if (head->fpn[i].conv_w)
			fill_prng((float *)head->fpn[i].conv_w->data,
				  sam3_tensor_nelems(head->fpn[i].conv_w),
				  0.1f, state);
		if (head->fpn[i].conv_b)
			fill_prng((float *)head->fpn[i].conv_b->data,
				  sam3_tensor_nelems(head->fpn[i].conv_b),
				  0.05f, state);
		if (head->fpn[i].gn_w) {
			/* Initialize GN gamma to small positive (PRNG+1). */
			float *w = (float *)head->fpn[i].gn_w->data;
			int n = sam3_tensor_nelems(head->fpn[i].gn_w);
			fill_prng(w, n, 0.1f, state);
			for (int k = 0; k < n; k++)
				w[k] += 1.0f;
		}
		if (head->fpn[i].gn_b)
			fill_prng((float *)head->fpn[i].gn_b->data,
				  sam3_tensor_nelems(head->fpn[i].gn_b),
				  0.05f, state);
	}

	if (head->inst_proj_w)
		fill_prng((float *)head->inst_proj_w->data,
			  sam3_tensor_nelems(head->inst_proj_w),
			  0.1f, state);
	if (head->inst_proj_b)
		fill_prng((float *)head->inst_proj_b->data,
			  sam3_tensor_nelems(head->inst_proj_b),
			  0.05f, state);

	for (int i = 0; i < SAM3_SEG_MASK_MLP_LAYERS; i++) {
		if (head->mask_mlp[i].w)
			fill_prng((float *)head->mask_mlp[i].w->data,
				  sam3_tensor_nelems(head->mask_mlp[i].w),
				  0.1f, state);
		if (head->mask_mlp[i].b)
			fill_prng((float *)head->mask_mlp[i].b->data,
				  sam3_tensor_nelems(head->mask_mlp[i].b),
				  0.05f, state);
	}
}

/*
 * dump_seg_head_fixture - Run the seg head on small synthetic inputs.
 *
 * Skips the prompt cross-attention branch — the migration only touches
 * the FPN + instance projection + mask-embed dot product, so only the
 * encoder state, FPN skip features, and final mask logits are captured.
 */
static enum sam3_error dump_seg_head_fixture(const char *out_dir,
					     struct sam3_cpu_backend *cpu)
{
	uint32_t rng = GEN_NHWC_SEED ^ 0x11111111u;
	struct sam3_seg_head head;
	struct sam3_graph graph;
	struct sam3_tensor *enc, *feat_2x, *feat_4x;
	struct sam3_tensor *queries, *masks;
	enum sam3_error err;

	err = sam3_seg_head_init(&head, SEG_D_MODEL, SEG_N_HEADS);
	if (err != SAM3_OK) {
		sam3_log_error("seg_head_init failed (%d)", err);
		return err;
	}

	err = sam3_seg_head_load(&head, NULL, &cpu->arena);
	if (err != SAM3_OK) {
		sam3_log_error("seg_head_load failed (%d)", err);
		return err;
	}

	fill_weight_tensors_seg(&head, &rng);

	int enc_dims[] = {SEG_SEQ, SEG_D_MODEL};
	int f2_dims[]  = {1, SEG_D_MODEL, SEG_FEAT2_H, SEG_FEAT2_W};
	int f4_dims[]  = {1, SEG_D_MODEL, SEG_FEAT4_H, SEG_FEAT4_W};
	int q_dims[]   = {SEG_N_QUERIES, SEG_D_MODEL};

	enc = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 2, enc_dims);
	feat_2x = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 4, f2_dims);
	feat_4x = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 4, f4_dims);
	queries = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 2, q_dims);
	if (!enc || !feat_2x || !feat_4x || !queries) {
		sam3_log_error("seg_head input alloc failed");
		return SAM3_ENOMEM;
	}

	fill_prng((float *)enc->data,
		  sam3_tensor_nelems(enc), 0.3f, &rng);
	fill_prng((float *)feat_2x->data,
		  sam3_tensor_nelems(feat_2x), 0.3f, &rng);
	fill_prng((float *)feat_4x->data,
		  sam3_tensor_nelems(feat_4x), 0.3f, &rng);
	fill_prng((float *)queries->data,
		  sam3_tensor_nelems(queries), 0.3f, &rng);

	err = dump_tensor(out_dir, "seg_head_enc.bin", enc);
	if (err != SAM3_OK)
		return err;
	err = dump_tensor(out_dir, "seg_head_feat_2x.bin", feat_2x);
	if (err != SAM3_OK)
		return err;
	err = dump_tensor(out_dir, "seg_head_feat_4x.bin", feat_4x);
	if (err != SAM3_OK)
		return err;

	sam3_graph_init(&graph);

	masks = sam3_seg_head_build(&head, &graph, queries, enc,
				    feat_2x, feat_4x,
				    SEG_ENC_H, SEG_ENC_W,
				    &cpu->arena);
	if (!masks) {
		sam3_log_error("seg_head_build returned NULL");
		return SAM3_ENOMEM;
	}

	err = cpu->base.ops->graph_eval(&cpu->base, &graph);
	if (err != SAM3_OK) {
		sam3_log_error("seg_head graph_eval failed (%d)", err);
		return err;
	}

	err = dump_tensor(out_dir, "seg_head_masks.bin", masks);
	if (err != SAM3_OK)
		return err;

	sam3_log_info("seg_head fixture written (seq %d, d %d)",
		      SEG_SEQ, SEG_D_MODEL);
	return SAM3_OK;
}

/*
 * fill_weight_tensors_mask_dec - Populate mask decoder weights with PRNG.
 *
 * Walks every tensor the loader touches. Zero-weights would give
 * degenerate outputs (masks all equal), which is not useful as a
 * regression check.
 */
static void fill_weight_tensors_mask_dec(struct sam3_mask_decoder *dec,
					 uint32_t *state)
{
	/* Learned tokens — uniform small PRNG. */
	fill_prng((float *)dec->mask_tokens->data,
		  sam3_tensor_nelems(dec->mask_tokens), 0.1f, state);
	fill_prng((float *)dec->iou_token->data,
		  sam3_tensor_nelems(dec->iou_token), 0.1f, state);
	fill_prng((float *)dec->obj_score_token->data,
		  sam3_tensor_nelems(dec->obj_score_token), 0.1f, state);

	for (int l = 0; l < SAM3_MASK_DEC_LAYERS; l++) {
		struct sam3_tensor *t[] = {
			dec->layers[l].sa_qkv_w,  dec->layers[l].sa_qkv_b,
			dec->layers[l].sa_out_w,  dec->layers[l].sa_out_b,
			dec->layers[l].ca_ti_q_w, dec->layers[l].ca_ti_q_b,
			dec->layers[l].ca_ti_k_w, dec->layers[l].ca_ti_k_b,
			dec->layers[l].ca_ti_v_w, dec->layers[l].ca_ti_v_b,
			dec->layers[l].ca_ti_out_w,
			dec->layers[l].ca_ti_out_b,
			dec->layers[l].mlp_fc1_w, dec->layers[l].mlp_fc1_b,
			dec->layers[l].mlp_fc2_w, dec->layers[l].mlp_fc2_b,
			dec->layers[l].ca_it_q_w, dec->layers[l].ca_it_q_b,
			dec->layers[l].ca_it_k_w, dec->layers[l].ca_it_k_b,
			dec->layers[l].ca_it_v_w, dec->layers[l].ca_it_v_b,
			dec->layers[l].ca_it_out_w,
			dec->layers[l].ca_it_out_b,
		};
		int nt = (int)(sizeof(t) / sizeof(t[0]));
		for (int i = 0; i < nt; i++) {
			if (!t[i])
				continue;
			fill_prng((float *)t[i]->data,
				  sam3_tensor_nelems(t[i]), 0.05f, state);
		}

		/* LayerNorm gammas: PRNG + 1 so scale is nonzero. */
		struct sam3_tensor *lnw[] = {
			dec->layers[l].ln1_w, dec->layers[l].ln2_w,
			dec->layers[l].ln3_w, dec->layers[l].ln4_w,
		};
		struct sam3_tensor *lnb[] = {
			dec->layers[l].ln1_b, dec->layers[l].ln2_b,
			dec->layers[l].ln3_b, dec->layers[l].ln4_b,
		};
		for (int i = 0; i < 4; i++) {
			if (lnw[i]) {
				float *w = (float *)lnw[i]->data;
				int n = sam3_tensor_nelems(lnw[i]);
				fill_prng(w, n, 0.05f, state);
				for (int k = 0; k < n; k++)
					w[k] += 1.0f;
			}
			if (lnb[i])
				fill_prng((float *)lnb[i]->data,
					  sam3_tensor_nelems(lnb[i]),
					  0.02f, state);
		}
	}

	struct sam3_tensor *final_t[] = {
		dec->final_q_w, dec->final_q_b,
		dec->final_k_w, dec->final_k_b,
		dec->final_v_w, dec->final_v_b,
		dec->final_out_w, dec->final_out_b,
	};
	for (int i = 0; i < (int)(sizeof(final_t) / sizeof(final_t[0])); i++) {
		if (!final_t[i])
			continue;
		fill_prng((float *)final_t[i]->data,
			  sam3_tensor_nelems(final_t[i]), 0.05f, state);
	}
	if (dec->final_ln_w) {
		float *w = (float *)dec->final_ln_w->data;
		int n = sam3_tensor_nelems(dec->final_ln_w);
		fill_prng(w, n, 0.05f, state);
		for (int k = 0; k < n; k++)
			w[k] += 1.0f;
	}
	if (dec->final_ln_b)
		fill_prng((float *)dec->final_ln_b->data,
			  sam3_tensor_nelems(dec->final_ln_b),
			  0.02f, state);

	/* Pixel decoder conv transposes, layer norm, skip convs. */
	struct sam3_tensor *px_t[] = {
		dec->up_conv1_w, dec->up_conv1_b,
		dec->up_conv2_w, dec->up_conv2_b,
		dec->conv_s0_w, dec->conv_s0_b,
		dec->conv_s1_w, dec->conv_s1_b,
	};
	for (int i = 0; i < (int)(sizeof(px_t) / sizeof(px_t[0])); i++) {
		if (!px_t[i])
			continue;
		fill_prng((float *)px_t[i]->data,
			  sam3_tensor_nelems(px_t[i]), 0.1f, state);
	}
	if (dec->up_ln_w) {
		float *w = (float *)dec->up_ln_w->data;
		int n = sam3_tensor_nelems(dec->up_ln_w);
		fill_prng(w, n, 0.05f, state);
		for (int k = 0; k < n; k++)
			w[k] += 1.0f;
	}
	if (dec->up_ln_b)
		fill_prng((float *)dec->up_ln_b->data,
			  sam3_tensor_nelems(dec->up_ln_b),
			  0.02f, state);

	/* Hypernetwork and IoU MLPs. */
	for (int i = 0; i < SAM3_MASK_DEC_MASKS; i++) {
		struct sam3_tensor *h[] = {
			dec->hyper[i].proj_in_w, dec->hyper[i].proj_in_b,
			dec->hyper[i].hidden_w, dec->hyper[i].hidden_b,
			dec->hyper[i].proj_out_w, dec->hyper[i].proj_out_b,
		};
		for (int k = 0; k < (int)(sizeof(h) / sizeof(h[0])); k++) {
			if (!h[k])
				continue;
			fill_prng((float *)h[k]->data,
				  sam3_tensor_nelems(h[k]), 0.05f, state);
		}
	}

	struct sam3_tensor *iou_t[] = {
		dec->iou_proj_in_w, dec->iou_proj_in_b,
		dec->iou_hidden_w, dec->iou_hidden_b,
		dec->iou_proj_out_w, dec->iou_proj_out_b,
	};
	for (int i = 0; i < (int)(sizeof(iou_t) / sizeof(iou_t[0])); i++) {
		if (!iou_t[i])
			continue;
		fill_prng((float *)iou_t[i]->data,
			  sam3_tensor_nelems(iou_t[i]), 0.05f, state);
	}

	if (dec->no_mask_embed)
		fill_prng((float *)dec->no_mask_embed->data,
			  sam3_tensor_nelems(dec->no_mask_embed),
			  0.05f, state);
	if (dec->pe_gaussian)
		fill_prng((float *)dec->pe_gaussian->data,
			  sam3_tensor_nelems(dec->pe_gaussian),
			  0.5f, state);
}

/*
 * dump_mask_dec_fixture - Run mask decoder on tiny synthetic inputs.
 *
 * The grid is kept to 4x4 so feat_s0 (16x16x256 floats) and the mask
 * output (4x16x16) are both small. Even the smallest run exercises the
 * full pipeline — the pixel decoder's conv transposes and skip adds are
 * what the NHWC migration will modify, so the captured mask logits are
 * the end-to-end reference.
 */
static enum sam3_error dump_mask_dec_fixture(const char *out_dir,
					     struct sam3_cpu_backend *cpu)
{
	uint32_t rng = GEN_NHWC_SEED ^ 0x22222222u;
	struct sam3_mask_decoder dec;
	struct sam3_graph graph;
	struct sam3_tensor *keys, *feat_s0, *feat_s1;
	struct sam3_tensor *masks = NULL;
	struct sam3_tensor *iou = NULL;
	enum sam3_error err;

	err = sam3_mask_decoder_init(&dec);
	if (err != SAM3_OK) {
		sam3_log_error("mask_decoder_init failed (%d)", err);
		return err;
	}

	err = sam3_mask_decoder_load(&dec, NULL, &cpu->arena);
	if (err != SAM3_OK) {
		sam3_log_error("mask_decoder_load failed (%d)", err);
		return err;
	}

	fill_weight_tensors_mask_dec(&dec, &rng);

	int keys_dims[]    = {MD_N_PIX, MD_D_MODEL};
	int s1_dims[]      = {1, MD_D_MODEL, MD_FEAT_S1_H, MD_FEAT_S1_W};
	int s0_dims[]      = {1, MD_D_MODEL, MD_FEAT_S0_H, MD_FEAT_S0_W};

	keys = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 2, keys_dims);
	feat_s1 = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 4, s1_dims);
	feat_s0 = gh_alloc_tensor(&cpu->arena, SAM3_DTYPE_F32, 4, s0_dims);
	if (!keys || !feat_s0 || !feat_s1) {
		sam3_log_error("mask_dec input alloc failed");
		return SAM3_ENOMEM;
	}

	fill_prng((float *)keys->data,
		  sam3_tensor_nelems(keys), 0.3f, &rng);
	fill_prng((float *)feat_s0->data,
		  sam3_tensor_nelems(feat_s0), 0.3f, &rng);
	fill_prng((float *)feat_s1->data,
		  sam3_tensor_nelems(feat_s1), 0.3f, &rng);

	err = dump_tensor(out_dir, "mask_dec_keys.bin", keys);
	if (err != SAM3_OK)
		return err;
	err = dump_tensor(out_dir, "mask_dec_feat_s0.bin", feat_s0);
	if (err != SAM3_OK)
		return err;
	err = dump_tensor(out_dir, "mask_dec_feat_s1.bin", feat_s1);
	if (err != SAM3_OK)
		return err;

	sam3_graph_init(&graph);

	err = sam3_mask_decoder_build(&dec, &graph, keys,
				      MD_GRID_H, MD_GRID_W,
				      NULL, feat_s0, feat_s1,
				      &cpu->arena, &masks, &iou, NULL,
				      NULL, NULL);
	if (err != SAM3_OK || !masks) {
		sam3_log_error("mask_decoder_build failed (%d)", err);
		return err ? err : SAM3_ENOMEM;
	}

	err = cpu->base.ops->graph_eval(&cpu->base, &graph);
	if (err != SAM3_OK) {
		sam3_log_error("mask_dec graph_eval failed (%d)", err);
		return err;
	}

	err = dump_tensor(out_dir, "mask_dec_output.bin", masks);
	if (err != SAM3_OK)
		return err;

	sam3_log_info("mask_dec fixture written (grid %dx%d)",
		      MD_GRID_H, MD_GRID_W);
	return SAM3_OK;
}

/*
 * ensure_dir - Create out_dir if it does not already exist.
 *
 * Only creates the final component; the parent path must already
 * exist. Returns SAM3_OK if the directory now exists and is writable.
 */
static enum sam3_error ensure_dir(const char *path)
{
	struct stat st;

	if (stat(path, &st) == 0) {
		if (!S_ISDIR(st.st_mode)) {
			sam3_log_error("%s exists but is not a directory",
				       path);
			return SAM3_EIO;
		}
		return SAM3_OK;
	}

	if (mkdir(path, 0775) != 0) {
		sam3_log_error("mkdir %s failed: %s",
			       path, strerror(errno));
		return SAM3_EIO;
	}
	return SAM3_OK;
}

int main(int argc, char **argv)
{
	struct sam3_backend *be = NULL;
	struct sam3_cpu_backend *cpu = NULL;
	enum sam3_error err;
	const char *out_dir;

	if (argc != 2) {
		fprintf(stderr,
			"usage: %s <output_dir>\n", argv[0]);
		return 1;
	}
	out_dir = argv[1];

	sam3_log_set_level(SAM3_LOG_INFO);

	err = ensure_dir(out_dir);
	if (err != SAM3_OK)
		return 1;

	be = sam3_backend_init(SAM3_BACKEND_CPU);
	if (!be) {
		sam3_log_error("sam3_backend_init(CPU) failed");
		return 1;
	}
	cpu = (struct sam3_cpu_backend *)be;

	err = dump_neck_fixture(out_dir, cpu);
	if (err != SAM3_OK)
		goto cleanup;

	err = dump_seg_head_fixture(out_dir, cpu);
	if (err != SAM3_OK)
		goto cleanup;

	err = dump_mask_dec_fixture(out_dir, cpu);
	if (err != SAM3_OK)
		goto cleanup;

	sam3_log_info("all NHWC migration fixtures written to %s",
		      out_dir);

cleanup:
	sam3_backend_free(be);
	return (err == SAM3_OK) ? 0 : 1;
}
