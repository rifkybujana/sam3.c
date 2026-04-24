/*
 * src/model/segmentation.h - SAM3 UniversalSegmentationHead
 *
 * Implements the MaskFormer-style segmentation head used for text/box
 * prompts. Contains an FPN PixelDecoder (interpolate + skip add + 3×3
 * conv + GroupNorm(8) + ReLU), instance projection (1×1 conv), mask
 * embedder (3-layer MLP), and optional prompt cross-attention. Mask
 * logits are computed via dot product of embedded queries and instance
 * pixel features.
 *
 * Weight prefix: detector_model.mask_decoder.*
 *
 * Key types:  sam3_seg_head
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3_image.c, tests/test_batched_ops.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_SEGMENTATION_H
#define SAM3_MODEL_SEGMENTATION_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_SEG_FPN_STAGES       3
#define SAM3_SEG_MASK_MLP_LAYERS  3
#define SAM3_SEG_GN_GROUPS        8

struct sam3_seg_head {
	int d_model;       /* 256 */
	int n_attn_heads;  /* 8 for prompt cross-attention */

	/* PixelDecoder FPN: 3 stages of 3×3 conv + GroupNorm(8) + ReLU */
	struct {
		struct sam3_tensor *conv_w; /* [d, d, 3, 3] */
		struct sam3_tensor *conv_b; /* [d] */
		struct sam3_tensor *gn_w;   /* [d] */
		struct sam3_tensor *gn_b;   /* [d] */
	} fpn[SAM3_SEG_FPN_STAGES];

	/* Instance projection: 1×1 conv [d → d] after pixel decoder */
	struct sam3_tensor *inst_proj_w; /* [d, d, 1, 1] */
	struct sam3_tensor *inst_proj_b; /* [d] */

	/* Mask embedder: 3-layer MLP on query embeddings */
	struct {
		struct sam3_tensor *w; /* [d, d] */
		struct sam3_tensor *b; /* [d] */
	} mask_mlp[SAM3_SEG_MASK_MLP_LAYERS];

	/* Prompt cross-attention: separate Q/K/V/O projections */
	struct sam3_tensor *pxattn_q_w, *pxattn_q_b;
	struct sam3_tensor *pxattn_k_w, *pxattn_k_b;
	struct sam3_tensor *pxattn_v_w, *pxattn_v_b;
	struct sam3_tensor *pxattn_o_w, *pxattn_o_b;
	struct sam3_tensor *pxattn_norm_w, *pxattn_norm_b;

	/* Debug: intermediate tensors (valid after build, data after eval) */
	struct sam3_tensor *_debug_pixel_embed; /* FPN output [1,H,W,d] */
	struct sam3_tensor *_debug_inst;        /* inst proj [1,H,W,d] */
	struct sam3_tensor *_debug_mask_embed;  /* MLP output [nq,d] */
	struct sam3_tensor *_debug_enc_nchw;    /* encoder NHWC [1,h,w,d] */
};

/*
 * sam3_seg_head_init - Initialize segmentation head.
 *
 * @head:         Seg head struct (caller-allocated)
 * @d_model:      Model dimension (256)
 * @n_attn_heads: Number of heads for prompt cross-attention (8)
 */
enum sam3_error sam3_seg_head_init(struct sam3_seg_head *head,
				   int d_model, int n_attn_heads);

/*
 * sam3_seg_head_load - Load weights from weight file.
 *
 * Weight prefix: detector_model.mask_decoder.*
 * Falls back to zero-initialized tensors when weights are missing.
 */
enum sam3_error sam3_seg_head_load(struct sam3_seg_head *head,
				   const struct sam3_weight_file *wf,
				   struct sam3_arena *arena);

/*
 * sam3_seg_head_build - Build segmentation head compute graph.
 *
 * @head:           Loaded seg head
 * @g:              Graph to add nodes to
 * @queries:        DETR decoder output [n_queries, d_model]
 * @encoder_states: Encoder output [seq, d_model] (seq = enc_h * enc_w)
 * @feat_2x:        Backbone feature at 2× [1, d, H2, W2]
 * @feat_4x:        Backbone feature at 4× [1, d, H4, W4]
 * @enc_h, enc_w:   Spatial dims of encoder output (72×72)
 * @arena:          Arena for intermediate tensors
 *
 * Pipeline:
 *  1. Reshape encoder_states to NHWC [1, enc_h, enc_w, d] (72×72)
 *  2. FPN: interpolate + skip add + 3×3 conv + GroupNorm(8) + ReLU (×2)
 *  3. Instance projection: NHWC 1×1 conv on pixel features
 *  4. Mask embedder: 3-layer MLP on queries
 *  5. Dot product: mask_embed @ instance_features → mask logits
 *
 * Returns mask logits [n_queries, final_h, final_w], or NULL on error.
 */
struct sam3_tensor *sam3_seg_head_build(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_tensor *encoder_states,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	int enc_h, int enc_w,
	struct sam3_arena *arena);

/*
 * sam3_seg_head_build_cross_attn - Build prompt cross-attention graph.
 *
 * Must be evaluated as a separate graph before calling seg_head_build
 * when the caller needs the cross-attention result materialized, for
 * example to feed it into a follow-on graph.
 *
 * @head:           Loaded seg head
 * @g:              Graph to add nodes to
 * @encoder_states: Encoder output [seq, d_model]
 * @text_features:  Text encoder output [n_text, d_model]
 * @arena:          Arena for intermediate tensors
 *
 * Returns cross-attended encoder states [seq, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_seg_head_build_cross_attn(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *encoder_states,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena);

/*
 * sam3_seg_head_build_cross_attn_batched - Batched prompt cross-attention.
 *
 * Batched variant of sam3_seg_head_build_cross_attn. Every tensor gains
 * a leading batch dim B. Per-head SDPA uses the [B, 1, head_len, hd]
 * 4D reshape so the existing Metal 4D SDPA path handles B > 1.
 *
 * @head:           Loaded seg head
 * @g:              Graph to add nodes to
 * @encoder_states: [B, n_pixels, d_model]
 * @text_features:  [B, n_text, d_model]
 * @arena:          Arena for intermediate tensors
 *
 * Returns cross-attended encoder states [B, n_pixels, d_model], or NULL.
 */
struct sam3_tensor *sam3_seg_head_build_cross_attn_batched(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *encoder_states,
	struct sam3_tensor *text_features,
	struct sam3_arena *arena);

/*
 * Build FPN pixel decoder only (no instance projection). Operates on
 * NHWC tensors [1, H, W, d].
 */
struct sam3_tensor *sam3_seg_head_build_pixel_decoder(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nhwc,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *arena);

/*
 * sam3_seg_head_build_fpn - Build FPN pixel decoder + instance
 *                           projection.
 *
 * Must be evaluated and persisted before building the dot product.
 *
 * @head:     Loaded seg head
 * @g:        Graph to add nodes to
 * @enc_nhwc: Encoder output in NHWC [1, 72, 72, d]
 * @feat_2x:  Backbone feature at 2× [1, H2, W2, d]
 * @feat_4x:  Backbone feature at 4× [1, H4, W4, d]
 * @arena:    Arena for intermediate tensors
 *
 * Returns instance-projected pixel features [1, H4, W4, d].
 */
struct sam3_tensor *sam3_seg_head_build_fpn(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nhwc,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *arena);

/*
 * sam3_seg_head_build_pixel_decoder_batched - Batched FPN pixel decoder.
 *
 * Same as sam3_seg_head_build_pixel_decoder but accepts [B, H, W, d]
 * for every NHWC input. All ops in the FPN (gh_upsample, gh_add,
 * gh_conv2d, gh_groupnorm, gh_relu) are N-aware, so this is a naming-
 * consistency wrapper — the actual pipeline is the same.
 *
 * @head:     Loaded seg head
 * @g:        Graph to add nodes to
 * @enc_nhwc: Encoder states [B, enc_h, enc_w, d_model]
 * @feat_2x:  Backbone feature at 2x, already broadcast to [B, H2, W2, d]
 * @feat_4x:  Backbone feature at 4x, already broadcast to [B, H4, W4, d]
 * @arena:    Arena for intermediate tensors
 *
 * Callers are responsible for tiling shared image features to [B, ...]
 * before calling (use gh_broadcast_batch). Returns [B, H4, W4, d], or NULL.
 */
struct sam3_tensor *sam3_seg_head_build_pixel_decoder_batched(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nhwc,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *arena);

/*
 * sam3_seg_head_build_fpn_batched - Batched FPN + 1×1 instance projection.
 *
 * Same as sam3_seg_head_build_fpn but accepts [B, H, W, d] inputs and
 * returns [B, H, W, d]. Delegates to the N-aware build_pixel_decoder +
 * gh_conv2d with N=B.
 *
 * @head, @g, @enc_nhwc, @feat_2x, @feat_4x, @arena — see
 * sam3_seg_head_build_pixel_decoder_batched.
 *
 * Returns instance features [B, H4, W4, d], or NULL.
 */
struct sam3_tensor *sam3_seg_head_build_fpn_batched(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *enc_nhwc,
	struct sam3_tensor *feat_2x,
	struct sam3_tensor *feat_4x,
	struct sam3_arena *arena);

/*
 * sam3_seg_head_build_mask_embed - Build mask embedder MLP on queries.
 *
 * @head:    Loaded seg head
 * @g:       Graph to add nodes to
 * @queries: DETR decoder output [n_queries, d_model]
 * @arena:   Arena for intermediate tensors
 *
 * Returns mask embeddings [n_queries, d_model].
 */
struct sam3_tensor *sam3_seg_head_build_mask_embed(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *queries,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_SEGMENTATION_H */
