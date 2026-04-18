/*
 * src/model/mask_decoder.h - SAM3 mask decoder (two-way transformer)
 *
 * Defines the mask decoder that produces segmentation masks and IoU
 * scores from image features and prompt embeddings. Architecture:
 *   - 6 learned tokens: 4 mask + 1 IoU + 1 obj_score
 *   - 2-layer two-way transformer with self-attention, bidirectional
 *     cross-attention (token<->image), and MLP
 *   - Pixel decoder: 2 transposed convolutions with GELU
 *   - 4 hypernetwork MLPs converting mask tokens to 32-dim embeddings
 *   - IoU prediction MLP
 *
 * Cross-attention uses 128-dim internal projections (not 256-dim),
 * implemented via custom attention that operates in reduced space.
 *
 * Key types:  sam3_mask_decoder
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3_image.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MASK_DECODER_H
#define SAM3_MODEL_MASK_DECODER_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_MASK_DEC_LAYERS  2
#define SAM3_MASK_DEC_MASKS   4
#define SAM3_MASK_DEC_D_INNER 128

struct sam3_mask_decoder {
	int d_model;	/* 256 */
	int d_inner;	/* 128 — cross-attention internal dim */
	int d_pixel;	/* 32  — pixel decoder output dim */
	int n_heads;	/* 8 */
	int n_masks;	/* 4 */

	/* Learned tokens */
	struct sam3_tensor *mask_tokens;	/* [4, 256] */
	struct sam3_tensor *iou_token;		/* [1, 256] */
	struct sam3_tensor *obj_score_token;	/* [1, 256] */

	/* 2-layer two-way transformer */
	struct {
		/* Self-attention on tokens (256-dim, 8 heads) */
		struct sam3_tensor *sa_qkv_w, *sa_qkv_b;
		struct sam3_tensor *sa_out_w, *sa_out_b;
		struct sam3_tensor *ln1_w, *ln1_b;

		/* Cross-attention: token queries -> image KV (128-dim) */
		struct sam3_tensor *ca_ti_q_w, *ca_ti_q_b;
		struct sam3_tensor *ca_ti_k_w, *ca_ti_k_b;
		struct sam3_tensor *ca_ti_v_w, *ca_ti_v_b;
		struct sam3_tensor *ca_ti_out_w, *ca_ti_out_b;
		struct sam3_tensor *ln2_w, *ln2_b;

		/* FFN: ReLU MLP */
		struct sam3_tensor *mlp_fc1_w, *mlp_fc1_b;
		struct sam3_tensor *mlp_fc2_w, *mlp_fc2_b;
		struct sam3_tensor *ln3_w, *ln3_b;

		/* Cross-attention: image queries -> token KV (128-dim) */
		struct sam3_tensor *ca_it_q_w, *ca_it_q_b;
		struct sam3_tensor *ca_it_k_w, *ca_it_k_b;
		struct sam3_tensor *ca_it_v_w, *ca_it_v_b;
		struct sam3_tensor *ca_it_out_w, *ca_it_out_b;
		struct sam3_tensor *ln4_w, *ln4_b;
	} layers[SAM3_MASK_DEC_LAYERS];

	/* Final cross-attention: token -> image (128-dim) */
	struct sam3_tensor *final_q_w, *final_q_b;
	struct sam3_tensor *final_k_w, *final_k_b;
	struct sam3_tensor *final_v_w, *final_v_b;
	struct sam3_tensor *final_out_w, *final_out_b;
	struct sam3_tensor *final_ln_w, *final_ln_b;

	/* Pixel decoder: transposed convolutions (OHWI after Task 10) */
	struct sam3_tensor *up_conv1_w, *up_conv1_b;	/* [64,2,2,256]/[64] */
	struct sam3_tensor *up_ln_w, *up_ln_b;		/* [64] */
	struct sam3_tensor *up_conv2_w, *up_conv2_b;	/* [32,2,2,64]/[32] */

	/* 4 hypernetwork MLPs: [256] -> [256] -> [256] -> [32] */
	struct {
		struct sam3_tensor *proj_in_w, *proj_in_b;
		struct sam3_tensor *hidden_w, *hidden_b;
		struct sam3_tensor *proj_out_w, *proj_out_b;
	} hyper[SAM3_MASK_DEC_MASKS];

	/* IoU prediction MLP: [256] -> [256] -> [256] -> [4] */
	struct sam3_tensor *iou_proj_in_w, *iou_proj_in_b;
	struct sam3_tensor *iou_hidden_w, *iou_hidden_b;
	struct sam3_tensor *iou_proj_out_w, *iou_proj_out_b;

	/*
	 * Object-score prediction MLP: 3-layer MLP on obj_score_token.
	 * Python: pred_obj_score_head = MLP(256, 256, 1, 3).
	 * Layers: [256->256]-relu-[256->256]-relu-[256->1].
	 */
	struct sam3_tensor *obj_score_fc0_w, *obj_score_fc0_b; /* [256,256]/[256] */
	struct sam3_tensor *obj_score_fc1_w, *obj_score_fc1_b; /* [256,256]/[256] */
	struct sam3_tensor *obj_score_fc2_w, *obj_score_fc2_b; /* [1,256]/[1] */

	/* Multi-scale skip connections: conv_s0 (32) and conv_s1 (64)
	 * (OHWI after Task 10) */
	struct sam3_tensor *conv_s0_w, *conv_s0_b;	/* [32,1,1,256]/[32] */
	struct sam3_tensor *conv_s1_w, *conv_s1_b;	/* [64,1,1,256]/[64] */

	/* Dense prompt embedding when no mask prompt is provided */
	struct sam3_tensor *no_mask_embed;		/* [1, 256] */

	/* Gaussian matrix for positional encoding */
	struct sam3_tensor *pe_gaussian;		/* [2, 128] */

	/* Debug: set by build, valid after eval (temporary) */
	struct sam3_tensor *_debug_px;
	struct sam3_tensor *_debug_hyper;
	struct sam3_tensor *_debug_keys;
	struct sam3_tensor *_debug_queries;
};

/*
 * sam3_mask_decoder_init - Initialize mask decoder with SAM3 config.
 *
 * @dec: Mask decoder struct (caller-allocated)
 *
 * Returns SAM3_OK.
 */
enum sam3_error sam3_mask_decoder_init(struct sam3_mask_decoder *dec);

/*
 * sam3_mask_decoder_load - Load mask decoder weights.
 *
 * @dec:   Initialized mask decoder
 * @wf:    Weight file (NULL for zero-init fallback)
 * @arena: Arena for weight tensors
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if arena full.
 */
enum sam3_error sam3_mask_decoder_load(struct sam3_mask_decoder *dec,
				       const struct sam3_weight_file *wf,
				       struct sam3_arena *arena);

/*
 * sam3_mask_decoder_build - Build mask decoder compute graph.
 *
 * Produces 4 mask logits and IoU scores from image features
 * and optional prompt tokens.
 *
 * @dec:       Loaded mask decoder
 * @g:         Graph to populate
 * @img_feat:  [n_pixels, d_model] fused image features, row-major
 *             (h*W + w, c) so it reshapes to NHWC as a pure view
 * @grid_h:    Spatial height of image features
 * @grid_w:    Spatial width of image features
 * @prompt:    [n_prompt, d_model] sparse prompt tokens, or NULL
 * @feat_s0:   [1, 4H, 4W, d_model] 2x-scale backbone feature (NHWC),
 *             or NULL
 * @feat_s1:   [1, 2H, 2W, d_model] 1x-scale backbone feature (NHWC),
 *             or NULL
 * @arena:         Arena for intermediates
 * @out_masks:     Receives [4, H, W] mask logits
 * @out_iou:       Receives [4] IoU predictions, or NULL
 * @out_obj_token: Optional. If non-NULL, receives the object-score
 *                 token of shape [1, d_model] (2-D view sliced from
 *                 the transformer token stack). Pass NULL to skip.
 * @out_obj_score_logits: Optional [1, 1] scalar logit from
 *                 pred_obj_score_head(obj_score_token). Positive means
 *                 object visible. Pass NULL to skip.
 * @out_mask_tokens: Optional [4, d_model] transformer-processed mask
 *                 tokens, used by the tracker to select the best-IoU
 *                 token for obj_ptr_proj. Pass NULL to skip.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM on allocation failure.
 */
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
	struct sam3_tensor **out_iou,
	struct sam3_tensor **out_obj_token,
	struct sam3_tensor **out_obj_score_logits,
	struct sam3_tensor **out_mask_tokens);

/*
 * sam3_mask_decoder_select_with_stability - Pick best mask using stability.
 *
 * @logits:           [n_masks * H * W] row-major f32 mask logits.
 * @iou_scores:       [n_masks] predicted IoU scores.
 * @n_masks:          Candidate count. Stability gating only kicks in for
 *                    n_masks >= 3; otherwise returns argmax(iou).
 * @H, @W:            Mask spatial dims.
 * @delta:            Perturbation offset (e.g. 0.05). 0 → falls back to
 *                    argmax(iou).
 * @stability_thresh: Stability floor (e.g. 0.98). 0 → falls back.
 *
 * Returns the chosen mask index in [0, n_masks). Among masks with
 * stability >= stability_thresh, prefers the highest IoU; if none
 * qualify, plain argmax(iou). Exposed for testing; the video path
 * calls it with session opts filled in.
 */
int sam3_mask_decoder_select_with_stability(const float *logits,
					     const float *iou_scores,
					     int n_masks, int H, int W,
					     float delta,
					     float stability_thresh);

#endif /* SAM3_MODEL_MASK_DECODER_H */
