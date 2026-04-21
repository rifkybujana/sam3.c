/*
 * src/model/vl_combiner.h - Vision-language backbone combiner
 *
 * Wraps the image encoder (Hiera ViT or EfficientViT), feature pyramid
 * neck, text encoder iface (CLIP or MobileCLIP), BPE tokenizer, and 2D
 * position encoding into a single composite module. This is a structural
 * wrapper that organizes sub-modules and provides unified init/load/build
 * entry points for the SAM3 inference pipeline. Backbone dispatch is
 * based on backbone_type.
 *
 * Key types:  sam3_vl_backbone
 * Depends on: image_encoder.h, image_encoder_efficientvit.h, necks.h,
 *             text_encoder.h, text_encoder_iface.h, tokenizer.h,
 *             position_encoding.h
 * Used by:    sam3.c (top-level image model)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_VL_COMBINER_H
#define SAM3_MODEL_VL_COMBINER_H

#include "image_encoder.h"
#include "image_encoder_efficientvit.h"
#include "image_encoder_tinyvit.h"
#include "necks.h"
#include "text_encoder.h"
#include "text_encoder_iface.h"
#include "tokenizer.h"
#include "position_encoding.h"

struct sam3_vl_backbone {
	int backbone_type;	/* enum sam3_backbone_type */
	union {
		struct sam3_vit vit;
		struct sam3_efficientvit evit;
		struct sam3_tinyvit tvit;
	} enc;
	struct sam3_neck neck;
	struct sam3_neck sam2_neck;	/* sam2_fpn_layers (EfficientSAM3) */
	int has_sam2_neck;		/* 1 if sam2_neck is initialized */
	struct sam3_text_encoder_iface text_iface;
	struct sam3_tokenizer tokenizer;
	struct sam3_pos_encoding pos_enc;
	int scalp;	/* 1 -- number of scales to keep from the end */
};

/*
 * sam3_vl_backbone_init - Initialize the VL backbone.
 *
 * @vl:            VL backbone struct (caller-allocated, zeroed)
 * @backbone_type: SAM3_BACKBONE_HIERA or SAM3_BACKBONE_EFFICIENTVIT
 * @n_fpn_scales:  Number of FPN scales for the main neck (3 for SAM 3.1,
 *                 4 for SAM 3). The tracker-side sam2_neck always uses
 *                 4 scales regardless.
 * @arena:         Arena for precomputation (RoPE, position encoding)
 *
 * Dispatches to the appropriate image encoder init based on backbone_type.
 * Sets up encoder, neck, tokenizer, and position encoding.
 *
 * Returns SAM3_OK on success, or an error code on failure.
 */
enum sam3_error sam3_vl_backbone_init(struct sam3_vl_backbone *vl,
				      int backbone_type,
				      int n_fpn_scales,
				      struct sam3_arena *arena);

/*
 * sam3_vl_backbone_load - Load all sub-module weights.
 *
 * @vl:    Initialized VL backbone
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Loads ViT, neck, and text encoder weights from the weight file.
 * Returns SAM3_OK on success, or an error code on failure.
 */
enum sam3_error sam3_vl_backbone_load(struct sam3_vl_backbone *vl,
				      const struct sam3_weight_file *wf,
				      struct sam3_arena *arena);

/*
 * sam3_vl_backbone_free - Free non-arena resources (tokenizer).
 *
 * @vl: VL backbone to free (may be NULL).
 *
 * Only the tokenizer allocates heap memory; all other sub-modules
 * use arena allocation and do not need explicit freeing.
 */
void sam3_vl_backbone_free(struct sam3_vl_backbone *vl);

/*
 * sam3_vl_backbone_img_size - Return the encoder's actual input image size.
 *
 * @vl: Initialized VL backbone.
 *
 * Returns the img_size that the image encoder was configured with
 * (e.g. 512 for EfficientViT, 1008 for Hiera/TinyViT).
 */
int sam3_vl_backbone_img_size(const struct sam3_vl_backbone *vl);

/*
 * sam3_vl_backbone_build_vision - Run vision pipeline with per-block eval.
 *
 * Evaluates the ViT per-block (resetting scratch between blocks),
 * then resets scratch and builds the neck graph for the caller to
 * evaluate.
 *
 * @vl:           Initialized and loaded backbone
 * @g:            Graph to build neck into (caller evaluates)
 * @be:           Backend for per-block ViT evaluation
 * @image:        Input image [3, img_size, img_size] normalized F32
 * @out_features: Array of 4 pointers filled with multi-scale features
 * @scratch:      Arena for intermediate tensors (reset between stages)
 * @persist:      Arena for persistent ViT output buffer
 * @profiler:     Profiler for sub-stage timing (may be NULL)
 *
 * Returns the raw ViT features [n_patches, embed_dim], or NULL on error.
 */
struct sam3_tensor *sam3_vl_backbone_build_vision(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *image,
	struct sam3_tensor *out_features[],
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_profiler *profiler);

/*
 * sam3_vl_backbone_build_vision_dual - As build_vision, but also produces
 * sam2-side features. When sam2_neck is initialized (has_sam2_neck=1),
 * sam2_features[] is populated with 4 multi-scale tensors from the
 * sam2 neck. The video tracker uses sam2_features as input to the SAM
 * mask decoder (Python's pix_feat_with_mem path).
 *
 * If sam2_features is NULL or has_sam2_neck=0, behaves identically to
 * sam3_vl_backbone_build_vision.
 */
struct sam3_tensor *sam3_vl_backbone_build_vision_dual(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *image,
	struct sam3_tensor *out_features[],
	struct sam3_tensor *sam2_features[],
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_profiler *profiler);

/*
 * sam3_vl_backbone_build_text - Build text pipeline graph.
 *
 * @vl:         Initialized and loaded backbone
 * @g:          Graph to add nodes to
 * @text:       Null-terminated text prompt string
 * @pooled_out: Receives pooled text feature [d_model]
 * @arena:      Arena for intermediate tensors
 *
 * Tokenizes text, runs text encoder. Returns per-token embeddings
 * [seq_len, d_model], or NULL on error.
 */
struct sam3_tensor *sam3_vl_backbone_build_text(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	const char *text,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena);

/*
 * sam3_vl_backbone_set_text_backbone - Reinitialize the text iface.
 *
 * Called by the loader after reading the .sam3 v4 header so the iface
 * matches the file's text_backbone. Must be called BEFORE
 * sam3_vl_backbone_load.
 *
 * @vl:            Initialized VL backbone
 * @text_backbone: enum sam3_text_backbone (SAM3_TEXT_CLIP or MOBILECLIP_*)
 * @arena:         Arena used during vl_backbone_init
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for unknown backbone.
 */
enum sam3_error sam3_vl_backbone_set_text_backbone(
	struct sam3_vl_backbone *vl, int text_backbone,
	struct sam3_arena *arena);

#endif /* SAM3_MODEL_VL_COMBINER_H */
