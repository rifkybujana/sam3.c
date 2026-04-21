/*
 * src/model/vl_combiner.c - Vision-language backbone combiner
 *
 * Implements the composite VL backbone that wires together the image
 * encoder (Hiera ViT, EfficientViT, or TinyViT), feature pyramid neck, CLIP text
 * encoder, BPE tokenizer, and 2D position encoding. Dispatches to the
 * correct encoder based on backbone_type. This module does not add new
 * computation -- it organizes the sub-modules and provides unified
 * init/load/build entry points for the SAM3 inference pipeline.
 *
 * Key types:  sam3_vl_backbone
 * Depends on: vl_combiner.h, graph_helpers.h
 * Used by:    sam3.c (top-level image model)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "vl_combiner.h"
#include "graph_helpers.h"
#include "util/log.h"

enum sam3_error sam3_vl_backbone_init(struct sam3_vl_backbone *vl,
				      int backbone_type,
				      int n_fpn_scales,
				      struct sam3_arena *arena)
{
	enum sam3_error err;
	int grid_size;
	int backbone_dim;

	vl->scalp = 1;
	vl->backbone_type = backbone_type;
	vl->has_sam2_neck = 0;
	vl->has_interactive_neck = 0;

	switch (backbone_type) {
	case SAM3_BACKBONE_HIERA:
		err = sam3_vit_init(&vl->enc.vit,
				     1008,	/* img_size */
				     14,	/* patch_size */
				     1024,	/* embed_dim */
				     32,	/* depth */
				     16,	/* n_heads */
				     24,	/* window_size */
				     4736,	/* mlp_dim = 1024 * 4.625 */
				     arena);
		if (err != SAM3_OK)
			return err;
		grid_size = vl->enc.vit.grid_size;
		backbone_dim = vl->enc.vit.embed_dim;
		break;

	case SAM3_BACKBONE_EFFICIENTVIT: {
		int width_list[] = {24, 48, 96, 192, 384};
		int depth_list[] = {1, 2, 3, 4, 6};
		err = sam3_efficientvit_init(&vl->enc.evit,
					      width_list, depth_list,
					      32,	/* attn_dim */
					      4,	/* expand_ratio */
					      512,	/* img_size */
					      1024);	/* embed_dim */
		if (err != SAM3_OK)
			return err;
		grid_size = vl->enc.evit.grid_size;
		backbone_dim = vl->enc.evit.embed_dim;
		break;
	}

	case SAM3_BACKBONE_TINYVIT: {
		int embed_dims[] = {96, 192, 384, 576};
		int depths[] = {2, 2, 6, 2};
		int num_heads[] = {3, 6, 12, 18};
		int window_sizes[] = {7, 7, 14, 7};
		err = sam3_tinyvit_init(&vl->enc.tvit,
					 embed_dims, depths,
					 num_heads, window_sizes,
					 4,	/* n_layers */
					 1008,	/* img_size */
					 1024,	/* embed_dim */
					 4);	/* mlp_ratio */
		if (err != SAM3_OK)
			return err;
		grid_size = vl->enc.tvit.grid_size;
		backbone_dim = vl->enc.tvit.embed_dim;
		break;
	}

	default:
		sam3_log_error("vl_backbone: unknown backbone_type %d",
			       backbone_type);
		return SAM3_EINVAL;
	}

	/* Init main neck: n_fpn_scales at {4x, 2x, 1x, [0.5x]} */
	float scales[] = {4.0f, 2.0f, 1.0f, 0.5f};
	if (n_fpn_scales < 1 || n_fpn_scales > 4) {
		sam3_log_error("vl_backbone: invalid n_fpn_scales %d",
			       n_fpn_scales);
		return SAM3_EINVAL;
	}
	err = sam3_neck_init(&vl->neck, 256, backbone_dim,
			      grid_size, n_fpn_scales, scales);
	if (err != SAM3_OK)
		return err;

	/*
	 * Tracker-side FPN neck. Loaded from the same destination prefix
	 * (`sam2_fpn_layers.*`) on disk regardless of whether the source
	 * checkpoint is SAM 3 (dual-neck: `sam2_convs`, 4 scales) or
	 * SAM 3.1 (tri-neck: `propagation_convs`, 3 scales). In both
	 * cases the tracker's memory-attn and mask decoder consume this
	 * neck's output as their image embedding — the detector-side
	 * `convs` neck (loaded separately as `vl->neck`) is trained with
	 * a different objective and produces different features.
	 */
	err = sam3_neck_init(&vl->sam2_neck, 256, backbone_dim,
			      grid_size, n_fpn_scales, scales);
	if (err != SAM3_OK)
		return err;
	vl->has_sam2_neck = 1;

	/*
	 * Interactive neck (SAM 3.1 tri-neck only). Feeds the interactive
	 * mask decoder's conv_s0/s1 skip path on seed frames (Python's
	 * `_use_mask_as_output` -> `_forward_sam_heads(is_interactive=True)`).
	 * The loader zeroes the weights when the checkpoint lacks
	 * `interactive_fpn_layers.*`, so the init is safe for SAM 3 too.
	 */
	err = sam3_neck_init(&vl->interactive_neck, 256, backbone_dim,
			      grid_size, n_fpn_scales, scales);
	if (err != SAM3_OK)
		return err;
	vl->has_interactive_neck = 1;

	/* Init tokenizer (byte-level fallback vocab) */
	err = sam3_tokenizer_init(&vl->tokenizer);
	if (err != SAM3_OK)
		return err;

	/*
	 * Init text encoder iface. Default to CLIP here; the variant
	 * passed to sam3_vl_backbone_init may not yet be wired (callers
	 * may overwrite via sam3_vl_backbone_set_text_backbone before
	 * load). The actual variant flows through sam3_load_model from
	 * the .sam3 v4 header.
	 */
	err = sam3_text_encoder_iface_init(&vl->text_iface,
					   SAM3_TEXT_CLIP, arena);
	if (err != SAM3_OK)
		return err;

	/*
	 * Lazy 2D sinusoidal position encoding for encoder grid.
	 * Python: num_pos_feats = input // 2 = 256 // 2 = 128.
	 * Output is [H, W, 128*2] = [grid, grid, 256] matching d_model.
	 * Actual computation deferred to first sam3_pos_encoding_get().
	 */
	vl->pos_enc.height = grid_size;
	vl->pos_enc.width = grid_size;
	vl->pos_enc.num_pos_feats = 128;
	vl->pos_enc.temperature = 10000.0f;
	vl->pos_enc.arena = arena;

	return SAM3_OK;
}

/*
 * Force eager computation of all lazy-init data (RoPE tables,
 * tiled pos_embed, 2D position encoding) so that it is included
 * in the arena offset before weights_end is saved. Without this,
 * arena rollbacks on repeated set_image/segment calls would
 * destroy lazily-computed data while stale pointers remain.
 */
static enum sam3_error precompute_lazy_data(struct sam3_vl_backbone *vl)
{
	/* Hiera: RoPE tables + tiled pos_embed */
	if (vl->backbone_type == SAM3_BACKBONE_HIERA) {
		enum sam3_error err = sam3_vit_precompute(&vl->enc.vit);
		if (err != SAM3_OK)
			return err;
	}

	/* 2D sinusoidal position encoding (all backbones) */
	if (!sam3_pos_encoding_get(&vl->pos_enc)) {
		sam3_log_error("vl_backbone: failed to precompute "
			       "position encoding");
		return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

enum sam3_error sam3_vl_backbone_load(struct sam3_vl_backbone *vl,
				      const struct sam3_weight_file *wf,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	switch (vl->backbone_type) {
	case SAM3_BACKBONE_HIERA:
		err = sam3_vit_load(&vl->enc.vit, wf, arena);
		break;
	case SAM3_BACKBONE_EFFICIENTVIT:
		err = sam3_efficientvit_load(&vl->enc.evit, wf, arena);
		break;
	case SAM3_BACKBONE_TINYVIT:
		err = sam3_tinyvit_load(&vl->enc.tvit, wf, arena);
		break;
	default:
		err = SAM3_EINVAL;
		break;
	}
	if (err != SAM3_OK)
		return err;

	err = sam3_neck_load(&vl->neck, wf, arena);
	if (err != SAM3_OK)
		return err;

	if (vl->has_sam2_neck) {
		err = sam3_neck_load_prefixed(
			&vl->sam2_neck, wf, arena,
			"detector_model.vision_encoder.neck."
			"sam2_fpn_layers.");
		if (err != SAM3_OK)
			return err;
		sam3_log_info("vl_backbone: sam2_fpn_layers loaded "
			      "(backbone_dim=%d)",
			      vl->sam2_neck.backbone_dim);
	}

	if (vl->has_interactive_neck) {
		const char *iact_prefix =
			"detector_model.vision_encoder.neck."
			"interactive_fpn_layers.";
		/*
		 * Probe for the first expected weight: only SAM 3.1 checkpoints
		 * ship interactive_convs / interactive_fpn_layers. For SAM 3
		 * we skip the load (and clear the flag) so the image encoder
		 * does not waste compute on a zero-weight neck.
		 */
		char probe[128];
		snprintf(probe, sizeof(probe), "%s0.proj1.weight", iact_prefix);
		if (wf && sam3_weight_find(wf, probe)) {
			err = sam3_neck_load_prefixed(
				&vl->interactive_neck, wf, arena,
				iact_prefix);
			if (err != SAM3_OK)
				return err;
			sam3_log_info("vl_backbone: interactive_fpn_layers "
				      "loaded (backbone_dim=%d)",
				      vl->interactive_neck.backbone_dim);
		} else {
			vl->has_interactive_neck = 0;
			sam3_log_debug("vl_backbone: interactive_fpn_layers "
				       "absent in checkpoint (SAM 3)");
		}
	}

	err = vl->text_iface.ops->load(&vl->text_iface, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Force lazy-init data to be computed now so that arena
	 * rollbacks in set_image/segment do not destroy them. */
	return precompute_lazy_data(vl);
}

int sam3_vl_backbone_img_size(const struct sam3_vl_backbone *vl)
{
	switch (vl->backbone_type) {
	case SAM3_BACKBONE_HIERA:
		return vl->enc.vit.img_size;
	case SAM3_BACKBONE_EFFICIENTVIT:
		return vl->enc.evit.img_size;
	case SAM3_BACKBONE_TINYVIT:
		return vl->enc.tvit.img_size;
	default:
		return 0;
	}
}

void sam3_vl_backbone_free(struct sam3_vl_backbone *vl)
{
	if (vl)
		sam3_tokenizer_free(&vl->tokenizer);
}

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

struct sam3_tensor *sam3_vl_backbone_build_vision(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *image,
	struct sam3_tensor *out_features[],
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_profiler *profiler)
{
	return sam3_vl_backbone_build_vision_dual(
		vl, g, be, image, out_features, NULL,
		scratch, persist, profiler);
}

struct sam3_tensor *sam3_vl_backbone_build_vision_dual(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	struct sam3_backend *be,
	struct sam3_tensor *image,
	struct sam3_tensor *out_features[],
	struct sam3_tensor *sam2_features[],
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_profiler *profiler)
{
	/*
	 * Run image encoder per-block: evaluates internally, returns
	 * materialized output in persist arena.
	 */
	struct sam3_tensor *enc_out;

	switch (vl->backbone_type) {
	case SAM3_BACKBONE_HIERA:
		enc_out = sam3_vit_build(&vl->enc.vit, be, image,
					  scratch, persist, profiler);
		break;
	case SAM3_BACKBONE_EFFICIENTVIT:
		enc_out = sam3_efficientvit_build(&vl->enc.evit, be, image,
						   scratch, persist, profiler);
		break;
	case SAM3_BACKBONE_TINYVIT:
		enc_out = sam3_tinyvit_build(&vl->enc.tvit, be, image,
					       scratch, persist, profiler);
		break;
	default:
		enc_out = NULL;
		break;
	}
	if (!enc_out) {
		sam3_log_error("vl_backbone: encoder build returned NULL");
		return NULL;
	}

	sam3_log_debug("vl_backbone: encoder done, scratch %zu/%zu, persist %zu/%zu",
		       scratch->offset, scratch->size,
		       persist->offset, persist->size);

	/*
	 * Reset scratch and build neck graph. The caller will
	 * evaluate this graph after we return.
	 */
	sam3_arena_reset(scratch);
	sam3_graph_init(g);

	enum sam3_error err;

	err = sam3_neck_build(&vl->neck, g, enc_out, out_features, scratch);
	if (err != SAM3_OK) {
		sam3_log_error("vl_backbone: neck_build failed, scratch %zu/%zu",
			       scratch->offset, scratch->size);
		return NULL;
	}

	sam3_log_debug("vl_backbone: neck built, scratch %zu/%zu, %d nodes",
		       scratch->offset, scratch->size, g->n_nodes);

	if (sam2_features && vl->has_sam2_neck) {
		err = sam3_neck_build(&vl->sam2_neck, g, enc_out,
				      sam2_features, scratch);
		if (err != SAM3_OK) {
			sam3_log_error("vl_backbone: sam2_neck_build failed");
			return NULL;
		}
		sam3_log_debug("vl_backbone: sam2_neck built, scratch %zu/%zu, %d nodes",
			       scratch->offset, scratch->size, g->n_nodes);
	}

	return enc_out;
}

struct sam3_tensor *sam3_vl_backbone_build_text(
	struct sam3_vl_backbone *vl,
	struct sam3_graph *g,
	const char *text,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena)
{
	int ctx = vl->text_iface.ctx_len;
	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN]; /* current value 32; clamps below */
	int n_tokens;
	struct sam3_tensor *tok_tensor;
	int dims[1];

	if (ctx > SAM3_TOKENIZER_CONTEXT_LEN)
		ctx = SAM3_TOKENIZER_CONTEXT_LEN;

	n_tokens = sam3_tokenizer_encode(&vl->tokenizer, text, tokens, ctx);
	if (n_tokens <= 0)
		return NULL;

	dims[0] = ctx;
	tok_tensor = gh_alloc_tensor(arena, SAM3_DTYPE_I32, 1, dims);
	if (!tok_tensor)
		return NULL;
	memcpy(tok_tensor->data, tokens, (size_t)ctx * sizeof(int32_t));

	return vl->text_iface.ops->build(&vl->text_iface, g, tok_tensor,
					 pooled_out, arena);
}

enum sam3_error sam3_vl_backbone_set_text_backbone(
	struct sam3_vl_backbone *vl, int text_backbone,
	struct sam3_arena *arena)
{
	if (!vl || !arena)
		return SAM3_EINVAL;
	return sam3_text_encoder_iface_init(&vl->text_iface,
					    text_backbone, arena);
}
