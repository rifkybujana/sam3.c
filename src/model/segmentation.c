/*
 * src/model/segmentation.c - Segmentation head graph construction
 *
 * Builds the compute graph for the pixel decoder and mask prediction.
 * The pixel decoder upsamples image features through three stages of
 * nearest-neighbor 2x upsampling, 1x1 convolution (implemented as
 * linear on reshaped features), layer normalization, and ReLU. The
 * final mask logits are computed as a dot product between query
 * embeddings and the upsampled pixel features, followed by sigmoid.
 *
 * Key types:  sam3_seg_head, sam3_pixel_decoder
 * Depends on: segmentation.h, graph_helpers.h
 * Used by:    sam3.c (top-level inference pipeline)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "segmentation.h"
#include "graph_helpers.h"

enum sam3_error sam3_seg_head_init(struct sam3_seg_head *head, int d_model)
{
	if (!head || d_model <= 0)
		return SAM3_EINVAL;

	memset(head, 0, sizeof(*head));
	head->d_model = d_model;
	head->pixel_dec.hidden_dim = d_model;

	return SAM3_OK;
}

enum sam3_error sam3_seg_head_load(struct sam3_seg_head *head,
				   const struct sam3_weight_file *wf,
				   struct sam3_arena *arena)
{
	int d = head->d_model;
	char name[128];

	int conv_w_dims[] = {d, d};
	int d_dims[] = {d};

	for (int i = 0; i < SAM3_SEG_UPSAMPLE_STAGES; i++) {
		/* Conv weight: [d_model, d_model] (1x1 conv as linear) */
		snprintf(name, sizeof(name),
			 "seg_head.pixel_dec.stages.%d.conv.weight", i);
		head->pixel_dec.stages[i].conv_w =
			gh_load_or_alloc(wf, name, arena,
				      SAM3_DTYPE_F32, 2, conv_w_dims);
		if (!head->pixel_dec.stages[i].conv_w)
			return SAM3_ENOMEM;

		/* Conv bias: [d_model] */
		snprintf(name, sizeof(name),
			 "seg_head.pixel_dec.stages.%d.conv.bias", i);
		head->pixel_dec.stages[i].conv_b =
			gh_load_or_alloc(wf, name, arena,
				      SAM3_DTYPE_F32, 1, d_dims);
		if (!head->pixel_dec.stages[i].conv_b)
			return SAM3_ENOMEM;

		/* Layer norm weight: [d_model] */
		snprintf(name, sizeof(name),
			 "seg_head.pixel_dec.stages.%d.ln.weight", i);
		head->pixel_dec.stages[i].ln_w =
			gh_load_or_alloc(wf, name, arena,
				      SAM3_DTYPE_F32, 1, d_dims);
		if (!head->pixel_dec.stages[i].ln_w)
			return SAM3_ENOMEM;

		/* Layer norm bias: [d_model] */
		snprintf(name, sizeof(name),
			 "seg_head.pixel_dec.stages.%d.ln.bias", i);
		head->pixel_dec.stages[i].ln_b =
			gh_load_or_alloc(wf, name, arena,
				      SAM3_DTYPE_F32, 1, d_dims);
		if (!head->pixel_dec.stages[i].ln_b)
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}

/*
 * build_upsample_stage - Build one pixel decoder upsample stage.
 *
 * Pipeline: upsample 2x -> reshape to [H*W, C] -> linear (1x1 conv) ->
 * layernorm -> ReLU -> reshape back to [1, C, H, W].
 *
 * @head:  Seg head with loaded weights
 * @stage: Stage index (0-2)
 * @g:     Graph to add nodes to
 * @x:     Input tensor [1, d_model, H, W]
 * @arena: Arena for intermediate tensors
 *
 * Returns output [1, d_model, H*2, W*2], or NULL on error.
 */
static struct sam3_tensor *build_upsample_stage(struct sam3_seg_head *head,
						int stage,
						struct sam3_graph *g,
						struct sam3_tensor *x,
						struct sam3_arena *arena)
{
	int d = head->d_model;

	/* Step 1: Nearest-neighbor 2x upsample [1, C, H, W] -> [1, C, 2H, 2W] */
	x = gh_upsample(g, arena, x, 2);
	if (!x)
		return NULL;

	/* Current spatial dims after upsample */
	int h = x->dims[2];
	int w = x->dims[3];
	int hw = h * w;

	/*
	 * Step 2: 1x1 conv as linear on reshaped features.
	 * Reshape [1, C, H, W] -> flatten channel-first to [C, H*W],
	 * transpose to [H*W, C], apply linear, transpose back,
	 * reshape to [1, C, H, W].
	 */
	int flat_dims[] = {d, hw};
	x = gh_reshape(g, arena, x, 2, flat_dims);
	if (!x)
		return NULL;

	/* [C, H*W] -> [H*W, C] */
	x = gh_transpose(g, arena, x);
	if (!x)
		return NULL;

	/* Linear: [H*W, C] @ W^T + bias -> [H*W, C] */
	x = gh_linear(g, arena, x,
		       head->pixel_dec.stages[stage].conv_w,
		       head->pixel_dec.stages[stage].conv_b);
	if (!x)
		return NULL;

	/* Step 3: Layer normalization over d_model dim */
	x = gh_layernorm(g, arena, x,
			  head->pixel_dec.stages[stage].ln_w,
			  head->pixel_dec.stages[stage].ln_b);
	if (!x)
		return NULL;

	/* Step 4: ReLU activation */
	x = gh_relu(g, arena, x);
	if (!x)
		return NULL;

	/*
	 * Reshape back to NCHW: [H*W, C] -> transpose to [C, H*W] ->
	 * reshape to [1, C, H, W].
	 */
	x = gh_transpose(g, arena, x);
	if (!x)
		return NULL;

	int nchw_dims[] = {1, d, h, w};
	x = gh_reshape(g, arena, x, 4, nchw_dims);
	if (!x)
		return NULL;

	return x;
}

struct sam3_tensor *sam3_seg_head_build(
	struct sam3_seg_head *head,
	struct sam3_graph *g,
	struct sam3_tensor *query_embed,
	struct sam3_tensor *pixel_features,
	int grid_h, int grid_w,
	struct sam3_arena *arena)
{
	int d = head->d_model;

	if (!head || !g || !query_embed || !pixel_features || !arena)
		return NULL;

	/*
	 * Step 1: Reshape pixel_features from [n_pixels, d_model] to
	 * [1, d_model, grid_h, grid_w] for spatial upsample ops.
	 *
	 * Data is [n_pixels, d_model] in row-major = each pixel is a
	 * d_model vector. We need NCHW, so transpose first:
	 * [n_pixels, d_model] -> [d_model, n_pixels] -> [1, d, H, W].
	 */
	struct sam3_tensor *x = pixel_features;

	x = gh_transpose(g, arena, x);
	if (!x)
		return NULL;
	/* x is [d_model, n_pixels] */

	int nchw_dims[] = {1, d, grid_h, grid_w};
	x = gh_reshape(g, arena, x, 4, nchw_dims);
	if (!x)
		return NULL;

	/*
	 * Step 2: Three upsample stages.
	 * Each stage: upsample 2x -> 1x1 conv -> LN -> ReLU.
	 * Spatial dims: (grid_h, grid_w) -> *2 -> *4 -> *8.
	 */
	for (int i = 0; i < SAM3_SEG_UPSAMPLE_STAGES; i++) {
		x = build_upsample_stage(head, i, g, x, arena);
		if (!x)
			return NULL;
	}

	/*
	 * After 3 stages of 2x upsampling:
	 * x is [1, d_model, grid_h*8, grid_w*8].
	 */
	int final_h = grid_h * 8;
	int final_w = grid_w * 8;
	int final_hw = final_h * final_w;

	/*
	 * Step 3: Reshape final pixel features for dot product.
	 * [1, d_model, H, W] -> [d_model, H*W] -> transpose -> [H*W, d_model].
	 */
	int flat_dims[] = {d, final_hw};
	x = gh_reshape(g, arena, x, 2, flat_dims);
	if (!x)
		return NULL;

	x = gh_transpose(g, arena, x);
	if (!x)
		return NULL;
	/* x is [H*W, d_model] */

	/*
	 * Step 4: Dot product to get mask logits.
	 * masks = query_embed @ pixel_features^T
	 * [n_queries, d_model] @ [d_model, H*W] -> [n_queries, H*W].
	 */
	struct sam3_tensor *pix_t = gh_transpose(g, arena, x);
	if (!pix_t)
		return NULL;
	/* pix_t is [d_model, H*W] */

	struct sam3_tensor *masks = gh_matmul(g, arena, query_embed, pix_t);
	if (!masks)
		return NULL;
	/* masks is [n_queries, H*W] */

	/* Step 5: Sigmoid -> final mask probabilities */
	masks = gh_sigmoid(g, arena, masks);
	if (!masks)
		return NULL;

	return masks;
}
