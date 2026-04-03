/*
 * src/model/necks.c - Multi-scale feature pyramid neck implementation
 *
 * Implements the feature pyramid network (FPN) neck that transforms
 * single-scale ViT output into multi-resolution feature maps. For each
 * scale stage, the pipeline is: 1x1 linear projection from backbone_dim
 * to d_model, layer normalization, and spatial rescaling. Upsampling
 * uses nearest-neighbor interpolation; downsampling uses a 1x1 conv2d
 * with stride 2.
 *
 * Key types:  sam3_neck
 * Depends on: necks.h, graph_helpers.h
 * Used by:    sam3.c (top-level image encoding pipeline)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "necks.h"
#include "graph_helpers.h"

enum sam3_error sam3_neck_init(struct sam3_neck *neck,
			       int d_model, int backbone_dim,
			       int grid_size, int n_scales,
			       const float *scale_factors)
{
	if (n_scales < 1 || n_scales > SAM3_NECK_MAX_SCALES)
		return SAM3_EINVAL;

	memset(neck, 0, sizeof(*neck));

	neck->d_model = d_model;
	neck->backbone_dim = backbone_dim;
	neck->grid_size = grid_size;
	neck->n_scales = n_scales;

	for (int i = 0; i < n_scales; i++)
		neck->stages[i].scale_factor = scale_factors[i];

	return SAM3_OK;
}

enum sam3_error sam3_neck_load(struct sam3_neck *neck,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	int d = neck->d_model;
	int b = neck->backbone_dim;
	char name[128];

	int proj_w_dims[] = {d, b};
	int d_dims[] = {d};

	for (int i = 0; i < neck->n_scales; i++) {
		/* Projection weight: [d_model, backbone_dim] */
		snprintf(name, sizeof(name),
			 "neck.stage.%d.proj.weight", i);
		neck->stages[i].proj_w = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       2, proj_w_dims);
		if (!neck->stages[i].proj_w)
			return SAM3_ENOMEM;

		/* Projection bias: [d_model] */
		snprintf(name, sizeof(name),
			 "neck.stage.%d.proj.bias", i);
		neck->stages[i].proj_b = gh_load_or_alloc(wf, name, arena,
						       SAM3_DTYPE_F32,
						       1, d_dims);
		if (!neck->stages[i].proj_b)
			return SAM3_ENOMEM;

		/* Layer norm weight: [d_model] */
		snprintf(name, sizeof(name),
			 "neck.stage.%d.ln.weight", i);
		neck->stages[i].ln_w = gh_load_or_alloc(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, d_dims);
		if (!neck->stages[i].ln_w)
			return SAM3_ENOMEM;

		/* Layer norm bias: [d_model] */
		snprintf(name, sizeof(name),
			 "neck.stage.%d.ln.bias", i);
		neck->stages[i].ln_b = gh_load_or_alloc(wf, name, arena,
						     SAM3_DTYPE_F32,
						     1, d_dims);
		if (!neck->stages[i].ln_b)
			return SAM3_ENOMEM;

		/* Downsample conv for scale < 1.0 */
		if (neck->stages[i].scale_factor < 1.0f) {
			int down_w_dims[] = {d, d, 1, 1};
			snprintf(name, sizeof(name),
				 "neck.stage.%d.down.weight", i);
			neck->stages[i].down_w = gh_load_or_alloc(
				wf, name, arena, SAM3_DTYPE_F32,
				4, down_w_dims);
			if (!neck->stages[i].down_w)
				return SAM3_ENOMEM;

			snprintf(name, sizeof(name),
				 "neck.stage.%d.down.bias", i);
			neck->stages[i].down_b = gh_load_or_alloc(
				wf, name, arena, SAM3_DTYPE_F32,
				1, d_dims);
			if (!neck->stages[i].down_b)
				return SAM3_ENOMEM;
		}
	}

	return SAM3_OK;
}

/*
 * build_stage - Build the compute graph for a single neck stage.
 *
 * Pipeline: linear projection -> layernorm -> spatial rescale.
 *
 * @neck:    Initialized and loaded neck
 * @stage:   Stage index
 * @g:       Graph
 * @flat_in: ViT features as [n_patches, backbone_dim]
 * @arena:   Arena for intermediate tensors
 *
 * Returns the output feature map [1, d_model, H_out, W_out], or NULL.
 */
static struct sam3_tensor *build_stage(struct sam3_neck *neck,
					int stage,
					struct sam3_graph *g,
					struct sam3_tensor *flat_in,
					struct sam3_arena *arena)
{
	int d = neck->d_model;
	int gs = neck->grid_size;
	float sf = neck->stages[stage].scale_factor;

	/*
	 * Step 1: Project from backbone_dim to d_model.
	 * flat_in is [n_patches, backbone_dim].
	 * linear(flat_in, proj_w, proj_b) -> [n_patches, d_model].
	 */
	struct sam3_tensor *x;
	x = gh_linear(g, arena, flat_in,
		       neck->stages[stage].proj_w,
		       neck->stages[stage].proj_b);
	if (!x)
		return NULL;
	/* x is [n_patches, d_model] */

	/*
	 * Step 2: Layer normalization (per-patch, over d_model dim).
	 */
	x = gh_layernorm(g, arena, x,
			  neck->stages[stage].ln_w,
			  neck->stages[stage].ln_b);
	if (!x)
		return NULL;

	/*
	 * Step 3: Reshape to spatial NCHW for rescaling.
	 * [n_patches, d_model] -> [gs, gs, d_model] -> need NCHW:
	 * [1, d_model, gs, gs].
	 *
	 * Since the data is row-major and patches are in raster order
	 * (row by row), the layout is [gs*gs, d_model]. We reshape to
	 * [gs, gs, d_model], then rearrange to [d_model, gs, gs].
	 *
	 * With our 2D-only transpose, the simplest approach:
	 * 1. Reshape [n_patches, d_model] -> [n_patches, d_model]
	 *    (already this shape)
	 * 2. Transpose -> [d_model, n_patches]
	 * 3. Reshape -> [1, d_model, gs, gs]
	 */
	x = gh_transpose(g, arena, x);
	if (!x)
		return NULL;
	/* x is [d_model, n_patches] */

	int nchw_dims[] = {1, d, gs, gs};
	x = gh_reshape(g, arena, x, 4, nchw_dims);
	if (!x)
		return NULL;

	/*
	 * Step 4: Spatial rescaling.
	 */
	if (sf > 1.0f) {
		/* Upsample by integer factor */
		int scale = (int)sf;
		x = gh_upsample(g, arena, x, scale);
		if (!x)
			return NULL;
	} else if (sf < 1.0f) {
		/*
		 * Downsample via 1x1 conv2d with stride 2.
		 * Input [1, d_model, gs, gs] ->
		 * Output [1, d_model, gs/2, gs/2].
		 */
		int half_gs = gs / 2;
		int out_dims[] = {1, d, half_gs, half_gs};
		struct sam3_tensor *conv_out;
		conv_out = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
					   4, out_dims);
		if (!conv_out)
			return NULL;

		struct sam3_tensor *inputs[] = {
			x, neck->stages[stage].down_w
		};
		conv_out = sam3_graph_add_op(g, SAM3_OP_CONV2D,
					      inputs, 2, conv_out);
		if (!conv_out)
			return NULL;

		struct sam3_node *node = &g->nodes[g->n_nodes - 1];
		node->params[0] = 2; /* stride */
		node->params[1] = 0; /* padding */

		/*
		 * Add downsample bias.
		 * conv_out is [1, d_model, half_gs, half_gs].
		 * Reshape to [d_model, half_gs*half_gs], add bias
		 * [d_model] via broadcast, reshape back.
		 */
		int n_spatial = half_gs * half_gs;
		int flat2d[] = {d, n_spatial};
		struct sam3_tensor *flat;
		flat = gh_reshape(g, arena, conv_out, 2, flat2d);
		if (!flat)
			return NULL;

		/* Transpose to [n_spatial, d_model] for bias add */
		flat = gh_transpose(g, arena, flat);
		if (!flat)
			return NULL;

		flat = gh_add(g, arena, flat,
			       neck->stages[stage].down_b);
		if (!flat)
			return NULL;

		/* Transpose back and reshape to NCHW */
		flat = gh_transpose(g, arena, flat);
		if (!flat)
			return NULL;

		x = gh_reshape(g, arena, flat, 4, out_dims);
		if (!x)
			return NULL;
	}
	/* scale == 1.0: no spatial change, x stays as-is */

	return x;
}

enum sam3_error sam3_neck_build(struct sam3_neck *neck,
				struct sam3_graph *g,
				struct sam3_tensor *vit_features,
				struct sam3_tensor *out_features[],
				struct sam3_arena *arena)
{
	if (!neck || !g || !vit_features || !out_features || !arena)
		return SAM3_EINVAL;

	for (int i = 0; i < neck->n_scales; i++) {
		out_features[i] = build_stage(neck, i, g,
					       vit_features, arena);
		if (!out_features[i])
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}
