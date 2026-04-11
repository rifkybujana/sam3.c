/*
 * src/model/necks.c - Multi-scale feature pyramid neck implementation
 *
 * Implements Meta's SAM3 FPN neck architecture on the NHWC layout.
 * For each scale stage, the pipeline is: reshape [np, dim] directly
 * to NHWC [1, gs, gs, dim], optional NHWC MaxPool, spatial rescaling
 * via NHWC ConvTranspose2d (with GELU), 1x1 NHWC conv projection to
 * d_model, 3x3 NHWC conv spatial mixing. Conv weights already ship
 * in OHWI on disk (permuted by sam3_convert, Task 12), so the load
 * path consumes them directly with no runtime transpose.
 *
 * Key types:  sam3_neck
 * Depends on: necks.h, graph_helpers.h, util/log.h
 * Used by:    vl_combiner.c (vision pipeline), tests/test_necks.c,
 *             tests/test_neck_nhwc.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "necks.h"
#include "graph_helpers.h"
#include "util/log.h"

/*
 * init_stage_4x - Scale 4.0: two ConvTranspose2d + 1x1 proj + 3x3 mix.
 *
 * Pipeline:
 *   ConvT(dim->dim/2, k=2, s=2) -> GELU ->
 *   ConvT(dim/2->dim/4, k=2, s=2) ->
 *   Conv1x1(dim/4->d_model) ->
 *   Conv3x3(d_model->d_model, pad=1)
 */
static void init_stage_4x(struct sam3_neck *neck, int i)
{
	neck->stages[i].has_maxpool = 0;
	neck->stages[i].n_convs = 4;

	/* ConvTranspose2d(dim -> dim/2, k=2, s=2) */
	neck->stages[i].is_transpose[0] = 1;
	neck->stages[i].kernel_size[0] = 2;
	neck->stages[i].gelu_after[0] = 1;
	neck->stages[i].seq_idx[0] = 0;

	/* ConvTranspose2d(dim/2 -> dim/4, k=2, s=2) */
	neck->stages[i].is_transpose[1] = 1;
	neck->stages[i].kernel_size[1] = 2;
	neck->stages[i].gelu_after[1] = 0;
	neck->stages[i].seq_idx[1] = 2;

	/* Conv1x1(dim/4 -> d_model) */
	neck->stages[i].is_transpose[2] = 0;
	neck->stages[i].kernel_size[2] = 1;
	neck->stages[i].gelu_after[2] = 0;
	neck->stages[i].seq_idx[2] = 3;

	/* Conv3x3(d_model -> d_model, pad=1) */
	neck->stages[i].is_transpose[3] = 0;
	neck->stages[i].kernel_size[3] = 3;
	neck->stages[i].gelu_after[3] = 0;
	neck->stages[i].seq_idx[3] = 4;
}

/*
 * init_stage_2x - Scale 2.0: one ConvTranspose2d + 1x1 proj + 3x3 mix.
 *
 * Pipeline:
 *   ConvT(dim->dim/2, k=2, s=2) ->
 *   Conv1x1(dim/2->d_model) ->
 *   Conv3x3(d_model->d_model, pad=1)
 */
static void init_stage_2x(struct sam3_neck *neck, int i)
{
	neck->stages[i].has_maxpool = 0;
	neck->stages[i].n_convs = 3;

	/* ConvTranspose2d(dim -> dim/2, k=2, s=2) */
	neck->stages[i].is_transpose[0] = 1;
	neck->stages[i].kernel_size[0] = 2;
	neck->stages[i].gelu_after[0] = 0;
	neck->stages[i].seq_idx[0] = 0;

	/* Conv1x1(dim/2 -> d_model) */
	neck->stages[i].is_transpose[1] = 0;
	neck->stages[i].kernel_size[1] = 1;
	neck->stages[i].gelu_after[1] = 0;
	neck->stages[i].seq_idx[1] = 1;

	/* Conv3x3(d_model -> d_model, pad=1) */
	neck->stages[i].is_transpose[2] = 0;
	neck->stages[i].kernel_size[2] = 3;
	neck->stages[i].gelu_after[2] = 0;
	neck->stages[i].seq_idx[2] = 2;
}

/*
 * init_stage_1x - Scale 1.0: 1x1 proj + 3x3 mix (no rescaling).
 *
 * Pipeline:
 *   Conv1x1(dim->d_model) ->
 *   Conv3x3(d_model->d_model, pad=1)
 */
static void init_stage_1x(struct sam3_neck *neck, int i)
{
	neck->stages[i].has_maxpool = 0;
	neck->stages[i].n_convs = 2;

	/* Conv1x1(dim -> d_model) */
	neck->stages[i].is_transpose[0] = 0;
	neck->stages[i].kernel_size[0] = 1;
	neck->stages[i].gelu_after[0] = 0;
	neck->stages[i].seq_idx[0] = 0;

	/* Conv3x3(d_model -> d_model, pad=1) */
	neck->stages[i].is_transpose[1] = 0;
	neck->stages[i].kernel_size[1] = 3;
	neck->stages[i].gelu_after[1] = 0;
	neck->stages[i].seq_idx[1] = 1;
}

/*
 * init_stage_half - Scale 0.5: MaxPool + 1x1 proj + 3x3 mix.
 *
 * Pipeline:
 *   MaxPool(k=2, s=2) ->
 *   Conv1x1(dim->d_model) ->
 *   Conv3x3(d_model->d_model, pad=1)
 */
static void init_stage_half(struct sam3_neck *neck, int i)
{
	neck->stages[i].has_maxpool = 1;
	neck->stages[i].n_convs = 2;

	/* Conv1x1(dim -> d_model) */
	neck->stages[i].is_transpose[0] = 0;
	neck->stages[i].kernel_size[0] = 1;
	neck->stages[i].gelu_after[0] = 0;
	neck->stages[i].seq_idx[0] = 1;

	/* Conv3x3(d_model -> d_model, pad=1) */
	neck->stages[i].is_transpose[1] = 0;
	neck->stages[i].kernel_size[1] = 3;
	neck->stages[i].gelu_after[1] = 0;
	neck->stages[i].seq_idx[1] = 2;
}

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

	for (int i = 0; i < n_scales; i++) {
		neck->stages[i].scale_factor = scale_factors[i];

		float sf = scale_factors[i];
		if (sf >= 3.5f)
			init_stage_4x(neck, i);
		else if (sf >= 1.5f)
			init_stage_2x(neck, i);
		else if (sf >= 0.75f)
			init_stage_1x(neck, i);
		else
			init_stage_half(neck, i);
	}

	return SAM3_OK;
}

/*
 * compute_conv_dims - Compute weight shape for a conv layer.
 *
 * Walks through the stage pipeline to determine input/output channels
 * for the j-th conv layer, based on scale factor and backbone_dim.
 */
static void compute_conv_dims(struct sam3_neck *neck, int i, int j,
			      int *c_in, int *c_out, int *kh, int *kw)
{
	int dim = neck->backbone_dim;
	int d = neck->d_model;
	float sf = neck->stages[i].scale_factor;

	*kh = neck->stages[i].kernel_size[j];
	*kw = *kh;

	if (sf >= 3.5f) {
		/* 4x: ConvT(dim->dim/2) -> ConvT(dim/2->dim/4) ->
		 *     Conv1x1(dim/4->d) -> Conv3x3(d->d) */
		switch (j) {
		case 0: *c_in = dim;     *c_out = dim / 2; break;
		case 1: *c_in = dim / 2; *c_out = dim / 4; break;
		case 2: *c_in = dim / 4; *c_out = d;       break;
		case 3: *c_in = d;       *c_out = d;       break;
		}
	} else if (sf >= 1.5f) {
		/* 2x: ConvT(dim->dim/2) -> Conv1x1(dim/2->d) -> Conv3x3(d->d) */
		switch (j) {
		case 0: *c_in = dim;     *c_out = dim / 2; break;
		case 1: *c_in = dim / 2; *c_out = d;       break;
		case 2: *c_in = d;       *c_out = d;       break;
		}
	} else if (sf >= 0.75f) {
		/* 1x: Conv1x1(dim->d) -> Conv3x3(d->d) */
		switch (j) {
		case 0: *c_in = dim; *c_out = d; break;
		case 1: *c_in = d;   *c_out = d; break;
		}
	} else {
		/* 0.5x: MaxPool -> Conv1x1(dim->d) -> Conv3x3(d->d) */
		switch (j) {
		case 0: *c_in = dim; *c_out = d; break;
		case 1: *c_in = d;   *c_out = d; break;
		}
	}
}

/*
 * Weight name prefix for neck weights in the .sam3 file.
 * Original PyTorch: detector_model.vision_encoder.neck.fpn_layers.*
 */
#define NECK_P "detector_model.vision_encoder.neck.fpn_layers."

/*
 * neck_weight_name - Build correct weight name for a neck conv layer.
 *
 * Maps internal (stage, conv_idx) to PyTorch naming convention:
 *   - ConvTranspose2d layers → scale_layers.{seq_idx}
 *   - Last two Conv2d layers → proj1 and proj2
 */
static void neck_weight_name(char *buf, size_t buflen,
			     struct sam3_neck *neck,
			     int stage, int j, const char *suffix)
{
	int n = neck->stages[stage].n_convs;

	if (neck->stages[stage].is_transpose[j]) {
		/* Scale layers: ConvTranspose2d for upsampling */
		int seq = neck->stages[stage].seq_idx[j];
		snprintf(buf, buflen,
			 NECK_P "%d.scale_layers.%d.%s",
			 stage, seq, suffix);
	} else if (j == n - 2) {
		/* Second-to-last non-transpose conv → proj1 (1x1) */
		snprintf(buf, buflen,
			 NECK_P "%d.proj1.%s", stage, suffix);
	} else {
		/* Last conv → proj2 (3x3) */
		snprintf(buf, buflen,
			 NECK_P "%d.proj2.%s", stage, suffix);
	}
}

enum sam3_error sam3_neck_load(struct sam3_neck *neck,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena)
{
	char name[128];

	for (int i = 0; i < neck->n_scales; i++) {
		for (int j = 0; j < neck->stages[i].n_convs; j++) {
			int c_in, c_out, kh, kw;
			compute_conv_dims(neck, i, j,
					   &c_in, &c_out, &kh, &kw);

			/*
			 * Conv weights ship in OHWI [OC, KH, KW, IC]
			 * for both Conv2d and ConvTranspose2d after
			 * Task 12's permute in sam3_convert.
			 */
			int w_dims[4] = {c_out, kh, kw, c_in};

			neck_weight_name(name, sizeof(name),
					 neck, i, j, "weight");
			neck->stages[i].conv_w[j] = gh_load_mmap(
				wf, name, arena, SAM3_DTYPE_F32,
				4, w_dims);
			if (!neck->stages[i].conv_w[j])
				return SAM3_ENOMEM;

			int b_dims[] = {c_out};
			neck_weight_name(name, sizeof(name),
					 neck, i, j, "bias");
			neck->stages[i].conv_b[j] = gh_load_mmap(
				wf, name, arena, SAM3_DTYPE_F32,
				1, b_dims);
			if (!neck->stages[i].conv_b[j])
				return SAM3_ENOMEM;
		}
	}

	return SAM3_OK;
}

/*
 * build_stage - Build the compute graph for a single FPN stage.
 *
 * Pipeline (NHWC):
 * 1. Optional NHWC MaxPool(k=2, s=2) on [1, H, W, dim]
 * 2. For each conv: NHWC ConvTranspose2d or Conv2d (OHWI weight),
 *    optional GELU
 *
 * Returns the output feature map [1, H_out, W_out, d_model], or NULL.
 */
static struct sam3_tensor *build_stage(struct sam3_neck *neck,
					int stage,
					struct sam3_graph *g,
					struct sam3_tensor *nhwc_in,
					struct sam3_arena *arena)
{
	struct sam3_tensor *x = nhwc_in;

	/* Optional MaxPool before convs */
	if (neck->stages[stage].has_maxpool) {
		x = gh_maxpool2d(g, arena, x, 2, 2);
		if (!x)
			return NULL;
	}

	/* Apply conv layers */
	for (int j = 0; j < neck->stages[stage].n_convs; j++) {
		int k = neck->stages[stage].kernel_size[j];
		int padding = (k == 3) ? 1 : 0;

		if (neck->stages[stage].is_transpose[j]) {
			x = gh_conv_transpose2d(g, arena, x,
				neck->stages[stage].conv_w[j],
				neck->stages[stage].conv_b[j],
				2, 0);
		} else {
			x = gh_conv2d(g, arena, x,
				neck->stages[stage].conv_w[j],
				neck->stages[stage].conv_b[j],
				1, padding);
		}
		if (!x)
			return NULL;

		if (neck->stages[stage].gelu_after[j]) {
			x = gh_gelu(g, arena, x);
			if (!x)
				return NULL;
		}
	}

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

	/*
	 * The ViT output is already channels-last: [n_patches,
	 * backbone_dim] is exactly [1, gs*gs, dim] so a plain reshape
	 * to NHWC [1, gs, gs, dim] is zero-copy. This replaces the
	 * old gh_transpose + NCHW reshape pair.
	 */
	int nhwc_dims[] = {1, neck->grid_size, neck->grid_size,
			   neck->backbone_dim};
	struct sam3_tensor *nhwc = gh_reshape(g, arena, vit_features,
					      4, nhwc_dims);
	if (!nhwc)
		return SAM3_ENOMEM;

	for (int i = 0; i < neck->n_scales; i++) {
		out_features[i] = build_stage(neck, i, g, nhwc, arena);
		if (!out_features[i])
			return SAM3_ENOMEM;
	}

	return SAM3_OK;
}
