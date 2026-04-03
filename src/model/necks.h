/*
 * src/model/necks.h - Multi-scale feature pyramid neck
 *
 * Defines the feature pyramid network (FPN) neck that converts the
 * single-scale ViT backbone output [n_patches, embed_dim] into
 * multi-resolution feature maps. Each scale applies a 1x1 projection
 * from backbone_dim to d_model, layer normalization, and spatial
 * rescaling (upsample for scale > 1, conv2d stride-2 for scale < 1).
 *
 * Key types:  sam3_neck
 * Depends on: core/tensor.h, core/graph.h, core/alloc.h, core/weight.h
 * Used by:    sam3.c (image encoding pipeline), tests/test_necks.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_NECKS_H
#define SAM3_MODEL_NECKS_H

#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

#define SAM3_NECK_MAX_SCALES 4

struct sam3_neck {
	int d_model;		/* 256 -- output channel dim */
	int backbone_dim;	/* 1024 -- from ViT */
	int n_scales;		/* up to SAM3_NECK_MAX_SCALES */
	int grid_size;		/* 72 -- spatial size from ViT */

	/* Per-scale projection and normalization */
	struct {
		struct sam3_tensor *proj_w;	/* [d_model, backbone_dim] */
		struct sam3_tensor *proj_b;	/* [d_model] */
		struct sam3_tensor *ln_w;	/* [d_model] */
		struct sam3_tensor *ln_b;	/* [d_model] */
		float scale_factor;		/* e.g. 4.0, 2.0, 1.0, 0.5 */

		/* For scale < 1: stride-2 downsample conv */
		struct sam3_tensor *down_w;	/* [d_model, d_model, 1, 1] */
		struct sam3_tensor *down_b;	/* [d_model] */
	} stages[SAM3_NECK_MAX_SCALES];
};

/*
 * sam3_neck_init - Initialize neck with configuration.
 *
 * @neck:          Neck struct (caller-allocated, zeroed)
 * @d_model:       Output channel dimension (256)
 * @backbone_dim:  Input channel dimension from ViT (1024)
 * @grid_size:     Spatial grid size from ViT (72)
 * @n_scales:      Number of output scales (1-4)
 * @scale_factors: Array of n_scales scale factors (e.g. {4.0, 2.0, 1.0, 0.5})
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if n_scales is out of range.
 */
enum sam3_error sam3_neck_init(struct sam3_neck *neck,
			       int d_model, int backbone_dim,
			       int grid_size, int n_scales,
			       const float *scale_factors);

/*
 * sam3_neck_load - Load neck weights from weight file.
 *
 * @neck:  Initialized neck struct
 * @wf:    Open weight file (may be NULL for zero-init fallback)
 * @arena: Arena for weight tensor allocation
 *
 * Looks up weight tensors by name and populates the struct. When a
 * weight is not found (or wf is NULL), a zero-initialized tensor of
 * the correct shape is allocated as a fallback.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM if the arena is full.
 */
enum sam3_error sam3_neck_load(struct sam3_neck *neck,
			       const struct sam3_weight_file *wf,
			       struct sam3_arena *arena);

/*
 * sam3_neck_build - Build neck compute graph.
 *
 * @neck:         Initialized and loaded neck
 * @g:            Graph to add nodes to
 * @vit_features: ViT output [n_patches, backbone_dim]
 * @out_features: Output array of n_scales feature tensors
 *                (caller-allocated array of pointers)
 * @arena:        Arena for intermediate tensors
 *
 * Fills out_features[0..n_scales-1] with feature maps at different
 * scales. Each output is [1, d_model, H_i, W_i] in NCHW format.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_neck_build(struct sam3_neck *neck,
				struct sam3_graph *g,
				struct sam3_tensor *vit_features,
				struct sam3_tensor *out_features[],
				struct sam3_arena *arena);

#endif /* SAM3_MODEL_NECKS_H */
