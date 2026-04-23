/*
 * include/sam3/sam3_types.h - Public type definitions for sam3
 *
 * Defines all public-facing types: error codes, tensor descriptors,
 * configuration structs, and inference results. This header is included
 * by sam3.h and should not be included directly by users.
 *
 * Key types:  sam3_error, sam3_dtype, sam3_tensor_desc, sam3_mask_result
 * Depends on: <stdint.h>, <stddef.h>
 * Used by:    sam3.h, all internal modules
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_TYPES_H
#define SAM3_TYPES_H

#include <stdint.h>
#include <stddef.h>

#define SAM3_MAX_DIMS 4

/* Error codes returned by all sam3 public functions. */
enum sam3_error {
	SAM3_OK       =  0,
	SAM3_EINVAL   = -1,  /* Invalid argument */
	SAM3_ENOMEM   = -2,  /* Out of memory */
	SAM3_EIO      = -3,  /* I/O error (file read, Metal shader) */
	SAM3_EBACKEND = -4,  /* Backend initialization failed */
	SAM3_EMODEL   = -5,  /* Model format error */
	SAM3_EDTYPE   = -6,  /* Unsupported or mismatched dtype */
	SAM3_EVIDEO   = -7,  /* Video tracking error */
};

/* Log severity levels. */
enum sam3_log_level {
	SAM3_LOG_DEBUG,
	SAM3_LOG_INFO,
	SAM3_LOG_WARN,
	SAM3_LOG_ERROR,
};

/* Supported tensor data types. */
enum sam3_dtype {
	SAM3_DTYPE_F32,
	SAM3_DTYPE_F16,
	SAM3_DTYPE_BF16,
	SAM3_DTYPE_I32,
	SAM3_DTYPE_I8,
	SAM3_DTYPE_Q8_0,   /* Block-quantized int8: 32 values + f32 scale */
};

#define SAM3_DTYPE_COUNT 6

/* Prompt type for segmentation. */
enum sam3_prompt_type {
	SAM3_PROMPT_POINT,
	SAM3_PROMPT_BOX,
	SAM3_PROMPT_MASK,
	SAM3_PROMPT_TEXT,
};

/* A 2D point prompt (x, y) with label (foreground=1, background=0). */
struct sam3_point {
	float x;
	float y;
	int   label;
};

/* A bounding box prompt. */
struct sam3_box {
	float x1;
	float y1;
	float x2;
	float y2;
};

/* Segmentation prompt (union of point, box, mask, or text input). */
struct sam3_prompt {
	enum sam3_prompt_type type;
	union {
		struct sam3_point point;
		struct sam3_box   box;
		/* Mask input: pointer to H*W float array */
		struct {
			const float *data;
			int          width;
			int          height;
		} mask;
		const char *text;  /* Null-terminated UTF-8 string */
	};
};

/*
 * A single prompt set for batch segmentation: one logical query (a bundle
 * of points/boxes/text that together describe one object or one textual
 * concept). Batch segmentation runs N prompt sets against the same cached
 * image and returns N independent results. The prompts pointer must remain
 * valid for the duration of the sam3_segment_batch call only.
 */
struct sam3_prompt_set {
	const struct sam3_prompt *prompts;
	int                       n_prompts;
};

/* Result of a segmentation inference. */
struct sam3_result {
	float *masks;        /* Output masks: n_masks * H * W floats */
	float *iou_scores;   /* IoU score per mask */
	int    n_masks;
	int    mask_height;
	int    mask_width;
	int    iou_valid;    /* 1 if iou_scores are model-predicted, 0 if placeholder */
	float *boxes;        /* Bounding boxes: n_masks * 4 floats (x1,y1,x2,y2 xyxy) */
	int    boxes_valid;  /* 1 if boxes are computed, 0 if not */
	int    best_mask;    /* Stability-selected best mask index, -1 if N/A */
};

/* Vision backbone type. */
enum sam3_backbone_type {
	SAM3_BACKBONE_HIERA        = 0,
	SAM3_BACKBONE_EFFICIENTVIT = 1,
	SAM3_BACKBONE_TINYVIT      = 2,
};

/* Model variant. SAM3_1 uses the tri-neck (3 FPN scales) and the
 * multiplex-capable tracker; SAM3 uses the dual-neck and the original
 * tracker. Stored in the .sam3 header's reserved[1] slot. */
enum sam3_variant {
	SAM3_VARIANT_SAM3   = 0,
	SAM3_VARIANT_SAM3_1 = 1,
};

/* Text encoder backbone variant. Stored in the .sam3 v4 header.
 * v3 files (no field) load as SAM3_TEXT_CLIP for backward compat. */
enum sam3_text_backbone {
	SAM3_TEXT_CLIP            = 0,
	SAM3_TEXT_MOBILECLIP_S0   = 1,
	SAM3_TEXT_MOBILECLIP_S1   = 2,
	SAM3_TEXT_MOBILECLIP_L    = 3,
};

/* Model configuration loaded from weights file. */
struct sam3_model_config {
	int image_size;       /* Input image size (e.g., 1024) */
	int encoder_dim;      /* Image encoder embedding dimension */
	int decoder_dim;      /* Mask decoder dimension */
	int n_encoder_layers;
	int n_decoder_layers;
	int backbone_type;    /* enum sam3_backbone_type */
	int n_fpn_scales;     /* 3 (SAM 3.1) or 4 (SAM 3) */
	int variant;          /* enum sam3_variant */
	int text_backbone;    /* enum sam3_text_backbone */
};

/* Propagation direction for video tracking. */
enum sam3_propagate_dir {
	SAM3_PROPAGATE_BOTH     = 0,
	SAM3_PROPAGATE_FORWARD  = 1,
	SAM3_PROPAGATE_BACKWARD = 2,
};

#define SAM3_MAX_OBJECTS         16   /* per-session object cap; matches Python multiplex bucket */
#define SAM3_MAX_MEMORY_FRAMES   16
#define SAM3_MAX_POINTS_PER_OBJ  16

/* Opaque context handle. */
typedef struct sam3_ctx sam3_ctx;

/* Opaque video session handle. */
typedef struct sam3_video_session sam3_video_session;

#endif /* SAM3_TYPES_H */
