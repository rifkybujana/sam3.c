/*
 * src/model/tracker.h - Core tracker module for SAM3 video segmentation
 *
 * Defines the sam3_tracker struct which wires together the mask decoder,
 * memory encoder, memory attention, and memory bank into a single
 * per-frame tracking pipeline. Corresponds to Sam3TrackerBase in the
 * Python reference. The tracker holds all sub-modules and learned
 * parameters, and provides init/load/track_frame/reset lifecycle methods.
 *
 * Key types:  sam3_tracker
 * Depends on: mask_decoder.h, memory_encoder.h, memory_attn.h,
 *             memory_bank.h, core/tensor.h, core/graph.h, core/alloc.h,
 *             core/weight.h
 * Used by:    model/video_session.h, model/sam3_video.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TRACKER_H
#define SAM3_MODEL_TRACKER_H

#include "mask_decoder.h"
#include "memory_encoder.h"
#include "memory_attn.h"
#include "memory_bank.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"

struct sam3_tracker {
	/* Sub-modules */
	struct sam3_mask_decoder  sam_decoder;
	struct sam3_memory_encoder mem_encoder;
	struct sam3_memory_attn   mem_attention;
	/*
	 * NOTE: mem_bank was removed in Task 2.2. Per-object memory banks
	 * live in sam3_video_object::bank inside the session. The tracker
	 * no longer owns any bank; callers pass a bank pointer to
	 * sam3_tracker_track_frame explicitly.
	 */

	/* Learned parameters */
	struct sam3_tensor *maskmem_tpos_enc;    /* [7, 1, 1, 64] temporal pos enc */
	struct sam3_tensor *no_mem_embed;        /* [1, 1, 256] no-memory placeholder */
	struct sam3_tensor *no_mem_pos_enc;      /* [1, 1, 256] pos for no-memory */
	struct sam3_tensor *no_obj_ptr;          /* [1, 256] obj ptr when not visible */
	struct sam3_tensor *no_obj_embed_spatial;/* [1, 64] spatial for occlusion */

	/*
	 * Object pointer projection: 3-layer MLP from best-IoU mask
	 * token to obj_ptr. Python: MLP(hidden_dim, hidden_dim,
	 * hidden_dim, 3). Layers: 3x Linear(256, 256) with ReLU between.
	 * Weight names: obj_ptr_proj.layers.{0,1,2}.{weight,bias}.
	 */
	struct sam3_tensor *obj_ptr_proj_fc0_w;  /* [256, 256] */
	struct sam3_tensor *obj_ptr_proj_fc0_b;  /* [256] */
	struct sam3_tensor *obj_ptr_proj_fc1_w;  /* [256, 256] */
	struct sam3_tensor *obj_ptr_proj_fc1_b;  /* [256] */
	struct sam3_tensor *obj_ptr_proj_fc2_w;  /* [256, 256] */
	struct sam3_tensor *obj_ptr_proj_fc2_b;  /* [256] */

	/*
	 * Temporal position projection for object pointers: Linear
	 * (hidden_dim, mem_dim) = Linear(256, 64). Projects 1D sine
	 * position encoding of frame distances into memory-dim space
	 * before concatenation into the memory-attention prompt.
	 */
	struct sam3_tensor *obj_ptr_tpos_proj_w; /* [64, 256] */
	struct sam3_tensor *obj_ptr_tpos_proj_b; /* [64] */

	/* Mask downsampler: Conv2d(1, 64, kernel_size=4, stride=4) for GT mask prompts */
	struct sam3_tensor *mask_downsample_w;   /* [64, 1, 4, 4] OHWI */
	struct sam3_tensor *mask_downsample_b;   /* [64] */

	/*
	 * SAM prompt encoder weights (sam_prompt_encoder.*). Used by the
	 * video tracker's prompt-token projection instead of the SAM3
	 * detector's geometry encoder. Python reference:
	 *   sparse_embed = pe_layer(point_coords_norm) + point_embeddings[label]
	 * with a "not_a_point" padding row when no box is provided.
	 *
	 * pe_gaussian:       [2, 128] Gaussian positional encoding matrix
	 * not_a_point:       [1, 256] embedding for padding rows
	 * pt_emb_neg:        [1, 256] added to negative-label points (label=0)
	 * pt_emb_pos:        [1, 256] added to positive-label points (label=1)
	 * pt_emb_box_tl:     [1, 256] added to box top-left corner (label=2)
	 * pt_emb_box_br:     [1, 256] added to box bottom-right corner (label=3)
	 */
	struct sam3_tensor *sam_pe_gaussian;     /* [2, 128] */
	struct sam3_tensor *sam_not_a_point;     /* [1, 256] */
	struct sam3_tensor *sam_pt_emb_neg;      /* [1, 256] */
	struct sam3_tensor *sam_pt_emb_pos;      /* [1, 256] */
	struct sam3_tensor *sam_pt_emb_box_tl;   /* [1, 256] */
	struct sam3_tensor *sam_pt_emb_box_br;   /* [1, 256] */

	/* Config matching Sam3TrackerBase */
	int   num_maskmem;               /* 7: total memory frames */
	int   max_cond_frames_in_attn;   /* 4: cap on cond frames admitted
					  * into memory attention per frame */
	int   image_size;                /* 1008: input resolution */
	int   backbone_stride;     /* 14: backbone feature stride */
	int   max_obj_ptrs;        /* 16: max object pointers in cross-attention */
	float sigmoid_scale;       /* 20.0: memory mask preprocessing */
	float sigmoid_bias;        /* -10.0: memory mask preprocessing */
	float mf_threshold;        /* 0.01: SAM3-Long memory selection threshold */

	int   multimask_output;    /* 1: use multimask */
	int   multimask_min_pt_num;/* 0 */
	int   multimask_max_pt_num;/* 1 */
};

/*
 * Optional Python parameters NOT modeled here:
 *   - cond_frame_spatial_embedding
 *   - cond_frame_obj_ptr_embedding
 *
 * sam3_tracker_base.py reads both via getattr(self, name, None) and
 * skips the addition when the attribute is missing. Upstream SAM3
 * checkpoints (sam3.safetensors, efficient_sam3_*.safetensors) ship
 * without these tensors, so the Python code paths are inert for the
 * released weights. If a future checkpoint adds them we would also
 * need to load the tensors and bias the cond-frame pos enc / obj_ptr
 * here; for now omitting them matches what Python actually runs.
 */

/*
 * sam3_tracker_init - Initialize tracker with default config values.
 *
 * @trk: Tracker struct (caller-allocated)
 *
 * Sets all config fields to SAM3 defaults and initializes sub-modules.
 * Returns SAM3_OK on success, SAM3_EINVAL if trk is NULL.
 */
enum sam3_error sam3_tracker_init(struct sam3_tracker *trk);

/*
 * sam3_tracker_load - Load all tracker weights.
 *
 * @trk:   Initialized tracker
 * @wf:    Weight file (NULL for zero-init testing)
 * @arena: Arena for weight tensor allocation
 *
 * Delegates to sub-module loaders and loads direct tensor weights.
 * Weight prefix: tracker_model.*
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad args,
 * SAM3_ENOMEM if arena is full.
 */
enum sam3_error sam3_tracker_load(struct sam3_tracker *trk,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena);

/*
 * sam3_tracker_track_frame - Process one frame through the tracking pipeline.
 *
 * @trk:           Loaded tracker
 * @g:             Graph to build into
 * @bank:          Per-object memory bank to read from and write to. Must not
 *                 be NULL. Owned by the caller (sam3_video_object::bank).
 * @backbone_feat: Image encoder output [feat_h*feat_w, d_model]
 * @feat_h:        Spatial height of backbone features (e.g. 72)
 * @feat_w:        Spatial width of backbone features (e.g. 72)
 * @prompt:        Sparse prompt tokens [n_prompt, 256] or NULL for propagation
 * @feat_s0:       High-res backbone feature [1, 4H, 4W, 256] or NULL
 * @feat_s1:       Mid-res backbone feature [1, 2H, 2W, 256] or NULL
 * @frame_idx:     Current frame index (for memory temporal encoding)
 * @is_cond:       True if this is a conditioning frame (user-provided prompt)
 * @arena:         Scratch arena for intermediates
 * @out_masks:     Output: mask logits [4, H, W]
 * @out_iou:       Output: IoU scores [4]
 * @out_obj_ptr:   Output: object pointer [1, 256] or NULL
 * @out_score:     Output: object existence score (scalar) or NULL
 *
 * Runs memory attention (with placeholder if bank is empty), then the
 * mask decoder, and sets output pointers.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL for bad args,
 * SAM3_ENOMEM if graph or arena allocation fails.
 */
enum sam3_error sam3_tracker_track_frame(
	struct sam3_tracker *trk,
	struct sam3_graph *g,
	struct sam3_memory_bank *bank,
	struct sam3_tensor *backbone_feat,
	int feat_h, int feat_w,
	struct sam3_tensor *prompt,
	struct sam3_tensor *feat_s0,
	struct sam3_tensor *feat_s1,
	int frame_idx, int is_cond,
	struct sam3_arena *arena,
	struct sam3_tensor **out_masks,
	struct sam3_tensor **out_iou,
	struct sam3_tensor **out_obj_ptr,
	struct sam3_tensor **out_score);

/*
 * sam3_tracker_sam_project_prompts - Encode point prompts into the
 * SAM prompt-encoder token format used by the mask-decoder two-way
 * transformer.
 *
 * Produces [n_tokens, d_model=256] tokens:
 *   - One token per point via
 *       pe = pe_layer(coords_normalized)       # 2*pi*coords @ gaussian
 *       tok = [cos(pe), sin(pe)] + point_embeddings[label]
 *   - One padding "not_a_point" token appended when no box prompt is
 *     present (pad=true) — Python adds this for SAM's two-way
 *     transformer to always receive >=2 prompt tokens.
 *
 * Returns NULL on allocation failure or if the sam_pe_gaussian
 * weight is missing (older checkpoints).
 */
struct sam3_prompt;
struct sam3_tensor *sam3_tracker_sam_project_prompts(
	const struct sam3_tracker *trk,
	const struct sam3_prompt *prompts,
	int n_prompts,
	int prompt_w, int prompt_h,
	struct sam3_arena *arena);

/*
 * sam3_tracker_reset - Clear memory bank for a new tracking sequence.
 *
 * @trk: Tracker to reset
 *
 * Returns SAM3_OK on success, SAM3_EINVAL if trk is NULL.
 */
enum sam3_error sam3_tracker_reset(struct sam3_tracker *trk);

#endif /* SAM3_MODEL_TRACKER_H */
