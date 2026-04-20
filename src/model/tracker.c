/*
 * src/model/tracker.c - Core tracker module implementation
 *
 * Implements the SAM3 video tracker lifecycle: init, load, track_frame,
 * and reset. Wires together the mask decoder, memory encoder, memory
 * attention, and memory bank sub-modules. The track_frame method runs
 * memory attention against the memory bank when populated (collecting
 * spatial features, temporal position encodings, and object pointers),
 * falling back to the no_mem_embed path when the bank is empty, then
 * invokes the mask decoder to produce mask logits and IoU scores. The
 * object-score token emitted by the decoder is projected through
 * obj_ptr_proj to yield the per-frame object pointer, and the IoU
 * tensor is exposed as the object existence score.
 *
 * Weight prefix: tracker_model.*
 *
 * Key types:  sam3_tracker
 * Depends on: tracker.h, graph_helpers.h, util/log.h
 * Used by:    model/session.c (future)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <string.h>

#include "tracker.h"
#include "graph_helpers.h"
#include "util/log.h"

#define WP "tracker_model."

/* --- Initialization ─ --- */

enum sam3_error sam3_tracker_init(struct sam3_tracker *trk)
{
	if (!trk) {
		sam3_log_error("tracker init: NULL tracker");
		return SAM3_EINVAL;
	}

	memset(trk, 0, sizeof(*trk));

	/* Config defaults matching Sam3TrackerBase */
	trk->num_maskmem            = 7;
	trk->max_cond_frames_in_attn = 4;
	trk->image_size             = 1008;
	trk->backbone_stride        = 14;
	trk->max_obj_ptrs           = 16;
	trk->sigmoid_scale          = 20.0f;
	trk->sigmoid_bias           = -10.0f;
	trk->mf_threshold           = 0.01f;
	trk->multimask_output       = 1;
	trk->multimask_min_pt_num   = 0;
	trk->multimask_max_pt_num   = 1;

	/* Initialize sub-modules */
	sam3_mask_decoder_init(&trk->sam_decoder);
	sam3_memory_encoder_init(&trk->mem_encoder, 256, 64);
	sam3_memory_attn_init(&trk->mem_attention, 256, 64, 4, 1, 72, 72);
	/*
	 * Per-object memory banks are owned by sam3_video_object::bank
	 * (Task 2.2). The tracker no longer owns a global bank — banks are
	 * initialised in sam3_session_get_or_add_obj and passed explicitly
	 * to sam3_tracker_track_frame.
	 */

	return SAM3_OK;
}

/* --- Weight loading ─ --- */

enum sam3_error sam3_tracker_load(struct sam3_tracker *trk,
				  const struct sam3_weight_file *wf,
				  struct sam3_arena *arena)
{
	enum sam3_error err;

	if (!trk || !arena) {
		sam3_log_error("tracker load: NULL argument");
		return SAM3_EINVAL;
	}

	/* Delegate to sub-module loaders */
	err = sam3_mask_decoder_load(&trk->sam_decoder, wf, arena);
	if (err != SAM3_OK) {
		sam3_log_error("tracker load: mask decoder failed (%d)", err);
		return err;
	}

	err = sam3_memory_encoder_load(&trk->mem_encoder, wf, arena);
	if (err != SAM3_OK) {
		sam3_log_error("tracker load: memory encoder failed (%d)", err);
		return err;
	}

	err = sam3_memory_attn_load(&trk->mem_attention, wf, arena);
	if (err != SAM3_OK) {
		sam3_log_error("tracker load: memory attention failed (%d)",
			       err);
		return err;
	}

	/* Load direct tensor weights */

	/* Temporal position encoding: [7, 1, 1, 64] */
	int tpos_dims[] = {7, 1, 1, 64};
	trk->maskmem_tpos_enc = gh_load_mmap(
		wf, WP "maskmem_tpos_enc", arena,
		SAM3_DTYPE_F32, 4, tpos_dims);
	if (!trk->maskmem_tpos_enc) {
		sam3_log_error("tracker load: maskmem_tpos_enc alloc failed");
		return SAM3_ENOMEM;
	}

	/* No-memory placeholder embed: [1, 1, 256] */
	int nomem_dims[] = {1, 1, 256};
	trk->no_mem_embed = gh_load_mmap(
		wf, WP "no_mem_embed", arena,
		SAM3_DTYPE_F32, 3, nomem_dims);
	if (!trk->no_mem_embed) {
		sam3_log_error("tracker load: no_mem_embed alloc failed");
		return SAM3_ENOMEM;
	}

	/* No-memory position encoding: [1, 1, 256] */
	trk->no_mem_pos_enc = gh_load_mmap(
		wf, WP "no_mem_pos_enc", arena,
		SAM3_DTYPE_F32, 3, nomem_dims);
	if (!trk->no_mem_pos_enc) {
		sam3_log_error("tracker load: no_mem_pos_enc alloc failed");
		return SAM3_ENOMEM;
	}

	/* No-object pointer: [1, 256] */
	int noobj_dims[] = {1, 256};
	trk->no_obj_ptr = gh_load_mmap(
		wf, WP "no_obj_ptr", arena,
		SAM3_DTYPE_F32, 2, noobj_dims);
	if (!trk->no_obj_ptr) {
		sam3_log_error("tracker load: no_obj_ptr alloc failed");
		return SAM3_ENOMEM;
	}

	/* No-object spatial embedding: [1, 64] */
	int noobj_spatial_dims[] = {1, 64};
	trk->no_obj_embed_spatial = gh_load_mmap(
		wf, WP "no_obj_embed_spatial", arena,
		SAM3_DTYPE_F32, 2, noobj_spatial_dims);
	if (!trk->no_obj_embed_spatial) {
		sam3_log_error("tracker load: no_obj_embed_spatial alloc failed");
		return SAM3_ENOMEM;
	}

	/*
	 * Object pointer projection: 3-layer MLP
	 * (weight names: obj_ptr_proj.layers.{0,1,2}.{weight,bias}).
	 * Each layer: [256, 256]/[256].
	 */
	{
		int w_dims[] = {256, 256};
		int b_dims[] = {256};

		trk->obj_ptr_proj_fc0_w = gh_load_mmap(
			wf, WP "obj_ptr_proj.layers.0.weight", arena,
			SAM3_DTYPE_F32, 2, w_dims);
		if (!trk->obj_ptr_proj_fc0_w) return SAM3_ENOMEM;
		trk->obj_ptr_proj_fc0_b = gh_load_mmap(
			wf, WP "obj_ptr_proj.layers.0.bias", arena,
			SAM3_DTYPE_F32, 1, b_dims);
		if (!trk->obj_ptr_proj_fc0_b) return SAM3_ENOMEM;

		trk->obj_ptr_proj_fc1_w = gh_load_mmap(
			wf, WP "obj_ptr_proj.layers.1.weight", arena,
			SAM3_DTYPE_F32, 2, w_dims);
		if (!trk->obj_ptr_proj_fc1_w) return SAM3_ENOMEM;
		trk->obj_ptr_proj_fc1_b = gh_load_mmap(
			wf, WP "obj_ptr_proj.layers.1.bias", arena,
			SAM3_DTYPE_F32, 1, b_dims);
		if (!trk->obj_ptr_proj_fc1_b) return SAM3_ENOMEM;

		trk->obj_ptr_proj_fc2_w = gh_load_mmap(
			wf, WP "obj_ptr_proj.layers.2.weight", arena,
			SAM3_DTYPE_F32, 2, w_dims);
		if (!trk->obj_ptr_proj_fc2_w) return SAM3_ENOMEM;
		trk->obj_ptr_proj_fc2_b = gh_load_mmap(
			wf, WP "obj_ptr_proj.layers.2.bias", arena,
			SAM3_DTYPE_F32, 1, b_dims);
		if (!trk->obj_ptr_proj_fc2_b) return SAM3_ENOMEM;
	}

	/*
	 * Temporal position projection for object pointers:
	 * Linear(hidden_dim=256, mem_dim=64).
	 */
	{
		int tpos_w_dims[] = {64, 256};
		int tpos_b_dims[] = {64};

		trk->obj_ptr_tpos_proj_w = gh_load_mmap(
			wf, WP "obj_ptr_tpos_proj.weight", arena,
			SAM3_DTYPE_F32, 2, tpos_w_dims);
		if (!trk->obj_ptr_tpos_proj_w) return SAM3_ENOMEM;
		trk->obj_ptr_tpos_proj_b = gh_load_mmap(
			wf, WP "obj_ptr_tpos_proj.bias", arena,
			SAM3_DTYPE_F32, 1, tpos_b_dims);
		if (!trk->obj_ptr_tpos_proj_b) return SAM3_ENOMEM;
	}

	/* Mask downsampler Conv2d: weight [64, 1, 4, 4], bias [64] */
	int mds_w_dims[] = {64, 1, 4, 4};
	trk->mask_downsample_w = gh_load_mmap(
		wf, WP "mask_downsample.weight", arena,
		SAM3_DTYPE_F32, 4, mds_w_dims);
	if (!trk->mask_downsample_w) {
		sam3_log_error("tracker load: mask_downsample.weight alloc failed");
		return SAM3_ENOMEM;
	}

	int mds_b_dims[] = {64};
	trk->mask_downsample_b = gh_load_mmap(
		wf, WP "mask_downsample.bias", arena,
		SAM3_DTYPE_F32, 1, mds_b_dims);
	if (!trk->mask_downsample_b) {
		sam3_log_error("tracker load: mask_downsample.bias alloc failed");
		return SAM3_ENOMEM;
	}

	/*
	 * SAM prompt encoder weights. Used by video-path prompt token
	 * projection. Names: tracker_model.sam_prompt_encoder.*
	 * (or, if rename is not applied, tracker.sam_prompt_encoder.*).
	 */
	{
		int pe_dims[] = {2, 128};
		int emb_dims[] = {1, 256};
		/* The weight_rename pipeline maps the PyTorch
		 * pe_layer.positional_encoding_gaussian_matrix to
		 * shared_embedding.positional_embedding in the .sam3 file. */
		trk->sam_pe_gaussian = gh_load_mmap(
			wf, WP "prompt_encoder.shared_embedding."
			"positional_embedding",
			arena, SAM3_DTYPE_F32, 2, pe_dims);
		if (!trk->sam_pe_gaussian) {
			sam3_log_warn("tracker load: sam_prompt_encoder "
				      "pe_gaussian not loaded (using zeros)");
		} else {
			const float *gm = (const float *)trk->sam_pe_gaussian->data;
			sam3_log_info("pe_gaussian: dims=[%d,%d,%d,%d] "
				      "[0,0..3]=%.4f,%.4f,%.4f,%.4f "
				      "[1,0..3]=%.4f,%.4f,%.4f,%.4f",
				      trk->sam_pe_gaussian->dims[0],
				      trk->sam_pe_gaussian->dims[1],
				      trk->sam_pe_gaussian->dims[2],
				      trk->sam_pe_gaussian->dims[3],
				      gm[0], gm[1], gm[2], gm[3],
				      gm[128], gm[129], gm[130], gm[131]);
		}
		trk->sam_not_a_point = gh_load_mmap(
			wf, WP "prompt_encoder.not_a_point_embed.weight",
			arena, SAM3_DTYPE_F32, 2, emb_dims);
		trk->sam_pt_emb_neg = gh_load_mmap(
			wf, WP "prompt_encoder.point_embeddings.0.weight",
			arena, SAM3_DTYPE_F32, 2, emb_dims);
		trk->sam_pt_emb_pos = gh_load_mmap(
			wf, WP "prompt_encoder.point_embeddings.1.weight",
			arena, SAM3_DTYPE_F32, 2, emb_dims);
		trk->sam_pt_emb_box_tl = gh_load_mmap(
			wf, WP "prompt_encoder.point_embeddings.2.weight",
			arena, SAM3_DTYPE_F32, 2, emb_dims);
		trk->sam_pt_emb_box_br = gh_load_mmap(
			wf, WP "prompt_encoder.point_embeddings.3.weight",
			arena, SAM3_DTYPE_F32, 2, emb_dims);
	}

	sam3_log_info("tracker loaded (%d maskmem, stride %d, image %d)",
		      trk->num_maskmem, trk->backbone_stride,
		      trk->image_size);
	return SAM3_OK;
}

/* --- SAM prompt encoder (point projection for tracker path) --- */

/*
 * Apply Gaussian positional encoding to a normalized coord pair
 * (x, y) in [0, 1]. Matches Python's PositionEmbeddingRandom:
 *   c = 2 * (x, y) - 1          # shift to [-1, 1]
 *   c = c @ positional_matrix   # [128], positional_matrix: [2, 128]
 *   c = 2 * pi * c
 *   return [sin(c), cos(c)]     # [256]
 */
static void sam_pe_encode(float xn, float yn,
			  const struct sam3_tensor *gaussian,
			  float *out_256)
{
	/* Input coords in [0,1] -> scale to [-1, 1] */
	float cx = 2.0f * xn - 1.0f;
	float cy = 2.0f * yn - 1.0f;

	/* Matmul [1, 2] @ [2, 128] = [1, 128] */
	const float *gm = (const float *)gaussian->data;
	float tmp[128];
	for (int i = 0; i < 128; i++) {
		/* Row-major [2, 128]: element (row, col) at gm[row*128 + col] */
		tmp[i] = cx * gm[0 * 128 + i] + cy * gm[1 * 128 + i];
	}
	const float TWO_PI = 6.28318530717958647692f;
	for (int i = 0; i < 128; i++)
		tmp[i] *= TWO_PI;

	/* Output: [sin(tmp), cos(tmp)] concat -> [256].
	 * NOTE: Python returns cat([sin, cos], dim=-1), i.e. sin first. */
	for (int i = 0; i < 128; i++) {
		out_256[i]       = sinf(tmp[i]);
		out_256[i + 128] = cosf(tmp[i]);
	}
}

struct sam3_tensor *sam3_tracker_sam_project_prompts(
	const struct sam3_tracker *trk,
	const struct sam3_prompt *prompts,
	int n_prompts,
	int prompt_w, int prompt_h,
	struct sam3_arena *arena)
{
	if (!trk || !prompts || n_prompts <= 0 || !arena)
		return NULL;
	if (!trk->sam_pe_gaussian) {
		sam3_log_warn("sam_project_prompts: pe_gaussian not loaded");
		return NULL;
	}

	/* Count points vs boxes. */
	int n_points = 0, n_boxes = 0;
	for (int i = 0; i < n_prompts; i++) {
		if (prompts[i].type == SAM3_PROMPT_POINT) n_points++;
		else if (prompts[i].type == SAM3_PROMPT_BOX) n_boxes++;
	}
	/* Point tokens = n_points; box tokens = 2 per box (TL+BR).
	 * Python pads with a "not_a_point" row when no box is present,
	 * regardless of point count. */
	int pad = (n_boxes == 0) ? 1 : 0;
	int n_tokens = n_points + 2 * n_boxes + pad;

	int out_dims[] = {n_tokens, 256};
	struct sam3_tensor *out = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
						  2, out_dims);
	if (!out)
		return NULL;
	float *od = (float *)out->data;
	memset(od, 0, (size_t)n_tokens * 256 * sizeof(float));

	/* Per-point: PE(coord) + point_embeddings[label]. */
	int row = 0;
	for (int i = 0; i < n_prompts; i++) {
		if (prompts[i].type != SAM3_PROMPT_POINT)
			continue;
		/* Python: coords = points + 0.5 (pixel center), then
		 * normalize by image size. */
		float x = (prompts[i].point.x + 0.5f) / (float)prompt_w;
		float y = (prompts[i].point.y + 0.5f) / (float)prompt_h;
		sam_pe_encode(x, y, trk->sam_pe_gaussian, od + (size_t)row * 256);

		/* Add point_embeddings[label]. Label 0 -> neg, 1 -> pos. */
		const struct sam3_tensor *emb = NULL;
		int label = prompts[i].point.label;
		if (label == 0 && trk->sam_pt_emb_neg)
			emb = trk->sam_pt_emb_neg;
		else if (label == 1 && trk->sam_pt_emb_pos)
			emb = trk->sam_pt_emb_pos;
		if (emb) {
			const float *ed = (const float *)emb->data;
			for (int c = 0; c < 256; c++)
				od[(size_t)row * 256 + c] += ed[c];
		}
		row++;
	}

	/* Per-box: 2 corner tokens. */
	for (int i = 0; i < n_prompts; i++) {
		if (prompts[i].type != SAM3_PROMPT_BOX)
			continue;
		float x1 = (prompts[i].box.x1 + 0.5f) / (float)prompt_w;
		float y1 = (prompts[i].box.y1 + 0.5f) / (float)prompt_h;
		float x2 = (prompts[i].box.x2 + 0.5f) / (float)prompt_w;
		float y2 = (prompts[i].box.y2 + 0.5f) / (float)prompt_h;
		sam_pe_encode(x1, y1, trk->sam_pe_gaussian,
			      od + (size_t)row * 256);
		if (trk->sam_pt_emb_box_tl) {
			const float *ed =
				(const float *)trk->sam_pt_emb_box_tl->data;
			for (int c = 0; c < 256; c++)
				od[(size_t)row * 256 + c] += ed[c];
		}
		row++;
		sam_pe_encode(x2, y2, trk->sam_pe_gaussian,
			      od + (size_t)row * 256);
		if (trk->sam_pt_emb_box_br) {
			const float *ed =
				(const float *)trk->sam_pt_emb_box_br->data;
			for (int c = 0; c < 256; c++)
				od[(size_t)row * 256 + c] += ed[c];
		}
		row++;
	}

	/* Pad row: not_a_point_embed (no PE added). */
	if (pad && trk->sam_not_a_point) {
		const float *ed = (const float *)trk->sam_not_a_point->data;
		for (int c = 0; c < 256; c++)
			od[(size_t)row * 256 + c] = ed[c];
		row++;
	}

	return out;
}

/* --- Per-frame tracking  --- */

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
	struct sam3_tensor **out_score)
{
	enum sam3_error err;
	struct sam3_tensor *conditioned_feat = NULL;
	struct sam3_tensor *obj_tok = NULL;

	if (!trk || !g || !bank || !backbone_feat || !arena ||
	    !out_masks || !out_iou) {
		sam3_log_error("tracker track_frame: NULL argument");
		return SAM3_EINVAL;
	}

	if (feat_h <= 0 || feat_w <= 0) {
		sam3_log_error("tracker track_frame: bad feature dims %dx%d",
			       feat_h, feat_w);
		return SAM3_EINVAL;
	}

	/*
	 * Step 1: Prepare conditioned features.
	 *
	 * If the memory bank is empty (first frame or after reset),
	 * follow the upstream "directly_add_no_mem_embed" path: add
	 * the learned no_mem_embed vector to the backbone features
	 * and skip memory attention entirely. Memory attention would
	 * otherwise require a memory tensor in mem_dim (=64), but
	 * no_mem_embed is stored in d_model (=256) — the two cannot
	 * share a tensor.
	 */
	int total_mem = sam3_memory_bank_total(bank);

	if (total_mem == 0) {
		/*
		 * Reshape no_mem_embed from [1, 1, 256] to [1, 256]
		 * so it broadcasts over the [seq, 256] backbone
		 * features in a single add.
		 */
		int nme_dims[] = {1, 256};
		struct sam3_tensor *no_mem = gh_reshape(
			g, arena, trk->no_mem_embed, 2, nme_dims);
		if (!no_mem) {
			sam3_log_error("tracker: no_mem reshape failed");
			return SAM3_ENOMEM;
		}

		conditioned_feat = gh_add(g, arena, backbone_feat, no_mem);
		if (!conditioned_feat) {
			sam3_log_error("tracker: no_mem add failed");
			return SAM3_ENOMEM;
		}
	} else {
		struct sam3_tensor *memory, *mem_pos, *obj_ptrs_cat;
		enum sam3_error attn_err;

		/*
		 * Build a per-frame view that picks the closest
		 * max_cond_frames_in_attn cond entries. Non-cond entries are
		 * taken in full storage order. Matches Python's selected_cond
		 * filtering in sam3_tracker_base.py:_prepare_memory_conditioned_features.
		 */
		struct sam3_memory_bank_view view;
		sam3_memory_bank_build_view(bank, frame_idx, &view);

		/* Collect spatial memory features [total_spatial, mem_dim] */
		struct sam3_tensor *spatial_cat =
			gh_concat_mem(g, arena, &view);
		if (!spatial_cat) {
			sam3_log_error("tracker: gh_concat_mem failed");
			return SAM3_ENOMEM;
		}

		/* Temporal position encoding for spatial tokens */
		struct sam3_tensor *mem_pos_spatial = gh_tpos_enc_mem(
			g, arena, &view,
			trk->maskmem_tpos_enc, frame_idx);
		if (!mem_pos_spatial) {
			sam3_log_error("tracker: gh_tpos_enc_mem failed");
			return SAM3_ENOMEM;
		}

		/* Collect object pointers (may be NULL when no obj_ptrs) */
		obj_ptrs_cat = gh_concat_obj_ptrs(
			g, arena, &view, trk->max_obj_ptrs);

		/* Assemble memory and mem_pos */
		if (obj_ptrs_cat) {
			/*
			 * Temporal PE for obj pointers: 1D sine PE over
			 * normalized frame-distance -> obj_ptr_tpos_proj
			 * Linear(hidden_dim=256, mem_dim=64). Python:
			 * sam3_tracker_base.py:_get_tpos_enc().
			 */
			struct sam3_tensor *obj_ptrs_pe =
				gh_obj_ptrs_tpos_sine(
					g, arena, &view,
					trk->max_obj_ptrs,
					frame_idx, 256);
			struct sam3_tensor *obj_pos = NULL;
			if (obj_ptrs_pe && trk->obj_ptr_tpos_proj_w) {
				obj_pos = gh_linear(
					g, arena, obj_ptrs_pe,
					trk->obj_ptr_tpos_proj_w,
					trk->obj_ptr_tpos_proj_b);
			}
			if (!obj_pos) {
				/* Fallback: zeros if sine PE or projection
				 * failed (keeps legacy behavior). */
				obj_pos = gh_zeros_like(g, arena,
							 obj_ptrs_cat);
			}
			if (!obj_pos) {
				sam3_log_error("tracker: obj_pos alloc failed");
				return SAM3_ENOMEM;
			}
			memory  = gh_concat_rows(g, arena, spatial_cat,
						 obj_ptrs_cat);
			mem_pos = gh_concat_rows(g, arena, mem_pos_spatial,
						 obj_pos);
		} else {
			memory  = spatial_cat;
			mem_pos = mem_pos_spatial;
		}
		if (!memory || !mem_pos) {
			sam3_log_error("tracker: memory assembly failed");
			return SAM3_ENOMEM;
		}

		/*
		 * Run memory attention: cross-attend current features
		 * to memory. The current features live in d_model=256
		 * and are projected down to mem_dim=64 for the K/V path
		 * inside the attention module.
		 */
		attn_err = sam3_memory_attn_build_full(
			&trk->mem_attention, g,
			backbone_feat, memory, mem_pos,
			arena, &conditioned_feat);
		if (attn_err != SAM3_OK) {
			sam3_log_error("tracker: memory attention failed (%d)",
				       attn_err);
			return attn_err;
		}
	}

	/*
	 * Step 3: Run SAM mask decoder.
	 *
	 * Receives:
	 *   out_masks               -> [4, H, W] raw logits
	 *   out_iou                 -> [4] IoU predictions
	 *   obj_tok                 -> [1, 256] obj_score_token post-xfmr
	 *   object_score_logits     -> [1, 1] occlusion logit from MLP
	 *   mask_tokens             -> [4, 256] post-xfmr mask tokens
	 *                              used as input to obj_ptr_proj.
	 */
	struct sam3_tensor *object_score_logits = NULL;
	struct sam3_tensor *mask_tokens = NULL;
	err = sam3_mask_decoder_build(
		&trk->sam_decoder, g, conditioned_feat,
		feat_h, feat_w, prompt, feat_s0, feat_s1,
		arena, out_masks, out_iou, &obj_tok,
		&object_score_logits, &mask_tokens);
	if (err != SAM3_OK) {
		sam3_log_error("tracker: mask decoder failed (%d)", err);
		return err;
	}

	/*
	 * Step 4: Existence score and object pointer.
	 *
	 * Python flow (sam3_tracker_base.py:312-376):
	 *   is_obj_appearing   = object_score_logits > 0         # [B, 1]
	 *   low_res_multimasks = where(is_obj_appearing, m, -1024.0)
	 *   obj_ptr            = obj_ptr_proj(best_mask_token)
	 *   obj_ptr            = is_obj_appearing ? obj_ptr : no_obj_ptr
	 *
	 * The existence score exposed to callers is the post-MLP logit
	 * (scalar), not the per-mask IoU.
	 *
	 * Because the graph is built ahead of evaluation, we cannot
	 * pick the best-IoU mask token at build time; instead we run
	 * obj_ptr_proj over ALL 4 mask tokens at once — the MLP is
	 * linear per row — and emit [4, 256]. The downstream consumer
	 * (sam3_video.c) picks the best-IoU row post-eval when storing
	 * in the memory bank. For the legacy single-pointer output
	 * path, slice row 0 as a sensible default (matches Python's
	 * multimask_output=False case).
	 */
	if (out_score) {
		if (object_score_logits)
			*out_score = object_score_logits;
		else
			*out_score = *out_iou; /* fallback */
	}

	if (out_obj_ptr) {
		if (!mask_tokens) {
			sam3_log_error("tracker: decoder did not emit mask_tokens");
			return SAM3_EMODEL;
		}

		struct sam3_tensor *h;
		/* 3-layer MLP over [4, 256] -> [4, 256] */
		h = gh_linear(g, arena, mask_tokens,
			      trk->obj_ptr_proj_fc0_w,
			      trk->obj_ptr_proj_fc0_b);
		if (!h) return SAM3_ENOMEM;
		h = gh_relu(g, arena, h);
		if (!h) return SAM3_ENOMEM;
		h = gh_linear(g, arena, h,
			      trk->obj_ptr_proj_fc1_w,
			      trk->obj_ptr_proj_fc1_b);
		if (!h) return SAM3_ENOMEM;
		h = gh_relu(g, arena, h);
		if (!h) return SAM3_ENOMEM;
		h = gh_linear(g, arena, h,
			      trk->obj_ptr_proj_fc2_w,
			      trk->obj_ptr_proj_fc2_b);
		if (!h) return SAM3_ENOMEM;
		/* h: [4, 256] obj_ptr per mask token */

		/* Occlusion gate on row 0 only: the downstream best-IoU
		 * selection in sam3_video.c will re-apply the gate, but
		 * this keeps the legacy [1, 256] return shape correct. */
		struct sam3_tensor *row0 = gh_slice(g, arena, h, 0, 0, 1);
		if (!row0) return SAM3_ENOMEM;
		*out_obj_ptr = row0;
	}

	sam3_log_debug("tracker: frame %d tracked (mem=%d, cond=%d)",
		       frame_idx, total_mem, is_cond);
	return SAM3_OK;
}

/* --- Reset  --- */

enum sam3_error sam3_tracker_reset(struct sam3_tracker *trk)
{
	if (!trk) {
		sam3_log_error("tracker reset: NULL tracker");
		return SAM3_EINVAL;
	}

	/*
	 * Task 2.2: mem_bank was removed from struct sam3_tracker. Per-object
	 * banks are owned by sam3_video_object::bank and cleared by
	 * sam3_video_reset directly. Nothing to do here other than return OK.
	 */
	sam3_log_debug("tracker: reset (per-object banks cleared by session)");
	return SAM3_OK;
}
