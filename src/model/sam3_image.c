/*
 * src/model/sam3_image.c - SAM3 top-level image model implementation
 *
 * Wires together all SAM3 sub-modules into a two-phase inference
 * pipeline: encode (run ViT + neck, cache image features) and segment
 * (geometry encoder + encoder fusion + decoder + segmentation head).
 * Both phases evaluate per-stage, resetting the scratch arena between
 * stages to keep peak memory bounded.
 *
 * Key types:  sam3_image_model
 * Depends on: sam3_image.h, graph_helpers.h
 * Used by:    sam3.c (top-level context)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <string.h>
#include <stdio.h>
#include <math.h>

#include "sam3_image.h"
#include "graph_helpers.h"
#include "util/log.h"

enum sam3_error sam3_image_model_init(struct sam3_image_model *model,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	memset(model, 0, sizeof(*model));

	/* Vision-language backbone (ViT + neck + text encoder + tokenizer) */
	err = sam3_vl_backbone_init(&model->backbone, arena);
	if (err != SAM3_OK)
		return err;

	/* Encoder fusion: 6-layer DETR encoder transformer */
	err = sam3_encoder_fusion_init(&model->encoder, 256, 8, 6, 2048);
	if (err != SAM3_OK)
		return err;

	/* Decoder: 6-layer transformer with 200 learned queries */
	err = sam3_decoder_init(&model->decoder, 256, 8, 6, 2048, 200);
	if (err != SAM3_OK)
		return err;

	/* Geometry encoder: 3-layer cross-attention for point/box prompts */
	err = sam3_geometry_encoder_init(&model->geom_enc, 256, 3);
	if (err != SAM3_OK)
		return err;

	/* Segmentation head: pixel decoder + mask prediction */
	err = sam3_seg_head_init(&model->seg_head, 256, 8);
	if (err != SAM3_OK)
		return err;

	/* SAM mask decoder: two-way transformer + pixel decoder */
	err = sam3_mask_decoder_init(&model->mask_dec);
	if (err != SAM3_OK)
		return err;

	/* Objectness scorer config (weights loaded separately) */
	model->scorer.input_dim = 256;
	model->scorer.hidden_dim = 256;

	return SAM3_OK;
}

enum sam3_error sam3_image_model_load(struct sam3_image_model *model,
				      const struct sam3_weight_file *wf,
				      const char *vocab_path,
				      struct sam3_arena *arena)
{
	enum sam3_error err;

	/* Load backbone weights (ViT + neck + text encoder) */
	err = sam3_vl_backbone_load(&model->backbone, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load full CLIP BPE vocabulary if a vocab path is provided */
	if (vocab_path) {
		err = sam3_tokenizer_load_bpe(&model->backbone.tokenizer,
					      vocab_path);
		if (err != SAM3_OK)
			return err;
	}

	/* Load encoder fusion weights */
	err = sam3_encoder_fusion_load(&model->encoder, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load decoder weights */
	err = sam3_decoder_load(&model->decoder, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load geometry encoder weights */
	err = sam3_geometry_encoder_load(&model->geom_enc, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load segmentation head weights */
	err = sam3_seg_head_load(&model->seg_head, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load SAM mask decoder weights */
	err = sam3_mask_decoder_load(&model->mask_dec, wf, arena);
	if (err != SAM3_OK)
		return err;

	/* Load dot-product scorer weights */
	err = sam3_dot_scorer_load(&model->scorer, wf, arena);
	if (err != SAM3_OK)
		return err;

	return SAM3_OK;
}

void sam3_image_model_free(struct sam3_image_model *model)
{
	if (model)
		sam3_vl_backbone_free(&model->backbone);
}

enum sam3_error sam3_image_model_encode(struct sam3_image_model *model,
					struct sam3_backend *be,
					struct sam3_tensor *image,
					struct sam3_arena *scratch,
					struct sam3_arena *persist)
{
	struct sam3_tensor *features[4];
	struct sam3_tensor *vit_out;
	struct sam3_graph g;
	enum sam3_error err;

	/*
	 * Run ViT per-block (evaluates internally), then build
	 * neck graph in scratch. sam3_vl_backbone_build_vision
	 * resets scratch and initializes g after ViT completes.
	 */
	vit_out = sam3_vl_backbone_build_vision(&model->backbone, &g,
						be, image, features,
						scratch, persist);
	if (!vit_out) {
		sam3_log_error("encode: vl_backbone_build_vision failed");
		return SAM3_ENOMEM;
	}

	sam3_log_debug("encode: vision built, %d graph nodes", g.n_nodes);

	/* Diagnostic: check ViT output values before neck eval */
	{
		const float *vd = (const float *)vit_out->data;
		int vn = 1;
		for (int vi = 0; vi < vit_out->n_dims; vi++)
			vn *= vit_out->dims[vi];
		float vmin = vd[0], vmax = vd[0];
		int vnz = 0;
		for (int vi = 0; vi < vn; vi++) {
			if (vd[vi] < vmin) vmin = vd[vi];
			if (vd[vi] > vmax) vmax = vd[vi];
			if (vd[vi] != 0.0f) vnz++;
		}
		sam3_log_debug("encode: vit_out [%d] dims=[%d,%d] "
			"min=%.4f max=%.4f nz=%d/%d",
			vn, vit_out->dims[0], vit_out->dims[1],
			vmin, vmax, vnz, vn);

		/* Check ALL neck conv weights to verify loading */
		for (int si = 0; si < model->backbone.neck.n_scales; si++) {
			for (int cj = 0; cj < model->backbone.neck.stages[si].n_convs; cj++) {
				struct sam3_tensor *nw = model->backbone.neck.stages[si].conv_w[cj];
				if (!nw || !nw->data) continue;
				const float *nd = (const float *)nw->data;
				int nn = 1;
				for (int ni = 0; ni < nw->n_dims; ni++)
					nn *= nw->dims[ni];
				float nmin = nd[0], nmax = nd[0];
				int nnz = 0;
				for (int ni = 0; ni < nn; ni++) {
					if (nd[ni] < nmin) nmin = nd[ni];
					if (nd[ni] > nmax) nmax = nd[ni];
					if (nd[ni] != 0.0f) nnz++;
				}
				sam3_log_debug("encode: neck[%d].conv[%d].w "
					"[%d] min=%.4f max=%.4f nz=%d/%d",
					si, cj, nn, nmin, nmax, nnz, nn);
			}
		}
	}

	/*
	 * Evaluate FPN neck stages one at a time.
	 *
	 * Evaluating all 4 stages in a single graph causes the shared
	 * NCHW input tensor to be corrupted (MLX reuses intermediate
	 * buffers). Instead, we: (1) evaluate transpose+reshape to get
	 * NCHW, (2) evaluate each stage as its own graph.
	 *
	 * All scratch allocations accumulate (no arena resets) so that
	 * stage outputs survive until we copy them to persist at the end.
	 */
	{
		/*
		 * Step 1: Manual C transpose from [pixels, C] to NCHW.
		 *
		 * ViT output is [HW, C] row-major. We need [1, C, H, W].
		 * The MLX graph-based transpose produces garbage due to
		 * intermediate buffer handling, so we do it in C.
		 *
		 * dst[c * HW + p] = src[p * C + c]
		 */
		sam3_arena_reset(scratch);

		int C = model->backbone.neck.backbone_dim;
		int gs = model->backbone.neck.grid_size;
		int HW = gs * gs;
		int nchw_dims[] = {1, C, gs, gs};
		size_t nchw_bytes = (size_t)C * HW * sizeof(float);
		void *nchw_buf = sam3_arena_alloc(scratch, nchw_bytes);
		if (!nchw_buf) return SAM3_ENOMEM;

		{
			const float *src = (const float *)vit_out->data;
			float *dst = (float *)nchw_buf;
			for (int p = 0; p < HW; p++)
				for (int c = 0; c < C; c++)
					dst[c * HW + p] = src[p * C + c];
		}

		sam3_log_debug("encode: nchw manual transpose done "
			"[%d,%d,%d,%d]", 1, C, gs, gs);

		/* Step 2: Evaluate each stage independently */
		struct sam3_tensor *stage_out[4] = {0};

		for (int si = 0; si < model->backbone.neck.n_scales; si++) {
			struct sam3_tensor *nchw_in;
			nchw_in = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
						  4, nchw_dims, nchw_buf);
			if (!nchw_in) return SAM3_ENOMEM;

			struct sam3_tensor *x = nchw_in;

			if (model->backbone.neck.stages[si].has_maxpool) {
				struct sam3_graph g_stage;
				sam3_graph_init(&g_stage);
				x = gh_maxpool2d(&g_stage, scratch, x, 2, 2);
				if (!x) return SAM3_ENOMEM;
				err = be->ops->graph_eval(be, &g_stage);
				if (err != SAM3_OK) return err;
				x = gh_tensor_wrap(scratch, x->dtype,
					x->n_dims, x->dims, x->data);
				if (!x) return SAM3_ENOMEM;
			}

			for (int j = 0; j < model->backbone.neck.stages[si].n_convs; j++) {
				struct sam3_graph g_stage;
				sam3_graph_init(&g_stage);
				int k = model->backbone.neck.stages[si].kernel_size[j];
				int pad = (k == 3) ? 1 : 0;

				if (model->backbone.neck.stages[si].is_transpose[j]) {
					x = gh_conv_transpose2d(&g_stage, scratch, x,
						model->backbone.neck.stages[si].conv_w[j],
						model->backbone.neck.stages[si].conv_b[j],
						2, 0);
				} else {
					x = gh_conv2d(&g_stage, scratch, x,
						model->backbone.neck.stages[si].conv_w[j],
						model->backbone.neck.stages[si].conv_b[j],
						1, pad);
				}
				if (!x) return SAM3_ENOMEM;

				if (model->backbone.neck.stages[si].gelu_after[j]) {
					x = gh_gelu(&g_stage, scratch, x);
					if (!x) return SAM3_ENOMEM;
				}

				err = be->ops->graph_eval(be, &g_stage);
				if (err != SAM3_OK) return err;

				/* Snapshot layer output + smoothness */
				{
					const float *ld =
						(const float *)x->data;
					int lC = x->dims[1];
					int lH = x->dims[2];
					int lW = x->dims[3];
					int ln = lC * lH * lW;
					float lmin = ld[0], lmax = ld[0];
					double la = 0, ldw = 0, ldh = 0;
					long ca = 0, cw2 = 0, ch3 = 0;
					for (int c = 0; c < lC; c++) {
						const float *lc =
							ld + (long)c*lH*lW;
						for (int h = 0; h < lH; h++)
						for (int w = 0; w < lW; w++) {
							float v = lc[h*lW+w];
							if (v < lmin) lmin = v;
							if (v > lmax) lmax = v;
							la += fabs(v); ca++;
							if (w+1<lW) {
								ldw += fabs(v-
								  lc[h*lW+w+1]);
								cw2++;
							}
							if (h+1<lH) {
								ldh += fabs(v-
								  lc[(h+1)*lW+w]);
								ch3++;
							}
						}
					}
					double lm = la / ca;
					double rw = (ldw/cw2)/(lm+1e-8);
					double rh = (ldh/ch3)/(lm+1e-8);
					sam3_log_debug("encode: neck[%d].layer[%d]"
						" [%d,%d,%d,%d] min=%.4f"
						" max=%.4f rw=%.3f rh=%.3f",
						si, j,
						x->dims[0], x->dims[1],
						x->dims[2], x->dims[3],
						lmin, lmax, rw, rh);
				}

				/* Wrap for next layer */
				x = gh_tensor_wrap(scratch, x->dtype,
					x->n_dims, x->dims, x->data);
				if (!x) return SAM3_ENOMEM;
			}

			/*
			 * Copy stage output to scratch immediately.
			 * MLX recycles internal buffers between
			 * graph_eval calls, so we must snapshot data
			 * before building the next stage's graph.
			 */
			stage_out[si] = gh_alloc_tensor(
				scratch, x->dtype, x->n_dims, x->dims);
			if (!stage_out[si]) return SAM3_ENOMEM;
			memcpy(stage_out[si]->data, x->data, x->nbytes);

			{
				const float *sd =
					(const float *)stage_out[si]->data;
				int sn = 1;
				for (int sj = 0; sj < x->n_dims; sj++)
					sn *= x->dims[sj];
				float smin = sd[0], smax = sd[0];
				for (int sj = 0; sj < sn; sj++) {
					if (sd[sj] < smin) smin = sd[sj];
					if (sd[sj] > smax) smax = sd[sj];
				}
				sam3_log_debug("encode: feat[%d] "
					"[%d,%d,%d,%d] "
					"min=%.4f max=%.4f",
					si, x->dims[0], x->dims[1],
					x->dims[2], x->dims[3],
					smin, smax);
			}
		}

		/* Step 3: Copy outputs to persist arena */
		struct sam3_tensor *pf[4];
		for (int si = 0; si < 4; si++) {
			pf[si] = gh_alloc_tensor(persist, stage_out[si]->dtype,
						  stage_out[si]->n_dims,
						  stage_out[si]->dims);
			if (!pf[si]) return SAM3_ENOMEM;
			memcpy(pf[si]->data, stage_out[si]->data,
			       stage_out[si]->nbytes);
		}

		model->cached_feat_4x = pf[0];
		model->cached_feat_s0 = pf[1];
		model->cached_feat_s1 = pf[2];
		model->cached_image_features = pf[3];

		sam3_log_debug("encode: cached features: main [%d,%d,%d,%d]"
			       " s0 [%d,%d,%d,%d] s1 [%d,%d,%d,%d]"
			       " 4x [%d,%d,%d,%d]",
			       pf[3]->dims[0], pf[3]->dims[1],
			       pf[3]->dims[2], pf[3]->dims[3],
			       pf[1]->dims[0], pf[1]->dims[1],
			       pf[1]->dims[2], pf[1]->dims[3],
			       pf[2]->dims[0], pf[2]->dims[1],
			       pf[2]->dims[2], pf[2]->dims[3],
			       pf[0]->dims[0], pf[0]->dims[1],
			       pf[0]->dims[2], pf[0]->dims[3]);
	}

	model->image_encoded = 1;
	return SAM3_OK;
}

/*
 * nchw_to_2d - Transpose [1, C, H, W] NCHW tensor to [H*W, C] 2D tensor.
 *
 * Needed because the neck stores features in NCHW format but the encoder
 * fusion, geometry encoder, and segmentation head expect [n_pixels, d_model].
 */
static struct sam3_tensor *nchw_to_2d(struct sam3_arena *arena,
				       struct sam3_tensor *nchw)
{
	int C = nchw->dims[1];
	int H = nchw->dims[2];
	int W = nchw->dims[3];
	int HW = H * W;
	int dims[2] = { HW, C };
	struct sam3_tensor *out;
	const float *src;
	float *dst;
	int hw, c;

	out = gh_alloc_tensor(arena, nchw->dtype, 2, dims);
	if (!out)
		return NULL;

	src = (const float *)nchw->data;
	dst = (float *)out->data;
	for (hw = 0; hw < HW; hw++)
		for (c = 0; c < C; c++)
			dst[hw * C + c] = src[c * HW + hw];

	return out;
}

/*
 * persist_tensor - Copy a tensor (struct + data) to the persist arena.
 */
static struct sam3_tensor *persist_tensor(struct sam3_arena *persist,
					  struct sam3_tensor *src)
{
	struct sam3_tensor *dst;

	dst = gh_alloc_tensor(persist, src->dtype, src->n_dims, src->dims);
	if (!dst)
		return NULL;
	memcpy(dst->data, src->data, src->nbytes);
	return dst;
}

/*
 * concat_2d_persist - Concatenate two 2D tensors along axis 0 in persist.
 *
 * Manually copies data without a graph op, so the result is immediately
 * materialized in the persist arena.
 */
static struct sam3_tensor *concat_2d_persist(struct sam3_arena *persist,
					     struct sam3_tensor *a,
					     struct sam3_tensor *b)
{
	int dims[2];
	struct sam3_tensor *out;

	dims[0] = a->dims[0] + b->dims[0];
	dims[1] = a->dims[1];
	out = gh_alloc_tensor(persist, a->dtype, 2, dims);
	if (!out)
		return NULL;
	memcpy(out->data, a->data, a->nbytes);
	memcpy((char *)out->data + a->nbytes, b->data, b->nbytes);
	return out;
}

enum sam3_error sam3_image_model_segment(
	struct sam3_image_model *model,
	struct sam3_backend *be,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *text_features,
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_tensor **out_masks,
	struct sam3_tensor **out_scores)
{
	struct sam3_graph g;
	struct sam3_tensor *img_2d;
	struct sam3_tensor *geom_out = NULL;
	struct sam3_tensor *context;
	struct sam3_tensor *fused;
	struct sam3_tensor *masks;
	struct sam3_tensor *scores = NULL;
	enum sam3_error err;
	size_t persist_save;
	int grid_h, grid_w;

	if (!model->image_encoded)
		return SAM3_EINVAL;

	if (!prompt_tokens && !text_features)
		return SAM3_EINVAL;

	/* Save persist offset so inter-stage data can be freed later */
	persist_save = persist->offset;

	/*
	 * Reshape cached image features from NCHW [1, C, H, W] to
	 * [H*W, d_model]. The neck stores features in NCHW but all
	 * downstream modules expect [n_pixels, d_model].
	 */
	img_2d = nchw_to_2d(persist, model->cached_image_features);
	if (!img_2d)
		return SAM3_ENOMEM;

	/*
	 * Stage 1: Geometry encoder — cross-attend prompt tokens to
	 * cached image features. Output is [N+1, d_model].
	 */
	if (prompt_tokens) {
		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		geom_out = sam3_geometry_encoder_build(
			&model->geom_enc, &g,
			prompt_tokens,
			img_2d,
			scratch);
		if (!geom_out) {
			err = SAM3_ENOMEM;
			goto fail;
		}

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK) {
			sam3_log_error("segment: geom eval failed: %d",
				       err);
			goto fail;
		}

		geom_out = persist_tensor(persist, geom_out);
		if (!geom_out) {
			err = SAM3_ENOMEM;
			goto fail;
		}

		sam3_log_debug("segment: geom encoder done, "
			       "persist %zu/%zu",
			       persist->offset, persist->size);
	}

	/*
	 * Build context for encoder fusion and decoder. Manual
	 * concat in persist arena (no graph op needed).
	 */
	if (text_features && geom_out) {
		context = concat_2d_persist(persist, text_features,
					    geom_out);
		if (!context) {
			err = SAM3_ENOMEM;
			goto fail;
		}
	} else if (text_features) {
		context = text_features;
	} else {
		context = geom_out;
	}

	/*
	 * Stage 2: Encoder fusion — fuse image features with context
	 * (text and/or geometry). Output is [n_pixels, d_model].
	 */
	{
		const float *id = (const float *)img_2d->data;
		int in = img_2d->dims[0] * img_2d->dims[1];
		float imin = id[0], imax = id[0];
		for (int ii = 0; ii < in; ii++) {
			if (id[ii] < imin) imin = id[ii];
			if (id[ii] > imax) imax = id[ii];
		}
		sam3_log_debug("segment: img_2d [%d,%d] min=%.4f max=%.4f",
			       img_2d->dims[0], img_2d->dims[1], imin, imax);

		const float *cd = (const float *)context->data;
		int cn = context->dims[0] * context->dims[1];
		float cmin = cd[0], cmax = cd[0];
		for (int ci = 0; ci < cn; ci++) {
			if (cd[ci] < cmin) cmin = cd[ci];
			if (cd[ci] > cmax) cmax = cd[ci];
		}
		sam3_log_debug("segment: context [%d,%d] min=%.4f max=%.4f",
			       context->dims[0], context->dims[1], cmin, cmax);
	}

	/*
	 * Evaluate encoder fusion per-layer to avoid MLX
	 * shared-buffer corruption in large graphs.
	 */
	{
		struct sam3_tensor *enc_x = img_2d;
		size_t x_bytes = enc_x->dims[0] * enc_x->dims[1]
				  * sizeof(float);
		void *x_buf = sam3_arena_alloc(persist, x_bytes);
		if (!x_buf) { err = SAM3_ENOMEM; goto fail; }
		memcpy(x_buf, enc_x->data, x_bytes);

		for (int li = 0; li < model->encoder.n_layers; li++) {
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int x_dims[] = {img_2d->dims[0],
					 img_2d->dims[1]};
			enc_x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
						2, x_dims, x_buf);
			if (!enc_x) { err = SAM3_ENOMEM; goto fail; }

			struct sam3_tensor *ctx_copy;
			ctx_copy = gh_tensor_wrap(scratch, context->dtype,
						   context->n_dims,
						   context->dims,
						   context->data);
			if (!ctx_copy) { err = SAM3_ENOMEM; goto fail; }

			enc_x = sam3_encoder_fusion_build_layer(
				&model->encoder, li, &g,
				enc_x, ctx_copy, scratch);
			if (!enc_x) { err = SAM3_ENOMEM; goto fail; }

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("segment: enc layer %d "
					       "eval failed: %d", li, err);
				goto fail;
			}
			memcpy(x_buf, enc_x->data, x_bytes);

			/* Diagnostic: per-layer stats */
			{
				const float *ld = (const float *)x_buf;
				int ln = img_2d->dims[0] * img_2d->dims[1];
				float lmin = ld[0], lmax = ld[0];
				for (int lj = 0; lj < ln; lj++) {
					if (ld[lj] < lmin) lmin = ld[lj];
					if (ld[lj] > lmax) lmax = ld[lj];
				}
				sam3_log_debug("segment: enc layer %d "
					       "min=%.4f max=%.4f",
					       li, lmin, lmax);
			}
		}

		/* Final layer norm */
		sam3_arena_reset(scratch);
		sam3_graph_init(&g);
		int x_dims[] = {img_2d->dims[0], img_2d->dims[1]};
		enc_x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					2, x_dims, x_buf);
		if (!enc_x) { err = SAM3_ENOMEM; goto fail; }

		fused = sam3_encoder_fusion_build_final(
			&model->encoder, &g, enc_x, scratch);
		if (!fused) { err = SAM3_ENOMEM; goto fail; }

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK) {
			sam3_log_error("segment: enc final ln eval "
				       "failed: %d", err);
			goto fail;
		}

		fused = persist_tensor(persist, fused);
		if (!fused) { err = SAM3_ENOMEM; goto fail; }
	}

	sam3_log_debug("segment: encoder fusion done, persist %zu/%zu",
		       persist->offset, persist->size);

	/* Diagnostic: check fused image feature stats */
	{
		const float *fd = (const float *)fused->data;
		int fn = fused->dims[0] * fused->dims[1];
		float fmin = fd[0], fmax = fd[0], fsum = 0;
		for (int fi = 0; fi < fn; fi++) {
			if (fd[fi] < fmin) fmin = fd[fi];
			if (fd[fi] > fmax) fmax = fd[fi];
			fsum += fd[fi];
		}
		sam3_log_debug("segment: fused features [%d,%d] "
			       "min=%.4f max=%.4f mean=%.4f",
			       fused->dims[0], fused->dims[1],
			       fmin, fmax, fsum / fn);
	}

	/*
	 * Stage 3: DETR decoder — per-layer evaluation to avoid MLX
	 * shared-buffer corruption in the 961-node combined graph.
	 */
	{
		struct sam3_tensor *queries;

		/*
		 * Initialize query and box state in persist arena so
		 * they survive across per-layer scratch resets.
		 */
		int nq = model->decoder.n_queries;
		int d = model->decoder.d_model;
		int q_dims[] = {nq, d};

		/* Copy query embeddings to persist as mutable state */
		void *q_buf = sam3_arena_alloc(persist,
			(size_t)nq * d * sizeof(float));
		if (!q_buf) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		memcpy(q_buf, model->decoder.query_embed->data,
		       (size_t)nq * d * sizeof(float));

		for (int li = 0; li < model->decoder.n_layers; li++) {
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			/*
			 * Wrap persisted q as graph input. Pass NULL
			 * for boxes to skip box refinement — this
			 * ensures q is the final graph node and the
			 * Metal backend copies its data back.
			 */
			struct sam3_tensor *q_in;
			struct sam3_tensor *enc_wrap;
			struct sam3_tensor *txt_wrap;

			q_in = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					       2, q_dims, q_buf);
			enc_wrap = gh_tensor_wrap(scratch,
						   fused->dtype,
						   fused->n_dims,
						   fused->dims,
						   fused->data);
			txt_wrap = gh_tensor_wrap(scratch,
						   text_features->dtype,
						   text_features->n_dims,
						   text_features->dims,
						   text_features->data);
			if (!q_in || !enc_wrap || !txt_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			struct sam3_tensor *q_out;
			q_out = sam3_decoder_build_layer(
				&model->decoder, li, &g,
				q_in, enc_wrap, txt_wrap,
				NULL, scratch);
			if (!q_out) {
				sam3_log_error("segment: dec layer %d "
					       "build failed", li);
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("segment: dec layer %d "
					       "eval failed: %d", li, err);
				goto fail;
			}

			/* Snapshot result back to persist buffer */
			memcpy(q_buf, q_out->data,
			       (size_t)nq * d * sizeof(float));

			/* Diagnostic: per-layer stats */
			{
				const float *ld = (const float *)q_buf;
				int ln = nq * d;
				float lmin = ld[0], lmax = ld[0];
				for (int lj = 0; lj < ln; lj++) {
					if (ld[lj] < lmin) lmin = ld[lj];
					if (ld[lj] > lmax) lmax = ld[lj];
				}
				sam3_log_debug("segment: dec layer %d "
					       "q min=%.4f max=%.4f",
					       li, lmin, lmax);
			}
		}

		/* Final: output layer norm */
		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		struct sam3_tensor *q_final;
		q_final = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					  2, q_dims, q_buf);
		if (!q_final) {
			err = SAM3_ENOMEM;
			goto fail;
		}

		queries = sam3_decoder_build_final(
			&model->decoder, &g, q_final, scratch);
		if (!queries) {
			sam3_log_error("segment: dec final ln failed");
			err = SAM3_ENOMEM;
			goto fail;
		}

		err = be->ops->graph_eval(be, &g);
		if (err != SAM3_OK) {
			sam3_log_error("segment: dec final eval failed");
			goto fail;
		}

		queries = persist_tensor(persist, queries);
		if (!queries) {
			err = SAM3_ENOMEM;
			goto fail;
		}

		/* Diagnostic: check decoder query values */
		{
			const float *qd = (const float *)queries->data;
			int qn = queries->dims[0] * queries->dims[1];
			float qmin = qd[0], qmax = qd[0], qsum = 0;
			int qnz = 0;
			for (int qi = 0; qi < qn; qi++) {
				if (qd[qi] < qmin) qmin = qd[qi];
				if (qd[qi] > qmax) qmax = qd[qi];
				qsum += qd[qi];
				if (qd[qi] != 0.0f) qnz++;
			}
			sam3_log_debug("segment: decoder queries [%d,%d] "
				"min=%.4f max=%.4f mean=%.4f nz=%d/%d",
				queries->dims[0], queries->dims[1],
				qmin, qmax, qsum / qn, qnz, qn);
		}

		/*
		 * Stage 4: Segmentation head — FPN pixel decoder +
		 * mask embedder MLP + dot product mask prediction.
		 */
		grid_h = model->cached_image_features->dims[2];
		grid_w = model->cached_image_features->dims[3];

		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		/* Diagnostic: check backbone feature stats */
		{
			struct sam3_tensor *bf[] = {
				fused,
				model->cached_feat_s1,
				model->cached_feat_s0,
				model->cached_feat_4x
			};
			const char *bn[] = {
				"fused", "feat_1x", "feat_2x", "feat_4x"
			};
			for (int bi = 0; bi < 4; bi++) {
				struct sam3_tensor *bt = bf[bi];
				const float *bd =
					(const float *)bt->data;
				int bsz = 1;
				for (int bj = 0; bj < bt->n_dims; bj++)
					bsz *= bt->dims[bj];
				float bmin = bd[0], bmax = bd[0];
				int bnz = 0;
				for (int bj = 0; bj < bsz; bj++) {
					if (bd[bj] < bmin) bmin = bd[bj];
					if (bd[bj] > bmax) bmax = bd[bj];
					if (bd[bj] != 0.0f) bnz++;
				}
				sam3_log_debug("seg input: %s [%d] "
					"min=%.4f max=%.4f nz=%d/%d",
					bn[bi], bsz,
					bmin, bmax, bnz, bsz);
			}
		}

		/* Diagnostic: check loaded weight stats */
		{
			struct sam3_seg_head *sh = &model->seg_head;
			struct sam3_tensor *wt[] = {
				sh->mask_mlp[0].w, sh->mask_mlp[0].b,
				sh->inst_proj_w,
				sh->pxattn_q_w,
				sh->fpn[0].conv_w,
			};
			const char *wn[] = {
				"mlp0_w", "mlp0_b",
				"inst_w", "xattn_q_w", "fpn0_conv_w",
			};
			for (int wi = 0; wi < 5; wi++) {
				struct sam3_tensor *ww = wt[wi];
				if (!ww || !ww->data) {
					sam3_log_debug("seg wt: %s NULL",
						       wn[wi]);
					continue;
				}
				const float *wd =
					(const float *)ww->data;
				int wn2 = 1;
				for (int wj = 0; wj < ww->n_dims; wj++)
					wn2 *= ww->dims[wj];
				float wmin = wd[0], wmax = wd[0];
				int wnz = 0;
				for (int wj = 0; wj < wn2; wj++) {
					if (wd[wj] < wmin) wmin = wd[wj];
					if (wd[wj] > wmax) wmax = wd[wj];
					if (wd[wj] != 0.0f) wnz++;
				}
				sam3_log_debug("seg wt: %s [%d] "
					"min=%.6f max=%.6f nz=%d/%d",
					wn[wi], wn2,
					wmin, wmax, wnz, wn2);
			}
		}

		/*
		 * Prompt cross-attention must be evaluated as a
		 * separate graph because seg_head_build does a
		 * manual NCHW transpose that reads tensor data.
		 */
		struct sam3_tensor *seg_enc = fused;
		if (text_features) {
			struct sam3_tensor *enc_wrap;
			struct sam3_tensor *txt_wrap;

			enc_wrap = gh_tensor_wrap(scratch,
				fused->dtype, fused->n_dims,
				fused->dims, fused->data);
			txt_wrap = gh_tensor_wrap(scratch,
				text_features->dtype,
				text_features->n_dims,
				text_features->dims,
				text_features->data);
			if (!enc_wrap || !txt_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			seg_enc = sam3_seg_head_build_cross_attn(
				&model->seg_head, &g,
				enc_wrap, txt_wrap, scratch);
			if (!seg_enc) {
				sam3_log_error("segment: cross-attn "
					       "build failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("segment: cross-attn "
					       "eval failed");
				goto fail;
			}

			/* Materialize result for NCHW transpose */
			seg_enc = persist_tensor(persist, seg_enc);
			if (!seg_enc) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			sam3_log_debug("segment: cross-attn done "
				"[%d,%d]", seg_enc->dims[0],
				seg_enc->dims[1]);

			sam3_arena_reset(scratch);
			sam3_graph_init(&g);
		}

		/*
		 * Stage 4a: Manual NCHW transpose of seg_enc.
		 */
		int seg_d = model->seg_head.d_model;
		int seg_seq = seg_enc->dims[0];
		int nchw_sd[] = {1, seg_d, grid_h, grid_w};
		struct sam3_tensor *enc_nchw;
		enc_nchw = gh_alloc_tensor(scratch, seg_enc->dtype,
					    4, nchw_sd);
		if (!enc_nchw) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		{
			const float *src = (const float *)seg_enc->data;
			float *dst = (float *)enc_nchw->data;
			for (int s = 0; s < seg_seq; s++)
				for (int c = 0; c < seg_d; c++)
					dst[c * seg_seq + s] =
						src[s * seg_d + c];
		}
		sam3_log_debug("seg: enc_nchw [%d,%d,%d,%d]",
			       1, seg_d, grid_h, grid_w);

		/* Diagnostic: enc_nchw stats */
		{
			const float *nd = (const float *)enc_nchw->data;
			int nn = seg_d * seg_seq;
			float nmin = nd[0], nmax = nd[0];
			for (int ni = 0; ni < nn; ni++) {
				if (nd[ni] < nmin) nmin = nd[ni];
				if (nd[ni] > nmax) nmax = nd[ni];
			}
			sam3_log_debug("seg: enc_nchw min=%.4f max=%.4f",
				       nmin, nmax);
		}

		/*
		 * Stage 4b: FPN pixel decoder + instance projection.
		 * Evaluated as separate graph to get real inst values.
		 */
		struct sam3_tensor *inst;
		{
			sam3_graph_init(&g);

			struct sam3_tensor *nchw_wrap;
			nchw_wrap = gh_tensor_wrap(scratch,
				enc_nchw->dtype, enc_nchw->n_dims,
				enc_nchw->dims, enc_nchw->data);
			if (!nchw_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			/* Smoothness of skip features */
			{
				const char *names[] = {"feat_s1(1x)",
					"feat_s0(2x)", "feat_4x"};
				struct sam3_tensor *skips[] = {
					model->cached_feat_s1,
					model->cached_feat_s0,
					model->cached_feat_4x};
				for (int si = 0; si < 3; si++) {
					struct sam3_tensor *f = skips[si];
					const float *fd =
						(const float *)f->data;
					int fC = f->dims[1];
					int fH = f->dims[2];
					int fW = f->dims[3];
					double fa = 0, fdw = 0, fdh = 0;
					long ca = 0, cw = 0, ch2 = 0;
					for (int c = 0; c < fC; c++) {
						const float *fc =
							fd + (long)c*fH*fW;
						for (int h = 0; h < fH; h++)
						for (int w = 0; w < fW; w++) {
							float v = fc[h*fW+w];
							fa += fabs(v); ca++;
							if (w+1<fW) {
								fdw += fabs(v -
								  fc[h*fW+w+1]);
								cw++;
							}
							if (h+1<fH) {
								fdh += fabs(v -
								  fc[(h+1)*fW+w]);
								ch2++;
							}
						}
					}
					double m = fa/ca;
					double rw = (fdw/cw)/(m+1e-8);
					double rh = (fdh/ch2)/(m+1e-8);
					sam3_log_debug("seg: %s [%d,%d,%d,%d]"
						" rw=%.3f rh=%.3f mean=%.4f",
						names[si],
						f->dims[0], f->dims[1],
						f->dims[2], f->dims[3],
						rw, rh, m);
				}
			}

			inst = sam3_seg_head_build_fpn(
				&model->seg_head, &g,
				nchw_wrap,
				model->cached_feat_s1,
				model->cached_feat_s0,
				model->cached_feat_4x,
				scratch);
			if (!inst) {
				sam3_log_error("seg: FPN build failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			sam3_log_debug("seg: FPN+inst built, %d nodes",
				       g.n_nodes);

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("seg: FPN eval failed");
				goto fail;
			}

			inst = persist_tensor(persist, inst);
			if (!inst) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			{
				const float *id =
					(const float *)inst->data;
				int in_sz = 1;
				for (int ij = 0; ij < inst->n_dims; ij++)
					in_sz *= inst->dims[ij];
				float imin = id[0], imax = id[0];
				for (int ij = 0; ij < in_sz; ij++) {
					if (id[ij] < imin) imin = id[ij];
					if (id[ij] > imax) imax = id[ij];
				}
				sam3_log_debug("seg: inst [%d,%d,%d,%d] "
					"min=%.4f max=%.4f",
					inst->dims[0], inst->dims[1],
					inst->dims[2], inst->dims[3],
					imin, imax);

				/* Spatial smoothness metric */
				int iC = inst->dims[1];
				int iH = inst->dims[2];
				int iW = inst->dims[3];
				double adj_h = 0, adj_w = 0, aabs = 0;
				long cnt_h = 0, cnt_w = 0, cnt_a = 0;
				for (int c = 0; c < iC; c++) {
					const float *ch =
						id + (long)c * iH * iW;
					for (int h = 0; h < iH; h++)
					for (int w = 0; w < iW; w++) {
						float v = ch[h*iW + w];
						aabs += fabs(v);
						cnt_a++;
						if (w+1 < iW) {
							adj_w += fabs(v -
								ch[h*iW+w+1]);
							cnt_w++;
						}
						if (h+1 < iH) {
							adj_h += fabs(v -
								ch[(h+1)*iW+w]);
							cnt_h++;
						}
					}
				}
				double ma = aabs / cnt_a;
				double dw = adj_w / cnt_w;
				double dh = adj_h / cnt_h;
				sam3_log_debug("seg: inst smooth: "
					"mean|v|=%.4f dw=%.4f dh=%.4f "
					"rw=%.3f rh=%.3f",
					ma, dw, dh,
					dw / (ma + 1e-8),
					dh / (ma + 1e-8));
			}
		}

		/*
		 * Stage 4c: Mask embedder MLP on queries.
		 */
		struct sam3_tensor *mask_embed;
		{
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			struct sam3_tensor *q_wrap;
			q_wrap = gh_tensor_wrap(scratch,
				queries->dtype, queries->n_dims,
				queries->dims, queries->data);
			if (!q_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			mask_embed = sam3_seg_head_build_mask_embed(
				&model->seg_head, &g, q_wrap, scratch);
			if (!mask_embed) {
				sam3_log_error("seg: mask MLP failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("seg: mask MLP eval failed");
				goto fail;
			}

			mask_embed = persist_tensor(persist, mask_embed);
			if (!mask_embed) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			{
				const float *md =
					(const float *)mask_embed->data;
				int mn = mask_embed->dims[0]
					  * mask_embed->dims[1];
				float mmin = md[0], mmax = md[0];
				for (int mi = 0; mi < mn; mi++) {
					if (md[mi] < mmin) mmin = md[mi];
					if (md[mi] > mmax) mmax = md[mi];
				}
				sam3_log_debug("seg: mask_embed [%d,%d] "
					"min=%.4f max=%.4f",
					mask_embed->dims[0],
					mask_embed->dims[1],
					mmin, mmax);
			}
		}

		/*
		 * Stage 4d: Dot product mask logits.
		 * mask_embed [nq, d] @ inst_flat [d, H*W] → [nq, H*W]
		 */
		{
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int final_h = inst->dims[2];
			int final_w = inst->dims[3];
			int final_hw = final_h * final_w;
			int n_q = queries->dims[0];

			struct sam3_tensor *me_wrap, *inst_wrap;
			me_wrap = gh_tensor_wrap(scratch,
				mask_embed->dtype, mask_embed->n_dims,
				mask_embed->dims, mask_embed->data);
			inst_wrap = gh_tensor_wrap(scratch,
				inst->dtype, inst->n_dims,
				inst->dims, inst->data);
			if (!me_wrap || !inst_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			/* Reshape inst [1,d,H,W] → [d, H*W] */
			int flat_dims[] = {seg_d, final_hw};
			struct sam3_tensor *inst_flat;
			inst_flat = gh_reshape(&g, scratch, inst_wrap,
						2, flat_dims);
			if (!inst_flat) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			/* masks = mask_embed @ inst_flat */
			masks = gh_matmul(&g, scratch,
					   me_wrap, inst_flat);
			if (!masks) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			/* Reshape [nq, H*W] → [nq, H, W] */
			int mask_dims[] = {n_q, final_h, final_w};
			masks = gh_reshape(&g, scratch, masks,
					    3, mask_dims);
			if (!masks) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("seg: dot product eval failed");
				goto fail;
			}

			{
				const float *dd =
					(const float *)masks->data;
				int dn = n_q * final_h * final_w;
				float dmin = dd[0], dmax = dd[0];
				for (int dj = 0; dj < dn; dj++) {
					if (dd[dj] < dmin) dmin = dd[dj];
					if (dd[dj] > dmax) dmax = dd[dj];
				}
				sam3_log_debug("seg: masks [%d,%d,%d] "
					"min=%.4f max=%.4f",
					n_q, final_h, final_w,
					dmin, dmax);

			/* Per-mask positive pixel stats */
			for (int qi = 0; qi < n_q; qi++) {
				const float *mrow = dd + qi * final_h * final_w;
				float mmin = mrow[0], mmax = mrow[0];
				int npos = 0;
				for (int j = 0; j < final_h * final_w; j++) {
					if (mrow[j] < mmin) mmin = mrow[j];
					if (mrow[j] > mmax) mmax = mrow[j];
					if (mrow[j] > 0.0f) npos++;
				}
				float frac = (float)npos / (final_h * final_w);
				if (qi < 5 || (frac > 0.05f && frac < 0.6f))
					sam3_log_debug("seg: mask %d "
						"min=%.2f max=%.2f "
						"pos=%.1f%%",
						qi, mmin, mmax,
						frac * 100);
			}

			/* Dump inst L2 norm spatial map */
			{
				const float *id =
					(const float *)inst->data;
				int ic = inst->dims[1];
				int ihw = final_h * final_w;
				FILE *fp = fopen("/tmp/inst_l2.pgm", "w");
				if (fp) {
					float l2min = 1e30f, l2max = -1e30f;
					float *l2 = (float *)sam3_arena_alloc(
						scratch, (size_t)ihw * sizeof(float));
					if (l2) {
						for (int p = 0; p < ihw; p++) {
							float sum = 0;
							for (int c = 0; c < ic; c++) {
								float v = id[c * ihw + p];
								sum += v * v;
							}
							l2[p] = sqrtf(sum);
							if (l2[p] < l2min) l2min = l2[p];
							if (l2[p] > l2max) l2max = l2[p];
						}
						sam3_log_debug("seg: inst L2 "
							"min=%.2f max=%.2f",
							l2min, l2max);
						fprintf(fp, "P2\n%d %d\n255\n",
							final_w, final_h);
						for (int p = 0; p < ihw; p++) {
							int v = (int)(255.0f *
								(l2[p] - l2min) /
								(l2max - l2min + 1e-8f));
							fprintf(fp, "%d ",
								v < 0 ? 0 : v > 255 ? 255 : v);
							if ((p + 1) % final_w == 0)
								fprintf(fp, "\n");
						}
						sam3_log_info("seg: wrote "
							"/tmp/inst_l2.pgm");
					}
					fclose(fp);
				}
			}
			}
		}

		/*
		 * Stage 4e: Objectness scorer on decoder queries.
		 *
		 * Produces per-query confidence [n_queries, 1]. Only
		 * computed when text_features are available, since the
		 * scorer graph references them (for the similarity dot
		 * product — the actual confidence comes from an MLP on
		 * queries alone, but the current build wires both).
		 */
		if (text_features) {
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			struct sam3_tensor *q_wrap2, *tf_wrap;
			q_wrap2 = gh_tensor_wrap(scratch,
				queries->dtype, queries->n_dims,
				queries->dims, queries->data);
			tf_wrap = gh_tensor_wrap(scratch,
				text_features->dtype,
				text_features->n_dims,
				text_features->dims,
				text_features->data);
			if (!q_wrap2 || !tf_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			scores = sam3_dot_scorer_build(&model->scorer,
				&g, q_wrap2, tf_wrap, scratch);
			if (!scores) {
				sam3_log_error("seg: scorer build failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			scores = gh_sigmoid(&g, scratch, scores);
			if (!scores) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("seg: scorer eval failed");
				goto fail;
			}

			scores = persist_tensor(persist, scores);
			if (!scores) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			{
				const float *sd =
					(const float *)scores->data;
				int sn = scores->dims[0];
				float smin = sd[0], smax = sd[0];
				double ssum = 0;
				for (int si = 0; si < sn; si++) {
					if (sd[si] < smin) smin = sd[si];
					if (sd[si] > smax) smax = sd[si];
					ssum += sd[si];
				}
				sam3_log_debug("seg: scores [%d] "
					"min=%.4f max=%.4f mean=%.4f",
					sn, smin, smax,
					(float)(ssum / sn));
			}
		}
	}

	/* Persist masks so they survive persist rollback */
	masks = persist_tensor(persist, masks);
	if (!masks) {
		err = SAM3_ENOMEM;
		goto fail;
	}

	if (out_scores)
		*out_scores = scores;

	/* Restore persist — inter-stage data no longer needed */
	persist->offset = persist_save;

	*out_masks = masks;
	return SAM3_OK;

fail:
	persist->offset = persist_save;
	return err;
}
