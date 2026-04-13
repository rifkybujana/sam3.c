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
#include "util/profile.h"

/* ── CPU helpers for decoder box refinement ──────────────────────── */

/*
 * cpu_layernorm_f32 - Apply LayerNorm on CPU.
 *
 * src [N, d] → dst [N, d], using weight w[d] and bias b[d].
 */
static void cpu_layernorm_f32(const float *src, const float *w,
			       const float *b, int N, int d, float *dst)
{
	const float eps = 1e-5f;
	for (int i = 0; i < N; i++) {
		const float *row = src + i * d;
		float *out = dst + i * d;
		float mean = 0, var = 0;
		for (int j = 0; j < d; j++)
			mean += row[j];
		mean /= (float)d;
		for (int j = 0; j < d; j++) {
			float diff = row[j] - mean;
			var += diff * diff;
		}
		var /= (float)d;
		float inv_std = 1.0f / sqrtf(var + eps);
		for (int j = 0; j < d; j++)
			out[j] = (row[j] - mean) * inv_std * w[j] + b[j];
	}
}

/*
 * cpu_linear_f32 - Compute dst = src × W^T + b on CPU.
 *
 * src [M, K], W [N, K] (row-major), b [N], dst [M, N].
 */
static void cpu_linear_f32(const float *src, const float *W,
			    const float *b, int M, int K, int N,
			    float *dst)
{
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float s = b ? b[j] : 0.0f;
			const float *a = src + i * K;
			const float *wj = W + j * K;
			for (int k = 0; k < K; k++)
				s += a[k] * wj[k];
			dst[i * N + j] = s;
		}
	}
}

/*
 * cpu_linear_relu_f32 - Linear + ReLU on CPU.
 */
static void cpu_linear_relu_f32(const float *src, const float *W,
				 const float *b, int M, int K, int N,
				 float *dst)
{
	cpu_linear_f32(src, W, b, M, K, N, dst);
	int n = M * N;
	for (int i = 0; i < n; i++)
		if (dst[i] < 0.0f)
			dst[i] = 0.0f;
}

/*
 * cpu_inverse_sigmoid - Compute logit from probability.
 *
 * Returns log(clamp(x, eps) / clamp(1-x, eps)).
 * Matches Python's inverse_sigmoid(x, eps=1e-3).
 */
static inline float cpu_inverse_sigmoid(float x)
{
	const float eps = 1e-3f;
	float x1 = x > eps ? x : eps;
	float x2 = (1.0f - x) > eps ? (1.0f - x) : eps;
	return logf(x1 / x2);
}

/*
 * cpu_box_refine - Apply box head MLP and update reference boxes.
 *
 * Applies output_ln to queries, runs box_head MLP, then updates
 * ref_boxes via: sigmoid(inverse_sigmoid(old_ref) + delta).
 * Matches Python's iterative box refinement with
 * use_normed_output_consistently=True.
 *
 * @q:         Query embeddings [nq, d] (raw, before output_ln)
 * @dec:       Decoder (for weights)
 * @ref_boxes: Reference boxes [nq, 4] — updated in place
 * @nq:        Number of queries
 * @d:         Model dimension
 * @tmp1:      Scratch buffer [nq * max(d, 4)]
 * @tmp2:      Scratch buffer [nq * d]
 */
static void cpu_box_refine(const float *q,
			    const struct sam3_decoder *dec,
			    float *ref_boxes, int nq, int d,
			    float *tmp1, float *tmp2)
{
	/* Step 1: LayerNorm(q) using output_ln weights */
	cpu_layernorm_f32(q,
		(const float *)dec->output_ln_w->data,
		(const float *)dec->output_ln_b->data,
		nq, d, tmp1);

	/* Step 2: box_head MLP (3 layers, ReLU on first two) */
	cpu_linear_relu_f32(tmp1,
		(const float *)dec->layers[0].box_fc1_w->data,
		(const float *)dec->layers[0].box_fc1_b->data,
		nq, d, d, tmp2);
	cpu_linear_relu_f32(tmp2,
		(const float *)dec->layers[0].box_fc2_w->data,
		(const float *)dec->layers[0].box_fc2_b->data,
		nq, d, d, tmp1);
	cpu_linear_f32(tmp1,
		(const float *)dec->layers[0].box_fc3_w->data,
		(const float *)dec->layers[0].box_fc3_b->data,
		nq, d, 4, tmp2); /* delta [nq, 4] in tmp2 */

	/* Step 3: ref_boxes = sigmoid(inverse_sigmoid(old) + delta) */
	for (int j = 0; j < nq * 4; j++) {
		float logit = cpu_inverse_sigmoid(ref_boxes[j]);
		ref_boxes[j] = 1.0f / (1.0f + expf(-(logit + tmp2[j])));
	}
}

#ifdef SAM3_DEBUG_DUMP
/* Debug: dump a tensor to a raw binary file for Python comparison */
static void dump_tensor(const char *path, const struct sam3_tensor *t)
{
	int n = 1;
	for (int i = 0; i < t->n_dims; i++)
		n *= t->dims[i];
	FILE *fp = fopen(path, "wb");
	if (!fp) return;
	fwrite(t->data, sizeof(float), (size_t)n, fp);
	fclose(fp);
	sam3_log_info("dump: wrote %s (%d floats)", path, n);
}
#else
static inline void dump_tensor(const char *path,
				const struct sam3_tensor *t)
{
	(void)path;
	(void)t;
}
#endif

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

	/* Dot-product scorer config (weights loaded separately) */
	model->scorer.d_model = 256;
	model->scorer.d_proj = 256;
	model->scorer.d_ffn = 2048;

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
					struct sam3_arena *persist,
					struct sam3_profiler *profiler)
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
						scratch, persist,
						profiler);
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
	 * NHWC input tensor to be corrupted (MLX reuses intermediate
	 * buffers). Instead, we wrap the ViT output as NHWC and then
	 * evaluate each stage as its own graph.
	 *
	 * All scratch allocations accumulate (no arena resets) so that
	 * stage outputs survive until we copy them to persist at the end.
	 */
	SAM3_PROF_BEGIN(profiler, "neck");
	{
		/*
		 * Step 1: Wrap the ViT output as an NHWC tensor.
		 *
		 * ViT output is [HW, C] row-major, which is already
		 * the NHWC byte layout for [1, gs, gs, C]. After the
		 * Task 8 neck migration, the neck conv weights are
		 * stored OHWI and the NHWC conv helpers consume NHWC
		 * input directly, so no transpose is needed here.
		 *
		 * All downstream consumers (seg_head, mask decoder,
		 * geom encoder / fusion) are NHWC after Task 10, so
		 * stage outputs are snapshotted only in NHWC. The
		 * geometry encoder / fusion path gets the 1x NHWC
		 * feature reshaped to [H*W, d_model] as a pure view
		 * (NHWC byte order already matches row-major [HW, d]).
		 */
		sam3_arena_reset(scratch);

		int C = model->backbone.neck.backbone_dim;
		int gs = model->backbone.neck.grid_size;
		int HW = gs * gs;
		int nhwc_dims[] = {1, gs, gs, C};
		size_t nhwc_bytes = (size_t)C * HW * sizeof(float);

		sam3_log_debug("encode: neck NHWC wrap [%d,%d,%d,%d]",
			       1, gs, gs, C);

		/* Dump raw ViT NHWC for debugging */
		{
			int nhwc_sd2[] = {1, gs, gs, C};
			struct sam3_tensor tmp;
			tmp.dtype = SAM3_DTYPE_F32;
			tmp.n_dims = 4;
			for (int k = 0; k < 4; k++) tmp.dims[k] = nhwc_sd2[k];
			tmp.data = vit_out->data;
			tmp.nbytes = nhwc_bytes;
			dump_tensor("/tmp/dbg_vit_nhwc.bin", &tmp);
		}

		/*
		 * Step 2: Evaluate each stage independently in NHWC.
		 *
		 * After Task 10 every downstream consumer is NHWC,
		 * so we snapshot only the NHWC payload.
		 */
		struct sam3_tensor *stage_out_nhwc[4] = {0};

		for (int si = 0; si < model->backbone.neck.n_scales; si++) {
			/*
			 * Copy the ViT NHWC payload per stage. MLX
			 * may invalidate input buffers after a
			 * graph_eval, so each stage gets its own
			 * private copy of the shared input.
			 */
			void *stage_nhwc = sam3_arena_alloc(scratch,
							    nhwc_bytes);
			if (!stage_nhwc) return SAM3_ENOMEM;
			memcpy(stage_nhwc, vit_out->data, nhwc_bytes);

			struct sam3_tensor *nhwc_in;
			nhwc_in = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
						  4, nhwc_dims, stage_nhwc);
			if (!nhwc_in) return SAM3_ENOMEM;

			struct sam3_tensor *x = nhwc_in;

			if (model->backbone.neck.stages[si].has_maxpool) {
				struct sam3_graph g_stage;
				sam3_graph_init(&g_stage);
				x = gh_maxpool2d(&g_stage, scratch,
						      x, 2, 2);
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

				/* Snapshot layer output + smoothness (NHWC) */
				{
					const float *ld =
						(const float *)x->data;
					int lH = x->dims[1];
					int lW = x->dims[2];
					int lC = x->dims[3];
					float lmin = ld[0], lmax = ld[0];
					double la = 0, ldw = 0, ldh = 0;
					long ca = 0, cw2 = 0, ch3 = 0;
					for (int h = 0; h < lH; h++) {
						for (int w = 0; w < lW; w++) {
							const float *pv =
							  ld + ((long)h*lW+w)*lC;
							for (int c = 0; c < lC; c++) {
								float v = pv[c];
								if (v < lmin) lmin = v;
								if (v > lmax) lmax = v;
								la += fabs(v); ca++;
							}
							if (w+1<lW) {
								const float *pn =
								  ld + ((long)h*lW+w+1)*lC;
								for (int c = 0; c < lC; c++) {
									ldw += fabs(pv[c]-pn[c]);
									cw2++;
								}
							}
							if (h+1<lH) {
								const float *pn =
								  ld + ((long)(h+1)*lW+w)*lC;
								for (int c = 0; c < lC; c++) {
									ldh += fabs(pv[c]-pn[c]);
									ch3++;
								}
							}
						}
					}
					double lm = la / ca;
					double rw = (ldw/cw2)/(lm+1e-8);
					double rh = (ldh/ch3)/(lm+1e-8);
					sam3_log_debug("encode: neck[%d].layer[%d]"
						" nhwc [%d,%d,%d,%d] min=%.4f"
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
			 * Snapshot stage output in NHWC. MLX recycles
			 * internal buffers between graph_eval calls,
			 * so we materialize a private copy in scratch
			 * before the next stage builds its graph.
			 */
			{
				int sH = x->dims[1];
				int sW = x->dims[2];
				int sC = x->dims[3];
				int nhwc_dims_out[] = {1, sH, sW, sC};

				stage_out_nhwc[si] = gh_alloc_tensor(
					scratch, x->dtype, 4, nhwc_dims_out);
				if (!stage_out_nhwc[si]) return SAM3_ENOMEM;
				memcpy(stage_out_nhwc[si]->data, x->data,
				       stage_out_nhwc[si]->nbytes);
			}

			{
				const float *sd =
					(const float *)stage_out_nhwc[si]->data;
				int sn = 1;
				for (int sj = 0; sj < stage_out_nhwc[si]->n_dims; sj++)
					sn *= stage_out_nhwc[si]->dims[sj];
				float smin = sd[0], smax = sd[0];
				for (int sj = 0; sj < sn; sj++) {
					if (sd[sj] < smin) smin = sd[sj];
					if (sd[sj] > smax) smax = sd[sj];
				}
				sam3_log_debug("encode: feat[%d] nhwc "
					"[%d,%d,%d,%d] "
					"min=%.4f max=%.4f",
					si,
					stage_out_nhwc[si]->dims[0],
					stage_out_nhwc[si]->dims[1],
					stage_out_nhwc[si]->dims[2],
					stage_out_nhwc[si]->dims[3],
					smin, smax);
			}
		}

		/*
		 * Step 3: Copy outputs to persist arena. Only NHWC
		 * snapshots are retained after Task 10 — every
		 * downstream consumer (geom encoder, fusion, decoder,
		 * seg_head, mask decoder) now reads NHWC directly.
		 */
		struct sam3_tensor *pfn[4];
		for (int si = 0; si < 4; si++) {
			pfn[si] = gh_alloc_tensor(persist,
				stage_out_nhwc[si]->dtype,
				stage_out_nhwc[si]->n_dims,
				stage_out_nhwc[si]->dims);
			if (!pfn[si]) return SAM3_ENOMEM;
			memcpy(pfn[si]->data, stage_out_nhwc[si]->data,
			       stage_out_nhwc[si]->nbytes);
		}

		model->cached_feat_4x_nhwc = pfn[0];
		model->cached_feat_s0_nhwc = pfn[1];
		model->cached_feat_s1_nhwc = pfn[2];
		model->cached_image_features = pfn[3];

		dump_tensor("/tmp/dbg_neck_4x.bin", pfn[0]);
		dump_tensor("/tmp/dbg_neck_2x.bin", pfn[1]);
		dump_tensor("/tmp/dbg_neck_1x.bin", pfn[2]);
		dump_tensor("/tmp/dbg_neck_05x.bin", pfn[3]);

		sam3_log_debug("encode: cached features (NHWC): "
			       "main [%d,%d,%d,%d] "
			       "s0 [%d,%d,%d,%d] s1 [%d,%d,%d,%d] "
			       "4x [%d,%d,%d,%d]",
			       pfn[3]->dims[0], pfn[3]->dims[1],
			       pfn[3]->dims[2], pfn[3]->dims[3],
			       pfn[1]->dims[0], pfn[1]->dims[1],
			       pfn[1]->dims[2], pfn[1]->dims[3],
			       pfn[2]->dims[0], pfn[2]->dims[1],
			       pfn[2]->dims[2], pfn[2]->dims[3],
			       pfn[0]->dims[0], pfn[0]->dims[1],
			       pfn[0]->dims[2], pfn[0]->dims[3]);
	}

	SAM3_PROF_END(profiler, "neck");

	model->image_encoded = 1;
	return SAM3_OK;
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
	struct sam3_backend *cpu_be,
	struct sam3_tensor *prompt_tokens,
	struct sam3_tensor *text_features,
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_tensor **out_masks,
	struct sam3_tensor **out_scores,
	struct sam3_profiler *profiler)
{
	/*
	 * dec_be: backend for decoder + scorer stages.
	 * Using CPU avoids GPU-specific float32 reduction order
	 * divergence that compounds through 6 decoder layers,
	 * causing scorer logits to deviate 10-15x from Python.
	 */
	struct sam3_backend *dec_be = cpu_be ? cpu_be : be;

	/* Helper: reset dec_be arena to reclaim memory between evals.
	 * Decoder cross-attention allocates ~33MB per layer, so without
	 * reset the CPU arena would overflow after a few layers. */
#define DEC_BE_ARENA_RESET() do { \
	if (dec_be->ops->arena_reset) \
		dec_be->ops->arena_reset(dec_be); \
} while (0)

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
	 * Reshape 1× backbone features from NHWC [1, H, W, d_model]
	 * to [H*W, d_model]. NHWC row-major byte order already
	 * matches the downstream [n_pixels, d_model] layout, so we
	 * just copy the payload into a fresh 2D tensor in persist.
	 * The Python model feeds the 72×72 (1×) features to the
	 * encoder, not the 36×36 (0.5×) features.
	 */
	{
		struct sam3_tensor *s1 = model->cached_feat_s1_nhwc;
		int HW = s1->dims[1] * s1->dims[2];
		int C = s1->dims[3];
		int dims[2] = {HW, C};
		img_2d = gh_alloc_tensor(persist, s1->dtype, 2, dims);
		if (!img_2d)
			return SAM3_ENOMEM;
		memcpy(img_2d->data, s1->data, s1->nbytes);
	}

	/*
	 * Stage 1: Geometry encoder — full transformer encoder attending
	 * prompt tokens to cached image features. Output is [N+1, d_model].
	 *
	 * Python order:
	 *   1. Append CLS token after prompt embeddings
	 *   2. Pre-encoder: final_proj (Linear) + norm (LayerNorm)
	 *   3. 3 encoder layers: self-attn → cross-attn → FFN
	 *   4. Post-encoder: encode_norm (LayerNorm)
	 *
	 * Cross-attention uses pos_enc_at_cross_attn_keys=True:
	 *   key = image_features + pos_enc,  value = image_features
	 */
	SAM3_PROF_BEGIN(profiler, "geometry_encode");
	if (prompt_tokens) {
		struct sam3_geometry_encoder *enc = &model->geom_enc;
		int d = enc->d_model;
		int n_heads = enc->n_heads;
		int head_dim = d / n_heads;

		/* Get position encoding [H, W, d] → flatten to [H*W, d] */
		struct sam3_tensor *pe_t;
		pe_t = sam3_pos_encoding_get(&model->backbone.pos_enc);
		int pe_hw = pe_t->dims[0] * pe_t->dims[1];

		/*
		 * Pre-compute img_with_pos = img_2d + pos for cross-attn
		 * keys. Persisted since it's reused across all layers.
		 */
		struct sam3_tensor *img_with_pos;
		img_with_pos = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						2, img_2d->dims);
		if (!img_with_pos) { err = SAM3_ENOMEM; goto fail; }
		{
			const float *id = (const float *)img_2d->data;
			const float *pd = (const float *)pe_t->data;
			float *od = (float *)img_with_pos->data;
			int n = pe_hw * d;
			for (int j = 0; j < n; j++)
				od[j] = id[j] + pd[j];
		}

		/* Step 1: Append CLS token after prompt tokens.
		 * Python: concat_padded_sequences(point_embeds, ..., cls, ...)
		 * Result order: [prompt_0, ..., prompt_n, CLS]
		 */
		struct sam3_tensor *x;
		{
			int xdims[2] = {prompt_tokens->dims[0] +
					enc->cls_token->dims[0],
					d};
			x = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
					     2, xdims);
			if (!x) { err = SAM3_ENOMEM; goto fail; }
			size_t pt_bytes = (size_t)prompt_tokens->dims[0]
				* (size_t)d * sizeof(float);
			size_t cls_bytes = (size_t)enc->cls_token->dims[0]
				* (size_t)d * sizeof(float);
			memcpy(x->data, prompt_tokens->data, pt_bytes);
			memcpy((char *)x->data + pt_bytes,
			       enc->cls_token->data, cls_bytes);
		}

		dump_tensor("/tmp/dbg_geom_after_concat.bin", x);

		/*
		 * Step 2: Pre-encoder projection + LayerNorm.
		 * Python: final_embeds = self.norm(self.final_proj(final_embeds))
		 *
		 * Manual CPU matmul to avoid Metal backend issues
		 * with small [2, 256] tensors.
		 */
		{
			int nrows = x->dims[0];
			struct sam3_tensor *proj;
			proj = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						2, x->dims);
			if (!proj) { err = SAM3_ENOMEM; goto fail; }

			const float *xd = (const float *)x->data;
			const float *wd = (const float *)enc->post_proj_w->data;
			const float *bd = (const float *)enc->post_proj_b->data;
			float *od = (float *)proj->data;

			/* out[r,c] = sum(x[r,k] * W[c,k]) + b[c] */
			for (int r = 0; r < nrows; r++) {
				for (int c = 0; c < d; c++) {
					float sum = bd[c];
					for (int k = 0; k < d; k++)
						sum += xd[r * d + k] * wd[c * d + k];
					od[r * d + c] = sum;
				}
			}
			x = proj;
		}
		dump_tensor("/tmp/dbg_geom_after_proj.bin", x);
		{
			/*
			 * LayerNorm on CPU — Metal can't read CPU-arena
			 * tensors as graph inputs.
			 */
			int nrows = x->dims[0];
			const float *gw = (const float *)enc->norm_w->data;
			const float *gb = (const float *)enc->norm_b->data;
			float *xd = (float *)x->data;
			const float eps = 1e-5f;

			for (int r = 0; r < nrows; r++) {
				float *row = xd + r * d;
				double sum = 0.0, sum2 = 0.0;
				for (int c = 0; c < d; c++) {
					sum += (double)row[c];
					sum2 += (double)row[c] * (double)row[c];
				}
				double mean = sum / d;
				double var = sum2 / d - mean * mean;
				double inv = 1.0 / sqrt(var + eps);
				for (int c = 0; c < d; c++)
					row[c] = (float)(((double)row[c] - mean) *
						  inv) * gw[c] + gb[c];
			}
		}

		dump_tensor("/tmp/dbg_geom_pre_enc.bin", x);

		/*
		 * Step 3: Transformer encoder layers (CPU path).
		 *
		 * The geometry encoder has only 2 tokens — Metal/MLX
		 * SDPA kernels degenerate on such tiny inputs, so we
		 * run the full 3-layer encoder on CPU.
		 */
#define GEOM_NROWS_MAX 8	/* CLS + up to 7 prompts */
		{
		int nq = x->dims[0];    /* 2 typically */
		int nkv = img_2d->dims[0]; /* 5184 */

		/* Scratch buffers on the arena (freed after loop) */
		size_t sa_save = persist->offset;
		float *buf_norm = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_qkv = (float *)sam3_arena_alloc(
			persist, (size_t)nq * 3 * (size_t)d * sizeof(float));
		float *buf_attn = (float *)sam3_arena_alloc(
			persist, (size_t)n_heads * (size_t)nq * (size_t)nq * sizeof(float));
		float *buf_sa = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_q = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_k = (float *)sam3_arena_alloc(
			persist, (size_t)nkv * (size_t)d * sizeof(float));
		float *buf_v = (float *)sam3_arena_alloc(
			persist, (size_t)nkv * (size_t)d * sizeof(float));
		float *buf_ca = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_ff1 = (float *)sam3_arena_alloc(
			persist, (size_t)nq * 2048 * sizeof(float));
		float *buf_ff2 = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		if (!buf_norm || !buf_qkv || !buf_attn || !buf_sa ||
		    !buf_q || !buf_k || !buf_v || !buf_ca ||
		    !buf_ff1 || !buf_ff2) {
			err = SAM3_ENOMEM; goto fail;
		}

		float *xd = (float *)x->data;
		const float eps = 1e-5f;
		const float scale = 1.0f / sqrtf((float)head_dim);

		for (int li = 0; li < enc->n_layers; li++) {
			/* --- 3a: Self-attention --- */
			/* LayerNorm: buf_norm = LN(xd) */
			{
				const float *gw = (const float *)enc->layers[li].norm1_w->data;
				const float *gb = (const float *)enc->layers[li].norm1_b->data;
				for (int r = 0; r < nq; r++) {
					const float *row = xd + r * d;
					double s = 0, s2 = 0;
					for (int c = 0; c < d; c++) {
						s += (double)row[c];
						s2 += (double)row[c] * (double)row[c];
					}
					double m = s / d, v2 = s2 / d - m * m;
					double inv = 1.0 / sqrt(v2 + eps);
					for (int c = 0; c < d; c++)
						buf_norm[r * d + c] = (float)(((double)row[c] - m) * inv) * gw[c] + gb[c];
				}
			}
			/* QKV projection: buf_qkv = buf_norm @ sa_qkv_w^T + sa_qkv_b */
			{
				const float *w = (const float *)enc->layers[li].sa_qkv_w->data;
				const float *b = (const float *)enc->layers[li].sa_qkv_b->data;
				int d3 = 3 * d;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d3; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_norm[r * d + k] * w[c * d + k];
						buf_qkv[r * d3 + c] = sum;
					}
				}
			}
			/* Per-head SDPA: softmax(Q@K^T / sqrt(dh)) @ V */
			for (int h = 0; h < n_heads; h++) {
				int hs = h * head_dim;
				/* Score matrix [nq, nq] */
				for (int qi = 0; qi < nq; qi++) {
					float maxs = -1e30f;
					for (int ki = 0; ki < nq; ki++) {
						float dot = 0;
						for (int c = 0; c < head_dim; c++)
							dot += buf_qkv[qi * 3 * d + hs + c] *
							       buf_qkv[ki * 3 * d + d + hs + c];
						dot *= scale;
						buf_attn[h * nq * nq + qi * nq + ki] = dot;
						if (dot > maxs) maxs = dot;
					}
					/* Softmax */
					float sexp = 0;
					for (int ki = 0; ki < nq; ki++) {
						float e = expf(buf_attn[h * nq * nq + qi * nq + ki] - maxs);
						buf_attn[h * nq * nq + qi * nq + ki] = e;
						sexp += e;
					}
					float inv_s = 1.0f / sexp;
					for (int ki = 0; ki < nq; ki++)
						buf_attn[h * nq * nq + qi * nq + ki] *= inv_s;
				}
				/* Weighted sum of V */
				for (int qi = 0; qi < nq; qi++) {
					for (int c = 0; c < head_dim; c++) {
						float sum = 0;
						for (int ki = 0; ki < nq; ki++)
							sum += buf_attn[h * nq * nq + qi * nq + ki] *
							       buf_qkv[ki * 3 * d + 2 * d + hs + c];
						buf_sa[qi * d + hs + c] = sum;
					}
				}
			}
			/* Output projection + residual */
			{
				const float *w = (const float *)enc->layers[li].sa_out_w->data;
				const float *b = (const float *)enc->layers[li].sa_out_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_sa[r * d + k] * w[c * d + k];
						xd[r * d + c] += sum; /* residual */
					}
				}
			}
			if (li == 0)
				dump_tensor("/tmp/dbg_geom_l0_sa.bin", x);

			/* --- 3b: Cross-attention --- */
			/* LayerNorm: buf_norm = LN(xd) using ca_ln */
			{
				const float *gw = (const float *)enc->layers[li].ca_ln_w->data;
				const float *gb = (const float *)enc->layers[li].ca_ln_b->data;
				for (int r = 0; r < nq; r++) {
					const float *row = xd + r * d;
					double s = 0, s2 = 0;
					for (int c = 0; c < d; c++) {
						s += (double)row[c];
						s2 += (double)row[c] * (double)row[c];
					}
					double m = s / d, v2 = s2 / d - m * m;
					double inv = 1.0 / sqrt(v2 + eps);
					for (int c = 0; c < d; c++)
						buf_norm[r * d + c] = (float)(((double)row[c] - m) * inv) * gw[c] + gb[c];
				}
			}
			/* Q = buf_norm @ ca_q_w^T + ca_q_b */
			{
				const float *w = (const float *)enc->layers[li].ca_q_w->data;
				const float *b = (const float *)enc->layers[li].ca_q_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_norm[r * d + k] * w[c * d + k];
						buf_q[r * d + c] = sum;
					}
				}
			}
			/* K = img_with_pos @ k_w^T + k_b (first half of ca_kv) */
			/* V = img_2d @ v_w^T + v_b (second half of ca_kv) */
			{
				const float *kv_w = (const float *)enc->layers[li].ca_kv_w->data;
				const float *kv_b = (const float *)enc->layers[li].ca_kv_b->data;
				/* kv_w is [2d, d], k_w = [0:d, :], v_w = [d:2d, :] */
				const float *kw = kv_w;
				const float *vw = kv_w + d * d;
				const float *kb = kv_b;
				const float *vb = kv_b + d;
				const float *img_pos_d = (const float *)img_with_pos->data;
				const float *img_d = (const float *)img_2d->data;

				for (int r = 0; r < nkv; r++) {
					for (int c = 0; c < d; c++) {
						float ks = kb[c], vs = vb[c];
						for (int k = 0; k < d; k++) {
							ks += img_pos_d[r * d + k] * kw[c * d + k];
							vs += img_d[r * d + k] * vw[c * d + k];
						}
						buf_k[r * d + c] = ks;
						buf_v[r * d + c] = vs;
					}
				}
			}
			/* Per-head cross-SDPA: softmax(Q@K^T / sqrt(dh)) @ V */
			for (int h = 0; h < n_heads; h++) {
				int hs = h * head_dim;
				for (int qi = 0; qi < nq; qi++) {
					/* For each query, compute attention over all KV */
					float maxs = -1e30f;
					/* We need [nq, nkv] scores — too big for stack.
					 * Process one query row at a time streaming. */
					/* Pass 1: find max score for numerical stability */
					for (int ki = 0; ki < nkv; ki++) {
						float dot = 0;
						for (int c = 0; c < head_dim; c++)
							dot += buf_q[qi * d + hs + c] *
							       buf_k[ki * d + hs + c];
						dot *= scale;
						if (dot > maxs) maxs = dot;
					}
					/* Pass 2: compute exp sums and weighted V */
					float sexp = 0;
					float vsum[64]; /* head_dim max 64 */
					for (int c = 0; c < head_dim; c++)
						vsum[c] = 0;
					for (int ki = 0; ki < nkv; ki++) {
						float dot = 0;
						for (int c = 0; c < head_dim; c++)
							dot += buf_q[qi * d + hs + c] *
							       buf_k[ki * d + hs + c];
						float e = expf(dot * scale - maxs);
						sexp += e;
						for (int c = 0; c < head_dim; c++)
							vsum[c] += e * buf_v[ki * d + hs + c];
					}
					float inv_s = 1.0f / sexp;
					for (int c = 0; c < head_dim; c++)
						buf_ca[qi * d + hs + c] = vsum[c] * inv_s;
				}
			}
			/* Output projection + residual */
			{
				const float *w = (const float *)enc->layers[li].ca_out_w->data;
				const float *b = (const float *)enc->layers[li].ca_out_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_ca[r * d + k] * w[c * d + k];
						xd[r * d + c] += sum;
					}
				}
			}
			if (li == 0)
				dump_tensor("/tmp/dbg_geom_l0_ca.bin", x);

			/* --- 3c: FFN: norm3 → fc1 → relu → fc2 → residual --- */
			/* LayerNorm */
			{
				const float *gw = (const float *)enc->layers[li].norm3_w->data;
				const float *gb = (const float *)enc->layers[li].norm3_b->data;
				for (int r = 0; r < nq; r++) {
					const float *row = xd + r * d;
					double s = 0, s2 = 0;
					for (int c = 0; c < d; c++) {
						s += (double)row[c];
						s2 += (double)row[c] * (double)row[c];
					}
					double m = s / d, v2 = s2 / d - m * m;
					double inv = 1.0 / sqrt(v2 + eps);
					for (int c = 0; c < d; c++)
						buf_norm[r * d + c] = (float)(((double)row[c] - m) * inv) * gw[c] + gb[c];
				}
			}
			/* fc1: [nq, d] → [nq, 2048] + relu */
			{
				const float *w = (const float *)enc->layers[li].ffn_fc1_w->data;
				const float *b = (const float *)enc->layers[li].ffn_fc1_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < 2048; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_norm[r * d + k] * w[c * d + k];
						buf_ff1[r * 2048 + c] = sum > 0 ? sum : 0; /* relu */
					}
				}
			}
			/* fc2: [nq, 2048] → [nq, d] + residual */
			{
				const float *w = (const float *)enc->layers[li].ffn_fc2_w->data;
				const float *b = (const float *)enc->layers[li].ffn_fc2_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < 2048; k++)
							sum += buf_ff1[r * 2048 + k] * w[c * 2048 + k];
						xd[r * d + c] += sum;
					}
				}
			}

			sam3_log_debug("geom: layer %d done", li);
			{
				char dpath[64];
				snprintf(dpath, sizeof(dpath),
					 "/tmp/dbg_geom_layer_%02d.bin", li);
				dump_tensor(dpath, x);
			}
		}
		/* Reclaim scratch buffers */
		persist->offset = sa_save;
		/* Re-allocate x as a proper persist tensor */
		{
			struct sam3_tensor *xp;
			xp = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
					      2, x->dims);
			if (!xp) { err = SAM3_ENOMEM; goto fail; }
			memcpy(xp->data, x->data, x->nbytes);
			x = xp;
		}
		}
#undef GEOM_NROWS_MAX

		/* Step 4: Post-encoder LayerNorm (encode_norm) — CPU */
		{
			int nrows = x->dims[0];
			const float *gw = (const float *)enc->encode_norm_w->data;
			const float *gb = (const float *)enc->encode_norm_b->data;
			float *xd = (float *)x->data;
			const float eps = 1e-5f;

			for (int r = 0; r < nrows; r++) {
				float *row = xd + r * d;
				double sum = 0.0, sum2 = 0.0;
				for (int c = 0; c < d; c++) {
					sum += (double)row[c];
					sum2 += (double)row[c] * (double)row[c];
				}
				double mean = sum / d;
				double var = sum2 / d - mean * mean;
				double inv = 1.0 / sqrt(var + eps);
				for (int c = 0; c < d; c++)
					row[c] = (float)(((double)row[c] - mean) *
						  inv) * gw[c] + gb[c];
			}
			geom_out = x;
		}

		sam3_log_debug("segment: geom encoder done, "
			       "persist %zu/%zu",
			       persist->offset, persist->size);
	} else if (text_features) {
		/*
		 * Text-only path: Python always runs the geometry encoder
		 * even without geometric prompts. It uses _get_dummy_prompt()
		 * which produces empty point/box sequences, so after
		 * concat_padded_sequences only the CLS token (1 row) remains.
		 * The CLS token then goes through:
		 *   final_proj → LayerNorm → 3 encoder layers → encode_norm
		 * producing a [1, d_model] output that is concatenated with
		 * text features as context for encoder fusion.
		 */
		struct sam3_geometry_encoder *enc = &model->geom_enc;
		int d = enc->d_model;
		int n_heads = enc->n_heads;
		int head_dim = d / n_heads;

		/* Position encoding for cross-attention keys */
		struct sam3_tensor *pe_t;
		pe_t = sam3_pos_encoding_get(&model->backbone.pos_enc);
		int pe_hw = pe_t->dims[0] * pe_t->dims[1];

		/* img_with_pos = img_2d + pos for cross-attn keys */
		struct sam3_tensor *img_with_pos;
		img_with_pos = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						2, img_2d->dims);
		if (!img_with_pos) { err = SAM3_ENOMEM; goto fail; }
		{
			const float *id = (const float *)img_2d->data;
			const float *pd = (const float *)pe_t->data;
			float *od = (float *)img_with_pos->data;
			int n = pe_hw * d;
			for (int j = 0; j < n; j++)
				od[j] = id[j] + pd[j];
		}

		/* x = CLS token only [1, d_model] */
		struct sam3_tensor *x;
		{
			int xdims[2] = {enc->cls_token->dims[0], d};
			x = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
					     2, xdims);
			if (!x) { err = SAM3_ENOMEM; goto fail; }
			memcpy(x->data, enc->cls_token->data,
			       (size_t)enc->cls_token->dims[0]
			       * (size_t)d * sizeof(float));
		}

		/* Pre-encoder: final_proj (Linear) */
		{
			int nrows = x->dims[0];
			struct sam3_tensor *proj;
			proj = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						2, x->dims);
			if (!proj) { err = SAM3_ENOMEM; goto fail; }

			const float *xd = (const float *)x->data;
			const float *wd = (const float *)enc->post_proj_w->data;
			const float *bd = (const float *)enc->post_proj_b->data;
			float *od = (float *)proj->data;

			for (int r = 0; r < nrows; r++) {
				for (int c = 0; c < d; c++) {
					float sum = bd[c];
					for (int k = 0; k < d; k++)
						sum += xd[r * d + k] * wd[c * d + k];
					od[r * d + c] = sum;
				}
			}
			x = proj;
		}

		/* Pre-encoder: LayerNorm */
		{
			int nrows = x->dims[0];
			const float *gw = (const float *)enc->norm_w->data;
			const float *gb = (const float *)enc->norm_b->data;
			float *xd = (float *)x->data;
			const float eps = 1e-5f;

			for (int r = 0; r < nrows; r++) {
				float *row = xd + r * d;
				double sum = 0.0, sum2 = 0.0;
				for (int c = 0; c < d; c++) {
					sum += (double)row[c];
					sum2 += (double)row[c] * (double)row[c];
				}
				double mean = sum / d;
				double var = sum2 / d - mean * mean;
				double inv = 1.0 / sqrt(var + eps);
				for (int c = 0; c < d; c++)
					row[c] = (float)(((double)row[c] - mean) *
						  inv) * gw[c] + gb[c];
			}
		}

		/* 3-layer transformer encoder (CLS-only, nq=1) */
		{
		int nq = x->dims[0];    /* 1 (CLS only) */
		int nkv = img_2d->dims[0]; /* 5184 */

		size_t sa_save = persist->offset;
		float *buf_norm = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_qkv = (float *)sam3_arena_alloc(
			persist, (size_t)nq * 3 * (size_t)d * sizeof(float));
		float *buf_attn = (float *)sam3_arena_alloc(
			persist, (size_t)n_heads * (size_t)nq * (size_t)nq * sizeof(float));
		float *buf_sa = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_q = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_k = (float *)sam3_arena_alloc(
			persist, (size_t)nkv * (size_t)d * sizeof(float));
		float *buf_v = (float *)sam3_arena_alloc(
			persist, (size_t)nkv * (size_t)d * sizeof(float));
		float *buf_ca = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		float *buf_ff1 = (float *)sam3_arena_alloc(
			persist, (size_t)nq * 2048 * sizeof(float));
		float *buf_ff2 = (float *)sam3_arena_alloc(
			persist, (size_t)nq * (size_t)d * sizeof(float));
		if (!buf_norm || !buf_qkv || !buf_attn || !buf_sa ||
		    !buf_q || !buf_k || !buf_v || !buf_ca ||
		    !buf_ff1 || !buf_ff2) {
			err = SAM3_ENOMEM; goto fail;
		}

		float *xd = (float *)x->data;
		const float eps = 1e-5f;
		const float scale = 1.0f / sqrtf((float)head_dim);

		for (int li = 0; li < enc->n_layers; li++) {
			/* Self-attention: LN → QKV → SDPA → out_proj + residual */
			{
				const float *gw = (const float *)enc->layers[li].norm1_w->data;
				const float *gb = (const float *)enc->layers[li].norm1_b->data;
				for (int r = 0; r < nq; r++) {
					const float *row = xd + r * d;
					double s = 0, s2 = 0;
					for (int c = 0; c < d; c++) {
						s += (double)row[c];
						s2 += (double)row[c] * (double)row[c];
					}
					double m = s / d, v2 = s2 / d - m * m;
					double inv2 = 1.0 / sqrt(v2 + eps);
					for (int c = 0; c < d; c++)
						buf_norm[r * d + c] = (float)(((double)row[c] - m) * inv2) * gw[c] + gb[c];
				}
			}
			{
				const float *w = (const float *)enc->layers[li].sa_qkv_w->data;
				const float *b = (const float *)enc->layers[li].sa_qkv_b->data;
				int d3 = 3 * d;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d3; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_norm[r * d + k] * w[c * d + k];
						buf_qkv[r * d3 + c] = sum;
					}
				}
			}
			for (int h = 0; h < n_heads; h++) {
				int hs = h * head_dim;
				for (int qi = 0; qi < nq; qi++) {
					float maxs = -1e30f;
					for (int ki = 0; ki < nq; ki++) {
						float dot = 0;
						for (int c = 0; c < head_dim; c++)
							dot += buf_qkv[qi * 3 * d + hs + c] *
							       buf_qkv[ki * 3 * d + d + hs + c];
						dot *= scale;
						buf_attn[h * nq * nq + qi * nq + ki] = dot;
						if (dot > maxs) maxs = dot;
					}
					float sexp = 0;
					for (int ki = 0; ki < nq; ki++) {
						float e = expf(buf_attn[h * nq * nq + qi * nq + ki] - maxs);
						buf_attn[h * nq * nq + qi * nq + ki] = e;
						sexp += e;
					}
					float inv_s = 1.0f / sexp;
					for (int ki = 0; ki < nq; ki++)
						buf_attn[h * nq * nq + qi * nq + ki] *= inv_s;
				}
				for (int qi = 0; qi < nq; qi++) {
					for (int c = 0; c < head_dim; c++) {
						float sum = 0;
						for (int ki = 0; ki < nq; ki++)
							sum += buf_attn[h * nq * nq + qi * nq + ki] *
							       buf_qkv[ki * 3 * d + 2 * d + hs + c];
						buf_sa[qi * d + hs + c] = sum;
					}
				}
			}
			{
				const float *w = (const float *)enc->layers[li].sa_out_w->data;
				const float *b = (const float *)enc->layers[li].sa_out_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_sa[r * d + k] * w[c * d + k];
						xd[r * d + c] += sum;
					}
				}
			}

			/* Cross-attention: LN → Q/KV → SDPA → out_proj + residual */
			{
				const float *gw = (const float *)enc->layers[li].ca_ln_w->data;
				const float *gb = (const float *)enc->layers[li].ca_ln_b->data;
				for (int r = 0; r < nq; r++) {
					const float *row = xd + r * d;
					double s = 0, s2 = 0;
					for (int c = 0; c < d; c++) {
						s += (double)row[c];
						s2 += (double)row[c] * (double)row[c];
					}
					double m = s / d, v2 = s2 / d - m * m;
					double inv2 = 1.0 / sqrt(v2 + eps);
					for (int c = 0; c < d; c++)
						buf_norm[r * d + c] = (float)(((double)row[c] - m) * inv2) * gw[c] + gb[c];
				}
			}
			{
				const float *w = (const float *)enc->layers[li].ca_q_w->data;
				const float *b = (const float *)enc->layers[li].ca_q_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_norm[r * d + k] * w[c * d + k];
						buf_q[r * d + c] = sum;
					}
				}
			}
			{
				const float *kv_w = (const float *)enc->layers[li].ca_kv_w->data;
				const float *kv_b = (const float *)enc->layers[li].ca_kv_b->data;
				const float *kw = kv_w;
				const float *vw = kv_w + d * d;
				const float *kb = kv_b;
				const float *vb = kv_b + d;
				const float *img_pos_d = (const float *)img_with_pos->data;
				const float *img_d = (const float *)img_2d->data;

				for (int r = 0; r < nkv; r++) {
					for (int c = 0; c < d; c++) {
						float ks = kb[c], vs = vb[c];
						for (int k = 0; k < d; k++) {
							ks += img_pos_d[r * d + k] * kw[c * d + k];
							vs += img_d[r * d + k] * vw[c * d + k];
						}
						buf_k[r * d + c] = ks;
						buf_v[r * d + c] = vs;
					}
				}
			}
			for (int h = 0; h < n_heads; h++) {
				int hs = h * head_dim;
				for (int qi = 0; qi < nq; qi++) {
					float maxs = -1e30f;
					for (int ki = 0; ki < nkv; ki++) {
						float dot = 0;
						for (int c = 0; c < head_dim; c++)
							dot += buf_q[qi * d + hs + c] *
							       buf_k[ki * d + hs + c];
						dot *= scale;
						if (dot > maxs) maxs = dot;
					}
					float sexp = 0;
					float vsum[64];
					for (int c = 0; c < head_dim; c++)
						vsum[c] = 0;
					for (int ki = 0; ki < nkv; ki++) {
						float dot = 0;
						for (int c = 0; c < head_dim; c++)
							dot += buf_q[qi * d + hs + c] *
							       buf_k[ki * d + hs + c];
						float e = expf(dot * scale - maxs);
						sexp += e;
						for (int c = 0; c < head_dim; c++)
							vsum[c] += e * buf_v[ki * d + hs + c];
					}
					float inv_s = 1.0f / sexp;
					for (int c = 0; c < head_dim; c++)
						buf_ca[qi * d + hs + c] = vsum[c] * inv_s;
				}
			}
			{
				const float *w = (const float *)enc->layers[li].ca_out_w->data;
				const float *b = (const float *)enc->layers[li].ca_out_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_ca[r * d + k] * w[c * d + k];
						xd[r * d + c] += sum;
					}
				}
			}

			/* FFN: norm3 → fc1 → relu → fc2 → residual */
			{
				const float *gw = (const float *)enc->layers[li].norm3_w->data;
				const float *gb = (const float *)enc->layers[li].norm3_b->data;
				for (int r = 0; r < nq; r++) {
					const float *row = xd + r * d;
					double s = 0, s2 = 0;
					for (int c = 0; c < d; c++) {
						s += (double)row[c];
						s2 += (double)row[c] * (double)row[c];
					}
					double m = s / d, v2 = s2 / d - m * m;
					double inv2 = 1.0 / sqrt(v2 + eps);
					for (int c = 0; c < d; c++)
						buf_norm[r * d + c] = (float)(((double)row[c] - m) * inv2) * gw[c] + gb[c];
				}
			}
			{
				const float *w = (const float *)enc->layers[li].ffn_fc1_w->data;
				const float *b = (const float *)enc->layers[li].ffn_fc1_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < 2048; c++) {
						float sum = b[c];
						for (int k = 0; k < d; k++)
							sum += buf_norm[r * d + k] * w[c * d + k];
						buf_ff1[r * 2048 + c] = sum > 0 ? sum : 0;
					}
				}
			}
			{
				const float *w = (const float *)enc->layers[li].ffn_fc2_w->data;
				const float *b = (const float *)enc->layers[li].ffn_fc2_b->data;
				for (int r = 0; r < nq; r++) {
					for (int c = 0; c < d; c++) {
						float sum = b[c];
						for (int k = 0; k < 2048; k++)
							sum += buf_ff1[r * 2048 + k] * w[c * 2048 + k];
						xd[r * d + c] += sum;
					}
				}
			}

			sam3_log_debug("geom (cls-only): layer %d done", li);
		}
		persist->offset = sa_save;
		{
			struct sam3_tensor *xp;
			xp = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
					      2, x->dims);
			if (!xp) { err = SAM3_ENOMEM; goto fail; }
			memcpy(xp->data, x->data, x->nbytes);
			x = xp;
		}
		}

		/* Post-encoder: encode_norm (LayerNorm) */
		{
			int nrows = x->dims[0];
			const float *gw = (const float *)enc->encode_norm_w->data;
			const float *gb = (const float *)enc->encode_norm_b->data;
			float *xd = (float *)x->data;
			const float eps = 1e-5f;

			for (int r = 0; r < nrows; r++) {
				float *row = xd + r * d;
				double sum = 0.0, sum2 = 0.0;
				for (int c = 0; c < d; c++) {
					sum += (double)row[c];
					sum2 += (double)row[c] * (double)row[c];
				}
				double mean = sum / d;
				double var = sum2 / d - mean * mean;
				double inv = 1.0 / sqrt(var + eps);
				for (int c = 0; c < d; c++)
					row[c] = (float)(((double)row[c] - mean) *
						  inv) * gw[c] + gb[c];
			}
			geom_out = x;
		}

		sam3_log_debug("segment: geom encoder (cls-only) done, "
			       "persist %zu/%zu",
			       persist->offset, persist->size);
	}

	SAM3_PROF_END(profiler, "geometry_encode");

	/* Dump text features and geometry encoder output for fixture comparison */
	if (text_features)
		dump_tensor("/tmp/dbg_text_features.bin", text_features);
	if (geom_out)
		dump_tensor("/tmp/dbg_geom_out.bin", geom_out);

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
	SAM3_PROF_BEGIN(profiler, "encoder_fusion");
	{
		struct sam3_tensor *enc_x = img_2d;
		size_t x_bytes = enc_x->dims[0] * enc_x->dims[1]
				  * sizeof(float);
		void *x_buf = sam3_arena_alloc(persist, x_bytes);
		if (!x_buf) { err = SAM3_ENOMEM; goto fail; }

		/*
		 * Copy encoder input into persist buffer for
		 * per-layer evaluation. No 0.1*pos addition here;
		 * that belongs to TransformerEncoderCrossAttention
		 * (tracking encoder), not TransformerEncoderFusion
		 * (DETR encoder).
		 */
		memcpy(x_buf, enc_x->data, x_bytes);

		/* Dump encoder input for fixture comparison */
		{
			struct sam3_tensor dtmp;
			dtmp.dtype = SAM3_DTYPE_F32;
			dtmp.n_dims = 2;
			dtmp.dims[0] = img_2d->dims[0];
			dtmp.dims[1] = img_2d->dims[1];
			dtmp.data = x_buf;
			dtmp.nbytes = x_bytes;
			dump_tensor("/tmp/dbg_enc_input.bin", &dtmp);
		}

		/* Flatten pos_enc for encoder layers [H*W, d] */
		struct sam3_tensor *enc_pe_t;
		enc_pe_t = sam3_pos_encoding_get(
				&model->backbone.pos_enc);
		int epe_n = enc_pe_t->dims[0] * enc_pe_t->dims[1];
		int epe_d = enc_pe_t->dims[2];
		int epe_dims[] = {epe_n, epe_d};

		for (int li = 0; li < model->encoder.n_layers; li++) {
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int x_dims[] = {img_2d->dims[0],
					 img_2d->dims[1]};
			enc_x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
						2, x_dims, x_buf);
			if (!enc_x) { err = SAM3_ENOMEM; goto fail; }

			struct sam3_tensor *epos_wrap;
			epos_wrap = gh_tensor_wrap(scratch,
						     SAM3_DTYPE_F32,
						     2, epe_dims,
						     enc_pe_t->data);
			if (!epos_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			struct sam3_tensor *ctx_copy;
			ctx_copy = gh_tensor_wrap(scratch, context->dtype,
						   context->n_dims,
						   context->dims,
						   context->data);
			if (!ctx_copy) { err = SAM3_ENOMEM; goto fail; }

			enc_x = sam3_encoder_fusion_build_layer(
				&model->encoder, li, &g,
				enc_x, epos_wrap, ctx_copy, scratch);
			if (!enc_x) { err = SAM3_ENOMEM; goto fail; }

			err = be->ops->graph_eval(be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("segment: enc layer %d "
					       "eval failed: %d", li, err);
				goto fail;
			}
			memcpy(x_buf, enc_x->data, x_bytes);

			/* Dump per-layer output for fixture comparison */
			{
				char dpath[64];
				snprintf(dpath, sizeof(dpath),
					 "/tmp/dbg_enc_layer_%02d.bin", li);
				struct sam3_tensor dtmp;
				dtmp.dtype = SAM3_DTYPE_F32;
				dtmp.n_dims = 2;
				dtmp.dims[0] = img_2d->dims[0];
				dtmp.dims[1] = img_2d->dims[1];
				dtmp.data = x_buf;
				dtmp.nbytes = x_bytes;
				dump_tensor(dpath, &dtmp);
			}

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

		/*
		 * No final layer norm — the DETR encoder has none.
		 * Wrap x_buf as fused tensor directly in persist.
		 */
		{
			int x_dims[] = {img_2d->dims[0],
					 img_2d->dims[1]};
			fused = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						 2, x_dims);
			if (!fused) { err = SAM3_ENOMEM; goto fail; }
			memcpy(fused->data, x_buf,
			       (size_t)x_dims[0] * x_dims[1]
			       * sizeof(float));
		}
	}

	SAM3_PROF_END(profiler, "encoder_fusion");

	sam3_log_debug("segment: encoder fusion done, persist %zu/%zu",
		       persist->offset, persist->size);

	dump_tensor("/tmp/dbg_fused.bin", fused);

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
	SAM3_PROF_BEGIN(profiler, "decoder");
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

		/*
		 * Compute initial query_pos from reference_points.
		 * Steps: sigmoid(ref_points) → sine_embed (CPU)
		 *        → ref_point_head MLP (graph eval once).
		 *
		 * After each decoder layer, box refinement updates
		 * ref_boxes and query_pos is recomputed. This matches
		 * Python's iterative box refinement with DAB-DETR
		 * conditional queries.
		 */
		float *ref_boxes = (float *)sam3_arena_alloc(
			persist, (size_t)nq * 4 * sizeof(float));
		float *qpos_buf = (float *)sam3_arena_alloc(
			persist, (size_t)nq * d * sizeof(float));

		/* Scratch for CPU box refinement between layers */
		float *br_tmp1 = (float *)sam3_arena_alloc(
			persist, (size_t)nq * d * sizeof(float));
		float *br_tmp2 = (float *)sam3_arena_alloc(
			persist, (size_t)nq * d * sizeof(float));
		if (!ref_boxes || !qpos_buf || !br_tmp1 || !br_tmp2) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		{
			const float *rp = (const float *)
				model->decoder.reference_points->data;
			int rp_n = nq * 4;
			for (int ri = 0; ri < rp_n; ri++)
				ref_boxes[ri] = 1.0f /
					(1.0f + expf(-rp[ri]));

			DEC_BE_ARENA_RESET();
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			struct sam3_tensor *qpos;
			qpos = sam3_decoder_compute_query_pos(
				&model->decoder, &g, scratch,
				ref_boxes);
			if (!qpos) {
				sam3_log_error("segment: query_pos "
					       "build failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = dec_be->ops->graph_eval(dec_be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("segment: query_pos "
					       "eval failed: %d", err);
				goto fail;
			}

			memcpy(qpos_buf, qpos->data,
			       (size_t)nq * d * sizeof(float));

			sam3_log_debug("segment: query_pos computed "
				       "[%d,%d]", nq, d);
		}

		int qpos_dims[] = {nq, d};

		/* Dump query_pos and initial queries for debugging */
		{
			struct sam3_tensor qptmp;
			qptmp.dtype = SAM3_DTYPE_F32;
			qptmp.n_dims = 2;
			qptmp.dims[0] = nq;
			qptmp.dims[1] = d;
			qptmp.data = qpos_buf;
			qptmp.nbytes = (size_t)nq * d * sizeof(float);
			dump_tensor("/tmp/dbg_query_pos.bin", &qptmp);

			qptmp.data = q_buf;
			dump_tensor("/tmp/dbg_q_init.bin", &qptmp);
		}

		/*
		 * Flatten position encoding [H, W, d] → [H*W, d]
		 * to match enc_features [n_pixels, d_model].
		 */
		struct sam3_tensor *pos_enc_t;
		pos_enc_t = sam3_pos_encoding_get(
				&model->backbone.pos_enc);
		int pe_n = pos_enc_t->dims[0] * pos_enc_t->dims[1];
		int pe_d = pos_enc_t->dims[2];
		int pe_dims[] = {pe_n, pe_d};
		int pe_H = pos_enc_t->dims[0];
		int pe_W = pos_enc_t->dims[1];

		/* Box-relative positional bias buffer [n_heads, nq, pe_n] */
		int rpb_size = model->decoder.n_heads * nq * pe_n;
		float *rpb_buf = (float *)sam3_arena_alloc(
			persist, (size_t)rpb_size * sizeof(float));
		if (!rpb_buf) {
			err = SAM3_ENOMEM;
			goto fail;
		}

		for (int li = 0; li < model->decoder.n_layers; li++) {
			/*
			 * Build/eval each decoder substep as a
			 * separate graph to isolate divergence.
			 */
			struct sam3_tensor *q_in;
			struct sam3_tensor *qpos_in;
			struct sam3_tensor *enc_wrap;
			struct sam3_tensor *enc_pos_wrap;
			struct sam3_tensor *txt_wrap;
			struct sam3_tensor *q_out;

#define DEC_WRAP_INPUTS() do { \
	q_in = gh_tensor_wrap(scratch, SAM3_DTYPE_F32, \
			       2, q_dims, q_buf); \
	qpos_in = gh_tensor_wrap(scratch, SAM3_DTYPE_F32, \
				  2, qpos_dims, qpos_buf); \
	enc_wrap = gh_tensor_wrap(scratch, fused->dtype, \
				   fused->n_dims, fused->dims, \
				   fused->data); \
	enc_pos_wrap = gh_tensor_wrap(scratch, SAM3_DTYPE_F32, \
				       2, pe_dims, \
				       pos_enc_t->data); \
	txt_wrap = context \
		? gh_tensor_wrap(scratch, context->dtype, \
				 context->n_dims, \
				 context->dims, \
				 context->data) \
		: NULL; \
	if (!q_in || !qpos_in || !enc_wrap || \
	    !enc_pos_wrap) { \
		err = SAM3_ENOMEM; \
		goto fail; \
	} \
} while (0)

			/* Substep A: Self-attention */
			SAM3_PROF_BEGIN(profiler, "dec_sa");
			DEC_BE_ARENA_RESET();
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);
			DEC_WRAP_INPUTS();
			q_out = sam3_decoder_build_sa(
				&model->decoder, li, &g,
				q_in, qpos_in, scratch);
			if (!q_out) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			err = dec_be->ops->graph_eval(dec_be, &g);
			if (err != SAM3_OK) goto fail;
			memcpy(q_buf, q_out->data,
			       (size_t)nq * d * sizeof(float));

			/* Dump after SA */
			if (li == 0) {
				struct sam3_tensor dt;
				dt.dtype = SAM3_DTYPE_F32;
				dt.n_dims = 2;
				dt.dims[0] = nq; dt.dims[1] = d;
				dt.data = q_buf;
				dt.nbytes = (size_t)nq * d *
					sizeof(float);
				dump_tensor(
					"/tmp/dbg_dec_l0_sa.bin",
					&dt);
			}

			SAM3_PROF_END(profiler, "dec_sa");

			/* Substep B: Text cross-attention */
			SAM3_PROF_BEGIN(profiler, "dec_tca");
			DEC_BE_ARENA_RESET();
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);
			DEC_WRAP_INPUTS();
			q_out = sam3_decoder_build_tca(
				&model->decoder, li, &g,
				q_in, qpos_in, txt_wrap, scratch);
			if (!q_out) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			err = dec_be->ops->graph_eval(dec_be, &g);
			if (err != SAM3_OK) goto fail;
			memcpy(q_buf, q_out->data,
			       (size_t)nq * d * sizeof(float));

			/* Dump after TCA */
			if (li == 0) {
				struct sam3_tensor dt;
				dt.dtype = SAM3_DTYPE_F32;
				dt.n_dims = 2;
				dt.dims[0] = nq; dt.dims[1] = d;
				dt.data = q_buf;
				dt.nbytes = (size_t)nq * d *
					sizeof(float);
				dump_tensor(
					"/tmp/dbg_dec_l0_tca.bin",
					&dt);
			}

			SAM3_PROF_END(profiler, "dec_tca");

			/* Substep C: Vision cross-attention with boxRPB */
			SAM3_PROF_BEGIN(profiler, "dec_rpb");
			sam3_decoder_compute_rpb(&model->decoder,
						  ref_boxes,
						  pe_H, pe_W, rpb_buf);
			SAM3_PROF_END(profiler, "dec_rpb");
			SAM3_PROF_BEGIN(profiler, "dec_ca");
			DEC_BE_ARENA_RESET();
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);
			DEC_WRAP_INPUTS();
			int rpb_dims[] = {model->decoder.n_heads,
					   nq, pe_n};
			struct sam3_tensor *rpb_wrap;
			rpb_wrap = gh_tensor_wrap(scratch,
				SAM3_DTYPE_F32, 3, rpb_dims,
				rpb_buf);
			if (!rpb_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			q_out = sam3_decoder_build_ca(
				&model->decoder, li, &g,
				q_in, qpos_in,
				enc_wrap, enc_pos_wrap,
				rpb_wrap, scratch);
			if (!q_out) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			err = dec_be->ops->graph_eval(dec_be, &g);
			if (err != SAM3_OK) goto fail;
			memcpy(q_buf, q_out->data,
			       (size_t)nq * d * sizeof(float));

			/* Dump after CA */
			if (li == 0) {
				struct sam3_tensor dt;
				dt.dtype = SAM3_DTYPE_F32;
				dt.n_dims = 2;
				dt.dims[0] = nq; dt.dims[1] = d;
				dt.data = q_buf;
				dt.nbytes = (size_t)nq * d *
					sizeof(float);
				dump_tensor(
					"/tmp/dbg_dec_l0_ca.bin",
					&dt);
			}

			SAM3_PROF_END(profiler, "dec_ca");

			/* Substep D: FFN */
			SAM3_PROF_BEGIN(profiler, "dec_ffn");
			DEC_BE_ARENA_RESET();
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);
			DEC_WRAP_INPUTS();
			q_out = sam3_decoder_build_ffn(
				&model->decoder, li, &g,
				q_in, scratch);
			if (!q_out) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			err = dec_be->ops->graph_eval(dec_be, &g);
			if (err != SAM3_OK) goto fail;
			memcpy(q_buf, q_out->data,
			       (size_t)nq * d * sizeof(float));

			SAM3_PROF_END(profiler, "dec_ffn");

#undef DEC_WRAP_INPUTS

			/* Dump per-layer decoder output */
			{
				char dpath[64];
				snprintf(dpath, sizeof(dpath),
					 "/tmp/dbg_dec_layer_%02d.bin",
					 li);
				struct sam3_tensor dtmp;
				dtmp.dtype = SAM3_DTYPE_F32;
				dtmp.n_dims = 2;
				dtmp.dims[0] = nq;
				dtmp.dims[1] = d;
				dtmp.data = q_buf;
				dtmp.nbytes = (size_t)nq * d *
					sizeof(float);
				dump_tensor(dpath, &dtmp);
			}

			/* Diagnostic: per-layer stats */
			{
				const float *ld = (const float *)q_buf;
				int ntot = nq * d;
				float lmin = ld[0], lmax = ld[0];
				for (int lj = 0; lj < ntot; lj++) {
					if (ld[lj] < lmin) lmin = ld[lj];
					if (ld[lj] > lmax) lmax = ld[lj];
				}
				sam3_log_debug("segment: dec layer %d "
					       "q min=%.4f max=%.4f",
					       li, lmin, lmax);
			}

			/*
			 * Box refinement + query_pos update (CPU).
			 *
			 * Python applies iterative box refinement after
			 * each layer:
			 *   delta = box_head(norm(output))
			 *   ref = sigmoid(inverse_sigmoid(ref) + delta)
			 *   query_pos = ref_point_head(sine_embed(ref))
			 *
			 * We do this on CPU since the Metal backend only
			 * copies back final graph outputs. The box head
			 * MLP on [200, 256] is trivially fast on CPU.
			 */
			SAM3_PROF_BEGIN(profiler, "dec_box_refine");
			cpu_box_refine(q_buf, &model->decoder,
				       ref_boxes, nq, d,
				       br_tmp1, br_tmp2);
			SAM3_PROF_END(profiler, "dec_box_refine");

			/* Recompute query_pos for next layer */
			if (li < model->decoder.n_layers - 1) {
				DEC_BE_ARENA_RESET();
				sam3_arena_reset(scratch);
				sam3_graph_init(&g);

				struct sam3_tensor *qpos;
				qpos = sam3_decoder_compute_query_pos(
					&model->decoder, &g, scratch,
					ref_boxes);
				if (!qpos) {
					sam3_log_error("segment: "
						"query_pos rebuild "
						"failed at layer %d",
						li);
					err = SAM3_ENOMEM;
					goto fail;
				}

				err = dec_be->ops->graph_eval(dec_be, &g);
				if (err != SAM3_OK) {
					sam3_log_error("segment: "
						"query_pos eval "
						"failed at layer %d: "
						"%d", li, err);
					goto fail;
				}

				memcpy(qpos_buf, qpos->data,
				       (size_t)nq * d *
				       sizeof(float));
			}
		}

		/* Final: output layer norm */
		DEC_BE_ARENA_RESET();
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

		err = dec_be->ops->graph_eval(dec_be, &g);
		if (err != SAM3_OK) {
			sam3_log_error("segment: dec final eval failed");
			goto fail;
		}

		queries = persist_tensor(persist, queries);
		if (!queries) {
			err = SAM3_ENOMEM;
			goto fail;
		}

		/* Dump decoder queries for Python comparison */
		dump_tensor("/tmp/dbg_queries.bin", queries);

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

		SAM3_PROF_END(profiler, "decoder");

		/*
		 * Stage 4: Segmentation head — FPN pixel decoder +
		 * mask embedder MLP + dot product mask prediction.
		 */
		SAM3_PROF_BEGIN(profiler, "seg_head");
		grid_h = model->cached_feat_s1_nhwc->dims[1];
		grid_w = model->cached_feat_s1_nhwc->dims[2];

		sam3_arena_reset(scratch);
		sam3_graph_init(&g);

		/* Diagnostic: check backbone feature stats */
		{
			struct sam3_tensor *bf[] = {
				fused,
				model->cached_feat_s1_nhwc,
				model->cached_feat_s0_nhwc,
				model->cached_feat_4x_nhwc
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
		 * separate graph so the cross-attended result is
		 * persisted before we wrap it as NHWC for the FPN.
		 */
		struct sam3_tensor *seg_enc = fused;
		if (context) {
			struct sam3_tensor *enc_wrap;
			struct sam3_tensor *txt_wrap;

			enc_wrap = gh_tensor_wrap(scratch,
				fused->dtype, fused->n_dims,
				fused->dims, fused->data);
			txt_wrap = gh_tensor_wrap(scratch,
				context->dtype,
				context->n_dims,
				context->dims,
				context->data);
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

			/* Materialize result so NHWC wrap can read it */
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
		 * Stage 4a: Reshape seg_enc to NHWC [1, H, W, d].
		 *
		 * seg_enc is [seq, d] = [grid_h*grid_w, d]. Because
		 * NHWC element order is N,H,W,C and seg_enc is already
		 * row-major over (h*W + w, d), the reshape is a pure
		 * view change with no data movement.
		 */
		int seg_d = model->seg_head.d_model;
		int nhwc_sd[] = {1, grid_h, grid_w, seg_d};
		struct sam3_tensor *enc_nhwc;
		enc_nhwc = gh_tensor_wrap(scratch, seg_enc->dtype,
					   4, nhwc_sd, seg_enc->data);
		if (!enc_nhwc) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		sam3_log_debug("seg: enc_nhwc [1,%d,%d,%d]",
			       grid_h, grid_w, seg_d);

		/*
		 * Stage 4b: FPN pixel decoder + instance projection.
		 * Evaluated as separate graph to get real inst values.
		 */
		struct sam3_tensor *inst;
		{
			sam3_graph_init(&g);

			dump_tensor("/tmp/dbg_enc_nhwc.bin", enc_nhwc);
			dump_tensor("/tmp/dbg_feat_1x.bin",
				model->cached_feat_s1_nhwc);
			dump_tensor("/tmp/dbg_feat_2x.bin",
				model->cached_feat_s0_nhwc);
			dump_tensor("/tmp/dbg_feat_4x.bin",
				model->cached_feat_4x_nhwc);

			/*
			 * Split FPN and inst_proj into two graph evals
			 * so pixel_embed data gets copied back.
			 */
			{
				/* Build FPN only → pixel_embed (NHWC) */
				struct sam3_tensor *pixel_embed;
				pixel_embed = sam3_seg_head_build_pixel_decoder(
					&model->seg_head, &g,
					enc_nhwc,
					model->cached_feat_s0_nhwc,
					model->cached_feat_4x_nhwc,
					scratch);
				if (!pixel_embed) {
					sam3_log_error("seg: FPN build failed");
					err = SAM3_ENOMEM;
					goto fail;
				}

				sam3_log_debug("seg: FPN built, %d nodes",
					       g.n_nodes);

				err = be->ops->graph_eval(be, &g);
				if (err != SAM3_OK) {
					sam3_log_error("seg: FPN eval failed");
					goto fail;
				}

				pixel_embed = persist_tensor(persist, pixel_embed);
				if (!pixel_embed) {
					err = SAM3_ENOMEM;
					goto fail;
				}

				dump_tensor("/tmp/dbg_pixel_embed.bin",
					    pixel_embed);

				/* Build inst_proj (1x1 NHWC conv) */
				sam3_arena_reset(scratch);
				sam3_graph_init(&g);

				struct sam3_tensor *pe_wrap;
				pe_wrap = gh_tensor_wrap(scratch,
					pixel_embed->dtype,
					pixel_embed->n_dims,
					pixel_embed->dims,
					pixel_embed->data);
				if (!pe_wrap) {
					err = SAM3_ENOMEM;
					goto fail;
				}

				inst = gh_conv2d(&g, scratch,
						  pe_wrap,
						  model->seg_head.inst_proj_w,
						  model->seg_head.inst_proj_b,
						  1, 0);
				if (!inst) {
					sam3_log_error("seg: inst proj build failed");
					err = SAM3_ENOMEM;
					goto fail;
				}

				err = be->ops->graph_eval(be, &g);
				if (err != SAM3_OK) {
					sam3_log_error("seg: inst proj eval failed");
					goto fail;
				}

				inst = persist_tensor(persist, inst);
				if (!inst) {
					err = SAM3_ENOMEM;
					goto fail;
				}
			}

			dump_tensor("/tmp/dbg_inst.bin", inst);
			dump_tensor("/tmp/dbg_inst_proj_w.bin",
				model->seg_head.inst_proj_w);
			dump_tensor("/tmp/dbg_inst_proj_b.bin",
				model->seg_head.inst_proj_b);

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

				/*
				 * Spatial smoothness metric — inst is NHWC
				 * [1, H, W, C], walk over channels with
				 * stride C between spatial samples.
				 */
				int iH = inst->dims[1];
				int iW = inst->dims[2];
				int iC = inst->dims[3];
				double adj_h = 0, adj_w = 0, aabs = 0;
				long cnt_h = 0, cnt_w = 0, cnt_a = 0;
				for (int c = 0; c < iC; c++) {
					for (int h = 0; h < iH; h++)
					for (int w = 0; w < iW; w++) {
						float v = id[(h*iW + w)*iC + c];
						aabs += fabs(v);
						cnt_a++;
						if (w+1 < iW) {
							adj_w += fabs(v -
								id[(h*iW+w+1)*iC + c]);
							cnt_w++;
						}
						if (h+1 < iH) {
							adj_h += fabs(v -
								id[((h+1)*iW+w)*iC + c]);
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

			dump_tensor("/tmp/dbg_mask_embed.bin", mask_embed);

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
		 * inst is NHWC [1, H, W, d]. Reshape to [H*W, d],
		 * transpose to [d, H*W], then matmul:
		 * mask_embed [nq, d] @ inst_flat_t [d, H*W] → [nq, H*W]
		 */
		{
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			int final_h = inst->dims[1];
			int final_w = inst->dims[2];
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

			/* Reshape inst [1,H,W,d] → [H*W, d] */
			int flat_dims[] = {final_hw, seg_d};
			struct sam3_tensor *inst_flat;
			inst_flat = gh_reshape(&g, scratch, inst_wrap,
						2, flat_dims);
			if (!inst_flat) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			/* Transpose [H*W, d] → [d, H*W] */
			struct sam3_tensor *inst_flat_t;
			inst_flat_t = gh_transpose(&g, scratch, inst_flat);
			if (!inst_flat_t) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			/* masks = mask_embed @ inst_flat_t */
			masks = gh_matmul(&g, scratch,
					   me_wrap, inst_flat_t);
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

			/*
			 * Dump inst L2 norm spatial map. inst is NHWC
			 * [1, H, W, d], so channel values for pixel p
			 * live at id[p*d + c].
			 */
			{
				const float *id =
					(const float *)inst->data;
				int ic = inst->dims[3];
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
								float v = id[p * ic + c];
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
		 * Stage 4e: Objectness scorer (DotProductScoring).
		 *
		 * Produces per-query confidence logits [n_queries, 1]
		 * via scaled dot product between projected decoder
		 * outputs and mean-pooled prompt features.
		 *
		 * The caller should apply sigmoid to get probabilities.
		 */
		if (context) {
			DEC_BE_ARENA_RESET();
			sam3_arena_reset(scratch);
			sam3_graph_init(&g);

			sam3_log_debug("seg: scorer queries "
				"[%d,%d] context [%d,%d]",
				queries->dims[0], queries->dims[1],
				context->dims[0], context->dims[1]);

			struct sam3_tensor *q_wrap2, *ctx_wrap;
			q_wrap2 = gh_tensor_wrap(scratch,
				queries->dtype, queries->n_dims,
				queries->dims, queries->data);
			ctx_wrap = gh_tensor_wrap(scratch,
				context->dtype, context->n_dims,
				context->dims, context->data);
			if (!q_wrap2 || !ctx_wrap) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			scores = sam3_dot_scorer_build(&model->scorer,
				&g, q_wrap2, ctx_wrap, scratch);
			if (!scores) {
				sam3_log_error("seg: scorer build failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			err = dec_be->ops->graph_eval(dec_be, &g);
			if (err != SAM3_OK) {
				sam3_log_error("seg: scorer eval failed");
				goto fail;
			}

			scores = persist_tensor(persist, scores);
			if (!scores) {
				err = SAM3_ENOMEM;
				goto fail;
			}

			dump_tensor("/tmp/dbg_scorer.bin", scores);
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

	SAM3_PROF_END(profiler, "seg_head");

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

#undef DEC_BE_ARENA_RESET

	*out_masks = masks;
	return SAM3_OK;

fail:
	persist->offset = persist_save;
	return err;
}
