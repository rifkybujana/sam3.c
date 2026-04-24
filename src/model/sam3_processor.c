/*
 * src/model/sam3_processor.c - High-level image processor implementation
 *
 * Implements the top-level processor that manages backend, arenas, and
 * model lifetime. Handles pixel normalization (uint8 RGB to float CHW),
 * prompt projection (point/box coordinates to d_model embeddings), and
 * orchestrates the encode/segment pipeline. Arena resets between
 * operations keep memory bounded.
 *
 * Key types:  sam3_processor
 * Depends on: sam3_processor.h, graph_helpers.h, core/weight.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    sam3.c (top-level context), tools/
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sam3_processor.h"
#include "graph_helpers.h"
#include "core/weight.h"
#include "backend/backend.h"
#include "backend/cpu/cpu_backend.h"
#include "util/hash.h"
#include "util/log.h"
#include "feature_cache.h"

#include "sam3/internal/processor_normalize.h"
#include "sam3/internal/mask_select.h"
#include "sam3/internal/mask_boxes.h"
#include "mask_decoder.h"
#include "util/profile.h"

/* Forward declarations for async text worker helpers (defined below). */
static void join_text_worker(struct sam3_processor *proc);
static void *text_worker_main(void *arg);

static void reset_image_cached_state(struct sam3_processor *proc)
{
	proc->model.image_encoded = 0;
	proc->model.cached_image_features = NULL;
	proc->model.cached_feat_s0_nhwc = NULL;
	proc->model.cached_feat_s1_nhwc = NULL;
	proc->model.cached_feat_4x_nhwc = NULL;
	proc->model.cached_sam2_05x_nhwc = NULL;
	proc->model.cached_sam2_1x_nhwc = NULL;
	proc->model.cached_sam2_2x_nhwc = NULL;
	proc->model.cached_sam2_4x_nhwc = NULL;
	proc->image_loaded = 0;
	proc->current_img_slot = -1;
}

void sam3_normalize_rgb_chw(const uint8_t *src, float *dst,
			    int width, int height)
{
	int c, y, x;

	for (c = 0; c < 3; c++) {
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				int src_idx = (y * width + x) * 3 + c;
				int dst_idx = c * height * width +
					      y * width + x;
				dst[dst_idx] =
					(float)src[src_idx] / 127.5f - 1.0f;
			}
		}
	}
}

enum sam3_error sam3_processor_init(struct sam3_processor *proc,
				    int backbone_type, int n_fpn_scales)
{
	return sam3_processor_init_ex(proc, backbone_type, n_fpn_scales,
				      0, 0, 0, NULL);
}

enum sam3_error sam3_processor_init_ex(struct sam3_processor *proc,
				       int backbone_type,
				       int n_fpn_scales,
				       int n_image_slots,
				       int n_text_slots,
				       size_t image_mem_budget_bytes,
				       const char *image_spill_dir)
{
	enum sam3_error err;

	memset(proc, 0, sizeof(*proc));

	/*
	 * Try Metal first (lazy eval, no pre-allocation of intermediates),
	 * fall back to CPU with large arena if unavailable.
	 */
#ifdef SAM3_HAS_METAL
	proc->backend = sam3_backend_init(SAM3_BACKEND_METAL);
#endif
	if (!proc->backend) {
		struct sam3_cpu_backend *cpu = calloc(1, sizeof(*cpu));
		if (!cpu)
			return SAM3_ENOMEM;
		cpu->base.type = SAM3_BACKEND_CPU;
		cpu->base.ops = sam3_cpu_backend_ops();
		/*
		 * CPU inference path: the backend arena is unused during
		 * normal inference (model code allocates from processor
		 * arenas via gh_alloc_tensor), so keep it minimal.
		 * The backend scratch handles conv2d im2col buffers;
		 * native NHWC kernels eliminated layout conversions,
		 * so the FPN 3×3 conv on 288×288×256 now needs only
		 * the im2col buffer (~729 MiB).
		 */
		cpu->arena_capacity = 64UL * 1024 * 1024;
		cpu->scratch_capacity = 1024UL * 1024 * 1024;
		err = cpu->base.ops->init(&cpu->base);
		if (err != SAM3_OK) {
			free(cpu);
			return err;
		}
		proc->backend = &cpu->base;
	}

	/*
	 * Arena sizes are backend-aware. The Metal path uses skip_data
	 * for intermediate tensors (MLX manages GPU memory), so arenas
	 * hold mostly tensor structs and need large capacity for the
	 * combined graph metadata. The CPU path allocates actual data
	 * buffers but resets between stages, so per-stage peak is the
	 * binding constraint.
	 *
	 * CPU model_arena must hold fused QKV weights (Hiera: 32 layers
	 * × 3×1024×1024 f32 = 384 MiB), tiled pos_embed (20 MiB),
	 * plus neck, decoder, and per-segment persistent data. The full
	 * Hiera model needs ~600 MiB; 1.5 GiB gives headroom.
	 *
	 * CPU scratch_arena must hold a full ViT block batch. Hiera
	 * with batch=2 peaks at ~2.1 GiB per batch (attention scores
	 * dominate: 9 windows × 16 heads × 576×576 × 4B = 188 MiB
	 * per block, plus MLP expansion at 94 MiB). 2.5 GiB gives
	 * headroom for the FPN pixel decoder as well.
	 */
	int is_cpu = (proc->backend->type == SAM3_BACKEND_CPU);
	/* Model arena holds weights + per-frame cached features. The video
	 * tracker now caches BOTH sam3 and sam2 neck outputs (4 scales each:
	 * 4x = 81 MiB, 2x = 20 MiB, 1x = 5 MiB, 0.5x = 1.3 MiB ≈ 107 MiB
	 * per neck), so the second neck adds ~107 MiB on top of single-image
	 * usage. Bump caps accordingly. */
	size_t model_cap = is_cpu ? 1792UL * 1024 * 1024
				  : 2304UL * 1024 * 1024;
	size_t scratch_cap = is_cpu ? 3072UL * 1024 * 1024
				    : 3584UL * 1024 * 1024;

	err = sam3_arena_init(&proc->model_arena, model_cap);
	if (err != SAM3_OK)
		goto cleanup_backend;

	err = sam3_arena_init(&proc->scratch_arena, scratch_cap);
	if (err != SAM3_OK)
		goto cleanup_model_arena;

	/*
	 * Async text encoder scratch arena (#11). Sized for the CLIP
	 * text encoder operating on a 77-token sequence with 4-block
	 * batching: per-block scratch peaks well under 256 MiB. The
	 * worker's persistent output now lives in the txt_cache slot
	 * arena (created below), not in a dedicated arena.
	 */
	err = sam3_arena_init(&proc->text_scratch_arena,
			      256UL * 1024 * 1024);
	if (err != SAM3_OK)
		goto cleanup_scratch_arena;

	/*
	 * Text feature cache. Each slot holds one [n_tokens, d_model]
	 * f32 tensor; max sized for 77 tokens × 1024 d_model = 308 KiB.
	 * Bundle a 1 MiB cell to leave room for tensor headers and
	 * worker-local scratch. Default 16 slots → 16 MiB total.
	 * (Observed real usage on realistic prompts: ~170 KiB per slot.)
	 */
	proc->txt_cache = sam3_text_cache_create(n_text_slots,
						 1UL * 1024 * 1024);
	if (!proc->txt_cache) {
		err = SAM3_ENOMEM;
		goto cleanup_text_scratch_arena;
	}
	proc->text_cached_bundle = NULL;
	proc->text_worker_slot = -1;

	/*
	 * Async text encoder backend (#11). The text worker runs against
	 * its own CPU backend so it does not contend with the image
	 * encoder's Metal device. We deliberately use CPU here even
	 * when Metal is available: MLX-C 0.6 keeps a process-wide
	 * device kernel cache that is not safe to mutate from two
	 * threads concurrently (validated by test_metal_cpu_concurrent),
	 * so two parallel Metal backends would race on get_kernel().
	 * CPU + Metal sit on disjoint hardware and synchronize cleanly,
	 * which gives us true parallelism for the text + image encoders.
	 */
	proc->text_backend = NULL;
	{
		struct sam3_cpu_backend *cpu_t = calloc(1, sizeof(*cpu_t));
		if (cpu_t) {
			cpu_t->base.type = SAM3_BACKEND_CPU;
			cpu_t->base.ops  = sam3_cpu_backend_ops();
			cpu_t->arena_capacity = 64UL * 1024 * 1024;
			if (cpu_t->base.ops->init(&cpu_t->base) == SAM3_OK)
				proc->text_backend = &cpu_t->base;
			else
				free(cpu_t);
		}
		if (!proc->text_backend)
			sam3_log_warn("processor: text backend unavailable, "
				      "async text encoding disabled");
	}

	/* Initialize all sub-modules with appropriate backbone config */
	err = sam3_image_model_init(&proc->model, backbone_type,
				    n_fpn_scales, &proc->model_arena);
	if (err != SAM3_OK)
		goto cleanup_text_backend;

	err = sam3_arena_init(&proc->video_scratch_arena,
			      384UL * 1024 * 1024);
	if (err != SAM3_OK)
		goto cleanup_image_model;

	/*
	 * Allocate the image feature cache. Slot arena is sized for
	 * encoder peak output (the FPN + SAM2 neck scales, all in f32).
	 * Observed usage on Hiera @ 1008×1008: ~235 MiB per bundle.
	 * Use 256 MiB per slot (~9% headroom). When a budget is set,
	 * excess slots spill to disk under image_spill_dir.
	 */
	proc->img_cache = sam3_image_cache_create_ex(n_image_slots,
						     256UL * 1024 * 1024,
						     image_mem_budget_bytes,
						     image_spill_dir);
	if (!proc->img_cache) {
		err = SAM3_ENOMEM;
		goto cleanup_video_arena;
	}
	proc->current_img_slot = -1;

	return SAM3_OK;

cleanup_video_arena:
	sam3_arena_free(&proc->video_scratch_arena);
cleanup_image_model:
	sam3_image_model_free(&proc->model);
cleanup_text_backend:
	if (proc->text_backend) {
		proc->text_backend->ops->free(proc->text_backend);
		free(proc->text_backend);
		proc->text_backend = NULL;
	}
	sam3_text_cache_destroy(proc->txt_cache);
	proc->txt_cache = NULL;
cleanup_text_scratch_arena:
	sam3_arena_free(&proc->text_scratch_arena);
cleanup_scratch_arena:
	sam3_arena_free(&proc->scratch_arena);
cleanup_model_arena:
	sam3_arena_free(&proc->model_arena);
cleanup_backend:
	proc->backend->ops->free(proc->backend);
	free(proc->backend);
	proc->backend = NULL;
	return err;
}

enum sam3_error sam3_processor_load(struct sam3_processor *proc,
				    const struct sam3_weight_file *wf,
				    const char *vocab_path)
{
	return sam3_image_model_load(&proc->model, wf, vocab_path,
				     &proc->model_arena);
}

int sam3_processor_img_size(const struct sam3_processor *proc)
{
	return sam3_vl_backbone_img_size(&proc->model.backbone);
}

void sam3_processor_free(struct sam3_processor *proc)
{
	if (!proc)
		return;

	/* Join any in-flight async text worker before tearing down its
	 * arenas and backend. */
	join_text_worker(proc);

	sam3_image_cache_destroy(proc->img_cache);
	proc->img_cache = NULL;

	sam3_image_model_free(&proc->model);

	if (proc->text_backend) {
		proc->text_backend->ops->free(proc->text_backend);
		free(proc->text_backend);
		proc->text_backend = NULL;
	}

	if (proc->backend) {
		proc->backend->ops->free(proc->backend);
		free(proc->backend);
		proc->backend = NULL;
	}

	sam3_text_cache_destroy(proc->txt_cache);
	proc->txt_cache = NULL;
	sam3_arena_free(&proc->text_scratch_arena);
	sam3_arena_free(&proc->model_arena);
	sam3_arena_free(&proc->video_scratch_arena);
	sam3_arena_free(&proc->scratch_arena);
}

enum sam3_error sam3_processor_set_image(struct sam3_processor *proc,
					 const uint8_t *pixels,
					 int width, int height)
{
	struct sam3_tensor *image;
	enum sam3_error err;
	int dims[3];
	float *dst;

	if (!proc || !pixels || width <= 0 || height <= 0)
		return SAM3_EINVAL;

	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)&width, sizeof(width), key);
	key = sam3_fnv1a_64((const uint8_t *)&height, sizeof(height), key);
	size_t n_bytes = (size_t)width * (size_t)height * 3;
	key = sam3_fnv1a_64(pixels, n_bytes, key);
	if (key == 0)
		key = 1;

	size_t pref_len = n_bytes < SAM3_CACHE_PREFIX_BYTES
			      ? n_bytes : SAM3_CACHE_PREFIX_BYTES;

	int hit = sam3_image_cache_lookup(proc->img_cache, key, pixels,
					  pref_len);
	if (hit >= 0) {
		struct sam3_image_bundle *b =
			&proc->img_cache->slots[hit].bundle;
		proc->model.cached_image_features = b->image_features;
		proc->model.cached_feat_s0_nhwc   = b->feat_s0_nhwc;
		proc->model.cached_feat_s1_nhwc   = b->feat_s1_nhwc;
		proc->model.cached_feat_4x_nhwc   = b->feat_4x_nhwc;
		proc->model.cached_sam2_05x_nhwc  = b->sam2_05x_nhwc;
		proc->model.cached_sam2_1x_nhwc   = b->sam2_1x_nhwc;
		proc->model.cached_sam2_2x_nhwc   = b->sam2_2x_nhwc;
		proc->model.cached_sam2_4x_nhwc   = b->sam2_4x_nhwc;
		proc->model.image_encoded = 1;
		proc->image_loaded = 1;
		proc->current_img_slot = hit;
		proc->prompt_w = b->prompt_w;
		proc->prompt_h = b->prompt_h;
		return SAM3_OK;
	}

	int slot = sam3_image_cache_claim_slot(proc->img_cache);
	if (slot < 0)
		return SAM3_ENOMEM;
	proc->current_img_slot = slot;
	struct sam3_arena *persist = &proc->img_cache->slots[slot].arena;

	sam3_arena_reset(&proc->scratch_arena);

	dims[0] = 3;
	dims[1] = height;
	dims[2] = width;
	image = gh_alloc_tensor(&proc->scratch_arena, SAM3_DTYPE_F32, 3, dims);
	if (!image) {
		reset_image_cached_state(proc);
		return SAM3_ENOMEM;
	}
	dst = (float *)image->data;
	SAM3_PROF_BEGIN(proc->profiler, "image_normalize");
	sam3_normalize_rgb_chw(pixels, dst, width, height);
	SAM3_PROF_END(proc->profiler, "image_normalize");

	SAM3_PROF_BEGIN(proc->profiler, "image_encode");
	err = sam3_image_model_encode(&proc->model, proc->backend, image,
				      &proc->scratch_arena, persist,
				      proc->profiler);
	SAM3_PROF_END(proc->profiler, "image_encode");
	if (err != SAM3_OK) {
		reset_image_cached_state(proc);
		return err;
	}

	proc->image_loaded = 1;
	proc->prompt_w = width;
	proc->prompt_h = height;

	struct sam3_image_bundle bundle = {0};
	bundle.image_features = proc->model.cached_image_features;
	bundle.feat_s0_nhwc   = proc->model.cached_feat_s0_nhwc;
	bundle.feat_s1_nhwc   = proc->model.cached_feat_s1_nhwc;
	bundle.feat_4x_nhwc   = proc->model.cached_feat_4x_nhwc;
	bundle.sam2_05x_nhwc  = proc->model.cached_sam2_05x_nhwc;
	bundle.sam2_1x_nhwc   = proc->model.cached_sam2_1x_nhwc;
	bundle.sam2_2x_nhwc   = proc->model.cached_sam2_2x_nhwc;
	bundle.sam2_4x_nhwc   = proc->model.cached_sam2_4x_nhwc;
	bundle.prompt_w = width;
	bundle.prompt_h = height;
	bundle.width = width;
	bundle.height = height;
	sam3_image_cache_register(proc->img_cache, slot, key,
				  pixels, pref_len, &bundle);
	return SAM3_OK;
}

/*
 * count_prompts_by_type - Count point and box prompts separately.
 */
static void count_prompts_by_type(const struct sam3_prompt *prompts,
				  int n_prompts,
				  int *n_points, int *n_boxes)
{
	int i;

	*n_points = 0;
	*n_boxes = 0;

	for (i = 0; i < n_prompts; i++) {
		switch (prompts[i].type) {
		case SAM3_PROMPT_POINT:
			(*n_points)++;
			break;
		case SAM3_PROMPT_BOX:
			(*n_boxes)++;
			break;
		default:
			break;
		}
	}
}

/*
 * cpu_linear - Manual CPU matmul: out[r,c] = sum(in[r,k]*W[c,k]) + b[c].
 *
 * @out:    Output buffer [nrows * out_dim]
 * @in:     Input buffer [nrows * in_dim]
 * @w:      Weight tensor [out_dim, in_dim] (row-major)
 * @b:      Bias tensor [out_dim]
 * @nrows:  Number of input rows
 * @in_dim: Input dimension
 * @out_dim: Output dimension
 */
static void cpu_linear(float *out, const float *in,
		       const float *w, const float *b,
		       int nrows, int in_dim, int out_dim)
{
	for (int r = 0; r < nrows; r++) {
		for (int c = 0; c < out_dim; c++) {
			float sum = b[c];
			for (int k = 0; k < in_dim; k++)
				sum += in[r * in_dim + k] * w[c * in_dim + k];
			out[r * out_dim + c] = sum;
		}
	}
}

/*
 * cpu_sinusoidal_posenc - Compute sinusoidal positional encoding.
 *
 * Matches PositionEmbeddingSine._encode_xy from the Python reference:
 *   dim_t[i] = temperature^(2*floor(i/2)/num_pos_feats)
 *   out[2k]   = sin(coord * scale / dim_t[2k])
 *   out[2k+1] = cos(coord * scale / dim_t[2k+1])
 *
 * @out:            Output buffer [num_pos_feats]
 * @coord:          Normalized coordinate in [0,1]
 * @num_pos_feats:  Number of output features (typically d_model/2 = 128)
 */
static void cpu_sinusoidal_posenc(float *out, float coord,
				  int num_pos_feats)
{
	const float temperature = 10000.0f;
	const float scale = 2.0f * 3.14159265358979323846f;
	float scaled = coord * scale;

	for (int i = 0; i < num_pos_feats / 2; i++) {
		float exp = 2.0f * (float)i / (float)num_pos_feats;
		float freq = powf(temperature, exp);
		float val = scaled / freq;
		out[2 * i] = sinf(val);
		out[2 * i + 1] = cosf(val);
	}
}

/*
 * cpu_bilinear_sample_nhwc - Bilinear interpolation from NHWC features.
 *
 * Matches PyTorch grid_sample with align_corners=False:
 *   pixel_pos = (grid_coord + 1) / 2 * size - 0.5
 *
 * @out:     Output buffer [C]
 * @data:    NHWC feature data (batch dim ignored, offset to n=0);
 *           element layout is (h, w, c) with channels innermost.
 * @C:       Number of channels
 * @H:       Height
 * @W:       Width
 * @norm_x:  Normalized x coordinate in [0,1]
 * @norm_y:  Normalized y coordinate in [0,1]
 */
static void cpu_bilinear_sample_nhwc(float *out, const float *data,
				     int C, int H, int W,
				     float norm_x, float norm_y)
{
	/* Convert [0,1] to grid [-1,1] then to pixel coords */
	float gx = 2.0f * norm_x - 1.0f;
	float gy = 2.0f * norm_y - 1.0f;
	float fx = (gx + 1.0f) / 2.0f * (float)W - 0.5f;
	float fy = (gy + 1.0f) / 2.0f * (float)H - 0.5f;

	int x0 = (int)floorf(fx);
	int y0 = (int)floorf(fy);
	float dx = fx - (float)x0;
	float dy = fy - (float)y0;

	/* Clamp to valid range */
	int x0c = x0 < 0 ? 0 : (x0 >= W ? W - 1 : x0);
	int x1c = (x0 + 1) < 0 ? 0 : ((x0 + 1) >= W ? W - 1 : x0 + 1);
	int y0c = y0 < 0 ? 0 : (y0 >= H ? H - 1 : y0);
	int y1c = (y0 + 1) < 0 ? 0 : ((y0 + 1) >= H ? H - 1 : y0 + 1);

	float w00 = (1.0f - dx) * (1.0f - dy);
	float w10 = dx * (1.0f - dy);
	float w01 = (1.0f - dx) * dy;
	float w11 = dx * dy;

	const float *p00 = data + (y0c * W + x0c) * C;
	const float *p10 = data + (y0c * W + x1c) * C;
	const float *p01 = data + (y1c * W + x0c) * C;
	const float *p11 = data + (y1c * W + x1c) * C;

	for (int c = 0; c < C; c++) {
		out[c] = w00 * p00[c] + w10 * p10[c] +
			 w01 * p01[c] + w11 * p11[c];
	}
}

/*
 * sam3_project_prompts - Project point/box coordinates to d_model
 * embeddings.
 *
 * Computes three encoding paths and sums them (matching Python):
 *   1. Direct projection: Linear(2, d) on normalized coords
 *   2. Positional encoding: sinusoidal pos enc + Linear(d, d)
 *   3. Pool projection: bilinear sample from LayerNormed img feats +
 *      Linear(d, d)
 * Then adds label embedding: type_embed + points_embed.
 *
 * Points use [N, 2] -> [N, d_model], boxes use [N, 4] -> [N, d_model].
 * Results are concatenated into a single [total, d_model] tensor.
 *
 * @model:         Loaded image model (for geom_enc weights)
 * @feat_s1_nhwc:  Cached 1x backbone feature [1, H, W, d_model] NHWC,
 *                 used for the pool-projection path. May be NULL; in
 *                 that case pool projection is skipped.
 * @prompts:       Array of point/box prompts
 * @n_prompts:     Number of prompts
 * @prompt_w:      Width for coord normalization
 * @prompt_h:      Height for coord normalization
 * @arena:         Arena for the output tensor and LN scratch
 */
struct sam3_tensor *sam3_project_prompts(
	struct sam3_image_model *model,
	const struct sam3_tensor *feat_s1_nhwc,
	const struct sam3_prompt *prompts,
	int n_prompts,
	int prompt_w, int prompt_h,
	struct sam3_arena *arena)
{
	struct sam3_geometry_encoder *enc = &model->geom_enc;
	int d = enc->d_model;
	int n_points, n_boxes;
	int i;
	float norm_w = (float)prompt_w;
	float norm_h = (float)prompt_h;

	count_prompts_by_type(prompts, n_prompts, &n_points, &n_boxes);

	if (n_points == 0 && n_boxes == 0)
		return NULL;

	int total = n_points + n_boxes;
	int out_dims[2] = {total, d};
	struct sam3_tensor *out;
	out = gh_alloc_tensor(arena, SAM3_DTYPE_F32, 2, out_dims);
	if (!out)
		return NULL;

	float *od = (float *)out->data;
	int row = 0;

	/*
	 * Prepare LayerNormed image features for pool projection.
	 * Python: img_pre_norm(img_feats[-1]) on [H*W, B, C] format.
	 * We normalize each spatial position across C channels.
	 */
	const float *img_normed = NULL;
	float *img_norm_buf = NULL;
	int feat_H = 0, feat_W = 0;
	if (n_points > 0 && enc->pool_proj_w && feat_s1_nhwc) {
		const struct sam3_tensor *feat = feat_s1_nhwc;
		/* NHWC: dims = [1, H, W, C] */
		feat_H = feat->dims[1];
		feat_W = feat->dims[2];
		int C = feat->dims[3];
		int HW = feat_H * feat_W;
		const float *fd = (const float *)feat->data;
		const float *nw = (const float *)enc->img_pre_norm_w->data;
		const float *nb = (const float *)enc->img_pre_norm_b->data;

		/* Allocate from arena for normed features NHWC [1,H,W,C] */
		int norm_dims[] = {1, feat_H, feat_W, C};
		struct sam3_tensor *norm_t = gh_alloc_tensor(arena,
			SAM3_DTYPE_F32, 4, norm_dims);
		if (!norm_t)
			return NULL;
		img_norm_buf = (float *)norm_t->data;

		/* LayerNorm per spatial position (along C axis).
		 * NHWC stores channels innermost: row = &fd[hw * C] */
		for (int hw = 0; hw < HW; hw++) {
			const float *row_in = fd + hw * C;
			float *row_out = img_norm_buf + hw * C;

			double mean = 0.0;
			for (int c = 0; c < C; c++)
				mean += (double)row_in[c];
			mean /= C;

			double var = 0.0;
			for (int c = 0; c < C; c++) {
				double diff = (double)row_in[c] - mean;
				var += diff * diff;
			}
			var /= C;

			float inv_std = 1.0f / sqrtf((float)var + 1e-5f);
			for (int c = 0; c < C; c++) {
				float normed = ((float)(row_in[c] -
					mean)) * inv_std;
				row_out[c] = normed * nw[c] + nb[c];
			}
		}
		img_normed = img_norm_buf;
	}

	/* Project points: direct + posenc + pool + label embed */
	if (n_points > 0) {
		const float *pw = (const float *)enc->point_proj_w->data;
		const float *pb = (const float *)enc->point_proj_b->data;
		const float *le = (const float *)enc->label_embed->data;

		for (i = 0; i < n_prompts; i++) {
			if (prompts[i].type != SAM3_PROMPT_POINT)
				continue;

			float coords[2] = {
				prompts[i].point.x / norm_w,
				prompts[i].point.y / norm_h
			};

			/* 1. Direct: out = coords @ W^T + b */
			cpu_linear(od + row * d, coords, pw, pb,
				   1, 2, d);

			/* 2. Positional encoding projection */
			if (enc->posenc_proj_w) {
				float pos_enc[512]; /* d_model max */
				float proj_tmp[512];
				int half_d = d / 2;
				cpu_sinusoidal_posenc(pos_enc,
					coords[0], half_d);
				cpu_sinusoidal_posenc(pos_enc + half_d,
					coords[1], half_d);
				cpu_linear(proj_tmp, pos_enc,
					(const float *)enc->posenc_proj_w->data,
					(const float *)enc->posenc_proj_b->data,
					1, d, d);
				for (int c = 0; c < d; c++)
					od[row * d + c] += proj_tmp[c];
			}

			/* 3. Pool projection: bilinear sample + linear */
			if (img_normed && enc->pool_proj_w) {
				float sampled[512];
				float proj_tmp[512];
				cpu_bilinear_sample_nhwc(sampled,
					img_normed, d,
					feat_H, feat_W,
					coords[0], coords[1]);
				cpu_linear(proj_tmp, sampled,
					(const float *)enc->pool_proj_w->data,
					(const float *)enc->pool_proj_b->data,
					1, d, d);
				for (int c = 0; c < d; c++)
					od[row * d + c] += proj_tmp[c];
			}

			/* 4. Add label embedding */
			int label = prompts[i].point.label;
			if (label >= 0 && label < enc->n_labels) {
				const float *lv = le + label * d;
				for (int c = 0; c < d; c++)
					od[row * d + c] += lv[c];
			}

			row++;
		}
	}

	/* Project boxes: normalize to [0,1], linear, add label embed */
	if (n_boxes > 0) {
		const float *bw = (const float *)enc->box_proj_w->data;
		const float *bb = (const float *)enc->box_proj_b->data;
		const float *le = (const float *)enc->label_embed->data;

		for (i = 0; i < n_prompts; i++) {
			if (prompts[i].type != SAM3_PROMPT_BOX)
				continue;

			float coords[4] = {
				prompts[i].box.x1 / norm_w,
				prompts[i].box.y1 / norm_h,
				prompts[i].box.x2 / norm_w,
				prompts[i].box.y2 / norm_h
			};

			/* Linear: out = coords @ W^T + b */
			cpu_linear(od + row * d, coords, bw, bb,
				   1, 4, d);

			/* Box label: default to 0 (positive) */
			const float *lv = le;  /* label_embed[0] */
			for (int c = 0; c < d; c++)
				od[row * d + c] += lv[c];

			row++;
		}
	}

	return out;
}

/*
 * find_text_prompt - Find the first text prompt in the array.
 *
 * Returns the text string, or NULL if no text prompt is present.
 */
static const char *find_text_prompt(const struct sam3_prompt *prompts,
				    int n_prompts)
{
	for (int i = 0; i < n_prompts; i++) {
		if (prompts[i].type == SAM3_PROMPT_TEXT)
			return prompts[i].text;
	}
	return NULL;
}

/*
 * join_text_worker - Wait for the text worker thread (if any) to
 * finish and clear the active flag. Safe to call even when no worker
 * is in flight.
 */
static void join_text_worker(struct sam3_processor *proc)
{
	if (proc->text_thread_active) {
		pthread_join(proc->text_thread, NULL);
		proc->text_thread_active = 0;
	}
}

/*
 * text_worker_main - pthread entry point for async text encoding.
 *
 * Reads pre-tokenized state from proc, runs the text encoder, writes
 * the result tensor into the pre-claimed text cache slot's arena, and
 * registers the bundle with the cache. The main thread must not touch
 * the slot arena or the cache's mutable tables until pthread_join
 * returns.
 */
static void *text_worker_main(void *arg)
{
	struct sam3_processor *proc = arg;
	struct sam3_text_encoder_iface *te_iface =
		&proc->model.backbone.text_iface;
	int ctx = te_iface->ctx_len;
	int slot = proc->text_worker_slot;
	struct sam3_arena *arena =
		sam3_text_cache_slot_arena(proc->txt_cache, slot);

	int tok_dims[1] = {ctx};
	struct sam3_tensor *tok = gh_alloc_tensor(arena, SAM3_DTYPE_I32,
						  1, tok_dims);
	if (!tok) {
		proc->text_thread_err = SAM3_ENOMEM;
		return NULL;
	}
	memcpy(tok->data, proc->text_tokens,
	       (size_t)ctx * sizeof(int32_t));

	struct sam3_tensor *features = te_iface->ops->build_perblock(
		te_iface, proc->text_backend, tok,
		&proc->text_scratch_arena, arena);
	if (!features) {
		proc->text_thread_err = SAM3_ENOMEM;
		return NULL;
	}

	/* Truncate to real tokens (mirrors the inline path). */
	if (features->dims[0] > proc->text_n_tokens) {
		int d = features->dims[1];
		int trunc_dims[2] = {proc->text_n_tokens, d};
		struct sam3_tensor *trunc = gh_alloc_tensor(arena,
				SAM3_DTYPE_F32, 2, trunc_dims);
		if (!trunc) {
			proc->text_thread_err = SAM3_ENOMEM;
			return NULL;
		}
		memcpy(trunc->data, features->data,
		       (size_t)proc->text_n_tokens * (size_t)d *
		       sizeof(float));
		features = trunc;
	}

	struct sam3_text_bundle b = {.features = features,
				     .n_tokens = proc->text_n_tokens};
	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)proc->text_tokens,
			    (size_t)proc->text_n_tokens * sizeof(int32_t),
			    key);
	if (key == 0)
		key = 1;

	sam3_text_cache_register(proc->txt_cache, slot, key,
				 proc->text_tokens, proc->text_n_tokens, &b);
	proc->text_thread_err = SAM3_OK;
	return NULL;
}

enum sam3_error sam3_processor_set_text(struct sam3_processor *proc,
					const char *text)
{
	struct sam3_text_encoder_iface *te_iface;
	int ctx, n_tokens, rc;

	if (!proc || !text)
		return SAM3_EINVAL;
	if (!proc->text_backend) {
		sam3_log_error("set_text: text_backend unavailable");
		return SAM3_EBACKEND;
	}

	/* Join any prior worker before stomping on shared state. */
	join_text_worker(proc);
	sam3_arena_reset(&proc->text_scratch_arena);
	proc->text_cached_bundle = NULL;
	proc->text_worker_slot = -1;
	proc->text_thread_err = SAM3_OK;

	te_iface = &proc->model.backbone.text_iface;
	ctx = te_iface->ctx_len;

	/* Tokenize on the caller thread (cheap, < 1ms). */
	n_tokens = sam3_tokenizer_encode(&proc->model.backbone.tokenizer,
					 text, proc->text_tokens, ctx);
	if (n_tokens <= 0) {
		sam3_log_error("set_text: tokenize failed");
		return SAM3_EINVAL;
	}
	proc->text_n_tokens = n_tokens;

	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)proc->text_tokens,
			    (size_t)n_tokens * sizeof(int32_t), key);
	if (key == 0)
		key = 1;

	int hit = sam3_text_cache_lookup(proc->txt_cache, key,
					 proc->text_tokens, n_tokens);
	if (hit >= 0) {
		proc->text_cached_bundle =
			&proc->txt_cache->slots[hit].bundle;
		return SAM3_OK;
	}

	int slot = sam3_text_cache_claim_slot(proc->txt_cache);
	if (slot < 0)
		return SAM3_ENOMEM;
	proc->text_worker_slot = slot;

	/*
	 * Spawn the worker. Request an 8 MiB stack — MLX-C plus ASan
	 * blows the default 512 KiB pthread stack on macOS, and the
	 * matching size is validated by test_metal_cpu_concurrent.
	 */
	pthread_attr_t attr;
	if (pthread_attr_init(&attr) != 0)
		return SAM3_EBACKEND;
	if (pthread_attr_setstacksize(&attr, 8UL * 1024 * 1024) != 0) {
		pthread_attr_destroy(&attr);
		return SAM3_EBACKEND;
	}
	rc = pthread_create(&proc->text_thread, &attr, text_worker_main,
			    proc);
	pthread_attr_destroy(&attr);
	if (rc != 0) {
		sam3_log_error("set_text: pthread_create failed (%d)", rc);
		return SAM3_EBACKEND;
	}
	proc->text_thread_active = 1;
	return SAM3_OK;
}

/*
 * sam3_processor_precache_image - Populate the image feature cache for
 * @pixels without changing the processor's current-image state.
 *
 * On a cache hit this is a no-op. On a miss we save model.cached_*
 * (plus image_loaded / current_img_slot / prompt_w/h), claim a slot,
 * encode into it, register the bundle, then restore the saved state.
 * The encode writes into its slot's arena so the image cache grows
 * exactly as it would under a set_image() miss.
 */
enum sam3_error sam3_processor_precache_image(struct sam3_processor *proc,
					      const uint8_t *pixels,
					      int width, int height)
{
	struct sam3_tensor *image;
	enum sam3_error err;
	int dims[3];

	if (!proc || !pixels || width <= 0 || height <= 0)
		return SAM3_EINVAL;

	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)&width, sizeof(width), key);
	key = sam3_fnv1a_64((const uint8_t *)&height, sizeof(height), key);
	size_t n_bytes = (size_t)width * (size_t)height * 3;
	key = sam3_fnv1a_64(pixels, n_bytes, key);
	if (key == 0)
		key = 1;

	size_t pref_len = n_bytes < SAM3_CACHE_PREFIX_BYTES
			      ? n_bytes : SAM3_CACHE_PREFIX_BYTES;

	int hit = sam3_image_cache_lookup(proc->img_cache, key, pixels,
					  pref_len);
	if (hit >= 0)
		return SAM3_OK;

	/* Save caller-visible state — the encoder call will overwrite
	 * model.cached_* as a side effect. */
	struct sam3_image_model *m = &proc->model;
	struct sam3_tensor *sv_feat = m->cached_image_features;
	struct sam3_tensor *sv_s0   = m->cached_feat_s0_nhwc;
	struct sam3_tensor *sv_s1   = m->cached_feat_s1_nhwc;
	struct sam3_tensor *sv_4x   = m->cached_feat_4x_nhwc;
	struct sam3_tensor *sv_s2_05 = m->cached_sam2_05x_nhwc;
	struct sam3_tensor *sv_s2_1  = m->cached_sam2_1x_nhwc;
	struct sam3_tensor *sv_s2_2  = m->cached_sam2_2x_nhwc;
	struct sam3_tensor *sv_s2_4  = m->cached_sam2_4x_nhwc;
	int sv_encoded  = m->image_encoded;
	int sv_loaded   = proc->image_loaded;
	int sv_slot     = proc->current_img_slot;
	int sv_prompt_w = proc->prompt_w;
	int sv_prompt_h = proc->prompt_h;

	int slot = sam3_image_cache_claim_slot(proc->img_cache);
	if (slot < 0)
		return SAM3_ENOMEM;
	struct sam3_arena *persist = &proc->img_cache->slots[slot].arena;

	sam3_arena_reset(&proc->scratch_arena);

	dims[0] = 3;
	dims[1] = height;
	dims[2] = width;
	image = gh_alloc_tensor(&proc->scratch_arena, SAM3_DTYPE_F32,
				3, dims);
	if (!image) {
		err = SAM3_ENOMEM;
		goto restore;
	}
	SAM3_PROF_BEGIN(proc->profiler, "image_normalize");
	sam3_normalize_rgb_chw(pixels, (float *)image->data, width, height);
	SAM3_PROF_END(proc->profiler, "image_normalize");

	SAM3_PROF_BEGIN(proc->profiler, "image_encode");
	err = sam3_image_model_encode(&proc->model, proc->backend, image,
				      &proc->scratch_arena, persist,
				      proc->profiler);
	SAM3_PROF_END(proc->profiler, "image_encode");
	if (err != SAM3_OK)
		goto restore;

	struct sam3_image_bundle bundle = {0};
	bundle.image_features = m->cached_image_features;
	bundle.feat_s0_nhwc   = m->cached_feat_s0_nhwc;
	bundle.feat_s1_nhwc   = m->cached_feat_s1_nhwc;
	bundle.feat_4x_nhwc   = m->cached_feat_4x_nhwc;
	bundle.sam2_05x_nhwc  = m->cached_sam2_05x_nhwc;
	bundle.sam2_1x_nhwc   = m->cached_sam2_1x_nhwc;
	bundle.sam2_2x_nhwc   = m->cached_sam2_2x_nhwc;
	bundle.sam2_4x_nhwc   = m->cached_sam2_4x_nhwc;
	bundle.prompt_w = width;
	bundle.prompt_h = height;
	bundle.width = width;
	bundle.height = height;
	sam3_image_cache_register(proc->img_cache, slot, key,
				  pixels, pref_len, &bundle);
	err = SAM3_OK;

restore:
	m->cached_image_features = sv_feat;
	m->cached_feat_s0_nhwc   = sv_s0;
	m->cached_feat_s1_nhwc   = sv_s1;
	m->cached_feat_4x_nhwc   = sv_4x;
	m->cached_sam2_05x_nhwc  = sv_s2_05;
	m->cached_sam2_1x_nhwc   = sv_s2_1;
	m->cached_sam2_2x_nhwc   = sv_s2_2;
	m->cached_sam2_4x_nhwc   = sv_s2_4;
	m->image_encoded = sv_encoded;
	proc->image_loaded = sv_loaded;
	proc->current_img_slot = sv_slot;
	proc->prompt_w = sv_prompt_w;
	proc->prompt_h = sv_prompt_h;
	return err;
}

/*
 * sam3_processor_precache_text - Synchronously encode @text into the
 * text cache without touching the caller's pending-prompt state.
 *
 * On hit: no-op. On miss: tokenize, claim a slot, run the text
 * encoder inline on proc->text_backend (no worker thread), and
 * register the bundle. The proc's text_tokens / text_n_tokens /
 * text_worker_slot / text_cached_bundle are saved and restored so a
 * prior set_text()'s pending state survives the call intact.
 */
enum sam3_error sam3_processor_precache_text(struct sam3_processor *proc,
					     const char *text)
{
	if (!proc || !text)
		return SAM3_EINVAL;
	if (!proc->text_backend) {
		sam3_log_error("precache_text: text_backend unavailable");
		return SAM3_EBACKEND;
	}

	/* Wait for any in-flight worker so we don't race on text_tokens /
	 * text_worker_slot. */
	join_text_worker(proc);

	/* Save caller-visible state. */
	int32_t saved_tokens[SAM3_PROCESSOR_MAX_TOKENS];
	memcpy(saved_tokens, proc->text_tokens, sizeof(saved_tokens));
	int saved_n_tokens = proc->text_n_tokens;
	int saved_worker_slot = proc->text_worker_slot;
	struct sam3_text_bundle *saved_bundle = proc->text_cached_bundle;

	struct sam3_text_encoder_iface *te_iface =
		&proc->model.backbone.text_iface;
	int ctx_len = te_iface->ctx_len;

	int n_tokens = sam3_tokenizer_encode(
		&proc->model.backbone.tokenizer, text,
		proc->text_tokens, ctx_len);
	if (n_tokens <= 0) {
		sam3_log_error("precache_text: tokenize failed");
		memcpy(proc->text_tokens, saved_tokens, sizeof(saved_tokens));
		proc->text_n_tokens = saved_n_tokens;
		return SAM3_EINVAL;
	}
	proc->text_n_tokens = n_tokens;

	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)proc->text_tokens,
			    (size_t)n_tokens * sizeof(int32_t), key);
	if (key == 0)
		key = 1;

	int hit = sam3_text_cache_lookup(proc->txt_cache, key,
					 proc->text_tokens, n_tokens);
	if (hit >= 0) {
		memcpy(proc->text_tokens, saved_tokens, sizeof(saved_tokens));
		proc->text_n_tokens = saved_n_tokens;
		proc->text_worker_slot = saved_worker_slot;
		proc->text_cached_bundle = saved_bundle;
		return SAM3_OK;
	}

	int slot = sam3_text_cache_claim_slot(proc->txt_cache);
	if (slot < 0) {
		memcpy(proc->text_tokens, saved_tokens, sizeof(saved_tokens));
		proc->text_n_tokens = saved_n_tokens;
		return SAM3_ENOMEM;
	}
	proc->text_worker_slot = slot;

	sam3_arena_reset(&proc->text_scratch_arena);
	proc->text_thread_err = SAM3_OK;

	/* Synchronous reuse of the worker entry point — no thread. */
	(void)text_worker_main(proc);
	enum sam3_error err = proc->text_thread_err;
	proc->text_thread_err = SAM3_OK;

	/* Restore caller-visible state regardless of success. */
	memcpy(proc->text_tokens, saved_tokens, sizeof(saved_tokens));
	proc->text_n_tokens = saved_n_tokens;
	proc->text_worker_slot = saved_worker_slot;
	proc->text_cached_bundle = saved_bundle;
	return err;
}

/*
 * segment_one - Shared body for sam3_processor_segment and
 * sam3_processor_segment_batch. Runs text encode + geometry project +
 * full mask pipeline for a single prompt set, writing mask/score/box
 * data into @result.
 *
 * @use_async_text: 1 to let the async text paths (hit on
 *                  proc->text_cached_bundle, or join of an in-flight
 *                  worker that set_text spawned) participate. 0 to
 *                  always take the inline text encode path for this
 *                  call — used by the batch entry point, which drops
 *                  pending async state up front so each set's text is
 *                  encoded synchronously.
 */
static enum sam3_error segment_one(struct sam3_processor *proc,
				   const struct sam3_prompt *prompts,
				   int n_prompts,
				   struct sam3_result *result,
				   int use_async_text)
{
	struct sam3_tensor *prompt_tokens = NULL;
	struct sam3_tensor *text_features = NULL;
	struct sam3_tensor *mask_logits;
	struct sam3_tensor *score_logits = NULL;
	enum sam3_error err;
	size_t mask_bytes, persist_save;
	int nelems;
	const char *text;

	if (!proc || !prompts || n_prompts <= 0 || !result)
		return SAM3_EINVAL;

	if (!proc->image_loaded)
		return SAM3_EINVAL;

	memset(result, 0, sizeof(*result));

	/*
	 * Save model arena offset so we can roll back inter-stage
	 * data after segmentation completes.
	 */
	persist_save = proc->model_arena.offset;

	/*
	 * Stage A: Encode text prompt. Build the text encoder graph,
	 * evaluate it, and copy the result to the model arena so it
	 * survives scratch resets between segmentation stages.
	 */
	text = find_text_prompt(prompts, n_prompts);

	/*
	 * When only geometric prompts are given (no text), inject
	 * the dummy text "visual" — the Python reference model does
	 * this automatically in Sam3Processor.add_geometric_prompt().
	 * The DETR encoder/decoder require text features as context.
	 */
	if (!text) {
		int n_pts, n_bxs;
		count_prompts_by_type(prompts, n_prompts,
				      &n_pts, &n_bxs);
		if (n_pts > 0 || n_bxs > 0) {
			text = "visual";
			sam3_log_info("segment: injecting dummy text "
				      "\"visual\" for geometric prompts");
		}
	}

	if (text) {
		SAM3_PROF_BEGIN(proc->profiler, "text_encode");

		if (use_async_text && proc->text_cached_bundle) {
			/*
			 * Hit path: set_text found the bundle already in
			 * txt_cache. Read features directly from the cache
			 * slot — no copy into model_arena, no worker.
			 */
			text_features = proc->text_cached_bundle->features;
			proc->text_cached_bundle = NULL;
			SAM3_PROF_END(proc->profiler, "text_encode");
		} else if (use_async_text && proc->text_thread_active) {
			/*
			 * Miss path: set_text spawned a worker. Join it,
			 * then pick up the features from the slot it wrote
			 * into. Features live in the cache slot's arena —
			 * no copy into model_arena.
			 */
			join_text_worker(proc);
			if (proc->text_thread_err != SAM3_OK) {
				sam3_log_error("segment: text worker "
					       "failed: %d",
					       proc->text_thread_err);
				err = proc->text_thread_err;
				proc->text_thread_err = SAM3_OK;
				goto fail;
			}
			int s = proc->text_worker_slot;
			if (s < 0) {
				err = SAM3_EBACKEND;
				goto fail;
			}
			text_features =
				proc->txt_cache->slots[s].bundle.features;
			proc->text_worker_slot = -1;
			SAM3_PROF_END(proc->profiler, "text_encode");
		} else {
			/*
			 * Legacy inline path — runs the text encoder on
			 * the main backend with proc->scratch_arena and
			 * proc->model_arena. Used when set_text() was not
			 * called before segment().
			 */
			struct sam3_text_encoder_iface *te_iface =
				&proc->model.backbone.text_iface;
			int ctx = te_iface->ctx_len;
			int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
			int n_tokens;
			struct sam3_tensor *tok_tensor;
			int tok_dims[1];

			SAM3_PROF_BEGIN(proc->profiler, "tokenize");
			n_tokens = sam3_tokenizer_encode(
				&proc->model.backbone.tokenizer, text,
				tokens, ctx);
			SAM3_PROF_END(proc->profiler, "tokenize");
			if (n_tokens <= 0) {
				sam3_log_error("segment: tokenize failed");
				err = SAM3_EINVAL;
				goto fail;
			}

			tok_dims[0] = ctx;
			sam3_arena_reset(&proc->scratch_arena);
			tok_tensor = gh_alloc_tensor(&proc->model_arena,
						      SAM3_DTYPE_I32, 1,
						      tok_dims);
			if (!tok_tensor) {
				err = SAM3_ENOMEM;
				goto fail;
			}
			memcpy(tok_tensor->data, tokens,
			       (size_t)ctx * sizeof(int32_t));

			SAM3_PROF_BEGIN(proc->profiler, "text_blocks");
			text_features = te_iface->ops->build_perblock(
				te_iface, proc->backend, tok_tensor,
				&proc->scratch_arena, &proc->model_arena);
			SAM3_PROF_END(proc->profiler, "text_blocks");
			if (!text_features) {
				sam3_log_error("segment: text encode "
					       "perblock failed");
				err = SAM3_ENOMEM;
				goto fail;
			}

			sam3_log_debug("segment: text encoded (perblock), "
				       "%d×%d",
				       text_features->dims[0],
				       text_features->dims[1]);

			/*
			 * Truncate text features to real tokens only.
			 *
			 * Python creates text_attention_mask = (tokenized
			 * != 0) and passes it as key_padding_mask to
			 * encoder cross-attn, masking out padding tokens.
			 * We achieve the same effect by dropping padding
			 * tokens entirely — softmax over non-masked
			 * tokens is identical to softmax over the
			 * truncated set.
			 */
			if (text_features->dims[0] > n_tokens) {
				int d = text_features->dims[1];
				int trunc_dims[] = {n_tokens, d};
				struct sam3_tensor *trunc;

				trunc = gh_alloc_tensor(&proc->model_arena,
							 SAM3_DTYPE_F32, 2,
							 trunc_dims);
				if (!trunc) {
					err = SAM3_ENOMEM;
					goto fail;
				}
				memcpy(trunc->data, text_features->data,
				       (size_t)n_tokens * (size_t)d *
				       sizeof(float));
				sam3_log_info("segment: truncated text "
					      "%d→%d tokens (dropped %d "
					      "padding)",
					      text_features->dims[0],
					      n_tokens,
					      text_features->dims[0] -
						      n_tokens);
				text_features = trunc;
			}
			SAM3_PROF_END(proc->profiler, "text_encode");
		}
	}

	/*
	 * Stage B: Project geometric prompt coordinates. Build a
	 * small graph, evaluate, copy result to model arena.
	 */
	{
		int n_points, n_boxes;

		count_prompts_by_type(prompts, n_prompts,
				      &n_points, &n_boxes);
		if (n_points > 0 || n_boxes > 0) {
			/*
			 * Project prompts on CPU: normalize coords to
			 * [0,1], linear projection, add label embedding.
			 * No graph/Metal needed for tiny prompt tensors.
			 */
			SAM3_PROF_BEGIN(proc->profiler, "prompt_project");
			prompt_tokens = sam3_project_prompts(
				&proc->model,
				proc->model.cached_feat_s1_nhwc,
				prompts, n_prompts,
				proc->prompt_w, proc->prompt_h,
				&proc->model_arena);
			SAM3_PROF_END(proc->profiler, "prompt_project");
			if (!prompt_tokens) {
				err = SAM3_ENOMEM;
				goto fail;
			}
		}
	}

	/* At least one of text or geometry must be present */
	if (!prompt_tokens && !text_features) {
		err = SAM3_EINVAL;
		goto fail;
	}

	/*
	 * Run per-stage segmentation: geometry encoder, encoder
	 * fusion, decoder, segmentation head — each evaluated
	 * independently with scratch reset between stages.
	 */
	SAM3_PROF_BEGIN(proc->profiler, "mask_decode");
	err = sam3_image_model_segment(&proc->model, proc->backend,
				       proc->text_backend,
				       prompt_tokens, text_features,
				       &proc->scratch_arena,
				       &proc->model_arena,
				       &mask_logits, &score_logits,
				       proc->profiler);
	SAM3_PROF_END(proc->profiler, "mask_decode");
	if (err != SAM3_OK) {
		sam3_log_error("segment: pipeline failed: %d", err);
		goto fail;
	}

	/*
	 * Copy mask logits to result. The mask tensor is expected to
	 * be [n_masks, H, W] from the segmentation head.
	 */
	if (mask_logits->n_dims == 3) {
		result->n_masks = mask_logits->dims[0];
		result->mask_height = mask_logits->dims[1];
		result->mask_width = mask_logits->dims[2];
	} else if (mask_logits->n_dims == 2) {
		result->n_masks = 1;
		result->mask_height = mask_logits->dims[0];
		result->mask_width = mask_logits->dims[1];
	} else {
		err = SAM3_EINVAL;
		goto fail;
	}

	nelems = result->n_masks * result->mask_height * result->mask_width;
	mask_bytes = (size_t)nelems * sizeof(float);

	result->masks = malloc(mask_bytes);
	if (!result->masks) {
		err = SAM3_ENOMEM;
		goto fail;
	}

	memcpy(result->masks, mask_logits->data, mask_bytes);

	/* Allocate IoU scores (one per mask), zero-initialized */
	result->iou_scores = calloc((size_t)result->n_masks,
				     sizeof(float));
	if (!result->iou_scores) {
		free(result->masks);
		result->masks = NULL;
		err = SAM3_ENOMEM;
		goto fail;
	}

	/*
	 * Copy scorer output into iou_scores. The scorer produces
	 * raw logits [n_queries, 1]; apply sigmoid to get [0,1]
	 * probabilities for NMS prefiltering.
	 */
	if (score_logits) {
		int n_scores = score_logits->dims[0];
		int n_copy = n_scores < result->n_masks
			     ? n_scores : result->n_masks;
		const float *sdata = (const float *)score_logits->data;

		for (int i = 0; i < n_copy; i++)
			result->iou_scores[i] = 1.0f /
				(1.0f + expf(-sdata[i]));

		result->iou_valid = 1;
		sam3_log_info("segment: %d IoU scores computed", n_copy);
	} else {
		result->iou_valid = 0;
		sam3_log_warn("segment: IoU scores unavailable "
			      "(no text features for scorer)");
	}

	/*
	 * Post-processing: stability-based mask selection.
	 * When the mask decoder produces a small number of masks
	 * (typically 4), apply the stability algorithm to select
	 * the best one. Segmentation head output (200 queries)
	 * is filtered by NMS in the caller instead.
	 */
	SAM3_PROF_BEGIN(proc->profiler, "postprocess");
	result->best_mask = -1;
	if (result->n_masks > 1 &&
	    result->n_masks <= SAM3_MASK_DEC_MASKS &&
	    result->iou_valid) {
		result->best_mask = sam3_mask_select_best(
			result->masks, result->iou_scores,
			result->n_masks,
			result->mask_height, result->mask_width,
			SAM3_STABILITY_DELTA,
			SAM3_STABILITY_THRESH);
		if (result->best_mask >= 0)
			sam3_log_info("segment: stability selected "
				      "mask %d (of %d)",
				      result->best_mask,
				      result->n_masks);
	}

	/*
	 * Post-processing: extract bounding boxes from masks.
	 * Boxes are in xyxy format with exclusive upper bounds.
	 */
	result->boxes = calloc((size_t)result->n_masks * 4,
			       sizeof(float));
	if (result->boxes) {
		sam3_masks_to_boxes(result->masks, result->n_masks,
				    result->mask_height,
				    result->mask_width,
				    result->boxes);
		result->boxes_valid = 1;
	} else {
		result->boxes_valid = 0;
		sam3_log_warn("segment: box extraction skipped "
			      "(alloc failed)");
	}

	SAM3_PROF_END(proc->profiler, "postprocess");

	/* Roll back inter-stage persist data and invalidate stale
	 * backend cache entries for the freed arena region. */
	{
		size_t end_off = proc->model_arena.offset;
		proc->model_arena.offset = persist_save;
		if (end_off > persist_save &&
		    proc->backend->ops->cache_invalidate) {
			char *base = (char *)proc->model_arena.base;
			proc->backend->ops->cache_invalidate(
				proc->backend,
				base + persist_save,
				end_off - persist_save);
		}
	}
	return SAM3_OK;

fail:
	{
		size_t end_off = proc->model_arena.offset;
		proc->model_arena.offset = persist_save;
		if (end_off > persist_save &&
		    proc->backend->ops->cache_invalidate) {
			char *base = (char *)proc->model_arena.base;
			proc->backend->ops->cache_invalidate(
				proc->backend,
				base + persist_save,
				end_off - persist_save);
		}
	}
	return err;
}

enum sam3_error sam3_processor_segment(struct sam3_processor *proc,
				       const struct sam3_prompt *prompts,
				       int n_prompts,
				       struct sam3_result *result)
{
	return segment_one(proc, prompts, n_prompts, result,
			   /* use_async_text= */ 1);
}

/*
 * prepare_set_inputs - Produce prompt_tokens and text_features for one
 * prompt set without running the image model. The tensors are
 * persist-allocated (proc->model_arena) so they survive across sets in
 * a batch.
 *
 * Either @out_prompt or @out_text may be written as NULL: NULL means
 * "this set has no geometric (or text) prompts". The existing inline
 * text path in segment_one injects a dummy "visual" text for pure
 * geometric sets; this helper mirrors that behavior.
 *
 * Returns SAM3_OK on success. On failure the caller is expected to
 * roll back proc->model_arena.
 */
static enum sam3_error prepare_set_inputs(
	struct sam3_processor *proc,
	const struct sam3_prompt *prompts, int n_prompts,
	struct sam3_tensor **out_prompt,
	struct sam3_tensor **out_text)
{
	struct sam3_tensor *prompt_tokens = NULL;
	struct sam3_tensor *text_features = NULL;
	const char *text;
	int n_points, n_boxes;

	*out_prompt = NULL;
	*out_text = NULL;

	text = find_text_prompt(prompts, n_prompts);
	if (!text) {
		count_prompts_by_type(prompts, n_prompts,
				      &n_points, &n_boxes);
		if (n_points > 0 || n_boxes > 0) {
			text = "visual";
			sam3_log_info("segment_batch: injecting dummy text "
				      "\"visual\" for geometric prompts");
		}
	}

	/* Stage A: Text encode (inline, on main backend). */
	if (text) {
		struct sam3_text_encoder_iface *te_iface =
			&proc->model.backbone.text_iface;
		int ctx = te_iface->ctx_len;
		int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
		int n_tokens;
		struct sam3_tensor *tok_tensor;
		int tok_dims[1];

		SAM3_PROF_BEGIN(proc->profiler, "tokenize");
		n_tokens = sam3_tokenizer_encode(
			&proc->model.backbone.tokenizer, text,
			tokens, ctx);
		SAM3_PROF_END(proc->profiler, "tokenize");
		if (n_tokens <= 0) {
			sam3_log_error("segment_batch: tokenize failed");
			return SAM3_EINVAL;
		}

		tok_dims[0] = ctx;
		sam3_arena_reset(&proc->scratch_arena);
		tok_tensor = gh_alloc_tensor(&proc->model_arena,
					      SAM3_DTYPE_I32, 1, tok_dims);
		if (!tok_tensor)
			return SAM3_ENOMEM;
		memcpy(tok_tensor->data, tokens,
		       (size_t)ctx * sizeof(int32_t));

		SAM3_PROF_BEGIN(proc->profiler, "text_blocks");
		text_features = te_iface->ops->build_perblock(
			te_iface, proc->backend, tok_tensor,
			&proc->scratch_arena, &proc->model_arena);
		SAM3_PROF_END(proc->profiler, "text_blocks");
		if (!text_features) {
			sam3_log_error("segment_batch: text encode "
				       "perblock failed");
			return SAM3_ENOMEM;
		}

		/*
		 * Truncate to real (non-padding) tokens. Mirrors segment_one:
		 * softmax over non-masked tokens equals softmax over truncated
		 * set, so dropping padding is safe and keeps shapes tight.
		 */
		if (text_features->dims[0] > n_tokens) {
			int d = text_features->dims[1];
			int trunc_dims[2] = {n_tokens, d};
			struct sam3_tensor *trunc;

			trunc = gh_alloc_tensor(&proc->model_arena,
						 SAM3_DTYPE_F32, 2,
						 trunc_dims);
			if (!trunc)
				return SAM3_ENOMEM;
			memcpy(trunc->data, text_features->data,
			       (size_t)n_tokens * (size_t)d * sizeof(float));
			text_features = trunc;
		}
	}

	/* Stage B: Project geometric prompts (CPU, on model_arena). */
	count_prompts_by_type(prompts, n_prompts, &n_points, &n_boxes);
	if (n_points > 0 || n_boxes > 0) {
		SAM3_PROF_BEGIN(proc->profiler, "prompt_project");
		prompt_tokens = sam3_project_prompts(
			&proc->model,
			proc->model.cached_feat_s1_nhwc,
			prompts, n_prompts,
			proc->prompt_w, proc->prompt_h,
			&proc->model_arena);
		SAM3_PROF_END(proc->profiler, "prompt_project");
		if (!prompt_tokens)
			return SAM3_ENOMEM;
	}

	if (!prompt_tokens && !text_features)
		return SAM3_EINVAL;

	*out_prompt = prompt_tokens;
	*out_text = text_features;
	return SAM3_OK;
}

/*
 * extract_result_slot - Populate results[i] from the i-th slot of the
 * stacked mask/score tensors produced by sam3_image_model_segment_batched.
 *
 * Mirrors the postprocessing tail of segment_one (mask malloc, iou sigmoid,
 * stability mask selection, masks-to-boxes). The only difference is where
 * we read the per-slot data from.
 */
static enum sam3_error extract_result_slot(
	struct sam3_result *result,
	const struct sam3_tensor *batched_masks,
	const struct sam3_tensor *batched_scores,
	int slot_idx)
{
	int nq = batched_masks->dims[1];
	int H = batched_masks->dims[2];
	int W = batched_masks->dims[3];
	size_t mask_bytes = (size_t)nq * (size_t)H * (size_t)W *
			    sizeof(float);
	const float *mask_src = (const float *)batched_masks->data +
				 (size_t)slot_idx * nq * H * W;

	memset(result, 0, sizeof(*result));
	result->n_masks = nq;
	result->mask_height = H;
	result->mask_width = W;

	result->masks = malloc(mask_bytes);
	if (!result->masks)
		return SAM3_ENOMEM;
	memcpy(result->masks, mask_src, mask_bytes);

	result->iou_scores = calloc((size_t)nq, sizeof(float));
	if (!result->iou_scores) {
		free(result->masks);
		result->masks = NULL;
		return SAM3_ENOMEM;
	}

	if (batched_scores) {
		const float *sdata = (const float *)batched_scores->data +
				      (size_t)slot_idx * nq;
		for (int i = 0; i < nq; i++)
			result->iou_scores[i] = 1.0f /
				(1.0f + expf(-sdata[i]));
		result->iou_valid = 1;
	} else {
		result->iou_valid = 0;
	}

	result->best_mask = -1;
	if (result->n_masks > 1 &&
	    result->n_masks <= SAM3_MASK_DEC_MASKS &&
	    result->iou_valid) {
		result->best_mask = sam3_mask_select_best(
			result->masks, result->iou_scores,
			result->n_masks,
			result->mask_height, result->mask_width,
			SAM3_STABILITY_DELTA,
			SAM3_STABILITY_THRESH);
	}

	result->boxes = calloc((size_t)result->n_masks * 4, sizeof(float));
	if (result->boxes) {
		sam3_masks_to_boxes(result->masks, result->n_masks,
				    result->mask_height, result->mask_width,
				    result->boxes);
		result->boxes_valid = 1;
	} else {
		result->boxes_valid = 0;
	}

	return SAM3_OK;
}

enum sam3_error sam3_processor_segment_batch(
	struct sam3_processor *proc,
	const struct sam3_prompt_set *sets,
	int n_sets,
	struct sam3_result *results)
{
	int i, j;
	enum sam3_error err;
	size_t persist_save;
	struct sam3_tensor **per_set_prompt = NULL;
	struct sam3_tensor **per_set_text = NULL;
	struct sam3_tensor *prompt_stacked = NULL;
	struct sam3_tensor *text_stacked = NULL;
	struct sam3_tensor *batched_masks = NULL;
	struct sam3_tensor *batched_scores = NULL;
	int all_prompt_compat = 1, all_text_compat = 1;
	int ref_prompt_len = -1, ref_text_len = -1;
	int ref_text_d = -1;
	int d_model;

	if (!proc || !sets || n_sets <= 0 || !results)
		return SAM3_EINVAL;

	if (!proc->image_loaded)
		return SAM3_EINVAL;

	/*
	 * Zero the full results array up front so the callee always
	 * overwrites every slot — even on early validation failure.
	 * Then validate every set before running any segment; we do not
	 * want a half-processed batch on malformed input.
	 */
	memset(results, 0, (size_t)n_sets * sizeof(results[0]));
	for (i = 0; i < n_sets; i++) {
		if (!sets[i].prompts || sets[i].n_prompts <= 0)
			return SAM3_EINVAL;
	}

	/*
	 * Drop any pending async text state. Batch always takes the
	 * inline text-encode path per set, so a leftover worker or
	 * cached-bundle pointer from a prior set_text would only cause
	 * confusion. The user is expected to rely on sam3_precache_text
	 * for text reuse across sets.
	 */
	if (proc->text_thread_active)
		join_text_worker(proc);
	proc->text_cached_bundle = NULL;
	proc->text_worker_slot = -1;

	/* Single-set shortcut: go straight to segment_one — the batched
	 * driver at B=1 would just do extra memcpys. segment_one is the
	 * byte-exact parity reference. */
	if (n_sets == 1) {
		err = segment_one(proc, sets[0].prompts, sets[0].n_prompts,
				  &results[0], /* use_async_text= */ 0);
		if (err != SAM3_OK) {
			memset(results, 0, sizeof(results[0]));
			return err;
		}
		sam3_log_info("segment_batch: 1 set completed");
		return SAM3_OK;
	}

	d_model = proc->model.decoder.d_model;
	persist_save = proc->model_arena.offset;

	/*
	 * Phase 1: per-set prompt + text encoding. Collect tensors and
	 * detect shape compatibility (same N+1, same trimmed seq).
	 */
	per_set_prompt = sam3_arena_alloc(&proc->model_arena,
					   (size_t)n_sets *
						   sizeof(*per_set_prompt));
	per_set_text = sam3_arena_alloc(&proc->model_arena,
					 (size_t)n_sets *
						 sizeof(*per_set_text));
	if (!per_set_prompt || !per_set_text) {
		err = SAM3_ENOMEM;
		goto fail;
	}
	memset(per_set_prompt, 0, (size_t)n_sets * sizeof(*per_set_prompt));
	memset(per_set_text, 0, (size_t)n_sets * sizeof(*per_set_text));

	for (i = 0; i < n_sets; i++) {
		err = prepare_set_inputs(proc,
					 sets[i].prompts, sets[i].n_prompts,
					 &per_set_prompt[i],
					 &per_set_text[i]);
		if (err != SAM3_OK) {
			sam3_log_error("segment_batch: prepare set %d/%d "
				       "failed: %d", i + 1, n_sets, err);
			goto fail;
		}

		/* Shape compat: prompt_tokens are [N+1, d]. */
		if (per_set_prompt[i]) {
			int len = per_set_prompt[i]->dims[0];
			if (ref_prompt_len < 0) {
				ref_prompt_len = len;
				/* If first present set is not slot 0, earlier
				 * slots were NULL — mixed present/absent. */
				if (i > 0)
					all_prompt_compat = 0;
			} else if (ref_prompt_len != len) {
				all_prompt_compat = 0;
			}
		} else if (ref_prompt_len >= 0) {
			all_prompt_compat = 0;  /* mixed present/absent */
		}

		/* Shape compat: text_features are [seq, d_text]. */
		if (per_set_text[i]) {
			int len = per_set_text[i]->dims[0];
			int d = per_set_text[i]->dims[1];
			if (ref_text_len < 0) {
				ref_text_len = len;
				ref_text_d = d;
				if (i > 0)
					all_text_compat = 0;
			} else if (ref_text_len != len || ref_text_d != d) {
				all_text_compat = 0;
			}
		} else if (ref_text_len >= 0) {
			all_text_compat = 0;
		}
	}

	/*
	 * Phase 2: dispatch. If shapes don't line up across sets, fall
	 * back to the serial per-set loop. The serial path is the same
	 * reference used by test_segment_batch_parity.
	 */
	if (!all_prompt_compat || !all_text_compat) {
		sam3_log_info("segment_batch: serial fallback "
			      "(n_sets=%d prompt_compat=%d text_compat=%d)",
			      n_sets, all_prompt_compat, all_text_compat);

		/* Roll back prepare_set_inputs allocations; segment_one
		 * will redo them cleanly per set. */
		proc->model_arena.offset = persist_save;

		for (i = 0; i < n_sets; i++) {
			err = segment_one(proc, sets[i].prompts,
					  sets[i].n_prompts, &results[i],
					  /* use_async_text= */ 0);
			if (err != SAM3_OK) {
				sam3_log_error("segment_batch: set %d/%d "
					       "failed: %d",
					       i + 1, n_sets, err);
				for (j = 0; j < i; j++)
					sam3_result_free(&results[j]);
				memset(results, 0,
				       (size_t)n_sets * sizeof(results[0]));
				return err;
			}
		}
		sam3_log_info("segment_batch: %d sets completed (serial)",
			      n_sets);
		return SAM3_OK;
	}

	/*
	 * Phase 3: stack per-set tensors along a new leading B dim.
	 */
	if (ref_prompt_len >= 0) {
		int dims[3] = {n_sets, ref_prompt_len, d_model};
		prompt_stacked = gh_alloc_tensor(&proc->model_arena,
						  SAM3_DTYPE_F32, 3, dims);
		if (!prompt_stacked) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		size_t per_slot_bytes = (size_t)ref_prompt_len *
					 (size_t)d_model * sizeof(float);
		for (i = 0; i < n_sets; i++)
			memcpy((char *)prompt_stacked->data +
				(size_t)i * per_slot_bytes,
			       per_set_prompt[i]->data, per_slot_bytes);
	}

	if (ref_text_len >= 0) {
		int dims[3] = {n_sets, ref_text_len, ref_text_d};
		text_stacked = gh_alloc_tensor(&proc->model_arena,
						SAM3_DTYPE_F32, 3, dims);
		if (!text_stacked) {
			err = SAM3_ENOMEM;
			goto fail;
		}
		size_t per_slot_bytes = (size_t)ref_text_len *
					 (size_t)ref_text_d * sizeof(float);
		for (i = 0; i < n_sets; i++)
			memcpy((char *)text_stacked->data +
				(size_t)i * per_slot_bytes,
			       per_set_text[i]->data, per_slot_bytes);
	}

	/*
	 * Phase 4: single batched pipeline call.
	 */
	SAM3_PROF_BEGIN(proc->profiler, "mask_decode");
	err = sam3_image_model_segment_batched(
		&proc->model, proc->backend, proc->text_backend,
		prompt_stacked, text_stacked, n_sets,
		&proc->scratch_arena, &proc->model_arena,
		&batched_masks, &batched_scores, proc->profiler);
	SAM3_PROF_END(proc->profiler, "mask_decode");
	if (err != SAM3_OK) {
		sam3_log_error("segment_batch: batched pipeline failed: %d",
			       err);
		goto fail;
	}

	/*
	 * Phase 5: unpack per-set results from the stacked outputs.
	 */
	SAM3_PROF_BEGIN(proc->profiler, "postprocess");
	for (i = 0; i < n_sets; i++) {
		err = extract_result_slot(&results[i], batched_masks,
					   batched_scores, i);
		if (err != SAM3_OK) {
			for (j = 0; j < i; j++)
				sam3_result_free(&results[j]);
			memset(results, 0,
			       (size_t)n_sets * sizeof(results[0]));
			SAM3_PROF_END(proc->profiler, "postprocess");
			goto fail;
		}
	}
	SAM3_PROF_END(proc->profiler, "postprocess");

	/* Phase 6: roll back inter-stage persist data and invalidate stale
	 * backend cache entries for the freed arena region. */
	{
		size_t end_off = proc->model_arena.offset;
		proc->model_arena.offset = persist_save;
		if (end_off > persist_save &&
		    proc->backend->ops->cache_invalidate) {
			char *base = (char *)proc->model_arena.base;
			proc->backend->ops->cache_invalidate(
				proc->backend,
				base + persist_save,
				end_off - persist_save);
		}
	}

	sam3_log_info("segment_batch: %d sets completed (batched)", n_sets);
	return SAM3_OK;

fail:
	{
		size_t end_off = proc->model_arena.offset;
		proc->model_arena.offset = persist_save;
		if (end_off > persist_save &&
		    proc->backend->ops->cache_invalidate) {
			char *base = (char *)proc->model_arena.base;
			proc->backend->ops->cache_invalidate(
				proc->backend,
				base + persist_save,
				end_off - persist_save);
		}
	}
	memset(results, 0, (size_t)n_sets * sizeof(results[0]));
	return err;
}

void sam3_processor_cache_clear(struct sam3_processor *proc, unsigned which)
{
	if (!proc)
		return;
	if (which == 0 || (which & 1u)) {
		sam3_image_cache_clear(proc->img_cache);
		/*
		 * The slot arenas were just reset — any live
		 * model.cached_* pointers now reference freed storage.
		 * Zero them (and image_loaded / current_img_slot) so a
		 * subsequent segment() before the next set_image fails
		 * predictably rather than reading poisoned memory.
		 */
		reset_image_cached_state(proc);
	}
	if (which == 0 || (which & 2u)) {
		join_text_worker(proc);
		sam3_text_cache_clear(proc->txt_cache);
		proc->text_cached_bundle = NULL;
		proc->text_worker_slot = -1;
	}
}

void sam3_processor_cache_stats(const struct sam3_processor *proc,
				struct sam3_cache_stats *out)
{
	if (!out)
		return;
	memset(out, 0, sizeof(*out));
	if (!proc)
		return;
	struct sam3_cache_stats img = {0}, txt = {0};
	sam3_image_cache_stats(proc->img_cache, &img);
	sam3_text_cache_stats(proc->txt_cache, &txt);
	out->image_hits = img.image_hits;
	out->image_misses = img.image_misses;
	out->image_evictions = img.image_evictions;
	out->text_hits = txt.text_hits;
	out->text_misses = txt.text_misses;
	out->text_evictions = txt.text_evictions;
}
