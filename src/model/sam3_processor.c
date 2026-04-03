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

#include <stdlib.h>
#include <string.h>

#include "sam3_processor.h"
#include "graph_helpers.h"
#include "core/weight.h"
#include "backend/cpu/cpu_backend.h"

enum sam3_error sam3_processor_init(struct sam3_processor *proc)
{
	struct sam3_cpu_backend *cpu;
	enum sam3_error err;

	memset(proc, 0, sizeof(*proc));

	/* Create CPU backend with 512 MiB arena for graph evaluation */
	cpu = calloc(1, sizeof(*cpu));
	if (!cpu)
		return SAM3_ENOMEM;

	cpu->base.type = SAM3_BACKEND_CPU;
	cpu->base.ops = sam3_cpu_backend_ops();
	cpu->arena_capacity = 512UL * 1024 * 1024;

	err = cpu->base.ops->init(&cpu->base);
	if (err != SAM3_OK) {
		free(cpu);
		return err;
	}
	proc->backend = &cpu->base;

	/* Model arena: 256 MiB for weights and cached features */
	err = sam3_arena_init(&proc->model_arena, 256UL * 1024 * 1024);
	if (err != SAM3_OK)
		goto cleanup_backend;

	/* Scratch arena: 128 MiB for per-inference temporaries */
	err = sam3_arena_init(&proc->scratch_arena, 128UL * 1024 * 1024);
	if (err != SAM3_OK)
		goto cleanup_model_arena;

	/* Initialize all sub-modules with default SAM3 config */
	err = sam3_image_model_init(&proc->model, &proc->model_arena);
	if (err != SAM3_OK)
		goto cleanup_scratch_arena;

	return SAM3_OK;

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
				    const char *model_path,
				    const char *vocab_path)
{
	struct sam3_weight_file wf;
	enum sam3_error err;

	memset(&wf, 0, sizeof(wf));

	err = sam3_weight_open(&wf, model_path);
	if (err != SAM3_OK)
		return err;

	err = sam3_image_model_load(&proc->model, &wf, vocab_path,
				    &proc->model_arena);
	sam3_weight_close(&wf);

	return err;
}

void sam3_processor_free(struct sam3_processor *proc)
{
	if (!proc)
		return;

	sam3_image_model_free(&proc->model);

	if (proc->backend) {
		proc->backend->ops->free(proc->backend);
		free(proc->backend);
		proc->backend = NULL;
	}

	sam3_arena_free(&proc->model_arena);
	sam3_arena_free(&proc->scratch_arena);
}

enum sam3_error sam3_processor_set_image(struct sam3_processor *proc,
					 const uint8_t *pixels,
					 int width, int height)
{
	struct sam3_tensor *image;
	struct sam3_graph graph;
	enum sam3_error err;
	int dims[3];
	int c, y, x;
	float *dst;

	if (!proc || !pixels || width <= 0 || height <= 0)
		return SAM3_EINVAL;

	/* Reset scratch arena for this encode pass */
	sam3_arena_reset(&proc->scratch_arena);

	/*
	 * Allocate image tensor [3, height, width] and normalize
	 * uint8 RGB interleaved pixels to float CHW planar layout.
	 */
	dims[0] = 3;
	dims[1] = height;
	dims[2] = width;
	image = gh_alloc_tensor(&proc->scratch_arena, SAM3_DTYPE_F32,
				3, dims);
	if (!image)
		return SAM3_ENOMEM;

	dst = (float *)image->data;
	for (c = 0; c < 3; c++) {
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				int src_idx = (y * width + x) * 3 + c;
				int dst_idx = c * height * width +
					      y * width + x;
				dst[dst_idx] = pixels[src_idx] / 255.0f;
			}
		}
	}

	/* Build and evaluate vision encoder graph */
	sam3_graph_init(&graph);
	err = sam3_image_model_encode(&proc->model, &graph,
				      proc->backend, image,
				      &proc->scratch_arena);
	if (err != SAM3_OK)
		return err;

	proc->image_loaded = 1;
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
 * project_prompts - Project point/box coordinates to d_model embeddings.
 *
 * Creates coordinate tensors from prompts and projects them through
 * the geometry encoder's linear layers. Points use [N, 2] -> [N, d_model]
 * and boxes use [N, 4] -> [N, d_model]. Results are concatenated into
 * a single [total, d_model] tensor.
 */
static struct sam3_tensor *project_prompts(
	struct sam3_image_model *model,
	struct sam3_graph *g,
	const struct sam3_prompt *prompts,
	int n_prompts,
	struct sam3_arena *arena)
{
	int n_points, n_boxes;
	struct sam3_tensor *point_coords = NULL;
	struct sam3_tensor *box_coords = NULL;
	struct sam3_tensor *point_proj = NULL;
	struct sam3_tensor *box_proj = NULL;
	struct sam3_tensor *parts[2];
	int n_parts = 0;
	int pi = 0, bi = 0;
	int i;
	float *pdata, *bdata;

	count_prompts_by_type(prompts, n_prompts, &n_points, &n_boxes);

	if (n_points == 0 && n_boxes == 0)
		return NULL;

	/* Build point coordinate tensor [n_points, 2] */
	if (n_points > 0) {
		int dims[2] = { n_points, 2 };

		point_coords = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
					       2, dims);
		if (!point_coords)
			return NULL;

		pdata = (float *)point_coords->data;
		for (i = 0; i < n_prompts; i++) {
			if (prompts[i].type != SAM3_PROMPT_POINT)
				continue;
			pdata[pi * 2 + 0] = prompts[i].point.x;
			pdata[pi * 2 + 1] = prompts[i].point.y;
			pi++;
		}

		/* Project: [n_points, 2] @ [2, d_model] -> [n_points, d_model] */
		point_proj = gh_linear(g, arena, point_coords,
				       model->geom_enc.point_proj_w,
				       model->geom_enc.point_proj_b);
		if (!point_proj)
			return NULL;

		parts[n_parts++] = point_proj;
	}

	/* Build box coordinate tensor [n_boxes, 4] */
	if (n_boxes > 0) {
		int dims[2] = { n_boxes, 4 };

		box_coords = gh_alloc_tensor(arena, SAM3_DTYPE_F32,
					     2, dims);
		if (!box_coords)
			return NULL;

		bdata = (float *)box_coords->data;
		for (i = 0; i < n_prompts; i++) {
			if (prompts[i].type != SAM3_PROMPT_BOX)
				continue;
			bdata[bi * 4 + 0] = prompts[i].box.x1;
			bdata[bi * 4 + 1] = prompts[i].box.y1;
			bdata[bi * 4 + 2] = prompts[i].box.x2;
			bdata[bi * 4 + 3] = prompts[i].box.y2;
			bi++;
		}

		/* Project: [n_boxes, 4] @ [4, d_model] -> [n_boxes, d_model] */
		box_proj = gh_linear(g, arena, box_coords,
				     model->geom_enc.box_proj_w,
				     model->geom_enc.box_proj_b);
		if (!box_proj)
			return NULL;

		parts[n_parts++] = box_proj;
	}

	/* Concatenate point and box projections along axis 0 */
	if (n_parts == 1)
		return parts[0];

	return gh_concat(g, arena, parts, n_parts, 0);
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

enum sam3_error sam3_processor_segment(struct sam3_processor *proc,
				       const struct sam3_prompt *prompts,
				       int n_prompts,
				       struct sam3_result *result)
{
	struct sam3_graph graph;
	struct sam3_tensor *prompt_tokens = NULL;
	struct sam3_tensor *text_features = NULL;
	struct sam3_tensor *mask_logits;
	enum sam3_error err;
	size_t mask_bytes;
	int nelems;
	const char *text;

	if (!proc || !prompts || n_prompts <= 0 || !result)
		return SAM3_EINVAL;

	if (!proc->image_loaded)
		return SAM3_EINVAL;

	memset(result, 0, sizeof(*result));

	/* Reset scratch arena for this segment pass */
	sam3_arena_reset(&proc->scratch_arena);

	sam3_graph_init(&graph);

	/* Encode text prompt if present */
	text = find_text_prompt(prompts, n_prompts);
	if (text) {
		text_features = sam3_vl_backbone_build_text(
			&proc->model.backbone, &graph,
			text, NULL, &proc->scratch_arena);
		if (!text_features)
			return SAM3_ENOMEM;
	}

	/* Project geometric prompt coordinates to d_model embeddings */
	prompt_tokens = project_prompts(&proc->model, &graph, prompts,
					n_prompts, &proc->scratch_arena);

	/* At least one of text or geometry must be present */
	if (!prompt_tokens && !text_features)
		return SAM3_EINVAL;

	/* Build segmentation graph */
	mask_logits = sam3_image_model_segment(&proc->model, &graph,
					       proc->backend, prompt_tokens,
					       text_features,
					       &proc->scratch_arena);
	if (!mask_logits)
		return SAM3_ENOMEM;

	/* Evaluate the segmentation graph */
	err = proc->backend->ops->graph_eval(proc->backend, &graph);
	if (err != SAM3_OK)
		return err;

	/*
	 * Copy mask logits to result. The mask tensor is expected to be
	 * [n_masks, H, W] from the segmentation head.
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
		return SAM3_EINVAL;
	}

	nelems = result->n_masks * result->mask_height * result->mask_width;
	mask_bytes = (size_t)nelems * sizeof(float);

	result->masks = malloc(mask_bytes);
	if (!result->masks)
		return SAM3_ENOMEM;

	memcpy(result->masks, mask_logits->data, mask_bytes);

	/* Allocate IoU scores (one per mask), zero-initialized for now */
	result->iou_scores = calloc((size_t)result->n_masks, sizeof(float));
	if (!result->iou_scores) {
		free(result->masks);
		result->masks = NULL;
		return SAM3_ENOMEM;
	}

	return SAM3_OK;
}
