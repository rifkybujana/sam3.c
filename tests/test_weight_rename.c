/*
 * tests/test_weight_rename.c - Unit tests for weight name remapping
 *
 * Verifies that the rename reader correctly maps PyTorch .pt checkpoint
 * key names (facebook/sam3) to the C model's expected weight names,
 * including prefix replacement, attribute renaming, and QKV splitting.
 *
 * Uses a mock weight reader that provides named tensors without real
 * data, then checks that the rename reader produces the expected
 * output names and split counts.
 *
 * Key types:  weight_reader, weight_tensor_info
 * Depends on: core/weight.h, test_helpers.h, weight_rename.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "test_helpers.h"
#include "core/weight.h"

/* Include weight_rename.h from tools/ — build system adds include path */
#include "weight_rename.h"

/* ── Mock reader ───────────────────────────────────────────────────── */

#define MOCK_MAX_TENSORS 64

struct mock_reader_state {
	int n;
	struct {
		char name[SAM3_WEIGHT_NAME_MAX];
		int  dims[SAM3_MAX_DIMS];
		int  n_dims;
	} tensors[MOCK_MAX_TENSORS];
};

static enum sam3_error mock_open(struct weight_reader *r, const char *path)
{
	(void)r; (void)path;
	return SAM3_OK;
}

static int mock_n_tensors(struct weight_reader *r)
{
	struct mock_reader_state *s = r->impl;
	return s->n;
}

static enum sam3_error mock_get_tensor_info(struct weight_reader *r, int idx,
					    struct weight_tensor_info *info)
{
	struct mock_reader_state *s = r->impl;
	if (idx < 0 || idx >= s->n)
		return SAM3_EINVAL;

	info->name = s->tensors[idx].name;
	info->dtype = SAM3_DTYPE_F32;
	info->n_dims = s->tensors[idx].n_dims;
	for (int i = 0; i < info->n_dims; i++)
		info->dims[i] = s->tensors[idx].dims[i];

	/* Compute nbytes */
	size_t nb = sizeof(float);
	for (int i = 0; i < info->n_dims; i++)
		nb *= (size_t)info->dims[i];
	info->nbytes = nb;

	return SAM3_OK;
}

static enum sam3_error mock_read_tensor_data(struct weight_reader *r, int idx,
					     void *dst, size_t dst_size)
{
	(void)r; (void)idx;
	memset(dst, 0, dst_size);
	return SAM3_OK;
}

static void mock_close(struct weight_reader *r)
{
	(void)r;
}

static const struct weight_reader_ops mock_ops = {
	.open             = mock_open,
	.n_tensors        = mock_n_tensors,
	.get_tensor_info  = mock_get_tensor_info,
	.read_tensor_data = mock_read_tensor_data,
	.close            = mock_close,
};

static void mock_add(struct mock_reader_state *s, const char *name,
		     int n_dims, const int *dims)
{
	int i = s->n++;
	snprintf(s->tensors[i].name, SAM3_WEIGHT_NAME_MAX, "%s", name);
	s->tensors[i].n_dims = n_dims;
	for (int d = 0; d < n_dims; d++)
		s->tensors[i].dims[d] = dims[d];
}

/* ── Helper: find renamed tensor by output name ────────────────────── */

static int find_output(struct weight_reader *rr, const char *expected_name)
{
	int n = rr->ops->n_tensors(rr);
	for (int i = 0; i < n; i++) {
		struct weight_tensor_info info;
		if (rr->ops->get_tensor_info(rr, i, &info) != SAM3_OK)
			continue;
		if (strcmp(info.name, expected_name) == 0)
			return i;
	}
	return -1;
}

/* ── ViT backbone tests ───────────────────────────────────────────── */

static void test_vit_qkv_split(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {384, 128};
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk."
		 "blocks.3.attn.qkv.weight",
		 2, dims2);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	enum sam3_error err = weight_reader_rename_init(&rr, &inner);
	ASSERT(err == SAM3_OK);

	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.3.attention.q_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.3.attention.k_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.3.attention.v_proj.weight") >= 0);

	/* Check dim splitting: original [384, 128] -> [128, 128] */
	struct weight_tensor_info info;
	int idx = find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.3.attention.q_proj.weight");
	ASSERT(rr.ops->get_tensor_info(&rr, idx, &info) == SAM3_OK);
	ASSERT_EQ(info.dims[0], 128);
	ASSERT_EQ(info.dims[1], 128);

	rr.ops->close(&rr);
}

static void test_vit_attn_proj(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {128, 128};
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk."
		 "blocks.0.attn.proj.weight",
		 2, dims2);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 1);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.0.attention.o_proj.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_vit_norms(void)
{
	struct mock_reader_state ms = {0};
	int dims1[] = {128};
	/* PT uses norm1/norm2 -> C uses layer_norm1/layer_norm2 */
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk."
		 "blocks.5.norm1.weight",
		 1, dims1);
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk."
		 "blocks.5.norm2.weight",
		 1, dims1);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.5.layer_norm1.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.5.layer_norm2.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_vit_pos_embed(void)
{
	struct mock_reader_state ms = {0};
	int dims3[] = {1, 64, 128};
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk.pos_embed",
		 3, dims3);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"embeddings.position_embedding.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_vit_mlp(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {512, 128};
	/* PT uses mlp.fc1 / mlp.fc2 (unchanged) */
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk."
		 "blocks.0.mlp.fc1.weight",
		 2, dims2);
	int dims2b[] = {128, 512};
	mock_add(&ms,
		 "detector.backbone.vision_backbone.trunk."
		 "blocks.0.mlp.fc2.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.0.mlp.fc1.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.backbone."
		"layers.0.mlp.fc2.weight") >= 0);
	rr.ops->close(&rr);
}

/* ── Neck tests ────────────────────────────────────────────────────── */

static void test_neck(void)
{
	struct mock_reader_state ms = {0};
	int dims4[] = {256, 128, 1, 1};
	/* PT: convs.{i}.conv_1x1.X -> fpn_layers.{i}.proj1.X */
	mock_add(&ms,
		 "detector.backbone.vision_backbone.convs."
		 "0.conv_1x1.weight",
		 4, dims4);
	mock_add(&ms,
		 "detector.backbone.vision_backbone.convs."
		 "1.conv_1x1.weight",
		 4, dims4);
	mock_add(&ms,
		 "detector.backbone.vision_backbone.convs."
		 "2.conv_1x1.weight",
		 4, dims4);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.neck."
		"fpn_layers.0.proj1.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.neck."
		"fpn_layers.1.proj1.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.vision_encoder.neck."
		"fpn_layers.2.proj1.weight") >= 0);
	rr.ops->close(&rr);
}

/* ── Mask decoder tests ────────────────────────────────────────────── */

static void test_mask_decoder_tokens(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {4, 256};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.mask_tokens.weight",
		 2, dims2);
	int dims2b[] = {1, 256};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.iou_token.weight",
		 2, dims2b);
	mock_add(&ms,
		 "tracker.sam_mask_decoder.obj_score_token.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.mask_tokens.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.iou_token.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"obj_score_token.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_upscaling(void)
{
	struct mock_reader_state ms = {0};
	int dims4[] = {64, 256, 2, 2};
	/* PT uses output_upscaling.{0,1,3} */
	mock_add(&ms,
		 "tracker.sam_mask_decoder.output_upscaling.0.weight",
		 4, dims4);
	int dims1[] = {64};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.output_upscaling.1.weight",
		 1, dims1);
	int dims4b[] = {32, 64, 2, 2};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.output_upscaling.3.weight",
		 4, dims4b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.upscale_conv1.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"upscale_layer_norm.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.upscale_conv2.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_transformer_attn(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {256, 256};
	/* PT uses out_proj -> C uses o_proj */
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "self_attn.out_proj.weight",
		 2, dims2);
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "cross_attn_token_to_image.out_proj.weight",
		 2, dims2);
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "self_attn.q_proj.weight",
		 2, dims2);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"self_attn.o_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"cross_attn_token_to_image.o_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"self_attn.q_proj.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_transformer_norms(void)
{
	struct mock_reader_state ms = {0};
	int dims1[] = {256};
	/* PT uses norm{1..4} -> C uses layer_norm{1..4} */
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "norm1.weight",
		 1, dims1);
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "norm2.weight",
		 1, dims1);
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "norm3.weight",
		 1, dims1);
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "norm4.weight",
		 1, dims1);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 4);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"layer_norm1.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"layer_norm2.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"layer_norm3.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"layer_norm4.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_transformer_mlp(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {2048, 256};
	/* PT uses mlp.lin1/lin2 -> C uses mlp.proj_in/proj_out */
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "mlp.lin1.weight",
		 2, dims2);
	int dims2b[] = {256, 2048};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer.layers.0."
		 "mlp.lin2.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"mlp.proj_in.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer.layers.0."
		"mlp.proj_out.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_hypernetworks_mlp(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {256, 256};
	/* PT uses layers.{0,1,2} -> C uses proj_in/layers.0/proj_out */
	mock_add(&ms,
		 "tracker.sam_mask_decoder."
		 "output_hypernetworks_mlps.0.layers.0.weight",
		 2, dims2);
	mock_add(&ms,
		 "tracker.sam_mask_decoder."
		 "output_hypernetworks_mlps.0.layers.1.weight",
		 2, dims2);
	int dims2b[] = {32, 256};
	mock_add(&ms,
		 "tracker.sam_mask_decoder."
		 "output_hypernetworks_mlps.0.layers.2.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"output_hypernetworks_mlps.0.proj_in.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"output_hypernetworks_mlps.0.layers.0.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"output_hypernetworks_mlps.0.proj_out.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_iou_head(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {256, 256};
	/* PT uses layers.{0,1,2} -> C uses proj_in/layers.0/proj_out */
	mock_add(&ms,
		 "tracker.sam_mask_decoder."
		 "iou_prediction_head.layers.0.weight",
		 2, dims2);
	mock_add(&ms,
		 "tracker.sam_mask_decoder."
		 "iou_prediction_head.layers.1.weight",
		 2, dims2);
	int dims2b[] = {4, 256};
	mock_add(&ms,
		 "tracker.sam_mask_decoder."
		 "iou_prediction_head.layers.2.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"iou_prediction_head.proj_in.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"iou_prediction_head.layers.0.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"iou_prediction_head.proj_out.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_obj_score_head(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {256, 256};
	/* obj score head goes through handle_scorer, not handle_mask_decoder */
	mock_add(&ms,
		 "detector.dot_prod_scoring.prompt_mlp.layers.0.weight",
		 2, dims2);
	int dims2b[] = {256, 2048};
	mock_add(&ms,
		 "detector.dot_prod_scoring.prompt_mlp.layers.1.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"pred_obj_score_head.prompt_mlp.fc1.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder."
		"pred_obj_score_head.prompt_mlp.fc2.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_final_attn(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {128, 256};
	/* PT uses out_proj -> C uses o_proj */
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer."
		 "final_attn_token_to_image.out_proj.weight",
		 2, dims2);
	int dims1[] = {256};
	/* PT uses norm_final_attn -> C uses layer_norm_final_attn */
	mock_add(&ms,
		 "tracker.sam_mask_decoder.transformer."
		 "norm_final_attn.weight",
		 1, dims1);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer."
		"final_attn_token_to_image.o_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.transformer."
		"layer_norm_final_attn.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_mask_decoder_conv_s(void)
{
	struct mock_reader_state ms = {0};
	int dims4[] = {32, 256, 1, 1};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.conv_s0.weight",
		 4, dims4);
	int dims4b[] = {64, 256, 1, 1};
	mock_add(&ms,
		 "tracker.sam_mask_decoder.conv_s1.weight",
		 4, dims4b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);

	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.conv_s0.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.mask_decoder.conv_s1.weight") >= 0);
	rr.ops->close(&rr);
}

/* ── Prompt encoder tests ──────────────────────────────────────────── */

static void test_prompt_encoder(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {1, 256};
	mock_add(&ms,
		 "tracker.sam_prompt_encoder.no_mask_embed.weight",
		 2, dims2);
	int dims2b[] = {2, 128};
	mock_add(&ms,
		 "tracker.sam_prompt_encoder."
		 "pe_layer.positional_encoding_gaussian_matrix",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);
	ASSERT(find_output(&rr,
		"tracker_model.prompt_encoder."
		"no_mask_embed.weight") >= 0);
	ASSERT(find_output(&rr,
		"tracker_model.prompt_encoder."
		"shared_embedding.positional_embedding") >= 0);
	rr.ops->close(&rr);
}

/* ── Memory attention tests ────────────────────────────────────────── */

static void test_memory_attention_self_attn(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {256, 256};
	/* PT uses out_proj -> C uses o_proj */
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.1."
		 "self_attn.out_proj.weight",
		 2, dims2);
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.1."
		 "self_attn.q_proj.weight",
		 2, dims2);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);

	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.1."
		"self_attn.o_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.1."
		"self_attn.q_proj.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_memory_attention_cross_attn(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {256, 256};
	/* cross_attn_image -> cross_attn (module rename) */
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0."
		 "cross_attn_image.q_proj.weight",
		 2, dims2);
	/* cross_attn_image.out_proj -> cross_attn.o_proj */
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0."
		 "cross_attn_image.out_proj.weight",
		 2, dims2);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);

	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"cross_attn.q_proj.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"cross_attn.o_proj.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_memory_attention_norms(void)
{
	struct mock_reader_state ms = {0};
	int dims1[] = {256};
	/* PT uses norm{1..3} -> C uses layer_norm{1..3} */
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0.norm1.weight",
		 1, dims1);
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0.norm2.weight",
		 1, dims1);
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0.norm3.weight",
		 1, dims1);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"layer_norm1.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"layer_norm2.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"layer_norm3.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_memory_attention_mlp(void)
{
	struct mock_reader_state ms = {0};
	int dims2[] = {2048, 256};
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0.linear1.weight",
		 2, dims2);
	int dims2b[] = {256, 2048};
	mock_add(&ms,
		 "tracker.transformer.encoder.layers.0.linear2.weight",
		 2, dims2b);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 2);

	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"mlp.fc1.weight") >= 0);
	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layers.0."
		"mlp.fc2.weight") >= 0);
	rr.ops->close(&rr);
}

static void test_memory_attention_top_norm(void)
{
	struct mock_reader_state ms = {0};
	int dims1[] = {256};
	/* Top-level norm (not inside layers.{i}) passes through */
	mock_add(&ms,
		 "tracker.transformer.encoder.layer_norm.weight",
		 1, dims1);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 1);

	ASSERT(find_output(&rr,
		"detector_model.detr_encoder.layer_norm.weight") >= 0);
	rr.ops->close(&rr);
}

/* ── Passthrough test ──────────────────────────────────────────────── */

static void test_passthrough(void)
{
	struct mock_reader_state ms = {0};
	int dims1[] = {256};
	/* Keys with no matching prefix pass through unchanged */
	mock_add(&ms, "no_memory_embedding", 1, dims1);
	mock_add(&ms, "no_object_pointer", 1, dims1);
	mock_add(&ms, "some_random_param.weight", 1, dims1);

	struct weight_reader inner = {.ops = &mock_ops, .impl = &ms};
	struct weight_reader rr;
	ASSERT(weight_reader_rename_init(&rr, &inner) == SAM3_OK);
	ASSERT_EQ(rr.ops->n_tensors(&rr), 3);

	ASSERT(find_output(&rr, "no_memory_embedding") >= 0);
	ASSERT(find_output(&rr, "no_object_pointer") >= 0);
	ASSERT(find_output(&rr, "some_random_param.weight") >= 0);
	rr.ops->close(&rr);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
	/* ViT backbone */
	test_vit_qkv_split();
	test_vit_attn_proj();
	test_vit_norms();
	test_vit_pos_embed();
	test_vit_mlp();

	/* Neck */
	test_neck();

	/* Mask decoder */
	test_mask_decoder_tokens();
	test_mask_decoder_upscaling();
	test_mask_decoder_transformer_attn();
	test_mask_decoder_transformer_norms();
	test_mask_decoder_transformer_mlp();
	test_mask_decoder_hypernetworks_mlp();
	test_mask_decoder_iou_head();
	test_mask_decoder_obj_score_head();
	test_mask_decoder_final_attn();
	test_mask_decoder_conv_s();

	/* Prompt encoder */
	test_prompt_encoder();

	/* Memory attention */
	test_memory_attention_self_attn();
	test_memory_attention_cross_attn();
	test_memory_attention_norms();
	test_memory_attention_mlp();
	test_memory_attention_top_norm();

	/* Passthrough */
	test_passthrough();

	TEST_REPORT();
}
