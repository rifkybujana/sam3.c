/*
 * tests/test_box_prompt.c - Box prompt geometry encoding parity.
 *
 * Verifies the image-path box token produced by sam3_project_prompts
 * against the Python reference's SequenceGeometryEncoder._encode_boxes
 * (direct_project + pos_enc_project + label_embed[1]). Pool projection
 * is deliberately excluded by feeding feat_s1_nhwc=NULL, so the checked
 * vector is exactly direct + posenc + label.
 *
 * Regression guard for the fix that restored cxcywh coords, the missing
 * pool / pos-enc projections, and the positive label default on boxes.
 *
 * Expected values come from reference/sam3/sam3/model/geometry_encoders.py
 * evaluated against models/sam3.safetensors — see the sibling Python
 * script referenced in the header comment of the fix commit.
 *
 * Key types:  sam3_image_model (geom_enc only populated)
 * Depends on: test_helpers.h, model/sam3_processor.h, model/prompt_encoder.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "sam3/sam3_types.h"
#include "model/sam3_processor.h"
#include "model/sam3_image.h"
#include "model/prompt_encoder.h"
#include "core/alloc.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Expected first 8 elements of the box-prompt token for box xyxy=
 * (201.6, 302.4, 403.2, 554.4) on a 1008×1008 prompt space → cxcywh=
 * (0.3, 0.425, 0.2, 0.25) normalized, label=1. Computed from the Python
 * reference using the shipped SAM3 weights.
 */
static const float expected_head[8] = {
	-1.34241867f,
	 2.73933268f,
	-2.74823284f,
	 0.23872589f,
	 1.24911439f,
	-1.23072577f,
	-2.32940722f,
	 2.25193739f,
};

/*
 * test_box_prompt_direct_posenc_label - End-to-end check of the three
 * direct contributions on top of the real SAM3 weights.
 *
 * The model file is only used to pull geom_enc tensors; no backbone or
 * decoder is initialized.
 */
static void test_box_prompt_direct_posenc_label(const char *model_path)
{
	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	enum sam3_error err = sam3_weight_open(&wf, model_path);
	if (err != SAM3_OK) {
		fprintf(stderr, "SKIP test_box_prompt: cannot open %s\n",
			model_path);
		return;
	}

	struct sam3_arena arena;
	/* geom_enc weights plus a small output + scratch: 128 MiB is ample. */
	ASSERT_EQ(sam3_arena_init(&arena, 128 * 1024 * 1024), SAM3_OK);

	struct sam3_image_model model;
	memset(&model, 0, sizeof(model));
	ASSERT_EQ(sam3_geometry_encoder_init(&model.geom_enc, 256, 3),
		  SAM3_OK);
	ASSERT_EQ(sam3_geometry_encoder_load(&model.geom_enc, &wf, &arena),
		  SAM3_OK);

	/* Box xyxy on 1008x1008 prompt space; matches the Python ref. */
	struct sam3_prompt prompt;
	memset(&prompt, 0, sizeof(prompt));
	prompt.type = SAM3_PROMPT_BOX;
	prompt.box.x1 = 201.6f;
	prompt.box.y1 = 302.4f;
	prompt.box.x2 = 403.2f;
	prompt.box.y2 = 554.4f;

	struct sam3_tensor *out = sam3_project_prompts(
		&model,
		/* feat_s1_nhwc = */ NULL,
		&prompt, 1,
		1008, 1008,
		&arena);
	ASSERT_NOT_NULL(out);
	ASSERT_EQ(out->dtype, SAM3_DTYPE_F32);
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], 1);    /* single box → single token */
	ASSERT_EQ(out->dims[1], 256);

	const float *token = (const float *)out->data;
	/* Relative tolerance 1e-4 absorbs float32 reorder-sum drift. */
	assert_tensor_close_f32(token, expected_head, 8,
				/* rtol */ 1e-4f,
				/* atol */ 1e-4f,
				"box_prompt_direct_posenc_label");

	sam3_arena_free(&arena);
	sam3_weight_close(&wf);
}

int main(int argc, char **argv)
{
	const char *model_path = (argc > 1) ? argv[1] : "models/sam3.sam3";
	test_box_prompt_direct_posenc_label(model_path);

	printf("test_box_prompt: %d passed, %d failed\n",
	       tests_run - tests_failed, tests_failed);
	return tests_failed ? 1 : 0;
}
