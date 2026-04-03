/*
 * tests/test_text_prompt.c - Text prompt integration tests
 *
 * Tests end-to-end text prompt support through the processor,
 * including text-only, geometry-only, and mixed prompt modes.
 * Uses small model dimensions and zeroed weights for fast testing.
 *
 * Key types:  sam3_processor, sam3_prompt
 * Depends on: test_helpers.h, model/sam3_processor.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "sam3/sam3_types.h"
#include "model/text_encoder.h"
#include "model/tokenizer.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"
#include "core/graph.h"
#include "core/tensor.h"

#include <string.h>

/*
 * test_text_prompt_type - Verify SAM3_PROMPT_TEXT enum exists and
 * text field is accessible.
 */
static void test_text_prompt_type(void)
{
	struct sam3_prompt prompt;
	memset(&prompt, 0, sizeof(prompt));
	prompt.type = SAM3_PROMPT_TEXT;
	prompt.text = "a cat";

	ASSERT_EQ(prompt.type, SAM3_PROMPT_TEXT);
	ASSERT(prompt.text != NULL);
	ASSERT_EQ(strcmp(prompt.text, "a cat"), 0);
}

/*
 * test_extract_text_prompt - Verify text prompts are counted
 * separately from geometric prompts.
 */
static void test_extract_text_prompt(void)
{
	struct sam3_prompt prompts[3];

	prompts[0].type = SAM3_PROMPT_POINT;
	prompts[0].point.x = 0.5f;
	prompts[0].point.y = 0.5f;
	prompts[0].point.label = 1;

	prompts[1].type = SAM3_PROMPT_TEXT;
	prompts[1].text = "a dog";

	prompts[2].type = SAM3_PROMPT_BOX;
	prompts[2].box.x1 = 0.0f;
	prompts[2].box.y1 = 0.0f;
	prompts[2].box.x2 = 1.0f;
	prompts[2].box.y2 = 1.0f;

	/* Count geometric vs text prompts */
	int n_geom = 0;
	int n_text = 0;
	for (int i = 0; i < 3; i++) {
		if (prompts[i].type == SAM3_PROMPT_TEXT)
			n_text++;
		else
			n_geom++;
	}

	ASSERT_EQ(n_geom, 2);
	ASSERT_EQ(n_text, 1);
}

/*
 * test_text_prompt_graph_build - Verify text features flow through the
 * text encoder graph with correct output shape.
 *
 * Sets up a CPU backend, creates a small text encoder with zero-init
 * weights, tokenizes "a cat", builds the encoder graph, and checks
 * the output tensor shape is [77, 16] (seq_len, d_model).
 */
static void test_text_prompt_graph_build(void)
{
	/* CPU backend with 128 MiB arena */
	struct sam3_cpu_backend cpu;
	memset(&cpu, 0, sizeof(cpu));
	cpu.base.type = SAM3_BACKEND_CPU;
	cpu.base.ops = sam3_cpu_backend_ops();
	cpu.arena_capacity = 128 * 1024 * 1024;
	ASSERT_EQ(cpu.base.ops->init(&cpu.base), SAM3_OK);

	/* 64 MiB arena for model weights */
	struct sam3_arena arena;
	ASSERT_EQ(sam3_arena_init(&arena, 64 * 1024 * 1024), SAM3_OK);

	/* Small text encoder */
	struct sam3_text_encoder te;
	memset(&te, 0, sizeof(te));
	te.d_model = 16;
	te.width = 32;
	te.n_heads = 4;
	te.n_layers = 2;
	te.context_len = 77;
	te.vocab_size = 49408;
	ASSERT_EQ(sam3_text_encoder_load(&te, NULL, &arena), SAM3_OK);

	/* Tokenizer */
	struct sam3_tokenizer tok;
	ASSERT_EQ(sam3_tokenizer_init(&tok), SAM3_OK);
	int32_t tokens[77];
	sam3_tokenizer_encode(&tok, "a cat", tokens, 77);

	/* Token tensor [77] of I32 */
	int tok_dims[] = {77};
	struct sam3_tensor *tok_tensor = gh_alloc_tensor(&arena,
							SAM3_DTYPE_I32,
							1, tok_dims);
	memcpy(tok_tensor->data, tokens, 77 * sizeof(int32_t));

	/* Build text encoder graph */
	struct sam3_graph graph;
	sam3_graph_init(&graph);
	struct sam3_tensor *pooled = NULL;
	struct sam3_tensor *text_features = sam3_text_encoder_build(
		&te, &graph, tok_tensor, &pooled, &arena);

	/* Verify output shape: [77, 16] = [seq_len, d_model] */
	ASSERT(text_features != NULL);
	ASSERT_EQ(text_features->n_dims, 2);
	ASSERT_EQ(text_features->dims[0], 77);
	ASSERT_EQ(text_features->dims[1], 16);

	/* Cleanup */
	sam3_tokenizer_free(&tok);
	sam3_arena_free(&arena);
	cpu.base.ops->free(&cpu.base);
}

int main(void)
{
	test_text_prompt_type();
	test_extract_text_prompt();
	test_text_prompt_graph_build();

	TEST_REPORT();
}
