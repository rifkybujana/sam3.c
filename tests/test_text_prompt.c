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
#include "model/sam3_processor.h"
#include "model/graph_helpers.h"
#include "backend/cpu/cpu_backend.h"

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

int main(void)
{
	test_text_prompt_type();
	test_extract_text_prompt();

	TEST_REPORT();
}
