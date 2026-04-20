/*
 * tests/test_mobileclip_text.c - MobileCLIP text encoder parity tests
 *
 * Loads fixtures dumped by scripts/dump_mobileclip_text_layers.py
 * (one directory per variant under tests/fixtures/) and exercises the
 * C-side encoder for shape, per-block, and final-pooled parity against
 * the PyTorch reference. This file covers Task 5.1 (factory smoke) only;
 * per-block parity tests are added in Tasks 5.2 and 5.3.
 *
 * Key types:  sam3_text_encoder_iface, sam3_mobileclip_config
 * Depends on: text_encoder_iface.h, mobileclip_text.h, core/alloc.h
 * Used by:    CTest (test_mobileclip_text)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sam3/sam3_types.h"
#include "core/alloc.h"
#include "model/mobileclip_text.h"
#include "model/text_encoder_iface.h"
#include "test_helpers.h"
#include "test_npy.h"

/* --- test_iface_factory_s0 --- */

/*
 * Smoke: factory creates a valid iface for the S0 variant with correct
 * ctx_len and d_model fields, without loading any weights.
 */
static int
test_iface_factory_s0(void)
{
	struct sam3_arena arena;
	struct sam3_text_encoder_iface iface;
	enum sam3_error err;

	if (sam3_arena_init(&arena, 4 * 1024 * 1024) != SAM3_OK) {
		fprintf(stderr, "FAIL: arena init\n");
		return 1;
	}

	memset(&iface, 0, sizeof(iface));
	err = sam3_text_encoder_iface_init(
		&iface, SAM3_TEXT_MOBILECLIP_S0, &arena);

	ASSERT(err == SAM3_OK);
	ASSERT(iface.text_backbone == SAM3_TEXT_MOBILECLIP_S0);
	ASSERT(iface.ctx_len == 16);
	ASSERT(iface.d_model == 256);
	ASSERT(iface.ops != NULL);
	ASSERT(iface.impl != NULL);

	sam3_arena_free(&arena);
	return 0;
}

/* --- test_iface_factory_s1 --- */

/*
 * Smoke: factory creates a valid iface for the S1 variant with correct
 * ctx_len and d_model fields, without loading any weights.
 */
static int
test_iface_factory_s1(void)
{
	struct sam3_arena arena;
	struct sam3_text_encoder_iface iface;
	enum sam3_error err;

	if (sam3_arena_init(&arena, 4 * 1024 * 1024) != SAM3_OK) {
		fprintf(stderr, "FAIL: arena init\n");
		return 1;
	}

	memset(&iface, 0, sizeof(iface));
	err = sam3_text_encoder_iface_init(
		&iface, SAM3_TEXT_MOBILECLIP_S1, &arena);

	ASSERT(err == SAM3_OK);
	ASSERT(iface.text_backbone == SAM3_TEXT_MOBILECLIP_S1);
	ASSERT(iface.ctx_len == 16);
	ASSERT(iface.d_model == 256);
	ASSERT(iface.ops != NULL);
	ASSERT(iface.impl != NULL);

	sam3_arena_free(&arena);
	return 0;
}

/* --- test_iface_factory_l --- */

/*
 * Smoke: factory creates a valid iface for the L variant with correct
 * ctx_len and d_model fields, without loading any weights.
 */
static int
test_iface_factory_l(void)
{
	struct sam3_arena arena;
	struct sam3_text_encoder_iface iface;
	enum sam3_error err;

	if (sam3_arena_init(&arena, 4 * 1024 * 1024) != SAM3_OK) {
		fprintf(stderr, "FAIL: arena init\n");
		return 1;
	}

	memset(&iface, 0, sizeof(iface));
	err = sam3_text_encoder_iface_init(
		&iface, SAM3_TEXT_MOBILECLIP_L, &arena);

	ASSERT(err == SAM3_OK);
	ASSERT(iface.text_backbone == SAM3_TEXT_MOBILECLIP_L);
	ASSERT(iface.ctx_len == 16);
	ASSERT(iface.d_model == 256);
	ASSERT(iface.ops != NULL);
	ASSERT(iface.impl != NULL);

	sam3_arena_free(&arena);
	return 0;
}

/* --- test_mobileclip_config_for --- */

/*
 * Smoke: sam3_mobileclip_config_for returns the correct static config for
 * each known variant and NULL for an unknown backbone id.
 */
static int
test_mobileclip_config_for(void)
{
	const struct sam3_mobileclip_config *cfg;

	cfg = sam3_mobileclip_config_for(SAM3_TEXT_MOBILECLIP_S0);
	ASSERT(cfg != NULL);
	ASSERT(cfg->text_backbone == SAM3_TEXT_MOBILECLIP_S0);
	ASSERT(cfg->ctx_len == 16);
	ASSERT(cfg->out_dim == 256);

	cfg = sam3_mobileclip_config_for(SAM3_TEXT_MOBILECLIP_S1);
	ASSERT(cfg != NULL);
	ASSERT(cfg->text_backbone == SAM3_TEXT_MOBILECLIP_S1);
	ASSERT(cfg->ctx_len == 16);
	ASSERT(cfg->out_dim == 256);
	ASSERT(cfg->n_repmixer_blocks == 0);

	cfg = sam3_mobileclip_config_for(SAM3_TEXT_MOBILECLIP_L);
	ASSERT(cfg != NULL);
	ASSERT(cfg->text_backbone == SAM3_TEXT_MOBILECLIP_L);
	ASSERT(cfg->ctx_len == 16);
	ASSERT(cfg->out_dim == 256);
	ASSERT(cfg->n_repmixer_blocks == 0);

	/* Unknown variant must return NULL, not crash. */
	cfg = sam3_mobileclip_config_for(9999);
	ASSERT(cfg == NULL);

	return 0;
}

int
main(void)
{
	test_iface_factory_s0();
	test_iface_factory_s1();
	test_iface_factory_l();
	test_mobileclip_config_for();

	if (tests_failed == 0)
		printf("test_mobileclip_text: PASS (%d tests, factory smoke)\n",
		       tests_run);

	TEST_REPORT();
}
