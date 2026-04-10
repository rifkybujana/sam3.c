/*
 * tests/test_profile.c - Unit tests for the profiler
 *
 * Tests profiler lifecycle, stage timing, op timing, memory tracking,
 * and report output. Built only when SAM3_PROFILE is enabled.
 *
 * Key types:  sam3_profiler
 * Depends on: util/profile.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "util/profile.h"

/* Task 2: Lifecycle tests */

static void test_profiler_create_free(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	ASSERT(p != NULL);
	sam3_profiler_free(p);
}

static void test_profiler_enable_disable(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	ASSERT_EQ(sam3_profiler_is_enabled(p), 0);
	sam3_profiler_enable(p);
	ASSERT_EQ(sam3_profiler_is_enabled(p), 1);
	sam3_profiler_disable(p);
	ASSERT_EQ(sam3_profiler_is_enabled(p), 0);
	sam3_profiler_free(p);
}

static void test_profiler_reset(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);
	sam3_prof_stage_begin(p, "test_stage");
	sam3_prof_stage_end(p, "test_stage");
	sam3_profiler_reset(p);
	/* After reset, should still be enabled but no data */
	ASSERT_EQ(sam3_profiler_is_enabled(p), 1);
	sam3_profiler_free(p);
}

/* Task 3: Stage timing tests */

static void test_stage_timing(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_stage_begin(p, "test_stage");
	/* Burn some time */
	volatile int x = 0;
	for (int i = 0; i < 100000; i++)
		x += i;
	(void)x;
	sam3_prof_stage_end(p, "test_stage");

	ASSERT_EQ(p->n_stages, 1);
	ASSERT_EQ(p->stages[0].calls, 1);
	ASSERT(p->stages[0].total_ns > 0);

	sam3_profiler_free(p);
}

static void test_stage_accumulates(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_stage_begin(p, "accum");
	sam3_prof_stage_end(p, "accum");
	sam3_prof_stage_begin(p, "accum");
	sam3_prof_stage_end(p, "accum");

	ASSERT_EQ(p->stages[0].calls, 2);

	sam3_profiler_free(p);
}

static void test_stage_disabled_noop(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	/* NOT enabled */
	sam3_prof_stage_begin(p, "noop");
	sam3_prof_stage_end(p, "noop");

	ASSERT_EQ(p->n_stages, 0);

	sam3_profiler_free(p);
}

/* Task 4: Op timing and memory tracking tests */

static void test_op_timing(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_op_begin(p, SAM3_OP_MATMUL);
	volatile int x = 0;
	for (int i = 0; i < 10000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_MATMUL);

	ASSERT_EQ(p->op_stats[SAM3_OP_MATMUL].calls, 1);
	ASSERT(p->op_stats[SAM3_OP_MATMUL].total_ns > 0);
	/* Other ops should be untouched */
	ASSERT_EQ(p->op_stats[SAM3_OP_ADD].calls, 0);

	sam3_profiler_free(p);
}

static void test_mem_tracking(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_mem_arena(p);
	ASSERT_EQ(p->mem.arena_count, 1);

	sam3_prof_mem_alloc(p, 1024);
	ASSERT_EQ(p->mem.alloc_count, 1);
	ASSERT_EQ((int)p->mem.current_bytes, 1024);
	ASSERT_EQ((int)p->mem.peak_bytes, 1024);

	sam3_prof_mem_alloc(p, 2048);
	ASSERT_EQ(p->mem.alloc_count, 2);
	ASSERT_EQ((int)p->mem.current_bytes, 3072);
	ASSERT_EQ((int)p->mem.peak_bytes, 3072);

	/* Simulate arena reset */
	sam3_prof_mem_arena_reset(p, 3072);
	ASSERT_EQ((int)p->mem.current_bytes, 0);
	/* Peak should not change */
	ASSERT_EQ((int)p->mem.peak_bytes, 3072);

	sam3_profiler_free(p);
}

static void test_report_no_crash(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	sam3_prof_stage_begin(p, "encode");
	sam3_prof_stage_end(p, "encode");
	sam3_prof_op_begin(p, SAM3_OP_MATMUL);
	sam3_prof_op_end(p, SAM3_OP_MATMUL);
	sam3_prof_mem_arena(p);
	sam3_prof_mem_alloc(p, 4096);

	/* Should not crash */
	sam3_profiler_report(p);

	ASSERT(1); /* If we got here, no crash */
	sam3_profiler_free(p);
}

/* Task 8: Full round-trip integration test */

static void test_full_round_trip(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	/* Simulate a mini inference run */
	sam3_prof_stage_begin(p, "image_encoder");

	sam3_prof_mem_arena(p);
	sam3_prof_mem_alloc(p, 1024 * 1024);  /* 1MB */

	sam3_prof_op_begin(p, SAM3_OP_CONV2D);
	volatile int x = 0;
	for (int i = 0; i < 50000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_CONV2D);

	sam3_prof_op_begin(p, SAM3_OP_MATMUL);
	for (int i = 0; i < 50000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_MATMUL);

	sam3_prof_stage_end(p, "image_encoder");

	sam3_prof_stage_begin(p, "mask_decoder");
	sam3_prof_op_begin(p, SAM3_OP_SOFTMAX);
	for (int i = 0; i < 10000; i++)
		x += i;
	(void)x;
	sam3_prof_op_end(p, SAM3_OP_SOFTMAX);
	sam3_prof_stage_end(p, "mask_decoder");

	/* Verify data collected */
	ASSERT_EQ(p->n_stages, 2);
	ASSERT(p->stages[0].total_ns > 0);
	ASSERT(p->stages[1].total_ns > 0);
	ASSERT_EQ(p->op_stats[SAM3_OP_CONV2D].calls, 1);
	ASSERT_EQ(p->op_stats[SAM3_OP_MATMUL].calls, 1);
	ASSERT_EQ(p->op_stats[SAM3_OP_SOFTMAX].calls, 1);
	ASSERT_EQ(p->mem.alloc_count, 1);
	ASSERT_EQ((int)p->mem.peak_bytes, 1024 * 1024);

	/* Print report */
	sam3_profiler_report(p);

	sam3_profiler_free(p);
}

static void test_inference_stages(void)
{
	struct sam3_profiler *p = sam3_profiler_create();
	sam3_profiler_enable(p);

	/* Simulate the stages wired in processor.c and sub-modules */
	sam3_prof_stage_begin(p, "model_load");
	sam3_prof_stage_end(p, "model_load");

	sam3_prof_stage_begin(p, "image_normalize");
	sam3_prof_stage_end(p, "image_normalize");

	sam3_prof_stage_begin(p, "image_encode");

	sam3_prof_stage_begin(p, "vit_precompute");
	sam3_prof_stage_end(p, "vit_precompute");

	sam3_prof_stage_begin(p, "vit_patch_embed");
	sam3_prof_stage_end(p, "vit_patch_embed");

	sam3_prof_stage_begin(p, "vit_blocks");
	sam3_prof_stage_end(p, "vit_blocks");

	sam3_prof_stage_begin(p, "neck");
	sam3_prof_stage_end(p, "neck");

	sam3_prof_stage_end(p, "image_encode");

	sam3_prof_stage_begin(p, "text_encode");

	sam3_prof_stage_begin(p, "tokenize");
	sam3_prof_stage_end(p, "tokenize");

	sam3_prof_stage_begin(p, "text_blocks");
	sam3_prof_stage_end(p, "text_blocks");

	sam3_prof_stage_end(p, "text_encode");

	sam3_prof_stage_begin(p, "prompt_project");
	sam3_prof_stage_end(p, "prompt_project");

	sam3_prof_stage_begin(p, "mask_decode");

	sam3_prof_stage_begin(p, "geometry_encode");
	sam3_prof_stage_end(p, "geometry_encode");

	sam3_prof_stage_begin(p, "encoder_fusion");
	sam3_prof_stage_end(p, "encoder_fusion");

	sam3_prof_stage_begin(p, "decoder");
	sam3_prof_stage_end(p, "decoder");

	sam3_prof_stage_begin(p, "seg_head");
	sam3_prof_stage_end(p, "seg_head");

	sam3_prof_stage_end(p, "mask_decode");

	sam3_prof_stage_begin(p, "postprocess");
	sam3_prof_stage_end(p, "postprocess");

	ASSERT_EQ(p->n_stages, 17);
	for (int i = 0; i < 17; i++) {
		ASSERT_EQ(p->stages[i].calls, 1);
		ASSERT(p->stages[i].total_ns >= 0);
	}

	sam3_profiler_report(p);
	sam3_profiler_free(p);
}

int main(void)
{
	/* Task 2: Lifecycle */
	test_profiler_create_free();
	test_profiler_enable_disable();
	test_profiler_reset();

	/* Task 3: Stage timing */
	test_stage_timing();
	test_stage_accumulates();
	test_stage_disabled_noop();

	/* Task 4: Op timing and memory */
	test_op_timing();
	test_mem_tracking();
	test_report_no_crash();

	/* Task 8: Integration */
	test_full_round_trip();

	/* Task 5: Stage names */
	test_inference_stages();

	TEST_REPORT();
}
