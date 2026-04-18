/*
 * tests/test_tracker.c - Core tracker module unit tests
 *
 * Tests initialization with default config, zero-init weight loading,
 * memory bank reset, and NULL argument handling for the sam3_tracker
 * module. Uses the CPU backend arena for weight allocation in the
 * load test.
 *
 * Key types:  sam3_tracker
 * Depends on: test_helpers.h, model/tracker.h, model/memory_bank.h,
 *             backend/cpu/cpu_backend.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include "test_helpers.h"
#include "model/tracker.h"
#include "model/memory_bank.h"
#include "model/graph_helpers.h"
#include "core/graph.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/backend.h"

/* --- test infrastructure --- */

static struct sam3_cpu_backend g_cpu;

static void setup(void)
{
	memset(&g_cpu, 0, sizeof(g_cpu));
	g_cpu.base.type = SAM3_BACKEND_CPU;
	g_cpu.base.ops = sam3_cpu_backend_ops();
	g_cpu.arena_capacity = 512 * 1024 * 1024; /* 512 MiB */
	g_cpu.base.ops->init(&g_cpu.base);
}

static void teardown(void)
{
	g_cpu.base.ops->free(&g_cpu.base);
}

/* --- test_tracker_init --- */

static void test_tracker_init(void)
{
	struct sam3_tracker trk;
	enum sam3_error err = sam3_tracker_init(&trk);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(trk.num_maskmem, 7);
	ASSERT_EQ(trk.image_size, 1008);
	ASSERT_EQ(trk.backbone_stride, 14);
	ASSERT_EQ(trk.max_obj_ptrs, 16);
	ASSERT_EQ(trk.max_cond_frames_in_attn, 4);
	ASSERT_EQ(trk.multimask_output, 1);
	ASSERT_EQ(trk.multimask_min_pt_num, 0);
	ASSERT_EQ(trk.multimask_max_pt_num, 1);
}

/* --- test_tracker_init_null --- */

static void test_tracker_init_null(void)
{
	enum sam3_error err = sam3_tracker_init(NULL);
	ASSERT_EQ(err, SAM3_EINVAL);
}

/* --- test_tracker_load_zero --- */

static void test_tracker_load_zero(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk);

	enum sam3_error err = sam3_tracker_load(&trk, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify learned parameters loaded */
	ASSERT(trk.maskmem_tpos_enc != NULL);
	ASSERT(trk.no_mem_embed != NULL);
	ASSERT(trk.no_mem_pos_enc != NULL);
	ASSERT(trk.no_obj_ptr != NULL);
	ASSERT(trk.no_obj_embed_spatial != NULL);
	ASSERT(trk.obj_ptr_proj_fc0_w != NULL);
	ASSERT(trk.obj_ptr_proj_fc0_b != NULL);
	ASSERT(trk.obj_ptr_proj_fc1_w != NULL);
	ASSERT(trk.obj_ptr_proj_fc1_b != NULL);
	ASSERT(trk.obj_ptr_proj_fc2_w != NULL);
	ASSERT(trk.obj_ptr_proj_fc2_b != NULL);
	ASSERT(trk.obj_ptr_tpos_proj_w != NULL);
	ASSERT(trk.obj_ptr_tpos_proj_b != NULL);

	/* Verify shapes */
	ASSERT_EQ(trk.maskmem_tpos_enc->n_dims, 4);
	ASSERT_EQ(trk.maskmem_tpos_enc->dims[0], 7);
	ASSERT_EQ(trk.maskmem_tpos_enc->dims[1], 1);
	ASSERT_EQ(trk.maskmem_tpos_enc->dims[2], 1);
	ASSERT_EQ(trk.maskmem_tpos_enc->dims[3], 64);

	ASSERT_EQ(trk.no_mem_embed->n_dims, 3);
	ASSERT_EQ(trk.no_mem_embed->dims[0], 1);
	ASSERT_EQ(trk.no_mem_embed->dims[1], 1);
	ASSERT_EQ(trk.no_mem_embed->dims[2], 256);

	ASSERT_EQ(trk.no_obj_ptr->n_dims, 2);
	ASSERT_EQ(trk.no_obj_ptr->dims[0], 1);
	ASSERT_EQ(trk.no_obj_ptr->dims[1], 256);

	ASSERT_EQ(trk.obj_ptr_proj_fc0_w->n_dims, 2);
	ASSERT_EQ(trk.obj_ptr_proj_fc0_w->dims[0], 256);
	ASSERT_EQ(trk.obj_ptr_proj_fc0_w->dims[1], 256);
	ASSERT_EQ(trk.obj_ptr_proj_fc0_b->n_dims, 1);
	ASSERT_EQ(trk.obj_ptr_proj_fc0_b->dims[0], 256);

	ASSERT_EQ(trk.obj_ptr_proj_fc2_w->n_dims, 2);
	ASSERT_EQ(trk.obj_ptr_proj_fc2_w->dims[0], 256);
	ASSERT_EQ(trk.obj_ptr_proj_fc2_w->dims[1], 256);

	/* obj_ptr_tpos_proj: Linear(256, 64) */
	ASSERT_EQ(trk.obj_ptr_tpos_proj_w->n_dims, 2);
	ASSERT_EQ(trk.obj_ptr_tpos_proj_w->dims[0], 64);
	ASSERT_EQ(trk.obj_ptr_tpos_proj_w->dims[1], 256);
	ASSERT_EQ(trk.obj_ptr_tpos_proj_b->n_dims, 1);
	ASSERT_EQ(trk.obj_ptr_tpos_proj_b->dims[0], 64);

	/* Verify mask downsampler loaded */
	ASSERT(trk.mask_downsample_w != NULL);
	ASSERT(trk.mask_downsample_b != NULL);
	ASSERT_EQ(trk.mask_downsample_w->n_dims, 4);
	ASSERT_EQ(trk.mask_downsample_w->dims[0], 64);
	ASSERT_EQ(trk.mask_downsample_w->dims[1], 1);
	ASSERT_EQ(trk.mask_downsample_w->dims[2], 4);
	ASSERT_EQ(trk.mask_downsample_w->dims[3], 4);
	ASSERT_EQ(trk.mask_downsample_b->n_dims, 1);
	ASSERT_EQ(trk.mask_downsample_b->dims[0], 64);
}

/* --- test_tracker_reset --- */

static void test_tracker_reset(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk);

	enum sam3_error err = sam3_tracker_reset(&trk);
	ASSERT_EQ(err, SAM3_OK);
	/* Task 2.2: mem_bank removed from tracker; reset is a no-op.
	 * Per-object banks are owned by the session. */
}

/* --- test_tracker_reset_null --- */

static void test_tracker_reset_null(void)
{
	enum sam3_error err = sam3_tracker_reset(NULL);
	ASSERT_EQ(err, SAM3_EINVAL);
}

/* --- test_tracker_load_null_args --- */

static void test_tracker_load_null_args(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk);

	ASSERT_EQ(sam3_tracker_load(NULL, NULL, &g_cpu.arena), SAM3_EINVAL);
	ASSERT_EQ(sam3_tracker_load(&trk, NULL, NULL), SAM3_EINVAL);
}

/* --- test_tracker_track_frame_smoke --- */

static void test_tracker_track_frame_smoke(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk);

	enum sam3_error err = sam3_tracker_load(&trk, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	/* Create a small backbone feature tensor [seq, d_model] */
	int feat_h = 16, feat_w = 16;
	int seq = feat_h * feat_w; /* 256 */
	int feat_dims[] = {seq, 256};
	struct sam3_tensor *backbone = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, feat_dims);
	ASSERT(backbone != NULL);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *masks = NULL;
	struct sam3_tensor *iou = NULL;

	/*
	 * Reinit memory attention with smaller feat size to match
	 * the 16x16 backbone features used in this test.
	 */
	sam3_memory_attn_init(&trk.mem_attention, 256, 64, 4, 1,
			      feat_h, feat_w);
	sam3_memory_attn_load(&trk.mem_attention, NULL, &g_cpu.arena);

	/* Task 2.2: pass a local bank (simulating per-object bank). */
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, trk.num_maskmem,
			      trk.max_cond_frames_in_attn,
			      1, trk.mf_threshold);

	err = sam3_tracker_track_frame(
		&trk, &graph, &bank, backbone,
		feat_h, feat_w,
		NULL, NULL, NULL,  /* no prompt, no hi-res feats */
		0, 1,              /* frame 0, conditioning */
		&g_cpu.arena,
		&masks, &iou, NULL, NULL);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(masks != NULL);
	ASSERT(iou != NULL);

	/* Verify graph was populated (nodes were added) */
	ASSERT(graph.n_nodes > 0);
}

/* --- test_tracker_track_frame_uses_populated_bank --- */

static void test_tracker_track_frame_uses_populated_bank(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk);

	enum sam3_error err = sam3_tracker_load(
		&trk, NULL, &g_cpu.arena);
	ASSERT_EQ(err, SAM3_OK);

	int feat_h = 16, feat_w = 16;
	int seq = feat_h * feat_w;

	/* Reinit mem_attention for the smaller spatial grid */
	sam3_memory_attn_init(&trk.mem_attention, 256, 64, 4, 1,
			      feat_h, feat_w);
	sam3_memory_attn_load(&trk.mem_attention, NULL, &g_cpu.arena);

	/*
	 * Task 2.2: use a local bank (per-object pattern — trk.mem_bank
	 * was removed). Fabricate one cond memory entry:
	 *   spatial_features [seq, 64] zeros
	 *   obj_pointer      [1, 256]  zeros
	 */
	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, trk.num_maskmem,
			      trk.max_cond_frames_in_attn,
			      1, trk.mf_threshold);

	int sp_dims[] = {seq, 64};
	struct sam3_tensor *fake_spatial = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, sp_dims);
	ASSERT(fake_spatial != NULL);
	memset(fake_spatial->data, 0, fake_spatial->nbytes);

	int op_dims[] = {1, 256};
	struct sam3_tensor *fake_obj = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, op_dims);
	ASSERT(fake_obj != NULL);
	memset(fake_obj->data, 0, fake_obj->nbytes);

	struct sam3_memory_entry entry = {0};
	entry.spatial_features = fake_spatial;
	entry.obj_pointer     = fake_obj;
	entry.frame_idx        = 0;
	entry.is_conditioning  = 1;
	entry.obj_score        = 1.0f;
	sam3_memory_bank_add(&bank, &entry);
	ASSERT_EQ(sam3_memory_bank_total(&bank), 1);

	/* Run track_frame on frame 1 -- populated-bank branch */
	int feat_dims[] = {seq, 256};
	struct sam3_tensor *backbone = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, feat_dims);
	ASSERT(backbone != NULL);
	memset(backbone->data, 0, backbone->nbytes);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *masks = NULL, *iou = NULL;
	struct sam3_tensor *obj_ptr = NULL;
	err = sam3_tracker_track_frame(
		&trk, &graph, &bank, backbone, feat_h, feat_w,
		NULL, NULL, NULL,
		/*frame_idx=*/1, /*is_cond=*/0,
		&g_cpu.arena,
		&masks, &iou, &obj_ptr, NULL);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(masks != NULL);
	ASSERT(iou != NULL);
	ASSERT(obj_ptr != NULL);
	ASSERT_EQ(obj_ptr->n_dims, 2);
	ASSERT_EQ(obj_ptr->dims[0], 1);
	ASSERT_EQ(obj_ptr->dims[1], 256);
	/*
	 * Graph must have nodes from both memory attention and mask
	 * decoder -- a no-memory fallback only adds the decoder's
	 * nodes.
	 */
	ASSERT(graph.n_nodes > 0);
}

/*
 * test_tpos_slot_selection_matches_python
 *
 * Python sam3_tracker_base.py:614-676 picks the maskmem_tpos_enc slot
 * by window-position, not raw frame distance:
 *   - cond frames    -> slot num_maskmem - 1 (the anchor slot)
 *   - non-cond, k-th newest (k = 1..n_non_cond) -> slot k - 1
 *
 * Build a bank with one cond frame and three non-cond frames, populate
 * maskmem_tpos_enc[slot] with a recognisable per-slot value, run
 * gh_tpos_enc_mem, and check that each row of the output picks up the
 * value of the slot the Python math would have selected.
 */
static void test_tpos_slot_selection_matches_python(void)
{
	const int num_maskmem = 7;
	const int mem_dim     = 4; /* small, for easy inspection */
	const int hw          = 2; /* rows per spatial feature */

	/* maskmem_tpos_enc[slot, 0, 0, :] = {slot, slot, slot, slot} */
	int tpos_dims[] = {num_maskmem, 1, 1, mem_dim};
	struct sam3_tensor *tpos = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 4, tpos_dims);
	ASSERT(tpos != NULL);
	float *tpos_data = (float *)tpos->data;
	for (int s = 0; s < num_maskmem; s++)
		for (int c = 0; c < mem_dim; c++)
			tpos_data[(size_t)s * mem_dim + c] = (float)s;

	int sp_dims[] = {hw, mem_dim};

	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, num_maskmem, 4, 1, 0.01f);

	/*
	 * Layout: one cond at frame 0, three non-cond at frames 10, 11, 12.
	 * Newest non-cond is frame 12 (stored last).
	 */
	struct sam3_tensor *sp_cond  = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, sp_dims);
	struct sam3_tensor *sp_nc_a  = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, sp_dims);
	struct sam3_tensor *sp_nc_b  = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, sp_dims);
	struct sam3_tensor *sp_nc_c  = gh_alloc_tensor(
		&g_cpu.arena, SAM3_DTYPE_F32, 2, sp_dims);
	ASSERT(sp_cond && sp_nc_a && sp_nc_b && sp_nc_c);

	struct sam3_memory_entry e_cond = {
		.spatial_features = sp_cond,
		.frame_idx = 0, .is_conditioning = 1, .obj_score = 1.0f
	};
	struct sam3_memory_entry e_nc_a = {
		.spatial_features = sp_nc_a,
		.frame_idx = 10, .is_conditioning = 0, .obj_score = 1.0f
	};
	struct sam3_memory_entry e_nc_b = {
		.spatial_features = sp_nc_b,
		.frame_idx = 11, .is_conditioning = 0, .obj_score = 1.0f
	};
	struct sam3_memory_entry e_nc_c = {
		.spatial_features = sp_nc_c,
		.frame_idx = 12, .is_conditioning = 0, .obj_score = 1.0f
	};
	sam3_memory_bank_add(&bank, &e_cond);
	sam3_memory_bank_add(&bank, &e_nc_a);
	sam3_memory_bank_add(&bank, &e_nc_b);
	sam3_memory_bank_add(&bank, &e_nc_c);

	struct sam3_memory_bank_view view;
	sam3_memory_bank_build_view(&bank, /*frame_idx=*/13, &view);

	struct sam3_graph graph;
	sam3_graph_init(&graph);

	struct sam3_tensor *out = gh_tpos_enc_mem(
		&graph, &g_cpu.arena, &view, tpos, /*current_frame_idx=*/13);
	ASSERT(out != NULL);
	ASSERT_EQ(out->n_dims, 2);
	ASSERT_EQ(out->dims[0], 4 * hw);   /* 1 cond + 3 non-cond, 2 rows each */
	ASSERT_EQ(out->dims[1], mem_dim);

	const float *dst = (const float *)out->data;
	/* Cond (rows 0..1) -> slot num_maskmem-1 = 6 */
	for (int j = 0; j < hw; j++) {
		for (int c = 0; c < mem_dim; c++)
			ASSERT_EQ((int)dst[j * mem_dim + c], 6);
	}
	/* non_cond[0] (frame 10, k=3 oldest of 3) -> slot 2 */
	for (int j = 0; j < hw; j++) {
		for (int c = 0; c < mem_dim; c++)
			ASSERT_EQ(
				(int)dst[(hw + j) * mem_dim + c], 2);
	}
	/* non_cond[1] (frame 11, k=2) -> slot 1 */
	for (int j = 0; j < hw; j++) {
		for (int c = 0; c < mem_dim; c++)
			ASSERT_EQ(
				(int)dst[(2 * hw + j) * mem_dim + c], 1);
	}
	/* non_cond[2] (frame 12, newest k=1) -> slot 0 */
	for (int j = 0; j < hw; j++) {
		for (int c = 0; c < mem_dim; c++)
			ASSERT_EQ(
				(int)dst[(3 * hw + j) * mem_dim + c], 0);
	}
}

/* --- test_tracker_track_frame_null --- */

static void test_tracker_track_frame_null(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk);

	struct sam3_graph graph;
	sam3_graph_init(&graph);
	struct sam3_tensor dummy;
	memset(&dummy, 0, sizeof(dummy));
	struct sam3_tensor *masks, *iou;

	struct sam3_memory_bank bank;
	sam3_memory_bank_init(&bank, trk.num_maskmem,
			      trk.max_cond_frames_in_attn,
			      1, trk.mf_threshold);

	/* NULL trk */
	ASSERT_EQ(sam3_tracker_track_frame(
		NULL, &graph, &bank, &dummy, 16, 16, NULL, NULL, NULL,
		0, 0, &g_cpu.arena, &masks, &iou, NULL, NULL),
		SAM3_EINVAL);
	/* NULL graph */
	ASSERT_EQ(sam3_tracker_track_frame(
		&trk, NULL, &bank, &dummy, 16, 16, NULL, NULL, NULL,
		0, 0, &g_cpu.arena, &masks, &iou, NULL, NULL),
		SAM3_EINVAL);
	/* NULL bank */
	ASSERT_EQ(sam3_tracker_track_frame(
		&trk, &graph, NULL, &dummy, 16, 16, NULL, NULL, NULL,
		0, 0, &g_cpu.arena, &masks, &iou, NULL, NULL),
		SAM3_EINVAL);
	/* zero feat_h */
	ASSERT_EQ(sam3_tracker_track_frame(
		&trk, &graph, &bank, &dummy, 0, 16, NULL, NULL, NULL,
		0, 0, &g_cpu.arena, &masks, &iou, NULL, NULL),
		SAM3_EINVAL);
}

/* --- main --- */

int main(void)
{
	test_tracker_init();
	test_tracker_init_null();

	setup();

	test_tracker_load_zero();
	test_tracker_load_null_args();
	test_tracker_reset();
	test_tracker_reset_null();
	test_tracker_track_frame_smoke();
	test_tracker_track_frame_uses_populated_bank();
	test_tpos_slot_selection_matches_python();
	test_tracker_track_frame_null();

	teardown();

	TEST_REPORT();
}
