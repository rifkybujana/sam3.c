/*
 * tests/test_tracker_v2_load.c - Round-trip sam3_tracker_v2 weight load.
 *
 * Opens models/sam3.1.sam3 (skips if absent), runs sam3_tracker_v2_init
 * + sam3_tracker_v2_load, and asserts that every phase-2.1 tensor
 * resolved to a non-NULL tensor with the expected shape. The test does
 * not exercise any forward graph — forward functions land in phase 2.2+.
 *
 * Key types:  sam3_tracker_v2
 * Depends on: model/tracker_v2.h, core/weight.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "core/weight.h"
#include "core/alloc.h"
#include "model/tracker_v2.h"
#include "test_helpers.h"

#ifndef SAM3_SOURCE_DIR
#error "SAM3_SOURCE_DIR must be defined by CMake"
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.1.sam3"

static void assert_tensor(const struct sam3_tensor *t, const char *label,
			  int expected_dims, ...)
{
	if (!t) {
		printf("  FAIL: %s is NULL\n", label);
		ASSERT(t != NULL);
		return;
	}
	ASSERT_EQ(t->n_dims, expected_dims);
	va_list ap;
	va_start(ap, expected_dims);
	for (int i = 0; i < expected_dims; i++) {
		int expected = va_arg(ap, int);
		if (t->dims[i] != expected) {
			printf("  FAIL: %s dim[%d] %d != %d\n",
			       label, i, t->dims[i], expected);
		}
		ASSERT_EQ(t->dims[i], expected);
	}
	va_end(ap);
}

int main(void)
{
	if (access(MODEL_PATH, F_OK) != 0) {
		printf("test_tracker_v2_load: SKIP (model missing at %s)\n",
		       MODEL_PATH);
		return 0;
	}

	struct sam3_weight_file wf;
	memset(&wf, 0, sizeof(wf));
	ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);

	struct sam3_arena arena;
	memset(&arena, 0, sizeof(arena));
	ASSERT_EQ(sam3_arena_init(&arena, 1024 * 1024), SAM3_OK);

	struct sam3_tracker_v2 trk;
	ASSERT_EQ(sam3_tracker_v2_init(&trk), SAM3_OK);
	ASSERT_EQ(sam3_tracker_v2_load(&trk, &wf, &arena), SAM3_OK);

	/* ── Maskmem mask downsampler (4 conv stages + final 1x1) ─────── */
	printf("  maskmem.mask_downsampler\n");
	int chans[5] = {32, 16, 64, 256, 1024};
	for (int s = 0; s < 4; s++) {
		assert_tensor(trk.maskmem.mask_downsampler.conv_w[s],
			       "mask_ds.conv_w", 4,
			       chans[s + 1], 3, 3, chans[s]);
		assert_tensor(trk.maskmem.mask_downsampler.conv_b[s],
			       "mask_ds.conv_b", 1, chans[s + 1]);
		assert_tensor(trk.maskmem.mask_downsampler.norm_w[s],
			       "mask_ds.norm_w", 1, chans[s + 1]);
		assert_tensor(trk.maskmem.mask_downsampler.norm_b[s],
			       "mask_ds.norm_b", 1, chans[s + 1]);
	}
	assert_tensor(trk.maskmem.mask_downsampler.proj_w,
		       "mask_ds.proj_w", 4, 256, 1, 1, 1024);
	assert_tensor(trk.maskmem.mask_downsampler.proj_b,
		       "mask_ds.proj_b", 1, 256);

	/* ── Maskmem pix_feat_proj + fuser ──────────────────────────────── */
	printf("  maskmem.pix_feat_proj + fuser\n");
	assert_tensor(trk.maskmem.pix_feat_proj_w,
		       "maskmem.pix_feat_proj_w", 4, 256, 1, 1, 256);
	assert_tensor(trk.maskmem.pix_feat_proj_b,
		       "maskmem.pix_feat_proj_b", 1, 256);
	for (int i = 0; i < 2; i++) {
		struct sam3_v2_cxblock *blk = &trk.maskmem.fuser[i];
		assert_tensor(blk->dwconv_w, "fuser.dwconv_w", 4,
			       256, 7, 7, 1);
		assert_tensor(blk->dwconv_b, "fuser.dwconv_b", 1, 256);
		assert_tensor(blk->norm_w,   "fuser.norm_w",   1, 256);
		assert_tensor(blk->norm_b,   "fuser.norm_b",   1, 256);
		assert_tensor(blk->pwconv1_w, "fuser.pwconv1_w", 2, 1024, 256);
		assert_tensor(blk->pwconv1_b, "fuser.pwconv1_b", 1, 1024);
		assert_tensor(blk->pwconv2_w, "fuser.pwconv2_w", 2, 256, 1024);
		assert_tensor(blk->pwconv2_b, "fuser.pwconv2_b", 1, 256);
		assert_tensor(blk->gamma,    "fuser.gamma",    1, 256);
	}

	/* ── obj_ptr_proj (3-layer MLP) ─────────────────────────────────── */
	printf("  obj_ptr_proj\n");
	for (int i = 0; i < 3; i++) {
		assert_tensor(trk.obj_ptr_proj.fc_w[i], "obj_ptr.fc_w", 2,
			       256, 256);
		assert_tensor(trk.obj_ptr_proj.fc_b[i], "obj_ptr.fc_b", 1,
			       256);
	}

	/* ── Small projections ──────────────────────────────────────────── */
	printf("  obj_ptr_tpos_proj + no_obj_ptr_linear\n");
	assert_tensor(trk.obj_ptr_tpos_proj_w, "obj_ptr_tpos.w", 2, 256, 256);
	assert_tensor(trk.obj_ptr_tpos_proj_b, "obj_ptr_tpos.b", 1, 256);
	assert_tensor(trk.no_obj_ptr_linear_w, "no_obj_ptr.w", 2, 256, 256);
	assert_tensor(trk.no_obj_ptr_linear_b, "no_obj_ptr.b", 1, 256);

	/* ── Memory-attention transformer (phase 2.3a) ──────────────────── */
	printf("  transformer.layers (4 decoupled layers)\n");
	for (int li = 0; li < 4; li++) {
		struct sam3_v2_memory_attn_layer *L =
			&trk.transformer.layers[li];

		/* self_attn QKV + out: all [256, 256] + [256] */
		assert_tensor(L->self_q_w, "self_q_w", 2, 256, 256);
		assert_tensor(L->self_q_b, "self_q_b", 1, 256);
		assert_tensor(L->self_out_w, "self_out_w", 2, 256, 256);

		/* cross_attn QKV + out */
		assert_tensor(L->cross_q_w, "cross_q_w", 2, 256, 256);
		assert_tensor(L->cross_v_w, "cross_v_w", 2, 256, 256);
		assert_tensor(L->cross_out_w, "cross_out_w", 2, 256, 256);

		/* image_cross_attn: q and k only */
		assert_tensor(L->img_q_w, "img_q_w", 2, 256, 256);
		assert_tensor(L->img_k_w, "img_k_w", 2, 256, 256);

		/* FFN */
		assert_tensor(L->lin1_w, "lin1_w", 2, 2048, 256);
		assert_tensor(L->lin1_b, "lin1_b", 1, 2048);
		assert_tensor(L->lin2_w, "lin2_w", 2, 256, 2048);

		/* 3 LayerNorms */
		assert_tensor(L->norm1_w, "norm1_w", 1, 256);
		assert_tensor(L->norm2_w, "norm2_w", 1, 256);
		assert_tensor(L->norm3_w, "norm3_w", 1, 256);
	}
	assert_tensor(trk.transformer.final_norm_w, "final_norm_w", 1, 256);
	assert_tensor(trk.transformer.final_norm_b, "final_norm_b", 1, 256);

	/* ── Singletons ─────────────────────────────────────────────────── */
	printf("  singleton embeddings\n");
	assert_tensor(trk.image_pe_gauss, "image_pe_gauss", 2, 2, 128);
	assert_tensor(trk.maskmem_tpos_enc, "maskmem_tpos_enc", 4,
		       7, 1, 1, 256);
	assert_tensor(trk.no_obj_embed_spatial, "no_obj_embed_spatial", 2,
		       16, 256);
	assert_tensor(trk.output_valid_embed, "output_valid_embed", 2,
		       16, 256);
	assert_tensor(trk.output_invalid_embed, "output_invalid_embed", 2,
		       16, 256);
	assert_tensor(trk.interactivity_no_mem_embed,
		       "interactivity_no_mem_embed", 3, 1, 1, 256);

	sam3_arena_free(&arena);
	sam3_weight_close(&wf);

	printf("test_tracker_v2_load: PASS (phase-2.3a all %d tensors)\n",
	       SAM3_V2_PHASE_2_3A_TENSORS);
	return 0;
}
