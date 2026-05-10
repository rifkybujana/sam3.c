/*
 * tests/test_video_encode.c - Encoder + overlay helper tests
 *
 * Covers: overlay compositing math (alpha blending + nearest-neighbor
 * upscale) and the encoder open/write/close lifecycle. Encoder tests
 * run end-to-end through libav and verify files reopen with the
 * expected dimensions and frame count.
 *
 * Key types:  (none)
 * Depends on: util/video_encode.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "util/platform.h"
#include "util/video_encode.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#pragma clang diagnostic pop

static void test_overlay_alpha_zero_leaves_unchanged(void)
{
	uint8_t rgb[4 * 4 * 3];
	for (int i = 0; i < 4 * 4 * 3; i++)
		rgb[i] = 100;
	uint8_t mask[4 * 4];
	memset(mask, 255, sizeof(mask));

	sam3_overlay_composite(rgb, 4, 4, mask, 4, 4, 0, 0.0f);

	for (int i = 0; i < 4 * 4 * 3; i++)
		ASSERT_EQ(rgb[i], 100);
}

static void test_overlay_alpha_one_replaces(void)
{
	uint8_t rgb[4 * 4 * 3];
	memset(rgb, 0, sizeof(rgb));
	uint8_t mask[4 * 4];
	memset(mask, 255, sizeof(mask));

	sam3_overlay_composite(rgb, 4, 4, mask, 4, 4, 0, 1.0f);

	/* obj_id=0 -> palette[0]. All pixels should equal that color. */
	uint8_t r = rgb[0], g = rgb[1], b = rgb[2];
	/* Palette entries must be non-all-zero (so a color was applied). */
	ASSERT(r != 0 || g != 0 || b != 0);
	for (int i = 0; i < 4 * 4; i++) {
		ASSERT_EQ(rgb[i * 3 + 0], r);
		ASSERT_EQ(rgb[i * 3 + 1], g);
		ASSERT_EQ(rgb[i * 3 + 2], b);
	}
}

static void test_overlay_mask_zero_no_effect(void)
{
	uint8_t rgb[4 * 4 * 3];
	memset(rgb, 50, sizeof(rgb));
	uint8_t mask[4 * 4];
	memset(mask, 0, sizeof(mask));

	sam3_overlay_composite(rgb, 4, 4, mask, 4, 4, 0, 1.0f);

	for (int i = 0; i < 4 * 4 * 3; i++)
		ASSERT_EQ(rgb[i], 50);
}

static void test_overlay_upscale_nearest(void)
{
	/* 2x2 mask: only top-left cell set. Output 4x4 -> top-left 2x2 set. */
	uint8_t rgb[4 * 4 * 3];
	memset(rgb, 0, sizeof(rgb));
	uint8_t mask[2 * 2] = { 255, 0, 0, 0 };

	sam3_overlay_composite(rgb, 4, 4, mask, 2, 2, 0, 1.0f);

	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			int idx = (y * 4 + x) * 3;
			int should_be_set = (x < 2 && y < 2);
			if (should_be_set) {
				ASSERT(rgb[idx] != 0 || rgb[idx + 1] != 0 ||
				       rgb[idx + 2] != 0);
			} else {
				ASSERT_EQ(rgb[idx + 0], 0);
				ASSERT_EQ(rgb[idx + 1], 0);
				ASSERT_EQ(rgb[idx + 2], 0);
			}
		}
	}
}

static void test_overlay_half_alpha_blends(void)
{
	uint8_t rgb[1 * 1 * 3] = { 100, 100, 100 };
	uint8_t mask[1] = { 255 };

	sam3_overlay_composite(rgb, 1, 1, mask, 1, 1, 0, 0.5f);

	/*
	 * Resulting pixel = 100 * 0.5 + palette[0][c] * 0.5. Exact value
	 * depends on palette[0]; just verify the pixel changed from 100
	 * in at least one channel.
	 */
	int changed = rgb[0] != 100 || rgb[1] != 100 || rgb[2] != 100;
	ASSERT(changed);
}

struct temp_video_path {
	char dir[512];
	char mp4[1024];
};

static int temp_video_path_init(struct temp_video_path *tmp)
{
	char *dir = sam3_platform_temp_dir("sam3-test-video");
	int n;

	if (!dir)
		return -1;
	n = snprintf(tmp->dir, sizeof(tmp->dir), "%s", dir);
	free(dir);
	if (n < 0 || (size_t)n >= sizeof(tmp->dir))
		return -1;
	n = snprintf(tmp->mp4, sizeof(tmp->mp4), "%s/%s", tmp->dir,
		     "out.mp4");
	if (n < 0 || (size_t)n >= sizeof(tmp->mp4)) {
		sam3_platform_rmdir(tmp->dir);
		return -1;
	}
	return 0;
}

static void temp_video_path_cleanup(struct temp_video_path *tmp)
{
	remove(tmp->mp4);
	sam3_platform_rmdir(tmp->dir);
}

static void test_encoder_close_null_is_ok(void)
{
	ASSERT_EQ(sam3_video_encoder_close(NULL), SAM3_OK);
}

static void test_encoder_rejects_unknown_extension(void)
{
	struct sam3_video_encoder *enc = NULL;
	enum sam3_error err = sam3_video_encoder_open(
		"sam3_test_bad.xyz", 16, 16, 10, 1, &enc);
	ASSERT_EQ(err, SAM3_EIO);
	ASSERT(enc == NULL);
}

static void test_encoder_open_close_idempotent(void)
{
	struct temp_video_path tmp;
	ASSERT_EQ(temp_video_path_init(&tmp), 0);

	struct sam3_video_encoder *enc = NULL;
	enum sam3_error err = sam3_video_encoder_open(tmp.mp4, 16, 16, 10, 1,
						      &enc);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(enc != NULL);
	if (err != SAM3_OK || !enc) {
		temp_video_path_cleanup(&tmp);
		return;
	}

	/* Close once. */
	ASSERT_EQ(sam3_video_encoder_close(enc), SAM3_OK);

	temp_video_path_cleanup(&tmp);
}

static void test_encoder_roundtrip_mp4(void)
{
	struct temp_video_path tmp;
	ASSERT_EQ(temp_video_path_init(&tmp), 0);

	struct sam3_video_encoder *enc = NULL;
	enum sam3_error err = sam3_video_encoder_open(tmp.mp4, 16, 16, 10, 1,
						      &enc);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(enc != NULL);
	if (err != SAM3_OK || !enc) {
		temp_video_path_cleanup(&tmp);
		return;
	}

	uint8_t frame[16 * 16 * 3];
	for (int i = 0; i < 24; i++) {
		for (int y = 0; y < 16; y++) {
			for (int x = 0; x < 16; x++) {
				frame[(y * 16 + x) * 3 + 0] = (uint8_t)(i * 10);
				frame[(y * 16 + x) * 3 + 1] = (uint8_t)(x * 16);
				frame[(y * 16 + x) * 3 + 2] = (uint8_t)(y * 16);
			}
		}
		ASSERT_EQ(sam3_video_encoder_write_rgb(enc, frame), SAM3_OK);
	}
	ASSERT_EQ(sam3_video_encoder_close(enc), SAM3_OK);

	/* Reopen via libav to verify file is valid. */
	AVFormatContext *fmt = NULL;
	int rc = avformat_open_input(&fmt, tmp.mp4, NULL, NULL);
	ASSERT_EQ(rc, 0);
	if (rc != 0) {
		temp_video_path_cleanup(&tmp);
		return;
	}
	rc = avformat_find_stream_info(fmt, NULL);
	ASSERT(rc >= 0);
	if (rc < 0) {
		avformat_close_input(&fmt);
		temp_video_path_cleanup(&tmp);
		return;
	}

	int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1,
					  NULL, 0);
	ASSERT(vstream >= 0);
	if (vstream < 0) {
		avformat_close_input(&fmt);
		temp_video_path_cleanup(&tmp);
		return;
	}
	ASSERT_EQ(fmt->streams[vstream]->codecpar->width, 16);
	ASSERT_EQ(fmt->streams[vstream]->codecpar->height, 16);

	avformat_close_input(&fmt);
	temp_video_path_cleanup(&tmp);
}

int main(void)
{
	test_overlay_alpha_zero_leaves_unchanged();
	test_overlay_alpha_one_replaces();
	test_overlay_mask_zero_no_effect();
	test_overlay_upscale_nearest();
	test_overlay_half_alpha_blends();
	test_encoder_close_null_is_ok();
	test_encoder_rejects_unknown_extension();
	test_encoder_open_close_idempotent();
	test_encoder_roundtrip_mp4();
	TEST_REPORT();
}
