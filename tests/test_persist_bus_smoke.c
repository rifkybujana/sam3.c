/*
 * tests/test_persist_bus_smoke.c - End-to-end cache persistence smoke test
 *
 * Loads assets/bus.jpg, runs the image encoder, saves the resulting
 * feature bundle to a .sam3cache file, and repeats for the text prompts
 * "person" and "bus". Then opens a fresh context, restores all three
 * cache files from disk, and runs segmentation for each prompt against
 * the reloaded cache. The test reports the on-disk size of each
 * compressed entry as a percentage of the raw feature bytes so the
 * real-world compression ratio is visible in the log.
 *
 * Skips silently when the model file is not present. Intended to be run
 * manually against a Metal release build; not a fast CTest target.
 *
 * Key types:  sam3_ctx, sam3_image, sam3_result, sam3_cache_stats
 * Depends on: test_helpers.h, sam3/sam3.h, util/image.h
 * Used by:    developers validating the .sam3cache persistence path
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "util/image.h"
#include "core/alloc.h"
#include "model/feature_cache.h"
#include "model/sam3_processor.h"
#include "model/sam3_internal.h"

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"
#define IMAGE_PATH SAM3_SOURCE_DIR "/assets/bus.jpg"
#define IMAGE_PATH2 SAM3_SOURCE_DIR "/assets/cat_2.png"
#define IMG_CACHE  "/tmp/sam3_bus_image.sam3cache"
#define TXT_PERSON "/tmp/sam3_bus_text_person.sam3cache"
#define TXT_BUS    "/tmp/sam3_bus_text_bus.sam3cache"

static long file_size_bytes(const char *path)
{
	struct stat st;
	return stat(path, &st) == 0 ? (long)st.st_size : -1;
}

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

#define TIME_START(t) double t = now_ms()
#define TIME_END(t)   (now_ms() - (t))

static int file_exists(const char *path)
{
	return access(path, F_OK) == 0;
}

static void run_text_prompt(sam3_ctx *ctx, const char *prompt,
			    const char *label)
{
	/*
	 * set_text consults the text feature cache; segment() does not.
	 * Call set_text first so the disk-restored entry is the one used.
	 */
	ASSERT_EQ(sam3_set_text(ctx, prompt), SAM3_OK);

	struct sam3_prompt p = { .type = SAM3_PROMPT_TEXT,
				 .text = prompt };
	struct sam3_result r = {0};
	enum sam3_error err = sam3_segment(ctx, &p, 1, &r);
	ASSERT_EQ(err, SAM3_OK);
	printf("  segment(%s): n_masks=%d best=%d "
	       "mask=%dx%d iou_valid=%d\n",
	       label, r.n_masks, r.best_mask,
	       r.mask_width, r.mask_height, r.iou_valid);
	ASSERT(r.n_masks > 0);
	sam3_result_free(&r);
}

static void smoke(void)
{
	if (!file_exists(MODEL_PATH)) {
		printf("  model missing at %s, skipping\n", MODEL_PATH);
		return;
	}
	if (!file_exists(IMAGE_PATH)) {
		printf("  bus.jpg missing at %s, skipping\n", IMAGE_PATH);
		return;
	}

	/* --- Phase 1: warm cache, save to disk --- */

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int img_size = sam3_get_image_size(ctx);
	ASSERT(img_size > 0);
	printf("  model image size: %d\n", img_size);

	struct sam3_image raw = {0};
	ASSERT_EQ(sam3_image_load(IMAGE_PATH, &raw), SAM3_OK);
	printf("  bus.jpg: %dx%d\n", raw.width, raw.height);

	struct sam3_image resized = {0};
	ASSERT_EQ(sam3_image_resize(&raw, &resized, img_size, img_size),
		  SAM3_OK);

	ASSERT_EQ(sam3_precache_image(ctx, resized.pixels,
				      resized.width, resized.height),
		  SAM3_OK);

	/*
	 * Probe the in-memory footprint. proc->img_cache holds per-slot
	 * arenas sized at sam3_image_cache_create() time; reading offset
	 * after a real encoder run tells us how much the peak bundle
	 * actually consumes vs the provisioned capacity.
	 */
	{
		const struct sam3_image_feature_cache *ic = ctx->proc.img_cache;
		if (ic && ic->n_slots > 0) {
			size_t used_max = 0;
			size_t cap = 0;
			for (int i = 0; i < ic->n_slots; i++) {
				if (ic->slots[i].arena.offset > used_max)
					used_max = ic->slots[i].arena.offset;
				cap = ic->slots[i].arena.size;
			}
			printf("  image slot: used=%.2f MiB / cap=%.2f MiB "
			       "(%.1f%%)\n",
			       used_max / (1024.0 * 1024.0),
			       cap / (1024.0 * 1024.0),
			       100.0 * used_max / (double)cap);
		}
	}

	ASSERT_EQ(sam3_precache_text(ctx, "person"), SAM3_OK);
	ASSERT_EQ(sam3_precache_text(ctx, "bus"),    SAM3_OK);

	{
		const struct sam3_text_feature_cache *tc = ctx->proc.txt_cache;
		if (tc && tc->n_slots > 0) {
			size_t used_max = 0;
			size_t cap = 0;
			for (int i = 0; i < tc->n_slots; i++) {
				if (tc->slots[i].arena.offset > used_max)
					used_max = tc->slots[i].arena.offset;
				cap = tc->slots[i].arena.size;
			}
			printf("  text slot:  used=%.2f KiB / cap=%.2f KiB "
			       "(%.1f%%)\n",
			       used_max / 1024.0,
			       cap / 1024.0,
			       100.0 * used_max / (double)cap);
		}
	}

	TIME_START(t_save_img);
	ASSERT_EQ(sam3_cache_save_image(ctx, resized.pixels,
					resized.width, resized.height,
					IMG_CACHE),
		  SAM3_OK);
	double save_img_ms = TIME_END(t_save_img);
	TIME_START(t_save_txt);
	ASSERT_EQ(sam3_cache_save_text(ctx, "person", TXT_PERSON), SAM3_OK);
	ASSERT_EQ(sam3_cache_save_text(ctx, "bus",    TXT_BUS),    SAM3_OK);
	double save_txt_ms = TIME_END(t_save_txt);
	printf("  timing: cache_save_image = %.2f ms, "
	       "cache_save_text x2 = %.2f ms\n",
	       save_img_ms, save_txt_ms);

	long img_sz    = file_size_bytes(IMG_CACHE);
	long person_sz = file_size_bytes(TXT_PERSON);
	long bus_sz    = file_size_bytes(TXT_BUS);
	printf("  compressed sizes:\n"
	       "    image bundle: %ld bytes (%.1f KiB)\n"
	       "    text[person]: %ld bytes\n"
	       "    text[bus]:    %ld bytes\n",
	       img_sz, img_sz / 1024.0, person_sz, bus_sz);
	ASSERT(img_sz    > 0);
	ASSERT(person_sz > 0);
	ASSERT(bus_sz    > 0);

	sam3_image_free(&raw);
	sam3_free(ctx);

	/* --- Phase 2: fresh ctx, load from disk, segment --- */

	sam3_ctx *ctx2 = sam3_init();
	ASSERT_NOT_NULL(ctx2);
	ASSERT_EQ(sam3_load_model(ctx2, MODEL_PATH), SAM3_OK);

	TIME_START(t_load_img);
	ASSERT_EQ(sam3_cache_load_image(ctx2, IMG_CACHE),  SAM3_OK);
	double load_img_ms = TIME_END(t_load_img);
	TIME_START(t_load_txt);
	ASSERT_EQ(sam3_cache_load_text(ctx2,  TXT_PERSON), SAM3_OK);
	ASSERT_EQ(sam3_cache_load_text(ctx2,  TXT_BUS),    SAM3_OK);
	double load_txt_ms = TIME_END(t_load_txt);
	printf("  timing: cache_load_image = %.2f ms, "
	       "cache_load_text x2 = %.2f ms\n",
	       load_img_ms, load_txt_ms);

	struct sam3_cache_stats before = {0};
	sam3_cache_stats(ctx2, &before);

	/*
	 * set_image with the *same* pixel bytes must hit the cache restored
	 * from disk instead of re-running the image encoder.
	 */
	TIME_START(t_set);
	ASSERT_EQ(sam3_set_image(ctx2, resized.pixels,
				 resized.width, resized.height),
		  SAM3_OK);
	double set_ms = TIME_END(t_set);
	printf("  timing: set_image (disk-restored cache hit) = %.2f ms\n",
	       set_ms);

	struct sam3_cache_stats after_set = {0};
	sam3_cache_stats(ctx2, &after_set);
	printf("  after set_image: image_hits=%llu misses=%llu\n",
	       (unsigned long long)after_set.image_hits,
	       (unsigned long long)after_set.image_misses);
	ASSERT(after_set.image_hits  > before.image_hits);

	run_text_prompt(ctx2, "person", "person");
	run_text_prompt(ctx2, "bus",    "bus");

	struct sam3_cache_stats final_stats = {0};
	sam3_cache_stats(ctx2, &final_stats);
	printf("  final: image_hits=%llu image_misses=%llu "
	       "text_hits=%llu text_misses=%llu\n",
	       (unsigned long long)final_stats.image_hits,
	       (unsigned long long)final_stats.image_misses,
	       (unsigned long long)final_stats.text_hits,
	       (unsigned long long)final_stats.text_misses);

	/*
	 * Both text prompts were restored from disk; segment() calls must
	 * find them without re-encoding.
	 */
	ASSERT(final_stats.text_hits >= 2u);

	sam3_image_free(&resized);
	sam3_free(ctx2);

	remove(IMG_CACHE);
	remove(TXT_PERSON);
	remove(TXT_BUS);
}

/*
 * tier_smoke - End-to-end exercise of the LRU tiered-spill policy.
 * Precache two distinct images so the second registration demotes the
 * first, then set_image on the first triggers a promote + demote
 * cycle. Verifies demote/promote counters advance and both bundles
 * survive the compress/decompress round trip well enough to produce
 * valid segmentation output.
 */
static void tier_smoke(void)
{
	if (!file_exists(MODEL_PATH) || !file_exists(IMAGE_PATH) ||
	    !file_exists(IMAGE_PATH2)) {
		printf("  tier: input missing, skipping\n");
		return;
	}

	/*
	 * Force n_hot_max=1 so registering the second image demotes the
	 * first. Without this, the default 1 GiB budget accommodates both
	 * images in RAM and no spill would occur.
	 */
	struct sam3_cache_opts opts = {0};
	opts.image_mem_budget_bytes = 256UL * 1024 * 1024;
	sam3_ctx *ctx = sam3_init_ex(&opts);
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int img_size = sam3_get_image_size(ctx);

	/* Load + resize both images. */
	struct sam3_image bus_raw = {0}, bus_rs = {0};
	ASSERT_EQ(sam3_image_load(IMAGE_PATH, &bus_raw), SAM3_OK);
	ASSERT_EQ(sam3_image_resize(&bus_raw, &bus_rs, img_size, img_size),
		  SAM3_OK);
	sam3_image_free(&bus_raw);

	struct sam3_image cat_raw = {0}, cat_rs = {0};
	ASSERT_EQ(sam3_image_load(IMAGE_PATH2, &cat_raw), SAM3_OK);
	ASSERT_EQ(sam3_image_resize(&cat_raw, &cat_rs, img_size, img_size),
		  SAM3_OK);
	sam3_image_free(&cat_raw);

	struct sam3_image_feature_cache *ic = ctx->proc.img_cache;
	ASSERT_NOT_NULL(ic);

	/* Cold miss: encoder runs for bus. No demote yet (only 1 populated). */
	TIME_START(t_bus_miss);
	ASSERT_EQ(sam3_precache_image(ctx, bus_rs.pixels, bus_rs.width,
				      bus_rs.height),
		  SAM3_OK);
	double bus_miss_ms = TIME_END(t_bus_miss);
	ASSERT_EQ(ic->demotions, 0u);
	printf("  timing: precache(bus) miss = %.2f ms\n", bus_miss_ms);

	/*
	 * Cold miss + implicit demote: cat encoder runs, then register
	 * demotes the previous bus slot. bus_reg_delta captures the
	 * *extra* cost the demote adds on top of a plain miss.
	 */
	TIME_START(t_cat_miss);
	ASSERT_EQ(sam3_precache_image(ctx, cat_rs.pixels, cat_rs.width,
				      cat_rs.height),
		  SAM3_OK);
	double cat_miss_ms = TIME_END(t_cat_miss);
	ASSERT(ic->demotions >= 1u);
	printf("  timing: precache(cat) miss + demote(bus) = %.2f ms "
	       "(demote delta ≈ %.2f ms)\n",
	       cat_miss_ms, cat_miss_ms - bus_miss_ms);

	int hot = 0, cold = 0;
	for (int i = 0; i < ic->n_slots; i++) {
		if (ic->slots[i].hash == 0) continue;
		if (ic->slots[i].disk_path) cold++;
		else                        hot++;
	}
	printf("  tier: hot=%d cold=%d (of %d populated)\n",
	       hot, cold, hot + cold);
	ASSERT_EQ(hot,  1);
	ASSERT_EQ(cold, 1);

	/* set_image on the currently-hot slot — should be ~0 ms. */
	TIME_START(t_hot_hit);
	ASSERT_EQ(sam3_set_image(ctx, cat_rs.pixels, cat_rs.width,
				 cat_rs.height),
		  SAM3_OK);
	double hot_hit_ms = TIME_END(t_hot_hit);
	printf("  timing: set_image(cat) hot hit    = %.3f ms\n", hot_hit_ms);

	/*
	 * Cold hit: bus is currently spilled. set_image must promote bus
	 * (decompress ~130 MiB) and demote cat (compress ~130 MiB).
	 */
	TIME_START(t_cold_hit);
	ASSERT_EQ(sam3_set_image(ctx, bus_rs.pixels, bus_rs.width,
				 bus_rs.height),
		  SAM3_OK);
	double cold_hit_ms = TIME_END(t_cold_hit);
	ASSERT(ic->promotions >= 1u);
	printf("  timing: set_image(bus) cold hit  = %.2f ms "
	       "(promote+demote)\n", cold_hit_ms);

	TIME_START(t_seg1);
	ASSERT_EQ(sam3_set_text(ctx, "person"), SAM3_OK);
	struct sam3_prompt p1 = { .type = SAM3_PROMPT_TEXT, .text = "person" };
	struct sam3_result r1 = {0};
	ASSERT_EQ(sam3_segment(ctx, &p1, 1, &r1), SAM3_OK);
	double seg1_ms = TIME_END(t_seg1);
	printf("  timing: set_text + segment(person) = %.2f ms "
	       "(masks=%d)\n", seg1_ms, r1.n_masks);
	ASSERT(r1.n_masks > 0);
	sam3_result_free(&r1);

	/* Second cold hit, reverse direction. */
	TIME_START(t_cold_hit2);
	ASSERT_EQ(sam3_set_image(ctx, cat_rs.pixels, cat_rs.width,
				 cat_rs.height),
		  SAM3_OK);
	double cold_hit2_ms = TIME_END(t_cold_hit2);
	ASSERT(ic->promotions >= 2u);
	printf("  timing: set_image(cat) cold hit  = %.2f ms "
	       "(promote+demote)\n", cold_hit2_ms);

	TIME_START(t_seg2);
	ASSERT_EQ(sam3_set_text(ctx, "cat"), SAM3_OK);
	struct sam3_prompt p2 = { .type = SAM3_PROMPT_TEXT, .text = "cat" };
	struct sam3_result r2 = {0};
	ASSERT_EQ(sam3_segment(ctx, &p2, 1, &r2), SAM3_OK);
	double seg2_ms = TIME_END(t_seg2);
	printf("  timing: set_text + segment(cat)    = %.2f ms "
	       "(masks=%d)\n", seg2_ms, r2.n_masks);
	ASSERT(r2.n_masks > 0);
	sam3_result_free(&r2);

	printf("  tier: final demotions=%llu promotions=%llu\n",
	       (unsigned long long)ic->demotions,
	       (unsigned long long)ic->promotions);

	sam3_image_free(&bus_rs);
	sam3_image_free(&cat_rs);
	sam3_free(ctx);
}

int main(void)
{
	smoke();
	tier_smoke();
	TEST_REPORT();
}
