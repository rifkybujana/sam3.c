/*
 * src/model/feature_cache.h - LRU caches in front of the image and text
 * encoders.
 *
 * Two independent LRU caches keyed by FNV-1a 64-bit content hash. The
 * image cache holds bundles of 8 cached_* tensor pointers (one per
 * scale) inside per-slot arenas. The text cache holds [n_tokens, d]
 * feature tensors inside fixed-size cells of a shared arena. Both are
 * thread-safe-by-construction: only the main thread mutates the cache
 * tables; the text encoder worker writes into a pre-claimed slot region
 * and is joined before the main thread inspects it.
 *
 * Key types:  sam3_image_feature_cache, sam3_text_feature_cache,
 *             sam3_image_bundle, sam3_text_bundle
 * Depends on: core/alloc.h, core/tensor.h
 * Used by:    src/model/sam3_processor.c, src/sam3.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_FEATURE_CACHE_H
#define SAM3_MODEL_FEATURE_CACHE_H

#include <stdint.h>
#include <stddef.h>

#include "core/alloc.h"
#include "core/tensor.h"

/* Defined in <sam3/sam3.h> in Task 7; forward-declared here so the
 * cache module compiles before the public API task is wired in. */
struct sam3_cache_stats {
	uint64_t image_hits, image_misses, image_evictions;
	uint64_t text_hits,  text_misses,  text_evictions;
};

#define SAM3_IMAGE_CACHE_DEFAULT_SLOTS 3
#define SAM3_IMAGE_CACHE_MAX_SLOTS     16
#define SAM3_TEXT_CACHE_DEFAULT_SLOTS  16
#define SAM3_TEXT_CACHE_MAX_SLOTS      64
#define SAM3_CACHE_PREFIX_BYTES        256

/* --- image bundle --- */

struct sam3_image_bundle {
	struct sam3_tensor *image_features;
	struct sam3_tensor *feat_s0_nhwc;
	struct sam3_tensor *feat_s1_nhwc;
	struct sam3_tensor *feat_4x_nhwc;
	struct sam3_tensor *sam2_05x_nhwc;
	struct sam3_tensor *sam2_1x_nhwc;
	struct sam3_tensor *sam2_2x_nhwc;
	struct sam3_tensor *sam2_4x_nhwc;
	int prompt_w;
	int prompt_h;
	int width;
	int height;
};

struct sam3_image_cache_slot {
	uint64_t hash;             /* 0 == empty */
	uint64_t lru_tick;
	struct sam3_arena arena;
	struct sam3_image_bundle bundle;
	uint8_t prefix[SAM3_CACHE_PREFIX_BYTES];
	size_t prefix_len;
};

struct sam3_image_feature_cache {
	int n_slots;
	uint64_t next_tick;
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	struct sam3_image_cache_slot *slots;
};

/* --- text bundle --- */

struct sam3_text_bundle {
	struct sam3_tensor *features; /* [n_tokens, d_model] */
	int n_tokens;
};

struct sam3_text_cache_slot {
	uint64_t hash;
	uint64_t lru_tick;
	struct sam3_text_bundle bundle;
	int32_t prefix_tokens[SAM3_CACHE_PREFIX_BYTES / 4];
	int     prefix_len;
	size_t  arena_offset;
};

struct sam3_text_feature_cache {
	int n_slots;
	uint64_t next_tick;
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	struct sam3_arena arena;
	size_t slot_bytes;
	struct sam3_text_cache_slot *slots;
};

/* --- image cache lifecycle --- */

struct sam3_image_feature_cache *
sam3_image_cache_create(int n_slots, size_t slot_arena_bytes);
void sam3_image_cache_destroy(struct sam3_image_feature_cache *c);
int  sam3_image_cache_n_slots(const struct sam3_image_feature_cache *c);
void sam3_image_cache_clear(struct sam3_image_feature_cache *c);
void sam3_image_cache_stats(const struct sam3_image_feature_cache *c,
			    struct sam3_cache_stats *out);

int sam3_image_cache_lookup(struct sam3_image_feature_cache *c,
			    uint64_t key,
			    const uint8_t *verify_prefix,
			    size_t verify_len);
int sam3_image_cache_claim_slot(struct sam3_image_feature_cache *c);
void sam3_image_cache_register(struct sam3_image_feature_cache *c, int idx,
			       uint64_t key,
			       const uint8_t *verify_prefix,
			       size_t verify_len,
			       const struct sam3_image_bundle *bundle);

/* --- text cache lifecycle --- */

struct sam3_text_feature_cache *
sam3_text_cache_create(int n_slots, size_t slot_bytes);
void sam3_text_cache_destroy(struct sam3_text_feature_cache *c);
int  sam3_text_cache_n_slots(const struct sam3_text_feature_cache *c);
void sam3_text_cache_clear(struct sam3_text_feature_cache *c);
void sam3_text_cache_stats(const struct sam3_text_feature_cache *c,
			   struct sam3_cache_stats *out);

int  sam3_text_cache_lookup(struct sam3_text_feature_cache *c,
			    uint64_t key,
			    const int32_t *verify_tokens,
			    int verify_len);
int  sam3_text_cache_claim_slot(struct sam3_text_feature_cache *c);
void sam3_text_cache_register(struct sam3_text_feature_cache *c, int idx,
			      uint64_t key,
			      const int32_t *verify_tokens,
			      int verify_len,
			      const struct sam3_text_bundle *bundle);

struct sam3_arena *
sam3_text_cache_slot_arena(struct sam3_text_feature_cache *c, int idx);

#endif /* SAM3_MODEL_FEATURE_CACHE_H */
