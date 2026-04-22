/*
 * src/model/feature_cache.h - LRU caches in front of the image and text
 * encoders.
 *
 * Two independent LRU caches keyed by FNV-1a 64-bit content hash. Each
 * cache owns one sam3_arena per slot — the image cache stores bundles
 * of 8 cached_* tensor pointers per scale, the text cache stores a
 * single [n_tokens, d_model] feature tensor. Thread-safe-by-
 * construction: only the main thread mutates the cache tables; the
 * text encoder worker writes into its pre-claimed slot's arena and is
 * joined before the main thread inspects it.
 *
 * Key types:  sam3_image_feature_cache, sam3_text_feature_cache,
 *             sam3_image_bundle, sam3_text_bundle
 * Depends on: core/alloc.h, core/tensor.h, sam3/sam3.h
 * Used by:    src/model/sam3_processor.c, src/sam3.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_FEATURE_CACHE_H
#define SAM3_MODEL_FEATURE_CACHE_H

#include <stdint.h>
#include <stddef.h>

#include "sam3/sam3.h"
#include "core/alloc.h"
#include "core/tensor.h"

#define SAM3_IMAGE_CACHE_DEFAULT_SLOTS 8
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

/*
 * Each slot is in one of three tiers:
 *   EMPTY: hash == 0, disk_path == NULL, arena live (reset).
 *   HOT:   hash != 0, disk_path == NULL, arena.base != NULL.
 *   DISK:  hash != 0, disk_path != NULL, arena.base == NULL.
 *
 * Memory budget caps the number of HOT slots at any time. When the
 * budget is saturated and a new entry needs RAM (either register or a
 * cold-lookup promote), the LRU HOT slot is demoted to DISK — its
 * bundle is streamed out to a file under spill_dir and its 256 MiB
 * arena is freed. On lookup hit, DISK slots are promoted back into a
 * fresh arena and their spill file is unlinked.
 */
struct sam3_image_cache_slot {
	uint64_t hash;             /* 0 == empty */
	uint64_t lru_tick;
	struct sam3_arena arena;
	size_t arena_bytes;        /* capacity to re-init on promote */
	struct sam3_image_bundle bundle;
	uint8_t prefix[SAM3_CACHE_PREFIX_BYTES];
	size_t prefix_len;
	char   *disk_path;         /* non-NULL iff DISK tier (malloc'd) */
};

struct sam3_image_feature_cache {
	int n_slots;
	int n_hot_max;             /* 0 == unlimited (all slots stay HOT) */
	char *spill_dir;           /* directory for DISK tier files */
	int  owns_spill_dir;       /* 1 iff we mkdir'd it and should rmdir */
	uint64_t next_tick;
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	uint64_t demotions;
	uint64_t promotions;
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
	struct sam3_arena arena;
	struct sam3_text_bundle bundle;
	int32_t prefix_tokens[SAM3_CACHE_PREFIX_BYTES / 4];
	int     prefix_len;
};

struct sam3_text_feature_cache {
	int n_slots;
	uint64_t next_tick;
	uint64_t hits;
	uint64_t misses;
	uint64_t evictions;
	struct sam3_text_cache_slot *slots;
};

/* --- image cache lifecycle --- */

struct sam3_image_feature_cache *
sam3_image_cache_create(int n_slots, size_t slot_arena_bytes);

/*
 * sam3_image_cache_create_ex - Create an image feature cache with a
 * memory budget cap. When @mem_budget_bytes > 0, at most
 * (mem_budget_bytes / slot_arena_bytes) slots stay HOT; additional
 * populated slots are transparently spilled to @spill_dir as
 * uncompressed bundle files. Pass 0 for @mem_budget_bytes (or NULL
 * for @spill_dir) to match the old all-hot behavior.
 *
 * If @spill_dir is NULL, a per-process directory is created under
 * /tmp via mkdtemp and owned by the cache (rm'd on destroy).
 */
struct sam3_image_feature_cache *
sam3_image_cache_create_ex(int n_slots, size_t slot_arena_bytes,
			   size_t mem_budget_bytes,
			   const char *spill_dir);
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
