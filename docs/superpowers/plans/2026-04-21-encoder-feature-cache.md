# Encoder Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-memory LRU caches in front of the SAM3 image encoder and text encoder, so an image-editing application can encode an image once and run many text-prompted segmentations without re-running encoders for repeated inputs.

**Architecture:** A new self-contained module `src/model/feature_cache.{h,c}` owns per-slot arenas for image feature bundles (default 3 slots) and a partitioned shared arena for text feature bundles (default 16 slots). Lookup is by FNV-1a 64-bit hash of input bytes/tokens. The processor calls into the cache from `set_image` / `set_text`; on hit it copies bundle pointers into `model.cached_*` and skips the encoder. The text worker thread writes its output directly into a pre-claimed cache slot, removing the `text_features_async` copy that exists today.

**Tech Stack:** C11, pthreads, existing `sam3_arena` allocator, existing `gh_alloc_tensor`, CTest.

**Spec:** `docs/superpowers/specs/2026-04-21-encoder-feature-cache-design.md`

---

## File Structure

**New files:**
- `src/util/hash.h` — FNV-1a 64-bit one-shot + incremental hash (header-only inline).
- `src/model/feature_cache.h` — public-to-the-module struct + function declarations for image and text caches.
- `src/model/feature_cache.c` — implementation.
- `tests/test_feature_cache.c` — unit tests for the cache module in isolation.
- `tests/test_processor_cache.c` — integration tests against `sam3_processor` + model.

**Modified files:**
- `include/sam3/sam3.h` — add `sam3_init_ex`, `sam3_cache_opts`, `sam3_cache_clear`, `sam3_cache_stats`.
- `src/sam3.c` — implement the new public API; flush caches in `sam3_load_model`.
- `src/model/sam3_internal.h` — add `cache_opts` field to `sam3_ctx`.
- `src/model/sam3_processor.h` — replace `text_persist_arena`, `text_features_async`, and `weights_end` with cache pointers and a `text_cached_bundle` field.
- `src/model/sam3_processor.c` — replace `weights_end` rollback with cache lookup in `set_image`; rewrite the text-worker integration to use the cache; remove the `text_features_async → model_arena` copy in `segment`.
- `tests/test_processor_async.c` — update assertions that reference removed fields (`text_thread_active`, `text_features_async`); replace with cache-stats based assertions.

---

## Task 1: FNV-1a 64-bit hash utility

**Files:**
- Create: `src/util/hash.h`
- Create: `tests/test_hash.c`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hash.c`:

```c
/*
 * tests/test_hash.c - FNV-1a 64-bit hash utility tests
 *
 * Verifies the constants and basic properties of sam3_fnv1a_64 used as
 * the cache key hash for the encoder feature cache.
 *
 * Key types:  (test-only)
 * Depends on: test_helpers.h, util/hash.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include <string.h>
#include "util/hash.h"

static void test_empty_input_returns_offset_basis(void)
{
	uint64_t h = sam3_fnv1a_64(NULL, 0, SAM3_FNV1A_64_OFFSET_BASIS);
	ASSERT_EQ(h, (uint64_t)0xcbf29ce484222325ULL);
}

static void test_known_vector_foobar(void)
{
	const char *s = "foobar";
	uint64_t h = sam3_fnv1a_64((const uint8_t *)s, strlen(s),
				   SAM3_FNV1A_64_OFFSET_BASIS);
	ASSERT_EQ(h, (uint64_t)0x85944171f73967e8ULL);
}

static void test_incremental_matches_one_shot(void)
{
	const char *s = "the quick brown fox";
	uint64_t one = sam3_fnv1a_64((const uint8_t *)s, strlen(s),
				     SAM3_FNV1A_64_OFFSET_BASIS);
	uint64_t inc = SAM3_FNV1A_64_OFFSET_BASIS;
	inc = sam3_fnv1a_64((const uint8_t *)s, 4, inc);
	inc = sam3_fnv1a_64((const uint8_t *)s + 4, strlen(s) - 4, inc);
	ASSERT_EQ(one, inc);
}

static void test_distinct_inputs_distinct_hashes(void)
{
	uint64_t a = sam3_fnv1a_64((const uint8_t *)"cat", 3,
				   SAM3_FNV1A_64_OFFSET_BASIS);
	uint64_t b = sam3_fnv1a_64((const uint8_t *)"dog", 3,
				   SAM3_FNV1A_64_OFFSET_BASIS);
	ASSERT(a != b);
}

int main(void)
{
	test_empty_input_returns_offset_basis();
	test_known_vector_foobar();
	test_incremental_matches_one_shot();
	test_distinct_inputs_distinct_hashes();
	TEST_REPORT();
}
```

- [ ] **Step 2: Run test to verify it fails (compile error)**

Run: `cd build && cmake --build . --target test_hash 2>&1 | tail -5`
Expected: compile error — `util/hash.h` does not exist.

- [ ] **Step 3: Write the implementation**

Create `src/util/hash.h`:

```c
/*
 * src/util/hash.h - FNV-1a 64-bit hash (header-only)
 *
 * One-shot and incremental hashing for the encoder feature cache.
 * 64-bit chosen so that two random images / token sequences have a
 * collision probability of ~5e-20 over a cache lifetime of millions
 * of insertions, well below the rate of single-bit memory errors.
 *
 * Key types:  (none — free functions)
 * Depends on: <stdint.h>, <stddef.h>
 * Used by:    src/model/feature_cache.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_HASH_H
#define SAM3_UTIL_HASH_H

#include <stdint.h>
#include <stddef.h>

#define SAM3_FNV1A_64_OFFSET_BASIS 0xcbf29ce484222325ULL
#define SAM3_FNV1A_64_PRIME        0x100000001b3ULL

/*
 * sam3_fnv1a_64 - FNV-1a hash, optionally seeded from a previous chunk.
 *
 * Pass SAM3_FNV1A_64_OFFSET_BASIS as @seed for a one-shot hash; pass
 * the previous return value to chain incremental updates. @data may be
 * NULL when @len == 0.
 */
static inline uint64_t sam3_fnv1a_64(const uint8_t *data, size_t len,
				     uint64_t seed)
{
	uint64_t h = seed;
	for (size_t i = 0; i < len; i++) {
		h ^= (uint64_t)data[i];
		h *= SAM3_FNV1A_64_PRIME;
	}
	return h;
}

#endif /* SAM3_UTIL_HASH_H */
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd build && cmake --build . --target test_hash && ctest -R '^test_hash$' --output-on-failure`
Expected: `4 tests, 0 failures` and `Passed`.

- [ ] **Step 5: Commit**

```bash
git add src/util/hash.h tests/test_hash.c
git commit -m "util: add FNV-1a 64-bit hash for feature cache keys"
```

---

## Task 2: Image cache module skeleton (init/free/stats)

**Files:**
- Create: `src/model/feature_cache.h`
- Create: `src/model/feature_cache.c`
- Create: `tests/test_feature_cache.c`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feature_cache.c`:

```c
/*
 * tests/test_feature_cache.c - Encoder feature cache unit tests
 *
 * Tests the standalone feature cache module: init/free, slot allocation,
 * LRU eviction, hit/miss accounting, and clear semantics. Uses synthetic
 * tensors so the tests do not require a loaded model.
 *
 * Key types:  sam3_image_feature_cache, sam3_text_feature_cache
 * Depends on: test_helpers.h, model/feature_cache.h, core/alloc.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include <string.h>
#include "model/feature_cache.h"

static void test_image_cache_init_default_slots(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(0, 1024);
	ASSERT_NOT_NULL(c);
	ASSERT_EQ(sam3_image_cache_n_slots(c), 3);
	struct sam3_cache_stats s = {0};
	sam3_image_cache_stats(c, &s);
	ASSERT_EQ(s.image_hits, 0u);
	ASSERT_EQ(s.image_misses, 0u);
	sam3_image_cache_destroy(c);
}

static void test_image_cache_clamps_slot_count(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(99, 1024);
	ASSERT_NOT_NULL(c);
	ASSERT_EQ(sam3_image_cache_n_slots(c), 16);
	sam3_image_cache_destroy(c);
}

static void test_text_cache_init_default_slots(void)
{
	struct sam3_text_feature_cache *c = sam3_text_cache_create(0, 4096);
	ASSERT_NOT_NULL(c);
	ASSERT_EQ(sam3_text_cache_n_slots(c), 16);
	sam3_text_cache_destroy(c);
}

int main(void)
{
	test_image_cache_init_default_slots();
	test_image_cache_clamps_slot_count();
	test_text_cache_init_default_slots();
	TEST_REPORT();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake --build . --target test_feature_cache 2>&1 | tail -5`
Expected: compile error — `model/feature_cache.h` does not exist.

- [ ] **Step 3: Write the header**

Create `src/model/feature_cache.h`:

```c
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

#include "core/alloc.h"
#include "core/tensor.h"
#include "sam3/sam3.h"

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
	size_t  arena_offset; /* slot's region in the shared arena */
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

/*
 * sam3_image_cache_lookup - Find a slot whose hash matches @key and
 * whose recorded prefix matches @verify_prefix (length @verify_len, up
 * to SAM3_CACHE_PREFIX_BYTES). On hit, returns the slot index and bumps
 * the LRU tick + hit counter. On miss, returns -1 and bumps the miss
 * counter.
 */
int sam3_image_cache_lookup(struct sam3_image_feature_cache *c,
			    uint64_t key,
			    const uint8_t *verify_prefix,
			    size_t verify_len);

/*
 * sam3_image_cache_claim_slot - Pick the LRU-oldest slot, reset its
 * arena, clear its hash, and return the slot index. Caller must then
 * encode into &slots[idx].arena and call sam3_image_cache_register.
 */
int sam3_image_cache_claim_slot(struct sam3_image_feature_cache *c);

/*
 * sam3_image_cache_register - Commit a freshly-encoded bundle into the
 * slot at @idx with hash @key. @verify_prefix (up to @verify_len bytes)
 * is copied for collision verification on later lookups.
 */
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

/* Get a slot's arena base pointer + capacity for the worker / encoder
 * to allocate into. @offset_out is the byte offset within
 * c->arena.base where the slot starts. */
struct sam3_arena *
sam3_text_cache_slot_arena(struct sam3_text_feature_cache *c, int idx);

#endif /* SAM3_MODEL_FEATURE_CACHE_H */
```

- [ ] **Step 4: Write the minimal implementation**

Create `src/model/feature_cache.c`:

```c
/*
 * src/model/feature_cache.c - Image and text feature LRU caches.
 *
 * Implements the cache module declared in feature_cache.h. Image cache
 * uses one arena per slot (sized for worst-case encoder output); text
 * cache uses a single shared arena partitioned into fixed-size cells.
 * All allocation outside the per-slot arenas (slots[] table, the cache
 * struct itself) goes through calloc/free since cache lifetime tracks
 * the processor, not an arena.
 *
 * Key types:  sam3_image_feature_cache, sam3_text_feature_cache
 * Depends on: feature_cache.h, util/log.h
 * Used by:    src/model/sam3_processor.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "feature_cache.h"

#include <stdlib.h>
#include <string.h>

#include "util/log.h"

/* --- image cache --- */

struct sam3_image_feature_cache *
sam3_image_cache_create(int n_slots, size_t slot_arena_bytes)
{
	if (n_slots <= 0)
		n_slots = SAM3_IMAGE_CACHE_DEFAULT_SLOTS;
	if (n_slots > SAM3_IMAGE_CACHE_MAX_SLOTS) {
		sam3_log_warn("image_cache: n_slots %d clamped to %d",
			      n_slots, SAM3_IMAGE_CACHE_MAX_SLOTS);
		n_slots = SAM3_IMAGE_CACHE_MAX_SLOTS;
	}

	struct sam3_image_feature_cache *c = calloc(1, sizeof(*c));
	if (!c)
		return NULL;

	c->n_slots = n_slots;
	c->slots = calloc((size_t)n_slots, sizeof(*c->slots));
	if (!c->slots) {
		free(c);
		return NULL;
	}

	for (int i = 0; i < n_slots; i++) {
		if (sam3_arena_init(&c->slots[i].arena, slot_arena_bytes)
		    != SAM3_OK) {
			for (int j = 0; j < i; j++)
				sam3_arena_free(&c->slots[j].arena);
			free(c->slots);
			free(c);
			return NULL;
		}
	}
	return c;
}

void sam3_image_cache_destroy(struct sam3_image_feature_cache *c)
{
	if (!c)
		return;
	for (int i = 0; i < c->n_slots; i++)
		sam3_arena_free(&c->slots[i].arena);
	free(c->slots);
	free(c);
}

int sam3_image_cache_n_slots(const struct sam3_image_feature_cache *c)
{
	return c ? c->n_slots : 0;
}

void sam3_image_cache_clear(struct sam3_image_feature_cache *c)
{
	if (!c)
		return;
	for (int i = 0; i < c->n_slots; i++) {
		sam3_arena_reset(&c->slots[i].arena);
		c->slots[i].hash = 0;
		c->slots[i].lru_tick = 0;
		c->slots[i].prefix_len = 0;
		memset(&c->slots[i].bundle, 0, sizeof(c->slots[i].bundle));
	}
	c->next_tick = 0;
}

void sam3_image_cache_stats(const struct sam3_image_feature_cache *c,
			    struct sam3_cache_stats *out)
{
	if (!out)
		return;
	if (!c) {
		out->image_hits = out->image_misses = out->image_evictions = 0;
		return;
	}
	out->image_hits = c->hits;
	out->image_misses = c->misses;
	out->image_evictions = c->evictions;
}

int sam3_image_cache_lookup(struct sam3_image_feature_cache *c,
			    uint64_t key,
			    const uint8_t *verify_prefix,
			    size_t verify_len)
{
	if (!c || key == 0)
		return -1;
	for (int i = 0; i < c->n_slots; i++) {
		struct sam3_image_cache_slot *s = &c->slots[i];
		if (s->hash != key)
			continue;
		size_t cmp = verify_len < s->prefix_len ? verify_len
						       : s->prefix_len;
		if (verify_prefix && cmp > 0 &&
		    memcmp(s->prefix, verify_prefix, cmp) != 0) {
			/* Hash collision — evict and treat as miss. */
			s->hash = 0;
			c->evictions++;
			c->misses++;
			return -1;
		}
		s->lru_tick = ++c->next_tick;
		c->hits++;
		return i;
	}
	c->misses++;
	return -1;
}

int sam3_image_cache_claim_slot(struct sam3_image_feature_cache *c)
{
	if (!c)
		return -1;
	int oldest = 0;
	uint64_t oldest_tick = c->slots[0].lru_tick;
	int empty = -1;
	for (int i = 0; i < c->n_slots; i++) {
		if (c->slots[i].hash == 0) {
			empty = i;
			break;
		}
		if (c->slots[i].lru_tick < oldest_tick) {
			oldest_tick = c->slots[i].lru_tick;
			oldest = i;
		}
	}
	int idx = (empty >= 0) ? empty : oldest;
	struct sam3_image_cache_slot *s = &c->slots[idx];
	if (s->hash != 0)
		c->evictions++;
	sam3_arena_reset(&s->arena);
	s->hash = 0;
	s->prefix_len = 0;
	memset(&s->bundle, 0, sizeof(s->bundle));
	return idx;
}

void sam3_image_cache_register(struct sam3_image_feature_cache *c, int idx,
			       uint64_t key,
			       const uint8_t *verify_prefix,
			       size_t verify_len,
			       const struct sam3_image_bundle *bundle)
{
	if (!c || idx < 0 || idx >= c->n_slots || key == 0)
		return;
	struct sam3_image_cache_slot *s = &c->slots[idx];
	s->hash = key;
	s->lru_tick = ++c->next_tick;
	s->bundle = *bundle;
	size_t cp = verify_len < SAM3_CACHE_PREFIX_BYTES ? verify_len
						       : SAM3_CACHE_PREFIX_BYTES;
	if (verify_prefix && cp > 0)
		memcpy(s->prefix, verify_prefix, cp);
	s->prefix_len = cp;
}

/* --- text cache --- */

struct sam3_text_feature_cache *
sam3_text_cache_create(int n_slots, size_t slot_bytes)
{
	if (n_slots <= 0)
		n_slots = SAM3_TEXT_CACHE_DEFAULT_SLOTS;
	if (n_slots > SAM3_TEXT_CACHE_MAX_SLOTS) {
		sam3_log_warn("text_cache: n_slots %d clamped to %d",
			      n_slots, SAM3_TEXT_CACHE_MAX_SLOTS);
		n_slots = SAM3_TEXT_CACHE_MAX_SLOTS;
	}

	struct sam3_text_feature_cache *c = calloc(1, sizeof(*c));
	if (!c)
		return NULL;
	c->n_slots = n_slots;
	c->slot_bytes = slot_bytes;
	c->slots = calloc((size_t)n_slots, sizeof(*c->slots));
	if (!c->slots) {
		free(c);
		return NULL;
	}
	if (sam3_arena_init(&c->arena,
			    (size_t)n_slots * slot_bytes) != SAM3_OK) {
		free(c->slots);
		free(c);
		return NULL;
	}
	for (int i = 0; i < n_slots; i++)
		c->slots[i].arena_offset = (size_t)i * slot_bytes;
	return c;
}

void sam3_text_cache_destroy(struct sam3_text_feature_cache *c)
{
	if (!c)
		return;
	sam3_arena_free(&c->arena);
	free(c->slots);
	free(c);
}

int sam3_text_cache_n_slots(const struct sam3_text_feature_cache *c)
{
	return c ? c->n_slots : 0;
}

void sam3_text_cache_clear(struct sam3_text_feature_cache *c)
{
	if (!c)
		return;
	for (int i = 0; i < c->n_slots; i++) {
		c->slots[i].hash = 0;
		c->slots[i].lru_tick = 0;
		c->slots[i].prefix_len = 0;
		c->slots[i].bundle.features = NULL;
		c->slots[i].bundle.n_tokens = 0;
	}
	c->next_tick = 0;
	/* Arena bytes stay allocated; cells are overwritten on next claim. */
}

void sam3_text_cache_stats(const struct sam3_text_feature_cache *c,
			   struct sam3_cache_stats *out)
{
	if (!out)
		return;
	if (!c) {
		out->text_hits = out->text_misses = out->text_evictions = 0;
		return;
	}
	out->text_hits = c->hits;
	out->text_misses = c->misses;
	out->text_evictions = c->evictions;
}

int sam3_text_cache_lookup(struct sam3_text_feature_cache *c,
			   uint64_t key,
			   const int32_t *verify_tokens,
			   int verify_len)
{
	if (!c || key == 0)
		return -1;
	for (int i = 0; i < c->n_slots; i++) {
		struct sam3_text_cache_slot *s = &c->slots[i];
		if (s->hash != key)
			continue;
		int cmp = verify_len < s->prefix_len ? verify_len
						    : s->prefix_len;
		if (verify_tokens && cmp > 0 &&
		    memcmp(s->prefix_tokens, verify_tokens,
			   (size_t)cmp * sizeof(int32_t)) != 0) {
			s->hash = 0;
			c->evictions++;
			c->misses++;
			return -1;
		}
		s->lru_tick = ++c->next_tick;
		c->hits++;
		return i;
	}
	c->misses++;
	return -1;
}

int sam3_text_cache_claim_slot(struct sam3_text_feature_cache *c)
{
	if (!c)
		return -1;
	int oldest = 0;
	uint64_t oldest_tick = c->slots[0].lru_tick;
	int empty = -1;
	for (int i = 0; i < c->n_slots; i++) {
		if (c->slots[i].hash == 0) {
			empty = i;
			break;
		}
		if (c->slots[i].lru_tick < oldest_tick) {
			oldest_tick = c->slots[i].lru_tick;
			oldest = i;
		}
	}
	int idx = (empty >= 0) ? empty : oldest;
	struct sam3_text_cache_slot *s = &c->slots[idx];
	if (s->hash != 0)
		c->evictions++;
	s->hash = 0;
	s->prefix_len = 0;
	s->bundle.features = NULL;
	s->bundle.n_tokens = 0;
	return idx;
}

void sam3_text_cache_register(struct sam3_text_feature_cache *c, int idx,
			      uint64_t key,
			      const int32_t *verify_tokens,
			      int verify_len,
			      const struct sam3_text_bundle *bundle)
{
	if (!c || idx < 0 || idx >= c->n_slots || key == 0)
		return;
	struct sam3_text_cache_slot *s = &c->slots[idx];
	s->hash = key;
	s->lru_tick = ++c->next_tick;
	s->bundle = *bundle;
	int cp = verify_len < (int)(sizeof(s->prefix_tokens) / sizeof(int32_t))
		     ? verify_len
		     : (int)(sizeof(s->prefix_tokens) / sizeof(int32_t));
	if (verify_tokens && cp > 0)
		memcpy(s->prefix_tokens, verify_tokens,
		       (size_t)cp * sizeof(int32_t));
	s->prefix_len = cp;
}

struct sam3_arena *
sam3_text_cache_slot_arena(struct sam3_text_feature_cache *c, int idx)
{
	(void)idx;
	if (!c)
		return NULL;
	return &c->arena; /* slots share; caller writes via arena_offset */
}
```

The `sam3_cache_stats` struct is referenced from `sam3/sam3.h` which doesn't exist yet — for now, add a temporary forward declaration in `feature_cache.h` so the module compiles in isolation:

Replace the `#include "sam3/sam3.h"` at the top of `feature_cache.h` with a forward declaration block:

```c
/* Defined in <sam3/sam3.h> in Task 7; forward-declared here so the
 * cache module compiles before the public API task is wired in. */
struct sam3_cache_stats {
	uint64_t image_hits, image_misses, image_evictions;
	uint64_t text_hits,  text_misses,  text_evictions;
};
```

(In Task 7 we move this struct to `sam3/sam3.h` and remove the forward declaration here.)

- [ ] **Step 5: Add to CMake build**

CMake already globs `src/model/*.c` into the `sam3` library and `tests/test_*.c` into individual test executables (see `CMakeLists.txt:240-245`). No edits needed.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd build && cmake --build . --target test_feature_cache && ctest -R '^test_feature_cache$' --output-on-failure`
Expected: `3 tests, 0 failures`.

- [ ] **Step 7: Commit**

```bash
git add src/model/feature_cache.h src/model/feature_cache.c tests/test_feature_cache.c
git commit -m "model: add feature_cache module with image/text LRU skeletons"
```

---

## Task 3: Image cache lookup, claim, and LRU eviction

**Files:**
- Modify: `tests/test_feature_cache.c`

- [ ] **Step 1: Add tests for lookup/claim/register and LRU**

Append to `tests/test_feature_cache.c` (before `main`):

```c
static struct sam3_tensor *make_dummy_tensor(struct sam3_arena *a, int v)
{
	struct sam3_tensor *t = sam3_arena_alloc(a, sizeof(*t));
	if (!t) return NULL;
	t->dtype = SAM3_DTYPE_F32;
	t->n_dims = 1;
	t->dims[0] = 1;
	t->data = sam3_arena_alloc(a, sizeof(float));
	if (!t->data) return NULL;
	((float *)t->data)[0] = (float)v;
	return t;
}

static void test_image_cache_miss_then_hit(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(2,
								     4 * 1024);
	int idx = sam3_image_cache_lookup(c, 0xdeadbeefULL, NULL, 0);
	ASSERT_EQ(idx, -1);

	int slot = sam3_image_cache_claim_slot(c);
	ASSERT(slot >= 0);
	struct sam3_image_bundle b = {0};
	b.image_features = make_dummy_tensor(&c->slots[slot].arena, 42);
	sam3_image_cache_register(c, slot, 0xdeadbeefULL, NULL, 0, &b);

	int hit = sam3_image_cache_lookup(c, 0xdeadbeefULL, NULL, 0);
	ASSERT_EQ(hit, slot);
	ASSERT_EQ(((float *)c->slots[hit].bundle.image_features->data)[0],
		  42.0f);

	struct sam3_cache_stats s = {0};
	sam3_image_cache_stats(c, &s);
	ASSERT_EQ(s.image_hits, 1u);
	ASSERT_EQ(s.image_misses, 1u);

	sam3_image_cache_destroy(c);
}

static void test_image_cache_lru_eviction(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(2,
								     4 * 1024);

	/* Insert A and B. */
	int sa = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle ba = {0};
	ba.image_features = make_dummy_tensor(&c->slots[sa].arena, 1);
	sam3_image_cache_register(c, sa, 0xAAULL, NULL, 0, &ba);

	int sb = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle bb = {0};
	bb.image_features = make_dummy_tensor(&c->slots[sb].arena, 2);
	sam3_image_cache_register(c, sb, 0xBBULL, NULL, 0, &bb);

	/* Touch A so B becomes the LRU victim. */
	int touch = sam3_image_cache_lookup(c, 0xAAULL, NULL, 0);
	ASSERT_EQ(touch, sa);

	/* Insert C. It should evict B, not A. */
	int sc = sam3_image_cache_claim_slot(c);
	ASSERT_EQ(sc, sb);
	struct sam3_image_bundle bc = {0};
	bc.image_features = make_dummy_tensor(&c->slots[sc].arena, 3);
	sam3_image_cache_register(c, sc, 0xCCULL, NULL, 0, &bc);

	ASSERT_EQ(sam3_image_cache_lookup(c, 0xAAULL, NULL, 0), sa);
	ASSERT_EQ(sam3_image_cache_lookup(c, 0xBBULL, NULL, 0), -1);
	ASSERT_EQ(sam3_image_cache_lookup(c, 0xCCULL, NULL, 0), sc);

	struct sam3_cache_stats s = {0};
	sam3_image_cache_stats(c, &s);
	ASSERT_EQ(s.image_evictions, 1u);

	sam3_image_cache_destroy(c);
}

static void test_image_cache_collision_verifies_prefix(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(1,
								     4 * 1024);
	int slot = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle b = {0};
	b.image_features = make_dummy_tensor(&c->slots[slot].arena, 7);
	uint8_t pref_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	sam3_image_cache_register(c, slot, 0x1234ULL, pref_a, 8, &b);

	/* Same hash, different prefix: must NOT hit. */
	uint8_t pref_b[8] = {9, 9, 9, 9, 9, 9, 9, 9};
	int collide = sam3_image_cache_lookup(c, 0x1234ULL, pref_b, 8);
	ASSERT_EQ(collide, -1);

	sam3_image_cache_destroy(c);
}

static void test_image_cache_clear(void)
{
	struct sam3_image_feature_cache *c = sam3_image_cache_create(2,
								     4 * 1024);
	int slot = sam3_image_cache_claim_slot(c);
	struct sam3_image_bundle b = {0};
	b.image_features = make_dummy_tensor(&c->slots[slot].arena, 1);
	sam3_image_cache_register(c, slot, 0xFEULL, NULL, 0, &b);

	sam3_image_cache_clear(c);
	ASSERT_EQ(sam3_image_cache_lookup(c, 0xFEULL, NULL, 0), -1);

	sam3_image_cache_destroy(c);
}
```

Add the new tests to `main`:

```c
int main(void)
{
	test_image_cache_init_default_slots();
	test_image_cache_clamps_slot_count();
	test_text_cache_init_default_slots();
	test_image_cache_miss_then_hit();
	test_image_cache_lru_eviction();
	test_image_cache_collision_verifies_prefix();
	test_image_cache_clear();
	TEST_REPORT();
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd build && cmake --build . --target test_feature_cache && ctest -R '^test_feature_cache$' --output-on-failure`
Expected: `7 tests, 0 failures`. The implementation from Task 2 already covers these — the new tests confirm it works end-to-end.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_cache.c
git commit -m "test: cover image cache lookup, LRU eviction, collision check"
```

---

## Task 4: Text cache lookup, claim, and LRU eviction

**Files:**
- Modify: `tests/test_feature_cache.c`

- [ ] **Step 1: Add text cache tests**

Append before `main`:

```c
static void test_text_cache_miss_then_hit(void)
{
	struct sam3_text_feature_cache *c = sam3_text_cache_create(4,
							       64 * 1024);
	int idx = sam3_text_cache_lookup(c, 0xCAFEULL, NULL, 0);
	ASSERT_EQ(idx, -1);

	int slot = sam3_text_cache_claim_slot(c);
	ASSERT(slot >= 0);
	struct sam3_text_bundle b = {.features = NULL, .n_tokens = 5};
	int32_t toks[3] = {10, 20, 30};
	sam3_text_cache_register(c, slot, 0xCAFEULL, toks, 3, &b);

	int hit = sam3_text_cache_lookup(c, 0xCAFEULL, toks, 3);
	ASSERT_EQ(hit, slot);
	ASSERT_EQ(c->slots[hit].bundle.n_tokens, 5);

	struct sam3_cache_stats s = {0};
	sam3_text_cache_stats(c, &s);
	ASSERT_EQ(s.text_hits, 1u);
	ASSERT_EQ(s.text_misses, 1u);

	sam3_text_cache_destroy(c);
}

static void test_text_cache_lru_eviction(void)
{
	struct sam3_text_feature_cache *c = sam3_text_cache_create(2,
							       64 * 1024);
	int sa = sam3_text_cache_claim_slot(c);
	struct sam3_text_bundle ba = {.features = NULL, .n_tokens = 1};
	sam3_text_cache_register(c, sa, 0xA1ULL, NULL, 0, &ba);

	int sb = sam3_text_cache_claim_slot(c);
	struct sam3_text_bundle bb = {.features = NULL, .n_tokens = 2};
	sam3_text_cache_register(c, sb, 0xB2ULL, NULL, 0, &bb);

	ASSERT_EQ(sam3_text_cache_lookup(c, 0xA1ULL, NULL, 0), sa);

	int sc = sam3_text_cache_claim_slot(c);
	ASSERT_EQ(sc, sb);
	struct sam3_text_bundle bc = {.features = NULL, .n_tokens = 3};
	sam3_text_cache_register(c, sc, 0xC3ULL, NULL, 0, &bc);

	ASSERT_EQ(sam3_text_cache_lookup(c, 0xA1ULL, NULL, 0), sa);
	ASSERT_EQ(sam3_text_cache_lookup(c, 0xB2ULL, NULL, 0), -1);
	ASSERT_EQ(sam3_text_cache_lookup(c, 0xC3ULL, NULL, 0), sc);

	sam3_text_cache_destroy(c);
}

static void test_text_cache_clear(void)
{
	struct sam3_text_feature_cache *c = sam3_text_cache_create(2,
							       64 * 1024);
	int slot = sam3_text_cache_claim_slot(c);
	struct sam3_text_bundle b = {.features = NULL, .n_tokens = 1};
	sam3_text_cache_register(c, slot, 0xEEULL, NULL, 0, &b);

	sam3_text_cache_clear(c);
	ASSERT_EQ(sam3_text_cache_lookup(c, 0xEEULL, NULL, 0), -1);

	sam3_text_cache_destroy(c);
}
```

Add these to `main` after the image tests:

```c
	test_text_cache_miss_then_hit();
	test_text_cache_lru_eviction();
	test_text_cache_clear();
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd build && cmake --build . --target test_feature_cache && ctest -R '^test_feature_cache$' --output-on-failure`
Expected: `10 tests, 0 failures`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_cache.c
git commit -m "test: cover text cache lookup, LRU eviction, clear"
```

---

## Task 5: Wire image cache into sam3_processor (replace weights_end)

**Files:**
- Modify: `src/model/sam3_processor.h`
- Modify: `src/model/sam3_processor.c`
- Modify: `src/model/sam3_internal.h` (add cache opts plumbing)
- Test: `tests/test_processor_cache.c` (created in this task)

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_processor_cache.c`:

```c
/*
 * tests/test_processor_cache.c - End-to-end image/text cache tests
 *
 * Loads a real model and verifies that repeated set_image / set_text
 * calls hit the cache. Skips if model weights are not present.
 *
 * Key types:  sam3_processor
 * Depends on: test_helpers.h, sam3/sam3.h, model/sam3_processor.h,
 *             model/feature_cache.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "sam3/sam3.h"
#include "core/weight.h"
#include "model/sam3_processor.h"
#include "model/feature_cache.h"

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif

#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"
#define VOCAB_PATH SAM3_SOURCE_DIR "/models/bpe_simple_vocab_16e6.txt.gz"

static int model_available(void)
{
	return access(MODEL_PATH, F_OK) == 0;
}

static void test_image_cache_hit_skips_encoder(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err;
	err = sam3_processor_init(&proc, SAM3_BACKBONE_HIERA, 4);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	int sz = sam3_processor_img_size(&proc);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	for (int i = 0; i < sz * sz * 3; i++)
		pixels[i] = (uint8_t)(i & 0xff);

	err = sam3_processor_set_image(&proc, pixels, sz, sz);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_cache_stats st0 = {0};
	sam3_image_cache_stats(proc.img_cache, &st0);
	ASSERT_EQ(st0.image_hits, 0u);
	ASSERT_EQ(st0.image_misses, 1u);

	err = sam3_processor_set_image(&proc, pixels, sz, sz);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_cache_stats st1 = {0};
	sam3_image_cache_stats(proc.img_cache, &st1);
	ASSERT_EQ(st1.image_hits, 1u);
	ASSERT_EQ(st1.image_misses, 1u);

	free(pixels);
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}

int main(void)
{
	test_image_cache_hit_skips_encoder();
	TEST_REPORT();
}
```

Update CMakeLists.txt: at the end of the per-test conditionals around line 287, add:

```cmake
	# test_processor_cache opens models/sam3.sam3
	if(TARGET test_processor_cache)
		target_compile_definitions(test_processor_cache PRIVATE
			SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
	endif()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && cmake --build . --target test_processor_cache 2>&1 | tail -10`
Expected: compile error — `proc.img_cache` is not a field of `struct sam3_processor`.

- [ ] **Step 3: Modify `src/model/sam3_processor.h`**

Add the include and replace fields:

At the top with other includes, add:
```c
#include "feature_cache.h"
```

Inside `struct sam3_processor`, replace:
```c
	size_t weights_end;              /* model_arena offset after load */
	int image_loaded;
```
with:
```c
	struct sam3_image_feature_cache *img_cache;
	int current_img_slot;            /* -1 == none */
	int image_loaded;                /* 1 iff cached_* pointers are live */
```

Add a new init signature alongside the existing one:
```c
/*
 * sam3_processor_init_ex - Like sam3_processor_init, but with caller-
 * supplied cache slot counts. Pass 0 for either to use the defaults
 * (3 image, 16 text). Pass 1 to disable multi-slot behavior.
 */
enum sam3_error sam3_processor_init_ex(struct sam3_processor *proc,
				       int backbone_type,
				       int n_fpn_scales,
				       int n_image_slots,
				       int n_text_slots);
```

Add a clear/stats pair:
```c
void sam3_processor_cache_clear(struct sam3_processor *proc, unsigned which);
void sam3_processor_cache_stats(const struct sam3_processor *proc,
				struct sam3_cache_stats *out);
```

(`SAM3_CACHE_IMAGE` / `SAM3_CACHE_TEXT` constants are added in Task 7 but used here as raw bits 1<<0 / 1<<1 — feature_cache.h does not define them.)

- [ ] **Step 4: Modify `src/model/sam3_processor.c` — init/free**

In `sam3_processor_init`, before the `return SAM3_OK;` at end, allocate the image cache. Convert the existing function to call a new `_ex` variant:

```c
enum sam3_error sam3_processor_init(struct sam3_processor *proc,
				    int backbone_type, int n_fpn_scales)
{
	return sam3_processor_init_ex(proc, backbone_type, n_fpn_scales,
				      0, 0);
}

enum sam3_error sam3_processor_init_ex(struct sam3_processor *proc,
				       int backbone_type, int n_fpn_scales,
				       int n_image_slots, int n_text_slots)
{
	/* ... existing init body, with the additions below ... */
}
```

Inside the new `_ex` body (replacing the old `init`), after `sam3_image_model_init` succeeds and before the `return SAM3_OK`:

```c
	/*
	 * Allocate the image feature cache. Slot arena size is sized for
	 * encoder peak output (matches the historical post-weights room
	 * inside model_arena: a few hundred MiB worst-case across all
	 * cached_* tensors). Use 384 MiB per slot — 3 slots default = 1.1
	 * GiB total, well inside the 2 GiB image-side budget that
	 * model_arena previously consumed.
	 */
	proc->img_cache = sam3_image_cache_create(n_image_slots,
						  384UL * 1024 * 1024);
	if (!proc->img_cache) {
		err = SAM3_ENOMEM;
		goto cleanup_text_backend;
	}
	proc->current_img_slot = -1;
	(void)n_text_slots; /* wired in Task 6 */
```

Add a new `cleanup_img_cache` label (the existing `cleanup_text_backend` chain stays the same; the new label sits between `return SAM3_OK` and `cleanup_text_backend`):

```c
	return SAM3_OK;

cleanup_img_cache:
	sam3_image_cache_destroy(proc->img_cache);
	proc->img_cache = NULL;
cleanup_text_backend:
	/* ... unchanged ... */
```

In `sam3_processor_free`, before `sam3_arena_free(&proc->model_arena);` add:
```c
	sam3_image_cache_destroy(proc->img_cache);
	proc->img_cache = NULL;
```

- [ ] **Step 5: Modify `sam3_processor_load` — drop `weights_end` capture**

Replace the body:

```c
enum sam3_error sam3_processor_load(struct sam3_processor *proc,
				    const struct sam3_weight_file *wf,
				    const char *vocab_path)
{
	return sam3_image_model_load(&proc->model, wf, vocab_path,
				     &proc->model_arena);
}
```

(`weights_end` is gone; encoder outputs no longer share `model_arena`.)

- [ ] **Step 6: Modify `sam3_processor_set_image`**

Replace the function body:

```c
enum sam3_error sam3_processor_set_image(struct sam3_processor *proc,
					 const uint8_t *pixels,
					 int width, int height)
{
	struct sam3_tensor *image;
	enum sam3_error err;
	int dims[3];
	float *dst;

	if (!proc || !pixels || width <= 0 || height <= 0)
		return SAM3_EINVAL;

	/* Hash the input bytes + dimensions to derive the cache key. */
	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)&width, sizeof(width), key);
	key = sam3_fnv1a_64((const uint8_t *)&height, sizeof(height), key);
	size_t n_bytes = (size_t)width * (size_t)height * 3;
	key = sam3_fnv1a_64(pixels, n_bytes, key);
	if (key == 0) key = 1; /* 0 reserved for "empty" */

	size_t pref_len = n_bytes < SAM3_CACHE_PREFIX_BYTES
			      ? n_bytes : SAM3_CACHE_PREFIX_BYTES;

	int hit = sam3_image_cache_lookup(proc->img_cache, key, pixels,
					  pref_len);
	if (hit >= 0) {
		struct sam3_image_bundle *b =
			&proc->img_cache->slots[hit].bundle;
		proc->model.cached_image_features = b->image_features;
		proc->model.cached_feat_s0_nhwc   = b->feat_s0_nhwc;
		proc->model.cached_feat_s1_nhwc   = b->feat_s1_nhwc;
		proc->model.cached_feat_4x_nhwc   = b->feat_4x_nhwc;
		proc->model.cached_sam2_05x_nhwc  = b->sam2_05x_nhwc;
		proc->model.cached_sam2_1x_nhwc   = b->sam2_1x_nhwc;
		proc->model.cached_sam2_2x_nhwc   = b->sam2_2x_nhwc;
		proc->model.cached_sam2_4x_nhwc   = b->sam2_4x_nhwc;
		proc->model.image_encoded = 1;
		proc->image_loaded = 1;
		proc->current_img_slot = hit;
		proc->prompt_w = b->prompt_w;
		proc->prompt_h = b->prompt_h;
		return SAM3_OK;
	}

	/* Miss: claim a slot and encode into its arena. */
	int slot = sam3_image_cache_claim_slot(proc->img_cache);
	if (slot < 0)
		return SAM3_ENOMEM;
	proc->current_img_slot = slot;
	struct sam3_arena *persist = &proc->img_cache->slots[slot].arena;

	sam3_arena_reset(&proc->scratch_arena);

	dims[0] = 3; dims[1] = height; dims[2] = width;
	image = gh_alloc_tensor(&proc->scratch_arena, SAM3_DTYPE_F32, 3, dims);
	if (!image)
		return SAM3_ENOMEM;
	dst = (float *)image->data;
	SAM3_PROF_BEGIN(proc->profiler, "image_normalize");
	sam3_normalize_rgb_chw(pixels, dst, width, height);
	SAM3_PROF_END(proc->profiler, "image_normalize");

	SAM3_PROF_BEGIN(proc->profiler, "image_encode");
	err = sam3_image_model_encode(&proc->model, proc->backend, image,
				      &proc->scratch_arena, persist,
				      proc->profiler);
	SAM3_PROF_END(proc->profiler, "image_encode");
	if (err != SAM3_OK)
		return err;

	proc->image_loaded = 1;
	proc->prompt_w = width;
	proc->prompt_h = height;

	struct sam3_image_bundle b = {0};
	b.image_features = proc->model.cached_image_features;
	b.feat_s0_nhwc   = proc->model.cached_feat_s0_nhwc;
	b.feat_s1_nhwc   = proc->model.cached_feat_s1_nhwc;
	b.feat_4x_nhwc   = proc->model.cached_feat_4x_nhwc;
	b.sam2_05x_nhwc  = proc->model.cached_sam2_05x_nhwc;
	b.sam2_1x_nhwc   = proc->model.cached_sam2_1x_nhwc;
	b.sam2_2x_nhwc   = proc->model.cached_sam2_2x_nhwc;
	b.sam2_4x_nhwc   = proc->model.cached_sam2_4x_nhwc;
	b.prompt_w = width;
	b.prompt_h = height;
	b.width = width;
	b.height = height;
	sam3_image_cache_register(proc->img_cache, slot, key,
				  pixels, pref_len, &b);
	return SAM3_OK;
}
```

Add at the top of the file with other includes:
```c
#include "util/hash.h"
#include "feature_cache.h"
```

- [ ] **Step 7: Add cache_clear / cache_stats helpers**

Append near the bottom of `sam3_processor.c`:

```c
void sam3_processor_cache_clear(struct sam3_processor *proc, unsigned which)
{
	if (!proc)
		return;
	if (which == 0 || (which & 1u))
		sam3_image_cache_clear(proc->img_cache);
	/* text cache cleared in Task 6 once it exists */
}

void sam3_processor_cache_stats(const struct sam3_processor *proc,
				struct sam3_cache_stats *out)
{
	if (!out)
		return;
	memset(out, 0, sizeof(*out));
	if (!proc)
		return;
	sam3_image_cache_stats(proc->img_cache, out);
}
```

- [ ] **Step 8: Update `tests/test_processor_async.c`**

The existing `test_set_text_returns_immediately` reads `proc.text_thread_active` and checks for `text_features_async` plumbing. Those fields still exist after this task (Task 6 removes them); leave that test file alone for now.

- [ ] **Step 9: Build everything and run the new + existing tests**

Run:
```
cd build && cmake --build . -- -j && \
  ctest -R '^test_(hash|feature_cache|processor_cache|processor_async|sam3_)' \
        --output-on-failure
```
Expected: all pass. The `image_loaded` semantics are preserved (still 1 after a successful set_image), so existing segment tests work unchanged.

- [ ] **Step 10: Commit**

```bash
git add src/model/sam3_processor.h src/model/sam3_processor.c \
        tests/test_processor_cache.c CMakeLists.txt
git commit -m "processor: route set_image through image feature cache"
```

---

## Task 6: Wire text cache into sam3_processor (replace text_features_async)

**Files:**
- Modify: `src/model/sam3_processor.h`
- Modify: `src/model/sam3_processor.c`
- Modify: `tests/test_processor_async.c`
- Modify: `tests/test_processor_cache.c`

- [ ] **Step 1: Add the text-cache integration test**

Append to `tests/test_processor_cache.c` before `main`:

```c
static void test_text_cache_hit_skips_worker(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err;
	err = sam3_processor_init(&proc, SAM3_BACKBONE_HIERA, 4);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_weight_file wf;
	err = sam3_weight_open(&wf, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);
	err = sam3_processor_load(&proc, &wf, VOCAB_PATH);
	ASSERT_EQ(err, SAM3_OK);

	/* First call → miss + worker spawned. */
	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);

	/* Drive the segment path so the worker output gets registered. */
	int sz = sam3_processor_img_size(&proc);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	err = sam3_processor_set_image(&proc, pixels, sz, sz);
	ASSERT_EQ(err, SAM3_OK);

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT,
				.text = {.text = "cat"}};
	struct sam3_result r = {0};
	err = sam3_processor_segment(&proc, &p, 1, &r);
	ASSERT_EQ(err, SAM3_OK);
	sam3_result_free(&r);

	struct sam3_cache_stats s0 = {0};
	sam3_text_cache_stats(proc.txt_cache, &s0);
	ASSERT_EQ(s0.text_misses, 1u);
	ASSERT_EQ(s0.text_hits, 0u);

	/* Second set_text("cat") → hit, no worker. */
	err = sam3_processor_set_text(&proc, "cat");
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(proc.text_thread_active, 0);

	struct sam3_cache_stats s1 = {0};
	sam3_text_cache_stats(proc.txt_cache, &s1);
	ASSERT_EQ(s1.text_hits, 1u);

	free(pixels);
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}
```

Add to `main`:
```c
	test_text_cache_hit_skips_worker();
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake --build . --target test_processor_cache 2>&1 | tail -5`
Expected: compile error — `proc.txt_cache` does not exist.

- [ ] **Step 3: Modify `src/model/sam3_processor.h`**

Inside `struct sam3_processor`, replace the async text block:

```c
	struct sam3_backend *text_backend;
	struct sam3_arena    text_scratch_arena;
	struct sam3_arena    text_persist_arena;
	pthread_t            text_thread;
	int                  text_thread_active;
	enum sam3_error      text_thread_err;
	struct sam3_tensor  *text_features_async;
	int32_t              text_tokens[SAM3_PROCESSOR_MAX_TOKENS];
	int                  text_n_tokens;
```

with:

```c
	struct sam3_backend *text_backend;
	struct sam3_arena    text_scratch_arena;
	struct sam3_text_feature_cache *txt_cache;
	struct sam3_text_bundle *text_cached_bundle; /* hit path; cleared by segment */
	int                      text_worker_slot;   /* -1 or txt_cache slot */
	pthread_t            text_thread;
	int                  text_thread_active;
	enum sam3_error      text_thread_err;
	int32_t              text_tokens[SAM3_PROCESSOR_MAX_TOKENS];
	int                  text_n_tokens;
```

- [ ] **Step 4: Modify `src/model/sam3_processor.c` — init/free**

In `sam3_processor_init_ex`, replace the `text_persist_arena` init block with text cache creation:

Remove:
```c
	err = sam3_arena_init(&proc->text_persist_arena,
			      16UL * 1024 * 1024);
	if (err != SAM3_OK)
		goto cleanup_text_scratch_arena;
```

Add (before the image cache block from Task 5):
```c
	/*
	 * Text feature cache. Each slot holds an [n_tokens, d_model] f32
	 * tensor; max sized for 77 tokens × 1024 d_model = 308 KiB. Add
	 * tensor-header headroom and bundle a 1 MiB cell to keep the math
	 * round-numbered. n_text_slots default 16 → 16 MiB total.
	 */
	proc->txt_cache = sam3_text_cache_create(n_text_slots,
						 1UL * 1024 * 1024);
	if (!proc->txt_cache) {
		err = SAM3_ENOMEM;
		goto cleanup_text_scratch_arena;
	}
	proc->text_cached_bundle = NULL;
	proc->text_worker_slot = -1;
```

Update the cleanup chain — replace `cleanup_text_backend:` block ordering so `txt_cache` is destroyed in the right place. Adjust labels:

```c
cleanup_img_cache:
	sam3_image_cache_destroy(proc->img_cache);
	proc->img_cache = NULL;
cleanup_text_backend:
	if (proc->text_backend) {
		proc->text_backend->ops->free(proc->text_backend);
		free(proc->text_backend);
		proc->text_backend = NULL;
	}
	sam3_text_cache_destroy(proc->txt_cache);
	proc->txt_cache = NULL;
cleanup_text_scratch_arena:
	sam3_arena_free(&proc->text_scratch_arena);
cleanup_scratch_arena:
	sam3_arena_free(&proc->scratch_arena);
cleanup_model_arena:
	sam3_arena_free(&proc->model_arena);
cleanup_backend:
	proc->backend->ops->free(proc->backend);
	free(proc->backend);
	proc->backend = NULL;
	return err;
```

In `sam3_processor_free`, replace:
```c
	sam3_arena_free(&proc->text_persist_arena);
```
with:
```c
	sam3_text_cache_destroy(proc->txt_cache);
	proc->txt_cache = NULL;
```

- [ ] **Step 5: Rewrite `text_worker_main`**

Replace the existing function body:

```c
static void *text_worker_main(void *arg)
{
	struct sam3_processor *proc = arg;
	struct sam3_text_encoder_iface *te_iface =
		&proc->model.backbone.text_iface;
	int ctx = te_iface->ctx_len;
	int slot = proc->text_worker_slot;
	struct sam3_arena *arena =
		sam3_text_cache_slot_arena(proc->txt_cache, slot);

	int tok_dims[1] = {ctx};
	struct sam3_tensor *tok = gh_alloc_tensor(arena, SAM3_DTYPE_I32,
						  1, tok_dims);
	if (!tok) {
		proc->text_thread_err = SAM3_ENOMEM;
		return NULL;
	}
	memcpy(tok->data, proc->text_tokens, (size_t)ctx * sizeof(int32_t));

	struct sam3_tensor *features = te_iface->ops->build_perblock(
		te_iface, proc->text_backend, tok,
		&proc->text_scratch_arena, arena);
	if (!features) {
		proc->text_thread_err = SAM3_ENOMEM;
		return NULL;
	}

	if (features->dims[0] > proc->text_n_tokens) {
		int d = features->dims[1];
		int trunc_dims[2] = {proc->text_n_tokens, d};
		struct sam3_tensor *trunc = gh_alloc_tensor(arena,
				SAM3_DTYPE_F32, 2, trunc_dims);
		if (!trunc) {
			proc->text_thread_err = SAM3_ENOMEM;
			return NULL;
		}
		memcpy(trunc->data, features->data,
		       (size_t)proc->text_n_tokens * (size_t)d * sizeof(float));
		features = trunc;
	}

	struct sam3_text_bundle b = {.features = features,
				     .n_tokens = proc->text_n_tokens};
	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)proc->text_tokens,
			    (size_t)proc->text_n_tokens * sizeof(int32_t),
			    key);
	if (key == 0) key = 1;

	sam3_text_cache_register(proc->txt_cache, slot, key,
				 proc->text_tokens, proc->text_n_tokens, &b);
	proc->text_thread_err = SAM3_OK;
	return NULL;
}
```

- [ ] **Step 6: Rewrite `sam3_processor_set_text`**

Replace the function body:

```c
enum sam3_error sam3_processor_set_text(struct sam3_processor *proc,
					const char *text)
{
	struct sam3_text_encoder_iface *te_iface;
	int ctx, n_tokens, rc;

	if (!proc || !text)
		return SAM3_EINVAL;
	if (!proc->text_backend) {
		sam3_log_error("set_text: text_backend unavailable");
		return SAM3_EBACKEND;
	}

	join_text_worker(proc);
	sam3_arena_reset(&proc->text_scratch_arena);
	proc->text_cached_bundle = NULL;
	proc->text_worker_slot = -1;
	proc->text_thread_err = SAM3_OK;

	te_iface = &proc->model.backbone.text_iface;
	ctx = te_iface->ctx_len;

	n_tokens = sam3_tokenizer_encode(&proc->model.backbone.tokenizer,
					 text, proc->text_tokens, ctx);
	if (n_tokens <= 0) {
		sam3_log_error("set_text: tokenize failed");
		return SAM3_EINVAL;
	}
	proc->text_n_tokens = n_tokens;

	uint64_t key = SAM3_FNV1A_64_OFFSET_BASIS;
	key = sam3_fnv1a_64((const uint8_t *)proc->text_tokens,
			    (size_t)n_tokens * sizeof(int32_t), key);
	if (key == 0) key = 1;

	int hit = sam3_text_cache_lookup(proc->txt_cache, key,
					 proc->text_tokens, n_tokens);
	if (hit >= 0) {
		proc->text_cached_bundle =
			&proc->txt_cache->slots[hit].bundle;
		return SAM3_OK;
	}

	int slot = sam3_text_cache_claim_slot(proc->txt_cache);
	if (slot < 0)
		return SAM3_ENOMEM;
	proc->text_worker_slot = slot;

	pthread_attr_t attr;
	if (pthread_attr_init(&attr) != 0)
		return SAM3_EBACKEND;
	if (pthread_attr_setstacksize(&attr, 8UL * 1024 * 1024) != 0) {
		pthread_attr_destroy(&attr);
		return SAM3_EBACKEND;
	}
	rc = pthread_create(&proc->text_thread, &attr, text_worker_main, proc);
	pthread_attr_destroy(&attr);
	if (rc != 0) {
		sam3_log_error("set_text: pthread_create failed (%d)", rc);
		return SAM3_EBACKEND;
	}
	proc->text_thread_active = 1;
	return SAM3_OK;
}
```

- [ ] **Step 7: Update text consumption in `sam3_processor_segment`**

In `sam3_processor_segment`, the current code (around lines 880-930) joins the worker, then has `if (proc->text_features_async) { ... copy into model_arena ... } else { /* inline encode */ ... }`. Replace **only** the leading `join_text_worker + if (proc->text_features_async) { ... } else {` part — i.e. lines 880 through the line `} else {` on line 932 — with the block below. **Do not touch anything inside the existing `else { ... }` body** (the inline-encode legacy path, lines 932-end-of-`else`); leave it intact and matched by the new `else` of the new structure.

Replacement (replaces lines 880-932 inclusive of the opening `if (text) {` and through `} else {`):

```c
	if (text) {
		SAM3_PROF_BEGIN(proc->profiler, "text_encode");

		if (proc->text_cached_bundle) {
			text_features = proc->text_cached_bundle->features;
			proc->text_cached_bundle = NULL;
			SAM3_PROF_END(proc->profiler, "text_encode");
		} else if (proc->text_thread_active) {
			join_text_worker(proc);
			if (proc->text_thread_err != SAM3_OK) {
				sam3_log_error("segment: text worker "
					       "failed: %d",
					       proc->text_thread_err);
				err = proc->text_thread_err;
				proc->text_thread_err = SAM3_OK;
				goto fail;
			}
			int s = proc->text_worker_slot;
			if (s < 0) {
				err = SAM3_EBACKEND;
				goto fail;
			}
			text_features =
				proc->txt_cache->slots[s].bundle.features;
			proc->text_worker_slot = -1;
			SAM3_PROF_END(proc->profiler, "text_encode");
		} else {
```

After this replacement, the file's flow is: new `if/else if/else` chain; the **existing inline-encode body** (originally lines 932-end-of-else) becomes the body of the new `else` and runs unchanged.

Verify after the edit by running `grep -n "text_features_async\|text_persist_arena" src/model/sam3_processor.c`. Expected: zero matches (all references removed).

- [ ] **Step 8: Update `sam3_processor_cache_clear` to flush text cache**

Replace the body added in Task 5:

```c
void sam3_processor_cache_clear(struct sam3_processor *proc, unsigned which)
{
	if (!proc)
		return;
	if (which == 0 || (which & 1u))
		sam3_image_cache_clear(proc->img_cache);
	if (which == 0 || (which & 2u)) {
		join_text_worker(proc);
		sam3_text_cache_clear(proc->txt_cache);
		proc->text_cached_bundle = NULL;
		proc->text_worker_slot = -1;
	}
}
```

And update `sam3_processor_cache_stats` to merge text stats:

```c
void sam3_processor_cache_stats(const struct sam3_processor *proc,
				struct sam3_cache_stats *out)
{
	if (!out)
		return;
	memset(out, 0, sizeof(*out));
	if (!proc)
		return;
	struct sam3_cache_stats img = {0}, txt = {0};
	sam3_image_cache_stats(proc->img_cache, &img);
	sam3_text_cache_stats(proc->txt_cache, &txt);
	out->image_hits = img.image_hits;
	out->image_misses = img.image_misses;
	out->image_evictions = img.image_evictions;
	out->text_hits = txt.text_hits;
	out->text_misses = txt.text_misses;
	out->text_evictions = txt.text_evictions;
}
```

- [ ] **Step 9: Update `tests/test_processor_async.c`**

The existing test `test_set_text_returns_immediately` checks `proc.text_thread_active` and references `proc.text_features_async` in a comment. The field still exists; the comment should now say "the worker writes into the txt_cache slot". Update the comment block (lines 66-74) and remove any reference to `text_features_async`. The actual asserts (`text_thread_active`, `text_n_tokens > 0`) remain valid. Replace lines 66-74 with:

```c
	/*
	 * processor_free() joins the worker. The worker writes its
	 * output into proc.txt_cache via the slot it pre-claimed; we
	 * just verify the join happens without crashing or leaking.
	 * processor_free MUST run before weight_close: while the worker
	 * is in flight it is reading mmap'd weight tensors, and
	 * unmapping them mid-flight would SEGV the worker.
	 */
```

If any other test in this file references `text_features_async` or `text_persist_arena`, remove those references — the worker now publishes through the cache.

- [ ] **Step 10: Run all affected tests**

Run:
```
cd build && cmake --build . -- -j && \
  ctest -R '^test_(hash|feature_cache|processor_cache|processor_async|sam3_)' \
        --output-on-failure
```
Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add src/model/sam3_processor.h src/model/sam3_processor.c \
        tests/test_processor_cache.c tests/test_processor_async.c
git commit -m "processor: route set_text through text feature cache"
```

---

## Task 7: Public API — `sam3_init_ex`, `sam3_cache_clear`, `sam3_cache_stats`

**Files:**
- Modify: `include/sam3/sam3.h`
- Modify: `src/model/sam3_internal.h`
- Modify: `src/sam3.c`
- Modify: `src/model/feature_cache.h` (remove forward declaration)

- [ ] **Step 1: Add API to `include/sam3/sam3.h`**

After the existing `sam3_init` declaration:

```c
struct sam3_cache_opts {
	int n_image_slots;  /* 0 -> default 3 */
	int n_text_slots;   /* 0 -> default 16 */
};

/*
 * sam3_init_ex - Like sam3_init, but with caller-supplied cache slot
 * counts. Pass NULL for defaults (equivalent to sam3_init).
 */
sam3_ctx *sam3_init_ex(const struct sam3_cache_opts *opts);

enum {
	SAM3_CACHE_IMAGE = 1 << 0,
	SAM3_CACHE_TEXT  = 1 << 1,
};

/*
 * sam3_cache_clear - Flush image and/or text encoder caches.
 *
 * @ctx:   Initialized context.
 * @which: Bitmask of SAM3_CACHE_IMAGE | SAM3_CACHE_TEXT (0 = both).
 */
void sam3_cache_clear(sam3_ctx *ctx, unsigned which);

struct sam3_cache_stats {
	uint64_t image_hits, image_misses, image_evictions;
	uint64_t text_hits,  text_misses,  text_evictions;
};

/*
 * sam3_cache_stats - Read hit/miss/eviction counters for both caches.
 */
void sam3_cache_stats(const sam3_ctx *ctx, struct sam3_cache_stats *out);
```

Add `#include <stdint.h>` to the header if not already pulled in via `sam3_types.h`.

- [ ] **Step 2: Remove forward declaration from `feature_cache.h`**

In `src/model/feature_cache.h`, remove the placeholder block:

```c
struct sam3_cache_stats {
	uint64_t image_hits, image_misses, image_evictions;
	uint64_t text_hits,  text_misses,  text_evictions;
};
```

and replace it (back) with:
```c
#include "sam3/sam3.h"
```
(near the top with the other includes).

- [ ] **Step 3: Add `cache_opts` to `sam3_ctx` and plumb init**

In `src/model/sam3_internal.h`, add to `struct sam3_ctx`:
```c
	struct sam3_cache_opts cache_opts;
```

In `src/sam3.c`, replace `sam3_init`:

```c
sam3_ctx *sam3_init(void)
{
	return sam3_init_ex(NULL);
}

sam3_ctx *sam3_init_ex(const struct sam3_cache_opts *opts)
{
	sam3_ctx *ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;
	if (opts)
		ctx->cache_opts = *opts;
	return ctx;
}
```

In `sam3_load_model`, replace the call to `sam3_processor_init` with:
```c
	err = sam3_processor_init_ex(&ctx->proc,
				     ctx->config.backbone_type,
				     ctx->config.n_fpn_scales,
				     ctx->cache_opts.n_image_slots,
				     ctx->cache_opts.n_text_slots);
```

At the start of `sam3_load_model`, just after the existing `if (ctx->loaded) { sam3_weight_close...; ctx->loaded = 0; }` block, also flush caches if a processor already exists:
```c
	if (ctx->proc_ready) {
		sam3_processor_cache_clear(&ctx->proc, 0);
	}
```

- [ ] **Step 4: Implement the new top-level functions**

Append to `src/sam3.c`:

```c
void sam3_cache_clear(sam3_ctx *ctx, unsigned which)
{
	if (!ctx || !ctx->proc_ready)
		return;
	sam3_processor_cache_clear(&ctx->proc, which);
}

void sam3_cache_stats(const sam3_ctx *ctx, struct sam3_cache_stats *out)
{
	if (!out)
		return;
	if (!ctx || !ctx->proc_ready) {
		memset(out, 0, sizeof(*out));
		return;
	}
	sam3_processor_cache_stats(&ctx->proc, out);
}
```

- [ ] **Step 5: Add a top-level integration test**

Append to `tests/test_processor_cache.c` before `main`:

```c
static void test_top_level_cache_clear_and_stats(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}
	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	enum sam3_error err = sam3_load_model(ctx, MODEL_PATH);
	ASSERT_EQ(err, SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);

	struct sam3_cache_stats st = {0};
	sam3_cache_stats(ctx, &st);
	ASSERT_EQ(st.image_hits, 1u);

	sam3_cache_clear(ctx, SAM3_CACHE_IMAGE);
	ASSERT_EQ(sam3_set_image(ctx, pixels, sz, sz), SAM3_OK);
	sam3_cache_stats(ctx, &st);
	ASSERT_EQ(st.image_hits, 1u); /* still 1; clear reset the entry */
	ASSERT(st.image_misses >= 2u);

	free(pixels);
	sam3_free(ctx);
}
```

Add to `main`:
```c
	test_top_level_cache_clear_and_stats();
```

- [ ] **Step 6: Run all tests**

Run:
```
cd build && cmake --build . -- -j && \
  ctest --output-on-failure
```
Expected: all tests pass. (Existing image / video / segment tests continue working — they call `sam3_init()` which now returns a context with default cache opts, and the on-disk behavior is unchanged.)

- [ ] **Step 7: Commit**

```bash
git add include/sam3/sam3.h src/sam3.c src/model/sam3_internal.h \
        src/model/feature_cache.h tests/test_processor_cache.c
git commit -m "sam3: public init_ex / cache_clear / cache_stats API"
```

---

## Task 8: Drop `model_arena` rollback safety code in segment

**Files:**
- Modify: `src/model/sam3_processor.c`

The `persist_save = proc->model_arena.offset` rollback at the end of `sam3_processor_segment` (around lines 854 / 1190 / 1205) was tied to the old `weights_end` model where image features lived in `model_arena`. Now that image features live in slot arenas, the rollback only needs to undo allocations made *during* the segment (text features used to be copied here too, but that copy is gone after Task 6). What remains is the prompt-token tensor allocation done by `sam3_project_prompts` and a few small inter-stage tensors. Those still want to be rolled back so segment is reentrant.

- [ ] **Step 1: Read the current rollback sites**

Read `src/model/sam3_processor.c` lines 850-870, 1180-1215. Confirm `persist_save` is captured once and restored on both fail and success paths. Keep this logic — it is still load-bearing.

- [ ] **Step 2: Add a regression test**

Append to `tests/test_processor_cache.c` before `main`:

```c
static void test_segment_does_not_grow_model_arena(void)
{
	if (!model_available()) {
		printf("  model weights missing, skipping\n");
		return;
	}

	struct sam3_processor proc;
	enum sam3_error err = sam3_processor_init(&proc,
						  SAM3_BACKBONE_HIERA, 4);
	ASSERT_EQ(err, SAM3_OK);
	struct sam3_weight_file wf;
	ASSERT_EQ(sam3_weight_open(&wf, MODEL_PATH), SAM3_OK);
	ASSERT_EQ(sam3_processor_load(&proc, &wf, VOCAB_PATH), SAM3_OK);

	int sz = sam3_processor_img_size(&proc);
	uint8_t *pixels = calloc((size_t)sz * sz * 3, 1);
	ASSERT_EQ(sam3_processor_set_image(&proc, pixels, sz, sz), SAM3_OK);

	size_t arena_after_load = proc.model_arena.offset;

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT,
				.text = {.text = "cat"}};
	struct sam3_result r = {0};

	for (int i = 0; i < 4; i++) {
		ASSERT_EQ(sam3_processor_segment(&proc, &p, 1, &r), SAM3_OK);
		sam3_result_free(&r);
		memset(&r, 0, sizeof(r));
	}

	/* model_arena offset should not grow across repeated segments
	 * once weights are loaded — all per-call data lives in scratch
	 * or cache arenas. */
	ASSERT_EQ(proc.model_arena.offset, arena_after_load);

	free(pixels);
	sam3_processor_free(&proc);
	sam3_weight_close(&wf);
}
```

Add to `main`:
```c
	test_segment_does_not_grow_model_arena();
```

- [ ] **Step 3: Run the test**

Run:
```
cd build && cmake --build . --target test_processor_cache && \
  ctest -R '^test_processor_cache$' --output-on-failure
```
Expected: pass. If it fails because `persist_save` rollback is missing some allocation site, that's a real bug — investigate before committing.

- [ ] **Step 4: Commit**

```bash
git add tests/test_processor_cache.c
git commit -m "test: model_arena does not grow across repeated segments"
```

---

## Task 9: Final regression sweep + docs note

**Files:**
- Modify: `docs/architecture.md` (one paragraph if the file documents the encoder caches; else skip)

- [ ] **Step 1: Run the full test suite**

Run:
```
cd build && cmake --build . -- -j && ctest --output-on-failure
```
Expected: all pre-existing tests pass plus the new ones. If a test under `test_sam3_*` fails because it reads `proc.image_loaded` semantics or `text_thread_active`, fix the test (semantics are preserved) — but do **not** revert any cache logic without the user's say-so.

- [ ] **Step 2: Search architecture doc for stale references**

Run: `grep -n "weights_end\|text_features_async\|text_persist_arena" docs/architecture.md`
Expected: zero or a small number of hits. For each hit, replace the reference with a one-line description of the new cache flow:
- `weights_end` → "image features cached in `img_cache` slot arenas"
- `text_features_async` → "text features cached in `txt_cache` slot bundles"
- `text_persist_arena` → "text cache shared arena (partitioned per slot)"

If `docs/architecture.md` does not mention these symbols, skip this step.

- [ ] **Step 3: Commit any doc changes**

```bash
git add docs/architecture.md
git commit -m "docs: note encoder feature cache replaces weights_end + text_features_async"
```

(If no changes, skip the commit.)

- [ ] **Step 4: Final commit summary**

Run: `git log --oneline -10`
Expected: 8–9 fresh commits implementing the cache, ending at the doc update.

---

## Self-Review Checklist

- All cache structs (`sam3_image_bundle`, `sam3_text_bundle`, `sam3_image_feature_cache`, `sam3_text_feature_cache`) are defined exactly once (in `feature_cache.h`).
- `sam3_cache_stats` is defined exactly once. Tasks 2-6 use a forward declaration in `feature_cache.h`; Task 7 moves the definition to `sam3/sam3.h` and the forward declaration is removed.
- `current_img_slot`, `text_worker_slot`, `text_cached_bundle` are introduced in Task 5/6 and used consistently afterwards.
- All allocator calls outside per-slot arenas (cache structs, slot tables) go through `calloc`/`free` per the design's "cache lifetime tracks the processor" rule. The hot path (encoder writing into a slot arena) uses `sam3_arena_alloc` only.
- Tests cover: hash properties, image cache miss/hit/LRU/collision/clear, text cache miss/hit/LRU/clear, end-to-end image hit, end-to-end text hit, top-level cache_clear/stats, model_arena non-growth.
