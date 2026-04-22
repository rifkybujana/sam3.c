/*
 * src/model/feature_cache.c - Image and text feature LRU caches.
 *
 * Implements the cache module declared in feature_cache.h. Both image
 * and text caches use one sam3_arena per slot, sized for the encoder's
 * worst-case output. Allocation outside the per-slot arenas (the
 * slots[] table and the cache struct itself) goes through calloc/free
 * since cache lifetime tracks the processor, not an arena.
 *
 * Key types:  sam3_image_feature_cache, sam3_text_feature_cache
 * Depends on: feature_cache.h, util/log.h
 * Used by:    src/model/sam3_processor.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "feature_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util/log.h"
#include "feature_cache_persist.h"

/* --- disk-tier helpers --- */

static char *build_spill_path(const char *dir, int slot_idx, uint64_t hash)
{
	char buf[512];
	int n = snprintf(buf, sizeof(buf), "%s/slot_%d_%016llx.bin",
			 dir, slot_idx, (unsigned long long)hash);
	if (n < 0 || (size_t)n >= sizeof(buf))
		return NULL;
	char *p = malloc((size_t)n + 1);
	if (!p)
		return NULL;
	memcpy(p, buf, (size_t)n + 1);
	return p;
}

static int count_hot_slots(const struct sam3_image_feature_cache *c)
{
	int n = 0;
	for (int i = 0; i < c->n_slots; i++) {
		const struct sam3_image_cache_slot *s = &c->slots[i];
		if (s->hash != 0 && s->disk_path == NULL)
			n++;
	}
	return n;
}

static int find_lru_hot_slot(const struct sam3_image_feature_cache *c,
			     int exclude_idx)
{
	int best = -1;
	uint64_t best_tick = 0;
	for (int i = 0; i < c->n_slots; i++) {
		if (i == exclude_idx)
			continue;
		const struct sam3_image_cache_slot *s = &c->slots[i];
		if (s->hash == 0 || s->disk_path != NULL)
			continue;
		if (best < 0 || s->lru_tick < best_tick) {
			best = i;
			best_tick = s->lru_tick;
		}
	}
	return best;
}

/*
 * Demote a populated HOT slot to DISK tier. Writes the bundle to a
 * file under spill_dir and releases the 256 MiB arena. Cost is
 * bounded by disk bandwidth (roughly 150 ms for a 235 MiB bundle on
 * a modern SSD, vs ~3-20 s with zlib).
 */
static void image_slot_demote_to_disk(struct sam3_image_feature_cache *c,
				      int idx)
{
	struct sam3_image_cache_slot *s = &c->slots[idx];
	if (s->hash == 0 || s->disk_path != NULL || !s->arena.base)
		return;
	if (!c->spill_dir) {
		sam3_log_warn("image_cache: no spill_dir, dropping slot");
		sam3_arena_reset(&s->arena);
		s->hash = 0;
		s->prefix_len = 0;
		memset(&s->bundle, 0, sizeof(s->bundle));
		return;
	}

	char *path = build_spill_path(c->spill_dir, idx, s->hash);
	if (!path) {
		sam3_log_warn("image_cache: spill path alloc failed, dropping");
		sam3_arena_reset(&s->arena);
		s->hash = 0;
		s->prefix_len = 0;
		memset(&s->bundle, 0, sizeof(s->bundle));
		return;
	}

	enum sam3_error err = sam3_image_bundle_write_uncompressed(path,
								   &s->bundle);
	if (err != SAM3_OK) {
		sam3_log_warn("image_cache: demote write failed (%d)", err);
		free(path);
		sam3_arena_reset(&s->arena);
		s->hash = 0;
		s->prefix_len = 0;
		memset(&s->bundle, 0, sizeof(s->bundle));
		return;
	}

	sam3_arena_free(&s->arena);
	s->disk_path = path;
	memset(&s->bundle, 0, sizeof(s->bundle));
	c->demotions++;
}

/*
 * Promote a DISK slot back to HOT. Re-initializes the arena, loads
 * the bundle from disk, unlinks the spill file. Returns SAM3_OK on
 * success; on failure the slot is invalidated (treated as a miss).
 */
static enum sam3_error
image_slot_promote_from_disk(struct sam3_image_feature_cache *c, int idx)
{
	struct sam3_image_cache_slot *s = &c->slots[idx];
	if (!s->disk_path)
		return SAM3_OK;

	enum sam3_error err = sam3_arena_init(&s->arena, s->arena_bytes);
	if (err != SAM3_OK) {
		sam3_log_error("image_cache: arena_init failed in promote");
		remove(s->disk_path);
		free(s->disk_path);
		s->disk_path = NULL;
		s->hash = 0;
		return err;
	}
	err = sam3_image_bundle_read_uncompressed(s->disk_path, &s->arena,
						  &s->bundle);
	if (err != SAM3_OK) {
		sam3_log_error("image_cache: promote read failed (%d)", err);
		sam3_arena_free(&s->arena);
		remove(s->disk_path);
		free(s->disk_path);
		s->disk_path = NULL;
		s->hash = 0;
		return err;
	}
	remove(s->disk_path);
	free(s->disk_path);
	s->disk_path = NULL;
	c->promotions++;
	return SAM3_OK;
}

/*
 * If the HOT budget is saturated, demote the LRU HOT slot (excluding
 * @exclude_idx, typically the slot about to become HOT) to disk.
 */
static void ensure_hot_budget(struct sam3_image_feature_cache *c,
			      int exclude_idx)
{
	if (c->n_hot_max <= 0)
		return;
	while (count_hot_slots(c) >= c->n_hot_max) {
		int victim = find_lru_hot_slot(c, exclude_idx);
		if (victim < 0)
			break;
		image_slot_demote_to_disk(c, victim);
	}
}

/* --- image cache --- */

struct sam3_image_feature_cache *
sam3_image_cache_create(int n_slots, size_t slot_arena_bytes)
{
	return sam3_image_cache_create_ex(n_slots, slot_arena_bytes, 0, NULL);
}

struct sam3_image_feature_cache *
sam3_image_cache_create_ex(int n_slots, size_t slot_arena_bytes,
			   size_t mem_budget_bytes,
			   const char *spill_dir)
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

	/*
	 * Compute hot budget. 0 bytes means "no cap" (all slots stay HOT,
	 * matching pre-tiering behavior).
	 */
	if (mem_budget_bytes == 0) {
		c->n_hot_max = 0;
	} else {
		int cap = (int)(mem_budget_bytes / slot_arena_bytes);
		if (cap <= 0)
			cap = 1;
		if (cap > n_slots)
			cap = n_slots;
		c->n_hot_max = cap;
	}

	/*
	 * Resolve spill directory. A user-supplied path is used as-is
	 * (caller owns creation/cleanup). A NULL path auto-creates a
	 * unique per-cache temp dir that the cache will rm on destroy.
	 */
	c->spill_dir = NULL;
	c->owns_spill_dir = 0;
	if (c->n_hot_max > 0) {
		if (spill_dir) {
			c->spill_dir = strdup(spill_dir);
			if (!c->spill_dir)
				goto fail_slots;
			/* mkdir best-effort; existing dir is fine. */
			mkdir(c->spill_dir, 0700);
		} else {
			char tmpl[] = "/tmp/sam3-imgcache-XXXXXX";
			char *d = mkdtemp(tmpl);
			if (!d) {
				sam3_log_error("image_cache: mkdtemp failed");
				goto fail_slots;
			}
			c->spill_dir = strdup(d);
			if (!c->spill_dir)
				goto fail_slots;
			c->owns_spill_dir = 1;
		}
	}

	for (int i = 0; i < n_slots; i++) {
		if (sam3_arena_init(&c->slots[i].arena, slot_arena_bytes)
		    != SAM3_OK) {
			for (int j = 0; j < i; j++)
				sam3_arena_free(&c->slots[j].arena);
			goto fail_spill_dir;
		}
		c->slots[i].arena_bytes = slot_arena_bytes;
	}
	return c;

fail_spill_dir:
	if (c->owns_spill_dir && c->spill_dir)
		rmdir(c->spill_dir);
	free(c->spill_dir);
fail_slots:
	free(c->slots);
	free(c);
	return NULL;
}

void sam3_image_cache_destroy(struct sam3_image_feature_cache *c)
{
	if (!c)
		return;
	for (int i = 0; i < c->n_slots; i++) {
		struct sam3_image_cache_slot *s = &c->slots[i];
		if (s->disk_path) {
			remove(s->disk_path);
			free(s->disk_path);
		}
		sam3_arena_free(&s->arena);
	}
	if (c->owns_spill_dir && c->spill_dir)
		rmdir(c->spill_dir);
	free(c->spill_dir);
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
		struct sam3_image_cache_slot *s = &c->slots[i];
		if (s->disk_path) {
			remove(s->disk_path);
			free(s->disk_path);
			s->disk_path = NULL;
			if (sam3_arena_init(&s->arena, s->arena_bytes)
			    != SAM3_OK)
				sam3_log_error("image_cache: arena_init "
					       "failed during clear");
		} else {
			sam3_arena_reset(&s->arena);
		}
		s->hash = 0;
		s->lru_tick = 0;
		s->prefix_len = 0;
		memset(&s->bundle, 0, sizeof(s->bundle));
	}
	c->next_tick  = 0;
	c->hits       = 0;
	c->misses     = 0;
	c->evictions  = 0;
	c->demotions  = 0;
	c->promotions = 0;
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
			/*
			 * Prefix collision: drop this entry. If it was on
			 * disk, delete the spill file first.
			 */
			if (s->disk_path) {
				remove(s->disk_path);
				free(s->disk_path);
				s->disk_path = NULL;
				if (sam3_arena_init(&s->arena, s->arena_bytes)
				    != SAM3_OK)
					sam3_log_error("image_cache: "
						       "arena_init failed "
						       "during collision");
			}
			s->hash = 0;
			c->evictions++;
			c->misses++;
			return -1;
		}
		if (s->disk_path) {
			/*
			 * Promoting adds one more HOT slot. Demote LRU
			 * HOT first if the budget is saturated.
			 */
			ensure_hot_budget(c, i);
			enum sam3_error err = image_slot_promote_from_disk(c, i);
			if (err != SAM3_OK) {
				c->misses++;
				return -1;
			}
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
	if (s->disk_path) {
		/*
		 * Reclaiming a DISK slot: delete its spill file and
		 * re-init the arena so the caller can write into it.
		 */
		remove(s->disk_path);
		free(s->disk_path);
		s->disk_path = NULL;
		if (sam3_arena_init(&s->arena, s->arena_bytes) != SAM3_OK) {
			sam3_log_error("image_cache: arena_init failed in "
				       "claim_slot");
			return -1;
		}
	} else {
		sam3_arena_reset(&s->arena);
	}
	s->hash = 0;
	s->prefix_len = 0;
	memset(&s->bundle, 0, sizeof(s->bundle));
	/*
	 * About to add one more HOT slot on register. If at budget,
	 * demote an older HOT slot now so we stay under the cap.
	 */
	ensure_hot_budget(c, idx);
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
	/*
	 * claim_slot already reserved headroom, so we should be within
	 * budget here. Re-check defensively — a caller who doesn't pair
	 * claim+register symmetrically still gets correct budget behavior.
	 */
	ensure_hot_budget(c, idx);
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
	c->slots = calloc((size_t)n_slots, sizeof(*c->slots));
	if (!c->slots) {
		free(c);
		return NULL;
	}
	for (int i = 0; i < n_slots; i++) {
		if (sam3_arena_init(&c->slots[i].arena, slot_bytes)
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

void sam3_text_cache_destroy(struct sam3_text_feature_cache *c)
{
	if (!c)
		return;
	for (int i = 0; i < c->n_slots; i++)
		sam3_arena_free(&c->slots[i].arena);
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
		sam3_arena_reset(&c->slots[i].arena);
		c->slots[i].hash = 0;
		c->slots[i].lru_tick = 0;
		c->slots[i].prefix_len = 0;
		c->slots[i].bundle.features = NULL;
		c->slots[i].bundle.n_tokens = 0;
	}
	c->next_tick = 0;
	c->hits = 0;
	c->misses = 0;
	c->evictions = 0;
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
	sam3_arena_reset(&s->arena);
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
	if (!c || idx < 0 || idx >= c->n_slots)
		return NULL;
	return &c->slots[idx].arena;
}
