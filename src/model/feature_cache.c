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
	c->hits = 0;
	c->misses = 0;
	c->evictions = 0;
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
