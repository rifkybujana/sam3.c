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
