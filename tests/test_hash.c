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
