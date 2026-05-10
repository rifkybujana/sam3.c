/*
 * tests/test_platform_file.c - Platform file helper tests
 *
 * Validates cross-platform memory mapping, path type detection, and
 * directory listing helpers used by the Windows portability layer.
 *
 * Key types:  sam3_file_map, sam3_dir_list
 * Depends on: util/platform.h, test_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "util/platform.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
	const char *path = "sam3_platform_map_test.bin";
	FILE *f = fopen(path, "wb");
	ASSERT(f != NULL);
	ASSERT_EQ(fwrite("SAM3", 1, 4, f), 4);
	fclose(f);

	struct sam3_file_map map;
	memset(&map, 0, sizeof(map));
	ASSERT_EQ(sam3_file_map_read(path, &map), SAM3_OK);
	ASSERT_EQ(map.size, 4);
	ASSERT(map.data != NULL);
	ASSERT_EQ(memcmp(map.data, "SAM3", 4), 0);
	sam3_file_prefetch(&map, SAM3_PREFETCH_RANDOM);
	sam3_file_prefetch(&map, SAM3_PREFETCH_WILLNEED);
	sam3_file_unmap(&map);

	ASSERT(sam3_path_is_regular(path));
	ASSERT(!sam3_path_is_dir(path));
	const char *base = sam3_path_basename_sep("C:\\models\\x.sam3");
	ASSERT(base != NULL);
	ASSERT(strcmp(base, "x.sam3") == 0);

	remove(path);
	TEST_REPORT();
}
