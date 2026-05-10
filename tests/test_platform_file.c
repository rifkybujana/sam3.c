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

static int accept_generated_frame(const char *name)
{
	return strcmp(name, "001.jpg") == 0 || strcmp(name, "002.png") == 0;
}

static int list_has_name(const struct sam3_dir_list *list, const char *name)
{
	for (int i = 0; i < list->n; i++) {
		if (strcmp(list->names[i], name) == 0)
			return 1;
	}
	return 0;
}

static void write_test_file(const char *path)
{
	FILE *f = fopen(path, "wb");
	ASSERT(f != NULL);
	if (!f)
		return;
	ASSERT_EQ(fwrite("x", 1, 1, f), 1);
	fclose(f);
}

int main(void)
{
	const char *path = "sam3_platform_map_test.bin";
	const char *dir_path = "sam3_platform_dir_test";
	const char *frame_a = "sam3_platform_dir_test/001.jpg";
	const char *frame_b = "sam3_platform_dir_test/002.png";
	const char *ignored = "sam3_platform_dir_test/notes.txt";
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

	remove(frame_a);
	remove(frame_b);
	remove(ignored);
	sam3_platform_rmdir(dir_path);
	ASSERT_EQ(sam3_platform_mkdir(dir_path), 0);
	write_test_file(frame_b);
	write_test_file(ignored);
	write_test_file(frame_a);

	struct sam3_dir_list list;
	memset(&list, 0, sizeof(list));
	ASSERT_EQ(sam3_dir_list_open(dir_path, accept_generated_frame, &list),
		  SAM3_OK);
	ASSERT_EQ(list.n, 2);
	ASSERT(list_has_name(&list, "001.jpg"));
	ASSERT(list_has_name(&list, "002.png"));
	sam3_dir_list_free(&list);

	remove(path);
	remove(frame_a);
	remove(frame_b);
	remove(ignored);
	sam3_platform_rmdir(dir_path);
	TEST_REPORT();
}
