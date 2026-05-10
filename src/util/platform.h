/*
 * src/util/platform.h - Cross-platform OS helpers
 *
 * Provides a narrow portability layer for file mapping, path inspection,
 * CPU-count detection, directory listing, temp directories, and thread
 * primitives. Runtime code uses these helpers instead of POSIX or Win32
 * APIs directly.
 *
 * Key types:  sam3_file_map, sam3_dir_list, sam3_thread
 * Depends on: sam3/sam3_types.h
 * Used by:    core/weight.c, weight_safetensors.c, util/threadpool.c,
 *             util/video.c, model/feature_cache.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_UTIL_PLATFORM_H
#define SAM3_UTIL_PLATFORM_H

#include <stddef.h>

#include "sam3/sam3_types.h"

enum sam3_prefetch_hint {
	SAM3_PREFETCH_SEQUENTIAL = 1,
	SAM3_PREFETCH_RANDOM = 2,
	SAM3_PREFETCH_WILLNEED = 3,
};

struct sam3_file_map {
	void *data;
	size_t size;
#ifdef _WIN32
	void *file_handle;
	void *mapping_handle;
#endif
};

struct sam3_dir_list {
	char **names;
	int n;
};

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef struct sam3_thread { HANDLE handle; } sam3_thread;
typedef struct sam3_mutex { CRITICAL_SECTION cs; } sam3_mutex;
typedef struct sam3_cond { CONDITION_VARIABLE cv; } sam3_cond;
#else
#include <pthread.h>
typedef struct sam3_thread { pthread_t thread; } sam3_thread;
typedef struct sam3_mutex { pthread_mutex_t mutex; } sam3_mutex;
typedef struct sam3_cond { pthread_cond_t cond; } sam3_cond;
#endif

enum sam3_error sam3_file_map_read(const char *path, struct sam3_file_map *out);
void sam3_file_unmap(struct sam3_file_map *map);
void sam3_file_prefetch(const struct sam3_file_map *map,
				enum sam3_prefetch_hint hint);
int sam3_platform_cpu_count(void);
int sam3_path_is_regular(const char *path);
int sam3_path_is_dir(const char *path);
const char *sam3_path_basename_sep(const char *path);

enum sam3_error sam3_dir_list_open(const char *path,
				   int (*accept)(const char *name),
				   struct sam3_dir_list *out);
void sam3_dir_list_free(struct sam3_dir_list *list);

int sam3_platform_mkdir(const char *path);
int sam3_platform_rmdir(const char *path);
char *sam3_platform_temp_dir(const char *prefix);

int sam3_thread_create(sam3_thread *thread, void *(*fn)(void *), void *arg);
int sam3_thread_create_with_stack(sam3_thread *thread, size_t stack_size,
					   void *(*fn)(void *), void *arg);
void sam3_thread_join(sam3_thread *thread);
int sam3_mutex_init(sam3_mutex *mutex);
void sam3_mutex_destroy(sam3_mutex *mutex);
void sam3_mutex_lock(sam3_mutex *mutex);
void sam3_mutex_unlock(sam3_mutex *mutex);
int sam3_cond_init(sam3_cond *cond);
void sam3_cond_destroy(sam3_cond *cond);
void sam3_cond_wait(sam3_cond *cond, sam3_mutex *mutex);
void sam3_cond_signal(sam3_cond *cond);
void sam3_cond_broadcast(sam3_cond *cond);

#endif /* SAM3_UTIL_PLATFORM_H */
