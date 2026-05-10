/*
 * src/util/platform.c - Cross-platform OS helper implementation
 *
 * Wraps the small set of OS facilities SAM3 needs behind a stable C
 * interface so runtime code can remain portable across POSIX and Windows.
 *
 * Key types:  sam3_file_map, sam3_dir_list, sam3_thread
 * Depends on: util/platform.h
 * Used by:    core/weight.c, weight_safetensors.c, util/threadpool.c,
 *             util/video.c, model/feature_cache.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "util/platform.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32

#include <io.h>

#else

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

static char *sam3_platform_strdup(const char *s)
{
	size_t n = strlen(s) + 1;
	char *out = malloc(n);
	if (!out)
		return NULL;
	memcpy(out, s, n);
	return out;
}

static enum sam3_error dir_list_push(struct sam3_dir_list *list,
					     const char *name)
{
	char **next;
	char *copy;

	if (list->n == INT32_MAX)
		return SAM3_ENOMEM;
	next = realloc(list->names, (size_t)(list->n + 1) * sizeof(*next));
	if (!next)
		return SAM3_ENOMEM;
	list->names = next;
	copy = sam3_platform_strdup(name);
	if (!copy)
		return SAM3_ENOMEM;
	list->names[list->n++] = copy;
	return SAM3_OK;
}

#ifdef _WIN32

enum sam3_error sam3_file_map_read(const char *path, struct sam3_file_map *out)
{
	HANDLE file;
	HANDLE mapping;
	LARGE_INTEGER size;
	void *data;

	if (!path || !out)
		return SAM3_EINVAL;
	memset(out, 0, sizeof(*out));

	file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
			   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file == INVALID_HANDLE_VALUE)
		return SAM3_EIO;
	if (!GetFileSizeEx(file, &size) || size.QuadPart < 0 ||
	    (uint64_t)size.QuadPart > (uint64_t)SIZE_MAX) {
		CloseHandle(file);
		return SAM3_EIO;
	}
	if (size.QuadPart == 0) {
		CloseHandle(file);
		return SAM3_OK;
	}

	mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
	if (!mapping) {
		CloseHandle(file);
		return SAM3_EIO;
	}
	data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
	if (!data) {
		CloseHandle(mapping);
		CloseHandle(file);
		return SAM3_EIO;
	}

	out->data = data;
	out->size = (size_t)size.QuadPart;
	out->file_handle = file;
	out->mapping_handle = mapping;
	return SAM3_OK;
}

void sam3_file_unmap(struct sam3_file_map *map)
{
	if (!map)
		return;
	if (map->data)
		UnmapViewOfFile(map->data);
	if (map->mapping_handle)
		CloseHandle((HANDLE)map->mapping_handle);
	if (map->file_handle)
		CloseHandle((HANDLE)map->file_handle);
	memset(map, 0, sizeof(*map));
}

void sam3_file_prefetch(const struct sam3_file_map *map,
				enum sam3_prefetch_hint hint)
{
	(void)map;
	(void)hint;
}

int sam3_platform_cpu_count(void)
{
	DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
	if (n == 0) {
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		n = info.dwNumberOfProcessors;
	}
	return n > 0 ? (int)n : 1;
}

int sam3_path_is_regular(const char *path)
{
	DWORD attrs;
	if (!path)
		return 0;
	attrs = GetFileAttributesA(path);
	return attrs != INVALID_FILE_ATTRIBUTES &&
	       !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

int sam3_path_is_dir(const char *path)
{
	DWORD attrs;
	if (!path)
		return 0;
	attrs = GetFileAttributesA(path);
	return attrs != INVALID_FILE_ATTRIBUTES &&
	       (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

enum sam3_error sam3_dir_list_open(const char *path,
				   int (*accept)(const char *name),
				   struct sam3_dir_list *out)
{
	WIN32_FIND_DATAA data;
	HANDLE find;
	char *pattern;
	size_t len;
	enum sam3_error err = SAM3_OK;

	if (!path || !out)
		return SAM3_EINVAL;
	memset(out, 0, sizeof(*out));

	len = strlen(path);
	pattern = malloc(len + 3);
	if (!pattern)
		return SAM3_ENOMEM;
	memcpy(pattern, path, len);
	if (len > 0 && (path[len - 1] == '/' || path[len - 1] == '\\')) {
		pattern[len++] = '*';
	} else {
		pattern[len++] = '\\';
		pattern[len++] = '*';
	}
	pattern[len] = '\0';

	find = FindFirstFileA(pattern, &data);
	free(pattern);
	if (find == INVALID_HANDLE_VALUE)
		return SAM3_EIO;

	do {
		const char *name = data.cFileName;
		if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
			continue;
		if (accept && !accept(name))
			continue;
		err = dir_list_push(out, name);
		if (err != SAM3_OK)
			break;
	} while (FindNextFileA(find, &data));

	FindClose(find);
	if (err != SAM3_OK)
		sam3_dir_list_free(out);
	return err;
}

int sam3_platform_mkdir(const char *path)
{
	if (!path)
		return -1;
	if (CreateDirectoryA(path, NULL))
		return 0;
	return GetLastError() == ERROR_ALREADY_EXISTS ? 0 : -1;
}

int sam3_platform_rmdir(const char *path)
{
	if (!path)
		return -1;
	return RemoveDirectoryA(path) ? 0 : -1;
}

char *sam3_platform_temp_dir(const char *prefix)
{
	char temp_root[MAX_PATH];
	DWORD root_len = GetTempPathA((DWORD)sizeof(temp_root), temp_root);
	DWORD pid = GetCurrentProcessId();
	DWORD tick = GetTickCount();
	const char *base = prefix ? prefix : "sam3";

	if (root_len == 0 || root_len >= sizeof(temp_root))
		return NULL;

	for (int i = 0; i < 100; i++) {
		char buf[MAX_PATH];
		int n = snprintf(buf, sizeof(buf), "%s%s-%lu-%lu-%d",
				 temp_root, base, (unsigned long)pid,
				 (unsigned long)tick, i);
		char *out;
		if (n < 0 || (size_t)n >= sizeof(buf))
			return NULL;
		if (!CreateDirectoryA(buf, NULL)) {
			if (GetLastError() == ERROR_ALREADY_EXISTS)
				continue;
			return NULL;
		}
		out = sam3_platform_strdup(buf);
		if (!out)
			RemoveDirectoryA(buf);
		return out;
	}
	return NULL;
}

struct thread_start {
	void *(*fn)(void *);
	void *arg;
};

static DWORD WINAPI thread_trampoline(LPVOID arg)
{
	struct thread_start *start = arg;
	void *(*fn)(void *) = start->fn;
	void *fn_arg = start->arg;
	free(start);
	fn(fn_arg);
	return 0;
}

int sam3_thread_create_with_stack(sam3_thread *thread, size_t stack_size,
					   void *(*fn)(void *), void *arg)
{
	struct thread_start *start;
	HANDLE handle;

	if (!thread || !fn)
		return -1;
	start = malloc(sizeof(*start));
	if (!start)
		return -1;
	start->fn = fn;
	start->arg = arg;
	handle = CreateThread(NULL, stack_size, thread_trampoline,
			      start, 0, NULL);
	if (!handle) {
		free(start);
		return -1;
	}
	thread->handle = handle;
	return 0;
}

void sam3_thread_join(sam3_thread *thread)
{
	if (!thread || !thread->handle)
		return;
	WaitForSingleObject(thread->handle, INFINITE);
	CloseHandle(thread->handle);
	thread->handle = NULL;
}

int sam3_mutex_init(sam3_mutex *mutex)
{
	if (!mutex)
		return -1;
	InitializeCriticalSection(&mutex->cs);
	return 0;
}

void sam3_mutex_destroy(sam3_mutex *mutex)
{
	if (mutex)
		DeleteCriticalSection(&mutex->cs);
}

void sam3_mutex_lock(sam3_mutex *mutex)
{
	EnterCriticalSection(&mutex->cs);
}

void sam3_mutex_unlock(sam3_mutex *mutex)
{
	LeaveCriticalSection(&mutex->cs);
}

int sam3_cond_init(sam3_cond *cond)
{
	if (!cond)
		return -1;
	InitializeConditionVariable(&cond->cv);
	return 0;
}

void sam3_cond_destroy(sam3_cond *cond)
{
	(void)cond;
}

void sam3_cond_wait(sam3_cond *cond, sam3_mutex *mutex)
{
	SleepConditionVariableCS(&cond->cv, &mutex->cs, INFINITE);
}

void sam3_cond_signal(sam3_cond *cond)
{
	WakeConditionVariable(&cond->cv);
}

void sam3_cond_broadcast(sam3_cond *cond)
{
	WakeAllConditionVariable(&cond->cv);
}

#else

enum sam3_error sam3_file_map_read(const char *path, struct sam3_file_map *out)
{
	int fd;
	struct stat st;
	void *data;

	if (!path || !out)
		return SAM3_EINVAL;
	memset(out, 0, sizeof(*out));

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return SAM3_EIO;
	if (fstat(fd, &st) < 0) {
		close(fd);
		return SAM3_EIO;
	}
	if (st.st_size < 0 || (uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
		close(fd);
		return SAM3_EIO;
	}
	if (st.st_size == 0) {
		close(fd);
		return SAM3_OK;
	}

	data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (data == MAP_FAILED)
		return SAM3_EIO;

	out->data = data;
	out->size = (size_t)st.st_size;
	return SAM3_OK;
}

void sam3_file_unmap(struct sam3_file_map *map)
{
	if (!map)
		return;
	if (map->data && map->size > 0)
		munmap(map->data, map->size);
	memset(map, 0, sizeof(*map));
}

void sam3_file_prefetch(const struct sam3_file_map *map,
				enum sam3_prefetch_hint hint)
{
	int advice;

	if (!map || !map->data || map->size == 0)
		return;
	switch (hint) {
	case SAM3_PREFETCH_SEQUENTIAL:
		advice = MADV_SEQUENTIAL;
		break;
	case SAM3_PREFETCH_RANDOM:
		advice = MADV_RANDOM;
		break;
	case SAM3_PREFETCH_WILLNEED:
		advice = MADV_WILLNEED;
		break;
	default:
		return;
	}
	madvise(map->data, map->size, advice);
}

int sam3_platform_cpu_count(void)
{
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	return n > 0 ? (int)n : 1;
}

int sam3_path_is_regular(const char *path)
{
	struct stat st;
	return path && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int sam3_path_is_dir(const char *path)
{
	struct stat st;
	return path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

enum sam3_error sam3_dir_list_open(const char *path,
				   int (*accept)(const char *name),
				   struct sam3_dir_list *out)
{
	DIR *dir;
	struct dirent *ent;
	enum sam3_error err = SAM3_OK;

	if (!path || !out)
		return SAM3_EINVAL;
	memset(out, 0, sizeof(*out));
	dir = opendir(path);
	if (!dir)
		return SAM3_EIO;
	while ((ent = readdir(dir)) != NULL) {
		const char *name = ent->d_name;
		if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
			continue;
		if (accept && !accept(name))
			continue;
		err = dir_list_push(out, name);
		if (err != SAM3_OK)
			break;
	}
	closedir(dir);
	if (err != SAM3_OK)
		sam3_dir_list_free(out);
	return err;
}

int sam3_platform_mkdir(const char *path)
{
	if (!path)
		return -1;
	if (mkdir(path, 0700) == 0)
		return 0;
	return errno == EEXIST ? 0 : -1;
}

int sam3_platform_rmdir(const char *path)
{
	return path ? rmdir(path) : -1;
}

char *sam3_platform_temp_dir(const char *prefix)
{
	const char *root = getenv("TMPDIR");
	const char *base = prefix ? prefix : "sam3";
	char *tmpl;
	int n;

	if (!root || !root[0])
		root = "/tmp";
	n = snprintf(NULL, 0, "%s/%s-XXXXXX", root, base);
	if (n < 0)
		return NULL;
	tmpl = malloc((size_t)n + 1);
	if (!tmpl)
		return NULL;
	snprintf(tmpl, (size_t)n + 1, "%s/%s-XXXXXX", root, base);
	if (!mkdtemp(tmpl)) {
		free(tmpl);
		return NULL;
	}
	return tmpl;
}

int sam3_thread_create_with_stack(sam3_thread *thread, size_t stack_size,
					   void *(*fn)(void *), void *arg)
{
	pthread_attr_t attr;
	pthread_attr_t *attrp = NULL;
	int rc;

	if (!thread || !fn)
		return -1;
	if (stack_size > 0) {
		if (pthread_attr_init(&attr) != 0)
			return -1;
		attrp = &attr;
		if (pthread_attr_setstacksize(&attr, stack_size) != 0) {
			pthread_attr_destroy(&attr);
			return -1;
		}
	}
	rc = pthread_create(&thread->thread, attrp, fn, arg);
	if (attrp)
		pthread_attr_destroy(attrp);
	return rc;
}

void sam3_thread_join(sam3_thread *thread)
{
	if (thread)
		pthread_join(thread->thread, NULL);
}

int sam3_mutex_init(sam3_mutex *mutex)
{
	return pthread_mutex_init(&mutex->mutex, NULL);
}

void sam3_mutex_destroy(sam3_mutex *mutex)
{
	pthread_mutex_destroy(&mutex->mutex);
}

void sam3_mutex_lock(sam3_mutex *mutex)
{
	pthread_mutex_lock(&mutex->mutex);
}

void sam3_mutex_unlock(sam3_mutex *mutex)
{
	pthread_mutex_unlock(&mutex->mutex);
}

int sam3_cond_init(sam3_cond *cond)
{
	return pthread_cond_init(&cond->cond, NULL);
}

void sam3_cond_destroy(sam3_cond *cond)
{
	pthread_cond_destroy(&cond->cond);
}

void sam3_cond_wait(sam3_cond *cond, sam3_mutex *mutex)
{
	pthread_cond_wait(&cond->cond, &mutex->mutex);
}

void sam3_cond_signal(sam3_cond *cond)
{
	pthread_cond_signal(&cond->cond);
}

void sam3_cond_broadcast(sam3_cond *cond)
{
	pthread_cond_broadcast(&cond->cond);
}

#endif

const char *sam3_path_basename_sep(const char *path)
{
	const char *slash;
	const char *backslash;

	if (!path)
		return NULL;
	slash = strrchr(path, '/');
	backslash = strrchr(path, '\\');
	if (!slash || (backslash && backslash > slash))
		slash = backslash;
	return slash ? slash + 1 : path;
}

void sam3_dir_list_free(struct sam3_dir_list *list)
{
	if (!list)
		return;
	for (int i = 0; i < list->n; i++)
		free(list->names[i]);
	free(list->names);
	memset(list, 0, sizeof(*list));
}

int sam3_thread_create(sam3_thread *thread, void *(*fn)(void *), void *arg)
{
	return sam3_thread_create_with_stack(thread, 0, fn, arg);
}
