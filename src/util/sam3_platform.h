/*
 * src/util/sam3_platform.h - Cross-platform shims for POSIX APIs
 *
 * MSYS2-UCRT64 / MinGW-w64 builds need shims for a handful of
 * POSIX APIs that sam3.c uses unconditionally on macOS/Linux:
 *   - <unistd.h>          -> <io.h> + <process.h> on Windows
 *   - sysconf(_SC_NPROC)  -> GetSystemInfo()
 *
 * <sys/mman.h> is provided on Windows by the MSYS2 mman-win32
 * package (installed to /ucrt64/include/sys/mman.h), so callers can
 * keep including it without a shim.
 *
 * Key types:  none
 * Depends on: <Windows.h> on Windows, <unistd.h> elsewhere
 * Used by:    src/util/threadpool.c (sysconf), and any caller that
 *             only needs unistd for read/write/close.
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */
#ifndef SAM3_UTIL_SAM3_PLATFORM_H
#define SAM3_UTIL_SAM3_PLATFORM_H

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#  endif
#  include <Windows.h>
#  include <io.h>
#  include <process.h>
#  include <direct.h>     /* _mkdir */
#  include <math.h>
/* MinGW provides <unistd.h> too, but MSVC does not; include it
   defensively only when present. */
#  if defined(__has_include)
#    if __has_include(<unistd.h>)
#      include <unistd.h>
#    endif
#  endif

/* mman-win32 ships <sys/mman.h> with mmap/munmap but no madvise.
   Provide a no-op shim and define the standard advice constants so
   callers that include sam3_platform.h after <sys/mman.h> still
   compile. madvise is purely advisory; ignoring it costs nothing. */
#  ifndef MADV_NORMAL
#    define MADV_NORMAL     0
#    define MADV_RANDOM     1
#    define MADV_SEQUENTIAL 2
#    define MADV_WILLNEED   3
#    define MADV_DONTNEED   4
#  endif
static inline int sam3_madvise_compat(void *addr, size_t length, int advice)
{
	(void)addr; (void)length; (void)advice;
	return 0;
}
#  ifndef madvise
#    define madvise(addr, len, advice) sam3_madvise_compat((addr), (len), (advice))
#  endif

/* The POSIX two-arg mkdir(path, mode) collapses to Windows _mkdir(path);
   the mode bits are not honored on NTFS anyway. */
#  ifndef sam3_mkdir_compat_defined
#    define sam3_mkdir_compat_defined 1
static inline int sam3_mkdir_compat(const char *path, int mode)
{
	(void)mode;
	return _mkdir(path);
}
#    define mkdir(path, mode) sam3_mkdir_compat((path), (mode))
#  endif

static inline long sam3_get_nprocessors_onln(void)
{
	SYSTEM_INFO si;
	GetSystemInfo(&si);
	return (long)si.dwNumberOfProcessors;
}

/* 64-bit-safe fstat for Windows: the default `struct stat` on MSVCRT
   has a 32-bit st_size and fstat() returns EOVERFLOW for files >= 2 GB.
   Use _fstat64 + struct __stat64 instead. Mirror under the POSIX
   names so call sites don't need #ifdef. */
#  include <sys/stat.h>
#  define sam3_stat_t      struct __stat64
#  define sam3_fstat(fd, sb) _fstat64((fd), (sb))

/* Open with O_BINARY on Windows so reads don't translate CRLF. */
#  include <fcntl.h>
#  ifndef O_BINARY
#    define O_BINARY 0
#  endif
static inline int sam3_open_rdonly(const char *path)
{
	return open(path, O_RDONLY | O_BINARY);
}

/* POSIX mkdtemp shim for Windows. The POSIX template is something like
 * "/tmp/sam3-imgcache-XXXXXX". Windows has no /tmp and no mkdtemp(), so
 * we resolve the basename, point it at GetTempPathA(), and try random
 * 6-char suffixes until _mkdir() succeeds. The returned pointer is to a
 * static buffer; callers must strdup() the result before another call. */
#  include <string.h>
#  include <stdio.h>
static inline char *sam3_mkdtemp_compat(char *tmpl)
{
	char tmpdir[MAX_PATH];
	DWORD n = GetTempPathA((DWORD)sizeof(tmpdir), tmpdir);
	if (n == 0 || n >= sizeof(tmpdir))
		return NULL;
	const char *base = strrchr(tmpl, '/');
	if (!base)
		base = strrchr(tmpl, '\\');
	base = base ? base + 1 : tmpl;
	char base_buf[MAX_PATH];
	strncpy(base_buf, base, sizeof(base_buf) - 1);
	base_buf[sizeof(base_buf) - 1] = 0;
	char *xxx = strstr(base_buf, "XXXXXX");
	if (!xxx)
		return NULL;
	static const char cs[] = "abcdefghijklmnopqrstuvwxyz0123456789";
	static char ret_buf[MAX_PATH * 2];
	for (int attempt = 0; attempt < 100; attempt++) {
		unsigned r = (unsigned)GetTickCount() ^
			((unsigned)GetCurrentProcessId() << 8) ^
			((unsigned)attempt * 2654435761u);
		for (int i = 0; i < 6; i++) {
			xxx[i] = cs[(r >> (i * 5)) % 36];
		}
		char path[MAX_PATH * 2];
		snprintf(path, sizeof(path), "%s%s", tmpdir, base_buf);
		if (_mkdir(path) == 0) {
			size_t plen = strlen(path);
			if (plen >= sizeof(ret_buf))
				return NULL;
			memcpy(ret_buf, path, plen + 1);
			return ret_buf;
		}
	}
	return NULL;
}
#  ifndef mkdtemp
#    define mkdtemp(tmpl) sam3_mkdtemp_compat((tmpl))
#  endif
#else
#  include <unistd.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  define sam3_stat_t      struct stat
#  define sam3_fstat(fd, sb) fstat((fd), (sb))
static inline int sam3_open_rdonly(const char *path)
{
	return open(path, O_RDONLY);
}
static inline long sam3_get_nprocessors_onln(void)
{
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	return n > 0 ? n : 1;
}
#endif

#endif /* SAM3_UTIL_SAM3_PLATFORM_H */
