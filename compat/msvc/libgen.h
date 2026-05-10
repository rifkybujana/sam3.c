#ifndef SAM3_COMPAT_MSVC_LIBGEN_H
#define SAM3_COMPAT_MSVC_LIBGEN_H

#include <string.h>

static inline char *dirname(char *path)
{
	char *sep;
	char *backslash;

	if (!path || !path[0])
		return ".";

	sep = strrchr(path, '/');
	backslash = strrchr(path, '\\');
	if (backslash && (!sep || backslash > sep))
		sep = backslash;

	if (!sep)
		return ".";
	if (sep == path) {
		sep[1] = '\0';
		return path;
	}

	*sep = '\0';
	return path;
}

#endif /* SAM3_COMPAT_MSVC_LIBGEN_H */