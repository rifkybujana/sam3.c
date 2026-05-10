#ifndef SAM3_COMPAT_MSVC_UNISTD_H
#define SAM3_COMPAT_MSVC_UNISTD_H

#include <io.h>
#include <sys/stat.h>

#ifndef F_OK
#define F_OK 0
#endif
#ifndef W_OK
#define W_OK 2
#endif
#ifndef R_OK
#define R_OK 4
#endif

#ifndef access
#define access _access
#endif

#ifndef isatty
#define isatty _isatty
#endif

#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & _S_IFDIR) == _S_IFDIR)
#endif

#endif /* SAM3_COMPAT_MSVC_UNISTD_H */