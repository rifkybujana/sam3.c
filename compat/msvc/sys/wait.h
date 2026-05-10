#ifndef SAM3_COMPAT_MSVC_SYS_WAIT_H
#define SAM3_COMPAT_MSVC_SYS_WAIT_H

#define WIFEXITED(status) (1)
#define WEXITSTATUS(status) (status)

#endif /* SAM3_COMPAT_MSVC_SYS_WAIT_H */