#ifndef SAM3_COMPAT_MSVC_PTHREAD_H
#define SAM3_COMPAT_MSVC_PTHREAD_H

static inline int
pthread_set_qos_class_self_np(int qos_class, int relative_priority)
{
	(void)qos_class;
	(void)relative_priority;
	return 0;
}

#endif /* SAM3_COMPAT_MSVC_PTHREAD_H */