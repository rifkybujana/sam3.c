#ifndef SAM3_COMPAT_MSVC_GETOPT_H
#define SAM3_COMPAT_MSVC_GETOPT_H

#include <string.h>

#define no_argument 0
#define required_argument 1
#define optional_argument 2

struct option {
	const char *name;
	int has_arg;
	int *flag;
	int val;
};

static char *optarg;
static int optind = 1;

static int getopt_long(int argc, char * const argv[], const char *optstring,
			       const struct option *longopts, int *longindex)
{
	const char *arg;

	optarg = 0;
	if (optind >= argc)
		return -1;

	arg = argv[optind];
	if (!arg || arg[0] != '-' || arg[1] == '\0')
		return -1;
	if (strcmp(arg, "--") == 0) {
		optind++;
		return -1;
	}

	if (arg[1] == '-') {
		const char *name = arg + 2;
		const char *eq = strchr(name, '=');
		size_t name_len = eq ? (size_t)(eq - name) : strlen(name);

		for (int i = 0; longopts && longopts[i].name; i++) {
			if (strncmp(name, longopts[i].name, name_len) != 0 ||
			    longopts[i].name[name_len] != '\0')
				continue;

			if (longindex)
				*longindex = i;
			if (longopts[i].has_arg == required_argument) {
				if (eq) {
					optarg = (char *)(eq + 1);
				} else if (optind + 1 < argc) {
					optarg = argv[++optind];
				} else {
					optind++;
					return '?';
				}
			}

			optind++;
			if (longopts[i].flag) {
				*longopts[i].flag = longopts[i].val;
				return 0;
			}
			return longopts[i].val;
		}

		optind++;
		return '?';
	}

	{
		char opt = arg[1];
		const char *spec = strchr(optstring, opt);
		if (!spec) {
			optind++;
			return '?';
		}
		if (spec[1] == ':') {
			if (arg[2]) {
				optarg = (char *)(arg + 2);
			} else if (optind + 1 < argc) {
				optarg = argv[++optind];
			} else {
				optind++;
				return '?';
			}
		}
		optind++;
		return opt;
	}
}

#endif /* SAM3_COMPAT_MSVC_GETOPT_H */