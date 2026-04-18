/*
 * tools/cli_info.c - SAM3 info subcommand
 *
 * Opens a .sam3 weight file and prints metadata (model config,
 * tensor count, file size). Supports --json output.
 *
 * Key types:  (uses sam3_weight_header, sam3_weight_file)
 * Depends on: cli_common.h, core/weight.h, util/error.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_common.h"
#include "cli_info.h"
#include "core/weight.h"
#include "util/error.h"

static void print_usage(const char *prog)
{
	fprintf(stderr, "Usage: %s <model.sam3> [options]\n\n", prog);
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  --json     Output as JSON\n");
	fprintf(stderr, "  -h, --help Show this help\n");
}

int cli_info(int argc, char **argv)
{
	const char *model_path = NULL;
	int json = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			print_usage(argv[0]);
			return SAM3_EXIT_OK;
		} else if (strcmp(argv[i], "--json") == 0) {
			json = 1;
		} else if (argv[i][0] != '-') {
			if (model_path) {
				fprintf(stderr,
					"error: multiple model paths\n");
				return SAM3_EXIT_USAGE;
			}
			model_path = argv[i];
		} else {
			fprintf(stderr, "error: unknown option '%s'\n",
				argv[i]);
			return SAM3_EXIT_USAGE;
		}
	}

	if (!model_path) {
		fprintf(stderr, "error: model path required\n");
		print_usage(argv[0]);
		return SAM3_EXIT_USAGE;
	}

	struct sam3_weight_file wf = {0};
	enum sam3_error err = sam3_weight_open(&wf, model_path);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to open '%s': %s\n",
			model_path, sam3_error_str(err));
		return sam3_error_to_exit(err);
	}

	if (json) {
		cli_json_model_info(stdout, wf.header, wf.mapped_size);
	} else {
		const struct sam3_weight_header *h = wf.header;
		printf("sam3 model info\n");
		printf("  file:            %s\n", model_path);
		printf("  format version:  %u\n", h->version);
		printf("  image_size:      %d\n", h->image_size);
		printf("  encoder_dim:     %d\n", h->encoder_dim);
		printf("  decoder_dim:     %d\n", h->decoder_dim);
		printf("  encoder_layers:  %d\n", h->n_encoder_layers);
		printf("  decoder_layers:  %d\n", h->n_decoder_layers);
		printf("  n_tensors:       %u\n", h->n_tensors);
		const char *variant_str = (h->reserved[1] == SAM3_VARIANT_SAM3_1)
					   ? "sam3.1" : "sam3";
		uint32_t scales = h->reserved[2]
				   ? h->reserved[2]
				   : (h->reserved[1] == SAM3_VARIANT_SAM3_1 ? 3 : 4);
		printf("  variant:         %s\n", variant_str);
		printf("  n_fpn_scales:    %u\n", scales);

		double gb = (double)wf.mapped_size /
			    (1024.0 * 1024.0 * 1024.0);
		double mb = (double)wf.mapped_size /
			    (1024.0 * 1024.0);
		if (gb >= 1.0)
			printf("  file_size:       %.1f GB\n", gb);
		else
			printf("  file_size:       %.1f MB\n", mb);
	}

	sam3_weight_close(&wf);
	return SAM3_EXIT_OK;
}
