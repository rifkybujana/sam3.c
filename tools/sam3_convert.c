/*
 * tools/sam3_convert.c - Model weight conversion tool
 *
 * Converts SAM3 model weights from SafeTensors format to sam3's native
 * binary format. Reads input through the weight_reader vtable and writes
 * .sam3 output via sam3_weight_write.
 *
 * Usage: sam3_convert -i <input.safetensors> -o <output.sam3> [options]
 *
 * Key types:  (standalone tool)
 * Depends on: core/weight.h, util/log.h, util/error.h
 * Used by:    end users (pre-inference step)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sam3/sam3_types.h"
#include "core/weight.h"
#include "util/log.h"
#include "util/error.h"

static void print_usage(const char *prog)
{
	printf("sam3 weight conversion tool\n\n");
	printf("Usage: %s -i <input> -o <output> [options]\n\n", prog);
	printf("Required:\n");
	printf("  -i <path>            Input weight file\n");
	printf("  -o <path>            Output .sam3 file\n");
	printf("\nOptions:\n");
	printf("  -f <format>          Input format: "
	       "\"safetensors\" (default)\n");
	printf("  --image-size <N>     Image input size "
	       "(default: 1024)\n");
	printf("  --encoder-dim <N>    Encoder dimension "
	       "(default: 1280)\n");
	printf("  --decoder-dim <N>    Decoder dimension "
	       "(default: 256)\n");
	printf("  --encoder-layers <N> Encoder layer count "
	       "(default: 32)\n");
	printf("  --decoder-layers <N> Decoder layer count "
	       "(default: 2)\n");
	printf("  -v                   Verbose output "
	       "(set log level to DEBUG)\n");
	printf("  -h, --help           Show this help\n");
}

struct convert_args {
	const char *input_path;
	const char *output_path;
	const char *format;
	int         image_size;
	int         encoder_dim;
	int         decoder_dim;
	int         encoder_layers;
	int         decoder_layers;
	int         verbose;
};

static int parse_args(int argc, char **argv, struct convert_args *args)
{
	args->input_path     = NULL;
	args->output_path    = NULL;
	args->format         = "safetensors";
	args->image_size     = 1024;
	args->encoder_dim    = 1280;
	args->decoder_dim    = 256;
	args->encoder_layers = 32;
	args->decoder_layers = 2;
	args->verbose        = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			return -1; /* signal: print help */
		} else if (strcmp(argv[i], "-i") == 0) {
			if (++i >= argc) {
				fprintf(stderr, "error: -i requires a path\n");
				return 1;
			}
			args->input_path = argv[i];
		} else if (strcmp(argv[i], "-o") == 0) {
			if (++i >= argc) {
				fprintf(stderr, "error: -o requires a path\n");
				return 1;
			}
			args->output_path = argv[i];
		} else if (strcmp(argv[i], "-f") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -f requires a format\n");
				return 1;
			}
			args->format = argv[i];
		} else if (strcmp(argv[i], "--image-size") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --image-size requires N\n");
				return 1;
			}
			args->image_size = atoi(argv[i]);
		} else if (strcmp(argv[i], "--encoder-dim") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --encoder-dim requires N\n");
				return 1;
			}
			args->encoder_dim = atoi(argv[i]);
		} else if (strcmp(argv[i], "--decoder-dim") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --decoder-dim requires N\n");
				return 1;
			}
			args->decoder_dim = atoi(argv[i]);
		} else if (strcmp(argv[i], "--encoder-layers") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --encoder-layers requires N\n");
				return 1;
			}
			args->encoder_layers = atoi(argv[i]);
		} else if (strcmp(argv[i], "--decoder-layers") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --decoder-layers requires N\n");
				return 1;
			}
			args->decoder_layers = atoi(argv[i]);
		} else if (strcmp(argv[i], "-v") == 0) {
			args->verbose = 1;
		} else {
			fprintf(stderr, "error: unknown option '%s'\n",
				argv[i]);
			return 1;
		}
	}

	if (!args->input_path) {
		fprintf(stderr, "error: -i <input> is required\n");
		return 1;
	}
	if (!args->output_path) {
		fprintf(stderr, "error: -o <output> is required\n");
		return 1;
	}

	return 0;
}

int main(int argc, char **argv)
{
	struct convert_args args;
	struct weight_reader reader;
	struct sam3_model_config config;
	enum sam3_error err;
	int n;
	int rc;

	rc = parse_args(argc, argv, &args);
	if (rc < 0) {
		print_usage(argv[0]);
		return 0;
	}
	if (rc > 0) {
		fprintf(stderr, "Run '%s -h' for usage.\n", argv[0]);
		return 1;
	}

	if (args.verbose)
		sam3_log_set_level(SAM3_LOG_DEBUG);
	else
		sam3_log_set_level(SAM3_LOG_INFO);

	/* Initialize reader based on format */
	if (strcmp(args.format, "safetensors") == 0) {
		weight_reader_safetensors_init(&reader);
	} else {
		fprintf(stderr, "error: unsupported format '%s' "
			"(only \"safetensors\" is supported)\n",
			args.format);
		return 1;
	}

	/* Open the input file */
	err = reader.ops->open(&reader, args.input_path);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to open '%s': %s\n",
			args.input_path, sam3_error_str(err));
		return 1;
	}

	n = reader.ops->n_tensors(&reader);

	/* Print conversion summary */
	printf("sam3_convert\n");
	printf("  input:          %s\n", args.input_path);
	printf("  output:         %s\n", args.output_path);
	printf("  format:         %s\n", args.format);
	printf("  tensors:        %d\n", n);
	printf("  image_size:     %d\n", args.image_size);
	printf("  encoder_dim:    %d\n", args.encoder_dim);
	printf("  decoder_dim:    %d\n", args.decoder_dim);
	printf("  encoder_layers: %d\n", args.encoder_layers);
	printf("  decoder_layers: %d\n", args.decoder_layers);

	/* Build model config */
	config.image_size       = args.image_size;
	config.encoder_dim      = args.encoder_dim;
	config.decoder_dim      = args.decoder_dim;
	config.n_encoder_layers = args.encoder_layers;
	config.n_decoder_layers = args.decoder_layers;

	/* Write .sam3 output */
	err = sam3_weight_write(args.output_path, &config, &reader);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: conversion failed: %s\n",
			sam3_error_str(err));
		reader.ops->close(&reader);
		return 1;
	}

	reader.ops->close(&reader);
	printf("Done.\n");

	return 0;
}
