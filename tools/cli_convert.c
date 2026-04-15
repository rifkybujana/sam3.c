/*
 * tools/cli_convert.c - SAM3 convert subcommand
 *
 * Converts model weights from SafeTensors to .sam3 format. Refactored
 * from the standalone sam3_convert tool into the unified CLI. Reads input
 * through the weight_reader vtable and writes .sam3 output via
 * sam3_weight_write.
 *
 * Key types:  convert_args, quant_reader_state, weight_reader_ops
 * Depends on: cli_common.h, cli_convert.h, core/weight.h, core/quant.h,
 *             core/half.h, util/log.h, util/error.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <libgen.h>

#include "sam3/sam3.h"
#include "sam3/sam3_types.h"
#include "core/weight.h"
#include "core/quant.h"
#include "core/half.h"
#include "util/log.h"
#include "util/error.h"
#include "weight_rename.h"
#include "weight_conv_perm.h"
#include "cli_common.h"
#include "cli_convert.h"

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
	       "(default: 1008, 512 for efficientvit)\n");
	printf("  --encoder-dim <N>    Encoder dimension "
	       "(default: 1280)\n");
	printf("  --decoder-dim <N>    Decoder dimension "
	       "(default: 256)\n");
	printf("  --encoder-layers <N> Encoder layer count "
	       "(default: 32)\n");
	printf("  --decoder-layers <N> Decoder layer count "
	       "(default: 2)\n");
	printf("  --backbone <type>    Vision backbone: "
	       "\"hiera\" (default), \"efficientvit\", "
	       "\"tinyvit\"\n");
	printf("  --quantize <type>    Quantize weights: "
	       "\"q8_0\" (default: none)\n");
	printf("  -v                   Verbose output "
	       "(set log level to DEBUG)\n");
	printf("  -q                   Quiet mode "
	       "(suppress progress)\n");
	printf("  --version            Print version and exit\n");
	printf("  -h, --help           Show this help\n");
}

struct convert_args {
	const char *input_path;
	const char *output_path;
	const char *format;
	const char *backbone;   /* "hiera", "efficientvit", or "tinyvit" */
	const char *quantize;   /* NULL or "q8_0" */
	int         image_size;
	int         encoder_dim;
	int         decoder_dim;
	int         encoder_layers;
	int         decoder_layers;
	int         backbone_type;
	int         verbose;
	int         quiet;
};

/*
 * parse_args - Parse command-line arguments into convert_args.
 *
 * Returns -1 for --help/--version (caller should exit 0),
 * 0 on success, 1 on error (caller should print "Run -h" and exit 1).
 */
static int parse_args(int argc, char **argv, struct convert_args *args)
{
	args->input_path     = NULL;
	args->output_path    = NULL;
	args->format         = "safetensors";
	args->backbone       = "hiera";
	args->quantize       = NULL;
	args->image_size     = -1; /* -1 = use backbone default */
	args->encoder_dim    = -1; /* -1 = use backbone default */
	args->decoder_dim    = 256;
	args->encoder_layers = -1; /* -1 = use backbone default */
	args->decoder_layers = 2;
	args->backbone_type  = SAM3_BACKBONE_HIERA;
	args->verbose        = 0;
	args->quiet          = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			return -1; /* signal: print help */
		} else if (strcmp(argv[i], "--version") == 0) {
			printf("sam3 %s\n", sam3_version());
			return -1;
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
		} else if (strcmp(argv[i], "--backbone") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --backbone requires a "
					"type\n");
				return 1;
			}
			args->backbone = argv[i];
		} else if (strcmp(argv[i], "--quantize") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --quantize requires a type\n");
				return 1;
			}
			args->quantize = argv[i];
		} else if (strcmp(argv[i], "-v") == 0) {
			args->verbose = 1;
		} else if (strcmp(argv[i], "-q") == 0) {
			args->quiet = 1;
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

	/* Resolve backbone type and apply defaults */
	if (strcmp(args->backbone, "hiera") == 0) {
		args->backbone_type = SAM3_BACKBONE_HIERA;
		if (args->encoder_dim < 0)
			args->encoder_dim = 1280;
		if (args->encoder_layers < 0)
			args->encoder_layers = 32;
	} else if (strcmp(args->backbone, "efficientvit") == 0) {
		args->backbone_type = SAM3_BACKBONE_EFFICIENTVIT;
		if (args->encoder_dim < 0)
			args->encoder_dim = 384;
		if (args->encoder_layers < 0)
			args->encoder_layers = 20;
	} else if (strcmp(args->backbone, "tinyvit") == 0) {
		args->backbone_type = SAM3_BACKBONE_TINYVIT;
		if (args->encoder_dim < 0)
			args->encoder_dim = 576;
		if (args->encoder_layers < 0)
			args->encoder_layers = 12;
	} else {
		fprintf(stderr,
			"error: unsupported backbone '%s' "
			"(use \"hiera\", \"efficientvit\", "
			"or \"tinyvit\")\n",
			args->backbone);
		return 1;
	}

	/*
	 * Apply backbone-aware image_size default when the user did
	 * not specify --image-size. EfficientViT hardcodes 512;
	 * Hiera and TinyViT use 1008.
	 */
	if (args->image_size < 0) {
		if (args->backbone_type == SAM3_BACKBONE_EFFICIENTVIT)
			args->image_size = 512;
		else
			args->image_size = 1008;
	}

	return 0;
}

/* ── Quantizing reader wrapper ─────────────────────────────────────── */

struct quant_reader_state {
	struct weight_reader *inner;
	int                   nelems_threshold;
};

static enum sam3_error qr_open(struct weight_reader *r, const char *path)
{
	(void)r; (void)path;
	return SAM3_OK; /* inner already opened */
}

static int qr_n_tensors(struct weight_reader *r)
{
	struct quant_reader_state *s = r->impl;
	return s->inner->ops->n_tensors(s->inner);
}

static enum sam3_error qr_get_tensor_info(struct weight_reader *r, int idx,
					  struct weight_tensor_info *info)
{
	struct quant_reader_state *s = r->impl;
	enum sam3_error err;

	err = s->inner->ops->get_tensor_info(s->inner, idx, info);
	if (err != SAM3_OK)
		return err;

	/* Compute nelems */
	int nelems = 1;
	for (int d = 0; d < info->n_dims; d++)
		nelems *= info->dims[d];

	/* Quantize large float tensors */
	if (nelems >= s->nelems_threshold &&
	    (info->dtype == SAM3_DTYPE_F32 ||
	     info->dtype == SAM3_DTYPE_F16 ||
	     info->dtype == SAM3_DTYPE_BF16)) {
		info->dtype  = SAM3_DTYPE_Q8_0;
		info->nbytes = sam3_q8_nbytes(nelems);
	}

	return SAM3_OK;
}

static enum sam3_error qr_read_tensor_data(struct weight_reader *r, int idx,
					   void *dst, size_t dst_size)
{
	struct quant_reader_state *s = r->impl;
	struct weight_tensor_info orig_info;
	enum sam3_error err;

	/* Get original (pre-quantize) info */
	err = s->inner->ops->get_tensor_info(s->inner, idx, &orig_info);
	if (err != SAM3_OK)
		return err;

	int nelems = 1;
	for (int d = 0; d < orig_info.n_dims; d++)
		nelems *= orig_info.dims[d];

	/* Check if this tensor should be quantized */
	int should_quantize = (nelems >= s->nelems_threshold &&
			       (orig_info.dtype == SAM3_DTYPE_F32 ||
				orig_info.dtype == SAM3_DTYPE_F16 ||
				orig_info.dtype == SAM3_DTYPE_BF16));

	if (!should_quantize) {
		return s->inner->ops->read_tensor_data(s->inner, idx,
						       dst, dst_size);
	}

	/* Read original data */
	void *orig_data = malloc(orig_info.nbytes);
	if (!orig_data)
		return SAM3_ENOMEM;

	err = s->inner->ops->read_tensor_data(s->inner, idx,
					       orig_data, orig_info.nbytes);
	if (err != SAM3_OK) {
		free(orig_data);
		return err;
	}

	/* Convert to f32 if needed, then quantize */
	float *f32_data;
	int need_free_f32 = 0;

	if (orig_info.dtype == SAM3_DTYPE_F32) {
		f32_data = (float *)orig_data;
	} else {
		f32_data = malloc((size_t)nelems * sizeof(float));
		if (!f32_data) {
			free(orig_data);
			return SAM3_ENOMEM;
		}
		need_free_f32 = 1;

		if (orig_info.dtype == SAM3_DTYPE_F16) {
			const uint16_t *fp16 = (const uint16_t *)orig_data;
			for (int i = 0; i < nelems; i++)
				f32_data[i] = fp16_to_f32(fp16[i]);
		} else { /* BF16 */
			const uint16_t *bf16 = (const uint16_t *)orig_data;
			for (int i = 0; i < nelems; i++)
				f32_data[i] = bf16_to_f32(bf16[i]);
		}
	}

	/* Quantize to Q8_0 */
	sam3_q8_quantize(f32_data, (struct sam3_q8_block *)dst, nelems);

	if (need_free_f32)
		free(f32_data);
	free(orig_data);

	return SAM3_OK;
}

static void qr_close(struct weight_reader *r)
{
	(void)r; /* inner closed separately */
}

static const struct weight_reader_ops quant_reader_ops = {
	.open             = qr_open,
	.n_tensors        = qr_n_tensors,
	.get_tensor_info  = qr_get_tensor_info,
	.read_tensor_data = qr_read_tensor_data,
	.close            = qr_close,
};

int cli_convert(int argc, char **argv)
{
	struct convert_args args;
	struct weight_reader reader;
	struct sam3_model_config config;
	enum sam3_error err;
	int n;
	int rc;

	rc = parse_args(argc, argv, &args);
	if (rc < 0) {
		/* --help prints usage, --version prints version */
		if (argc >= 2 &&
		    strcmp(argv[argc - 1], "--version") != 0)
			print_usage(argv[0]);
		return SAM3_EXIT_OK;
	}
	if (rc > 0) {
		fprintf(stderr, "Run '%s -h' for usage.\n", argv[0]);
		return SAM3_EXIT_USAGE;
	}

	/* Validate paths before expensive operations */
	if (access(args.input_path, R_OK) != 0) {
		fprintf(stderr, "error: input file not found: '%s'\n",
			args.input_path);
		return SAM3_EXIT_IO;
	}
	{
		char *tmp = strdup(args.output_path);
		if (tmp) {
			const char *dir = dirname(tmp);
			struct stat dir_st;
			if (stat(dir, &dir_st) != 0 ||
			    !S_ISDIR(dir_st.st_mode)) {
				fprintf(stderr,
					"error: output directory not "
					"found: '%s'\n", dir);
				free(tmp);
				return SAM3_EXIT_IO;
			}
			if (access(dir, W_OK) != 0) {
				fprintf(stderr,
					"error: output directory not "
					"writable: '%s'\n", dir);
				free(tmp);
				return SAM3_EXIT_IO;
			}
			free(tmp);
		}
	}

	/* Set log level */
	if (args.quiet)
		sam3_log_set_level(SAM3_LOG_ERROR);
	else if (args.verbose)
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
		return SAM3_EXIT_USAGE;
	}

	/* Open the input file */
	err = reader.ops->open(&reader, args.input_path);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to open '%s': %s\n",
			args.input_path, sam3_error_str(err));
		return sam3_error_to_exit(err);
	}

	/* Wrap with rename reader (remaps Python names to C names) */
	struct weight_reader rename_reader;
	err = weight_reader_rename_init(&rename_reader, &reader);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to build rename table: %s\n",
			sam3_error_str(err));
		reader.ops->close(&reader);
		return sam3_error_to_exit(err);
	}

	/*
	 * Wrap with conv_perm reader: detects conv2d / conv_transpose2d
	 * weights by name and permutes them to OHWI on the fly, so the
	 * .sam3 output ships with weights already in the layout the
	 * NHWC backend kernels expect.
	 */
	struct weight_reader conv_perm_reader;
	err = weight_reader_conv_perm_init(&conv_perm_reader, &rename_reader);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to init conv_perm reader: "
			"%s\n", sam3_error_str(err));
		rename_reader.ops->close(&rename_reader);
		reader.ops->close(&reader);
		return sam3_error_to_exit(err);
	}

	n = conv_perm_reader.ops->n_tensors(&conv_perm_reader);

	/* Print conversion summary */
	if (!args.quiet) {
		cli_progress("sam3_convert\n");
		cli_progress("  input:          %s\n", args.input_path);
		cli_progress("  output:         %s\n", args.output_path);
		cli_progress("  format:         %s\n", args.format);
		cli_progress("  backbone:       %s\n", args.backbone);
		cli_progress("  tensors:        %d (after rename/split)\n", n);
		cli_progress("  image_size:     %d\n", args.image_size);
		cli_progress("  encoder_dim:    %d\n", args.encoder_dim);
		cli_progress("  decoder_dim:    %d\n", args.decoder_dim);
		cli_progress("  encoder_layers: %d\n", args.encoder_layers);
		cli_progress("  decoder_layers: %d\n", args.decoder_layers);
	}

	/* Build model config */
	config.image_size       = args.image_size;
	config.encoder_dim      = args.encoder_dim;
	config.decoder_dim      = args.decoder_dim;
	config.n_encoder_layers = args.encoder_layers;
	config.n_decoder_layers = args.decoder_layers;
	config.backbone_type    = args.backbone_type;

	/* Set up quantizing wrapper if requested */
	struct weight_reader *write_reader = &conv_perm_reader;
	struct quant_reader_state qstate;
	struct weight_reader quant_reader;

	if (args.quantize) {
		if (strcmp(args.quantize, "q8_0") != 0) {
			fprintf(stderr,
				"error: unsupported quantize type '%s' "
				"(only \"q8_0\" is supported)\n",
				args.quantize);
			conv_perm_reader.ops->close(&conv_perm_reader);
			rename_reader.ops->close(&rename_reader);
			reader.ops->close(&reader);
			return SAM3_EXIT_USAGE;
		}

		qstate.inner = &conv_perm_reader;
		qstate.nelems_threshold = 1024;
		quant_reader.ops = &quant_reader_ops;
		quant_reader.impl = &qstate;
		write_reader = &quant_reader;

		if (!args.quiet)
			cli_progress("  quantize:       %s\n", args.quantize);
	}

	/* Write .sam3 output */
	err = sam3_weight_write(args.output_path, &config, write_reader);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: conversion failed: %s\n",
			sam3_error_str(err));
		conv_perm_reader.ops->close(&conv_perm_reader);
		rename_reader.ops->close(&rename_reader);
		reader.ops->close(&reader);
		return sam3_error_to_exit(err);
	}

	conv_perm_reader.ops->close(&conv_perm_reader);
	rename_reader.ops->close(&rename_reader);
	reader.ops->close(&reader);

	if (!args.quiet)
		cli_progress("Done.\n");

	return SAM3_EXIT_OK;
}
