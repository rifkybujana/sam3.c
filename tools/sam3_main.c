/*
 * tools/sam3_main.c - SAM3 inference CLI
 *
 * Main command-line tool for running SAM3 segmentation. Takes an image
 * and point/box prompts, runs the full inference pipeline, and writes
 * output masks in binary, PNG, overlay, or cutout formats.
 *
 * Usage: sam3 -m <model> -i <image> -p <x,y,label> [-o <dir>] [options]
 *
 * Key types:  struct main_args, sam3_prompt, sam3_result
 * Depends on: sam3/sam3.h, util/image.h, util/log.h, util/error.h
 * Used by:    end users
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sam3/sam3.h"
#include "util/image.h"
#include "util/log.h"
#include "util/error.h"

/* Suppress warnings in vendored stb header (declarations only) */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wcomma"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-prototypes"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include "util/vendor/stb_image_write.h"

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define MAX_PROMPTS 64

struct main_args {
	const char          *model_path;
	const char          *image_path;
	const char          *output_dir;
	struct sam3_prompt   prompts[MAX_PROMPTS];
	int                  n_prompts;
	int                  write_png;
	int                  write_overlay;
	int                  write_cutout;
	float                threshold;
	int                  profile;
	int                  verbose;
};

/*
 * parse_point - Parse a "x,y,label" string into a point prompt.
 *
 * @str:    Input string in format "x,y,label"
 * @prompt: Output prompt struct
 *
 * Returns 0 on success, 1 on parse error.
 */
static int parse_point(const char *str, struct sam3_prompt *prompt)
{
	float x, y;
	int label;

	if (sscanf(str, "%f,%f,%d", &x, &y, &label) != 3)
		return 1;

	prompt->type = SAM3_PROMPT_POINT;
	prompt->point.x = x;
	prompt->point.y = y;
	prompt->point.label = label;
	return 0;
}

/*
 * parse_box - Parse a "x1,y1,x2,y2" string into a box prompt.
 *
 * @str:    Input string in format "x1,y1,x2,y2"
 * @prompt: Output prompt struct
 *
 * Returns 0 on success, 1 on parse error.
 */
static int parse_box(const char *str, struct sam3_prompt *prompt)
{
	float x1, y1, x2, y2;

	if (sscanf(str, "%f,%f,%f,%f", &x1, &y1, &x2, &y2) != 4)
		return 1;

	prompt->type = SAM3_PROMPT_BOX;
	prompt->box.x1 = x1;
	prompt->box.y1 = y1;
	prompt->box.x2 = x2;
	prompt->box.y2 = y2;
	return 0;
}

static void print_usage(const char *prog)
{
	printf("sam3 inference tool v%s\n\n", sam3_version());
	printf("Usage: %s -m <model> -i <image> [prompts] "
	       "[options]\n\n", prog);
	printf("Required:\n");
	printf("  -m <path>          Model weights file (.sam3)\n");
	printf("  -i <path>          Input image (PNG/JPEG/BMP)\n");
	printf("\nPrompts (at least one required):\n");
	printf("  -t <text>          Text prompt\n");
	printf("  -p x,y,label       Point prompt (repeatable, "
	       "label: 1=fg, 0=bg)\n");
	printf("  -b x1,y1,x2,y2    Box prompt (repeatable)\n");
	printf("\nOutput options:\n");
	printf("  -o <dir>           Output directory "
	       "(default: \".\")\n");
	printf("  --png              Write grayscale PNG masks\n");
	printf("  --overlay          Write color overlay PNGs\n");
	printf("  --cutout           Write cutout PNGs\n");
	printf("  --all              Enable all output formats\n");
	printf("  --threshold <f>    Mask threshold "
	       "(default: 0.5)\n");
	printf("\nOther:\n");
	printf("  --profile          Print profiling report\n");
	printf("  -v                 Verbose logging\n");
	printf("  -h, --help         Show this help\n");
}

/*
 * parse_args - Parse command-line arguments into main_args.
 *
 * Returns -1 for --help (caller should print usage and exit 0),
 * 0 on success, 1 on error (caller should print "Run -h" and exit 1).
 */
static int parse_args(int argc, char **argv, struct main_args *args)
{
	args->model_path    = NULL;
	args->image_path    = NULL;
	args->output_dir    = ".";
	args->n_prompts     = 0;
	args->write_png     = 0;
	args->write_overlay = 0;
	args->write_cutout  = 0;
	args->threshold     = 0.5f;
	args->profile       = 0;
	args->verbose       = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			return -1;
		} else if (strcmp(argv[i], "-m") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -m requires a path\n");
				return 1;
			}
			args->model_path = argv[i];
		} else if (strcmp(argv[i], "-i") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -i requires a path\n");
				return 1;
			}
			args->image_path = argv[i];
		} else if (strcmp(argv[i], "-o") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -o requires a path\n");
				return 1;
			}
			args->output_dir = argv[i];
		} else if (strcmp(argv[i], "-t") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -t requires text\n");
				return 1;
			}
			if (args->n_prompts >= MAX_PROMPTS) {
				fprintf(stderr,
					"error: too many prompts "
					"(max %d)\n", MAX_PROMPTS);
				return 1;
			}
			args->prompts[args->n_prompts].type =
				SAM3_PROMPT_TEXT;
			args->prompts[args->n_prompts].text = argv[i];
			args->n_prompts++;
		} else if (strcmp(argv[i], "-p") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -p requires x,y,label\n");
				return 1;
			}
			if (args->n_prompts >= MAX_PROMPTS) {
				fprintf(stderr,
					"error: too many prompts "
					"(max %d)\n", MAX_PROMPTS);
				return 1;
			}
			if (parse_point(argv[i],
					&args->prompts[args->n_prompts])) {
				fprintf(stderr,
					"error: invalid point '%s' "
					"(expected x,y,label)\n",
					argv[i]);
				return 1;
			}
			args->n_prompts++;
		} else if (strcmp(argv[i], "-b") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -b requires "
					"x1,y1,x2,y2\n");
				return 1;
			}
			if (args->n_prompts >= MAX_PROMPTS) {
				fprintf(stderr,
					"error: too many prompts "
					"(max %d)\n", MAX_PROMPTS);
				return 1;
			}
			if (parse_box(argv[i],
				      &args->prompts[args->n_prompts])) {
				fprintf(stderr,
					"error: invalid box '%s' "
					"(expected x1,y1,x2,y2)\n",
					argv[i]);
				return 1;
			}
			args->n_prompts++;
		} else if (strcmp(argv[i], "--png") == 0) {
			args->write_png = 1;
		} else if (strcmp(argv[i], "--overlay") == 0) {
			args->write_overlay = 1;
		} else if (strcmp(argv[i], "--cutout") == 0) {
			args->write_cutout = 1;
		} else if (strcmp(argv[i], "--all") == 0) {
			args->write_png     = 1;
			args->write_overlay = 1;
			args->write_cutout  = 1;
		} else if (strcmp(argv[i], "--threshold") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --threshold requires "
					"a value\n");
				return 1;
			}
			args->threshold = (float)atof(argv[i]);
		} else if (strcmp(argv[i], "--profile") == 0) {
			args->profile = 1;
		} else if (strcmp(argv[i], "-v") == 0) {
			args->verbose = 1;
		} else {
			fprintf(stderr, "error: unknown option '%s'\n",
				argv[i]);
			return 1;
		}
	}

	if (!args->model_path) {
		fprintf(stderr, "error: -m <model> is required\n");
		return 1;
	}
	if (!args->image_path) {
		fprintf(stderr, "error: -i <image> is required\n");
		return 1;
	}
	if (args->n_prompts == 0) {
		fprintf(stderr,
			"error: at least one prompt (-p or -b) "
			"is required\n");
		return 1;
	}

	return 0;
}

/* --- Mask output helpers --- */

/*
 * write_mask_bin - Write raw float32 mask data to a binary file.
 *
 * @path: Output file path
 * @data: Float array of mask values
 * @n:    Number of floats to write
 *
 * Returns 0 on success, 1 on I/O error.
 */
static int write_mask_bin(const char *path, const float *data, int n)
{
	FILE *f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "error: cannot open '%s' for writing\n",
			path);
		return 1;
	}

	size_t written = fwrite(data, sizeof(float), (size_t)n, f);
	fclose(f);

	if ((int)written != n) {
		fprintf(stderr, "error: short write to '%s'\n", path);
		return 1;
	}
	return 0;
}

/*
 * write_mask_png - Write a thresholded grayscale PNG mask.
 *
 * Pixels above threshold become 255 (white), below become 0 (black).
 *
 * @path:      Output PNG path
 * @data:      Float mask data (H * W)
 * @w:         Mask width
 * @h:         Mask height
 * @threshold: Binarization threshold
 *
 * Returns 0 on success, 1 on error.
 */
static int write_mask_png(const char *path, const float *data,
			  int w, int h, float threshold)
{
	size_t npix = (size_t)w * h;
	uint8_t *gray = malloc(npix);
	if (!gray) {
		fprintf(stderr, "error: out of memory for PNG mask\n");
		return 1;
	}

	for (size_t i = 0; i < npix; i++)
		gray[i] = data[i] >= threshold ? 255 : 0;

	int ok = stbi_write_png(path, w, h, 1, gray, w);
	free(gray);

	if (!ok) {
		fprintf(stderr, "error: failed to write PNG '%s'\n",
			path);
		return 1;
	}
	return 0;
}

/*
 * resize_mask_nn - Nearest-neighbor resize of a float mask.
 *
 * @src:    Source mask data (src_h * src_w)
 * @src_w:  Source width
 * @src_h:  Source height
 * @dst:    Output buffer (dst_h * dst_w floats, caller-allocated)
 * @dst_w:  Target width
 * @dst_h:  Target height
 */
static void resize_mask_nn(const float *src, int src_w, int src_h,
			   float *dst, int dst_w, int dst_h)
{
	for (int y = 0; y < dst_h; y++) {
		int sy = y * src_h / dst_h;
		for (int x = 0; x < dst_w; x++) {
			int sx = x * src_w / dst_w;
			dst[y * dst_w + x] = src[sy * src_w + sx];
		}
	}
}

/*
 * write_overlay - Write a color overlay PNG on the original image.
 *
 * Overlays a semi-transparent blue (30, 144, 255) at 50% alpha on
 * pixels where the mask exceeds the threshold. The mask is resized
 * to match the original image dimensions if needed.
 *
 * @path:      Output PNG path
 * @mask:      Float mask data (mask_h * mask_w)
 * @mask_w:    Mask width
 * @mask_h:    Mask height
 * @img:       Original image (RGB)
 * @threshold: Mask threshold
 *
 * Returns 0 on success, 1 on error.
 */
static int write_overlay(const char *path, const float *mask,
			 int mask_w, int mask_h,
			 const struct sam3_image *img, float threshold)
{
	int w = img->width;
	int h = img->height;
	size_t npix = (size_t)w * h;

	/* Resize mask to image dimensions if needed */
	float *resized = NULL;
	const float *m = mask;
	if (mask_w != w || mask_h != h) {
		resized = malloc(npix * sizeof(float));
		if (!resized) {
			fprintf(stderr,
				"error: out of memory for overlay\n");
			return 1;
		}
		resize_mask_nn(mask, mask_w, mask_h,
			       resized, w, h);
		m = resized;
	}

	/* Build RGB overlay */
	uint8_t *out = malloc(npix * 3);
	if (!out) {
		free(resized);
		fprintf(stderr, "error: out of memory for overlay\n");
		return 1;
	}

	for (size_t i = 0; i < npix; i++) {
		uint8_t r = img->pixels[i * 3 + 0];
		uint8_t g = img->pixels[i * 3 + 1];
		uint8_t b = img->pixels[i * 3 + 2];

		if (m[i] >= threshold) {
			/* Blend with (30, 144, 255) at 50% alpha */
			out[i * 3 + 0] = (uint8_t)((r + 30) / 2);
			out[i * 3 + 1] = (uint8_t)((g + 144) / 2);
			out[i * 3 + 2] = (uint8_t)((b + 255) / 2);
		} else {
			out[i * 3 + 0] = r;
			out[i * 3 + 1] = g;
			out[i * 3 + 2] = b;
		}
	}

	int ok = stbi_write_png(path, w, h, 3, out, w * 3);
	free(out);
	free(resized);

	if (!ok) {
		fprintf(stderr,
			"error: failed to write overlay '%s'\n", path);
		return 1;
	}
	return 0;
}

/*
 * write_cutout - Write an RGBA cutout PNG.
 *
 * Original pixels where mask > threshold are kept; all other pixels
 * become fully transparent.
 *
 * @path:      Output PNG path
 * @mask:      Float mask data (mask_h * mask_w)
 * @mask_w:    Mask width
 * @mask_h:    Mask height
 * @img:       Original image (RGB)
 * @threshold: Mask threshold
 *
 * Returns 0 on success, 1 on error.
 */
static int write_cutout(const char *path, const float *mask,
			int mask_w, int mask_h,
			const struct sam3_image *img, float threshold)
{
	int w = img->width;
	int h = img->height;
	size_t npix = (size_t)w * h;

	/* Resize mask to image dimensions if needed */
	float *resized = NULL;
	const float *m = mask;
	if (mask_w != w || mask_h != h) {
		resized = malloc(npix * sizeof(float));
		if (!resized) {
			fprintf(stderr,
				"error: out of memory for cutout\n");
			return 1;
		}
		resize_mask_nn(mask, mask_w, mask_h,
			       resized, w, h);
		m = resized;
	}

	/* Build RGBA cutout */
	uint8_t *out = malloc(npix * 4);
	if (!out) {
		free(resized);
		fprintf(stderr, "error: out of memory for cutout\n");
		return 1;
	}

	for (size_t i = 0; i < npix; i++) {
		if (m[i] >= threshold) {
			out[i * 4 + 0] = img->pixels[i * 3 + 0];
			out[i * 4 + 1] = img->pixels[i * 3 + 1];
			out[i * 4 + 2] = img->pixels[i * 3 + 2];
			out[i * 4 + 3] = 255;
		} else {
			out[i * 4 + 0] = 0;
			out[i * 4 + 1] = 0;
			out[i * 4 + 2] = 0;
			out[i * 4 + 3] = 0;
		}
	}

	int ok = stbi_write_png(path, w, h, 4, out, w * 4);
	free(out);
	free(resized);

	if (!ok) {
		fprintf(stderr,
			"error: failed to write cutout '%s'\n", path);
		return 1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	struct main_args args;
	struct sam3_image orig_img = {0};
	struct sam3_result result = {0};
	sam3_ctx *ctx = NULL;
	enum sam3_error err;
	int ret = 1;
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

	/* Set log level */
	if (args.verbose)
		sam3_log_set_level(SAM3_LOG_DEBUG);
	else
		sam3_log_set_level(SAM3_LOG_INFO);

	/* Initialize context */
	ctx = sam3_init();
	if (!ctx) {
		fprintf(stderr, "error: failed to initialize sam3\n");
		goto cleanup;
	}

	/* Enable profiling if requested */
	if (args.profile) {
		err = sam3_profile_enable(ctx);
		if (err != SAM3_OK) {
			fprintf(stderr,
				"warning: profiling not available: %s\n",
				sam3_error_str(err));
		}
	}

	/* Load model */
	printf("Loading model: %s\n", args.model_path);
	err = sam3_load_model(ctx, args.model_path);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to load model '%s': %s\n",
			args.model_path, sam3_error_str(err));
		goto cleanup;
	}

	/* Load original image for overlay/cutout */
	if (args.write_overlay || args.write_cutout) {
		err = sam3_image_load(args.image_path, &orig_img);
		if (err != SAM3_OK) {
			fprintf(stderr,
				"error: failed to load image '%s': "
				"%s\n", args.image_path,
				sam3_error_str(err));
			goto cleanup;
		}
	}

	/* Encode image */
	printf("Encoding image: %s\n", args.image_path);
	err = sam3_set_image_file(ctx, args.image_path);
	if (err != SAM3_OK) {
		fprintf(stderr,
			"error: failed to encode image '%s': %s\n",
			args.image_path, sam3_error_str(err));
		goto cleanup;
	}

	/* Run segmentation */
	printf("Segmenting with %d prompt(s)...\n", args.n_prompts);
	err = sam3_segment(ctx, args.prompts, args.n_prompts, &result);
	if (err != SAM3_OK) {
		fprintf(stderr, "error: segmentation failed: %s\n",
			sam3_error_str(err));
		goto cleanup;
	}

	/* Print results summary */
	printf("Results: %d mask(s), %dx%d\n",
	       result.n_masks, result.mask_width, result.mask_height);
	{
		int mask_pixels = result.mask_width * result.mask_height;

		for (int i = 0; i < result.n_masks; i++) {
			if (result.iou_valid)
				printf("  mask %d: IoU = %.4f\n",
				       i, result.iou_scores[i]);
			else
				printf("  mask %d: IoU = N/A "
				       "(no scorer input)\n", i);

			/* Debug: per-mask statistics */
			if (args.verbose) {
				const float *md = result.masks +
						  i * mask_pixels;
				float mn = md[0], mx = md[0], sum = 0.0f;

				for (int j = 0; j < mask_pixels; j++) {
					if (md[j] < mn) mn = md[j];
					if (md[j] > mx) mx = md[j];
					sum += md[j];
				}
				sam3_log_debug("  mask %d stats: "
					       "min=%.4f max=%.4f "
					       "mean=%.4f",
					       i, mn, mx,
					       sum / mask_pixels);
			}
		}
	}

	/* Write outputs */
	{
		int mask_pixels = result.mask_width * result.mask_height;
		char path_buf[1024];

		for (int i = 0; i < result.n_masks; i++) {
			const float *mask_data =
				result.masks + i * mask_pixels;

			/* Always write raw binary */
			snprintf(path_buf, sizeof(path_buf),
				 "%s/mask_%d.bin", args.output_dir, i);
			if (write_mask_bin(path_buf, mask_data,
					   mask_pixels))
				fprintf(stderr,
					"warning: failed to write %s\n",
					path_buf);
			else
				printf("  wrote %s\n", path_buf);

			/* Grayscale PNG */
			if (args.write_png) {
				snprintf(path_buf, sizeof(path_buf),
					 "%s/mask_%d.png",
					 args.output_dir, i);
				if (write_mask_png(path_buf, mask_data,
						   result.mask_width,
						   result.mask_height,
						   args.threshold))
					fprintf(stderr,
						"warning: failed to "
						"write %s\n",
						path_buf);
				else
					printf("  wrote %s\n",
					       path_buf);
			}

			/* Color overlay */
			if (args.write_overlay) {
				snprintf(path_buf, sizeof(path_buf),
					 "%s/overlay_%d.png",
					 args.output_dir, i);
				if (write_overlay(path_buf, mask_data,
						  result.mask_width,
						  result.mask_height,
						  &orig_img,
						  args.threshold))
					fprintf(stderr,
						"warning: failed to "
						"write %s\n",
						path_buf);
				else
					printf("  wrote %s\n",
					       path_buf);
			}

			/* RGBA cutout */
			if (args.write_cutout) {
				snprintf(path_buf, sizeof(path_buf),
					 "%s/cutout_%d.png",
					 args.output_dir, i);
				if (write_cutout(path_buf, mask_data,
						 result.mask_width,
						 result.mask_height,
						 &orig_img,
						 args.threshold))
					fprintf(stderr,
						"warning: failed to "
						"write %s\n",
						path_buf);
				else
					printf("  wrote %s\n",
					       path_buf);
			}
		}
	}

	/* Profiling report */
	if (args.profile)
		sam3_profile_report(ctx);

	ret = 0;

cleanup:
	sam3_result_free(&result);
	sam3_image_free(&orig_img);
	sam3_free(ctx);
	return ret;
}
