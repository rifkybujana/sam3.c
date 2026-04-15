/*
 * tools/cli_segment.c - SAM3 segment subcommand
 *
 * Runs segmentation inference with point/box/text prompts. Supports
 * stdin image input (-i -), stdout mask output (-o -), JSON metadata
 * (--json), stdout PNG (--stdout-png), and quiet mode (-q).
 *
 * Key types:  segment_args
 * Depends on: cli_common.h, cli_segment.h, sam3/sam3.h, util/image.h,
 *             util/error.h, util/log.h
 * Used by:    tools/sam3_cli.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <pthread.h>
#include <sys/qos.h>

#include "cli_common.h"
#include "cli_segment.h"

#include "sam3/sam3.h"
#include "sam3/internal/mask_nms.h"
#include "sam3/internal/mask_postprocess.h"
#include "sam3/internal/mask_boxes.h"
#include "sam3/internal/mask_resize.h"
#include "sam3/internal/tensor_dump.h"
#include "core/tensor.h"
#include "util/image.h"
#include "util/log.h"
#include "util/error.h"
#include "util/time.h"

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

struct segment_args {
	const char          *model_path;
	const char          *image_path;
	const char          *output_dir;
	struct sam3_prompt   prompts[MAX_PROMPTS];
	int                  n_prompts;
	int                  write_png;
	int                  write_overlay;
	int                  write_cutout;
	float                threshold;
	int                  nms_enabled;
	float                nms_prob_thresh;
	float                nms_iou_thresh;
	float                min_mask_quality;
	int                  profile;
	int                  verbose;
	int                  smooth;
	const char          *dump_dir;
	int                  quiet;
	int                  json;
	int                  stdout_png;
	int                  use_stdin;
	int                  use_stdout;
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
	printf("sam3 segment - inference tool v%s\n\n",
	       sam3_version());
	printf("Usage: %s -m <model> -i <image> [prompts] "
	       "[options]\n\n", prog);
	printf("Required:\n");
	printf("  -m <path>          Model weights file (.sam3)\n");
	printf("  -i <path>          Input image (PNG/JPEG/BMP, "
	       "or \"-\" for stdin)\n");
	printf("\nPrompts (at least one required):\n");
	printf("  -t <text>          Text prompt\n");
	printf("  -p x,y,label       Point prompt (repeatable, "
	       "label: 1=fg, 0=bg)\n");
	printf("  -b x1,y1,x2,y2    Box prompt (repeatable)\n");
	printf("\nOutput options:\n");
	printf("  -o <dir>           Output directory "
	       "(default: \".\", or \"-\" for stdout)\n");
	printf("  --png              Write grayscale PNG masks\n");
	printf("  --overlay          Write color overlay PNGs\n");
	printf("  --cutout           Write cutout PNGs\n");
	printf("  --all              Enable all output formats\n");
	printf("  --smooth           Apply morphological cleanup "
	       "to masks\n");
	printf("  --threshold <f>    Mask threshold "
	       "(default: 0.0)\n");
	printf("  --no-nms           Disable NMS post-processing\n");
	printf("  --nms-prob-thresh <f>  Score prefilter "
	       "(default: 0.5)\n");
	printf("  --nms-iou-thresh <f>   Mask IoU NMS threshold "
	       "(default: 0.5)\n");
	printf("  --min-mask-quality <f>  Min confident pixel fraction "
	       "(default: 0.0)\n");
	printf("  --stdout-png       Write best mask PNG to stdout\n");
	printf("\nMetadata:\n");
	printf("  --json             Output result metadata as JSON\n");
	printf("  -q                 Quiet mode (suppress progress)\n");
	printf("\nDebug:\n");
	printf("  --dump-tensors <dir>   Dump intermediate tensors "
	       "for validation\n");
	printf("\nOther:\n");
	printf("  --profile          Print profiling report\n");
	printf("  -v                 Verbose logging\n");
	printf("  --version          Print version and exit\n");
	printf("  -h, --help         Show this help\n");
}

/*
 * parse_args - Parse command-line arguments into segment_args.
 *
 * Returns -1 for --help/--version (caller should exit 0),
 * 0 on success, 1 on error (caller should print "Run -h" and exit 1).
 */
static int parse_args(int argc, char **argv, struct segment_args *args)
{
	args->model_path      = NULL;
	args->image_path      = NULL;
	args->output_dir      = ".";
	args->n_prompts       = 0;
	args->write_png       = 0;
	args->write_overlay   = 0;
	args->write_cutout    = 0;
	args->threshold       = 0.0f;
	args->nms_enabled     = 1;
	args->nms_prob_thresh = 0.5f;
	args->nms_iou_thresh  = 0.5f;
	args->min_mask_quality = 0.0f;
	args->profile         = 0;
	args->verbose         = 0;
	args->smooth          = 0;
	args->dump_dir        = NULL;
	args->quiet           = 0;
	args->json            = 0;
	args->stdout_png      = 0;
	args->use_stdin       = 0;
	args->use_stdout      = 0;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			return -1;
		} else if (strcmp(argv[i], "--version") == 0) {
			printf("sam3 %s\n", sam3_version());
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
			if (strcmp(argv[i], "-") == 0)
				args->use_stdin = 1;
		} else if (strcmp(argv[i], "-o") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: -o requires a path\n");
				return 1;
			}
			args->output_dir = argv[i];
			if (strcmp(argv[i], "-") == 0)
				args->use_stdout = 1;
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
		} else if (strcmp(argv[i], "--no-nms") == 0) {
			args->nms_enabled = 0;
		} else if (strcmp(argv[i], "--nms-prob-thresh") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --nms-prob-thresh "
					"requires a value\n");
				return 1;
			}
			args->nms_prob_thresh = (float)atof(argv[i]);
		} else if (strcmp(argv[i], "--nms-iou-thresh") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --nms-iou-thresh "
					"requires a value\n");
				return 1;
			}
			args->nms_iou_thresh = (float)atof(argv[i]);
		} else if (strcmp(argv[i], "--min-mask-quality") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --min-mask-quality "
					"requires a value\n");
				return 1;
			}
			args->min_mask_quality = strtof(argv[i], NULL);
		} else if (strcmp(argv[i], "--smooth") == 0) {
			args->smooth = 1;
		} else if (strcmp(argv[i], "--dump-tensors") == 0) {
			if (++i >= argc) {
				fprintf(stderr,
					"error: --dump-tensors "
					"requires a directory\n");
				return 1;
			}
			args->dump_dir = argv[i];
		} else if (strcmp(argv[i], "--profile") == 0) {
			args->profile = 1;
		} else if (strcmp(argv[i], "-v") == 0) {
			args->verbose = 1;
		} else if (strcmp(argv[i], "-q") == 0) {
			args->quiet = 1;
		} else if (strcmp(argv[i], "--json") == 0) {
			args->json = 1;
		} else if (strcmp(argv[i], "--stdout-png") == 0) {
			args->stdout_png = 1;
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
 * draw_box_rgb - Draw a bounding box rectangle on an RGB buffer.
 *
 * Draws a 2-pixel thick rectangle at the given xyxy coordinates.
 * Coordinates are clamped to image bounds.
 *
 * @buf:  RGB pixel buffer (w * h * 3)
 * @w:    Image width
 * @h:    Image height
 * @x0:   Left edge (inclusive)
 * @y0:   Top edge (inclusive)
 * @x1:   Right edge (exclusive)
 * @y1:   Bottom edge (exclusive)
 * @r, @g, @b: Box color
 */
static void draw_box_rgb(uint8_t *buf, int w, int h,
			 int x0, int y0, int x1, int y1,
			 uint8_t r, uint8_t g, uint8_t b)
{
	/* Clamp to image bounds */
	if (x0 < 0) x0 = 0;
	if (y0 < 0) y0 = 0;
	if (x1 > w) x1 = w;
	if (y1 > h) y1 = h;
	if (x0 >= x1 || y0 >= y1)
		return;

	/* 2-pixel thick lines */
	for (int t = 0; t < 2; t++) {
		/* Top edge */
		if (y0 + t < y1) {
			for (int x = x0; x < x1; x++) {
				size_t idx = ((size_t)(y0 + t) * w + x) * 3;
				buf[idx + 0] = r;
				buf[idx + 1] = g;
				buf[idx + 2] = b;
			}
		}
		/* Bottom edge */
		if (y1 - 1 - t >= y0) {
			for (int x = x0; x < x1; x++) {
				size_t idx = ((size_t)(y1 - 1 - t) * w + x) * 3;
				buf[idx + 0] = r;
				buf[idx + 1] = g;
				buf[idx + 2] = b;
			}
		}
		/* Left edge */
		if (x0 + t < x1) {
			for (int y = y0; y < y1; y++) {
				size_t idx = ((size_t)y * w + (x0 + t)) * 3;
				buf[idx + 0] = r;
				buf[idx + 1] = g;
				buf[idx + 2] = b;
			}
		}
		/* Right edge */
		if (x1 - 1 - t >= x0) {
			for (int y = y0; y < y1; y++) {
				size_t idx = ((size_t)y * w + (x1 - 1 - t)) * 3;
				buf[idx + 0] = r;
				buf[idx + 1] = g;
				buf[idx + 2] = b;
			}
		}
	}
}

/*
 * write_overlay - Write a color overlay PNG on the original image.
 *
 * Overlays a semi-transparent blue (30, 144, 255) at 50% alpha on
 * pixels where the mask exceeds the threshold. Draws a bounding box
 * around the mask region. The mask is resized to match the original
 * image dimensions if needed.
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
		sam3_mask_resize_bilinear(mask, mask_w, mask_h,
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
			int x = (int)(i % (size_t)w);
			int y = (int)(i / (size_t)w);
			/* Edge: any 4-neighbor outside mask */
			int edge = (x == 0 || m[i - 1] < threshold ||
				    x == w - 1 || m[i + 1] < threshold ||
				    y == 0 || m[i - w] < threshold ||
				    y == h - 1 || m[i + w] < threshold);
			if (edge) {
				/* White outline for visibility */
				out[i * 3 + 0] = 255;
				out[i * 3 + 1] = 255;
				out[i * 3 + 2] = 255;
			} else {
				/* Blend with dodger blue at 60% */
				out[i * 3 + 0] = (uint8_t)(r * 0.4f + 18.f);
				out[i * 3 + 1] = (uint8_t)(g * 0.4f + 86.4f);
				out[i * 3 + 2] = (uint8_t)(b * 0.4f + 153.f);
			}
		} else {
			out[i * 3 + 0] = r;
			out[i * 3 + 1] = g;
			out[i * 3 + 2] = b;
		}
	}

	/* Draw bounding box around mask region */
	float box[4];
	if (sam3_masks_to_boxes(m, 1, h, w, box) == 0 &&
	    (box[2] > box[0] || box[3] > box[1])) {
		draw_box_rgb(out, w, h,
			     (int)box[0], (int)box[1],
			     (int)box[2], (int)box[3],
			     30, 144, 255);
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
 * write_overlay_merged - Write a single overlay with all masks in distinct colors.
 *
 * Each mask is blended at 40% opacity with a cycling 6-color palette.
 * White edges are drawn where any mask boundary exists.
 *
 * @path:      Output PNG path
 * @masks:     Float mask data (n_masks * mask_h * mask_w, contiguous)
 * @n_masks:   Number of masks
 * @mask_w:    Mask width
 * @mask_h:    Mask height
 * @img:       Original image (RGB)
 * @threshold: Mask threshold
 *
 * Returns 0 on success, 1 on error.
 */
static int write_overlay_merged(const char *path,
				const float *masks, int n_masks,
				int mask_w, int mask_h,
				const struct sam3_image *img,
				float threshold)
{
	static const uint8_t palette[][3] = {
		{30, 144, 255},   /* dodger blue */
		{255, 100, 80},   /* coral red */
		{0, 200, 100},    /* emerald green */
		{255, 200, 0},    /* gold */
		{200, 80, 255},   /* violet */
		{0, 210, 210},    /* cyan */
	};
	static const int n_colors = 6;

	int w = img->width;
	int h = img->height;
	size_t npix = (size_t)w * h;
	size_t mask_npix = (size_t)mask_w * mask_h;

	/* Resize all masks to image dimensions */
	float *resized_all = NULL;
	const float *m_base;
	int m_w, m_h;

	if (mask_w != w || mask_h != h) {
		resized_all = malloc(npix * (size_t)n_masks *
				     sizeof(float));
		if (!resized_all) {
			fprintf(stderr,
				"error: out of memory for "
				"merged overlay\n");
			return 1;
		}
		for (int k = 0; k < n_masks; k++) {
			sam3_mask_resize_bilinear(
				masks + (size_t)k * mask_npix,
				mask_w, mask_h,
				resized_all + (size_t)k * npix,
				w, h);
		}
		m_base = resized_all;
		m_w = w;
		m_h = h;
	} else {
		m_base = masks;
		m_w = mask_w;
		m_h = mask_h;
	}

	size_t m_npix = (size_t)m_w * m_h;

	uint8_t *out = malloc(npix * 3);
	if (!out) {
		free(resized_all);
		fprintf(stderr,
			"error: out of memory for merged overlay\n");
		return 1;
	}

	for (size_t i = 0; i < npix; i++) {
		float r = (float)img->pixels[i * 3 + 0];
		float g = (float)img->pixels[i * 3 + 1];
		float b = (float)img->pixels[i * 3 + 2];

		int x = (int)(i % (size_t)m_w);
		int y = (int)(i / (size_t)m_w);
		int any_edge = 0;

		/* Blend each active mask */
		for (int k = 0; k < n_masks; k++) {
			const float *m = m_base + (size_t)k * m_npix;
			if (m[i] < threshold)
				continue;

			/* Edge detection for this mask */
			int edge = (x == 0 || m[i - 1] < threshold ||
				    x == m_w - 1 ||
				    m[i + 1] < threshold ||
				    y == 0 ||
				    m[i - m_w] < threshold ||
				    y == m_h - 1 ||
				    m[i + m_w] < threshold);
			if (edge)
				any_edge = 1;

			/* 40% blend with palette color */
			int ci = k % n_colors;
			r = r * 0.6f + (float)palette[ci][0] * 0.4f;
			g = g * 0.6f + (float)palette[ci][1] * 0.4f;
			b = b * 0.6f + (float)palette[ci][2] * 0.4f;
		}

		if (any_edge) {
			out[i * 3 + 0] = 255;
			out[i * 3 + 1] = 255;
			out[i * 3 + 2] = 255;
		} else {
			out[i * 3 + 0] = (uint8_t)(r > 255.f ? 255 :
						    (int)r);
			out[i * 3 + 1] = (uint8_t)(g > 255.f ? 255 :
						    (int)g);
			out[i * 3 + 2] = (uint8_t)(b > 255.f ? 255 :
						    (int)b);
		}
	}

	/* Draw bounding boxes for each mask in matching palette color */
	float boxes[4 * 64];
	float *box_buf = (n_masks <= 64)
		? boxes
		: malloc((size_t)n_masks * 4 * sizeof(float));
	if (box_buf &&
	    sam3_masks_to_boxes(m_base, n_masks, m_h, m_w, box_buf) == 0) {
		for (int k = 0; k < n_masks; k++) {
			float *bk = box_buf + k * 4;
			if (bk[2] <= bk[0] && bk[3] <= bk[1])
				continue;
			int ci = k % n_colors;
			draw_box_rgb(out, w, h,
				     (int)bk[0], (int)bk[1],
				     (int)bk[2], (int)bk[3],
				     palette[ci][0], palette[ci][1],
				     palette[ci][2]);
		}
	}
	if (box_buf != boxes)
		free(box_buf);

	int ok = stbi_write_png(path, w, h, 3, out, w * 3);
	free(out);
	free(resized_all);

	if (!ok) {
		fprintf(stderr,
			"error: failed to write merged overlay "
			"'%s'\n", path);
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
		sam3_mask_resize_bilinear(mask, mask_w, mask_h,
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

/* Callback for stbi_write_png_to_func: writes PNG bytes to stdout. */
static void stdout_write_func(void *context, void *data, int size)
{
	(void)context;
	fwrite(data, 1, (size_t)size, stdout);
}

int cli_segment(int argc, char **argv)
{
	struct segment_args args;
	struct sam3_image orig_img = {0};
	struct sam3_image stdin_img = {0};
	struct sam3_result result = {0};
	sam3_ctx *ctx = NULL;
	uint8_t *stdin_buf = NULL;
	size_t stdin_size = 0;
	enum sam3_error err;
	int ret = SAM3_EXIT_RUNTIME;
	int rc;

	rc = parse_args(argc, argv, &args);
	if (rc < 0) {
		/* --help prints usage, --version prints version */
		if (rc == -1 && argc >= 2 &&
		    strcmp(argv[argc - 1], "--version") != 0)
			print_usage(argv[0]);
		return SAM3_EXIT_OK;
	}
	if (rc > 0) {
		fprintf(stderr, "Run '%s -h' for usage.\n", argv[0]);
		return SAM3_EXIT_USAGE;
	}

	/* Validate paths before expensive operations */
	if (!args.use_stdin && access(args.image_path, R_OK) != 0) {
		fprintf(stderr, "error: image not found: '%s'\n",
			args.image_path);
		return SAM3_EXIT_IO;
	}
	if (access(args.model_path, R_OK) != 0) {
		fprintf(stderr, "error: model not found: '%s'\n",
			args.model_path);
		return SAM3_EXIT_IO;
	}
	if (!args.use_stdout) {
		struct stat dir_st;
		if (stat(args.output_dir, &dir_st) != 0 ||
		    !S_ISDIR(dir_st.st_mode)) {
			fprintf(stderr,
				"error: output directory not found: "
				"'%s'\n", args.output_dir);
			return SAM3_EXIT_IO;
		}
		if (access(args.output_dir, W_OK) != 0) {
			fprintf(stderr,
				"error: output directory not "
				"writable: '%s'\n", args.output_dir);
			return SAM3_EXIT_IO;
		}
	}

	/*
	 * Bias the macOS scheduler toward P-cores so the kernel-launch
	 * dispatch loop and CPU-side tensor wrappers don't get migrated
	 * onto E-cores between Metal command buffer submissions. Cuts
	 * P-core/E-core migration jitter on Apple Silicon. Best-effort:
	 * a failure here is not fatal.
	 */
	(void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

	/* Set log level */
	if (args.quiet)
		sam3_log_set_level(SAM3_LOG_ERROR);
	else if (args.verbose)
		sam3_log_set_level(SAM3_LOG_DEBUG);
	else
		sam3_log_set_level(SAM3_LOG_INFO);

	/* Read stdin once if -i - is set */
	if (args.use_stdin) {
		if (cli_read_stdin(&stdin_buf, &stdin_size)) {
			fprintf(stderr,
				"error: failed to read image "
				"from stdin\n");
			goto cleanup;
		}
	}

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
	if (!args.quiet)
		cli_progress("Loading model: %s\n", args.model_path);
	uint64_t t0 = sam3_time_ns();
	err = sam3_load_model(ctx, args.model_path);
	uint64_t t1 = sam3_time_ns();
	if (err != SAM3_OK) {
		fprintf(stderr, "error: failed to load model '%s': %s\n",
			args.model_path, sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}
	if (!args.quiet)
		cli_progress("  model loaded in %.1fms\n",
			     (double)(t1 - t0) / 1e6);

	/* Load original image for overlay/cutout */
	if (args.write_overlay || args.write_cutout) {
		if (args.use_stdin) {
			err = sam3_image_load_memory(stdin_buf,
						    stdin_size,
						    &orig_img);
			if (err != SAM3_OK) {
				fprintf(stderr,
					"error: failed to decode "
					"stdin image: %s\n",
					sam3_error_str(err));
				ret = (int)sam3_error_to_exit(err);
				goto cleanup;
			}
		} else {
			err = sam3_image_load(args.image_path, &orig_img);
			if (err != SAM3_OK) {
				fprintf(stderr,
					"error: failed to load image "
					"'%s': %s\n",
					args.image_path,
					sam3_error_str(err));
				ret = (int)sam3_error_to_exit(err);
				goto cleanup;
			}
		}
	}

	/*
	 * If a text prompt is present, kick off async text encoding
	 * BEFORE the image encoder so the two run concurrently
	 * (image on Metal, text on a CPU worker). The encoded text
	 * features are consumed automatically by sam3_segment().
	 */
	const char *async_text = NULL;
	for (int i = 0; i < args.n_prompts; i++) {
		if (args.prompts[i].type == SAM3_PROMPT_TEXT &&
		    args.prompts[i].text) {
			async_text = args.prompts[i].text;
			break;
		}
	}
	if (async_text) {
		enum sam3_error te_err = sam3_set_text(ctx, async_text);
		if (te_err != SAM3_OK) {
			fprintf(stderr,
				"warning: async set_text failed (%s); "
				"falling back to inline encode\n",
				sam3_error_str(te_err));
		}
	}

	/* Encode image (overlaps with the async text worker if any) */
	if (!args.quiet)
		cli_progress("Encoding image: %s\n", args.image_path);
	t0 = sam3_time_ns();

	if (args.use_stdin) {
		struct sam3_image si = {0};
		err = sam3_image_load_memory(stdin_buf, stdin_size, &si);
		if (err != SAM3_OK) {
			fprintf(stderr,
				"error: failed to decode stdin "
				"image: %s\n",
				sam3_error_str(err));
			ret = (int)sam3_error_to_exit(err);
			goto cleanup;
		}
		/*
		 * sam3_set_image_file() internally resizes to
		 * image_size x image_size. Replicate that here
		 * for the stdin path.
		 */
		int target = sam3_get_image_size(ctx);
		int orig_w = si.width;
		int orig_h = si.height;
		struct sam3_image resized = {0};
		err = sam3_image_resize(&si, &resized, target, target);
		sam3_image_free(&si);
		if (err != SAM3_OK) {
			fprintf(stderr,
				"error: failed to resize stdin "
				"image: %s\n",
				sam3_error_str(err));
			ret = (int)sam3_error_to_exit(err);
			goto cleanup;
		}
		err = sam3_set_image(ctx, resized.pixels,
				     resized.width, resized.height);
		sam3_image_free(&resized);
		if (err == SAM3_OK)
			sam3_set_prompt_space(ctx, orig_w, orig_h);
	} else {
		err = sam3_set_image_file(ctx, args.image_path);
	}

	t1 = sam3_time_ns();
	if (err != SAM3_OK) {
		fprintf(stderr,
			"error: failed to encode image '%s': %s\n",
			args.image_path, sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}
	if (!args.quiet)
		cli_progress("  image encoded in %.1fms\n",
			     (double)(t1 - t0) / 1e6);

	/* Run segmentation */
	if (!args.quiet)
		cli_progress("Segmenting with %d prompt(s)...\n",
			     args.n_prompts);
	t0 = sam3_time_ns();
	err = sam3_segment(ctx, args.prompts, args.n_prompts, &result);
	t1 = sam3_time_ns();
	if (err != SAM3_OK) {
		fprintf(stderr, "error: segmentation failed: %s\n",
			sam3_error_str(err));
		ret = (int)sam3_error_to_exit(err);
		goto cleanup;
	}
	if (!args.quiet)
		cli_progress("  segmentation completed in %.1fms\n",
			     (double)(t1 - t0) / 1e6);

	/* Dump tensors before NMS (which compacts masks in-place) */
	if (args.dump_dir) {
		/* Dump cached image features via public API */
		enum sam3_error dump_err =
			sam3_dump_tensors(ctx, args.dump_dir);
		if (dump_err != SAM3_OK)
			fprintf(stderr,
				"warning: tensor dump failed: %s\n",
				sam3_error_str(dump_err));

		/* Dump mask logits from result */
		if (result.n_masks > 0) {
			struct sam3_tensor mask_t = {0};
			mask_t.dtype = SAM3_DTYPE_F32;
			mask_t.n_dims = 3;
			mask_t.dims[0] = result.n_masks;
			mask_t.dims[1] = result.mask_height;
			mask_t.dims[2] = result.mask_width;
			mask_t.data = result.masks;
			mask_t.nbytes =
				(size_t)result.n_masks *
				result.mask_height *
				result.mask_width * sizeof(float);

			char dump_path[1024];
			snprintf(dump_path, sizeof(dump_path),
				 "%s/mask_logits.bin",
				 args.dump_dir);
			sam3_tensor_dump(dump_path, &mask_t);
			if (!args.quiet)
				cli_progress("  dumped mask_logits.bin "
					     "[%d, %d, %d]\n",
					     result.n_masks,
					     result.mask_height,
					     result.mask_width);
		}

		/* Dump IoU scores from result */
		if (result.iou_valid && result.n_masks > 0) {
			struct sam3_tensor score_t = {0};
			score_t.dtype = SAM3_DTYPE_F32;
			score_t.n_dims = 1;
			score_t.dims[0] = result.n_masks;
			score_t.data = result.iou_scores;
			score_t.nbytes = (size_t)result.n_masks *
					 sizeof(float);

			char dump_path[1024];
			snprintf(dump_path, sizeof(dump_path),
				 "%s/scores.bin", args.dump_dir);
			sam3_tensor_dump(dump_path, &score_t);
			if (!args.quiet)
				cli_progress("  dumped scores.bin "
					     "[%d]\n",
					     result.n_masks);
		}
	}

	/* Post-process: greedy mask NMS. kept_buf and tmp_scores
	 * are sized to match the 512-mask cap enforced by
	 * sam3_mask_nms (see include/sam3/internal/mask_nms.h). */
	if (args.nms_enabled && result.iou_valid && result.n_masks > 0) {
		size_t mask_pixels = (size_t)result.mask_width *
				     (size_t)result.mask_height;
		int kept_buf[512];
		float tmp_scores[512];
		int n_kept = sam3_mask_nms(
			result.masks, result.iou_scores,
			result.n_masks,
			result.mask_height, result.mask_width,
			args.nms_prob_thresh,
			args.nms_iou_thresh,
			args.min_mask_quality,
			kept_buf);

		if (n_kept < 0) {
			fprintf(stderr, "warning: NMS failed, "
				"falling back to raw output\n");
		} else if (n_kept == 0) {
			if (!args.quiet)
				cli_progress("NMS: %d -> 0 masks\n",
					     result.n_masks);
			result.n_masks = 0;
		} else {
			/* Compact result.masks and result.iou_scores
			 * to hold only kept entries, in descending-
			 * score order. kept_buf contains indices into
			 * the ORIGINAL mask array which are NOT
			 * guaranteed to be monotonic in position, so
			 * we stage into a scratch buffer to avoid
			 * clobbering source slots that later
			 * iterations still need to read. */
			size_t total = (size_t)n_kept * mask_pixels;
			float *tmp = (float *)malloc(total *
				sizeof(float));
			if (!tmp) {
				fprintf(stderr,
					"warning: NMS compaction "
					"malloc (%zu bytes) failed, "
					"using raw output\n",
					total * sizeof(float));
			} else {
				for (int k = 0; k < n_kept; k++) {
					int src = kept_buf[k];
					memcpy(tmp +
						(size_t)k * mask_pixels,
						result.masks +
						(size_t)src * mask_pixels,
						mask_pixels *
						sizeof(float));
					tmp_scores[k] =
						result.iou_scores[src];
				}
				memcpy(result.masks, tmp,
					total * sizeof(float));
				memcpy(result.iou_scores, tmp_scores,
					(size_t)n_kept *
					sizeof(float));
				free(tmp);
				if (!args.quiet)
					cli_progress("NMS: %d -> "
						     "%d masks\n",
						     result.n_masks,
						     n_kept);
				result.n_masks = n_kept;
			}
		}
	}

	/* Print results summary */
	if (!args.quiet) {
		cli_progress("Results: %d mask(s), %dx%d\n",
			     result.n_masks, result.mask_width,
			     result.mask_height);
		if (result.best_mask >= 0)
			cli_progress("  best mask: %d "
				     "(stability-selected)\n",
				     result.best_mask);
	}
	{
		int mask_pixels = result.mask_width * result.mask_height;

		for (int i = 0; i < result.n_masks; i++) {
			if (!args.quiet) {
				if (result.iou_valid)
					cli_progress("  mask %d: "
						     "IoU = %.4f",
						     i,
						     result.iou_scores[i]);
				else
					cli_progress("  mask %d: "
						     "IoU = N/A", i);

				if (result.boxes_valid) {
					const float *b =
						result.boxes + i * 4;
					cli_progress(
						"  box=[%.0f,%.0f,"
						"%.0f,%.0f]",
						b[0], b[1], b[2], b[3]);
				}
				cli_progress("\n");
			}

			/* Debug: per-mask statistics */
			if (args.verbose) {
				const float *md = result.masks +
						  i * mask_pixels;
				float mn = md[0], mx = md[0],
				      sum = 0.0f;

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

	/* stdout raw mask output (-o -) */
	if (args.use_stdout) {
		uint32_t hdr[3] = {(uint32_t)result.n_masks,
				   (uint32_t)result.mask_height,
				   (uint32_t)result.mask_width};
		fwrite(hdr, sizeof(uint32_t), 3, stdout);
		fwrite(result.masks, sizeof(float),
		       (size_t)result.n_masks * result.mask_height *
		       result.mask_width,
		       stdout);
		fflush(stdout);
		goto post_output;
	}

	/* stdout PNG (--stdout-png) */
	if (args.stdout_png) {
		int mask_pixels = result.mask_width * result.mask_height;
		int best = result.best_mask >= 0 ? result.best_mask : 0;
		if (best >= result.n_masks)
			best = 0;
		if (result.n_masks > 0) {
			const float *md = result.masks +
					  best * mask_pixels;
			uint8_t *gray = malloc((size_t)mask_pixels);
			if (gray) {
				for (int j = 0; j < mask_pixels; j++)
					gray[j] = md[j] >= args.threshold
						  ? 255 : 0;
				stbi_write_png_to_func(
					stdout_write_func, NULL,
					result.mask_width,
					result.mask_height,
					1, gray, result.mask_width);
				fflush(stdout);
				free(gray);
			}
		}
		goto post_output;
	}

	/* Write outputs to files */
	{
		int mask_pixels = result.mask_width * result.mask_height;
		char path_buf[1024];

		/* Post-processing buffers for --smooth */
		unsigned char *smooth_bin = NULL;
		unsigned char *smooth_work = NULL;
		unsigned char *smooth_out = NULL;
		int *smooth_labels = NULL;
		int *smooth_stack = NULL;
		float *smooth_probs = NULL;

		if (args.smooth) {
			size_t mp = (size_t)mask_pixels;
			smooth_bin = malloc(mp);
			smooth_work = malloc(mp);
			smooth_out = malloc(mp);
			smooth_labels = malloc(mp * sizeof(int));
			smooth_stack = malloc(mp * sizeof(int));
			smooth_probs = malloc(mp * sizeof(float));
			if (!smooth_bin || !smooth_work ||
			    !smooth_out || !smooth_labels ||
			    !smooth_stack || !smooth_probs) {
				fprintf(stderr,
					"warning: --smooth malloc "
					"failed, falling back to "
					"raw output\n");
				free(smooth_bin);
				free(smooth_work);
				free(smooth_out);
				free(smooth_labels);
				free(smooth_stack);
				free(smooth_probs);
				smooth_bin = NULL;
				args.smooth = 0;
			}
		}

		for (int i = 0; i < result.n_masks; i++) {
			const float *mask_data =
				result.masks + i * mask_pixels;

			/* Always write raw binary */
			snprintf(path_buf, sizeof(path_buf),
				 "%s/mask_%d.bin",
				 args.output_dir, i);
			if (write_mask_bin(path_buf, mask_data,
					   mask_pixels))
				fprintf(stderr,
					"warning: failed to write "
					"%s\n", path_buf);
			else if (!args.quiet)
				cli_progress("  wrote %s\n", path_buf);

			/* Grayscale PNG */
			if (args.write_png) {
				snprintf(path_buf, sizeof(path_buf),
					 "%s/mask_%d.png",
					 args.output_dir, i);
				if (args.smooth && smooth_bin) {
					/* Binarize, morpho open, remove small */
					for (int j = 0; j < mask_pixels; j++)
						smooth_bin[j] =
							(mask_data[j] >= args.threshold) ? 1 : 0;
					sam3_mask_morpho_open(
						smooth_bin, smooth_out,
						result.mask_width,
						result.mask_height,
						smooth_work);
					sam3_mask_remove_small(
						smooth_out,
						result.mask_width,
						result.mask_height,
						16, smooth_labels,
						smooth_stack);
					uint8_t *gray = malloc(
						(size_t)mask_pixels);
					if (gray) {
						for (int j = 0;
						     j < mask_pixels; j++)
							gray[j] = smooth_out[j]
								? 255 : 0;
						stbi_write_png(
							path_buf,
							result.mask_width,
							result.mask_height,
							1, gray,
							result.mask_width);
						free(gray);
						if (!args.quiet)
							cli_progress(
								"  wrote %s "
								"(smoothed)\n",
								path_buf);
					}
				} else {
					if (write_mask_png(path_buf,
							   mask_data,
							   result.mask_width,
							   result.mask_height,
							   args.threshold))
						fprintf(stderr,
							"warning: failed "
							"to write %s\n",
							path_buf);
					else if (!args.quiet)
						cli_progress(
							"  wrote %s\n",
							path_buf);
				}
			}

			/* Color overlay */
			if (args.write_overlay) {
				snprintf(path_buf, sizeof(path_buf),
					 "%s/overlay_%d.png",
					 args.output_dir, i);
				if (args.smooth && smooth_probs) {
					sam3_mask_sigmoid(
						mask_data,
						smooth_probs,
						mask_pixels);
					if (write_overlay(path_buf,
							  smooth_probs,
							  result.mask_width,
							  result.mask_height,
							  &orig_img,
							  0.5f))
						fprintf(stderr,
							"warning: failed "
							"to write %s\n",
							path_buf);
					else if (!args.quiet)
						cli_progress(
							"  wrote %s "
							"(smooth "
							"overlay)\n",
							path_buf);
				} else {
					if (write_overlay(path_buf,
							  mask_data,
							  result.mask_width,
							  result.mask_height,
							  &orig_img,
							  args.threshold))
						fprintf(stderr,
							"warning: failed "
							"to write %s\n",
							path_buf);
					else if (!args.quiet)
						cli_progress(
							"  wrote %s\n",
							path_buf);
				}
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
				else if (!args.quiet)
					cli_progress("  wrote %s\n",
						     path_buf);
			}
		}

		/* Merged overlay with distinct colors per mask */
		if (args.write_overlay && result.n_masks > 0) {
			snprintf(path_buf, sizeof(path_buf),
				 "%s/overlay_all.png",
				 args.output_dir);
			if (write_overlay_merged(path_buf,
						 result.masks,
						 result.n_masks,
						 result.mask_width,
						 result.mask_height,
						 &orig_img,
						 args.threshold))
				fprintf(stderr,
					"warning: failed to write "
					"%s\n", path_buf);
			else if (!args.quiet)
				cli_progress("  wrote %s\n",
					     path_buf);
		}

		free(smooth_bin);
		free(smooth_work);
		free(smooth_out);
		free(smooth_labels);
		free(smooth_stack);
		free(smooth_probs);
	}

post_output:
	/* JSON output (--json) */
	if (args.json) {
		FILE *json_fp = args.use_stdout ? stderr : stdout;
		cli_json_result(json_fp, &result);
	}

	/* Profiling report */
	if (args.profile)
		sam3_profile_report(ctx);

	ret = SAM3_EXIT_OK;

cleanup:
	sam3_result_free(&result);
	sam3_image_free(&orig_img);
	sam3_image_free(&stdin_img);
	free(stdin_buf);
	sam3_free(ctx);
	return ret;
}
