/*
 * tools/cli_common.c - Shared CLI utilities implementation
 *
 * Implements JSON output, stdin reading, and other utilities shared
 * across sam3 subcommands. JSON is hand-written (no dependency) to
 * keep the build simple.
 *
 * Key types:  (none -- implements cli_common.h)
 * Depends on: cli_common.h, core/weight.h
 * Used by:    cli_segment.c, cli_convert.c, cli_info.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_common.h"
#include "core/weight.h"

void cli_json_result(FILE *fp, const struct sam3_result *result)
{
	fprintf(fp, "{\n");
	fprintf(fp, "  \"n_masks\": %d,\n", result->n_masks);
	fprintf(fp, "  \"mask_width\": %d,\n", result->mask_width);
	fprintf(fp, "  \"mask_height\": %d,\n", result->mask_height);
	fprintf(fp, "  \"best_mask\": %d,\n", result->best_mask);
	fprintf(fp, "  \"masks\": [\n");

	for (int i = 0; i < result->n_masks; i++) {
		fprintf(fp, "    {\"index\": %d", i);
		if (result->iou_valid)
			fprintf(fp, ", \"iou_score\": %.6f",
				result->iou_scores[i]);
		if (result->boxes_valid) {
			const float *b = result->boxes + i * 4;
			fprintf(fp, ", \"box\": [%.1f, %.1f, %.1f, %.1f]",
				b[0], b[1], b[2], b[3]);
		}
		fprintf(fp, "}%s\n",
			i < result->n_masks - 1 ? "," : "");
	}

	fprintf(fp, "  ]\n");
	fprintf(fp, "}\n");
}

void cli_json_model_info(FILE *fp,
			 const struct sam3_weight_header *header,
			 size_t file_size)
{
	fprintf(fp, "{\n");
	fprintf(fp, "  \"format_version\": %u,\n", header->version);
	fprintf(fp, "  \"n_tensors\": %u,\n", header->n_tensors);
	fprintf(fp, "  \"image_size\": %d,\n", header->image_size);
	fprintf(fp, "  \"encoder_dim\": %d,\n", header->encoder_dim);
	fprintf(fp, "  \"decoder_dim\": %d,\n", header->decoder_dim);
	fprintf(fp, "  \"encoder_layers\": %d,\n",
		header->n_encoder_layers);
	fprintf(fp, "  \"decoder_layers\": %d,\n",
		header->n_decoder_layers);
	fprintf(fp, "  \"file_size\": %zu\n", file_size);
	fprintf(fp, "}\n");
}

int cli_read_stdin(uint8_t **out_buf, size_t *out_size)
{
	size_t cap = 4096;
	size_t len = 0;
	uint8_t *buf = malloc(cap);

	if (!buf)
		return 1;

	for (;;) {
		size_t n = fread(buf + len, 1, cap - len, stdin);
		len += n;
		if (n == 0)
			break;
		if (len == cap) {
			cap *= 2;
			uint8_t *nb = realloc(buf, cap);
			if (!nb) {
				free(buf);
				return 1;
			}
			buf = nb;
		}
	}

	if (ferror(stdin)) {
		free(buf);
		return 1;
	}

	*out_buf = buf;
	*out_size = len;
	return 0;
}
