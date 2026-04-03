/*
 * src/model/tokenizer.c - BPE tokenizer for CLIP text encoding
 *
 * Implements a byte-level fallback tokenizer compatible with the CLIP
 * vocabulary layout. The vocabulary has 49408 slots matching CLIP: the
 * first 256 are single-byte tokens, and the last two (49406, 49407) are
 * the start-of-text and end-of-text special tokens. BPE merges are not
 * loaded in this fallback mode, so each byte maps directly to its token
 * ID. This is CPU-only text processing with no GPU involvement.
 *
 * Key types:  sam3_tokenizer
 * Depends on: tokenizer.h
 * Used by:    model/text_encoder.c (future)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

/* CLIP vocabulary constants */
#define CLIP_VOCAB_SIZE 49408
#define CLIP_SOT_TOKEN  49406
#define CLIP_EOT_TOKEN  49407

enum sam3_error sam3_tokenizer_init(struct sam3_tokenizer *tok)
{
	if (!tok)
		return SAM3_EINVAL;

	memset(tok, 0, sizeof(*tok));

	tok->vocab_size = CLIP_VOCAB_SIZE;
	tok->vocab = calloc((size_t)tok->vocab_size, sizeof(char *));
	if (!tok->vocab)
		return SAM3_ENOMEM;

	/* Initialize byte tokens (first 256 entries) */
	for (int i = 0; i < 256; i++) {
		tok->vocab[i] = malloc(2);
		if (!tok->vocab[i]) {
			sam3_tokenizer_free(tok);
			return SAM3_ENOMEM;
		}
		tok->vocab[i][0] = (char)i;
		tok->vocab[i][1] = '\0';
	}

	/* Special tokens */
	tok->sot_token = CLIP_SOT_TOKEN;
	tok->eot_token = CLIP_EOT_TOKEN;

	/* No merges in byte-level fallback mode */
	tok->n_merges = 0;
	tok->merge_first = NULL;
	tok->merge_second = NULL;

	return SAM3_OK;
}

int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens)
{
	if (!tok || !text || !tokens || max_tokens < 2)
		return 0;

	int pos = 0;

	/* Start-of-text token */
	tokens[pos++] = (int32_t)tok->sot_token;

	/* Encode each byte of the lowercased text */
	for (const char *p = text; *p && pos < max_tokens - 1; p++) {
		unsigned char c = (unsigned char)*p;

		/* ASCII lowercase */
		if (c >= 'A' && c <= 'Z')
			c = c - 'A' + 'a';

		tokens[pos++] = (int32_t)c;
	}

	/* End-of-text token */
	if (pos < max_tokens)
		tokens[pos++] = (int32_t)tok->eot_token;

	/* Record meaningful token count before padding */
	int n_tokens = pos;

	/* Pad remainder with EOT */
	while (pos < max_tokens)
		tokens[pos++] = (int32_t)tok->eot_token;

	return n_tokens;
}

void sam3_tokenizer_free(struct sam3_tokenizer *tok)
{
	if (!tok)
		return;

	if (tok->vocab) {
		for (int i = 0; i < tok->vocab_size; i++)
			free(tok->vocab[i]);
		free(tok->vocab);
		tok->vocab = NULL;
	}

	free(tok->merge_first);
	tok->merge_first = NULL;

	free(tok->merge_second);
	tok->merge_second = NULL;

	tok->vocab_size = 0;
	tok->n_merges = 0;
}
