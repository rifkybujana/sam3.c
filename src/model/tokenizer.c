/*
 * src/model/tokenizer.c - BPE tokenizer for CLIP text encoding
 *
 * Implements a BPE tokenizer compatible with the CLIP vocabulary layout.
 * The vocabulary has 49408 slots: the first 256 are single-byte tokens,
 * merged tokens start at index 256, and the last two (49406, 49407) are
 * start-of-text and end-of-text special tokens. Initialization creates
 * a byte-level fallback; sam3_tokenizer_load_bpe() loads a merge table
 * for production-quality subword tokenization.
 *
 * Key types:  sam3_tokenizer
 * Depends on: tokenizer.h
 * Used by:    model/text_encoder.c (future)
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "tokenizer.h"

#include <stdio.h>
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

/*
 * vocab_find - Linear scan to find token ID by string.
 *
 * Returns the index of the token in vocab, or -1 if not found.
 * Only called at load time so linear scan is acceptable.
 */
static int vocab_find(const struct sam3_tokenizer *tok, const char *str,
		      int search_limit)
{
	for (int i = 0; i < search_limit; i++) {
		if (tok->vocab[i] && strcmp(tok->vocab[i], str) == 0)
			return i;
	}
	return -1;
}

enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
					const char *path)
{
	FILE *fp = NULL;
	int *mf = NULL;
	int *ms = NULL;
	char line[512];
	int n_merges = 0;
	int idx = 0;
	enum sam3_error err;

	if (!tok || !path || !tok->vocab)
		return SAM3_EINVAL;

	/* Only allow loading once; re-loading would corrupt rollback state */
	if (tok->n_merges > 0)
		return SAM3_EINVAL;

	fp = fopen(path, "r");
	if (!fp)
		return SAM3_EIO;

	/* First pass: count non-comment, non-empty lines */
	while (fgets(line, (int)sizeof(line), fp)) {
		/* Skip comments and empty lines */
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
			continue;
		n_merges++;
	}

	if (n_merges == 0) {
		fclose(fp);
		return SAM3_EINVAL;
	}

	/* Check that merge count fits in vocab (256 + n_merges <= vocab_size - 2) */
	if (256 + n_merges > tok->vocab_size - 2) {
		fclose(fp);
		return SAM3_EINVAL;
	}

	mf = malloc((size_t)n_merges * sizeof(int));
	ms = malloc((size_t)n_merges * sizeof(int));
	if (!mf || !ms) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Second pass: parse merge lines */
	rewind(fp);

	while (fgets(line, (int)sizeof(line), fp) && idx < n_merges) {
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
			continue;

		/* Strip trailing newline */
		size_t len = strlen(line);
		while (len > 0 && (line[len - 1] == '\n' ||
				   line[len - 1] == '\r')) {
			line[--len] = '\0';
		}

		/* Split at first space */
		char *space = strchr(line, ' ');
		if (!space || space == line || *(space + 1) == '\0') {
			err = SAM3_EINVAL;
			goto cleanup;
		}
		*space = '\0';

		const char *token_a = line;
		const char *token_b = space + 1;

		/* Look up token IDs (search byte tokens + previously merged) */
		int search_limit = 256 + idx;
		int id_a = vocab_find(tok, token_a, search_limit);
		int id_b = vocab_find(tok, token_b, search_limit);

		if (id_a < 0 || id_b < 0) {
			err = SAM3_EINVAL;
			goto cleanup;
		}

		/* Build merged token string */
		size_t len_a = strlen(token_a);
		size_t len_b = strlen(token_b);
		char *merged = malloc(len_a + len_b + 1);
		if (!merged) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(merged, token_a, len_a);
		memcpy(merged + len_a, token_b, len_b);
		merged[len_a + len_b] = '\0';

		/* Store merged token in vocab at 256 + idx */
		free(tok->vocab[256 + idx]);
		tok->vocab[256 + idx] = merged;

		mf[idx] = id_a;
		ms[idx] = id_b;
		idx++;
	}

	if (idx != n_merges) {
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/* Success: commit merge tables to tokenizer */
	free(tok->merge_first);
	free(tok->merge_second);
	tok->merge_first = mf;
	tok->merge_second = ms;
	tok->n_merges = n_merges;
	fclose(fp);
	return SAM3_OK;

cleanup:
	/* Roll back any vocab entries we wrote during the second pass */
	for (int i = 0; i < idx; i++) {
		free(tok->vocab[256 + i]);
		tok->vocab[256 + i] = NULL;
	}
	free(mf);
	free(ms);
	fclose(fp);
	return err;
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

	/* Apply BPE merges if merge table is loaded */
	if (tok->n_merges > 0) {
		int changed = 1;
		while (changed) {
			changed = 0;
			int best_merge = tok->n_merges;
			int best_pos = -1;
			for (int i = 1; i < pos; i++) {
				for (int m = 0; m < best_merge; m++) {
					if (tokens[i] == tok->merge_first[m] &&
					    i + 1 < pos &&
					    tokens[i + 1] == tok->merge_second[m]) {
						best_merge = m;
						best_pos = i;
						break;
					}
				}
			}
			if (best_pos >= 0) {
				tokens[best_pos] = (int32_t)(256 + best_merge);
				for (int i = best_pos + 1; i < pos - 1; i++)
					tokens[i] = tokens[i + 1];
				pos--;
				changed = 1;
			}
		}
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
