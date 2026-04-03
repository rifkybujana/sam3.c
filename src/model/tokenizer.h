/*
 * src/model/tokenizer.h - BPE tokenizer for CLIP text encoding
 *
 * Defines a byte-pair encoding tokenizer compatible with the CLIP text
 * encoder used by SAM3. Initializes with a byte-level fallback vocabulary
 * (256 byte tokens + special tokens) that works without an external vocab
 * file. Full BPE merge tables can be loaded later for production quality.
 *
 * Key types:  sam3_tokenizer
 * Depends on: sam3/sam3_types.h
 * Used by:    model/text_encoder.c (future), tests/test_tokenizer.c
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TOKENIZER_H
#define SAM3_MODEL_TOKENIZER_H

#include "sam3/sam3_types.h"
#include <stdint.h>

#define SAM3_TOKENIZER_CONTEXT_LEN 77	/* max sequence length */

struct sam3_tokenizer {
	char **vocab;		/* array of token strings */
	int    vocab_size;
	int   *merge_first;	/* BPE merge pairs: first token ID */
	int   *merge_second;/* BPE merge pairs: second token ID */
	int    n_merges;
	int    sot_token;	/* start-of-text token ID */
	int    eot_token;	/* end-of-text token ID */
	int    bpe_loaded;	/* 0 = byte-level fallback, 1 = CLIP BPE */
	char   byte_unicode[256][5]; /* bytes_to_unicode table (UTF-8) */
	void  *encoder_map;	/* string -> token_id hash table (private) */
	void  *merge_rank_map;	/* "tokA\x01tokB" -> rank hash table */
};

/*
 * sam3_tokenizer_init - Initialize tokenizer with byte-level CLIP vocabulary.
 *
 * @tok: Tokenizer struct (caller-allocated, zeroed)
 *
 * Sets up a minimal byte-level tokenizer: 256 byte tokens (one per byte
 * value) plus CLIP special tokens (SOT=49406, EOT=49407). BPE merge
 * tables are left empty; the full vocabulary can be loaded later via a
 * dedicated load function.
 *
 * Returns SAM3_OK on success, SAM3_ENOMEM on allocation failure.
 * Caller must call sam3_tokenizer_free() to release resources.
 */
enum sam3_error sam3_tokenizer_init(struct sam3_tokenizer *tok);

/*
 * sam3_tokenizer_load_bpe - Load BPE merge table from a text file.
 *
 * @tok:  Initialized tokenizer (must have byte-level vocab set up)
 * @path: Path to BPE merges file (one merge per line, space-separated)
 *
 * Reads a CLIP-format BPE merge file. Lines starting with '#' are skipped
 * as comments. Each non-comment line contains two space-separated token
 * strings representing a merge pair. For merge at index i, the merged
 * token is stored at vocab[256 + i] and merge_first[i] / merge_second[i]
 * record the constituent token IDs.
 *
 * Returns SAM3_OK on success, SAM3_EINVAL on bad arguments or malformed
 * file, SAM3_EIO if the file cannot be opened, SAM3_ENOMEM on allocation
 * failure. On error, existing tokenizer state is not modified.
 */
enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
					const char *path);

/*
 * sam3_tokenizer_encode - Encode text to token IDs.
 *
 * @tok:        Initialized tokenizer
 * @text:       Null-terminated UTF-8 text string
 * @tokens:     Output array for token IDs (caller-allocated)
 * @max_tokens: Size of output array (should be SAM3_TOKENIZER_CONTEXT_LEN)
 *
 * The first token is always SOT. In byte-level fallback mode, each byte
 * of the lowercased text maps to its byte token ID and padding uses EOT.
 * In CLIP BPE mode (after load_bpe), text is pre-tokenized and BPE-merged
 * per the CLIP vocabulary; padding uses 0. The sequence is terminated
 * with EOT. If the text is too long, it is truncated to fit within
 * max_tokens (SOT at start, EOT at end).
 *
 * Returns number of meaningful tokens written (including SOT and EOT),
 * or 0 on invalid arguments.
 */
int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens);

/*
 * sam3_tokenizer_free - Free tokenizer resources.
 *
 * @tok: Tokenizer to free (may be NULL). Vocab strings, vocab array,
 *       and merge arrays are freed. Fields are zeroed after free.
 */
void sam3_tokenizer_free(struct sam3_tokenizer *tok);

#endif /* SAM3_MODEL_TOKENIZER_H */
