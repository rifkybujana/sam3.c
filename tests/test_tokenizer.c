/*
 * tests/test_tokenizer.c - Unit tests for BPE tokenizer
 *
 * Tests tokenizer initialization, byte-level encoding, case folding,
 * empty input, padding behavior, truncation, and CLIP BPE mode with
 * verified reference token IDs from the Python CLIP tokenizer.
 *
 * Key types:  sam3_tokenizer
 * Depends on: test_helpers.h, model/tokenizer.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/tokenizer.h"

#include <stdio.h>
#include <string.h>

/* Path to CLIP BPE vocabulary (may not exist in CI) */
#define BPE_VOCAB_PATH "models/bpe_simple_vocab_16e6.txt"

/* --- test_tokenizer_init --- */

static void test_tokenizer_init(void)
{
	struct sam3_tokenizer tok;
	enum sam3_error err = sam3_tokenizer_init(&tok);

	ASSERT_EQ(err, SAM3_OK);
	ASSERT(tok.vocab != NULL);
	ASSERT_EQ(tok.vocab_size, 49408);
	ASSERT_EQ(tok.sot_token, 49406);
	ASSERT_EQ(tok.eot_token, 49407);
	ASSERT_EQ(tok.n_merges, 0);
	ASSERT(tok.merge_first == NULL);
	ASSERT(tok.merge_second == NULL);
	ASSERT_EQ(tok.bpe_loaded, 0);
	ASSERT(tok.encoder_map == NULL);
	ASSERT(tok.merge_rank_map == NULL);

	/* Verify byte tokens are initialized */
	ASSERT(tok.vocab[0] != NULL);
	ASSERT_EQ((unsigned char)tok.vocab[0][0], 0);
	ASSERT_EQ(tok.vocab[0][1], '\0');

	ASSERT(tok.vocab[65] != NULL);
	ASSERT_EQ(tok.vocab[65][0], 'A');

	ASSERT(tok.vocab[255] != NULL);
	ASSERT_EQ((unsigned char)tok.vocab[255][0], 255);

	/* Slots beyond 256 should be NULL (not initialized) */
	ASSERT(tok.vocab[256] == NULL);

	/* Init with NULL should return EINVAL */
	ASSERT_EQ(sam3_tokenizer_init(NULL), SAM3_EINVAL);

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_simple --- */

static void test_encode_simple(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "cat", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	/* SOT + 'c' + 'a' + 't' + EOT = 5 meaningful tokens */
	ASSERT_EQ(n, 5);
	ASSERT_EQ(tokens[0], 49406);		/* SOT */
	ASSERT_EQ(tokens[1], (int32_t)'c');	/* 99 */
	ASSERT_EQ(tokens[2], (int32_t)'a');	/* 97 */
	ASSERT_EQ(tokens[3], (int32_t)'t');	/* 116 */
	ASSERT_EQ(tokens[4], 49407);		/* EOT */

	/* Rest should be padded with EOT */
	for (int i = 5; i < SAM3_TOKENIZER_CONTEXT_LEN; i++)
		ASSERT_EQ(tokens[i], 49407);

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_uppercase --- */

static void test_encode_uppercase(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	int32_t tokens_upper[SAM3_TOKENIZER_CONTEXT_LEN];
	int32_t tokens_lower[SAM3_TOKENIZER_CONTEXT_LEN];

	sam3_tokenizer_encode(&tok, "Cat", tokens_upper,
			      SAM3_TOKENIZER_CONTEXT_LEN);
	sam3_tokenizer_encode(&tok, "cat", tokens_lower,
			      SAM3_TOKENIZER_CONTEXT_LEN);

	/* Uppercase should be lowercased to match */
	for (int i = 0; i < SAM3_TOKENIZER_CONTEXT_LEN; i++)
		ASSERT_EQ(tokens_upper[i], tokens_lower[i]);

	/* Also verify 'C' (67) becomes 'c' (99) */
	ASSERT_EQ(tokens_upper[1], (int32_t)'c');

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_empty --- */

static void test_encode_empty(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	/* SOT + EOT = 2 meaningful tokens */
	ASSERT_EQ(n, 2);
	ASSERT_EQ(tokens[0], 49406);	/* SOT */
	ASSERT_EQ(tokens[1], 49407);	/* EOT */

	/* Rest padded with EOT */
	for (int i = 2; i < SAM3_TOKENIZER_CONTEXT_LEN; i++)
		ASSERT_EQ(tokens[i], 49407);

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_padding --- */

static void test_encode_padding(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	int32_t tokens[10];
	int n = sam3_tokenizer_encode(&tok, "hi", tokens, 10);

	/* SOT + 'h' + 'i' + EOT = 4 meaningful */
	ASSERT_EQ(n, 4);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], (int32_t)'h');
	ASSERT_EQ(tokens[2], (int32_t)'i');
	ASSERT_EQ(tokens[3], 49407);

	/* Positions 4-9 padded with EOT */
	for (int i = 4; i < 10; i++)
		ASSERT_EQ(tokens[i], 49407);

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_truncation --- */

static void test_encode_truncation(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* Use a tiny buffer: max_tokens=5 */
	int32_t tokens[5];
	int n = sam3_tokenizer_encode(&tok, "abcdefghij", tokens, 5);

	/* SOT + 'a' + 'b' + 'c' + EOT = 5 */
	ASSERT_EQ(n, 5);
	ASSERT_EQ(tokens[0], 49406);		/* SOT */
	ASSERT_EQ(tokens[1], (int32_t)'a');
	ASSERT_EQ(tokens[2], (int32_t)'b');
	ASSERT_EQ(tokens[3], (int32_t)'c');
	ASSERT_EQ(tokens[4], 49407);		/* EOT */

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_invalid_args --- */

static void test_encode_invalid_args(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	int32_t tokens[10];

	/* NULL tokenizer */
	ASSERT_EQ(sam3_tokenizer_encode(NULL, "hi", tokens, 10), 0);

	/* NULL text */
	ASSERT_EQ(sam3_tokenizer_encode(&tok, NULL, tokens, 10), 0);

	/* NULL output buffer */
	ASSERT_EQ(sam3_tokenizer_encode(&tok, "hi", NULL, 10), 0);

	/* max_tokens too small */
	ASSERT_EQ(sam3_tokenizer_encode(&tok, "hi", tokens, 1), 0);
	ASSERT_EQ(sam3_tokenizer_encode(&tok, "hi", tokens, 0), 0);

	sam3_tokenizer_free(&tok);
}

/* --- test_tokenizer_free_null --- */

static void test_tokenizer_free_null(void)
{
	/* Should not crash */
	sam3_tokenizer_free(NULL);

	/* Double free should be safe (fields zeroed after first free) */
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	sam3_tokenizer_free(&tok);
	sam3_tokenizer_free(&tok);
}

/* --- test_load_bpe_missing_file --- */

static void test_load_bpe_missing_file(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	enum sam3_error err = sam3_tokenizer_load_bpe(
		&tok, "/tmp/nonexistent_bpe_file.txt");
	ASSERT_EQ(err, SAM3_EIO);

	/* Tokenizer should be unchanged */
	ASSERT_EQ(tok.n_merges, 0);
	ASSERT_EQ(tok.bpe_loaded, 0);

	sam3_tokenizer_free(&tok);
}

/* --- test_load_bpe_invalid_args --- */

static void test_load_bpe_invalid_args(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* NULL tokenizer */
	ASSERT_EQ(sam3_tokenizer_load_bpe(NULL, "/tmp/test.txt"),
		  SAM3_EINVAL);

	/* NULL path */
	ASSERT_EQ(sam3_tokenizer_load_bpe(&tok, NULL), SAM3_EINVAL);

	sam3_tokenizer_free(&tok);
}

/* --- test_bytes_to_unicode --- */

/*
 * test_bytes_to_unicode - Verify the bytes_to_unicode table matches CLIP.
 */
static void test_bytes_to_unicode(void)
{
	struct sam3_tokenizer tok;
	enum sam3_error err = sam3_tokenizer_init(&tok);
	ASSERT_EQ(err, SAM3_OK);

	/* Printable ASCII: byte 65 ('A') -> "A" */
	ASSERT_EQ(strcmp(tok.byte_unicode[65], "A"), 0);

	/* Printable ASCII: byte 97 ('a') -> "a" */
	ASSERT_EQ(strcmp(tok.byte_unicode[97], "a"), 0);

	/* Non-printable: byte 0 maps to chr(256) = U+0100 = UTF-8: C4 80 */
	ASSERT_EQ((unsigned char)tok.byte_unicode[0][0], 0xC4);
	ASSERT_EQ((unsigned char)tok.byte_unicode[0][1], 0x80);

	/* Byte 32 (space) is non-printable in CLIP */
	ASSERT(tok.byte_unicode[32][0] != ' ');

	/* Byte 33 ('!') is printable -> maps to "!" */
	ASSERT_EQ(strcmp(tok.byte_unicode[33], "!"), 0);

	sam3_tokenizer_free(&tok);
}

/* --- test_encode_bpe_backward_compat --- */

static void test_encode_bpe_backward_compat(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* No BPE loaded — should produce byte-level tokens */
	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "hi!", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	/* SOT + 'h' + 'i' + '!' + EOT = 5 meaningful tokens */
	ASSERT_EQ(n, 5);
	ASSERT_EQ(tokens[0], 49406);		/* SOT */
	ASSERT_EQ(tokens[1], (int32_t)'h');	/* 104 */
	ASSERT_EQ(tokens[2], (int32_t)'i');	/* 105 */
	ASSERT_EQ(tokens[3], (int32_t)'!');	/* 33 */
	ASSERT_EQ(tokens[4], 49407);		/* EOT */

	sam3_tokenizer_free(&tok);
}

/* ---- CLIP BPE mode tests (skip if vocab file not found) ---- */

static int load_clip_bpe(struct sam3_tokenizer *tok)
{
	enum sam3_error err = sam3_tokenizer_load_bpe(tok, BPE_VOCAB_PATH);
	if (err == SAM3_EIO) {
		printf("  [SKIP] BPE file not found: %s\n", BPE_VOCAB_PATH);
		return 0;
	}
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_EQ(tok->bpe_loaded, 1);
	return (err == SAM3_OK);
}

static void test_clip_hello_world(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "hello world", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 4);
	ASSERT_EQ(tokens[0], 49406);	/* SOT */
	ASSERT_EQ(tokens[1], 3306);	/* hello */
	ASSERT_EQ(tokens[2], 1002);	/* world */
	ASSERT_EQ(tokens[3], 49407);	/* EOT */

	/* CLIP pads with 0 */
	for (int i = 4; i < SAM3_TOKENIZER_CONTEXT_LEN; i++)
		ASSERT_EQ(tokens[i], 0);

	sam3_tokenizer_free(&tok);
}

static void test_clip_cat(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "cat", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 3);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 2368);
	ASSERT_EQ(tokens[2], 49407);

	sam3_tokenizer_free(&tok);
}

static void test_clip_contraction(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "it's", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 4);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 585);	/* it */
	ASSERT_EQ(tokens[2], 568);	/* 's */
	ASSERT_EQ(tokens[3], 49407);

	sam3_tokenizer_free(&tok);
}

static void test_clip_multi_word(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "the quick brown fox", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 6);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 518);	/* the */
	ASSERT_EQ(tokens[2], 3712);	/* quick */
	ASSERT_EQ(tokens[3], 2866);	/* brown */
	ASSERT_EQ(tokens[4], 3240);	/* fox */
	ASSERT_EQ(tokens[5], 49407);

	sam3_tokenizer_free(&tok);
}

static void test_clip_single_char(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "a", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 3);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 320);	/* a */
	ASSERT_EQ(tokens[2], 49407);

	sam3_tokenizer_free(&tok);
}

static void test_clip_phrase(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "a photo of a cat", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 7);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 320);	/* a */
	ASSERT_EQ(tokens[2], 1125);	/* photo */
	ASSERT_EQ(tokens[3], 539);	/* of */
	ASSERT_EQ(tokens[4], 320);	/* a */
	ASSERT_EQ(tokens[5], 2368);	/* cat */
	ASSERT_EQ(tokens[6], 49407);

	sam3_tokenizer_free(&tok);
}

static void test_clip_person(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "person", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 3);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 2533);	/* person */
	ASSERT_EQ(tokens[2], 49407);

	sam3_tokenizer_free(&tok);
}

static void test_clip_case_insensitive(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);
	if (!load_clip_bpe(&tok)) {
		sam3_tokenizer_free(&tok);
		return;
	}

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "A dog", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	ASSERT_EQ(n, 4);
	ASSERT_EQ(tokens[0], 49406);
	ASSERT_EQ(tokens[1], 320);	/* a */
	ASSERT_EQ(tokens[2], 1929);	/* dog */
	ASSERT_EQ(tokens[3], 49407);

	sam3_tokenizer_free(&tok);
}

/* --- Main --- */

int main(void)
{
	/* Byte-level fallback tests */
	test_tokenizer_init();
	test_bytes_to_unicode();
	test_encode_simple();
	test_encode_uppercase();
	test_encode_empty();
	test_encode_padding();
	test_encode_truncation();
	test_encode_invalid_args();
	test_tokenizer_free_null();
	test_load_bpe_missing_file();
	test_load_bpe_invalid_args();
	test_encode_bpe_backward_compat();

	/* CLIP BPE mode tests */
	test_clip_hello_world();
	test_clip_cat();
	test_clip_contraction();
	test_clip_multi_word();
	test_clip_single_char();
	test_clip_phrase();
	test_clip_person();
	test_clip_case_insensitive();

	TEST_REPORT();
}
