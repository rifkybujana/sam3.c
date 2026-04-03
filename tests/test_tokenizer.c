/*
 * tests/test_tokenizer.c - Unit tests for BPE tokenizer
 *
 * Tests tokenizer initialization, byte-level encoding, case folding,
 * empty input, padding behavior, and truncation of long input. Uses
 * malloc-based allocation (no arena needed for tokenizer tests).
 *
 * Key types:  sam3_tokenizer
 * Depends on: test_helpers.h, model/tokenizer.h
 * Used by:    CTest
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"
#include "model/tokenizer.h"

#include <stdio.h>
#include <string.h>

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

	/* SOT + 'a' + 'b' + 'c' + 'd' = 5, truncated (no room for EOT
	 * beyond the 4 byte tokens that fit after SOT)
	 *
	 * Actually: pos=0 SOT, then loop adds bytes while pos < 4
	 * (max_tokens - 1), so we get SOT + 'a' + 'b' + 'c', then
	 * EOT at pos 4.
	 */
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

/* --- test_load_bpe --- */

static void test_load_bpe(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* Write a small BPE merge file */
	const char *path = "/tmp/test_bpe.txt";
	FILE *fp = fopen(path, "w");
	ASSERT(fp != NULL);
	fprintf(fp, "# version comment\n");
	fprintf(fp, "h i\n");
	fprintf(fp, "hi !\n");
	fclose(fp);

	enum sam3_error err = sam3_tokenizer_load_bpe(&tok, path);
	ASSERT_EQ(err, SAM3_OK);

	/* Verify merge count */
	ASSERT_EQ(tok.n_merges, 2);
	ASSERT(tok.merge_first != NULL);
	ASSERT(tok.merge_second != NULL);

	/* First merge: 'h' + 'i' -> "hi" at vocab[256] */
	ASSERT_EQ(tok.merge_first[0], (int)'h');
	ASSERT_EQ(tok.merge_second[0], (int)'i');
	ASSERT(tok.vocab[256] != NULL);
	ASSERT(strcmp(tok.vocab[256], "hi") == 0);

	/* Second merge: "hi" + '!' -> "hi!" at vocab[257] */
	ASSERT_EQ(tok.merge_first[1], 256);	/* "hi" token ID */
	ASSERT_EQ(tok.merge_second[1], (int)'!');
	ASSERT(tok.vocab[257] != NULL);
	ASSERT(strcmp(tok.vocab[257], "hi!") == 0);

	sam3_tokenizer_free(&tok);
	remove(path);
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
	ASSERT(tok.merge_first == NULL);
	ASSERT(tok.merge_second == NULL);

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

/* --- test_load_bpe_malformed --- */

static void test_load_bpe_malformed(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* Write a malformed BPE file (missing second token) */
	const char *path = "/tmp/test_bpe_bad.txt";
	FILE *fp = fopen(path, "w");
	ASSERT(fp != NULL);
	fprintf(fp, "h\n");  /* no space-separated pair */
	fclose(fp);

	enum sam3_error err = sam3_tokenizer_load_bpe(&tok, path);
	ASSERT_EQ(err, SAM3_EINVAL);

	/* Tokenizer merges should be unchanged */
	ASSERT_EQ(tok.n_merges, 0);

	sam3_tokenizer_free(&tok);
	remove(path);
}

/* --- test_encode_bpe_two_merges --- */

static void test_encode_bpe_two_merges(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* Write BPE merge file: "h i" then "hi !" */
	const char *path = "/tmp/test_bpe_encode.txt";
	FILE *fp = fopen(path, "w");
	ASSERT(fp != NULL);
	fprintf(fp, "h i\n");
	fprintf(fp, "hi !\n");
	fclose(fp);

	enum sam3_error err = sam3_tokenizer_load_bpe(&tok, path);
	ASSERT_EQ(err, SAM3_OK);

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "hi!", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	/* SOT + "hi!" (257) + EOT = 3 meaningful tokens */
	ASSERT_EQ(n, 3);
	ASSERT_EQ(tokens[0], 49406);	/* SOT */
	ASSERT_EQ(tokens[1], 257);	/* "hi!" merged token */
	ASSERT_EQ(tokens[2], 49407);	/* EOT */

	sam3_tokenizer_free(&tok);
	remove(path);
}

/* --- test_encode_bpe_one_merge --- */

static void test_encode_bpe_one_merge(void)
{
	struct sam3_tokenizer tok;
	sam3_tokenizer_init(&tok);

	/* Write BPE merge file: "h i" then "hi !" */
	const char *path = "/tmp/test_bpe_encode2.txt";
	FILE *fp = fopen(path, "w");
	ASSERT(fp != NULL);
	fprintf(fp, "h i\n");
	fprintf(fp, "hi !\n");
	fclose(fp);

	enum sam3_error err = sam3_tokenizer_load_bpe(&tok, path);
	ASSERT_EQ(err, SAM3_OK);

	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int n = sam3_tokenizer_encode(&tok, "hi", tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	/* SOT + "hi" (256) + EOT = 3 meaningful tokens */
	ASSERT_EQ(n, 3);
	ASSERT_EQ(tokens[0], 49406);	/* SOT */
	ASSERT_EQ(tokens[1], 256);	/* "hi" merged token */
	ASSERT_EQ(tokens[2], 49407);	/* EOT */

	sam3_tokenizer_free(&tok);
	remove(path);
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

/* --- Main --- */

int main(void)
{
	test_tokenizer_init();
	test_encode_simple();
	test_encode_uppercase();
	test_encode_empty();
	test_encode_padding();
	test_encode_truncation();
	test_encode_invalid_args();
	test_tokenizer_free_null();
	test_load_bpe();
	test_load_bpe_missing_file();
	test_load_bpe_invalid_args();
	test_load_bpe_malformed();
	test_encode_bpe_two_merges();
	test_encode_bpe_one_merge();
	test_encode_bpe_backward_compat();

	TEST_REPORT();
}
