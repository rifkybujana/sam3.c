/*
 * tests/bench_tokenizer.c - Tokenizer encoding performance benchmark
 *
 * Benchmarks sam3_tokenizer_encode throughput for byte-level and BPE modes
 * across varying input lengths. Reports tokens/sec and chars/sec. Also
 * measures BPE vocabulary load time. Not included in CTest; run manually
 * via ./bench_tokenizer.
 *
 * Key types:  sam3_tokenizer
 * Depends on: model/tokenizer.h, sam3/sam3_types.h
 * Used by:    manual benchmarking
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model/tokenizer.h"

/* ── Configuration ─────────────────────────────────────────────────── */

#define WARMUP_ITERS	1000
#define TIMED_ITERS	100000

static const char *BPE_VOCAB_PATH = "models/bpe_simple_vocab_16e6.txt.gz";

/* Test inputs of varying length and complexity */
static const struct {
	const char *label;
	const char *text;
} test_inputs[] = {
	{ "short word",    "cat" },
	{ "two words",     "a photo of a cat" },
	{ "sentence",      "a high quality photograph of a golden retriever" },
	{ "long sentence",
	  "a detailed photograph of a beautiful sunset over the ocean "
	  "with dramatic clouds and vibrant orange and purple colors" },
	{ "max length",
	  "segment anything model three is a foundation model for image "
	  "segmentation that can handle diverse prompting including text "
	  "points boxes and masks to produce high quality segmentation "
	  "output for any object in any image" },
};
#define N_INPUTS (int)(sizeof(test_inputs) / sizeof(test_inputs[0]))

/* ── Timing helper ─────────────────────────────────────────────────── */

static double get_time_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Benchmark runner ──────────────────────────────────────────────── */

static void bench_encode(const struct sam3_tokenizer *tok,
			 const char *label, const char *text,
			 const char *mode_label)
{
	int32_t tokens[SAM3_TOKENIZER_CONTEXT_LEN];
	int text_len = (int)strlen(text);

	/* Warmup */
	for (int i = 0; i < WARMUP_ITERS; i++)
		sam3_tokenizer_encode(tok, text, tokens,
				      SAM3_TOKENIZER_CONTEXT_LEN);

	/* Timed */
	int n_tokens = 0;
	double t0 = get_time_ms();
	for (int i = 0; i < TIMED_ITERS; i++)
		n_tokens = sam3_tokenizer_encode(tok, text, tokens,
						 SAM3_TOKENIZER_CONTEXT_LEN);
	double t1 = get_time_ms();

	double total_ms = t1 - t0;
	double avg_ns = (total_ms / TIMED_ITERS) * 1e6;
	double encodes_per_sec = TIMED_ITERS / (total_ms / 1000.0);
	/* Input bytes processed per second */
	double mb_per_sec = (double)text_len * encodes_per_sec / (1024.0 * 1024.0);

	printf("  %-15s | %3d chars | %3d tok | %7.1f ns | %10.0f enc/s | %8.2f MB/s\n",
	       label, text_len, n_tokens, avg_ns,
	       encodes_per_sec, mb_per_sec);

	(void)mode_label;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
	printf("SAM3 Tokenizer Benchmark\n");
	printf("========================\n");
	printf("Warmup: %d | Timed: %d\n\n", WARMUP_ITERS, TIMED_ITERS);

	/* ── Byte-level mode ──────────────────────────────────────── */
	struct sam3_tokenizer tok;
	enum sam3_error err = sam3_tokenizer_init(&tok);
	if (err != SAM3_OK) {
		fprintf(stderr, "Failed to init tokenizer: %d\n", err);
		return 1;
	}

	printf("Byte-level mode:\n");
	printf("  %-15s | %9s | %7s | %10s | %16s | %13s\n",
	       "Input", "Length", "Tokens", "Latency",
	       "Throughput", "Bandwidth");
	printf("  %-15s-+-%9s-+-%7s-+-%10s-+-%16s-+-%13s\n",
	       "---------------", "---------", "-------",
	       "----------", "----------------", "-------------");

	for (int i = 0; i < N_INPUTS; i++)
		bench_encode(&tok, test_inputs[i].label,
			     test_inputs[i].text, "byte");

	sam3_tokenizer_free(&tok);

	/* ── BPE mode ─────────────────────────────────────────────── */
	printf("\nBPE mode:\n");

	err = sam3_tokenizer_init(&tok);
	if (err != SAM3_OK) {
		fprintf(stderr, "Failed to init tokenizer: %d\n", err);
		return 1;
	}

	double t0 = get_time_ms();
	err = sam3_tokenizer_load_bpe(&tok, BPE_VOCAB_PATH);
	double t1 = get_time_ms();

	if (err != SAM3_OK) {
		printf("  (skipped — BPE vocab not found at %s)\n",
		       BPE_VOCAB_PATH);
		sam3_tokenizer_free(&tok);
		printf("\nDone.\n");
		return 0;
	}

	printf("  Vocab loaded in %.1f ms (%d tokens, %d merges)\n\n",
	       t1 - t0, tok.vocab_size, tok.n_merges);

	printf("  %-15s | %9s | %7s | %10s | %16s | %13s\n",
	       "Input", "Length", "Tokens", "Latency",
	       "Throughput", "Bandwidth");
	printf("  %-15s-+-%9s-+-%7s-+-%10s-+-%16s-+-%13s\n",
	       "---------------", "---------", "-------",
	       "----------", "----------------", "-------------");

	for (int i = 0; i < N_INPUTS; i++)
		bench_encode(&tok, test_inputs[i].label,
			     test_inputs[i].text, "bpe");

	sam3_tokenizer_free(&tok);

	printf("\nDone.\n");
	return 0;
}
