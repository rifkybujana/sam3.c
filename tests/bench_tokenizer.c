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

/* ── Corpus loader ─────────────────────────────────────────────────── */

/*
 * load_corpus - Read a text file into a single buffer and index lines.
 *
 * @path:      Path to UTF-8 text file, one line per entry.
 * @out_buf:   Receives malloc'd buffer holding the whole file.
 * @out_lines: Receives malloc'd array of char* pointing into the buffer.
 *             Each pointer is a null-terminated C string (newlines
 *             replaced with \0 in place).
 * @out_count: Receives number of lines.
 * @out_size:  Receives total file size in bytes.
 *
 * Returns 0 on success, -1 on failure (prints error to stderr).
 * Caller frees *out_buf and *out_lines on success.
 */
static int load_corpus(const char *path, char **out_buf, char ***out_lines,
		       int *out_count, long *out_size)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "error: cannot open corpus '%s'\n", path);
		fprintf(stderr,
			"hint: run 'python3 tests/data/gen_bench_corpus.py'\n");
		return -1;
	}

	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (sz <= 0) {
		fprintf(stderr, "error: corpus '%s' is empty\n", path);
		fclose(f);
		return -1;
	}

	char *buf = malloc((size_t)sz + 1);
	if (!buf) {
		fprintf(stderr, "error: out of memory loading corpus\n");
		fclose(f);
		return -1;
	}
	if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
		fprintf(stderr, "error: short read on corpus '%s'\n", path);
		free(buf);
		fclose(f);
		return -1;
	}
	fclose(f);
	buf[sz] = '\0';

	/* Count lines and replace \n with \0 in place. */
	int n = 0;
	for (long i = 0; i < sz; i++) {
		if (buf[i] == '\n') {
			buf[i] = '\0';
			n++;
		}
	}
	/* Handle missing trailing newline. */
	if (sz > 0 && buf[sz - 1] != '\0')
		n++;

	char **lines = malloc((size_t)n * sizeof(char *));
	if (!lines) {
		fprintf(stderr, "error: out of memory indexing corpus\n");
		free(buf);
		return -1;
	}

	int idx = 0;
	const char *p = buf;
	const char *end = buf + sz;
	while (p < end && idx < n) {
		/* Skip empty lines produced by consecutive newlines. */
		if (*p == '\0') {
			p++;
			continue;
		}
		lines[idx++] = (char *)p;
		p += strlen(p) + 1;
	}

	*out_buf = buf;
	*out_lines = lines;
	*out_count = idx;
	*out_size = sz;
	return 0;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
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

	/* ── Corpus throughput ────────────────────────────────────── */
	const char *corpus_path = (argc > 1)
		? argv[1]
		: "tests/data/bench_corpus.txt";

	char *corpus_buf = NULL;
	char **corpus_lines = NULL;
	int n_lines = 0;
	long corpus_size = 0;

	printf("\n");
	if (load_corpus(corpus_path, &corpus_buf, &corpus_lines,
			&n_lines, &corpus_size) != 0) {
		printf("Done.\n");
		return 1;
	}

	printf("Corpus: %s (%d lines, %ld KB)\n\n",
	       corpus_path, n_lines, corpus_size / 1024);
	printf("  %-12s | %10s | %12s | %12s | %8s\n",
	       "Mode", "Total", "Lines/sec", "Tokens/sec", "MB/s");
	printf("  %-12s-+-%10s-+-%12s-+-%12s-+-%8s\n",
	       "------------", "----------", "------------",
	       "------------", "--------");

	#define PASSES 3
	for (int mode = 0; mode < 2; mode++) {
		struct sam3_tokenizer ctok;
		if (sam3_tokenizer_init(&ctok) != SAM3_OK) {
			fprintf(stderr, "failed to init tokenizer\n");
			free(corpus_buf);
			free(corpus_lines);
			return 1;
		}
		const char *mode_label = "byte-level";
		if (mode == 1) {
			if (sam3_tokenizer_load_bpe(&ctok, BPE_VOCAB_PATH)
			    != SAM3_OK) {
				sam3_tokenizer_free(&ctok);
				continue;
			}
			mode_label = "BPE";
		}

		int32_t ctokens[SAM3_TOKENIZER_CONTEXT_LEN];
		long total_tokens = 0;

		/* Warmup pass. */
		for (int i = 0; i < n_lines; i++) {
			sam3_tokenizer_encode(&ctok, corpus_lines[i],
					      ctokens,
					      SAM3_TOKENIZER_CONTEXT_LEN);
		}

		/* Timed passes; report best. */
		double best_ms = 1e30;
		for (int p = 0; p < PASSES; p++) {
			total_tokens = 0;
			double t0 = get_time_ms();
			for (int i = 0; i < n_lines; i++) {
				total_tokens += sam3_tokenizer_encode(
					&ctok, corpus_lines[i], ctokens,
					SAM3_TOKENIZER_CONTEXT_LEN);
			}
			double t1 = get_time_ms();
			double ms = t1 - t0;
			if (ms < best_ms)
				best_ms = ms;
		}

		double secs = best_ms / 1000.0;
		double lines_per_sec = (double)n_lines / secs;
		double tokens_per_sec = (double)total_tokens / secs;
		double mb_per_sec = (double)corpus_size
				  / (secs * 1024.0 * 1024.0);

		printf("  %-12s | %7.1f ms | %10.0f/s | %10.0f/s | %6.2f MB/s\n",
		       mode_label, best_ms, lines_per_sec,
		       tokens_per_sec, mb_per_sec);

		sam3_tokenizer_free(&ctok);
	}
	#undef PASSES

	free(corpus_lines);
	free(corpus_buf);

	printf("\nDone.\n");
	return 0;
}
