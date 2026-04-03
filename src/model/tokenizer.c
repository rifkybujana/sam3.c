/*
 * src/model/tokenizer.c - BPE tokenizer for CLIP text encoding
 *
 * Implements a BPE tokenizer compatible with the CLIP vocabulary layout.
 * The vocabulary has 49408 slots: bytes_to_unicode tokens at [0-255],
 * their </w> variants at [256-511], merged tokens from [512..49405],
 * and SOT/EOT at [49406-49407]. Initialization creates a byte-level
 * fallback; sam3_tokenizer_load_bpe() loads the real CLIP BPE vocab
 * for production-quality subword tokenization.
 *
 * Key types:  sam3_tokenizer, tok_hash_table
 * Depends on: tokenizer.h
 * Used by:    model/text_encoder.c (future)
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

/* CLIP vocabulary constants */
#define CLIP_VOCAB_SIZE 49408
#define CLIP_SOT_TOKEN  49406
#define CLIP_EOT_TOKEN  49407
#define CLIP_MAX_MERGES 48894	/* 49408 - 256 - 256 - 2 */

/*
 * byte_to_unicode - CLIP's bytes_to_unicode mapping table.
 *
 * Maps each byte value [0-255] to a Unicode codepoint such that all
 * printable ASCII and Latin-1 chars map to themselves, while control
 * bytes and the gap at 0xAD map to U+0100..U+0143. Max codepoint is
 * U+0143, so all values fit in 2-byte UTF-8.
 */
static const uint32_t byte_to_unicode[256] = {
	0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
	0x0108, 0x0109, 0x010A, 0x010B, 0x010C, 0x010D, 0x010E, 0x010F,
	0x0110, 0x0111, 0x0112, 0x0113, 0x0114, 0x0115, 0x0116, 0x0117,
	0x0118, 0x0119, 0x011A, 0x011B, 0x011C, 0x011D, 0x011E, 0x011F,
	0x0120, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
	0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
	0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
	0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F,
	0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
	0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F,
	0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
	0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
	0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
	0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F,
	0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
	0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x0121,
	0x0122, 0x0123, 0x0124, 0x0125, 0x0126, 0x0127, 0x0128, 0x0129,
	0x012A, 0x012B, 0x012C, 0x012D, 0x012E, 0x012F, 0x0130, 0x0131,
	0x0132, 0x0133, 0x0134, 0x0135, 0x0136, 0x0137, 0x0138, 0x0139,
	0x013A, 0x013B, 0x013C, 0x013D, 0x013E, 0x013F, 0x0140, 0x0141,
	0x0142, 0x00A1, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7,
	0x00A8, 0x00A9, 0x00AA, 0x00AB, 0x00AC, 0x0143, 0x00AE, 0x00AF,
	0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
	0x00B8, 0x00B9, 0x00BA, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF,
	0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C4, 0x00C5, 0x00C6, 0x00C7,
	0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE, 0x00CF,
	0x00D0, 0x00D1, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D6, 0x00D7,
	0x00D8, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00DE, 0x00DF,
	0x00E0, 0x00E1, 0x00E2, 0x00E3, 0x00E4, 0x00E5, 0x00E6, 0x00E7,
	0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF,
	0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F5, 0x00F6, 0x00F7,
	0x00F8, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x00FE, 0x00FF,
};

/* Encode a Unicode codepoint to UTF-8. Returns number of bytes written. */
static int utf8_encode(uint32_t cp, char *buf)
{
	if (cp < 0x80) {
		buf[0] = (char)cp;
		return 1;
	}
	if (cp < 0x800) {
		buf[0] = (char)(0xC0 | (cp >> 6));
		buf[1] = (char)(0x80 | (cp & 0x3F));
		return 2;
	}
	/* Not needed for CLIP (max U+0143) but included for safety */
	buf[0] = (char)(0xE0 | (cp >> 12));
	buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
	buf[2] = (char)(0x80 | (cp & 0x3F));
	return 3;
}

/* Convert a byte to its CLIP UTF-8 representation. Returns byte count. */
static int byte_to_utf8(unsigned char b, char *buf)
{
	return utf8_encode(byte_to_unicode[b], buf);
}

/*
 * is_clip_printable - Check if byte is in CLIP's printable range.
 *
 * CLIP considers these byte ranges printable (mapped to chr(byte)):
 * 33-126, 161-172, 174-255
 */
static int is_clip_printable(int b)
{
	return (b >= 33 && b <= 126) ||
	       (b >= 161 && b <= 172) ||
	       (b >= 174 && b <= 255);
}

/*
 * build_bytes_to_unicode - Build the CLIP bytes_to_unicode table.
 *
 * Populates a 256-entry table where each byte maps to a short UTF-8
 * string. Printable bytes map to themselves (as UTF-8), non-printable
 * bytes map to codepoints starting at U+0100.
 */
static void build_bytes_to_unicode(char table[256][5])
{
	int n = 0;

	for (int b = 0; b < 256; b++) {
		int len;
		if (is_clip_printable(b)) {
			len = utf8_encode((uint32_t)b, table[b]);
		} else {
			len = utf8_encode((uint32_t)(256 + n), table[b]);
			n++;
		}
		table[b][len] = '\0';
	}
}

/*
 * build_clip_byte_order - Build the CLIP byte ordering table.
 *
 * CLIP's vocab lists printable bytes first (33-126, 161-172, 174-255)
 * then non-printable bytes (0-32, 127-160, 173). This determines which
 * byte goes at vocab position i for i in [0..255].
 */
static void build_clip_byte_order(int order[256])
{
	int n = 0;
	for (int b = 33; b <= 126; b++)
		order[n++] = b;
	for (int b = 161; b <= 172; b++)
		order[n++] = b;
	for (int b = 174; b <= 255; b++)
		order[n++] = b;
	for (int b = 0; b < 256; b++) {
		if (!is_clip_printable(b))
			order[n++] = b;
	}
}

/* ---- Open-addressing hash table ---- */

struct tok_hash_entry {
	char *key;
	int   value;
	int   key_owned;	/* 1 if key was malloc'd by the table */
};

struct tok_hash_table {
	struct tok_hash_entry *entries;
	int capacity;
	int count;
};

static uint32_t fnv1a_hash(const char *str)
{
	uint32_t h = 2166136261u;
	for (const unsigned char *p = (const unsigned char *)str; *p; p++) {
		h ^= *p;
		h *= 16777619u;
	}
	return h;
}

static struct tok_hash_table *hash_table_create(int min_capacity)
{
	struct tok_hash_table *ht;
	int cap = 64;

	while (cap < min_capacity * 2)
		cap *= 2;

	ht = calloc(1, sizeof(*ht));
	if (!ht)
		return NULL;

	ht->entries = calloc((size_t)cap, sizeof(struct tok_hash_entry));
	if (!ht->entries) {
		free(ht);
		return NULL;
	}
	ht->capacity = cap;
	ht->count = 0;
	return ht;
}

static int hash_table_insert(struct tok_hash_table *ht, const char *key,
			     int value, int copy_key)
{
	uint32_t idx = fnv1a_hash(key) & (uint32_t)(ht->capacity - 1);

	for (;;) {
		struct tok_hash_entry *e = &ht->entries[idx];
		if (!e->key) {
			if (copy_key) {
				e->key = strdup(key);
				if (!e->key)
					return -1;
				e->key_owned = 1;
			} else {
				e->key = (char *)key;
				e->key_owned = 0;
			}
			e->value = value;
			ht->count++;
			return 0;
		}
		if (strcmp(e->key, key) == 0) {
			e->value = value;
			return 0;
		}
		idx = (idx + 1) & (uint32_t)(ht->capacity - 1);
	}
}

/* Returns pointer to value, or NULL if not found */
static int *hash_table_lookup(const struct tok_hash_table *ht,
			      const char *key)
{
	if (!ht || !key)
		return NULL;

	uint32_t idx = fnv1a_hash(key) & (uint32_t)(ht->capacity - 1);

	for (;;) {
		struct tok_hash_entry *e = &ht->entries[idx];
		if (!e->key)
			return NULL;
		if (strcmp(e->key, key) == 0)
			return &e->value;
		idx = (idx + 1) & (uint32_t)(ht->capacity - 1);
	}
}

static void hash_table_free(struct tok_hash_table *ht)
{
	if (!ht)
		return;
	if (ht->entries) {
		for (int i = 0; i < ht->capacity; i++) {
			if (ht->entries[i].key_owned)
				free(ht->entries[i].key);
		}
		free(ht->entries);
	}
	free(ht);
}

/* Build "str_a\x01str_b" key for merge-pair lookup. Caller frees. */
static char *make_pair_key(const char *a, const char *b)
{
	size_t la = strlen(a);
	size_t lb = strlen(b);
	char *key = malloc(la + 1 + lb + 1);
	if (!key)
		return NULL;
	memcpy(key, a, la);
	key[la] = '\x01';
	memcpy(key + la + 1, b, lb);
	key[la + 1 + lb] = '\0';
	return key;
}

/*
 * lookup_pair_rank - Look up merge rank for a symbol pair using a stack buffer.
 *
 * Returns the merge rank (>= 0) or -1 if the pair has no merge rule.
 */
static int lookup_pair_rank(const struct tok_hash_table *ranks,
			    const char symbols[][64], const int *sym_len,
			    int pos, char *pair_buf)
{
	int la = sym_len[pos];
	int lb = sym_len[pos + 1];
	memcpy(pair_buf, symbols[pos], (size_t)la);
	pair_buf[la] = '\x01';
	memcpy(pair_buf + la + 1, symbols[pos + 1], (size_t)lb);
	pair_buf[la + 1 + lb] = '\0';
	int *rank = hash_table_lookup(ranks, pair_buf);
	return rank ? *rank : -1;
}

/* Pre-computed EOT fill for fast memcpy-based padding */
#define E_ CLIP_EOT_TOKEN
static const int32_t eot_pad[SAM3_TOKENIZER_CONTEXT_LEN] = {
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
};
#undef E_

/*
 * pretokenize_next - Scan the next pre-token from text.
 *
 * Matches CLIP's regex pattern: contractions, letter runs, single
 * digits, and symbol runs. Whitespace between tokens is stripped
 * rather than prepended to the following token as in the original
 * CLIP regex (\s+(?=[^\s])). This simplification works because
 * the BPE vocabulary encodes word boundaries via </w> suffixes
 * rather than leading-space tokens, so stripping inter-word
 * whitespace produces equivalent tokenization.
 *
 * Returns 1 if a token was found (sets *start and *len), 0 at end.
 */
static int pretokenize_next(const char **cursor, const char **start, int *len)
{
	const char *p = *cursor;

	/* Skip whitespace */
	while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
		p++;

	if (!*p) {
		*cursor = p;
		return 0;
	}

	/* Check for contractions: 's 't 're 've 'm 'll 'd */
	if (*p == '\'') {
		if (p[1] == 's' || p[1] == 't' || p[1] == 'm' ||
		    p[1] == 'd') {
			*start = p;
			*len = 2;
			*cursor = p + 2;
			return 1;
		}
		if (p[1] == 'r' && p[2] == 'e') {
			*start = p;
			*len = 3;
			*cursor = p + 3;
			return 1;
		}
		if (p[1] == 'v' && p[2] == 'e') {
			*start = p;
			*len = 3;
			*cursor = p + 3;
			return 1;
		}
		if (p[1] == 'l' && p[2] == 'l') {
			*start = p;
			*len = 3;
			*cursor = p + 3;
			return 1;
		}
	}

	/* Letter run: [a-zA-Z]+ */
	if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
		*start = p;
		while (*p && ((*p >= 'a' && *p <= 'z') ||
			      (*p >= 'A' && *p <= 'Z')))
			p++;
		*len = (int)(p - *start);
		*cursor = p;
		return 1;
	}

	/* Single digit: [0-9] */
	if (*p >= '0' && *p <= '9') {
		*start = p;
		*len = 1;
		*cursor = p + 1;
		return 1;
	}

	/* Symbol run: [^\s a-zA-Z0-9]+ */
	*start = p;
	while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
	       !((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) &&
	       !(*p >= '0' && *p <= '9'))
		p++;
	*len = (int)(p - *start);
	*cursor = p;
	return 1;
}

/*
 * bpe_encode_word - BPE-encode one pre-tokenized word.
 *
 * Converts word bytes to bytes_to_unicode strings, appends </w> to the
 * last symbol, then iteratively merges the lowest-rank bigram until no
 * more merges are possible. Final symbols are looked up in encoder_map.
 *
 * Returns number of token IDs written to out_ids, or -1 on error.
 */
#define BPE_MAX_SYMBOLS 128

static int bpe_encode_word(const struct sam3_tokenizer *tok,
			   const char *word, int word_len,
			   int32_t *out_ids, int max_ids)
{
	const struct tok_hash_table *enc =
		(const struct tok_hash_table *)tok->encoder_map;
	const struct tok_hash_table *ranks =
		(const struct tok_hash_table *)tok->merge_rank_map;

	/* Convert each byte to its bytes_to_unicode UTF-8 string */
	char symbols[BPE_MAX_SYMBOLS][64];
	int sym_len[BPE_MAX_SYMBOLS];
	int n_sym = 0;

	for (int i = 0; i < word_len && n_sym < BPE_MAX_SYMBOLS - 1; i++) {
		int nb = byte_to_utf8((unsigned char)word[i], symbols[n_sym]);
		symbols[n_sym][nb] = '\0';
		sym_len[n_sym] = nb;
		n_sym++;
	}

	if (n_sym == 0)
		return 0;

	/* Append </w> to the last symbol */
	int last = n_sym - 1;
	if (sym_len[last] + 4 < (int)sizeof(symbols[0])) {
		memcpy(symbols[last] + sym_len[last], "</w>", 5);
		sym_len[last] += 4;
	}

	/* Stack buffer for pair key lookups (avoids malloc per pair) */
	char pair_buf[132]; /* 64 + 1 + 64 + 1 + padding */

	/* Cache pair ranks — avoids re-hashing unchanged pairs each iteration */
	int pair_rank[BPE_MAX_SYMBOLS];
	for (int i = 0; i < n_sym - 1; i++)
		pair_rank[i] = lookup_pair_rank(ranks, symbols, sym_len,
						i, pair_buf);

	/* Iterative BPE merging */
	while (n_sym > 1) {
		int best_rank = -1;
		int best_pos = -1;

		/* Find lowest-rank bigram from cached ranks */
		for (int i = 0; i < n_sym - 1; i++) {
			if (pair_rank[i] >= 0 &&
			    (best_rank < 0 || pair_rank[i] < best_rank)) {
				best_rank = pair_rank[i];
				best_pos = i;
			}
		}

		if (best_pos < 0)
			break;

		/* Merge symbols[best_pos] and symbols[best_pos + 1] */
		int len_a = sym_len[best_pos];
		int len_b = sym_len[best_pos + 1];
		if (len_a + len_b >= (int)sizeof(symbols[0]))
			break;
		memcpy(symbols[best_pos] + len_a,
		       symbols[best_pos + 1], (size_t)len_b + 1);
		sym_len[best_pos] = len_a + len_b;

		/* Shift remaining symbols and cached ranks left */
		for (int i = best_pos + 1; i < n_sym - 1; i++) {
			memcpy(symbols[i], symbols[i + 1],
			       (size_t)sym_len[i + 1] + 1);
			sym_len[i] = sym_len[i + 1];
			pair_rank[i] = pair_rank[i + 1];
		}
		n_sym--;

		/* Re-lookup only the 2 pairs affected by the merge */
		if (best_pos > 0)
			pair_rank[best_pos - 1] = lookup_pair_rank(
				ranks, symbols, sym_len,
				best_pos - 1, pair_buf);
		if (best_pos < n_sym - 1)
			pair_rank[best_pos] = lookup_pair_rank(
				ranks, symbols, sym_len,
				best_pos, pair_buf);
	}

	/* Look up each final symbol in the encoder */
	int n_out = 0;
	for (int i = 0; i < n_sym && n_out < max_ids; i++) {
		int *id = hash_table_lookup(enc, symbols[i]);
		if (id) {
			out_ids[n_out++] = (int32_t)*id;
		}
	}

	return n_out;
}

#ifdef __aarch64__
/*
 * neon_lower_widen - NEON-accelerated lowercase + byte-to-int32 widening.
 *
 * Loads 16 bytes at a time, lowercases A-Z, widens to int32_t, and stores.
 * Stops when a NUL byte is found or @limit chars are processed. The 16-byte
 * NEON load may read past the NUL terminator within the same page; this is
 * architecturally safe on ARM but trips ASan, hence no_sanitize.
 *
 * Returns number of characters processed (always a multiple of 16 or less
 * if NUL was found).
 */
__attribute__((no_sanitize("address")))
static int neon_lower_widen(const unsigned char *src, int32_t *dst, int limit)
{
	const uint8x16_t v_A = vdupq_n_u8('A');
	const uint8x16_t v_Z = vdupq_n_u8('Z');
	const uint8x16_t v_bit5 = vdupq_n_u8(0x20);
	const uint8x16_t v_zero = vdupq_n_u8(0);
	int i = 0;

	while (i + 16 <= limit) {
		uint8x16_t chunk = vld1q_u8(src + i);

		/* Bail if chunk contains NUL terminator */
		if (vmaxvq_u8(vceqq_u8(chunk, v_zero)))
			break;

		/* Branchless lowercase: set bit 5 if A <= c <= Z */
		uint8x16_t is_upper = vandq_u8(vcgeq_u8(chunk, v_A),
						vcleq_u8(chunk, v_Z));
		chunk = vorrq_u8(chunk, vandq_u8(is_upper, v_bit5));

		/* Widen u8 → u16 → u32 and store */
		uint8x8_t lo8 = vget_low_u8(chunk);
		uint8x8_t hi8 = vget_high_u8(chunk);
		uint16x8_t lo16 = vmovl_u8(lo8);
		uint16x8_t hi16 = vmovl_u8(hi8);

		vst1q_u32((uint32_t *)(dst + i),
			  vmovl_u16(vget_low_u16(lo16)));
		vst1q_u32((uint32_t *)(dst + i + 4),
			  vmovl_u16(vget_high_u16(lo16)));
		vst1q_u32((uint32_t *)(dst + i + 8),
			  vmovl_u16(vget_low_u16(hi16)));
		vst1q_u32((uint32_t *)(dst + i + 12),
			  vmovl_u16(vget_high_u16(hi16)));

		i += 16;
	}

	return i;
}
#endif

/* ---- Public API ---- */

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
	tok->bpe_loaded = 0;
	tok->encoder_map = NULL;
	tok->merge_rank_map = NULL;
	tok->bpe_cache = NULL;

	build_bytes_to_unicode(tok->byte_unicode);

	return SAM3_OK;
}

enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
					const char *path)
{
	gzFile fp = NULL;
	char **merge_a = NULL;
	char **merge_b = NULL;
	char **new_vocab = NULL;
	struct tok_hash_table *enc = NULL;
	struct tok_hash_table *ranks = NULL;
	char line[512];
	int n_merges = 0;
	enum sam3_error err;

	if (!tok || !path || !tok->vocab)
		return SAM3_EINVAL;

	if (tok->bpe_loaded)
		return SAM3_EINVAL;

	fp = gzopen(path, "rb");
	if (!fp)
		return SAM3_EIO;

	/* Skip header line (starts with '"') */
	if (!gzgets(fp, line, (int)sizeof(line))) {
		gzclose(fp);
		return SAM3_EINVAL;
	}

	/* Read merge lines into parallel arrays */
	merge_a = calloc(CLIP_MAX_MERGES, sizeof(char *));
	merge_b = calloc(CLIP_MAX_MERGES, sizeof(char *));
	if (!merge_a || !merge_b) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	while (n_merges < CLIP_MAX_MERGES &&
	       gzgets(fp, line, (int)sizeof(line))) {
		/* Strip trailing newline */
		size_t len = strlen(line);
		while (len > 0 && (line[len - 1] == '\n' ||
				   line[len - 1] == '\r'))
			line[--len] = '\0';

		if (len == 0)
			continue;

		/* Split at first space */
		char *space = strchr(line, ' ');
		if (!space || space == line || *(space + 1) == '\0')
			continue;
		*space = '\0';

		merge_a[n_merges] = strdup(line);
		merge_b[n_merges] = strdup(space + 1);
		if (!merge_a[n_merges] || !merge_b[n_merges]) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		n_merges++;
	}
	gzclose(fp);
	fp = NULL;

	if (n_merges == 0) {
		err = SAM3_EINVAL;
		goto cleanup;
	}

	/* Build new vocabulary: 49408 entries */
	new_vocab = calloc(CLIP_VOCAB_SIZE, sizeof(char *));
	if (!new_vocab) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* [0-255]: bytes_to_unicode strings in CLIP ordering */
	int clip_order[256];
	build_clip_byte_order(clip_order);

	for (int i = 0; i < 256; i++) {
		char buf[4];
		int nb = byte_to_utf8((unsigned char)clip_order[i], buf);
		new_vocab[i] = malloc((size_t)nb + 1);
		if (!new_vocab[i]) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(new_vocab[i], buf, (size_t)nb);
		new_vocab[i][nb] = '\0';
	}

	/* [256-511]: same + "</w>" suffix */
	for (int i = 0; i < 256; i++) {
		size_t base_len = strlen(new_vocab[i]);
		new_vocab[256 + i] = malloc(base_len + 5);
		if (!new_vocab[256 + i]) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(new_vocab[256 + i], new_vocab[i], base_len);
		memcpy(new_vocab[256 + i] + base_len, "</w>", 5);
	}

	/* [512..512+n_merges-1]: concatenation of merge pairs */
	for (int i = 0; i < n_merges; i++) {
		size_t la = strlen(merge_a[i]);
		size_t lb = strlen(merge_b[i]);
		new_vocab[512 + i] = malloc(la + lb + 1);
		if (!new_vocab[512 + i]) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		memcpy(new_vocab[512 + i], merge_a[i], la);
		memcpy(new_vocab[512 + i] + la, merge_b[i], lb);
		new_vocab[512 + i][la + lb] = '\0';
	}

	/* [49406]: <|startoftext|> */
	new_vocab[CLIP_SOT_TOKEN] = strdup("<|startoftext|>");
	/* [49407]: <|endoftext|> */
	new_vocab[CLIP_EOT_TOKEN] = strdup("<|endoftext|>");
	if (!new_vocab[CLIP_SOT_TOKEN] || !new_vocab[CLIP_EOT_TOKEN]) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	/* Build encoder hash table: vocab[i] -> i */
	enc = hash_table_create(CLIP_VOCAB_SIZE);
	if (!enc) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	for (int i = 0; i < CLIP_VOCAB_SIZE; i++) {
		if (!new_vocab[i])
			continue;
		if (hash_table_insert(enc, new_vocab[i], i, 1) < 0) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
	}

	/* Build merge-rank hash table: "tokA\x01tokB" -> rank */
	ranks = hash_table_create(n_merges);
	if (!ranks) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}
	for (int i = 0; i < n_merges; i++) {
		char *pk = make_pair_key(merge_a[i], merge_b[i]);
		if (!pk) {
			err = SAM3_ENOMEM;
			goto cleanup;
		}
		if (hash_table_insert(ranks, pk, i, 0) < 0) {
			free(pk);
			err = SAM3_ENOMEM;
			goto cleanup;
		}
	}
	/* Mark all inserted pair keys as table-owned for cleanup */
	for (int i = 0; i < ranks->capacity; i++) {
		if (ranks->entries[i].key)
			ranks->entries[i].key_owned = 1;
	}

	/* Commit: free old state and install new */
	if (tok->vocab) {
		for (int i = 0; i < tok->vocab_size; i++)
			free(tok->vocab[i]);
		free(tok->vocab);
	}
	free(tok->merge_first);
	free(tok->merge_second);
	hash_table_free(tok->encoder_map);
	hash_table_free(tok->merge_rank_map);

	tok->vocab = new_vocab;
	tok->vocab_size = CLIP_VOCAB_SIZE;
	tok->n_merges = n_merges;
	tok->merge_first = NULL;
	tok->merge_second = NULL;
	tok->encoder_map = enc;
	tok->merge_rank_map = ranks;
	tok->bpe_loaded = 1;
	tok->sot_token = CLIP_SOT_TOKEN;
	tok->eot_token = CLIP_EOT_TOKEN;

	/* Allocate BPE word cache */
	free(tok->bpe_cache);
	tok->bpe_cache = calloc(SAM3_BPE_CACHE_SIZE,
				sizeof(struct sam3_bpe_cache_entry));

	/* Free merge string arrays (vocab took ownership of concat'd strings) */
	for (int i = 0; i < n_merges; i++) {
		free(merge_a[i]);
		free(merge_b[i]);
	}
	free(merge_a);
	free(merge_b);

	return SAM3_OK;

cleanup:
	if (fp)
		gzclose(fp);
	if (new_vocab) {
		for (int i = 0; i < CLIP_VOCAB_SIZE; i++)
			free(new_vocab[i]);
		free(new_vocab);
	}
	hash_table_free(enc);
	hash_table_free(ranks);
	if (merge_a) {
		for (int i = 0; i < n_merges; i++)
			free(merge_a[i]);
		free(merge_a);
	}
	if (merge_b) {
		for (int i = 0; i < n_merges; i++)
			free(merge_b[i]);
		free(merge_b);
	}
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

	if (tok->bpe_loaded) {
		/* CLIP BPE mode */
		char lower[1024];
		int tlen = 0;

		/* Lowercase into stack buffer */
		for (const char *p = text; *p && tlen < (int)sizeof(lower) - 1; p++)
			lower[tlen++] = (char)tolower((unsigned char)*p);
		lower[tlen] = '\0';

		/* Pre-tokenize and BPE-encode each word */
		const char *cursor = lower;
		const char *wstart;
		int wlen;
		int limit = max_tokens - 1;

		while (pretokenize_next(&cursor, &wstart, &wlen)) {
			int32_t word_ids[128];
			int n = 0;

			/*
			 * Check BPE word cache. The cache is a
			 * mutable performance optimization behind
			 * an otherwise const interface.
			 */
			struct sam3_bpe_cache_entry *cache =
				((struct sam3_tokenizer *)tok)->bpe_cache;
			int cached = 0;

			if (cache && wlen < SAM3_BPE_CACHE_MAX_KEY) {
				uint32_t h = 2166136261u;
				for (int ci = 0; ci < wlen; ci++) {
					h ^= (unsigned char)wstart[ci];
					h *= 16777619u;
				}
				int slot = (int)(h & (SAM3_BPE_CACHE_SIZE - 1));
				struct sam3_bpe_cache_entry *ce =
					&cache[slot];
				if (ce->key_len == wlen &&
				    memcmp(ce->key, wstart,
					   (size_t)wlen) == 0) {
					n = ce->n_ids;
					memcpy(word_ids, ce->ids,
					       (size_t)n * sizeof(int32_t));
					cached = 1;
				}
			}

			if (!cached) {
				n = bpe_encode_word(tok, wstart, wlen,
						    word_ids, 128);

				/* Insert into cache */
				if (cache &&
				    wlen < SAM3_BPE_CACHE_MAX_KEY &&
				    n <= SAM3_BPE_CACHE_MAX_IDS) {
					uint32_t h = 2166136261u;
					for (int ci = 0; ci < wlen; ci++) {
						h ^= (unsigned char)wstart[ci];
						h *= 16777619u;
					}
					int slot = (int)(h & (SAM3_BPE_CACHE_SIZE - 1));
					struct sam3_bpe_cache_entry *ce =
						&cache[slot];
					memcpy(ce->key, wstart, (size_t)wlen);
					ce->key[wlen] = '\0';
					ce->key_len = wlen;
					ce->n_ids = n;
					memcpy(ce->ids, word_ids,
					       (size_t)n * sizeof(int32_t));
				}
			}

			for (int i = 0; i < n && pos < limit; i++)
				tokens[pos++] = word_ids[i];
			if (pos >= limit)
				break;
		}

		/* End-of-text token */
		tokens[pos++] = (int32_t)tok->eot_token;
		int n_tokens = pos;

		/* Pad with 0 (CLIP convention) */
		if (pos < max_tokens)
			memset(tokens + pos, 0,
			       (size_t)(max_tokens - pos) * sizeof(int32_t));

		return n_tokens;
	}

	/* Byte-level fallback mode */
	int limit = max_tokens - 2;
	const unsigned char *src = (const unsigned char *)text;
	int i = 0;

#ifdef __aarch64__
	i = neon_lower_widen(src, tokens + pos, limit);
	pos += i;
#endif

	/* Scalar tail (or full path on non-NEON) */
	while (i < limit && src[i]) {
		unsigned char c = src[i];
		/* Branchless ASCII lowercase */
		c |= (unsigned char)(((unsigned)(c - 'A') < 26u) << 5);
		tokens[pos++] = (int32_t)c;
		i++;
	}

	/* End-of-text token */
	tokens[pos++] = (int32_t)tok->eot_token;
	int n_tokens = pos;

	/* Pad remainder with EOT via bulk memcpy */
	int pad = max_tokens - pos;
	if (pad > 0 && pad <= SAM3_TOKENIZER_CONTEXT_LEN) {
		memcpy(tokens + pos, eot_pad,
		       (size_t)pad * sizeof(int32_t));
	} else {
		int32_t eot = (int32_t)tok->eot_token;
		while (pos < max_tokens)
			tokens[pos++] = eot;
	}

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

	hash_table_free(tok->encoder_map);
	tok->encoder_map = NULL;

	hash_table_free(tok->merge_rank_map);
	tok->merge_rank_map = NULL;

	free(tok->bpe_cache);
	tok->bpe_cache = NULL;

	tok->vocab_size = 0;
	tok->n_merges = 0;
	tok->bpe_loaded = 0;
}
