# Tokenizer

The SAM3 tokenizer converts UTF-8 text into a fixed-length sequence of integer
token IDs that the CLIP text encoder consumes. It implements the same
byte-pair encoding (BPE) algorithm used by OpenAI's CLIP model, producing
identical output for the same input text.

## Overview

```
"a photo of a cat"
       |
       v
  +-----------+     +--------------+     +------------+
  | Lowercase |---->| Pre-tokenize |---->| BPE Merge  |----> [49406, 320, 1125, 539, 320, 2368, 49407, 0, ...]
  +-----------+     +--------------+     +------------+
                     splits into          merges byte
                     words by regex       pairs iteratively
```

The output is always `SAM3_TOKENIZER_CONTEXT_LEN` (77) int32 values:

    [SOT] [token] [token] ... [token] [EOT] [pad] [pad] ... [pad]

- **SOT** (49406): start-of-text, always at position 0
- **EOT** (49407): end-of-text, marks the end of meaningful tokens
- **Pad**: 0 in BPE mode, EOT in byte-level fallback mode

## Two Operating Modes

### Byte-level fallback

Available immediately after `sam3_tokenizer_init()` with no external files.
Each input byte maps directly to a token ID equal to its ASCII value
(after lowercasing). Useful for testing and as a fallback when the BPE
vocabulary file is not available.

```
"Cat" -> lowercase -> "cat"
      -> byte values: [99, 97, 116]
      -> tokens: [49406, 99, 97, 116, 49407, 49407, ..., 49407]
                  SOT    c   a   t    EOT    padding
```

### Full CLIP BPE

Activated by loading the vocabulary file via `sam3_tokenizer_load_bpe()`.
This is the production mode that produces tokenization identical to the
original Python CLIP tokenizer.

```
"a cat" -> lowercase -> "a cat"
        -> pre-tokenize: ["a", "cat"]
        -> BPE encode each word
        -> tokens: [49406, 320, 2368, 49407, 0, 0, ..., 0]
                    SOT    a    cat   EOT    padding
```

## Vocabulary Layout

The CLIP vocabulary has 49408 entries arranged in a specific order:

| Range | Count | Content |
|-------|-------|---------|
| 0-255 | 256 | `bytes_to_unicode` tokens — one per byte value, in CLIP order (printable bytes first: 33-126, 161-172, 174-255, then non-printable: 0-32, 127-160, 173) |
| 256-511 | 256 | Same tokens with `</w>` suffix (end-of-word markers) |
| 512-49405 | up to 48894 | BPE merge results — each is the concatenation of a merge pair |
| 49406 | 1 | `<\|startoftext\|>` (SOT) |
| 49407 | 1 | `<\|endoftext\|>` (EOT) |

### bytes_to_unicode

CLIP does not operate on raw bytes. Instead, each byte value is mapped to a
Unicode codepoint via the `bytes_to_unicode` table:

- Printable ASCII and Latin-1 bytes (33-126, 161-172, 174-255) map to
  themselves: byte `65` ('A') maps to codepoint `U+0041` ('A').
- Non-printable bytes (0-32, 127-160, 173) map to codepoints starting at
  `U+0100`: byte `0` maps to `U+0100`, byte `1` to `U+0101`, etc.

This ensures every byte has a visible Unicode representation, which is required
because the BPE merge table uses string-based tokens, not raw byte values.

## Encoding Pipeline (BPE Mode)

### Step 1: Lowercase

The entire input string is lowercased into a stack buffer. CLIP's tokenizer
is case-insensitive for the text encoder.

### Step 2: Pre-tokenize

The lowercased text is split into words using a regex-like scanner that
matches these patterns in priority order:

1. **Contractions**: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
2. **Letter runs**: `[a-zA-Z]+`
3. **Single digits**: `[0-9]` (one at a time)
4. **Symbol runs**: everything else that is not whitespace

Whitespace between words is stripped. This matches CLIP's original regex:
`'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s\w]+`

Example: `"don't stop 3 times!"` splits into:
`["don", "'t", "stop", "3", "times", "!"]`

### Step 3: BPE encode each word

Each pre-tokenized word goes through the BPE merge algorithm:

**3a. Convert to symbols.** Each byte of the word is converted to its
`bytes_to_unicode` UTF-8 string. The last symbol gets a `</w>` suffix
appended to mark the word boundary.

```
"cat" -> symbols: ["c", "a", "t</w>"]
```

**3b. Iterative merging.** Repeatedly find the adjacent symbol pair with the
lowest merge rank (from the merge table) and merge them into one symbol.
Stop when no more mergeable pairs exist.

```
symbols: ["c", "a", "t</w>"]
  merge("c", "a") -> rank 233 exists
  merge("a", "t</w>") -> rank 2146 exists
  best: ("c", "a") at rank 233
  -> ["ca", "t</w>"]

  merge("ca", "t</w>") -> rank 2112 exists
  -> ["cat</w>"]
  no more pairs -> done
```

**3c. Token lookup.** Each final symbol is looked up in the encoder hash
table to get its integer token ID.

```
"cat</w>" -> token ID 2368
```

### Step 4: Assemble output

SOT is placed at position 0, then all word token IDs are concatenated, EOT
is appended, and the remainder is zero-padded to 77 tokens.

## Data Structures

### `sam3_tokenizer`

The main tokenizer struct holds:

- `vocab` — array of 49408 token strings
- `encoder_map` — hash table mapping token string to token ID (for encoding)
- `merge_rank_map` — hash table mapping `"tokA\x01tokB"` to merge rank
  (for BPE merge priority)
- `bpe_cache` — 256-entry direct-mapped cache for word-to-token-IDs results
- `byte_unicode[256][5]` — precomputed bytes_to_unicode UTF-8 strings

### Hash tables

Two open-addressing hash tables with FNV-1a hashing:

- **encoder_map**: `token_string -> token_id` (49408 entries, used for final
  symbol lookup after BPE merging)
- **merge_rank_map**: `"symbolA\x01symbolB" -> rank` (up to 48894 entries,
  used to find the lowest-rank bigram during merging)

## Performance

The tokenizer is optimized for throughput on ARM64:

| Technique | Where | Impact |
|-----------|-------|--------|
| NEON vectorized lowercase + widen | Byte-level encode | 16 chars/iteration instead of 1 |
| Stack-based pair key buffer | BPE merge inner loop | Eliminates malloc/free per bigram check |
| Cached pair ranks | BPE merge loop | Only 2 hash lookups per merge instead of N-1 |
| BPE word cache | BPE encode | Skips O(n^2) merge for repeated words |
| `memcpy` from static const array | EOT padding | Bulk fill instead of scalar loop |
| Branchless ASCII lowercase | Byte-level encode | Bitwise op instead of branch |

Benchmark numbers (Release build, Apple M-series):

```
Byte-level mode:
  short word (3 chars)    ~8 ns     119M enc/s    341 MB/s
  sentence (47 chars)     ~19 ns     52M enc/s   2281 MB/s
  max length (217 chars)  ~16 ns     61M enc/s  12588 MB/s
```

## API

```c
/* Initialize with byte-level fallback vocabulary */
enum sam3_error sam3_tokenizer_init(struct sam3_tokenizer *tok);

/* Load full CLIP BPE vocabulary from gzipped text file */
enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
                                        const char *path);

/* Encode text to token IDs (always 77 output values) */
int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
                          const char *text,
                          int32_t *tokens, int max_tokens);

/* Free all tokenizer resources */
void sam3_tokenizer_free(struct sam3_tokenizer *tok);
```

## Files

- `src/model/tokenizer.h` — public header, struct definition
- `src/model/tokenizer.c` — implementation
- `tests/test_tokenizer.c` — unit tests
- `tests/bench_tokenizer.c` — performance benchmark
- `models/bpe_simple_vocab_16e6.txt.gz` — CLIP BPE vocabulary (not in repo)
