# Tutorial 01 — Building a CLIP BPE Tokenizer in C

This is the first article in a series that walks through how SAM3 is built
from the ground up. We start with the **tokenizer**: the small, self-contained
component that turns a string like `"a photo of a cat"` into a fixed-length
array of integers the rest of the model can consume.

We start here for three reasons:

1. **It has no dependencies.** No tensors, no backends, no model weights. Just
   strings, bytes, and integers. You can build it with `cc` alone.
2. **It exercises every rule in the project's coding standard.** Arena-free
   memory, hash tables, branchless lowercase, SIMD, and a direct-mapped cache.
   By the end you will have seen all of them at work on one small module.
3. **It is verifiable.** We have reference token IDs from the Python CLIP
   tokenizer. At every step you can check your output against
   `tests/test_tokenizer.c` and know you are correct.

The final source is about 1000 lines of C in `src/model/tokenizer.c`. We will
build it twice: first a simplified teaching version that is correct but slow,
then the production version with the optimizations that make it fast.

> **Two audiences, one article.** This tutorial tries to serve two kinds of
> readers at the same time, and you should feel welcome as either one:
>
> - **Programmers** who want to rebuild SAM3's tokenizer from scratch. You
>   should be comfortable reading C code, know what UTF-8 is in broad
>   strokes, and have a C11 compiler handy. The code blocks are meant to be
>   typed into an editor and run; the math boxes explain what those blocks
>   are doing in formal terms.
>
> - **Curious non-programmers** — students, writers, designers, researchers
>   in other fields — who just want to *understand* how a modern AI model
>   turns the sentence you typed into numbers the network can read. You can
>   skim every code block and every math box and still follow the story.
>   Any time we introduce a technical word, you'll find a short sidebar
>   labelled **"In plain English"** underneath that explains the idea with
>   an everyday analogy.
>
> You do **not** need to know what BPE, NEON, hash tables, or pointers are
> going in. Everything is explained as we encounter it. If a paragraph
> feels too dense, look for the plain-English box nearby; it's probably
> saying the same thing in friendlier words.
>
> One more thing: the tutorial is long, but it's meant to be *casual*
> reading, not a textbook. Grab a drink, settle in, and treat the code
> blocks the way a movie fan treats subtitles in a foreign film — look at
> them when you want the detail, look away when you don't.

---

## Table of contents

1. [What a tokenizer does](#1-what-a-tokenizer-does)
2. [Concept — byte-pair encoding in one page](#2-concept--byte-pair-encoding-in-one-page)
3. [Step 1 — The header and skeleton](#3-step-1--the-header-and-skeleton)
4. [Concept — byte-level fallback](#4-concept--byte-level-fallback)
5. [Step 2 — Byte-level encode (teaching version)](#5-step-2--byte-level-encode-teaching-version)
6. [Step 3 — Byte-level encode (production version)](#6-step-3--byte-level-encode-production-version)
7. [Concept — the CLIP vocabulary layout](#7-concept--the-clip-vocabulary-layout)
8. [Step 4 — Building the vocab from a merge file](#8-step-4--building-the-vocab-from-a-merge-file)
9. [Concept — pre-tokenization](#9-concept--pre-tokenization)
10. [Step 5 — `pretokenize_next`](#10-step-5--pretokenize_next)
11. [Concept — the BPE merge loop](#11-concept--the-bpe-merge-loop)
12. [Step 6 — `bpe_encode_word` (teaching version)](#12-step-6--bpe_encode_word-teaching-version)
13. [Step 7 — `bpe_encode_word` (production version)](#13-step-7--bpe_encode_word-production-version)
14. [Step 8 — NEON lowercase](#14-step-8--neon-lowercase)
15. [Step 9 — BPE word cache](#15-step-9--bpe-word-cache)
16. [Step 10 — Putting `encode()` together](#16-step-10--putting-encode-together)
17. [Verification](#17-verification)
18. [What's next](#18-whats-next)

---

## 1. What a tokenizer does

> **In plain English.** Neural networks can't read letters — they can only
> do math on numbers. So before a model like SAM3 can understand the
> sentence `"a photo of a cat"`, someone has to turn every word (or
> word-piece) into a number the model was trained to recognize. That
> someone is the **tokenizer**.
>
> Imagine you're sending a telegram, but the telegraph operator only knows
> a fixed dictionary of about 50 000 entries, each one numbered. To send
> your message you look at your sentence, find which dictionary entries
> match, and hand the operator a list of numbers. When the message arrives
> at the other end, another operator reads the numbers back off the same
> dictionary. A tokenizer is exactly the first operator: it takes text in
> and hands back numbers.
>
> Those numbers are called **token IDs**, and the dictionary is called the
> **vocabulary**. The rest of this article is about how to build the
> operator who does the looking-up efficiently.

The SAM3 text encoder expects inputs of a very specific shape: an `int32`
tensor of length 32 containing token IDs, with the first slot always being
the *start-of-text* token and the end of the meaningful content marked by an
*end-of-text* token. Everything after EOT is padding.

> **In plain English.** "`int32` tensor of length 32" is just a fancy way
> of saying "a list of 32 whole numbers". The model always expects exactly
> 32 numbers, no more and no less — if your sentence is shorter, the empty
> slots get filled with a placeholder; if it's longer, the tail gets
> chopped off. Think of it like a mailing label with 32 fixed boxes: one
> box is always "START", one box is always "END", and whatever text you're
> sending has to fit in between.

```
"a photo of a cat"
        |
        v
+-----------+     +--------------+     +------------+
| Lowercase |---->| Pre-tokenize |---->| BPE encode |---->
+-----------+     +--------------+     +------------+
                                                          |
                                                          v
               [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, ..., 0]
                  ^    ^    ^    ^   ^    ^    ^     ^
                  |    |    |    |   |    |    |     |
                  SOT "a" "photo" "of" "a" "cat" EOT  pad
```

That's the whole job. The tokenizer does not care about the model at all; it
only needs to produce the same sequence of integers that Python's CLIP
tokenizer would produce for the same input, because SAM3's text encoder
weights were trained with those IDs.

> **In plain English.** The model was trained once, long ago, using
> Python's CLIP tokenizer. Every word the model "knows" is associated with
> a specific number from that original training. If *our* tokenizer in C
> produces even slightly different numbers — say, number `2368` for "cat"
> instead of some other value — the model will see a word it's never been
> taught and get confused. So our job is not to invent a new tokenizer;
> our job is to rebuild the exact same one in a new language, byte for
> byte.

We want our function signature to look like:

```c
int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
                          const char *text,
                          int32_t *tokens, int max_tokens);
```

`max_tokens` will always be 32 in production (defined as
`SAM3_TOKENIZER_CONTEXT_LEN`). The function returns the number of meaningful
tokens written including SOT and EOT; the rest of the buffer is padding.

Four properties we need:

- **Deterministic.** Same input, same output, always.
- **Lossless within the vocabulary.** Any UTF-8 string that consists of
  recognized characters will round-trip to a sequence of valid token IDs.
- **Fast.** The text encoder will call this on every prompt. A few microseconds
  matter.
- **No heap churn.** Tokenization runs on the inference hot path, so we should
  not `malloc` per call.

Keep those four in mind. They drive every design decision below.

---

## 2. Concept — byte-pair encoding in one page

Before we write any code, you need a mental model of what BPE actually does.

> **In plain English — the LEGO analogy.** Imagine you build words out of
> LEGO bricks. The smallest possible brick is a single letter — `c`, `a`,
> `t`. You could build every word in English that way, but it takes a lot
> of bricks. Long words like `"tokenization"` would be twelve bricks each,
> and the model would have to look at all twelve to understand the word.
>
> Now imagine a smarter LEGO set: in addition to single-letter bricks, it
> also has pre-assembled chunks like `"ing"`, `"tion"`, `"ization"`, and
> even whole common words like `"cat"` and `"photo"`. Building a word
> becomes a puzzle: you look at the letters you need and ask "what's the
> biggest pre-made chunk I have that matches the start? And then the
> start of what's left over? And so on." Fewer pieces, fewer decisions,
> fewer things for the model to read.
>
> That's **byte-pair encoding (BPE)**. It's a clever way of deciding
> *which* pre-assembled chunks to include in the LEGO set. Decades of
> experience with text have shown that certain letter combinations appear
> over and over (`th`, `ing`, `tion`), and BPE learns these automatically
> from a huge pile of real sentences. The result is a vocabulary of about
> 50 000 chunks ranging from single letters to whole common words.

Given a vocabulary of, say, 50 000 "tokens", BPE is an algorithm that splits
a word into the *smallest number of longest tokens* that are present in the
vocabulary. That sentence is a mouthful, so let's see it.

Suppose our vocabulary contains:

```
"c", "a", "t", "ca", "at", "cat", "c</w>", "a</w>", "t</w>",
"ca</w>", "at</w>", "cat</w>"
```

And we want to tokenize the word `cat`. A naïve character-by-character split
gives three tokens: `c` + `a` + `t`. But the vocabulary also contains the
whole word as a single token `cat</w>`, which is better. How does BPE find it?

By **merging**. BPE is trained offline on a large corpus. Training produces
an ordered list of *merge rules*, each of the form "pair A and B should be
merged into token AB".

> **In plain English.** Think of merge rules as assembly instructions for
> the LEGO set, written in order of priority. Rule number 1 might say:
> "whenever you see an `i` brick next to an `n` brick, replace them with a
> single `in` brick." Rule number 2 says: "whenever you see `t` next to
> `h`, replace them with `th`." The rules keep going, thousands of them,
> and they're sorted by importance — the most useful merges come first.
>
> When you want to build a word, you follow these instructions greedily:
> apply the highest-priority rule that fits, then the next, and so on,
> until no more rules apply. Whatever LEGO pieces you're left with at the
> end are your tokens.

A snippet of CLIP's merge file looks like:

```
i n
t h
a n
...
c a
a t</w>
ca t</w>
```

Each line is a merge rule, and **earlier lines have higher priority**. During
encoding, the algorithm:

1. Starts with each byte of the input mapped to its own symbol (we will see
   exactly how in §7 — for now think "one symbol per byte").
2. Looks at every adjacent pair of symbols.
3. Finds the pair whose merge rule has the lowest rank (i.e. highest priority).
4. Merges that pair into a single symbol.
5. Repeats until no adjacent pair has a merge rule.

For `cat` (ranks below are illustrative — the real numbers depend on where
each rule sits in CLIP's merge file):

```
Start:  [c] [a] [t</w>]
        pair (c,a) has rank R1        <- lowest wins
        pair (a,t</w>) has rank R2
Merge:  [ca] [t</w>]
        pair (ca,t</w>) has rank R3
Merge:  [cat</w>]
        no pairs left. done.
```

The final symbol is looked up in the vocabulary: `cat</w>` → token ID 2368.

Two things to remember:

- **`</w>` is not a separator you add between tokens.** It is a suffix that
  marks "this symbol was at the end of a word". It is why `"cat"` (standalone)
  tokenizes differently from `"cat"` in `"catfish"`.
- **The merge rule's priority matters, not the pair's priority in the
  vocabulary.** Merge rank comes from the *order* of rules in the merge file,
  not from token IDs.

That's all BPE is. The rest of this tutorial is making a computer do it fast.

---

## 3. Step 1 — The header and skeleton

We start from the outside in: the public header. This is the contract every
caller will see, and every later step will flesh it out.

> **In plain English.** A `struct` in C is a labeled box with named slots,
> like a paper form: `Name: ___  Age: ___  Address: ___`. We're about to
> define a `sam3_tokenizer` form with slots for the vocabulary, a couple of
> lookup tables, and a few small integers.
>
> A "header file" (the `.h`) is the menu at a restaurant — it lists what you
> can order without explaining how the kitchen cooks it. The matching `.c`
> file is the kitchen: it contains the actual recipes. Programmers read the
> header first to learn what a piece of code *can do*, then dive into the
> `.c` file only when they want to see *how*.

Create `src/model/tokenizer.h`:

```c
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

#define SAM3_TOKENIZER_CONTEXT_LEN 32	/* SAM3 text encoder length */

struct sam3_tokenizer {
	char **vocab;		/* 49408 token strings */
	int    vocab_size;
	int    sot_token;	/* start-of-text ID (49406) */
	int    eot_token;	/* end-of-text   ID (49407) */
	int    bpe_loaded;	/* 0 = byte fallback, 1 = real BPE */
	char   byte_unicode[256][5];	/* bytes→unicode UTF-8 strings */

	/* filled in later — leave as void* for now */
	void  *encoder_map;	/* string -> token id */
	void  *merge_rank_map;	/* "A\x01B" -> rank (teaching) */
	void  *pair_rank_map;	/* (id_a<<32|id_b) -> rank (production) */
};

enum sam3_error sam3_tokenizer_init(struct sam3_tokenizer *tok);

enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
					const char *path);

int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens);

void sam3_tokenizer_free(struct sam3_tokenizer *tok);

#endif /* SAM3_MODEL_TOKENIZER_H */
```

Three design decisions worth flagging:

**Why are the three map pointers `void *`?** Because their concrete types are
internal hash tables we don't want the header to know about. The header is
the public API; exposing the hash-table struct would leak implementation
details. Inside `tokenizer.c` we cast them to their real type.

**Why both `merge_rank_map` and `pair_rank_map`?** The teaching version in
step 6 uses a string-keyed rank map (`"symA\x01symB" -> rank`), and the
production version in step 7 replaces it with an integer-keyed map
(`(id_a << 32 | id_b) -> rank`). Carrying both fields here lets us show both
code paths without rearranging the struct later. The *real* source in
`src/model/tokenizer.c` only has `pair_rank_map` because it ships only the
production path — once you finish step 7 you can delete `merge_rank_map`.

**Why is `byte_unicode` sized `[256][5]`?** CLIP maps every byte to a Unicode
codepoint. The non-printable bytes land on codepoints in `U+0100..U+0143`,
which encode to 2 UTF-8 bytes. Printable ASCII bytes encode to 1. Two bytes
plus a NUL terminator is 3, and we round up to 5 to leave headroom in case a
future change introduces anything above U+07FF (3 UTF-8 bytes) plus the NUL.
Storing these as fixed-size strings in the struct means we can hand them out
without allocation.

Create `src/model/tokenizer.c` with empty stubs:

```c
#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

enum sam3_error sam3_tokenizer_init(struct sam3_tokenizer *tok)
{
	if (!tok)
		return SAM3_EINVAL;
	memset(tok, 0, sizeof(*tok));
	return SAM3_OK;
}

enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
					const char *path)
{
	(void)tok; (void)path;
	return SAM3_OK;
}

int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens)
{
	(void)tok; (void)text; (void)tokens; (void)max_tokens;
	return 0;
}

void sam3_tokenizer_free(struct sam3_tokenizer *tok)
{
	(void)tok;
}
```

This compiles and does nothing. That's the checkpoint for step 1: the contract
is defined, the skeleton is in place, the rest of the file fills things in.

---

## 4. Concept — byte-level fallback

Production-quality BPE requires loading a 700 KB vocabulary file at startup.
We don't always want to do that — tests run without one, CI doesn't need it,
and during early development you want the tokenizer to just *work* before
you've downloaded anything.

So we build two modes into the same struct:

- **Byte-level fallback** — activated by `sam3_tokenizer_init()` alone.
  Each lowercased input byte becomes a token. Token IDs are the byte values
  themselves (0–255). No vocabulary file required.
- **CLIP BPE** — activated by calling `sam3_tokenizer_load_bpe()` on an
  already-initialized tokenizer. This replaces the simple byte vocabulary
  with the full 49 408-entry CLIP vocabulary and the full merge table.

The `bpe_loaded` field in the struct is the switch. The `encode()` function
checks it and picks the right path.

The byte-level mode will not produce the same token IDs as Python CLIP for
arbitrary text. That's fine — it's not *meant* to. Its job is to exercise the
SOT-text-EOT-padding machinery so the rest of the model can be tested end to
end before we wire up the real vocabulary.

---

## 5. Step 2 — Byte-level encode (teaching version)

> **In plain English.** Imagine a mailroom with 32 slots on the wall. For
> every incoming letter we do the same four things: put a "START" card in
> slot 0, copy the letter one character at a time into the next free slots
> (shouting becomes whispering — uppercase becomes lowercase on the way),
> drop an "END" card right after the last character, and fill every leftover
> slot with more "END" cards so the wall is always full. That's it. No
> dictionary, no clever merging. Just "copy and mark the edges". We'll
> replace this with the fancy BPE version later, but this dumb version is
> useful because it works with zero configuration and makes the rest of the
> code testable today.

Before we write the code, let's stare at what the function actually *is* in
math terms.

We have an input string of L bytes `b_0 b_1 ... b_{L-1}` and we want to
produce a fixed-length output vector of N = 32 integers. Define the
lowercase map on a single byte as

```
    lower(b) = b | 0x20   if 'A' <= b <= 'Z'
    lower(b) = b          otherwise
```

Then byte-level encoding is the piecewise function

```
                  | SOT                if i = 0
    tokens[i]  =  | lower(b_{i-1})     if 1 <= i <= min(L, N-2)
                  | EOT                if i = L+1  (or N-1, whichever comes first)
                  | EOT                otherwise   (padding)
```

Pictorially, for text = `"Cat"` (so L = 3) and N = 32:

```
    position:  0     1     2     3     4     5              31
             +-----+-----+-----+-----+-----+-----+-- ... --+-----+
    tokens:  | SOT | 'c' | 'a' | 't' | EOT | EOT |   ...   | EOT |
             +-----+-----+-----+-----+-----+-----+-- ... --+-----+
               ^     ^     ^     ^     ^
            49406   99    97   116  49407
               |                       |
               +-- start marker        +-- end marker (rest is padding)
```

Three invariants to notice:

- `tokens[0] = SOT` always. No input, no special case, always.
- There is exactly one EOT. Everything after it is literally the same EOT
  value — the padding is just a repeat.
- The output length is fixed at N, regardless of how long or short the
  input is. Short inputs pad; long inputs get truncated.

Now the code. Add this to `tokenizer.c`:

```c
#define CLIP_VOCAB_SIZE 49408
#define CLIP_SOT_TOKEN  49406
#define CLIP_EOT_TOKEN  49407

enum sam3_error sam3_tokenizer_init(struct sam3_tokenizer *tok)
{
	if (!tok)
		return SAM3_EINVAL;

	memset(tok, 0, sizeof(*tok));
	tok->vocab_size = CLIP_VOCAB_SIZE;
	tok->sot_token  = CLIP_SOT_TOKEN;
	tok->eot_token  = CLIP_EOT_TOKEN;
	tok->bpe_loaded = 0;
	return SAM3_OK;
}

int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens)
{
	if (!tok || !text || !tokens || max_tokens < 2)
		return 0;

	int pos = 0;
	tokens[pos++] = tok->sot_token;

	/* Reserve room for EOT */
	int limit = max_tokens - 1;

	for (int i = 0; text[i] && pos < limit; i++) {
		unsigned char c = (unsigned char)text[i];
		if (c >= 'A' && c <= 'Z')
			c += 32;	/* lowercase */
		tokens[pos++] = (int32_t)c;
	}

	tokens[pos++] = tok->eot_token;
	int n = pos;

	/* Pad with EOT */
	while (pos < max_tokens)
		tokens[pos++] = tok->eot_token;

	return n;
}
```

That is a working tokenizer. You can write a test against it right now:

```c
struct sam3_tokenizer tok;
sam3_tokenizer_init(&tok);

int32_t t[32];
int n = sam3_tokenizer_encode(&tok, "cat", t, 32);

assert(n == 5);
assert(t[0] == 49406);        /* SOT */
assert(t[1] == 'c');          /* 99  */
assert(t[2] == 'a');          /* 97  */
assert(t[3] == 't');          /* 116 */
assert(t[4] == 49407);        /* EOT */
```

Every byte becomes its ASCII value; uppercase letters are folded to lowercase
with a branch. Short, obvious, easy to audit.

> **About that `max_tokens - 1`.** We reserve *one* slot for EOT, because the
> teaching version pads the tail with EOT too — if we ran out of room the
> padding loop would still happily write EOTs on top of the unfinished text.
> The production version uses `max_tokens - 2`, reserving one slot for the
> final EOT *and* one for the zero-padding switch. We'll come back to this
> in §6.

So why do we rewrite it? Because this function will be called millions of
times during an evaluation and it has three problems that the production
version fixes.

---

## 6. Step 3 — Byte-level encode (production version)

> **In plain English.** The teaching version works, but it's slow in the
> way that handwriting letters is slow. Imagine you run a mail-order shop
> and every envelope needs the same stamp. You could ink one stamp at a
> time (teaching version), or you could clamp 16 envelopes side-by-side and
> press a single giant stamp across all of them at once (production version).
> Computers have "giant stamps" called SIMD instructions — they take the
> same simple operation (lowercase this byte) and apply it to 16 bytes in
> parallel. On top of that, every `if` statement in a tight loop is a small
> gamble: the CPU has to *guess* which way the branch will go and start
> working before it knows the answer. When it guesses wrong it throws the
> work away and starts over, wasting about 15 cycles. By rewriting the
> uppercase test as pure math — no `if` — we never gamble at all.

Three things are wrong with the teaching version from a performance standpoint:

1. **The lowercase branch is predictable but unpredictable enough** — for
   mixed-case text, the branch predictor will mis-predict on roughly half the
   characters. A mis-prediction on modern cores costs around 15 cycles. We
   can replace the branch with a single bitwise OR that always runs.

2. **The loop processes one byte at a time.** A NEON load can grab 16 bytes
   at once, lowercase them in parallel, and widen them from `u8` to `i32` in
   four vector stores. That's ~16× fewer iterations.

3. **The padding loop runs up to 32 times.** `memcpy` is faster than a scalar
   loop for any non-trivial size, even when the payload is `int32`, because
   libc's `memcpy` uses SIMD internally.

Let's address them one by one. First, branchless lowercase:

```c
unsigned char c = src[i];
c |= (unsigned char)(((unsigned)(c - 'A') < 26u) << 5);
```

This looks dense. Read it from the inside out:

- `c - 'A'` wraps around for any byte outside `['A'..'Z']` and becomes a
  value ≥ 26.
- `(unsigned)... < 26u` gives 1 when `c` is uppercase, 0 otherwise. No branch;
  the compiler emits a compare-and-set.
- `<< 5` shifts that 0/1 to bit 5 (value 32) — the bit that differs between
  uppercase and lowercase ASCII.
- `|=` sets it. If `c` was already lowercase the bit was already set and we
  OR in the same value. Idempotent.

One instruction sequence, no mispredicts. This is rule #5 from the project's
performance rules: *branchless over branchy for simple predicates*.

Now, the bulk padding. We can't `memset` because we want the value 49407, not
a repeating byte pattern. But we *can* pre-compute a full EOT-filled buffer
and `memcpy` from it:

```c
#define E_ CLIP_EOT_TOKEN
static const int32_t eot_pad[SAM3_TOKENIZER_CONTEXT_LEN] = {
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
	E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,E_,
};
#undef E_
```

Now padding is a single `memcpy` from `eot_pad` into the tail. Rule #4: *bulk
memory ops over scalar loops*.

Finally, the NEON path. This is the one non-trivial change.

**Mathematical view.** We want to process 16 bytes of input in one step.
Let `v ∈ (u8)^{16}` be a 16-byte vector loaded from `src + i`. Define
the per-lane operations:

```
    is_upper(v)[k] = 1 if 'A' <= v[k] <= 'Z' else 0       (mask)
    lower(v)[k]    = v[k] | (is_upper(v)[k] << 5)         (apply bit 5)
    widen(v)       = (u32)^{16}   with widen(v)[k] = v[k] (zero-extend)
```

Then one vector step writes `tokens[i..i+16] = widen(lower(v))`, and the
whole loop is just the obvious tiled version of that:

```
      src:                        dst: tokens + pos

      +----+----+----+----+       +----+----+----+----+----+----+----+----+
      | b0 | b1 |....| b15|       | b0 | b1 |....| b15|  0 |  0 |....|  0 |
      +----+----+----+----+       +----+----+----+----+----+----+----+----+
      u8 x 16                     u32 x 16  (4 u32x4 stores)

      step 1: vld1q_u8   --- load 16 bytes into one 128-bit register
      step 2: is_upper   --- (v >= 'A') & (v <= 'Z')
      step 3: lower      --- v |= (is_upper & 0x20)
      step 4: widen x2   --- u8x16 -> u16x8 (lo) + u16x8 (hi)
      step 5: widen x4   --- u16x8 -> u32x4 (lo,lo,hi,hi)
      step 6: 4 vst1q_u32 at dst + 0, +4, +8, +12
```

Five NEON instructions and four stores do the work of roughly sixty scalar
instructions.

We need to:

- load 16 bytes
- lowercase them as a vector
- widen from `u8` to `u32`
- store the four resulting 4-lane `u32` vectors into `tokens[pos..pos+16]`
- stop if we hit a NUL in the middle of a chunk

Here is the helper, dropped in guarded by `__aarch64__`:

```c
#ifdef __aarch64__
#include <arm_neon.h>

__attribute__((no_sanitize("address")))
static int neon_lower_widen(const unsigned char *src, int32_t *dst, int limit)
{
	const uint8x16_t v_A    = vdupq_n_u8('A');
	const uint8x16_t v_Z    = vdupq_n_u8('Z');
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

		/* Widen u8 → u16 → u32 in four stores */
		uint8x8_t  lo8  = vget_low_u8(chunk);
		uint8x8_t  hi8  = vget_high_u8(chunk);
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
```

Two things warrant explanation:

**Why `__attribute__((no_sanitize("address")))`?** Because `vld1q_u8` reads 16
bytes unconditionally. The loop guard `i + 16 <= limit` means we only ever
issue a load whose 16 bytes sit entirely inside the caller's `limit`-byte
region — so on real hardware the load is always in-bounds. The subtlety is
the *scalar* NUL check that follows: we read 16 bytes even if the logical
string ends earlier at a NUL, then detect the NUL and bail. Those trailing
bytes are inside the buffer but past the C-string terminator, and
AddressSanitizer considers any read past a string's used length to be
suspicious. The attribute tells ASan to trust this one function.

> **An important caveat.** `neon_lower_widen` is only safe when the caller
> guarantees the source buffer has at least `limit` bytes of *allocated*
> space, not just `limit` bytes *up to a NUL*. In `sam3_tokenizer_encode`
> that caller is us, and we pass the user's `text` pointer directly — so the
> safety argument reduces to "the user's string was allocated with enough
> headroom for a 16-byte read". Well-formed C strings almost always sit
> inside a buffer that's at least NUL-terminator + 1 byte long, and the
> 16-byte NEON read never crosses a page boundary starting from any valid
> pointer, so in practice this holds. If you ever reuse this helper on
> inputs you didn't allocate yourself, check this argument first.

**Why the vector NUL check?** We cannot allow a NUL to slip into the output,
because the NUL would be written as token 0 — which in CLIP BPE is meaningful
(it's the pad token!). The NEON path explicitly bails out of any 16-byte
chunk that contains a zero byte, leaving the tail for the scalar loop to
handle carefully.

The final production `encode()` function stitches these together:

```c
int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens)
{
	if (!tok || !text || !tokens || max_tokens < 2)
		return 0;

	int pos = 0;
	tokens[pos++] = tok->sot_token;

	/* (BPE branch comes later — for now only byte-level fallback) */

	int limit = max_tokens - 2;
	const unsigned char *src = (const unsigned char *)text;
	int i = 0;

#ifdef __aarch64__
	i = neon_lower_widen(src, tokens + pos, limit);
	pos += i;
#endif

	/* Scalar tail (or the whole thing on non-NEON) */
	while (i < limit && src[i]) {
		unsigned char c = src[i];
		c |= (unsigned char)(((unsigned)(c - 'A') < 26u) << 5);
		tokens[pos++] = (int32_t)c;
		i++;
	}

	tokens[pos++] = tok->eot_token;
	int n_tokens = pos;

	int pad = max_tokens - pos;
	if (pad > 0 && pad <= SAM3_TOKENIZER_CONTEXT_LEN) {
		memcpy(tokens + pos, eot_pad, (size_t)pad * sizeof(int32_t));
	} else {
		int32_t eot = tok->eot_token;
		while (pos < max_tokens)
			tokens[pos++] = eot;
	}

	return n_tokens;
}
```

The benchmark result for this byte-level path on an Apple M-series:

```
short word (3 chars)    ~8 ns     119M enc/s    341 MB/s
sentence (47 chars)     ~19 ns     52M enc/s   2281 MB/s
max length (217 chars)  ~16 ns     61M enc/s  12588 MB/s
```

At 12 GB/s we are bottlenecked on memory bandwidth, not CPU. That is the
ceiling. We cannot make byte-level encoding go faster without changing what
the function does.

Two general principles to note from this step:

- **Compile a simple version first, then replace it.** The simplified version
  acts as a reference implementation. If the NEON version ever disagrees with
  it, the scalar version wins and we debug NEON.
- **Always leave a scalar fallback.** The `#ifdef __aarch64__` guard is not
  optional. On x86-64 CI, on Windows developer boxes, or inside an emulator,
  the fallback is what actually runs. Portability is not optional.

---

## 7. Concept — the CLIP vocabulary layout

Byte-level is enough for testing, but to produce the same token IDs as Python
CLIP we need the full 49 408-entry CLIP vocabulary. Its layout is peculiar
enough to deserve its own section.

```
vocab index    count     contents
-----------    -----     --------
    0 -   255    256     bytes_to_unicode tokens (one per byte, CLIP ordering)
  256 -   511    256     same tokens with </w> suffix
  512 - 49405  48894     merged tokens (one per merge rule)
         49406     1     "<|startoftext|>"
         49407     1     "<|endoftext|>"
```

Three things to understand.

**bytes_to_unicode.** CLIP was originally a Python model. In Python, BPE
training operates on Unicode strings, not raw bytes. But raw bytes contain
values (0–31, 127–160, 173) that have no printable Unicode glyph, and that
Python's regex library handles poorly. CLIP's solution is to define a fixed
mapping from every byte to a *printable* Unicode codepoint:

- Bytes 33–126, 161–172, 174–255 map to themselves (all printable Latin-1).
- The remaining 68 non-printable bytes are assigned to codepoints starting
  at U+0100 in order.

So byte `65` ('A') stays as U+0041 ('A'), but byte `0` becomes U+0100 ('Ā'),
byte `1` becomes U+0101 ('ā'), and so on. Byte `32` (space) is *not*
printable in CLIP's convention — it becomes U+0120.

Why should you care? Because the merge rules in CLIP's merge file are strings
of these mapped characters, not raw bytes. When we compare "are these two
symbols the same" during BPE merging, we need to compare the mapped strings,
not the raw bytes. Our implementation will build a 256-entry lookup table
`byte_unicode[256][5]` giving the UTF-8 encoding of each byte's mapped
codepoint.

**The `</w>` suffix.** BPE trained on whole words needs to know when a symbol
is at the end of a word. The trick is to append `</w>` to the last symbol of
every word before merging begins. So `cat` is actually tokenized as
`[c] [a] [t</w>]`, not `[c] [a] [t]`. This is why the vocabulary has both
`t` (ID 116 in CLIP's order) and `t</w>` (ID 372). They are *different*
tokens.

**Merge rules live in the vocab too.** Every merge rule `A B` produces a new
token `AB` that gets appended to the vocabulary. If the 17th merge rule is
`c a`, then `vocab[512 + 17]` = `"ca"`. This gives us a clean invariant: the
token ID of a merge result is always `512 + rank`, where `rank` is the merge
rule's position in the file. No lookup required.

That invariant will save us a lot of work in step 7.

---

## 8. Step 4 — Building the vocab from a merge file

The CLIP vocab file is a gzipped plain-text file shipped as
`models/bpe_simple_vocab_16e6.txt.gz`. Its format is dead simple:

```
#version: 0.2
i n
t h
a n
e r
...
```

One header line, then one merge rule per line, two tokens separated by a
single space. `load_bpe` reads this file and populates the tokenizer's vocab
array, encoder map, and merge rank map.

**Mathematical view of the vocabulary.** §7 showed this as a table; here is
the same layout as a disjoint union, because that makes the index arithmetic
unambiguous.

```
    V  =  V_byte  ⊔  V_byte_w  ⊔  V_merge  ⊔  { SOT, EOT }

    | V_byte  | = 256                 (bytes_to_unicode chars)
    | V_byte_w| = 256                 (same chars + "</w>")
    | V_merge | = M                   (one per merge rule, M = 48894)
    | V       | = 256 + 256 + M + 2  = 49408

              0 .......... 255  256 ........... 511  512 ........ 512+M-1   49406  49407
            +-------------------+--------------------+----------------------+------+------+
    vocab:  |    V_byte         |     V_byte_w       |      V_merge         | SOT  | EOT  |
            +-------------------+--------------------+----------------------+------+------+
             └── index  i  ──┘   └── index 256+i ──┘  └── index 512+r ────┘
                (byte cp)        (byte cp + "</w>")    (merge rule at rank r)
```

And the fundamental invariant that makes the whole thing fast later on:

```
    vocab[512 + r]  =  concat( merge_a[r], merge_b[r] )
```

Read that carefully. It says the token ID of a merge result is *always*
`512 + r`, where `r` is the merge rule's position in the file. No lookup,
no string building, just an addition. §13 exploits this to drop a whole
hash table out of the BPE inner loop.

### Helpers we'll need

Before we write `load_bpe`, we need two small things: a UTF-8 encoder so we
can materialize the `bytes_to_unicode` table, and a minimal string-keyed
hash table for the encoder and merge-rank maps. The real source has both
earlier in `tokenizer.c`; pretend they were already there when you read the
teaching version below.

```c
/*
 * utf8_encode - Write the UTF-8 bytes for a single Unicode codepoint.
 * Returns the number of bytes written (1..4). No NUL terminator.
 */
static int utf8_encode(int cp, char *out)
{
	if (cp < 0x80) {
		out[0] = (char)cp;
		return 1;
	}
	if (cp < 0x800) {
		out[0] = (char)(0xC0 | (cp >> 6));
		out[1] = (char)(0x80 | (cp & 0x3F));
		return 2;
	}
	if (cp < 0x10000) {
		out[0] = (char)(0xE0 | (cp >> 12));
		out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
		out[2] = (char)(0x80 | (cp & 0x3F));
		return 3;
	}
	out[0] = (char)(0xF0 | (cp >> 18));
	out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
	out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
	out[3] = (char)(0x80 | (cp & 0x3F));
	return 4;
}

/*
 * byte_to_utf8 - CLIP's bytes_to_unicode mapping, UTF-8 encoded.
 * Given a raw byte in [0..255], writes the UTF-8 bytes of the
 * corresponding printable Unicode codepoint and returns the length.
 */
static int byte_to_utf8(int byte, char *out)
{
	int cp;
	if ((byte >= 33  && byte <= 126) ||
	    (byte >= 161 && byte <= 172) ||
	    (byte >= 174 && byte <= 255)) {
		cp = byte;	/* printable: maps to itself */
	} else {
		/* non-printable bytes packed into U+0100..U+0143 */
		static int n_assigned = 0;
		static int assigned[256];
		static int initialized = 0;
		if (!initialized) {
			int n = 0;
			for (int b = 0; b < 256; b++) {
				int printable =
				  (b >= 33  && b <= 126) ||
				  (b >= 161 && b <= 172) ||
				  (b >= 174 && b <= 255);
				if (!printable)
					assigned[b] = 256 + n++;
			}
			n_assigned = n;
			initialized = 1;
		}
		(void)n_assigned;
		cp = assigned[byte];
	}
	return utf8_encode(cp, out);
}
```

> *That static-init trick is fine in a single-threaded teaching example but
> you would not ship it — the real source in `tokenizer.c` builds the table
> once at startup inside `build_bytes_to_unicode` and stores it on the
> tokenizer struct. We use the pattern above to keep this snippet
> self-contained.*

And the hash table that `encoder_map` and `merge_rank_map` will point at.
Open addressing, FNV-1a hash, `int` values, string keys:

```c
struct tok_hash_table {
	char    **keys;		/* NULL = empty slot */
	int      *vals;
	int       capacity;	/* power of 2 */
	int       n_entries;
};

static uint32_t fnv1a(const char *s, int len)
{
	uint32_t h = 2166136261u;
	for (int i = 0; i < len; i++) {
		h ^= (unsigned char)s[i];
		h *= 16777619u;
	}
	return h;
}

static struct tok_hash_table *ht_create(int n_entries)
{
	int cap = 16;
	while (cap < n_entries * 2) cap <<= 1;
	struct tok_hash_table *ht = calloc(1, sizeof(*ht));
	ht->keys = calloc((size_t)cap, sizeof(char *));
	ht->vals = calloc((size_t)cap, sizeof(int));
	ht->capacity = cap;
	return ht;
}

static void ht_insert(struct tok_hash_table *ht, const char *key, int val)
{
	uint32_t mask = (uint32_t)(ht->capacity - 1);
	uint32_t idx  = fnv1a(key, (int)strlen(key)) & mask;
	while (ht->keys[idx]) {
		if (strcmp(ht->keys[idx], key) == 0) {
			ht->vals[idx] = val;
			return;
		}
		idx = (idx + 1) & mask;
	}
	ht->keys[idx] = strdup(key);
	ht->vals[idx] = val;
	ht->n_entries++;
}

static int *ht_lookup(const struct tok_hash_table *ht, const char *key)
{
	if (!ht) return NULL;
	uint32_t mask = (uint32_t)(ht->capacity - 1);
	uint32_t idx  = fnv1a(key, (int)strlen(key)) & mask;
	while (ht->keys[idx]) {
		if (strcmp(ht->keys[idx], key) == 0)
			return &((struct tok_hash_table *)ht)->vals[idx];
		idx = (idx + 1) & mask;
	}
	return NULL;
}
```

`hash_lookup(tok->encoder_map, key)` in the rest of the tutorial is exactly
`ht_lookup(tok->encoder_map, key)`; pick whichever name you like as long as
you use it consistently.

### The teaching load_bpe

Here is a teaching version that focuses on the *shape* of the logic. We will
swap in the real one shortly.

```c
#include <zlib.h>

#define CLIP_MAX_MERGES 48894

enum sam3_error sam3_tokenizer_load_bpe(struct sam3_tokenizer *tok,
					const char *path)
{
	if (!tok || !path)
		return SAM3_EINVAL;

	gzFile fp = gzopen(path, "rb");
	if (!fp)
		return SAM3_EIO;

	/* --- 1. Read every merge rule into two parallel arrays --- */
	char **merge_a = calloc(CLIP_MAX_MERGES, sizeof(char *));
	char **merge_b = calloc(CLIP_MAX_MERGES, sizeof(char *));
	int n_merges = 0;

	char line[512];
	gzgets(fp, line, sizeof(line));	/* skip version header */

	while (n_merges < CLIP_MAX_MERGES &&
	       gzgets(fp, line, sizeof(line))) {
		/* Strip trailing newline */
		size_t len = strlen(line);
		while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
			line[--len] = '\0';
		if (len == 0) continue;

		char *space = strchr(line, ' ');
		if (!space) continue;
		*space = '\0';

		merge_a[n_merges] = strdup(line);
		merge_b[n_merges] = strdup(space + 1);
		n_merges++;
	}
	gzclose(fp);

	/* --- 2. Build the 49408-entry vocab --- */
	char **vocab = calloc(CLIP_VOCAB_SIZE, sizeof(char *));

	/* [0..255]: bytes_to_unicode in CLIP's printable-first order. */
	int clip_order[256];
	build_clip_byte_order(clip_order);

	for (int i = 0; i < 256; i++) {
		char buf[4];
		int nb = byte_to_utf8(clip_order[i], buf);
		vocab[i] = malloc(nb + 1);
		memcpy(vocab[i], buf, nb);
		vocab[i][nb] = '\0';
		/* Also populate tok->byte_unicode for the raw byte value,
		 * which BPE encoding will need in §12.                  */
		memcpy(tok->byte_unicode[clip_order[i]], buf, nb);
		tok->byte_unicode[clip_order[i]][nb] = '\0';
	}

	/* [256..511]: same strings with </w> appended */
	for (int i = 0; i < 256; i++) {
		size_t base = strlen(vocab[i]);
		vocab[256 + i] = malloc(base + 5);
		memcpy(vocab[256 + i], vocab[i], base);
		memcpy(vocab[256 + i] + base, "</w>", 5);
	}

	/* [512..512+n_merges-1]: concatenation of each merge pair */
	for (int i = 0; i < n_merges; i++) {
		size_t la = strlen(merge_a[i]);
		size_t lb = strlen(merge_b[i]);
		vocab[512 + i] = malloc(la + lb + 1);
		memcpy(vocab[512 + i], merge_a[i], la);
		memcpy(vocab[512 + i] + la, merge_b[i], lb);
		vocab[512 + i][la + lb] = '\0';
	}

	/* [49406], [49407]: the special tokens */
	vocab[CLIP_SOT_TOKEN] = strdup("<|startoftext|>");
	vocab[CLIP_EOT_TOKEN] = strdup("<|endoftext|>");

	/* --- 3. Build encoder_map: vocab string -> token id --- */
	struct tok_hash_table *enc = ht_create(CLIP_VOCAB_SIZE);
	for (int i = 0; i < CLIP_VOCAB_SIZE; i++) {
		if (vocab[i])
			ht_insert(enc, vocab[i], i);
	}

	/* --- 4. Build merge_rank_map: "A\x01B" -> merge rank --- */
	struct tok_hash_table *mr = ht_create(n_merges);
	for (int i = 0; i < n_merges; i++) {
		char key[512];
		int la = (int)strlen(merge_a[i]);
		int lb = (int)strlen(merge_b[i]);
		if (la + 1 + lb + 1 > (int)sizeof(key)) continue;
		memcpy(key, merge_a[i], la);
		key[la] = '\x01';
		memcpy(key + la + 1, merge_b[i], lb);
		key[la + 1 + lb] = '\0';
		ht_insert(mr, key, i);
	}

	/* --- 5. Free the old byte-level vocab, if any --- */
	if (tok->vocab) {
		for (int i = 0; i < tok->vocab_size; i++)
			free(tok->vocab[i]);
		free(tok->vocab);
	}

	/* --- 6. Install everything on the struct --- */
	tok->vocab          = vocab;
	tok->encoder_map    = enc;
	tok->merge_rank_map = mr;
	tok->bpe_loaded     = 1;

	/* --- 7. Drop the scratch merge-pair arrays --- */
	for (int i = 0; i < n_merges; i++) {
		free(merge_a[i]);
		free(merge_b[i]);
	}
	free(merge_a);
	free(merge_b);

	return SAM3_OK;
}
```

Notice what we did *not* do here: we didn't check for allocation failure, we
didn't roll back on error, and we didn't handle malformed merge files. All
three are present in `src/model/tokenizer.c` behind `goto cleanup`. The
teaching version leaves them out so the happy path reads top-to-bottom
without interruption; add them before you ship.

The `build_clip_byte_order` helper is short and worth its own function because
the ordering is not obvious:

```c
static int is_clip_printable(int b)
{
	return (b >= 33  && b <= 126) ||
	       (b >= 161 && b <= 172) ||
	       (b >= 174 && b <= 255);
}

static void build_clip_byte_order(int order[256])
{
	int n = 0;
	/* Printable bytes come first, in ascending byte value order */
	for (int b = 33;  b <= 126; b++) order[n++] = b;
	for (int b = 161; b <= 172; b++) order[n++] = b;
	for (int b = 174; b <= 255; b++) order[n++] = b;
	/* Then the non-printable bytes */
	for (int b = 0; b < 256; b++)
		if (!is_clip_printable(b))
			order[n++] = b;
}
```

If you mess this up — say, you put the bytes in `0..255` order — your
`vocab[0]` will be `"!"` instead of the non-printable placeholder, every
token ID will be off by some amount, and nothing will match Python CLIP.
Don't mess it up.

The production `sam3_tokenizer_load_bpe` in `src/model/tokenizer.c` does the
same thing with three differences: (a) it skips `merge_rank_map` entirely
and builds the integer-pair `pair_rank_map` from step 7 directly; (b) it
commits changes atomically so a mid-load failure does not leave the
tokenizer in a half-loaded state (see `goto cleanup` in the real source);
(c) it does a full roll-back on `ENOMEM` or a malformed merge file. The
teaching version above skips all of that to keep the happy path legible.

---

## 9. Concept — pre-tokenization

> **In plain English.** Think about slicing a pizza before handing it out.
> If you try to eat the whole pie in one bite, you'll choke. Same story
> here: BPE can't chew a whole sentence at once — it needs bite-sized
> "words" to work on. Pre-tokenization is the pizza cutter. It walks along
> the sentence from left to right and snips off one chunk at a time: a run
> of letters, a contraction like `'t`, a single digit, or a clump of
> punctuation. Whitespace gets thrown away (it's the crumbs). Each chunk
> then gets its own little BPE session, and the results are stitched back
> together at the end.

BPE operates on *words*, not on entire sentences. Before we can merge
anything, we need to split the input into word-like chunks. CLIP's Python
tokenizer uses the regex

```
's|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s\w]+
```

which we translate into a hand-written scanner. In priority order the
patterns are:

1. **Contractions** (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`) — matched
   first so that `"it's"` becomes `["it", "'s"]`, not `["it'", "s"]`.
2. **Letter runs** (`[a-zA-Z]+`) — one or more consecutive letters.
3. **Single digits** (`[0-9]`) — note the "+" is absent: 123 tokenizes as
   three separate words, which is how CLIP handles numbers.
4. **Symbol runs** (`[^\s\w]+`) — everything else that isn't whitespace.

Whitespace between words is **stripped**, not attached to the next word.
CLIP's original regex attached leading whitespace as `\s+(?=[^\s])`, but we
get equivalent output by stripping inter-word whitespace because the BPE
vocab uses `</w>` suffixes instead of leading-space tokens to mark word
boundaries.

Worked example: `"Don't stop, 3 times!"`

```
Input:      D o n ' t   s t o p ,   3   t i m e s !
After lowercasing:
           d o n ' t   s t o p ,   3   t i m e s !

Pre-tokenization walk:
  "d"..."don"     -> letter run "don"
  "'t"            -> contraction "'t"
  " "             -> skip whitespace
  "stop"          -> letter run "stop"
  ","             -> symbol run ","
  " "             -> skip whitespace
  "3"             -> single digit "3"
  " "             -> skip whitespace
  "times"         -> letter run "times"
  "!"             -> symbol run "!"

Words: ["don", "'t", "stop", ",", "3", "times", "!"]
```

Each of those words will be BPE-encoded independently, and their token IDs
will be concatenated into the output.

---

## 10. Step 5 — `pretokenize_next`

We implement the scanner as a stateful function that returns one word at a
time. This keeps the caller in control of when to stop (e.g. when the output
buffer is full).

**Mathematical view.** The scanner is a function

```
    next :  cursor  -->  (class, start, len, cursor')
```

that advances the cursor past any whitespace, then matches the *longest*
prefix of the remaining string against one of five regular languages, in
priority order:

```
    T_contraction = { 's, 't, 're, 've, 'm, 'll, 'd }     priority 1
    T_letter      = [a-zA-Z]+                              priority 2
    T_digit       = [0-9]        (exactly one digit)       priority 3
    T_symbol      = [^\s\w]+                               priority 4
    end-of-input  = ""                                     priority 5
```

The decision tree at each cursor position:

```
              +----------------+
              |   *p == NUL?   |-- yes --> return 0  (EOF)
              +----------------+
                     | no
                     v
              +----------------+
              | *p is space?   |-- yes --> skip, loop
              +----------------+
                     | no
                     v
              +----------------+
              |  *p == '\''?   |-- yes --> try contraction
              +----------------+           (fall through on fail)
                     | no / fail
                     v
              +----------------+
              | *p is letter?  |-- yes --> longest [a-zA-Z]+ run
              +----------------+
                     | no
                     v
              +----------------+
              | *p is digit?   |-- yes --> single digit
              +----------------+
                     | no
                     v
              +----------------+
              |  else: symbol  |---------> longest [^\s\w]+ run
              +----------------+
```

Priority order matters: contractions must be tried *before* letter runs,
otherwise `"it's"` would match as the letter run `"it"` followed by the
symbol run `"'"` followed by the letter run `"s"` — three tokens instead of
two.

```c
/*
 * pretokenize_next - Scan the next pre-token from text.
 *
 * Returns 1 if a token was found (sets *start and *len), 0 at end.
 */
static int pretokenize_next(const char **cursor, const char **start, int *len)
{
	const char *p = *cursor;

	/* 1. Skip whitespace */
	while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
		p++;
	if (!*p) {
		*cursor = p;
		return 0;
	}

	/* 2. Contractions: 's 't 're 've 'm 'll 'd */
	if (*p == '\'') {
		if (p[1]=='s' || p[1]=='t' || p[1]=='m' || p[1]=='d') {
			*start = p; *len = 2; *cursor = p + 2;
			return 1;
		}
		if ((p[1]=='r' && p[2]=='e') ||
		    (p[1]=='v' && p[2]=='e') ||
		    (p[1]=='l' && p[2]=='l')) {
			*start = p; *len = 3; *cursor = p + 3;
			return 1;
		}
	}

	/* 3. Letter run: [a-zA-Z]+ */
	if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
		*start = p;
		while ((*p >= 'a' && *p <= 'z') ||
		       (*p >= 'A' && *p <= 'Z'))
			p++;
		*len = (int)(p - *start);
		*cursor = p;
		return 1;
	}

	/* 4. Single digit */
	if (*p >= '0' && *p <= '9') {
		*start = p; *len = 1; *cursor = p + 1;
		return 1;
	}

	/* 5. Symbol run: everything else non-whitespace */
	*start = p;
	while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
	       !((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) &&
	       !(*p >= '0' && *p <= '9'))
		p++;
	*len = (int)(p - *start);
	*cursor = p;
	return 1;
}
```

Usage:

```c
const char *cursor = lowered_text;
const char *word;
int wlen;
while (pretokenize_next(&cursor, &word, &wlen)) {
	/* encode this one word */
}
```

Two things worth noting about this function:

**It is production-ready as-is.** There is no "simplified" version of the
scanner; we only need to get the priority order right. Scanner code is
rarely a hot spot in tokenization (BPE is), so we optimize elsewhere.

**It returns pointers into the input, not copies.** No `strdup`, no `malloc`,
no temporary buffers. The caller sees `(start, len)` and can compute over
the string in place. Rule #1: *no allocations in hot paths*.

---

## 11. Concept — the BPE merge loop

> **In plain English.** Remember the LEGO analogy from §2? Here's where we
> actually *do* the building. We lay out the word as a row of single-brick
> letters: `c a t</w>`. Then we flip through our rulebook — thousands of
> "whenever you see X next to Y, snap them together" instructions sorted
> by importance. We scan the row, find the most important pair that has
> any rule at all, snap those two bricks into one fused brick, and repeat.
> Eventually no rule fits anywhere, and we stop. The bricks left on the
> table are our tokens. A child could do this by hand; a computer does it
> a few million times per second.

Once we have a word, we need to apply BPE to it. The algorithm is short:

```
symbols = [bytes_to_unicode(b) for b in word_bytes]
symbols[-1] += "</w>"

loop:
  find the pair (symbols[i], symbols[i+1]) whose merge rule has
      the lowest rank (or stop if no pair has any rank at all)
  symbols[i] = merge(symbols[i], symbols[i+1])
  remove symbols[i+1]
repeat until no more merges are possible

token_ids = [vocab_lookup(s) for s in symbols]
```

A naïve implementation does this with strings: each symbol is a C string,
each pair is looked up in a `string -> rank` hash table, and merges
concatenate strings.

That works, but it is slower than necessary for three reasons:

1. **String comparison is not free.** Even with a fast hash function we pay
   one or two memory loads per character of each key.
2. **Merges allocate.** Concatenating two C strings into a new one means
   `malloc` + `memcpy` + `free`, twice per merge.
3. **Pair ranks are recomputed every iteration.** When we merge positions
   `i` and `i+1`, the pairs at `i-1, i` and `i, i+1` are the only ones whose
   ranks change. But the naïve loop re-looks-up *all* pairs.

We will write the string version first to see the shape, then rewrite it.

---

## 12. Step 6 — `bpe_encode_word` (teaching version)

**Mathematical view of one word through BPE.** Let `w = b_0 b_1 ... b_{k-1}`
be the word bytes, and let

```
    μ :  (symbol, symbol)  -->  ℕ ∪ { ∞ }
```

be the merge-rank function, where `μ(A, B)` is the position of the rule
`A B` in CLIP's merge file, or `∞` if no such rule exists. Lower rank wins.

We start from the initial symbol sequence — one character per byte, with
`"</w>"` appended to the last symbol:

```
    S_0  =  [ u(b_0),  u(b_1),  ... ,  u(b_{k-1}) + "</w>" ]
```

(where `u` is the `bytes_to_unicode` map.) At each step `t`, define the best
adjacent pair:

```
    i*  =  argmin_{0 <= i < |S_t|-1}  μ( S_t[i], S_t[i+1] )
    r*  =  μ( S_t[i*], S_t[i*+1] )
```

Transition rule:

```
    if r* = ∞:     S* = S_t               (no mergeable pair → stop)
    else:          S_{t+1}  =  S_t with positions (i*, i*+1)
                              replaced by the single symbol S_t[i*] ⊕ S_t[i*+1]
```

Iterate until the fixed point `S*`. Then the output token IDs are

```
    out[j]  =  encoder_map( S*[j] )
```

For the word `"cat"` (k = 3), the trajectory looks like:

```
                 i*=0                     i*=0
    S_0  =  [c, a, t</w>]  --->  S_1 = [ca, t</w>]  --->  S_2 = [cat</w>]
             └─┬─┘ └──┬──┘              └────┬────┘              └────┬────┘
               R1    R2                      R3                      (no pairs)
              ^^^^^                          ^^^^
              best                           best

    out  =  [ encoder_map("cat</w>") ]  =  [ 2368 ]
```

(`R1`, `R2`, `R3` are the ranks of the three candidate pairs — their actual
values depend on CLIP's merge file order. What matters is the algorithm
picks the lowest, then re-evaluates, then stops when no rule applies.)

Here is the straightforward implementation. It builds a dynamic array of
C-string symbols, loops while merges are possible, and returns an array of
token IDs.

```c
#define MAX_SYMBOLS 128

/*
 * Teaching version: uses strings everywhere. Produces correct output but
 * allocates inside the loop and rescans all pairs each iteration.
 */
static int bpe_encode_word_simple(const struct sam3_tokenizer *tok,
				  const char *word, int word_len,
				  int32_t *out_ids, int max_ids)
{
	/* --- 1. Build initial symbols: one per byte, </w> on the last --- */
	char *sym[MAX_SYMBOLS];
	int n = 0;

	for (int i = 0; i < word_len && n < MAX_SYMBOLS - 1; i++) {
		const char *mapped = tok->byte_unicode[(unsigned char)word[i]];
		sym[n++] = strdup(mapped);
	}
	if (n == 0) return 0;

	/* Append </w> to the last symbol */
	{
		size_t base = strlen(sym[n-1]);
		char *with = malloc(base + 5);
		memcpy(with, sym[n-1], base);
		memcpy(with + base, "</w>", 5);
		free(sym[n-1]);
		sym[n-1] = with;
	}

	/* --- 2. Iterative merge loop --- */
	while (n > 1) {
		int best_rank = -1;
		int best_pos  = -1;

		/* Scan every adjacent pair */
		for (int i = 0; i < n - 1; i++) {
			char key[256];
			snprintf(key, sizeof(key), "%s\x01%s", sym[i], sym[i+1]);
			int *rank = ht_lookup(tok->merge_rank_map, key);
			if (rank && (best_rank < 0 || *rank < best_rank)) {
				best_rank = *rank;
				best_pos  = i;
			}
		}

		if (best_pos < 0) break;	/* no mergeable pair */

		/* Merge positions best_pos and best_pos+1 */
		size_t la = strlen(sym[best_pos]);
		size_t lb = strlen(sym[best_pos + 1]);
		char *merged = malloc(la + lb + 1);
		memcpy(merged,       sym[best_pos],     la);
		memcpy(merged + la,  sym[best_pos + 1], lb);
		merged[la + lb] = '\0';

		free(sym[best_pos]);
		free(sym[best_pos + 1]);
		sym[best_pos] = merged;

		/* Shift tail left */
		for (int i = best_pos + 1; i < n - 1; i++)
			sym[i] = sym[i + 1];
		n--;
	}

	/* --- 3. Look up each final symbol's token ID --- */
	int n_out = (n < max_ids) ? n : max_ids;
	for (int i = 0; i < n_out; i++) {
		int *id = ht_lookup(tok->encoder_map, sym[i]);
		out_ids[i] = id ? *id : 0;
	}

	/* Clean up */
	for (int i = 0; i < n; i++)
		free(sym[i]);
	return n_out;
}
```

This works. Hooked up to a complete tokenizer it will produce the correct
CLIP token IDs. You could ship it and pass every correctness test in
`tests/test_tokenizer.c`:

```c
ASSERT_EQ(tokens[1], 3306);	/* "hello" */
ASSERT_EQ(tokens[2], 1002);	/* "world" */
```

So why not stop here?

Count the heap operations for one merge in a 5-symbol word:

- `snprintf` into a stack key: fine.
- `ht_lookup`: one FNV-1a hash over ~8 bytes, one or two `strcmp`s: fine.
- **`malloc` for the merged string.** Every merge.
- **`free` for the two old strings.** Every merge.

For a 6-character word that BPE merges down to 1 token, we do 5 merges. Each
merge does ~1 malloc and 2 frees. That's 15 heap operations for one word.
At 100 words per prompt, that's 1500 heap operations per `encode()` call, and
the heap allocator is a global synchronization point we do not want to touch
on every call.

Also count the pair rank lookups: for a word of `n` symbols, the first pass
does `n-1` lookups, after one merge we do `n-2` lookups, and so on. Total
O(n²) lookups per word, every one of which builds a string key.

Both of these go away in the production version.

---

## 13. Step 7 — `bpe_encode_word` (production version)

Here is the punchline of the whole tutorial. Four optimizations applied
together:

1. **Work in integer IDs from the start.** Convert each input byte to a
   vocab index once, and track symbols as `int32_t` values for the rest of
   the function. Strings never enter the merge loop.
2. **Use an int-pair hash map for rank lookups.** The key is
   `(uint64_t)id_a << 32 | id_b`, an unambiguous 64-bit integer. Hashing and
   comparing two 64-bit integers is faster than hashing and comparing strings.
3. **Skip the encoder-map lookup for merge results.** Because of the
   invariant `vocab[512 + rank] = merge_result`, the ID of a merged symbol is
   trivially `512 + rank`. No lookup needed.
4. **Cache pair ranks in a parallel array.** When we merge positions `i` and
   `i+1`, only the ranks at `i-1` and `i` change. The rest are still valid.
   Re-look-up only those two.

With these changes, the merge loop allocates nothing and does only two hash
lookups per merge regardless of word length.

**Mathematical view of the optimizations.** We are replacing four
expensive operations with four cheap ones:

```
    Optimization            Before                  After
    ----------------------------------------------------------------
    1. symbols as IDs       S_t : seq<string>       I_t : seq<int32>
    2. pair key             "A\x01B"                κ(a,b) = (a<<32)|b
    3. merge result         encoder_map lookup      512 + rank  (direct)
    4. pair rank cache      rescan every step       update only 2 slots
```

Let `I_t` be the integer symbol sequence at step `t`, and let `R_t` be a
parallel array of cached pair ranks:

```
    R_t[i]  =  pair_rank_map( κ(I_t[i], I_t[i+1]) )       for 0 <= i < |I_t|-1
```

At each step, pick `i* = argmin_i R_t[i]` (ignoring `-1` sentinels for
"no rule"), set

```
    I_{t+1}[i*]  =  512 + R_t[i*]                 (!!!  no lookup)
```

and re-compute *only* `R[i*-1]` and `R[i*]`, because those are the two
pairs whose left/right symbol just changed. Everything else in `R` is still
correct.

Visualized for `|I_t| = 5`, with a merge chosen at `i* = 2`:

```
   before the merge:

       I_t:       [ I0   I1   I2   I3   I4 ]
                   └─┬┘ └─┬┘ └─┬┘ └─┬┘
       R_t:          R0   R1   R2   R3            (pair ranks,
                                                   R2 is the minimum)
                                ^^
                              chosen

   after the merge at i* = 2:

       I_{t+1}:   [ I0   I1   M    I4         ]       M = 512 + R2
                   └─┬┘ └──┬──┘ └─┬┘
       R_{t+1}:      R0    R1'    R2'                 R0: unchanged
                           ^^^^  ^^^^                 R1', R2': re-looked-up
                           new    new                 (2 hash calls total)
```

Complexity, per word of `n` bytes with `m` merges applied:

```
                            | hash lookups  |  heap ops  |  string ops
    ------------------------+---------------+------------+-------------
    Teaching (§12)          |    O(n · m)   |   O(m)     |   O(m)
    Production (§13)        |  (n-1) + 2·m  |     0      |     0
```

For `n = 12`, `m = 4` (the `"tokenization"` example below): 11 + 8 = 19
lookups, vs. 11 + 10 + 9 + 8 = 38 for the teaching version, and zero heap
allocations vs. twelve `strdup`s plus four `malloc`s plus eight `free`s.

### Setup: the byte→vocab-index table

First, we need to know which vocab slot each *raw* byte sits in. Because
CLIP puts printable bytes first, byte `'a'` (97) is *not* at `vocab[97]` —
it's at `vocab[64]`, because `'a'` is the 65th printable byte (counting from
33). We already computed this ordering in step 4 as `clip_order[]`; we just
invert it into `byte_to_vocab[]`:

```c
uint8_t byte_to_vocab[256];
for (int i = 0; i < 256; i++)
	byte_to_vocab[clip_order[i]] = (uint8_t)i;
```

Stored on the tokenizer as `tok->byte_to_vocab[256]`. Lookup is a single
array index.

### Setup: the int-pair rank map

We replace the string-keyed `merge_rank_map` with a new structure that uses
64-bit integer keys. Open addressing, same FNV-style multiplicative hash:

```c
struct pair_rank_map {
	uint64_t *keys;		/* UINT64_MAX = empty slot */
	int32_t  *vals;		/* merge rank */
	int       capacity;	/* power of 2 */
};

static struct pair_rank_map *pair_map_create(int n_entries)
{
	int cap = 16;
	while (cap < n_entries * 2) cap <<= 1;

	struct pair_rank_map *m = calloc(1, sizeof(*m));
	m->keys = malloc((size_t)cap * sizeof(uint64_t));
	m->vals = malloc((size_t)cap * sizeof(int32_t));
	for (int i = 0; i < cap; i++) m->keys[i] = UINT64_MAX;
	m->capacity = cap;
	return m;
}

static inline int pair_map_lookup(const struct pair_rank_map *m, uint64_t key)
{
	uint64_t h   = key * 0x9E3779B97F4A7C15ull;   /* Knuth */
	uint32_t mask = (uint32_t)(m->capacity - 1);
	uint32_t idx = (uint32_t)(h >> 32) & mask;
	for (;;) {
		uint64_t k = m->keys[idx];
		if (k == key)        return m->vals[idx];
		if (k == UINT64_MAX) return -1;
		idx = (idx + 1) & mask;
	}
}
```

`UINT64_MAX` as the empty sentinel is safe because no valid pair key can be
that large — both halves are vocab indices in `[0, 49407]`.

`load_bpe` builds this map during startup: for each merge rule, it looks
up both halves in the *string-keyed* encoder hash table to find their
vocab IDs, constructs the 64-bit key, and inserts with value = rank.

That's the setup. Now the encode loop.

### The merge loop

```c
#define BPE_MAX_SYMBOLS 128

static int bpe_encode_word(const struct sam3_tokenizer *tok,
			   const char *word, int word_len,
			   int32_t *out_ids, int max_ids)
{
	const struct pair_rank_map *prm = tok->pair_rank_map;
	const uint8_t *b2v = tok->byte_to_vocab;

	/* --- 1. Initial symbols are vocab IDs in [0..255] --- */
	int32_t sym[BPE_MAX_SYMBOLS];
	int n = 0;
	for (int i = 0; i < word_len && n < BPE_MAX_SYMBOLS - 1; i++)
		sym[n++] = (int32_t)b2v[(unsigned char)word[i]];
	if (n == 0) return 0;

	/* </w> suffix: base vocab slot + 256 is the </w> variant */
	sym[n - 1] += 256;

	/* --- 2. Cache the rank of every adjacent pair --- */
	int pair_rank[BPE_MAX_SYMBOLS];
	for (int i = 0; i < n - 1; i++) {
		uint64_t k = ((uint64_t)(uint32_t)sym[i] << 32) |
			     (uint32_t)sym[i + 1];
		pair_rank[i] = pair_map_lookup(prm, k);
	}

	/* --- 3. Iterative merging --- */
	while (n > 1) {
		int best_rank = -1, best_pos = -1;

		for (int i = 0; i < n - 1; i++) {
			if (pair_rank[i] >= 0 &&
			    (best_rank < 0 || pair_rank[i] < best_rank)) {
				best_rank = pair_rank[i];
				best_pos  = i;
			}
		}
		if (best_pos < 0) break;

		/* Merged symbol ID is 512 + rank, by the vocab invariant */
		sym[best_pos] = 512 + best_rank;

		/* Shift symbols and pair_rank left by one */
		for (int i = best_pos + 1; i < n - 1; i++) {
			sym[i] = sym[i + 1];
			pair_rank[i] = pair_rank[i + 1];
		}
		n--;

		/* Re-lookup only the 2 pairs affected by the merge */
		if (best_pos > 0) {
			uint64_t k =
			  ((uint64_t)(uint32_t)sym[best_pos - 1] << 32) |
			  (uint32_t)sym[best_pos];
			pair_rank[best_pos - 1] = pair_map_lookup(prm, k);
		}
		if (best_pos < n - 1) {
			uint64_t k =
			  ((uint64_t)(uint32_t)sym[best_pos] << 32) |
			  (uint32_t)sym[best_pos + 1];
			pair_rank[best_pos] = pair_map_lookup(prm, k);
		}
	}

	/* --- 4. Copy out --- */
	int n_out = (n < max_ids) ? n : max_ids;
	memcpy(out_ids, sym, (size_t)n_out * sizeof(int32_t));
	return n_out;
}
```

Compare this to the teaching version:

| | Teaching | Production |
|---|---|---|
| Symbol type | `char *` | `int32_t` |
| Allocations per merge | 1 malloc + 2 free | 0 |
| Pair rank lookups per merge | `n-1` | 2 |
| Hash key build | `snprintf` (~16 bytes) | 1 shift + 1 OR |
| Merge result lookup | hash table | `512 + rank` |

The only heap memory ever touched in this function is `sym[]` and
`pair_rank[]`, both on the stack. The only hash lookups are through the
flat `pair_rank_map` arrays, which are cache-friendly linear probes on a
power-of-two capacity.

There is one subtlety worth pointing out: **the `n-1` pair ranks live in
`pair_rank[0..n-2]`, and when we merge at `best_pos` we also need to drop
`pair_rank[best_pos]` from the array** (because that pair no longer exists).
The shift loop `for (i = best_pos+1; i < n-1; i++) { sym[i] = sym[i+1];
pair_rank[i] = pair_rank[i+1]; }` handles this, and then we re-lookup the
two neighbours of the merge point. Get these indices wrong and you will
silently produce slightly-different token IDs for multi-merge words; the
tests in `test_clip_contraction` (which tokenizes `it's` and expects
`[49406, 585, 568, 49407]`) will catch it immediately.

### Why this is faster

For the word `"tokenization"` (12 characters → ~4 BPE merges) the production
version executes roughly:

- 12 byte-to-vocab lookups (table index)
- 11 initial pair-rank lookups (one each)
- 4 merge iterations, each doing 2 lookups
- 0 allocations

Total hash lookups: 11 + 8 = 19. Total allocations: 0. The teaching version
would have done 11 + 10 + 9 + 8 = 38 lookups just for pair ranks, 4 mallocs,
and 8 frees.

This is rule #2 (*don't compute what you can cache*) and rule #1 (*no
allocations in hot paths*) working together.

---

## 14. Step 8 — NEON lowercase

The BPE path needs lowercased text just like the byte-level path does, but
it needs a *lowercased copy* of the input, not the widened `int32_t` form.
That's a second, slightly different NEON helper:

```c
#ifdef __aarch64__
__attribute__((no_sanitize("address")))
static int neon_lower_copy(const unsigned char *src, char *dst, int limit)
{
	const uint8x16_t v_A    = vdupq_n_u8('A');
	const uint8x16_t v_Z    = vdupq_n_u8('Z');
	const uint8x16_t v_bit5 = vdupq_n_u8(0x20);
	const uint8x16_t v_zero = vdupq_n_u8(0);
	int i = 0;

	while (i + 16 <= limit) {
		uint8x16_t chunk = vld1q_u8(src + i);
		if (vmaxvq_u8(vceqq_u8(chunk, v_zero)))
			break;
		uint8x16_t is_upper = vandq_u8(vcgeq_u8(chunk, v_A),
					       vcleq_u8(chunk, v_Z));
		chunk = vorrq_u8(chunk, vandq_u8(is_upper, v_bit5));
		vst1q_u8((unsigned char *)(dst + i), chunk);
		i += 16;
	}
	return i;
}
#endif
```

The body is the same branchless-lowercase idiom we saw in `neon_lower_widen`,
but instead of four widening stores it does one `u8` store. In the production
`encode()` function we call this before entering the BPE loop to fill a
stack-allocated `char lower[1024]` buffer.

Why 1024? Because `SAM3_TOKENIZER_CONTEXT_LEN` is 32, and a 1024-byte input
is already longer than anything that will fit in 32 tokens. Anything longer
than that gets truncated anyway, so we never lose data.

---

## 15. Step 9 — BPE word cache

> **In plain English.** You've seen someone tie their shoelaces thousands
> of times and by now they don't even think about it — their fingers
> remember. That's what we're building for the tokenizer. The first time
> it encodes the word `"person"` it has to go through the whole BPE
> rigmarole: look up bricks, check rules, snap them together, one step at
> a time. But the *second* time the word shows up (and in real prompts it
> shows up a lot) we'd love to just say "oh, that one — here are the
> tokens" and skip everything. That's a cache: a small lookup table
> mapping word → answer. We give it 1024 slots, and each word lands in
> one slot based on a fingerprint (a hash). If the slot already has the
> same word, we reuse the answer. If not, we do the real work and
> overwrite the slot. Simple, tiny, and it turns the slowest case in the
> tokenizer into a memcpy.

There's one more optimization left, and it's the biggest single win of all.
Look at a realistic workload: the SAM3 text encoder tokenizes prompts from
user input. In a segmentation session the same prompts appear repeatedly:
`"person"`, `"car"`, `"a cat"`, `"a dog"`. Many prompts share common English
words: `"a"`, `"the"`, `"of"`. And even within a single prompt, words repeat.

Encoding the word `"person"` to `[2533]` is fast — a few hundred nanoseconds
— but it's still dozens of times slower than a hash lookup. If we cache the
result we can skip the whole BPE loop for repeat words.

We use a direct-mapped cache: power-of-two bucket count, one entry per
bucket, hash determines the bucket, collisions evict. No chaining. No
allocation. 1024 buckets × 80 bytes per entry = 80 KB of RAM for the entire
cache — cheap.

**Mathematical view.** Let `K = 2^10 = 1024` be the cache size. For a word
`w` of length `ℓ`, define

```
    h(w)     =  FNV1a(w)                     32-bit hash
    slot(w)  =  h(w) mod K  =  h(w) & (K-1)  because K is a power of 2
```

A cache entry at slot `s` holds `(key_len, key[64], n_ids, ids[16])`, and
the cache hit predicate is

```
    hit(w, s)   iff   cache[s].key_len == ℓ  ∧  cache[s].key == w  (byte-eq)
```

On hit, we `memcpy` the cached IDs out and skip the merge loop entirely.
On miss, we run `bpe_encode_word`, write the result through to `cache[s]`
(overwriting whatever was there), and return.

```
     input word "person"
          │
          v                                 +--------------------+
      +-------+       slot = h("person")    |  cache[slot]       |
      | FNV1a |----------& (K-1) ---------->| key_len | key[64]  |
      +-------+                             | n_ids   | ids[16]  |
                                            +--------------------+
                                                │
                                                v
                                        +--------------+
                                        | key matches? |
                                        +--------------+
                                         / yes     no  \
                                        v              v
                                  memcpy(ids)    bpe_encode_word
                                  skip merge      + write-through
                                    loop
```

Cost comparison per word, for a word of `n` bytes:

```
    Cache hit   :  FNV1a hash (n) + memcmp (n) + memcpy (m)   ≈ O(n + m)
    Cache miss  :  everything in §13                           ≈ O(n log n)
```

Any workload with even modest word repetition is dominated by the hit path.

```c
#define SAM3_BPE_CACHE_SIZE    1024	/* power of 2 */
#define SAM3_BPE_CACHE_MAX_IDS 16
#define SAM3_BPE_CACHE_MAX_KEY 64

struct sam3_bpe_cache_entry {
	char    key[SAM3_BPE_CACHE_MAX_KEY];
	int     key_len;
	int32_t ids[SAM3_BPE_CACHE_MAX_IDS];
	int     n_ids;
};
```

Cache lookup is inlined into the encode loop:

```c
/* Hash the word */
uint32_t h = 2166136261u;
for (int ci = 0; ci < wlen; ci++) {
	h ^= (unsigned char)wstart[ci];
	h *= 16777619u;
}
int slot = (int)(h & (SAM3_BPE_CACHE_SIZE - 1));
struct sam3_bpe_cache_entry *ce = &cache[slot];

int cached = 0;
if (ce->key_len == wlen && memcmp(ce->key, wstart, wlen) == 0) {
	n = ce->n_ids;
	memcpy(word_ids, ce->ids, n * sizeof(int32_t));
	cached = 1;
}

if (!cached) {
	n = bpe_encode_word(tok, wstart, wlen, word_ids, 128);
	if (n <= SAM3_BPE_CACHE_MAX_IDS) {
		/* Write-through: overwrite whatever was in this slot */
		memcpy(ce->key, wstart, wlen);
		ce->key[wlen] = '\0';
		ce->key_len = wlen;
		ce->n_ids   = n;
		memcpy(ce->ids, word_ids, n * sizeof(int32_t));
	}
}
```

Two things make this correct:

- **We check `key_len` before `memcmp`.** A hash collision between two words
  of different lengths would otherwise read past the end of one of them. If
  lengths don't match, the cache misses — no harm done.
- **We bound the word length at 64 bytes** (`SAM3_BPE_CACHE_MAX_KEY`). Longer
  words skip the cache entirely. This keeps entries small, and very long
  words are rare enough that it doesn't matter.

This is rule #7 in action: *cache results of expensive pure functions*.
`bpe_encode_word` is deterministic and side-effect-free (except for this
very cache), so caching it is always safe.

The cache lives in a heap-allocated array owned by the tokenizer
(`tok->bpe_cache`), zeroed at startup so every slot has `key_len == 0` and
therefore can never falsely match a real word.

---

## 16. Step 10 — Putting `encode()` together

We now have all the pieces. The final `sam3_tokenizer_encode` function looks
like this:

```c
int sam3_tokenizer_encode(const struct sam3_tokenizer *tok,
			  const char *text,
			  int32_t *tokens, int max_tokens)
{
	if (!tok || !text || !tokens || max_tokens < 2)
		return 0;

	int pos = 0;
	tokens[pos++] = tok->sot_token;

	if (tok->bpe_loaded) {
		/* --- CLIP BPE path --- */
		char lower[1024];
		int tlen = 0;
		const int lim = sizeof(lower) - 1;

#ifdef __aarch64__
		tlen = neon_lower_copy((const unsigned char *)text, lower, lim);
#endif
		while (tlen < lim && text[tlen]) {
			unsigned char c = (unsigned char)text[tlen];
			c |= (unsigned char)(((unsigned)(c - 'A') < 26u) << 5);
			lower[tlen++] = (char)c;
		}
		lower[tlen] = '\0';

		const char *cursor = lower;
		const char *wstart;
		int wlen;
		int limit = max_tokens - 1;
		struct sam3_bpe_cache_entry *cache = tok->bpe_cache;

		while (pretokenize_next(&cursor, &wstart, &wlen)) {
			int32_t word_ids[128];
			int n = 0;

			/* Cacheable words: hash into direct-mapped cache */
			if (wlen > 0 && wlen < SAM3_BPE_CACHE_MAX_KEY) {
				uint32_t h = 2166136261u;
				for (int ci = 0; ci < wlen; ci++) {
					h ^= (unsigned char)wstart[ci];
					h *= 16777619u;
				}
				int slot = (int)(h &
					  (SAM3_BPE_CACHE_SIZE - 1));
				struct sam3_bpe_cache_entry *ce =
					&cache[slot];

				if (ce->key_len == wlen &&
				    memcmp(ce->key, wstart, wlen) == 0) {
					n = ce->n_ids;
					memcpy(word_ids, ce->ids,
					       n * sizeof(int32_t));
				} else {
					n = bpe_encode_word(tok, wstart,
							    wlen, word_ids,
							    128);
					if (n <= SAM3_BPE_CACHE_MAX_IDS) {
						memcpy(ce->key, wstart, wlen);
						ce->key[wlen] = '\0';
						ce->key_len = wlen;
						ce->n_ids   = n;
						memcpy(ce->ids, word_ids,
						       n * sizeof(int32_t));
					}
				}
			} else {
				n = bpe_encode_word(tok, wstart, wlen,
						    word_ids, 128);
			}

			for (int i = 0; i < n && pos < limit; i++)
				tokens[pos++] = word_ids[i];
			if (pos >= limit) break;
		}

		tokens[pos++] = tok->eot_token;
		int n_tokens = pos;

		/* CLIP pads with 0, not EOT */
		if (pos < max_tokens)
			memset(tokens + pos, 0,
			       (max_tokens - pos) * sizeof(int32_t));
		return n_tokens;
	}

	/* ---------- byte-level fallback path ---------- */
	/* (exactly what we wrote in step 3) */
}
```

One last quirk to notice: **CLIP BPE mode pads with zero, but byte-level
mode pads with EOT.** This is because the real CLIP text encoder was trained
with zero padding, so we must match it, but the byte-level fallback is a
testing mode and EOT padding is more informative in test output. Don't mix
these up.

---

## 17. Verification

Correctness first. `tests/test_tokenizer.c` has reference token IDs from the
Python CLIP tokenizer for a handful of inputs. If your implementation
matches, you're done:

```c
/* "hello world" */
ASSERT_EQ(n, 4);
ASSERT_EQ(tokens[0], 49406);   /* SOT */
ASSERT_EQ(tokens[1], 3306);    /* hello */
ASSERT_EQ(tokens[2], 1002);    /* world */
ASSERT_EQ(tokens[3], 49407);   /* EOT */

/* "a photo of a cat" */
ASSERT_EQ(n, 7);
ASSERT_EQ(tokens[0], 49406);
ASSERT_EQ(tokens[1], 320);     /* a */
ASSERT_EQ(tokens[2], 1125);    /* photo */
ASSERT_EQ(tokens[3], 539);     /* of */
ASSERT_EQ(tokens[4], 320);     /* a */
ASSERT_EQ(tokens[5], 2368);    /* cat */
ASSERT_EQ(tokens[6], 49407);

/* "it's" - tests contraction handling */
ASSERT_EQ(tokens[1], 585);     /* it */
ASSERT_EQ(tokens[2], 568);     /* 's */

/* "A dog" - tests case insensitivity */
ASSERT_EQ(tokens[1], 320);     /* a */
ASSERT_EQ(tokens[2], 1929);    /* dog */
```

If any of these disagree, the bug is almost always in one of four places:

- **`clip_order[]` / `byte_to_vocab[]`** — CLIP's printable-first ordering
  is easy to get subtly wrong. Check `vocab[0]` and `vocab[64]`.
- **`</w>` suffix** — forgetting to append it to the last symbol, or
  appending it to *every* symbol, breaks everything.
- **Pair priority** — "lowest rank wins", not "highest". Merge rules earlier
  in the file have *higher* priority, which means *lower* numeric rank.
- **Padding value** — CLIP BPE pads with 0, not with EOT. If your tests
  disagree on `tokens[7..]`, this is it.

Performance second. Build a Release binary (no sanitizers):

```
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j bench_tokenizer
./bench_tokenizer
```

On an Apple M-series you should see numbers in the vicinity of:

```
Byte-level mode:
  short word (3 chars)    ~8 ns     119M enc/s    341 MB/s
  sentence (47 chars)     ~19 ns     52M enc/s   2281 MB/s
  max length (217 chars)  ~16 ns     61M enc/s  12588 MB/s
```

A sanitized debug build will be 5–20× slower. Don't panic if your first
benchmark is horrible — rerun it with `Release`. The performance rules in
`CLAUDE.md` exist partly because it is dangerously easy to micro-optimize
code under ASan and find out later you didn't actually help.

---

## 18. What's next

You now have a working, fast, byte-exact CLIP BPE tokenizer in about 1000
lines of C. Every rule in the project's coding standard has appeared at
least once:

| Rule | Where it showed up |
|---|---|
| No allocs in hot paths | `bpe_encode_word`, scanner returning pointers |
| Don't compute what you can cache | Pair rank cache inside the merge loop |
| Don't scan data twice | Pre-tokenizer returns `(start, len)` directly |
| Bulk memory ops over scalar loops | `eot_pad` `memcpy`, `neon_lower_*` |
| Branchless for simple predicates | `c \|= ((c - 'A') < 26u) << 5` |
| SIMD for byte-level bulk work | `neon_lower_widen`, `neon_lower_copy` |
| Cache results of expensive pure functions | 1024-entry BPE word cache |
| Benchmark without sanitizers | `cmake -DCMAKE_BUILD_TYPE=Release` |

The tokenizer is also a good template for the modules we'll build next in
this series, because it teaches the three habits that show up in every
later component:

1. **Build the simplest correct version first.** You can't optimize what you
   can't verify. The teaching versions of `encode()` and `bpe_encode_word`
   are not wasted work — they are the ground truth you regression-test
   against.
2. **Replace the slow parts in place.** We didn't rewrite the whole module
   to add NEON; we dropped `neon_lower_copy` into one spot and left a scalar
   fallback for everything else. Same for the cache. Same for integer IDs.
3. **Let the hardware and the data structures drive the design.** The
   `512 + rank = merge_id` invariant is not a clever hack; it's the
   consequence of laying out the vocab such that merges naturally get a
   dense range. When the data layout cooperates with the algorithm, the
   code gets shorter, not longer.

In the next tutorial we pick up the next component in the pipeline: the
`.sam3` weight format and the mmap-based weight loader. We will see the
same three habits applied to a very different problem — reading 600 MB
of model weights into memory without a single user-space copy.

**Further reading.**

- The full source lives at `src/model/tokenizer.h` and
  `src/model/tokenizer.c`.
- The reference documentation (data-structure view, API signatures) is at
  [`docs/tokenizer.md`](../tokenizer.md).
- The reference implementation from OpenAI:
  <https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py>. Our
  C version should produce byte-identical output to this Python code for
  any input of fewer than 32 tokens.
- The BPE algorithm was introduced for NMT by Sennrich, Haddow & Birch
  (2015): <https://arxiv.org/abs/1508.07909>. The paper is five pages and
  worth reading once.

See you in tutorial 02.
