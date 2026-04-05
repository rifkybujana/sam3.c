# Text Encoder: Tokenizer + Text Transformer

The text encoder converts a short phrase into a fixed-length sequence of
d_model=256 embeddings that condition the fusion transformer. It consists of
a CLIP-style BPE tokenizer, a 24-layer text transformer at width=1024, and a
final linear projection to the model dimension.

## Files

- Tokenizer: `reference/sam3/sam3/model/tokenizer_ve.py`
- Transformer: `reference/sam3/sam3/model/text_encoder_ve.py`
- BPE merges: `reference/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- Assembled in: `reference/sam3/sam3/model_builder.py:500-509`

## Pipeline

```
"a red apple"
      |
      v
   clean (lowercase + whitespace collapse)
      |
      v
   regex pre-tokenize + BPE merge
      |
      v
   [49406, 320, 736, 3055, 49407]   -- SOT, "a", "red", "apple", EOT
      |
      v
   pad to context_length = 32
      |
      v
   (B, 32) int64 token IDs
      |
      v
   TextTransformer
     token_embedding     : (B, 32, 1024)
     + positional_embed  : (B, 32, 1024)
     24 x ResidualAttentionBlock with causal mask
     ln_final            : (B, 32, 1024)
      |
      v
   resizer: Linear(1024, 256)
      |
      v
   text_memory: (32, B, 256)   -- seq-first
```

## Tokenizer (`tokenizer_ve.py:130-256`)

`SimpleTokenizer` is a straight port of OpenAI CLIP's BPE tokenizer. SAM3
loads it with `context_length=32` (not 77) and `clean="lower"`.

**Cleaning** (`_clean_lower`, `tokenizer_ve.py:87-89`):
- `ftfy.fix_text` (fix mojibake)
- `html.unescape` twice
- Collapse whitespace runs to single space, strip, lowercase.

**Pre-tokenization regex** (`tokenizer_ve.py:159-162`):

```
<start_of_text>|<end_of_text>|
's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
```

Contractions, then letter runs, then single digits, then symbol runs. (The
regex uses Unicode property classes `\p{L}` = letter, `\p{N}` = number.)

**Byte-to-unicode mapping** (`tokenizer_ve.py:31-55`):
The 256 byte values are mapped to 256 distinct printable Unicode codepoints.
Printable ASCII/Latin-1 bytes map to themselves; the remaining 68 non-printable
bytes map to codepoints at 256 (0x100), 257, 258, ... This lets BPE operate
on string tokens without special-casing whitespace or control bytes.

**BPE merge** (`tokenizer_ve.py:170-206`):
Standard iterative pair merging. Each word has `</w>` appended to its last
character as the end-of-word marker. Find the bigram with the lowest merge
rank and merge it, repeat until no mergeable pair remains. A per-tokenizer
`self.cache` memoizes word-to-merged-string results.

**Vocabulary layout** (`tokenizer_ve.py:144-154`):

| Range    | Count | Contents                                         |
|----------|-------|--------------------------------------------------|
| 0-255    | 256   | `bytes_to_unicode` tokens (no end marker)        |
| 256-511  | 256   | Same tokens with `</w>` suffix                   |
| 512-49405 | 48894 | BPE merge results                               |
| 49406    | 1     | `<start_of_text>` (SOT)                          |
| 49407    | 1     | `<end_of_text>` (EOT)                            |

Total vocab size: **49408**.

**Output assembly** (`tokenizer_ve.py:227-255`):
Each input string becomes `[SOT_ID] + encoded_ids + [EOT_ID]` and is
right-padded with zeros to 32. If the sequence exceeds 32 tokens, it is
truncated and the last slot is forced to EOT_ID:

```python
result = torch.zeros(B, 32, dtype=torch.long)
for i, tokens in enumerate(all_tokens):
    if len(tokens) > 32:
        tokens = tokens[:32]
        tokens[-1] = 49407
    result[i, :len(tokens)] = torch.tensor(tokens)
```

## Text Transformer (`text_encoder_ve.py:166-252`)

Standard CLIP `TextTransformer` with SAM3-specific settings from
`model_builder.py:500-509`:

| Parameter         | Value   |
|-------------------|---------|
| context_length    | 32      |
| vocab_size        | 49408   |
| width             | 1024    |
| heads             | 16      |
| layers            | 24      |
| mlp_ratio         | 4.0     |
| pool_type         | "none"  |
| output_tokens     | True    |
| use_ln_post       | True    |
| no_causal_mask    | False   |
| act_layer         | GELU    |

**Embedding** (`text_encoder_ve.py:197-198`):
- `token_embedding = nn.Embedding(49408, 1024)`
- `positional_embedding = nn.Parameter(torch.empty(32, 1024))` (learned, not sine)

**Causal mask** (`text_encoder_ve.py:222-228`):
Upper-triangular additive mask filled with `-inf`, so position *t* can only
attend to positions ≤ *t*:

```python
mask = torch.full((32, 32), float("-inf"))
mask.triu_(1)   # zero on and below diagonal, -inf above
```

**ResidualAttentionBlock** (`text_encoder_ve.py:15-89`):

```python
# Pre-norm self-attention
x = q_x + attention(ln_1(q_x), attn_mask=causal)
# Pre-norm MLP
x = x + mlp(ln_2(x))
```

Each block:
- `nn.MultiheadAttention(d_model=1024, heads=16, batch_first=True)`
- MLP: `Linear(1024, 4096) → GELU → Linear(4096, 1024)` (mlp_ratio=4.0 → 4096)
- `LayerNorm` (eps=1e-5) before attention and before MLP
- `LayerScale` disabled (`ls_init_value=None` → `nn.Identity`)

**Final projection (`text_encoder_ve.py:210`):**
`ln_final = nn.LayerNorm(1024)` applied to the output of block 23.

**Pooling (`text_encoder_ve.py:244`):**
With `pool_type="none"`, no pooled output is produced — the full token
sequence is returned. The `text_projection` parameter (`(1024, 512)` in
stock CLIP) is unused for SAM3; projection to d_model=256 is handled outside
the transformer.

## VETextEncoder Wrapper (`text_encoder_ve.py:255-330`)

Wraps `TextTransformer` and adds a projection layer that drops the width from
1024 to d_model=256.

```python
self.encoder = TextTransformer(
    context_length=32, vocab_size=49408, width=1024,
    heads=16, layers=24, output_tokens=True, use_ln_post=True,
)
self.resizer = nn.Linear(1024, 256)
```

**Forward pass** (`text_encoder_ve.py:288-330`):

```python
tokenized = self.tokenizer(text, context_length=32).to(device)   # (B, 32)
text_attention_mask = (tokenized != 0).bool()                    # (B, 32), True = real
inputs_embeds = self.encoder.token_embedding(tokenized)          # (B, 32, 1024)
_, text_memory = self.encoder(tokenized)                         # (B, 32, 1024)

# Invert mask to PyTorch "True = pad" convention
text_attention_mask = text_attention_mask.ne(1)                  # (B, 32), True = pad

# Seq-first for downstream MultiheadAttention
text_memory = text_memory.transpose(0, 1)                        # (32, B, 1024)
text_memory_resized = self.resizer(text_memory)                  # (32, B, 256)
inputs_embeds = inputs_embeds.transpose(0, 1)                    # (32, B, 1024)

return text_attention_mask, text_memory_resized, inputs_embeds
```

### Attention mask sign convention

There are two mask conventions in play:
1. **Tokenizer**: `(tokenized != 0)` where True = real token. This matches
   the usual `attention_mask` convention in HuggingFace.
2. **PyTorch transformer**: `key_padding_mask` where True = *ignore this
   position*.

`VETextEncoder` flips the mask with `.ne(1)` on `text_encoder_ve.py:312` so
that downstream `language_mask` has True = padded, matching the format
expected by `nn.MultiheadAttention(..., key_padding_mask=...)`.

## SAM3VLBackbone.forward_text (`vl_combiner.py:123-178`)

Wraps `VETextEncoder` and packages the outputs into the backbone_out dict:

```python
output["language_features"] = text_memory_resized   # (32, B, 256)
output["language_mask"]     = text_attention_mask   # (B, 32) bool, True = pad
output["language_embeds"]   = inputs_embeds         # (32, B, 1024)
```

`language_features` is what the fusion transformer consumes as the prompt.
`language_embeds` (the 1024-dim embeddings before the transformer runs) is
passed through unused in the text-only inference path but reserved for
training-time auxiliary losses.

## Shape Table

| Stage                        | Tensor shape       | Dtype  | Layout |
|------------------------------|--------------------|--------|--------|
| Raw input                    | list of B strings  | str    | -      |
| Tokenizer output             | (B, 32)            | int64  | BS     |
| token_embedding              | (B, 32, 1024)      | float  | BSD    |
| + positional_embedding       | (B, 32, 1024)      | float  | BSD    |
| After 24 transformer blocks  | (B, 32, 1024)      | float  | BSD    |
| After ln_final               | (B, 32, 1024)      | float  | BSD    |
| After transpose (seq-first)  | (32, B, 1024)      | float  | SBD    |
| After resizer linear         | (32, B, 256)       | float  | SBD    |

## Parameter Count

- Token embedding: 49408 × 1024 = ~50.6M
- Positional embedding: 32 × 1024 = 33k
- Per block: qkv 3.15M + proj 1.05M + MLP 8.4M + 2*LN = ~12.6M
- 24 blocks: ~302M
- ln_final: 2k
- Resizer (1024→256): ~262k
- **Text path total: ~353M**

Text and vision together account for most of the 848M parameters; the
transformer encoder, decoder, and mask head share the remaining ~50M.
