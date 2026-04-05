# Fusion Transformer: Encoder + Decoder

The fusion stack binds image features to the text prompt and refines 200
object queries into box+mask predictions. It consists of a 6-layer encoder
that updates image tokens via cross-attention to text, followed by a 6-layer
decoder that has object queries cross-attend to both text and image.

## Files

- Encoder layer/stack: `reference/sam3/sam3/model/encoder.py`
- Decoder layer/stack: `reference/sam3/sam3/model/decoder.py`
- Orchestrator: `reference/sam3/sam3/model/sam3_image.py` (`_encode_prompt`,
  `_run_encoder`, `_run_decoder`)
- Builders: `reference/sam3/sam3/model_builder.py:125-200`

## Pipeline

```
image features (72x72, 5184 tokens)     text memory (32 tokens)
   (5184, B, 256)                          (32, B, 256)
        |                                        |
        |          geometry features (optional)  |
        |               (Q_geo, B, 256)          |
        |                       |                |
        +--------------+--------+----------------+
                       |
                       v
         prompt = cat(text, geometry, visual)
              (seq_total, B, 256)
                       |
                       v
         +------------------------------+
         | Transformer Encoder (6 lyr)  |
         |  image self-attn             |
         |  image -> text cross-attn    |
         |  FFN                         |
         +------------------------------+
                       |
                       v
         encoder_out.memory  (B, 5184, 256)
         encoder_out.prompt_after_enc  (seq_total, B, 256)  -- unchanged
                       |
                       v
         +------------------------------+
         | Transformer Decoder (6 lyr)  |
         |  200 object queries          |
         |  + presence token            |
         |  query self-attn             |
         |  query -> text cross-attn    |
         |  query -> image cross-attn   |
         |  FFN + box refinement        |
         +------------------------------+
                       |
                       v
         { hs          : (6, B, 200, 256)
           ref_boxes   : (6, B, 200, 4)
           presence    : (6, B, 1) }
```

## Prompt Assembly (`sam3_image.py:167-210`)

`_encode_prompt` constructs the prompt tensor that the encoder and decoder
attend to. It concatenates three sources along the sequence dimension:

```python
prompt      = cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
prompt_mask = cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
```

For text-only inference (no geometric prompts, no visual prompts):
- `txt_feats`: `(32, B, 256)` from `language_features`
- `geo_feats`: empty `(1, B, 256)` CLS-only placeholder from
  `SequenceGeometryEncoder` if no boxes/points are supplied
- `visual_prompt_embed`: empty `(0, B, 256)`

Typical text-only shape: `prompt ≈ (33, B, 256)`, `prompt_mask ≈ (B, 33)`.

## Encoder (`encoder.py:254-596`)

### TransformerEncoderFusion (`encoder.py:464-579`)

6 stacked `TransformerEncoderLayer` instances. The "fusion" in the name refers
to the option `add_pooled_text_to_img_feat` which mixes a pooled text vector
into image tokens before attention. **This option is disabled in SAM3**
(`model_builder.py:159`), so the fusion reduces to stacked cross-attention.

Config from `model_builder.py:152-161`:

| Parameter                   | Value  |
|-----------------------------|--------|
| num_layers                  | 6      |
| d_model                     | 256    |
| num_feature_levels          | 1      |
| add_pooled_text_to_img_feat | False  |
| pool_text_with_mask         | True   |

With `num_feature_levels=1`, no `level_embed` parameter is created, and all
image tokens come from a single scale.

**Forward** (`encoder.py:515-579`):

1. Reshape the input image feature list from seq-first `(HW, B, C)` to
   NCHW `(B, C, H, W)` so `_prepare_multilevel_features` can work with it.
2. Flatten multilevel features to `(B, HW_total, C)`. With one level at 72×72,
   `HW_total = 5184`.
3. Call `super().forward(src, prompt=prompt.transpose(0, 1), ...)`. Note the
   `transpose(0, 1)`: the encoder layer treats the **image** features as
   `tgt` (batch-first internally) and the **text prompt** as `memory`
   (batch-first).
4. Return dict:

```python
{
    "memory":            out,                    # (seq=5184, B, 256)
    "padding_mask":      None,                   # no image padding
    "pos_embed":         lvl_pos_embed_flatten,  # (seq=5184, B, 256)
    "memory_text":       prompt,                 # (seq_text, B, 256) unchanged
    "level_start_index": tensor([0]),
    "spatial_shapes":    tensor([[72, 72]]),
    "valid_ratios":      ones((B, 1, 2)),
}
```

### TransformerEncoderLayer (`encoder.py:15-251`)

Each encoder layer is a **pre-norm** cross-attention block (`pre_norm=True`).
Despite the name, the architecture is the DETR decoder layer: self-attention
over image tokens, cross-attention to the text prompt, then FFN.

Config from `model_builder.py:127-150`:

| Parameter                      | Value |
|--------------------------------|-------|
| activation                     | relu  |
| d_model                        | 256   |
| dim_feedforward                | 2048  |
| dropout                        | 0.1   |
| num_heads (self + cross)       | 8     |
| pos_enc_at_attn                | True  |
| pos_enc_at_cross_attn_keys     | False |
| pos_enc_at_cross_attn_queries  | False |
| pre_norm                       | True  |

**forward_pre** (`encoder.py:141-203`):

```python
# Self-attention over image tokens (query = key = image + pos_img)
tgt2 = norm1(tgt)
q = k = tgt2 + query_pos                 # add image pos embed
tgt2 = self_attn(q, k, value=tgt2, ...)
tgt  = tgt + dropout1(tgt2)

# Cross-attention: image queries attend to text prompt (no pos added)
tgt2 = norm2(tgt)
tgt2 = cross_attn_image(
    query=tgt2,                          # image
    key=memory,                          # text prompt
    value=memory,
    key_padding_mask=memory_key_padding_mask,  # True = padded text
)
tgt  = tgt + dropout2(tgt2)

# FFN
tgt2 = norm3(tgt)
tgt2 = linear2(dropout(relu(linear1(tgt2))))
tgt  = tgt + dropout3(tgt2)
```

**Important directionality:** `tgt` is the image feature sequence and
`memory` is the text prompt. The **image** cross-attends to the **text**
(image queries text). The text prompt itself is not updated by the encoder —
it passes through unchanged and is returned as `memory_text`.

### Running the Encoder (`sam3_image.py:212-250`)

`_run_encoder` creates zero positional embeddings for the prompt
(`prompt_pos_embed = torch.zeros_like(prompt)`) and passes everything to
`TransformerEncoderFusion.forward`:

```python
memory = self.transformer.encoder(
    src=img_feats.copy(),                  # [(5184, B, 256)]
    src_key_padding_mask=None,
    src_pos=img_pos_embeds.copy(),         # [(5184, B, 256)]
    prompt=prompt,                         # (seq_total, B, 256)
    prompt_pos=prompt_pos_embed,           # zeros
    prompt_key_padding_mask=prompt_mask,   # (B, seq_total)
    feat_sizes=[(72, 72)],
)
```

The `src=img_feats.copy()` is a list copy because the encoder reshapes the
tensors in-place.

Output stored in `encoder_out`:

```python
encoder_out = {
    "encoder_hidden_states": memory["memory"],            # (5184, B, 256)
    "pos_embed":             memory["pos_embed"],         # (5184, B, 256)
    "padding_mask":          None,
    "level_start_index":     tensor([0]),
    "spatial_shapes":        tensor([[72, 72]]),
    "valid_ratios":          ones((B, 1, 2)),
    "vis_feat_sizes":        [(72, 72)],
    "prompt_before_enc":     prompt,                      # (seq, B, 256)
    "prompt_after_enc":      prompt,                      # (seq, B, 256) same
    "prompt_mask":           prompt_mask,                 # (B, seq)
}
```

## Decoder (`decoder.py:192-613`)

### TransformerDecoder (`decoder.py:192-613`)

6 stacked `TransformerDecoderLayer` instances that refine 200 learned object
queries through iterative box refinement. A presence token is prepended to
queries in self-attention only.

Config from `model_builder.py:182-199`:

| Parameter             | Value       |
|-----------------------|-------------|
| num_layers            | 6           |
| num_queries           | 200         |
| num_o2m_queries       | 0           |
| d_model               | 256         |
| return_intermediate   | True        |
| box_refine            | True        |
| dac                   | True        |
| boxRPB                | "log"       |
| presence_token        | True        |
| resolution            | 1008        |
| stride                | 14          |
| dac_use_selfatt_ln    | True        |

**dac** (Denoising Anchor Contrastive) doubles the query count to 400 during
training (200 one-to-one + 200 one-to-many). **At inference time `apply_dac`
is `False`** (`sam3_image.py:266: apply_dac = ... and self.training`), so only
200 queries are used.

**Learned parameters:**

```python
self.query_embed = nn.Embedding(200, 256)          # content query
self.reference_points = nn.Embedding(200, 4)       # reference boxes (unsigmoid)
self.presence_token = nn.Embedding(1, 256)
self.bbox_embed = MLP(256, 256, 4, 3)              # box delta MLP
self.ref_point_head = MLP(512, 256, 256, 2)        # sine2pos -> query pos
self.presence_token_head = MLP(256, 256, 1, 3)
self.presence_token_out_norm = nn.LayerNorm(256)
self.norm = nn.LayerNorm(256)
self.boxRPB_embed_x = MLP(2, 256, 8, 2)            # box-relative pos bias x
self.boxRPB_embed_y = MLP(2, 256, 8, 2)            # box-relative pos bias y
```

### Per-layer forward (`decoder.py:412-613`)

The decoder runs a loop over the 6 layers. Each iteration:

**1. Build sine position embedding from current reference boxes:**

```python
reference_points_input = reference_boxes[:, :, None] * valid_ratios_cat  # (nq, B, 1, 4)
query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], 256)
# shape: (nq=200, B, 512) — sine embed for each of (cx, cy, w, h)
query_pos = self.ref_point_head(query_sine_embed)   # (nq, B, 256)
```

This is DAB-DETR style conditional query: the positional embedding of each
query is derived from its current reference box estimate, not a fixed
learned query embedding.

**2. Compute box-relative positional bias (boxRPB="log"):**

```python
memory_mask = self._get_rpb_matrix(reference_boxes, (72, 72))
# shape: (B*8, 200, 5184) — per-head additive attn bias
```

For each of the 8 attention heads, for each of 200 queries, for each of 5184
image positions, produces a bias based on the signed log distance between
the image pixel and the box edges. The bias is added to the cross-attention
scores in the next step.

**3. Run the decoder layer:**

```python
output, presence_out = layer(
    tgt=output,                              # (200, B, 256)
    tgt_query_pos=query_pos,                 # (200, B, 256)
    memory_text=memory_text,                 # (33, B, 256)
    text_attention_mask=text_attention_mask, # (B, 33) True = pad
    memory=memory,                           # (5184, B, 256)
    memory_key_padding_mask=None,
    memory_pos=pos,                          # (5184, B, 256)
    cross_attn_mask=memory_mask,             # boxRPB bias
    presence_token=presence_out,             # (1, B, 256)
)
```

**4. Box refinement:**

```python
reference_before_sigmoid = inverse_sigmoid(reference_boxes)
delta_unsig = self.bbox_embed(self.norm(output))   # (200, B, 4)
new_reference = (delta_unsig + reference_before_sigmoid).sigmoid()
reference_boxes = new_reference.detach()
```

The reference box is updated layer by layer, so the conditional query
position (step 1) also moves with it. `.detach()` blocks the gradient from
flowing back through the reference box history during training.

**5. Record intermediate outputs:**

```python
intermediate.append(self.norm(output))
intermediate_ref_boxes.append(new_reference_points)
intermediate_presence_logits.append(
    self.presence_token_head(self.presence_token_out_norm(presence_out)).squeeze(-1)
)
```

### TransformerDecoderLayer (`decoder.py:33-189`)

Each decoder layer has 4 sub-blocks, all **post-norm** (unlike the encoder):

**Sub-block 1 — Self-attention (with presence token prepended):**

```python
# prepend presence token to queries
tgt_o2o = cat([presence_token, tgt], dim=0)      # (1+200, B, 256)
tgt_query_pos_o2o = cat([zeros, tgt_query_pos], dim=0)

q = k = tgt_o2o + tgt_query_pos_o2o
tgt2 = self.self_attn(q, k, value=tgt_o2o)[0]
tgt  = tgt_o2o + dropout2(tgt2)
tgt  = norm2(tgt)                                 # (1+200, B, 256)
```

**Sub-block 2 — Cross-attention to text (`use_text_cross_attention=True`):**

```python
tgt2 = self.ca_text(
    query=tgt + tgt_query_pos,
    key=memory_text,                              # (33, B, 256)
    value=memory_text,
    key_padding_mask=text_attention_mask,         # (B, 33) True = pad
)[0]
tgt  = tgt + catext_dropout(tgt2)
tgt  = catext_norm(tgt)
```

Note: the presence token goes through text cross-attention too (it was
prepended to `tgt` in sub-block 1 and is not split off until after FFN).

**Sub-block 3 — Cross-attention to image:**

```python
# zero-pad the cross-attn bias for the presence token
presence_token_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
cross_attn_mask = cat([presence_token_mask, cross_attn_mask], dim=1)

tgt2 = self.cross_attn(
    query=tgt + tgt_query_pos,                    # (1+200, B, 256)
    key=memory + memory_pos,                      # (5184, B, 256)
    value=memory,
    attn_mask=cross_attn_mask,                    # (B*8, 1+200, 5184) boxRPB
    key_padding_mask=None,
)[0]
tgt  = tgt + dropout1(tgt2)
tgt  = norm1(tgt)
```

**Sub-block 4 — FFN:**

```python
tgt2 = self.linear2(self.dropout3(relu(self.linear1(tgt))))
tgt  = tgt + dropout4(tgt2)
tgt  = norm3(tgt)

# Split presence token back out
presence_token_out = tgt[:1]    # (1, B, 256)
tgt                = tgt[1:]    # (200, B, 256)
return tgt, presence_token_out
```

### Running the Decoder (`sam3_image.py:252-298`)

```python
bs = memory.shape[1]
tgt = self.transformer.decoder.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
# tgt shape: (200, B, 256) — initial object query content

hs, reference_boxes, dec_presence_out, dec_presence_feats = (
    self.transformer.decoder(
        tgt=tgt,
        memory=memory,                     # (5184, B, 256)
        memory_key_padding_mask=None,
        pos=pos_embed,                     # (5184, B, 256)
        reference_boxes=None,              # start from learned reference_points
        level_start_index=...,
        spatial_shapes=tensor([[72, 72]]),
        valid_ratios=ones((B, 1, 2)),
        tgt_mask=None,
        memory_text=prompt,                # (33, B, 256) text+geometry prompt
        text_attention_mask=prompt_mask,   # (B, 33)
        apply_dac=False,                   # training-only
    )
)
```

Since `reference_boxes=None`, the decoder uses `self.reference_points.weight`
(the learned initial anchors) as the starting point for refinement.

**Return values:**

| Tensor                | Shape                  | Contents                             |
|-----------------------|------------------------|--------------------------------------|
| `hs`                  | `(6, B, 200, 256)`     | Per-layer query embeddings (normed)  |
| `reference_boxes`     | `(6, B, 200, 4)`       | Per-layer refined boxes (cxcywh)     |
| `dec_presence_out`    | `(6, B, 1)`            | Per-layer presence logits            |
| `dec_presence_feats`  | `(1, B, 256)`          | Final presence token feature         |

## Score Head: DotProductScoring (`sam3_image.py:300-384`)

Each of the 200 final decoder queries is scored by dot product against a
pooled text embedding. The score head combines three components:

```python
out["queries"] = hs[-1][:, :num_o2o]                      # (B, 200, 256)
outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)  # (6, B, 200, 1)
```

Internally, `DotProductScoring`:
1. Passes the pooled text through an MLP (2 layers, hidden=2048, residual,
   LayerNorm) to produce a `(B, 256)` query.
2. Takes the dot product with each of the 200 query embeddings: `(6, B, 200, 1)`.

## Box Head (`sam3_image.py:327-337`)

Final box coordinates are computed by adding the layer-6 bbox delta to the
layer-6 inverse-sigmoid reference:

```python
anchor_box_offsets = self.transformer.decoder.bbox_embed(hs)  # (6, B, 200, 4)
reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
outputs_coord     = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)
```

Note that this box head is **separate from** the iterative refinement inside
the decoder: the decoder's `bbox_embed` is applied at each layer to compute
the running reference box, and this final `bbox_embed(hs)` is applied once
for the final output. Both share the same `MLP` parameters (`self.bbox_embed`
is used in both places).

## Shape Table (Text-Only Inference, B=1)

| Stage                                      | Tensor shape       | Layout |
|--------------------------------------------|--------------------|--------|
| Image features (from backbone)             | (5184, 1, 256)     | SBD    |
| Image pos embed                            | (5184, 1, 256)     | SBD    |
| Text features (from text encoder)          | (32, 1, 256)       | SBD    |
| Geometry CLS (from `SequenceGeometryEncoder`) | (1, 1, 256)     | SBD    |
| Concatenated prompt                        | (33, 1, 256)       | SBD    |
| Prompt mask                                | (1, 33)            | BS     |
| Encoder output memory                      | (5184, 1, 256)     | SBD    |
| Encoder output `memory_text`               | (33, 1, 256)       | SBD    |
| Initial query `tgt`                        | (200, 1, 256)      | SBD    |
| Per-layer query output `hs`                | (6, 1, 200, 256)   | LBQD   |
| Per-layer reference boxes                  | (6, 1, 200, 4)     | LBQ4   |
| Per-layer presence logit                   | (6, 1, 1)          | LB1    |
| Final box logits `pred_boxes`              | (1, 200, 4)        | BQ4    |
| Final class logits `pred_logits`           | (1, 200, 1)        | BQ1    |

## Parameter Count

Encoder (6 layers):
- Per layer: self-attn 262k + cross-attn 262k + FFN 1.05M + 3×LN ≈ 1.58M
- 6 layers: ~9.5M

Decoder (6 layers):
- Per layer: self-attn 262k + text-cross 262k + img-cross 262k + FFN 1.05M
  + 4×LN ≈ 1.85M
- 6 layers: ~11.1M
- Query embed: 200 × 256 = 51k
- Reference points: 200 × 4 = 800
- Presence token: 256
- bbox MLP: ~200k
- ref_point_head MLP: ~200k
- boxRPB MLPs (x + y): ~6k
- **Decoder total: ~11.5M**

Dot product scorer MLP: ~1M

**Fusion total: ~22M**

This is a small fraction of the 848M total — the weight is in the image
and text backbones. The fusion stack is what distinguishes SAM3 from CLIP:
it's the bridge between open-vocabulary text understanding and
instance-level segmentation.
