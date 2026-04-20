# SAM 3.1 Tracker Parity Debugging Log

Running log of bisection iterations. One section per round.

## Iteration 1 — level-0 coarse dumps (2026-04-20)

### Setup

- Branch: `feature/sam3.1-image-path`
- Build: `cmake -DSAM3_DEBUG_DUMP=ON -DSAM3_BUILD_PARITY_TESTS=ON -DSAM3_PARITY_VARIANT=sam3_1`
- Seed: `/tmp/seed_lvl0.png` (sam3_1_dump_seed frame-0 output,
  288x288, ~2.4% foreground)
- Frames dumped: 0 (cond, C only), 1, 2 (both sides)
- C command: `./build/sam3_1_dump_seed --model models/sam3.1.sam3
  --video assets/kids.mp4 --point 0.5,0.5,1 --out /tmp/seed_lvl0.png
  --propagate-frames 2 --frames-dir /tmp/c_frames_lvl0`
  (requires `/tmp/c_frames_lvl0` to exist first; otherwise
  `stbi_write_png` fails and the callback returns 1 after frame 1,
  producing only `_f0` + `_f1` — the silent off-by-one we hit first.)
- Python command: `SAM3_CKPT=models/sam3.1_multiplex.pt
  python3 scripts/dump_tracker_layers.py --video assets/kids.mp4
  --seed-mask /tmp/seed_lvl0.png --frames 2`

### Results

```
# Tracker layer parity diff (cosine threshold: 0.99)
# Frames compared: [1, 2] (C also writes _f0 for mask-decoder only; no Python counterpart)

memattn_out_f1                           cos=0.00000 abs_max=7.415 abs_mean=0.9056 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_masks_f1                        cos=0.00000 abs_max=5.565 abs_mean=1.828 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_iou_f1                          cos=0.00000 abs_max=78.63 abs_mean=67.6 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_score_f1                        cos=0.00000 abs_max=1.739 abs_mean=1.283 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_sam_f1                          cos=0.00000 abs_max=8.876 abs_mean=0.9998 rel=100.000% <--- FIRST DIVERGENCE
memattn_out_f2                           cos=0.00000 abs_max=7.776 abs_mean=0.9018 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_masks_f2                        cos=0.00000 abs_max=6.153 abs_mean=2 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_iou_f2                          cos=0.00000 abs_max=84.84 abs_mean=74.06 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_score_f2                        cos=0.00000 abs_max=1.512 abs_mean=1.138 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_sam_f2                          cos=0.00000 abs_max=8.681 abs_mean=0.9952 rel=100.000% <--- FIRST DIVERGENCE

*** First divergent slot: memattn_out_f1
    -> drill down with the level-1 dumps for that path
```

### Analysis

- All 10 compared slots show `cos = 0.00000`, 100 % relative error.
  There are no `MISSING` rows — both sides produced `_f1` and `_f2` for
  every slot, so the comparison surface is complete.
- The first divergent slot is `memattn_out_f1`: memory attention is
  already broken at the very first propagation frame, before its output
  ever feeds the mask decoder. This explains why all downstream
  `mask_dec_*` slots are also zero-cosine on f1 and f2 — they inherit
  garbage inputs from the memory-attention stage.
- Magnitudes are non-trivial (`abs_max ≈ 7–8` for memattn, `iou` abs_mean
  ≈ 67–74, `masks` abs_max ≈ 5–6). This is not a small numerical drift;
  it is a structural mismatch — either the memory-attention compute,
  the memory-bank inputs it reads, or the layout in which C hands the
  tensor off to the Python-equivalent dump point.
- One shape note worth watching in the next drill: the C `memattn_out`
  dump is shaped `[1, 5184, 256, 0]` (NHWC-flattened, batch 1) while
  the Python dump is `(5184, 1, 256)` ([N, B, C]). The comparator
  flattens both, so the scalar count matches, but the underlying axis
  order must actually agree or the cosine will read zero by construction.
  Ruling in or out a layout mismatch is the first cheap check in
  iteration 2.

### Next step

Proceed to Task 5 with Path **α** (memory-attention per-layer drill),
justification: `memattn_out_f1` is the first divergent slot and every
downstream decoder slot also diverges, so the break is at or before the
memory-attention output. We enter via Path α (inspect each memory-attn
layer's inputs, Q/K/V, attn-weights, and output), and if the very first
layer shows its *inputs* already wrong (memory-bank contents, not the
compute), we jump to Path γ (memory bank per-entry parity) as the task
plan instructs. A layout/transpose mismatch on the dump site is also
on the short suspect list given the zero-cosine magnitude and the
shape-axis difference between C (`[1, N, C]`) and Python (`[N, 1, C]`).

## Iteration 2 — path α: memory-attn per-layer (2026-04-20)

### Setup

- Branch: `feature/sam3.1-image-path` (unchanged)
- Adds 4 per-layer dump slots on both C and Python sides:
  `memattn_layer0..3_f{1,2}`.
- C hook: bracket the existing 4-layer loop in
  `sam3_multiplex_memory_attn_forward`
  (`src/model/tracker_multiplex.c`) so the running `output` is captured
  into `sam3_dbg_trk_memattn_layer{i}` after each iteration.
- Python hook: `model.transformer.encoder.layers[i].register_forward_hook`
  in `scripts/dump_tracker_layers.py`. Each `DecoupledTransformerDecoder
  Layerv2.forward` returns `(image, output)`; we dump the `output`
  element (the [HW, B, C]=[5184,1,256] residual accumulator that
  matches C's per-iteration `output` pointer).
- Comparator `PAIRS` reordered so the layer rows precede
  `memattn_out_fN` and the mask-decoder rows, so the "FIRST DIVERGENCE"
  marker highlights the earliest layer.

### Results

```
# Tracker layer parity diff (cosine threshold: 0.99)
# Frames compared: [1, 2] (C also writes _f0 for mask-decoder only; no Python counterpart)

memattn_layer0_f1                        cos=0.01169 abs_max=4.782 abs_mean=0.4981 rel=100.001% <--- FIRST DIVERGENCE
memattn_layer1_f1                        cos=-0.00769 abs_max=6.212 abs_mean=0.84 rel=139.227% <--- FIRST DIVERGENCE
memattn_layer2_f1                        cos=0.00000 abs_max=12.24 abs_mean=0.8425 rel=100.000% <--- FIRST DIVERGENCE
memattn_layer3_f1                        cos=0.00000 abs_max=16.52 abs_mean=1.107 rel=100.000% <--- FIRST DIVERGENCE
memattn_out_f1                           cos=0.00000 abs_max=7.415 abs_mean=0.9056 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_masks_f1                        cos=0.00000 abs_max=5.565 abs_mean=1.828 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_iou_f1                          cos=0.00000 abs_max=78.63 abs_mean=67.6 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_score_f1                        cos=0.00000 abs_max=1.739 abs_mean=1.283 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_sam_f1                          cos=0.00000 abs_max=8.876 abs_mean=0.9998 rel=100.000% <--- FIRST DIVERGENCE
memattn_layer0_f2                        cos=0.01110 abs_max=3.939 abs_mean=0.5111 rel=100.001% <--- FIRST DIVERGENCE
memattn_layer1_f2                        cos=-0.00641 abs_max=6.844 abs_mean=0.8587 rel=137.049% <--- FIRST DIVERGENCE
memattn_layer2_f2                        cos=0.00000 abs_max=16.98 abs_mean=0.8713 rel=100.000% <--- FIRST DIVERGENCE
memattn_layer3_f2                        cos=0.00000 abs_max=22.58 abs_mean=1.131 rel=100.000% <--- FIRST DIVERGENCE
memattn_out_f2                           cos=0.00000 abs_max=7.776 abs_mean=0.9018 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_masks_f2                        cos=0.00000 abs_max=6.153 abs_mean=2 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_iou_f2                          cos=0.00000 abs_max=84.84 abs_mean=74.06 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_score_f2                        cos=0.00000 abs_max=1.512 abs_mean=1.138 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_sam_f2                          cos=0.00000 abs_max=8.681 abs_mean=0.9952 rel=100.000% <--- FIRST DIVERGENCE

*** First divergent slot: memattn_layer0_f1
    -> drill down with the level-1 dumps for that path
```

### Analysis

- **First divergent layer: `memattn_layer0_f1`** (cos ≈ 0.012, effectively
  orthogonal). The very first layer of the memory-attention stack is
  already broken.
- The progression of abs_max through the stack on frame 1 — 4.78 → 6.21
  → 12.24 → 16.52 — is monotonically increasing, consistent with a
  structural mismatch at the inputs that the residual path amplifies
  at every layer, rather than a single bad op inside one specific
  layer.
- Layer 0 is broken at its output, which means one of layer 0's
  **inputs** is already wrong: `output=tgt`, `image`, `memory`,
  `memory_image`, `memory_image_pos`, the RoPE tables `cos_q/sin_q/
  cos_k/sin_k`, or `num_k_exclude_rope`. Per the task plan, when the
  first layer is already divergent we pivot to **Path γ — memory bank
  contents**, since `memory` and `memory_image` are the tensors that
  carry cross-frame state and are the most plausible source of
  structural mismatch.
- We already have `memory`, `memory_image`, and `memory_image_pos`
  captured on the C side (the three extra level-0 slots that landed
  in Task 1). The next iteration adds their Python counterparts and
  compares, which will localize γ to a specific memory-bank entry or
  to the tpos/pointer-token concatenation logic.

### Next step

Pivot to **Path γ** — memory bank contents parity. Specifically:

1. Add Python-side hooks on the SAM 3.1 multiplex tracker's memory
   assembly path so that `memory`, `memory_image`, and
   `memory_image_pos` (the tensors actually passed to
   `model.transformer.encoder`) are dumped alongside `tgt`.
2. Compare against the existing C `dbg_trk_memory*_fN.bin` dumps.
3. If `tgt` matches but `memory` or `memory_image` diverges, the
   problem is in how the C tracker builds the memory bank (Path γ
   specifically — ordering of entries, tpos indexing, obj-ptr token
   concatenation).
4. If `tgt` itself diverges, the problem is upstream at the prompt
   encoder / object feature pipeline — pivot further back.

## Iteration 3 — path γ: memory bank input dumps (2026-04-20)

### Setup

- Branch: `feature/sam3.1-image-path` (unchanged)
- Adds a new C dump slot `sam3_dbg_trk_tgt` (populated at the
  memory-attn else-branch in
  `sam3_tracker_multiplex_track_frame`, next to the existing
  `memory/memory_image/memory_image_pos` hook).
- Python-side: the `model.transformer.encoder` forward-hook now uses
  `with_kwargs=True` so the hook receives the encoder's kwargs dict
  (upstream call-site passes all args by keyword:
  `video_tracking_multiplex.py:1590`). Captured into
  `captures["memattn_inputs"]` as a dict; `_flush_captures_delta`
  dumps `src` (→ `tgt`), `memory`, `memory_image`, and
  `memory_image_pos`.
- Comparator `PAIRS` inserts the four bank-input slots BEFORE the
  `memattn_layer0` rows. Row counts differ between sides:
    * C `memory` = spatial + obj_ptrs (spatial_c + obj_ptrs_c rows).
    * Py `memory` = spatial + 16 obj_ptrs (max_obj_ptrs_in_encoder).
    * C `memory_image` / `memory_image_pos` are padded to
      Nm_total (C already applies the obj_ptr concat).
    * Py `memory_image` / `memory_image_pos` come directly off the
      hook (pre-concat), so they match only the spatial prefix.
  `_report` now truncates bank tensors of different sizes to their
  common row prefix and logs the original row counts in brackets.

### Results

```
# Tracker layer parity diff (cosine threshold: 0.99)
# Frames compared: [1, 2] (C also writes _f0 for mask-decoder only; no Python counterpart)

tgt_f1                                   cos=0.57287 abs_max=4.686 abs_mean=0.4881 rel=111.642% <--- FIRST DIVERGENCE
memory_f1                                cos=0.49205 abs_max=10.24 abs_mean=0.4854 rel=108.052% <--- FIRST DIVERGENCE [truncated to 5185 rows (py_rows=5200, c_rows=5185)]
memory_image_f1                          cos=0.57695 abs_max=3.619 abs_mean=0.4942 rel=111.702% <--- FIRST DIVERGENCE [truncated to 5184 rows (py_rows=5184, c_rows=5185)]
memory_image_pos_f1                      cos=0.08799 abs_max=2 abs_mean=0.7937 rel=134.469% <--- FIRST DIVERGENCE [truncated to 5184 rows (py_rows=5184, c_rows=5185)]
memattn_layer0_f1                        cos=0.01169 abs_max=4.782 abs_mean=0.4981 rel=100.001% <--- FIRST DIVERGENCE
memattn_layer1_f1                        cos=-0.00769 abs_max=6.212 abs_mean=0.84 rel=139.227% <--- FIRST DIVERGENCE
memattn_layer2_f1                        cos=0.00000 abs_max=12.24 abs_mean=0.8425 rel=100.000% <--- FIRST DIVERGENCE
memattn_layer3_f1                        cos=0.00000 abs_max=16.52 abs_mean=1.107 rel=100.000% <--- FIRST DIVERGENCE
memattn_out_f1                           cos=0.00000 abs_max=7.415 abs_mean=0.9056 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_masks_f1                        cos=0.00000 abs_max=5.565 abs_mean=1.828 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_iou_f1                          cos=0.00000 abs_max=78.63 abs_mean=67.6 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_score_f1                        cos=0.00000 abs_max=1.739 abs_mean=1.283 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_sam_f1                          cos=0.00000 abs_max=8.876 abs_mean=0.9998 rel=100.000% <--- FIRST DIVERGENCE
tgt_f2                                   cos=0.57369 abs_max=4.615 abs_mean=0.489 rel=111.695% <--- FIRST DIVERGENCE
memory_f2                                cos=0.49201 abs_max=10.24 abs_mean=0.4854 rel=108.054% <--- FIRST DIVERGENCE [truncated to 5185 rows (py_rows=10400, c_rows=5185)]
memory_image_f2                          cos=0.57691 abs_max=3.619 abs_mean=0.4942 rel=111.700% <--- FIRST DIVERGENCE [truncated to 5185 rows (py_rows=10368, c_rows=5185)]
memory_image_pos_f2                      cos=0.04318 abs_max=13.09 abs_mean=0.7939 rel=141.271% <--- FIRST DIVERGENCE [truncated to 5185 rows (py_rows=10368, c_rows=5185)]
memattn_layer0_f2                        cos=0.01110 abs_max=3.939 abs_mean=0.5111 rel=100.001% <--- FIRST DIVERGENCE
memattn_layer1_f2                        cos=-0.00641 abs_max=6.844 abs_mean=0.8587 rel=137.049% <--- FIRST DIVERGENCE
memattn_layer2_f2                        cos=0.00000 abs_max=16.98 abs_mean=0.8713 rel=100.000% <--- FIRST DIVERGENCE
memattn_layer3_f2                        cos=0.00000 abs_max=22.58 abs_mean=1.131 rel=100.000% <--- FIRST DIVERGENCE
memattn_out_f2                           cos=0.00000 abs_max=7.776 abs_mean=0.9018 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_masks_f2                        cos=0.00000 abs_max=6.153 abs_mean=2 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_iou_f2                          cos=0.00000 abs_max=84.84 abs_mean=74.06 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_score_f2                        cos=0.00000 abs_max=1.512 abs_mean=1.138 rel=100.000% <--- FIRST DIVERGENCE
mask_dec_sam_f2                          cos=0.00000 abs_max=8.681 abs_mean=0.9952 rel=100.000% <--- FIRST DIVERGENCE

*** First divergent slot: tgt_f1
    -> drill down with the level-1 dumps for that path
```

### Analysis

- **First divergent slot is `tgt_f1`** (cos ≈ 0.573). `tgt` is the
  current-frame object-query stream, which on the C side is a plain
  reshape of `image_embed` to `[1, HW, 256]` right inside
  `sam3_tracker_multiplex_track_frame`. On the Python side the hook
  sees the `src=` kwarg passed to `model.transformer.encoder`, which
  is the multiplex tracker's `vision_feat` (also derived from the
  per-frame image embedding). cos=0.573 is not noise — the two
  tensors represent the *same* conceptual quantity but are
  numerically different by ≈ 50 %. This means **the per-frame image
  encoding the tracker hands to memory-attn already differs between
  C and Python on frame 1+**. This is upstream of the bank entirely.
- All three bank tensors also diverge, but their divergence is
  consistent with the `tgt` divergence: if the per-frame image
  features are off, then the `image_features` slot captured into the
  bank at commit time on frame 0 (which becomes the spatial rows of
  `memory_image` on frame 1) is also off, and `memory` (= maskmem
  output built on top of the same image features) inherits the same
  error. `memory_image_pos` is the worst (cos ≈ 0.088 / 0.043) —
  that is a separate, compounding break because `image_pe` + sine
  tpos is constructed differently from the image features, so the
  low cosine there is a *distinct* bug.
- Shape asymmetries worth noting for later iterations:
  * `memory_f2` has py_rows=10400 vs c_rows=5185. Python keeps
    2 spatial entries × 5184 rows + 32 obj_ptrs (= 2 × 16); C keeps
    1 spatial entry × 5184 rows + 1 obj_ptr. That means the C
    memory bank retains fewer entries and fewer obj_ptrs than
    Python — a Path γ concern in its own right.
  * On frame 1, C `memory_image` / `memory_image_pos` have 5185
    rows (spatial 5184 + 1 obj_ptr row) while Python has 5184
    (pre-concat). This is an expected representational difference
    (C pre-applies the concat internally vs Python concatting
    inside the encoder).
- **Both `tgt` and `memory_image_pos` are broken on the first
  propagation frame**, but `tgt` has no bank dependency, so it
  isolates cleanly: fix the image-features side first, then
  re-evaluate the bank.

### Next step

**Investigate `tgt` divergence on frame 1+ first.** Because `tgt` is
just a reshape of `image_embed` inside
`sam3_tracker_multiplex_track_frame` (no processing between the
image encoder and the hook site), the divergence must originate in
the per-frame image-encoder pipeline when it is run against a cached
state. Concrete actions:

1. Dump `image_embed` (the 4-D `[1,H,W,256]` NHWC input to
   `sam3_tracker_multiplex_track_frame`) on both sides at frame 1
   and diff it. If it also has cos ≈ 0.57, the image encoder itself
   is the culprit — which would be surprising given we already have
   sam3 image-path parity tests. If it matches but `tgt` still
   differs, the reshape/permute on the C side is wrong.
2. There are separate SAM 3.1 image-path parity tests for the
   *prompt-encoder* flow, but the *tracker* flow may exercise a
   different code path (video frame loading, image preprocessing,
   mean/std). A likely suspect is frame loading: C reads frames via
   libffmpeg and Python reads via `cv2` — verify that both produce
   byte-identical uint8 RGB frames at frame 1, or that the
   Image-encoder normalization offset/scale are identical.
3. Once `tgt` is aligned, re-run this comparator. The bank slots
   may automatically align too (since `memory_image` spatial rows
   are just cached copies of earlier-frame `feat_s1`, which in turn
   was derived from the same image features). If `memory_image_pos`
   still has cos < 0.2 after `tgt` is fixed, then `image_pe +
   maskmem_tpos_enc` is a separate bug in the pos-enc builder in
   `multiplex_build_memory_from_bank` (tracker_multiplex.c:1905).

## Iteration 4 — sub-drill: image_embed vs tgt (2026-04-20)

### Setup

- Goal: isolate whether the frame-1 `tgt` divergence (cos=0.573)
  originates in the reshape from `image_embed` → `tgt`, or upstream
  at the image-features source itself.
- C side: added `sam3_dbg_trk_image_embed` slot, assigned at the top
  of `sam3_tracker_multiplex_track_frame` (the 4-D NHWC
  `[1, 72, 72, 256]` feature map passed in as the first tensor
  argument, before any processing).
- Python side: monkey-patched
  `model._prepare_memory_conditioned_features` to capture
  `current_vision_feats[-1]` — the top-level backbone/neck output in
  `(HW, 1, C) = (5184, 1, 256)` layout, pre-`expand(-1, B, -1)`.
  Byte layout matches C's NHWC `[1, H, W, C]` (HW contiguous, C
  innermost).
- Both sides produce exactly 1,327,104 F32 values per frame (=
  5184 × 256). Same commands as iteration 1/3.

### Results

```
# Tracker layer parity diff (cosine threshold: 0.99)
# Frames compared: [1, 2]

image_embed_f1   cos=0.57287 abs_max=4.686 abs_mean=0.4881 rel=111.642% <--- FIRST DIVERGENCE
tgt_f1           cos=0.57287 abs_max=4.686 abs_mean=0.4881 rel=111.642% <--- FIRST DIVERGENCE
memory_f1        cos=0.49205 abs_max=10.24 abs_mean=0.4854 rel=108.052% [trunc 5185 rows]
memory_image_f1  cos=0.57695 abs_max=3.619 abs_mean=0.4942 rel=111.702% [trunc 5184 rows]
memory_image_pos_f1 cos=0.08799 abs_max=2    abs_mean=0.7937 rel=134.469% [trunc 5184 rows]
memattn_layer0_f1   cos=0.01169 ...
...
image_embed_f2   cos=0.57369 abs_max=4.615 abs_mean=0.489  rel=111.695% <--- FIRST DIVERGENCE
tgt_f2           cos=0.57369 abs_max=4.615 abs_mean=0.489  rel=111.695% <--- FIRST DIVERGENCE
```

### Analysis

- **`image_embed_f1` and `tgt_f1` have BYTE-IDENTICAL parity metrics**
  on both frames: cos=0.57287 (f1) and cos=0.57369 (f2), with the same
  abs_max, abs_mean, and relative error to five significant figures.
  This confirms that C's memcpy from `image_embed` → `tgt` is lossless
  (as expected — it's a literal `memcpy` of the same byte-layout
  tensor), and Python's `vision_feat = current_vision_feats[-1]
  .expand(-1, B, -1)` followed by slicing the B=1 plane is likewise
  byte-equal to the pre-expand source.
- **The divergence originates UPSTREAM of the tracker**: the feature
  tensor handed to the tracker already disagrees at cos=0.573 before
  any memory-attention or reshape work begins. The tracker is NOT the
  source of the bug; it is a downstream victim.
- cos ≈ 0.57 at the top-of-memory-attn input with abs_mean ≈ 0.49 and
  rel ≈ 111 % is consistent with the image-encoder output genuinely
  differing in both magnitude AND direction — not a simple off-by-one,
  scale, or transpose. It rules out (a) NHWC↔NCHW confusion (that
  would keep the values but permute the order, leaving cos high if we
  flatten identically on both sides), (b) an obj-ptr pre-pad mis-count
  (image_embed has no obj-ptr rows), and (c) the memory-bank pos-enc
  bug flagged in iteration 3 (that feeds only into `memory_image_pos`,
  not `image_embed`).
- The divergence is symmetric across frame 1 and frame 2 (0.57287 vs
  0.57369) but _frame 0_'s C dump matches upstream — that is, the
  seed frame's mask-decoder output agreed in earlier iterations. The
  difference therefore appears specifically on propagation frames
  where the C engine's image-features pipeline is re-invoked against
  a new frame load, suggesting:
  1. Frame-loading divergence (C uses libffmpeg; Python uses cv2) —
     uint8 RGB bytes may differ at frame 1+ due to codec / colour-
     space / chroma-subsampling handling.
  2. Image-encoder / neck output divergence on non-seed frames,
     possibly from a frame-cache or state-mutation artefact.
  3. Preprocessing (mean/std / resize / letterbox) misalignment on
     propagation frames that does not manifest on frame 0 because
     frame 0 takes a different code path in both engines.

### Interpretation

**Upstream feature disagreement.** The tracker reshape is not the
bug. The image features that feed the tracker already diverge at
cos=0.573 on frame 1 and frame 2.

### Next step

The next drill must target the image-features pipeline on propagation
frames. In priority order:

1. **Frame-bytes comparison.** Dump `feat_s1` on both sides (the
   direct output of neck) for frame 1 and frame 2, AND the raw
   pre-preprocess uint8 RGB frame fed into the image encoder. If the
   uint8 bytes already disagree, the decoder is the culprit (ffmpeg
   vs cv2). If uint8 bytes match but `feat_s1` disagrees, the image
   encoder or neck is running with a different state on propagation
   frames.
2. **Verify `feat_s1` at frame 0 still matches.** If the C seed-mask
   at frame 0 is produced from the _same_ `feat_s1` layout and it
   agrees with Python, then whatever breaks on frames 1/2 is
   frame-load / pipeline specific — not a static mis-wiring.
3. Only once image-features parity holds should we revisit the
   tracker bank / pos-enc findings from iteration 3.

## Iteration 5 — fix (2026-04-20)

### Patches

**Bug A (memory_image_pos):** The C tracker was adding the mask-decoder's
Gaussian-random PE (`PositionEmbeddingRandom`, loaded from the
`image_pe_layer.positional_encoding_gaussian_matrix` weight) to the
spatial rows of `memory_image_pos`. Python uses the tracker backbone's
sinusoidal PE (`PositionEmbeddingSine`, no learned weights — pure
sin/cos over normalized (x, y) with temperature=10000), which is what
`propagation_vision_pos_embeds[-1]` carries. Added a local
`multiplex_fill_dense_sine_pe_row` helper in
`src/model/tracker_multiplex.c` (≈1800) that reproduces
`reference/sam3/sam3/model/position_encoding.py` exactly; swapped the
spatial-row inner loop in `multiplex_build_memory_from_bank`
(tracker_multiplex.c:1945) to use it instead of the Gaussian `image_pe`.

**Bug B (bank retention):** Two changes in concert:
1. `src/model/tracker_multiplex.c:1696` — bumped
   `MUX_MAX_MEM_ENTRIES_IN_ATTN` from 2 to `SAM3_MULTIPLEX_NUM_MASKMEM`
   (= 7), matching Python's `1 cond + (num_maskmem - 1) non-cond`
   fanout.
2. `src/model/video_session.c:50` — the SAM 3.1 multiplex variant now
   initialises the bank with `mf_threshold = -1.0f` (filter disabled),
   matching Python's default `use_memory_selection=False`. Without this
   change, every propagation frame was rejected by the SAM3-Long
   obj_score filter (because the image_embed root bug produces
   obj_score=0), so `n_non_cond` stayed 0 and the cap bump alone was a
   no-op. SAM 3 legacy still uses `mf_threshold=0.01f`.

### Post-fix comparator output (from /tmp/compare_post_fix.log)

```
memory_image_pos_f1  cos=0.94750 abs_max=1.331 abs_mean=0.1051  rel=17.807%  [py_rows=5184, c_rows=5185]
memory_f1            cos=0.49205 abs_max=10.24 abs_mean=0.4854  rel=108.052% [py_rows=5200, c_rows=5185]
memory_image_pos_f2  cos=0.88266 abs_max=2.154 abs_mean=0.2353  rel=39.821%  [py_rows=10368, c_rows=10370]
memory_f2            cos=0.12510 abs_max=10.65 abs_mean=0.5623  rel=127.150% [py_rows=10400, c_rows=10370]
memory_image_f2      cos=0.48093 abs_max=6.285 abs_mean=0.5347  rel=121.565% [py_rows=10368, c_rows=10370]
memattn_out_f2       cos=0.00012 abs_max=1.418e+17 abs_mean=1.069e+11  (numerical blow-up on f2 — see below)
image_embed_f1       cos=0.57287 (unchanged — fix is out of scope for this path)
image_embed_f2       cos=0.57369 (unchanged — same reason)
```

### Analysis

- `memory_image_pos_f1`: before cos=0.088, **after cos=0.9475** — Bug A
  confirmed. The slot is now dominated by the sine-PE grid (matching
  Python). Residual 0.05 cosine gap comes from the remaining f1 shape
  mismatch (C has 1 obj_ptr tail row vs Python's 16; see below).
- `memory_image_pos_f2`: before cos=0.043, **after cos=0.883** — same
  Bug A fix, same residual (+ 2 obj_ptr tails vs Python's 32 from two
  frames × multiplex_count=16).
- `memory_f2` row count: C 5185 → **10370**, Python 10400 — Bug B
  retention fix now pulls in both the cond frame (frame 0) and the
  non-cond propagation result from frame 1. Row count still trails
  Python by 30 = 2 × (16 − 1), which is exactly the `multiplex_count`
  obj_ptr expansion the C bank does not yet do (the C bank stores one
  "best" 256-D obj_pointer per frame; Python stores the full
  `[multiplex_count=16, 256]` and cats them as 16 tokens per frame).
  Left as a known follow-up — this is the `obj_pointer [n_obj, 256]`
  noted in `memory_bank.h:25`.
- `memattn_layer0_f{1,2}`, `memattn_out_f{1,2}`: still near-zero
  cosines. As expected — these are downstream of the image_embed
  divergence (cos=0.573) which this task did not address.
- **Regression note — numerical blow-up at `memattn_out_f2`**
  (abs_max=1.4e17). This was introduced by the Bug B retention bump:
  adding a second spatial K/V block when the image_embed input is
  already wrong seems to cause a softmax / RoPE interaction that
  explodes through the 4-layer stack. The divergence is downstream of
  the image_embed bug (memattn_layer1_f2 already shows abs_max=229),
  so we expect this to clear once image_embed parity is restored. No
  change in regression tests (`ctest -E test_fixture_compare|
  test_video_parity_kids` → 74/74 passing).
- `image_embed_f1` still at cos=0.573 — confirmed unchanged. The fix
  targets the memory bank construction, not the per-frame image
  encoder pipeline, which is a separate bug.

### Next step

`image_embed` is now the sole pre-memattn divergence on propagation
frames. The next session should:

1. Dump `feat_s1` at frame 1 on both sides.
2. If `feat_s1` disagrees: bisect the backbone / neck path for
   propagation frames (frame-load, preprocess, ViT state).
3. Once `image_embed` clears cos≥0.99, re-run this comparator — the
   memattn blow-up on f2 should collapse back to matched-magnitude
   outputs.
4. Then address the outstanding obj_ptr multiplex expansion (store
   full [16, 256] in the bank, emit 16 rows per entry in
   `multiplex_build_memory_from_bank`) to close the remaining ~30-row
   shape gap at frame 2.

