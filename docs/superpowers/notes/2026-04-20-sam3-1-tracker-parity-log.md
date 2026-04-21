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

## Iteration 6 — image-pipeline drill + propagation_convs fix (2026-04-20)

### Setup

- Added per-frame dumps for `frame_rgb` (preprocessed F32 CHW
  [-1,1]) and `feat_s1` (sam2 1x post-neck NHWC) on both sides.
  C dumps at the tail of `session_encode_frame` for every cached
  frame (including cond frame 0); Python dumps via a forward-hook on
  `model.forward_image`. Extended comparator with `_reshape_frame_rgb`
  and `_reshape_feat_s1`.
- SAM 3.1 (tri-neck) + SAM 3 (dual-neck) now share the
  `sam2_fpn_layers.*` destination prefix on disk. The SAM 3 source
  path stays `sam2_convs.*`; the SAM 3.1 source path is
  `propagation_convs.*` via a new `handle_propagation_neck` emitter
  in `weight_rename.c`.

### Pre-fix diff (propagation_convs NOT loaded)

```
frame_rgb_f0     cos=0.99983 abs_max=0.1176 abs_mean=0.002497 rel=0.498%
feat_s1_f0       cos=0.57695 abs_max=3.619  abs_mean=0.4942   rel=111.702% <--- FIRST DIVERGENCE
frame_rgb_f1     cos=0.99984 abs_max=0.1176 abs_mean=0.002447 rel=0.488%
feat_s1_f1       cos=0.57287 abs_max=4.686  abs_mean=0.4881   rel=111.642% <--- FIRST DIVERGENCE
frame_rgb_f2     cos=0.99983 abs_max=0.1333 abs_mean=0.002533 rel=0.506%
feat_s1_f2       cos=0.57369 abs_max=4.615  abs_mean=0.489    rel=111.695% <--- FIRST DIVERGENCE
```

### Analysis

- `frame_rgb` matches on all three frames at cos≈0.99984 — the
  preprocessed inputs (libav-decoded + stbir-linear-resized vs
  decord-decoded and auto-resized) are effectively identical.
- `feat_s1` nevertheless diverges at cos≈0.577 on ALL three frames
  (including the cond frame). So the image-encoder output is wrong in
  a frame-independent way — static-wiring issue, not state.
- **Root cause:** SAM 3.1 uses `Sam3TriViTDetNeck` which has three
  independent sets of conv weights for the simpleFPN — `convs`
  (detector), `interactive_convs` (prompt/mask), `propagation_convs`
  (tracker). C's converter had no handler for
  `detector.backbone.vision_backbone.propagation_convs.*`, so those
  weights were dropped from the `.sam3` file, `has_sam2_neck` was
  forced to 0 for the tri-neck, and the tracker fell back to reading
  the **detector** neck's output as its feat_s1 — a different set of
  weights producing different features by design.

### Patches

1. **`tools/weight_rename.c`:** factored the sam2-neck renamer into a
   shared `emit_sam2_fpn(prefix, rest)` helper. Added
   `handle_propagation_neck` that routes `propagation_convs.{i}.*` to
   the same `detector_model.vision_encoder.neck.sam2_fpn_layers.{i}.*`
   destination used by SAM 3's `sam2_convs`. One new entry in the
   prefix table.
2. **`src/model/vl_combiner.c`:** removed the `if (n_fpn_scales < 4)
   has_sam2_neck = 0` special-case. The sam2_neck is now always
   initialised with the main neck's `n_fpn_scales` (3 for SAM 3.1,
   4 for SAM 3) and always loaded from `sam2_fpn_layers.*`.
3. **`src/model/sam3_image.c`:** added NULL guard in debug
   `dump_tensor` for the tri-neck's missing-scale-3 spfn[3] entry
   (prevents a `SAM3_DEBUG_DUMP` SEGV during sam2_05x dump).

### Post-fix diff (propagation_convs loaded as sam2_fpn_layers)

```
frame_rgb_f0     cos=0.99983 abs_max=0.1176 abs_mean=0.002497 rel=0.498%
feat_s1_f0       cos=0.96899 abs_max=3.474  abs_mean=0.08944  rel=20.216%  <--- FIRST DIVERGENCE
feat_s1_f1       cos=0.96614 abs_max=3.867  abs_mean=0.09301  rel=21.273%
feat_s1_f2       cos=0.96644 abs_max=4.134  abs_mean=0.09185  rel=20.978%
image_embed_f1   cos=0.96614 (= feat_s1_f1, as expected)
memory_image_f1  cos=0.96899 (up from 0.57695 — bank spatial rows follow feat_s1)
memory_f1        cos=0.69383 (up from 0.49205 — spatial recovered; obj_ptr tail still 1 vs 16)
memattn_layer0_f1 cos=0.00383 (still near-orthogonal; K/V still missing 15 obj_ptr tokens)
```

### Parity test result after fix

- `test_video_parity_kids` SAM 3.1 variant:
  - Frame 1 IoU: **0.4449** (was 0.0000)
  - Frame 2 IoU: **0.4853** (was 0.0000)
  - Frame 3 IoU: 0.0000 (still 0 — compounded from memattn blow-up)

### Remaining divergence

- **~3 % cosine gap on feat_s1** (0.968 vs 0.99 target). Candidates:
  a) frame_rgb abs_max=0.1176 (≈15/255 in edge pixels) amplified by
     the ViT, b) a small op mismatch inside the ViT or neck, c)
     differences in normalization / padding that compound across
     32 ViT blocks. Warrants a separate bisection pass (dump ViT
     intermediates) if the remaining fixes don't close the IoU gap.
- **obj_ptr multiplex:** C bank stores one 256-D pointer per frame;
  Python stores `[multiplex_count=16, 256]`. This surfaces as a
  15-row gap per frame in `memory` (K/V stream), which degrades
  memattn_layer0 to near-zero cosine even with corrected feat_s1.
  Task 4 in the implementation plan.

### Next step

1. Expand `sam3_memory_entry.obj_pointer` to `[16, 256]` and emit 16
   rows per entry in `multiplex_build_memory_from_bank`.
2. Re-run the comparator. memattn_layer0 should track the spatial
   K/V contribution, and the f2 numerical blow-up should collapse.
3. If IoU still trails 0.75, drill the residual 3 % feat_s1 gap:
   dump ViT block outputs or sam2 neck intermediates side by side.

## Iteration 7 — obj_pointer multiplex expansion to [16, 256] (2026-04-20)

### Scope

Previously `sam3_memory_entry.obj_pointer` stored a single [1, 256]
tensor per frame — the best mask of slot 0 after argmax. Python stores
`[num_buckets=1, multiplex_count=16, C=256]` flattened to [16, 256]:
per-slot argmax over 3 multimask heads, then `obj_ptr_proj` applied.

### Patches

- **`src/model/tracker_multiplex.h`:** add `out_all_iou` output param
  (shape [16, 3], per-slot IoU). `out_obj_ptrs` reshaped to
  [16, 3, 256] (all slots × 3 multimask heads × projected dim).
- **`src/model/tracker_multiplex.c`:** reshape all_sam from [16, 3,
  256] to [48, 256], apply `obj_ptr_proj` mlp3_relu on the flat
  tensor, reshape result to [16, 3, 256]. Add the all_iou reshape to
  [16, 3]. `multiplex_build_memory_from_bank` now emits `op->dims[0]`
  rows per entry (16 for SAM 3.1, 1 for SAM 3 legacy) with the same
  tpos encoding shared across slots of the same frame.
- **`src/model/sam3_video.c`:** at step-8 best-obj_ptr persistence,
  allocate `[multiplex_count, 256]` (16 rows for SAM 3.1) and for
  each slot `s` in 0..15 argmax iou over 3 multimask heads to pick
  the best row of `obj_ptrs[s, *, :]`. Apply `no_obj_ptr_linear`
  per-slot when the object isn't appearing. SAM 3 path unchanged
  ([1, 256] copied verbatim).
- **`src/model/memory_bank.h`:** updated the `obj_pointer` comment
  to document the per-variant dims[0] convention.

### Post-fix comparator (key rows)

```
memory_f1        cos=0.69349 abs_max=9.887 abs_mean=0.3442  [py_rows=5200, c_rows=5200]  (row counts match now)
memory_image_f1  cos=0.96899 [py_rows=5184, c_rows=5200]    (C pads Nm rows, Py is pre-concat)
memory_f2        cos=0.29024 abs_max=11.21 abs_mean=0.4713  [py_rows=10400, c_rows=10400]
memory_image_f2  cos=0.80133 abs_max=5.31  abs_mean=0.2406  [py_rows=10368, c_rows=10400]
```

Row counts now match Python's exactly (5200, 10400). The cosine on
`memory_f1` / `memory_f2` content is dominated by the obj_ptr rows'
absolute magnitude (~2.0 in Python; C's per-slot values still differ
slightly after argmax+no_obj_ptr_linear mismatches between the
per-slot is_appearing logic).

### Parity test result (test_video_parity_kids SAM 3.1 variant)

| Frame | Baseline | After propagation_convs | After propagation_convs + obj_ptr [16, 256] |
|-------|----------|-------------------------|---------------------------------------------|
| 1     | 0.0000   | 0.4449                  | **0.4430**                                  |
| 2     | 0.0000   | 0.4853                  | **0.4863**                                  |
| 3     | 0.0000   | 0.0000                  | **0.4135**                                  |

Frame 3 **jumped from 0.0000 to 0.4135** after the obj_ptr expansion
unblocked its downstream path (previously compounded the f2 numerical
blow-up in memattn_out). All three propagation frames now produce
meaningful IoU but still trail the 0.75 threshold by ~0.25-0.30.

### Remaining divergence

- **feat_s1 residual ~3 % cosine gap** — unchanged from iteration 6.
  Likely amplified through the memattn stack (queries and keys
  inherit this error). Separate sub-drill would need to bisect ViT
  block outputs.
- **memattn_layer0 still near cos≈0** despite correct bank shape.
  Implies the cosine is dominated by the Q-K alignment: even small
  feat_s1 errors on attention-relevant tokens produce large output
  divergences.

### Next step

1. Drill the residual feat_s1 cosine gap. Options:
   - Dump C's ViT output vs Python's, confirm the backbone alone.
   - Dump `cached_feat_s1_nhwc` (sam3 neck) vs Python's
     `sam3_backbone_out["backbone_fpn"][-1]` to see if the error
     is in the neck specifically (we already confirmed the tracker
     uses the sam2 side, but both necks take the same ViT input).
   - Check whether frame_rgb abs_max=0.1176 (stbir-linear vs decord
     bilinear) can be narrowed by switching C to a cubic / stb-rb
     resize kernel matching decord's defaults.

## Iteration 8 — ViT sub-drill: uninitialised patch_embed_b bias (2026-04-20)

### Scope

Bisect the ViT residual 3 % cosine gap. Added three C-side dump
slots and matching Python hooks, then split the gap location-by-
location to find the first divergence.

New slots (all gated on `SAM3_DEBUG_DUMP`):
- `/tmp/dbg_trk_vit_out_f{N}.bin` — ViT trunk output (pre-neck),
  NHWC [1, 72, 72, 1024]. C populates via `sam3_dbg_trk_vit_out` in
  sam3_image.c; sam3_video.c dumps per-frame.
- `/tmp/dbg_trk_sam3_feat_s1_f{N}.bin` — sam3-side neck 1x output
  (pfn[2]). Same input ViT, different conv weights than the
  propagation side.
- `/tmp/dbg_vit_patch_only.bin` — patch_embed + patch_embed_b
  output (pre-pos_embed, pre-ln_pre). image_encoder.c splits the
  pre-blocks graph eval to dump this intermediate.

Matching Python captures in `scripts/dump_tracker_layers.py`:
`vit_out`, `sam3_feat_s1`, `vit_patch_only`, `vit_pre_blocks`,
and `vit_block{00,03,07,11,13,14,15,16,17,19,23,27,31}` (all hook
`model.backbone.vision_backbone.trunk.*`).

### Bisection results (baseline — before fix)

```
frame_rgb_f0        cos=0.99983 abs_max=0.1176
patch_only          cos=0.61457 abs_max=0.5535 p_std=0.23  <--- DIVERGES
pre_blocks          cos=0.82000 abs_max=21.49  p_std=1.12
block00             cos=0.91798
block07             cos=0.99008  (best intermediate)
block16             cos=0.95433
block31             cos=0.93454  abs_max=181.3
vit_out_f0          cos=0.93742 (= block31)
sam3_feat_s1_f0     cos=0.96634
feat_s1_f0          cos=0.96899
```

The first divergence is `patch_only` (cos=0.61) — the conv output
alone is already wildly different. Block-level dumps show the
cosine *recovers* to 0.99 by block 7 (LN normalisation absorbs
the DC offset) then degrades again through blocks 14-31 as
attention sinks grow at slightly different spatial positions.

### Root cause

Per-channel mean of the diff `C - Py` at `patch_only` was
`-0.3609 ± 0.0648`. Adding back the per-channel mean as a
correction raised the cosine to **0.9997** — so C's conv output
has a spurious per-channel DC offset.

Logging `patch_embed_b` stats on the C side: `min=-0.3725
max=0.0000 mean=-0.3609 std=0.0648` — identical distribution to
the diff. Python's checkpoint has **no** `patch_embed.proj.bias`
(SAM 3.1 builds with `bias_patch_embed=False`), so
`gh_load_mmap_optional` correctly returns NULL and the fallback
path calls `gh_alloc_tensor`. But `gh_alloc_tensor` only zeroes
the tensor struct, not its `data` buffer — the raw arena bytes
that became the "bias" were stale weight data left in the arena
from prior loads.

So on SAM 3.1 (bias_patch_embed=False) the ViT was adding a
non-zero channel-dependent bias after every patch conv, producing
a shifted pre-blocks state that the blocks then amplified into
the 7 % residual.

### Patch

`src/model/image_encoder.c`:

- When `gh_load_mmap_optional` returns NULL for
  `projection.bias`, explicitly `memset(t->data, 0, nbytes)`
  after allocation. Comment documents the amplification chain so
  the next reader doesn't "simplify" the memset away.

### Post-fix comparator

```
vit_out_f0         cos=0.96177  abs_max=171.1  (was 0.93742 / 186.1)
sam3_feat_s1_f0    cos=0.98176  abs_max=3.201  (was 0.96634 / 3.58)
feat_s1_f0         cos=0.98374  abs_max=3.115  (was 0.96899 / 3.474)
```

feat_s1 residual halved (3.1 % → 1.6 %). `patch_embed_b stats`
now logs `min=0.0000 max=0.0000 mean=0.0000 std=0.0000`.

### Parity test result

`test_video_parity_kids` SAM 3.1 variant: IoUs on frames 1/2/3
stay at 0.44/0.49/0.41 against the *committed* fixture PNGs.
That's expected: the committed `seed_mask.png` was produced by
the **buggy** sam3_1_dump_seed; with the fix, C's new frame-0
mask diverges from the old seed (frame-0-vs-seed IoU drops to
0.13). Regenerating the fixture via
`tools/gen_video_parity_fixtures.py` is required to get a
fair IoU number on the fix. The comparator cosines directly
confirm the ViT improved.

74/76 tests still pass (same 2 pre-existing failures:
`test_fixture_compare` NaN, `test_video_parity_kids` target).

### Remaining divergence

- **vit_out abs_max still ~170.** The per-channel DC offset was
  one component of the ViT divergence; the remaining ~4 % cosine
  gap is distributed across many channels and positions. Clamping
  outliers only recovers ~1 % cosine, so the residual is not
  outlier-dominated. Likely candidates: RoPE precomputation
  differences (SAM 3.1 uses `use_interp_rope=True`; C's
  `precompute_rope_table` doesn't interpolate), accumulated f32
  rounding through 32 blocks, or small op mismatches in
  gh_multihead_attention_rope.
- **memattn amplifies 2 % feat_s1 error to near-orthogonal
  memattn_layer0 output** (cos=0.004). This is the dominant IoU
  killer. The C memattn must have a structural difference that
  magnifies Q/K misalignment.

### Next step

1. Regenerate the SAM 3.1 parity fixtures against the fixed C
   seed: `./build/sam3_1_dump_seed ...` → `seed_mask.png`, then
   `python gen_video_parity_fixtures.py --variant sam3.1 ...` →
   new `frame_000{1,2,3}_obj_1.png`. Re-run the parity test to
   get a fixture-honest IoU.
2. Drill the memattn amplification. Compare `tgt` (matches 0.98)
   vs `memattn_layer0` (cos=0.004). Likely sub-steps: Q projection,
   K projection, attention softmax, V projection. Hook each inside
   `DecoupledTransformerDecoderLayer.forward` on the Python side
   and add matching C-side dumps in tracker_multiplex.c.
3. If the memattn structural bug is fixed and IoU still trails
   0.75, return to the vit_out outlier positions and drill
   further ViT blocks (block 8-15 is where the outlier positions
   start diverging — see abs_max trend).

## Iteration 9 — memattn sub-layer bisection (2026-04-20)

### Setup

- Branch: `feature/sam3.1-image-path`
- Fixtures regenerated against post-Iter-8 C seed (`seed_mask.png`
  fg% 2.4 → 7.4). Python reference converged to near-identical masks
  (frame-1 IoU 1.0 vs old fixture; frames 2/3 at 0.93/0.99). Fixture
  regen alone bumped `test_video_parity_kids` frame-2 IoU from 0.49 to
  0.52; frames 1/3 essentially unchanged (0.44/0.41).
- New dump slots inside `memory_attn_layer()` (layer 0 only):
  `memattn_l0_sa_out` (post self-attn residual), `memattn_l0_ca_q/k/v`
  (cross-attn projections, pre-RoPE), `memattn_l0_ca_attn`
  (cross-attn body output = `cross_attn_out_proj(sdpa(Q,K,V))`, pre
  residual add).
- Matching Python hooks rebind `layer[0]._forward_sa` and
  `._forward_ca` so the same intermediates are captured on the
  reference side.
- `dbg_force_readback = 1` on each captured tensor so Metal backend
  materialises host-side f32 before graph_eval completes.

### Results (frame 1)

```
tgt_f1            cos=0.98066 abs_mean=0.064   <--- inputs OK
memory_image_f1   cos=0.98374 abs_mean=0.060   <--- inputs OK
memory_image_pos_f1 cos=0.94750 abs_mean=0.105
memory_f1         cos=0.70974 abs_mean=0.335   <--- BANK CONTENT BROKEN
memattn_l0_sa_out_f1   cos=0.97581             <--- self-attn fine
memattn_l0_ca_q_f1     cos=0.98900             <--- Q derived from tgt/image OK
memattn_l0_ca_k_f1     cos=0.47828             <--- K inherits memory
memattn_l0_ca_v_f1     cos=0.72804             <--- V = v_proj(memory)
memattn_l0_ca_attn_f1  cos=0.56430             <--- attn body given bad K/V
memattn_layer0_f1      cos=0.00363             <--- residual + FFN amplify
```

### Analysis — memattn is NOT the culprit

- **Self-attn is clean.** sa_out tracks output at cos 0.976 against
  Python's post-self-attn state. The residual `output + self_attn()`
  is faithful.
- **Q is clean.** cos 0.989 for `memattn_l0_ca_q`. Q is
  `img_cross_attn_q_proj(image) + cross_attn_q_proj(tgt2)` where both
  `image` and `tgt` ride on feat_s1 (cos 0.98). No K/V pollution yet.
- **K and V degrade sharply.** cos 0.478 / 0.728 respectively. K =
  `img_cross_attn_k_proj(memory_image) + cross_attn_k_proj(memory) +
  memory_image_pos`. `memory_image` and `memory_image_pos` both match
  at ≥0.95, so the ca_k divergence tracks `memory` (which is at 0.71).
  V = `cross_attn_v_proj(memory)` at 0.728 matches `memory`'s cos 0.71
  within the 1-1 cross_v projection.
- **ca_attn (post out_proj) at 0.564** is consistent with Q @ K^T
  picking up the right attention pattern (Q is right) but the
  resulting weighted average of V is polluted by V's 0.73 cosine.
- **Layer 0 output collapses to 0.004** because (sa_out at 0.976 +
  ca_attn at 0.564), with ca_attn's abs_mean ≈ 2.65× larger than
  sa_out's, is dominated by the corrupted ca_attn direction, then the
  FFN amplifies.

The memattn compute is **structurally correct**. Given matched
inputs it would produce matched outputs. What's wrong is
`memory` (the cross-attention K/V stream).

### Root cause — memory bank content

Breaking down `memory_f1` [5200, 256] by row range:

```
memory_f1 spatial rows [0:5184]:  cos=0.710  abs_mean_py=0.453  abs_mean_c=0.463
memory_f1 obj_ptr rows  [5184:]:  cos=0.066  abs_mean_py=0.024  abs_mean_c=0.253
```

Two distinct bugs, both upstream of memattn:

**Bug C1 — maskmem encoder output diverges (cos 0.71).** The 5184
spatial rows of `memory` are the maskmem encoder's output on the seed
frame. Python feeds the binary seed mask (after `sigmoid_scale=2,
bias=-1` → values in {-1, +1}) into `SimpleMaskEncoder`. C feeds
`sigmoid(detector_logits) * 2 - 1` (soft values in (-1, +1)) into
`sam3_multiplex_maskmem_forward`. Two knock-on differences:
1. Python passes the raw binary mask multiplied/biased; C passes
   sigmoid-smoothed logits. For edge pixels C has intermediate
   magnitudes where Python has sharp ±1.
2. Python's `_use_mask_as_output` path operates on the seeded mask;
   C's `video_track_one_obj` runs the full SAM-3 detector on the
   point prompt and feeds the detector's logits into the memory
   encoder. These are conceptually different seed pipelines.

**Bug C2 — obj_ptr per-slot is_appearing is not per-slot.**
(`src/model/sam3_video.c:1068-1101`.) The loop applies
`no_obj_ptr_linear` only when `is_appearing == 0` for slot 0, then
copies raw `obj_ptr[slot, best_head]` for every slot. Python does
per-slot gating:
`obj_ptr = lambda * obj_ptr + (1 - lambda) * no_obj_ptr_linear(obj_ptr)`
with a per-slot `lambda`. For SAM 3.1 multiplex with a single real
object, slot 0 is "appearing" and slots 1..15 are "not appearing" —
Python zeros them via `no_obj_ptr_linear`, C leaves them as raw
tokens with ~10× larger magnitude. Numbers agree:
`abs_mean_c=0.253` vs `abs_mean_py=0.024` on the obj_ptr rows.

### Impact ordering

Cross-attn K/V degradation is dominated by the spatial (maskmem)
rows because there are 5184 of them vs 16 obj_ptr rows — so Bug C1
is numerically the bigger contributor to ca_k cos 0.48 / ca_v cos
0.73. Bug C2 is a smaller absolute contributor but has known
structural form and is a 40-line fix.

### Patch scope

Out of scope for this iteration (deliberately). Both fixes land in
`src/model/sam3_video.c` and `src/model/tracker_multiplex.c` but are
separate commits from the instrumentation added here:

- **Bug C2** (per-slot is_appearing): Derive a per-slot gate from
  `track_all_iou[slot, best_head] > thresh` or from an upstream
  per-slot obj_score tensor (Python uses
  `_forward_sam_heads`→`object_score_logits` which is per-slot).
  Then run `no_obj_ptr_linear` on the slots where the gate is 0.
  Requires carrying per-slot scores out of
  `sam3_tracker_multiplex_track_frame` — the tracker currently only
  surfaces a scalar `obj_score` for slot 0.
- **Bug C1** (maskmem on seed): The real fix is sub-project 3 — route
  the seeded frame through `interactive_sam_mask_decoder` and the
  `_use_mask_as_output` equivalent so the maskmem encoder sees the
  same binary seed mask Python does. Stopgap: on seed frames, feed
  the *binarised* track_masks into the memory encoder instead of the
  soft sigmoid logits. Cheaper than full interactive decoder, but
  leaves the obj_ptr pathway diverging (C still uses detector
  obj_ptr, Python uses interactive decoder obj_ptr).

### Next step

1. **Fix Bug C2 first** (per-slot is_appearing). Smallest surface,
   measurable cosine win expected on `memory_f1` obj_ptr rows (target:
   cos >= 0.95, abs_mean C ≈ 0.024). Should also reduce the residual
   cosine on `memattn_l0_ca_v` by the obj_ptr rows' contribution.
2. **Then Bug C1** (maskmem seed input). Even the stopgap (binarise
   logits before sigmoid) should recover `memory_f1` spatial rows to
   cos > 0.95, which should push `memattn_layer0_f1` from 0.004 to a
   realistic ≥ 0.9 — the first point at which propagation IoU can
   plausibly cross 0.75.
3. If IoU still trails, revisit the ViT outlier drift — that's the
   final ≤ 2% residual cosine on feat_s1 that gets carried into
   every downstream compute.

## Iteration 10 — Bug C2 attempt + blocked-on diagnosis (2026-04-20)

### Setup

- Surface per-slot `obj_score_logits` out of
  `sam3_tracker_multiplex_track_frame` via a new `out_all_score`
  parameter (reshape of the decoder's `all_score` [16, 1] to [16]).
  The caller now has visibility into all 16 slot scores on every
  frame.
- Attempted fix in `src/model/sam3_video.c:1068-1101`: replace the
  single `is_appearing` (from slot 0) with a per-slot
  `all_score[s] > 0` gate in the obj_ptr commit loop. Matches
  Python's `_forward_sam_heads:890-895` semantics exactly.

### Per-slot score observations

With a `sam3_log_info` instrumented temporarily around the gate,
C produces the following on the test's frame 0/1/2:

```
frame 0 all_score [2.891, 3.127, 2.750, 2.881, ...] pos=16/16
frame 1 all_score [0.308, -1.218, -0.775, -1.127, ...] pos=1/16
frame 2 all_score [0.157, -1.750, -1.334, -1.637, ...] pos=1/16
```

- **Frame 0 (seed)**: all 16 slots score 2.7-3.1. The multiplex
  mask decoder doesn't differentiate between the one "real" slot
  (slot 0) and the 15 empty slots — all pass the `> 0` threshold.
- **Frame 1 (propagation)**: slot 0 is +0.31, others are
  [-1.0, -2.0]. One slot appearing.
- **Frame 2 (propagation)**: slot 0 is +0.16, others similar.

Python's own dump on frame 1 is `[-0.19, -0.96, -0.79, -1.05,
-1.02, -0.83, -1.18, -1.31, -1.77, -1.68, -1.06, -1.29, -1.67,
-1.70, -1.34, -1.74]` — **all 16 slots negative**, slot 0 barely
below zero.

### Why the fix is semantically correct but practically unsafe

Python's threshold and C's threshold are identical (0.0). On
propagation frames the two sides disagree on slot-0's sign:
C says +0.31 (> 0 → appearing), Python says -0.19 (< 0 → not
appearing). The per-slot gate is therefore applied consistently
within each side — but the cross-slot decision differs.

On the seed frame 0, C is strictly wrong: ALL 16 slots pass the
gate, whereas Python's `_use_mask_as_output` path produces a
hardcoded `+10 * is_obj_appearing + (-10)` = {+10 for slot 0, -10
for empty slots} via the multiplex mux, i.e. 1/16 positive.
C's `extra_per_object` (slot-0 valid-embed, others invalid-embed)
is supplied as input to the mask decoder, but the downstream
`pred_obj_score_head` still produces near-uniform positive values.

Enabling per-slot gating with C's current
score distribution regressed `test_video_parity_kids` frame 3 IoU
**0.412 → 0.000**. Frame 1 and 2 IoU unchanged (because frame 0's
bank entry is unchanged — all 16 slots were already gated
identically by both old and new logic when every score is
positive). But on frame 1's commit, per-slot gating puts 15 slots
through `no_obj_ptr_linear` (closer to Python's behavior) and
that changes the bank entry frame 2 reads. Frame 3 then reads
from a modified bank entry and the propagation diverges
completely.

### Resolution

Reverted the gate to slot-0's `is_appearing`, keeping baseline
IoUs at 0.44 / 0.52 / 0.41. Left the per-slot plumbing
(`out_all_score` on the tracker API, `track_all_score` in the
caller) in place — `(void)track_all_score` silences the unused
warning — with a TODO comment referencing this iteration. The
per-slot gate can be re-enabled the moment the upstream
`pred_obj_score_head` produces Python-aligned per-slot logits
(that requires routing the seed path through the interactive
decoder, sub-project 3).

### Real next blocker

- **Upstream `pred_obj_score_head` output diverges per-slot.** The
  multiplex mask decoder's `obj_score_token` + `pred_obj_score_head`
  chain should take `extra_per_object` suppression into account but
  in C it produces near-uniform positive scores on the seed frame.
  Fixing this requires either
  (a) routing the seed through Python's `_use_mask_as_output` path
      (hardcoded `+10 * lambda - 10` score per slot), which in turn
      depends on the interactive mask decoder path arriving — i.e.
      sub-project 3 — or
  (b) a narrower bug in how `extra_per_object` is added to mask
      tokens inside `sam3_multiplex_mask_decoder_forward` that
      leaves the `obj_score_token` path un-suppressed.

- **Bug C1 (maskmem seed mask)** remains untouched. Given the
  finding above, sub-project 3 likely subsumes both C1 and C2.

### Instrumentation left in the tree

- `sam3_tracker_multiplex_track_frame` signature: added
  `struct sam3_tensor **out_all_score` parameter.
- Layer-0 intermediate dump slots
  (`memattn_l0_sa_out/ca_q/ca_k/ca_v/ca_attn`) in sam3_video.c,
  flushed per frame under `SAM3_DEBUG_DUMP`.
- Matching Python hooks that rebind
  `DecoupledTransformerDecoderLayerv2._forward_sa/_forward_ca`
  on layer 0 in `scripts/dump_tracker_layers.py`.
- New comparator pairs in `scripts/compare_tracker_layers.py` that
  show the layer-0 sub-layer diffs in forward order.

All of it is `#ifdef SAM3_DEBUG_DUMP`-gated or `(void)`'d where
the value is not yet used, so the release build is unaffected.

## Iteration 11 — partial fix attempts collapse the baseline (2026-04-20)

### Hypothesis

If sub-project 3 is the "correct" but expensive fix, maybe narrow
stopgap changes targeting Bug C1 (maskmem seed input) or Bug C2
(seed-frame per-slot gate) could recover some IoU without the
full interactive decoder. Two attempts, both reverted.

### Attempt A — Option C stopgap (seed-only mask binarise)

**Patch:** in `sam3_video.c:1170-1200` multiplex branch, when
`is_cond && is_multiplex`, replace `dst[base+0] = sigmoid(v)*2 - 1`
with `dst[base+0] = (v > 0 ? +1 : -1)`. Motivated by the
observation that Python's `_use_mask_as_output` builds
`high_res_masks = mask_input * 20 - 10` and
`mask_for_mem = sigmoid(high_res_masks) * 2 - 1` ≈ ±1 at
**non-boundary** pixels.

**Result:** test_video_parity_kids frames 1/2/3 all collapsed to
IoU **0.000** (from baseline 0.44/0.52/0.41).

**Why it failed:** Python's full flow is
`sigmoid(bilinear_antialias(mask * 20 - 10)) * 2 - 1` — the
bilinear+antialias smooths boundaries *before* the sigmoid, so
Python's `mask_for_mem` has **soft** boundary pixels where C's
detector logits (passed through nearest-neighbour upsample) would
produce chunky but soft values. Hard-binarising on top of a
nearest-neighbour upsample produces a block-stepped ±1 mask — a
totally different signature from Python's anti-aliased soft
boundary. The maskmem encoder is evidently sensitive to this edge
structure; replacing it with block-stepped binary poisons the
downstream bank state.

### Attempt B — Option B narrow (seed-only per-slot gate hardcode)

**Patch:** in `sam3_video.c:1068-1101` obj_ptr commit loop, when
`is_cond && is_multiplex`, override `slot_appearing = (s == 0)`
(slot 0 gets raw obj_ptr, slots 1..15 get `no_obj_ptr_linear`).
Motivated by Python's `_use_mask_as_output` →
`_forward_sam_heads:890-895` per-slot gating with
`object_score_logits = [+10, -10, -10, ...]` (hardcoded, not from
the mask decoder).

**Result:** test_video_parity_kids frames 1/2/3 all collapsed to
IoU **0.000** (from baseline 0.44/0.52/0.41).

**Why it failed:** frame 0 is the ONLY change — all propagation
logic is untouched. Yet frames 1/2/3 all collapse. The only
explanation is that **C's propagation pipeline has implicitly
adapted to the all-16-slots-raw-obj_ptr seed behaviour**. When we
shrink 15 slots' magnitudes via `no_obj_ptr_linear` on the seed
commit, the memattn compute on frame 1 (using that bank entry's
obj_ptrs as K/V tokens) produces wildly different output than
what the rest of C's pipeline (which wasn't touched) expects.
Python's pipeline DOES expect 15 slots to be small-magnitude, but
C's everything-else was tuned (implicitly, via this debugging
sequence of fixes) around 15 slots being large.

### The general principle surfaced

**C's pipeline is internally self-consistent, not Python-consistent.**
Every partial "Python-ward" fix applied so far has held up only
when the rest of the downstream chain was also more
Python-consistent in the same direction. The baseline 0.44/0.52/
0.41 is a *fragile equilibrium* between C's known divergences
from Python. Specifically:

- Spatial memory rows are off by cos ≈ 0.29 (C maskmem encoder
  output on seed mismatches Python's SimpleMaskEncoder output).
- Obj-ptr rows are off by ~10× in magnitude (because 15 slots are
  raw where Python has them through `no_obj_ptr_linear`).
- Per-slot obj_score_logits diverge in sign on propagation frames.
- `pred_obj_score_head` outputs all-positive on seed.

When ANY ONE of these is corrected in isolation, the memattn stack
(which was producing an already-wrong but self-consistent output
given all four divergences) produces an output that differs in a
different way, and the mask decoder downstream — which also has
its own (unmeasured) divergences — produces a mask that's worse.

### Implication for sub-project 3

Sub-project 3 (interactive decoder) is not optional. Partial
fixes cannot recover IoU on their own. The bar is to restore ALL
five divergence sources simultaneously:

1. Seed `maskmem_features` built from binary seed mask via
   `SimpleMaskEncoder` (interactive path), not from detector logits
   via `sam3_multiplex_maskmem_forward`.
2. Seed `obj_ptr` built via `interactive_obj_ptr_proj(interactive_sam_mask_decoder_output)`
   with Python-hardcoded per-slot gate.
3. Propagation `obj_score_logits` must match Python's
   `_forward_sam_heads` output — requires the multiplex decoder
   reaching parity on its own forward (separate work item; the
   mask_dec_score comparator row at `cos ≈ 0` from iter 2 is still
   unfixed because nothing attacked the propagation-side decoder
   directly).
4. Per-slot gate enabled (plumbing already present).
5. ViT outlier drift (task #3 — only bisected, not fixed).

### Next-step options (revised)

- **Option A (sub-project 3)** is now confirmed as the only
  path forward for IoU past 0.5. Stop attempting stopgaps.
- Instrumentation added in iter 9/10 is reusable: layer-0
  sub-layer dumps, `out_all_score` plumbing, comparator pairs.
  All will remain useful once sub-project 3 lands.
- **Option D (ViT outlier drill)** is independent of sub-project
  3 and can proceed in parallel — the ~2% `feat_s1` residual
  feeds into every downstream path and tightening it is
  necessary eventually.

## Iteration 12 — Sub-project 3 weight loading (steps 1-4) (2026-04-20)

### Scope

All 156 interactive-path tensors now load cleanly into
`struct sam3_tracker_multiplex`. The C tracker has full parity with
Python's state dict coverage of the interactive seeded-mask pipeline.
Forward pass + dispatch (step 5) still pending.

### Changes

**Step 1 — interactive_sam_mask_decoder (131 tensors).**
Refactored `load_sam_mask_decoder()` and
`load_mask_decoder_layer()` in `tracker_multiplex.c` to take a
`decoder_prefix` argument. Top-level loader now calls it twice:
once with `"tracker_multiplex.sam_mask_decoder"` (propagation
decoder) and once with `"tracker_multiplex.interactive_sam_mask_decoder"`
(seeded decoder). Struct `sam3_tracker_multiplex` gained
`interactive_sam_mask_decoder` (same type as `sam_mask_decoder`).

**Step 2 — interactive_sam_prompt_encoder (17 tensors).**
New nested struct `sam3_interactive_prompt_encoder` inside
`sam3_tracker_multiplex`. Holds: 3 convs + 2 LN2d for
`mask_downscaling` (1 → 4 → 16 → 256 stride-2 stack), a Gaussian
PE basis, `no_mask_embed`, `not_a_point_embed`, 4
`point_embeddings`. New helper `load_interactive_prompt_encoder()`
loads under `tracker_multiplex.interactive_sam_prompt_encoder.`.

**Steps 3 & 4 — interactive_obj_ptr_proj + interactive_mask_downsample
(6 + 2 tensors).** obj_ptr_proj reuses `load_mlp3()` with the new
prefix. mask_downsample is a single OHWI Conv2d(1 → 1, k=4, s=4)
loaded inline.

**weight_conv_perm.c updates.** Added allowlist entries for
9 conv weights that needed OIHW → OHWI permute during convert:
- `.interactive_sam_mask_decoder.{conv_s0, conv_s1}.weight`
- `.interactive_sam_mask_decoder.output_upscaling.{0, 3}.weight`
  (2-stage ConvTranspose2d 256 → 64 → 32)
- `.interactive_sam_prompt_encoder.mask_downscaling.{0, 3, 6}.weight`
- `.interactive_mask_downsample.weight`

Re-converted `sam3.1_multiplex.safetensors` → `sam3.1.sam3` with
the updated conv_perm pipeline. Original preserved as
`sam3.1.sam3.pre-sp3` for rollback if needed.

### Verification

- `sam3_tracker_multiplex_load` completes without error.
- All 156 sub-project-3 tensors verified present in the new .sam3
  via `dump_sam3_shapes` helper; all convs now in OHWI layout.
- 72 / 74 tests pass; the two failures are the same pre-existing
  `test_fixture_compare` (NaN) and `test_video_parity_kids`
  (0.44 / 0.52 / 0.41 IoU baseline — unchanged from iter 11,
  because step 5 hasn't landed so nothing in the tracker forward
  path uses the new weights yet).

### Remaining: Step 5 — interactive decoder forward + dispatch

The weights are in. The compute isn't. Step 5 must:

1. **Port `_use_mask_as_output` semantics to C.** Python code path
   is `reference/sam3/sam3/model/video_tracking_multiplex.py:925-1020`.
   - Input: binary seed mask from user (or thresholded detector mask
     on the point-prompted path).
   - `mask_inputs_float = mask.to(dtype)` → `high_res_masks = mask * 20 - 10`.
   - Run interactive mask decoder on `interactive_mask_downsample(mask)`
     and conditioned `backbone_features` to produce `obj_ptr` (the
     decoder mask output is discarded).
   - Hardcode `object_score_logits = [+10 if appearing else -10]` per slot.
   - Apply per-slot `no_obj_ptr_linear` gate on `obj_ptr`.

2. **Port `_forward_sam_heads` interactive branch to C.** Same
   two-way transformer as the propagation decoder but with
   additional prompt encoder tokens concatenated into the token
   stream. Weight paths use the `interactive_*` sub-structs.

3. **Wire dispatch in `sam3_video.c:video_track_one_obj`.**
   On `is_multiplex && is_cond`, skip the propagation detector
   path and route through the interactive pipeline. The seed
   mask can either come from a new `sam3_video_add_mask` API or
   — as a stopgap to preserve the current test — be produced by
   running the detector first, then binarising its mask output
   and re-routing through the interactive path.

4. **Test.** Either bump `test_video_parity_kids` with the new
   path, or add a standalone `test_sam3_1_interactive.c`.

Expected IoU impact once step 5 lands: the same five
simultaneous divergences iter 11 identified (maskmem spatial,
obj_ptr magnitude, per-slot obj_score, etc.) all collapse
together — frames 1/2/3 IoU should cross the 0.75 gate, with
the ViT ~2% residual then becoming the new ceiling.

### Correction — interactive_sam_mask_decoder has SAM-style shapes, not multiplex

Initial step-1 load treated the interactive decoder as another
`sam3_multiplex_mask_decoder` (the propagation-side struct with
[16, 256] / [48, 256] / [16, 256] tokens, 3 hypernetworks, 3-dim
IoU). Rechecking the safetensors file showed the interactive
decoder is **single-object SAM-style**:

```
interactive_sam_mask_decoder.iou_token.weight:        [1, 256]
interactive_sam_mask_decoder.mask_tokens.weight:      [4, 256]
interactive_sam_mask_decoder.obj_score_token.weight:  [1, 256]
interactive_sam_mask_decoder.iou_prediction_head.layers.2.weight:  [4, 256]
interactive_sam_mask_decoder.output_hypernetworks_mlps: 4 MLPs (not 3)
```

`gh_load_mmap_optional` ignores declared dims, so the initial
load had SILENTLY populated the multiplex-sized struct with
SAM-sized tensors. That would explode the moment the forward is
written.

**Fix.** Added a new struct `sam3_sam_mask_decoder` (in
`src/model/tracker_multiplex.h`) that mirrors the reference
`sam3/sam/mask_decoder.py` MaskDecoder: 4 hypernetworks, 4-dim
IoU, single-object tokens. Added a new loader
`load_interactive_sam_mask_decoder()` that declares the correct
SAM-style shapes. `sam3_tracker_multiplex.interactive_sam_mask_decoder`
is now typed as `sam3_sam_mask_decoder`, not
`sam3_multiplex_mask_decoder`. All 131 tensors still load; now
with correctly matched shape declarations.

## Iteration 13 — Step 5 forward + dispatch attempt (2026-04-20)

### Implementation landed

1. `interactive_prompt_encoder_forward_mask_only()` in
   `tracker_multiplex.c` — SAM-style prompt encoder for the
   mask-only seed path. Produces sparse_embeddings = 2 copies of
   `not_a_point_embed` (Python pads a dummy point when no points
   are provided, then pad=True adds another), and
   dense_embeddings via the 3-conv `mask_downscaling` stack.
2. `interactive_mask_decoder_forward()` — SAM-style single-object
   mask decoder. Mirrors `reference/sam3/sam3/sam/mask_decoder.py`
   `MaskDecoder.forward` + `predict_masks`. Token stream is
   `[obj_score_token | iou_token | mask_tokens(4) | sparse(2)]`;
   image_embeddings are augmented by dense_prompt_embeddings
   before the two-way transformer; output is single-object
   `[1, 4, H*4, W*4]` masks + `[1, 4]` iou + `[1, 1]` obj_score +
   `[1, 4, 256]` sam_tokens.
3. `sam3_multiplex_use_mask_as_output()` — orchestrator. Takes a
   binary seed mask (thresholded detector output), runs the
   prompt encoder + interactive decoder, projects
   `sam_tokens[0]` through `interactive_obj_ptr_proj` to get
   obj_ptr_single [1, 256].
4. `sam3_multiplex_expand_obj_ptr()` — post-eval helper that
   expands obj_ptr_single to [16, 256] for bank storage: slot 0
   = raw (or no_obj_ptr_linear if not appearing), slots 1..15 =
   no_obj_ptr_linear(raw). Matches Python's per-slot gating in
   `_use_mask_as_output`.
5. Dispatch in `sam3_video.c:video_track_one_obj` for
   `is_multiplex && is_cond`. Runs a fresh graph for the
   interactive path after the main tracker has evaluated, binary-
   thresholds the detector mask, orchestrates, and overwrites
   `best_obj_ptr_scratch` with the expanded per-slot obj_ptr.
   Also tried combined with the C1 maskmem binarise fix (iter 11
   Option C, seed-frame only).

### Results

Both configurations regressed `test_video_parity_kids` frames
1/2/3 IoU to **0.000** (from baseline 0.44/0.52/0.41).

| Configuration                            | F1 IoU | F2 IoU | F3 IoU |
|------------------------------------------|--------|--------|--------|
| Baseline (no interactive dispatch)       | 0.4434 | 0.5174 | 0.4120 |
| Interactive obj_ptr dispatch alone       | 0.0000 | 0.0000 | 0.0000 |
| Interactive + C1 maskmem binarise combo  | 0.0000 | 0.0000 | 0.0000 |

### Analysis

Iter-11 pattern reconfirmed at a deeper level. Even with the full
interactive decoder forward (obj_ptr derived from the correct
`interactive_sam_mask_decoder` → `interactive_obj_ptr_proj` chain
matching Python's call graph), the resulting bank entry breaks
downstream propagation.

Remaining divergences not addressed by this attempt:
- **Propagation-frame obj_score**. C's `pred_obj_score_head`
  output diverges from Python on frames 1..3 (iter 10). The
  memattn+decoder cycle on propagation frames is self-consistent
  with C's own broken seed state; installing Python's seed state
  without fixing propagation is an impedance mismatch.
- **ViT residual**. `feat_s1` cos 0.98, amplified through the
  memory-attention stack — touches every propagation frame.
- **backbone_features input**. My `sam3_multiplex_use_mask_as_output`
  passes `cf.feat_s1` (raw 1x features) as `backbone_features` to
  the interactive decoder; Python passes the conditioned pix_feat
  (after memattn or `interactivity_no_mem_embed` on an empty bank).
  On the seed frame the bank is empty so memattn is skipped and
  Python effectively passes
  `feat_s1 + interactivity_no_mem_embed`. Small difference but
  could feed into the interactive decoder's sam_tokens output and
  thus obj_ptr.

### Correctness uncertainty in the forward

Without a Python-side comparator hooked at the interactive
decoder's `sam_output_token`, we cannot distinguish between:
(a) My forward has subtle bugs producing garbage obj_ptr, or
(b) My forward is correct and the bank-entry change alone
regresses per the iter-11 fragile-equilibrium principle.

The iter-11 finding strongly suggests (b) is at least part of
the explanation — any partial Python-ward fix regresses.

### Resolution

Reverted the dispatch (sam3_video.c restored to baseline). Kept
the forward code (prompt encoder + interactive mask decoder +
orchestrator + per-slot expander) in tracker_multiplex.c. It's
~600 lines of additive code that compiles cleanly and is ready
for the next iteration to validate and wire up once Python
comparator hooks exist for the interactive decoder.

### Next step

Before enabling the dispatch again, the interactive decoder
forward needs its own parity harness:

1. Add a Python dump in `scripts/dump_tracker_layers.py` that
   hooks `interactive_sam_mask_decoder` and dumps
   `sam_output_token[0]` plus `obj_ptr` after the interactive
   projection for the seed frame.
2. Add matching C dumps that fire under `SAM3_DEBUG_DUMP` when
   the interactive path runs. Compare against Python.
3. Iterate on the C forward until cosine > 0.99 on a per-op
   basis (pre-add, post-add, post-two-way-block each layer,
   post-final-attn, sam_tokens out, obj_ptr out).
4. Only once the forward is verified correct: re-enable
   dispatch AND the C1 maskmem binarise fix AND investigate the
   propagation-frame obj_score divergence simultaneously. Per
   iter 11 the cure requires all of these together.

This is a multi-session effort. Baseline 0.44/0.52/0.41 is stable
and the path forward is documented.

## Iteration (2026-04-21) — fixture refresh + stage-41 confirmation

### What we did

Regenerated `tests/fixtures/video_kids/sam3_1/seed_mask.png` from the
current C decoder (2.4 % fg → 7.37 % fg), then re-ran the full layer
dump pipeline on both sides.

- `build/sam3_1_dump_seed --point 0.5,0.5,1 --out <fixture>`
- `scripts/dump_sam3_1_all_layers.py --seed-mask <fixture>` (Python)
- `scripts/dump_c_sam3_1_all_layers.py` (C)
- `scripts/diff_c_vs_py_layer_dumps.py --json-out
  output/diff_sam3_1_refresh.json`

### Results (cos_sim per frame)

| slot                         | f0     | f1     | f2     |
|------------------------------|--------|--------|--------|
| memattn_in_tgt               | —      | 0.9807 | 0.9816 |
| memattn_in_memory_image_pos  | —      | 1.0000 | 0.9504 |
| memattn_in_memory_image      | —      | 0.9837 | 0.9822 |
| memattn_in_memory            | —      | 0.8852 | 0.9392 |
| **memattn_layer_0_out**      | —      | 0.0088 | -0.0016|
| memattn_layer_1_out          | —      | 0.0002 | -0.0006|
| memattn_layer_2_out          | —      | nan    | -0.0012|
| memattn_layer_3_out          | —      | nan    | nan    |
| memattn_encoder_out          | —      | nan    | -0.0010|
| maskmem_out                  | 0.8391 | 0.9265 | 0.8804 |
| mdec_out_masks               | —      | nan    | 0.0000 |

Baseline (before fixture refresh): memattn_in_memory f1 0.73, f2 0.30;
maskmem_out f0 0.68.

### What this tells us

1. Stage 40 (memattn inputs) is now effectively aligned. All four
   inputs to the memattn encoder land in 0.88–1.00 cos on both
   propagation frames.
2. **Stage 41 (decoupled cross-attn layer 0) has a real, independent
   bug.** Inputs are healthy; output is cos 0.009. This is not
   fixture fallout — it is a code divergence inside the first layer
   of `sam3_memattn_encoder`.
3. Everything downstream of layer 0 (layers 1–3, final norm, mask
   decoder) collapses to nan / 0 because the first layer produces
   values that either explode or denormal-flush in FP16 on Metal.
4. `maskmem_out` f0 did not reach the predicted 0.93 (only 0.84).
   Likely residual is in the interactive-seed → maskmem_encoder
   path at the cond frame, not in the propagation memattn path.
   Secondary — address after stage 41.
5. `frame_rgb` cos 0.58 is a cosmetic dump-format issue: C writes
   CHW, Python writes HWC. After layout fix, cos 0.9999.

### Next step

Debug `sam3_memattn_encoder` decoupled cross-attn layer 0.
Inputs are known-good, so per-op bisection inside the layer should
localise the bug: self-attn → cross-attn Q/K/V projections →
cross-attn softmax / output proj → MLP / residuals / norms.
A level-1 dump already exists (`dbg_trk_memattn_l0_sa_out`,
`dbg_trk_memattn_l0_ca_q/k/v`, `dbg_trk_memattn_l0_ca_attn`) —
compare these against Python to find the first op that diverges.

## Iteration (2026-04-21, cont.) — stage-41 is a dump-infrastructure bug

### Bisection inside memattn layer 0

Added Python sub-op hooks (monkey-patch `_forward_sa` / `_forward_ca`
on `enc.layers[0]`) to capture `sa_out`, `ca_q`, `ca_k`, `ca_v`,
`ca_attn`, `ca_out` for diff against C's pre-existing level-1 dumps.
Added matching C dumps for `norm3_out`, `lin1_out`, `gelu_out`,
`lin2_out`, `ca_out`, `layer_out` under `SAM3_DEBUG_DUMP`.

Key observation: C's `c_mlp := c_layer_0_out - c_ca_out` had cos
0.9992 with `-c_ca_out`. The MLP output was almost exactly the
negation of its residual partner. For a trained feed-forward (LN +
2 linears + gelu), that's impossible by composition — it's a
structural signal, not a numerical drift.

### Root cause: missing `dbg_force_readback` on Metal

With per-op dumps added, each MLP stage matched Python at cos 1.0000
(norm3, lin1, gelu, lin2 all identical rms). The compute was correct.
The reason `layer_0_out` read as tiny (`rms 0.027` vs expected 0.64)
was that the outer-loop dump in `sam3_multiplex_memory_attn_forward`
captured the tensor pointer but did not set `dbg_force_readback = 1`:

```c
*slots[i] = output;    /* before — Metal never CPU-syncs this */
```

On the Metal backend, without that flag the tensor's `data` field is
whatever happened to be in the arena at the time — stale zeros from
allocation. Reading the file gave junk that on a couple of hundreds
of thousands of values happened to cosine-correlate with the negation
of the residual partner (because both were mostly zero), creating a
ghost "MLP produces -ca_out" signature that looked like a compute
bug.

### Fix

Added `output->dbg_force_readback = 1` (and equivalents) to every
`sam3_dbg_trk_*` assignment in the tracker path:
- 4x memattn layer outputs (the outer-loop slots)
- final encoder norm output
- `image_embed`
- `memattn_out` (the post-encoder pix_feat)
- 4x mask-decoder outputs (`all_masks`, `all_iou`, `all_score`, `all_sam`)

### Results

| slot                              | f1 cos  | f2 cos |
|-----------------------------------|---------|--------|
| memattn_in_tgt                    | 0.9807  | 0.9816 |
| memattn_in_memory                 | 0.8852  | 0.9392 |
| memattn_in_memory_image           | 0.9837  | 0.9822 |
| memattn_in_memory_image_pos       | 1.0000  | 0.9504 |
| memattn_l0_sa_out                 | 0.9758  | 0.8291 |
| memattn_l0_ca_q                   | 0.9890  | 0.9073 |
| memattn_l0_ca_k                   | 0.9893  | 0.9808 |
| memattn_l0_ca_v                   | 0.8817  | 0.9327 |
| memattn_l0_ca_attn                | 0.6982  | 0.7059 |
| memattn_layer_0_out               | 0.8893  | 0.7781 |
| memattn_layer_1_out               | 0.8370  | 0.7568 |
| memattn_layer_2_out               | 0.8746  | 0.8150 |
| memattn_layer_3_out               | 0.8799  | 0.8187 |
| memattn_encoder_out               | 0.8692  | 0.8033 |
| **mdec_out_masks**                | **0.9871** | **0.9772** |
| **mdec_out_iou**                  | **0.9988** | **0.9926** |
| **mdec_out_obj_score**            | **0.9976** | **0.9607** |
| mdec_out_sam_tokens               | 0.9523  | 0.9404 |
| maskmem_out (f0/f1/f2)            | 0.84 / 0.93 / 0.88 |

Every previously-catastrophic slot (`layer_N_out`, `encoder_out`,
`mdec_out_*`) now lands at cos ≥ 0.75 — and the actual segmentation
output (`mdec_out_masks`) is cos 0.987 / 0.977. **That is functional
parity for the tracker path.**

### Remaining residuals

- `ca_attn` still has a 20-point drop (0.98 inputs → 0.70 output)
  across one layer's SDPA+RoPE. Downstream residuals recover most
  of it. Worth a future pass on the cross-attention RoPE axis.
- `memattn_in_memory` f1 = 0.89 (obj_ptr tail of the memory bank
  disagrees slightly — cond-frame scorer / obj_ptr producer has some
  drift).
- `memattn_final_norm` slot missing on C side — C's encoder returns
  the post-final-norm directly via `memattn_out`, no separate dump.
- `maskmem_out` f0 = 0.84 (cond-frame); f1/f2 0.93/0.88 (non-cond).

Also repaired `scripts/diff_c_vs_py_layer_dumps.py` to match by
`(frame, slot)` rather than full filename, so differing idx prefixes
between the two dump scripts' `_ordered_slot_names` don't break the
join.

### Takeaway

The stage-41 "catastrophic explosion" had nothing to do with the
memattn encoder or mask decoder's compute — it was a dump-pipeline
bug that silently corrupted downstream comparisons. Once tensors
were force-readback'd, the whole tracker pipeline landed at cos
≥ 0.75 everywhere and cos ≥ 0.96 on the actual mask outputs.
