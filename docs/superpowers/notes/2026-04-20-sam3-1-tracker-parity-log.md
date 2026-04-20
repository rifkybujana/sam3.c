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

