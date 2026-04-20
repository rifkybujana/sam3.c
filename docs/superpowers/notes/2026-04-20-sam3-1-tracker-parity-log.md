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
