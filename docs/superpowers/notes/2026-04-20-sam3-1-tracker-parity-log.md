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
