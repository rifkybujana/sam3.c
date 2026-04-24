# Batched Segmentation Status (feat/batched-decoder)

## What landed

A 25-commit series introducing graph-level batching (B > 1) to SAM3
inference. Every new substep is parity-proved against the serial path at
B=1..3 with unit tests:

- `sam3_decoder_ffn_batched`, `sam3_decoder_build_layer_batched`,
  `sam3_decoder_build_final_batched` (per-slot token streams).
- `sam3_image_model_segment_batched` — end-to-end batched driver wiring
  Stages 1+2 (geom + fusion, serial per slot), Stage 3 (CPU prompt
  encoder), Stage 4 (batched Metal decoder).
- `sam3_segment_batch` public API (single call, single failure path).
- FPN correctness fixes: `gh_add` arg-order swap in `segmentation.c`
  and removal of the eager `gh_broadcast_batch` on shared image features
  in `sam3_image.c`. Both needed for any B > 1 caller.
- A `bench_segment_batch` harness for wall-clock comparison.

## In-tree but NOT on default code path

`sam3_segment_batch` dispatches a serial `sam3_segment` loop. The batched
builders (`*_batched`) and `sam3_image_model_segment_batched` remain
callable directly but are unreferenced by the public API.

## Benchmark (full-res Metal, single text prompt, 5 iters + warmup)

    B | serial | batched | speedup
    --+--------+---------+--------
    1 |  1563  |  1210   | 1.29x
    2 |  2400  |  2917   | 0.82x   (-18%)
    4 |  5340  |  5668   | 0.94x   (-6%)
    8 | 10505  |  8458*  | 1.24x*  (*groupnorm OOM, returns early)

## Bottlenecks

1. Stages 1+2 remain serial per slot with CPU-side memcpy stacking —
   dominates at B>=2.
2. Stage 3 on CPU, Stage 4 on Metal — every batch pays the backend hop.
3. RPB buffer (re-)allocated per decoder layer, scaling with B.

## Follow-up TODOs

1. Batch Stages 1+2 — build a truly batched fusion graph instead of
   per-slot loop + trailing `stack`.
2. Keep the whole batched pipeline on a single backend (Metal) to
   eliminate the Stage 3 / Stage 4 hop.
3. Size the RPB buffer once outside the decoder layer loop.
