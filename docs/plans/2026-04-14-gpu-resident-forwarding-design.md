# GPU-Resident Forwarding Design

**Date:** 2026-04-14
**Status:** Approved
**TODO item:** #5 (Multi-stream pipelining — reimagined as zero-copy forwarding)

---

## Problem

All GPU work runs on a single `mlx_stream`. Between consecutive `graph_eval`
calls (e.g., ViT batch 0 → batch 1), Phase 3 reads results back to host and
the next Phase 1 re-uploads them to GPU. For the ViT's 8 batch evaluations,
this creates 7 unnecessary readback-then-reupload round-trips of the ~5.3 MB
residual stream tensor.

## Approach: `no_readback` flag

Instead of adding a second MLX stream (the original TODO plan), skip Phase 3
readback entirely for intermediate stages. The output `mlx_array` stays in the
persistent tensor map. The next `graph_eval` finds it via `metal_wrap_tensor`
and uses it directly — zero data transfer.

### Why not multi-stream?

Multi-stream would overlap Phase 3 of block N with Phase 1 of block N+1. But
Phase 1 is fast (~ms of CPU graph building), so the overlap window is small.
GPU-resident forwarding eliminates the readback entirely — strictly better.

## Design

### 1. Graph-level flag

```c
struct sam3_graph {
    struct sam3_node nodes[SAM3_GRAPH_MAX_NODES];
    int              n_nodes;
    bool             no_readback;
};
```

`sam3_graph_init()` sets `no_readback = false` (safe default).

### 2. Metal backend changes (`metal_graph_eval`)

When `g->no_readback == true`:
- Phase 2 (mlx_eval) runs normally
- Phase 3 replaced with: evict within-graph intermediates only (already
  identified by Phase 1.5), keep final outputs in the tensor map
- Phase 4 (ephemeral eviction) runs normally

When `g->no_readback == false`: current behavior (full readback + eviction).

### 3. ViT caller changes (`image_encoder.c`)

- Allocate residual tensor once in persist arena (stable pointer across
  scratch resets)
- Reuse same tensor as both output of batch N and input of batch N+1
- Set `g.no_readback = true` for batches 0–6, `false` for batch 7

### 4. FPN caller changes (`sam3_image.c`)

Within each FPN scale, intermediate graph_eval calls (maxpool, early convs)
set `no_readback = true`. Final call per scale reads back (outputs consumed
by CPU-based mask decoder).

## Scope

- **ViT:** 7 eliminated readback cycles (batches 0–6 of 8)
- **FPN:** ~11 eliminated readback cycles (intermediate calls within 4 scales)
- **Mask decoder:** Not affected (runs on CPU backend)
- **CPU backend:** Ignores the flag (Phase 3 is a local memcpy)

## Data transfer savings

Per eliminated cycle: ~10.6 MB round-trip (5.3 MB readback + 5.3 MB re-upload)
for the ViT residual. Total: ~74 MB for ViT + additional savings from FPN.

## Edge cases

1. **Tensor lifetime:** Residual tensor header must live in persist arena (not
   scratch) so the map key survives scratch resets.
2. **GPU memory:** One live residual tensor (~10 MB F16). Freed on final
   readback eviction.
3. **Stale host data:** Caller must not read `output->data` after `no_readback`.
   This is opt-in, so caller's responsibility.
4. **Map capacity:** +1 entry per skipped readback. Negligible vs 8192 slots.

## Testing

1. **Correctness:** Two sequential graphs with `no_readback=true` on first.
   Verify second graph's output matches baseline.
2. **Map retention:** After `no_readback` eval, verify output tensor remains
   in tensor map.
3. **Final readback:** After `no_readback=false` eval, verify host data is
   correct.
