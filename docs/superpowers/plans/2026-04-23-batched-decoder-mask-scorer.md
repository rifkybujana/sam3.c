# SAM3 Batched Decoder / Mask-Head / Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `sam3_segment_batch(N prompt sets)` into a graph-level batched segmentation so the DETR decoder, mask head, and objectness scorer run once with leading batch dim `B = N`, instead of being invoked N times in serial.

**Architecture:** Per-set text encode, geometry encode, and encoder fusion remain serial (those stages are either cheap, cached, or per-set-text). Each set's fused encoder output `[n_pixels, d]` is persisted and stacked into `[B, n_pixels, d]` at the boundary between fusion and decoder. From that boundary onward, every tensor carries a leading batch dim: queries `[B, 200, d]`, ref_boxes `[B, 200, 4]`, RPB `[B, n_heads, 200, n_pixels]`, FPN features `[B, H, W, d]`, mask logits `[B, 200, H, W]`, scorer output `[B, 200, 1]`. Most `gh_*` ops (matmul, linear, conv2d, layernorm, reshape, transpose) already thread a leading batch dim transparently. The non-trivial batching work is concentrated in: per-head SDPA call sites (need `[B, nq, hd]` handling — reshape to `[B, 1, nq, hd]` 4D pattern), the CPU-only `sam3_decoder_compute_rpb`, the CPU-only `cpu_box_refine`, and the CPU-only `sam3_decoder_compute_query_pos`. Scorer mean pooling over the prompt sequence also needs a per-batch weight.

**Tech Stack:** C11 / clang, CMake, MLX + Metal backend (`src/backend/metal/`), pthread, sam3 arena + graph helpers (`src/model/graph_helpers.c`), single-binary tests under `tests/` via CTest.

---

## File Structure

New/modified files, with one-line responsibilities:

- **Modified: `src/model/decoder.h`, `src/model/decoder.c`** — all `sam3_decoder_build_*` functions accept a leading `B` dim in `q`/`query_pos`/`enc_features`/`rpb_mask`; `sam3_decoder_compute_query_pos` and `sam3_decoder_compute_rpb` gain a `B` parameter and produce batched CPU output.
- **Modified: `src/model/segmentation.h`, `src/model/segmentation.c`** — `sam3_seg_head_build_cross_attn` and `sam3_seg_head_*` internals accept `[B, seq, d]` / `[B, H, W, d]`; mask MLP accepts `[B, nq, d]`; dot-product mask logits produces `[B, nq, H, W]`.
- **Modified: `src/model/model_misc.h`, `src/model/model_misc.c`** — scorer accepts `[B, seq, d]` prompt, `[B, nq, d]` queries, produces `[B, nq, 1]`; mean pooling weight becomes per-batch.
- **Modified: `src/model/sam3_image.h`, `src/model/sam3_image.c`** — new `sam3_image_model_segment_batch(stacked_prompts, stacked_text, ...)` entry point next to the existing `sam3_image_model_segment`. Stages 3/4/4a-4e reshaped to operate on a leading `B` dim. The existing single-shot function becomes a thin wrapper that calls the batched version with `B=1`, to guarantee no parity drift.
- **Modified: `src/model/sam3_processor.c`** — `sam3_processor_segment_batch` runs fusion+geometry+text serially per set, persists each set's fused output, stacks into one tensor, then calls the batched image-model pipeline once; per-set results unpacked from the batched output.
- **New: `tests/test_segment_batch_parity.c`** — the parity harness. Fixture-based (uses `models/sam3.sam3` + a deterministic image). Runs single-shot segment N times, captures reference mask/score/box bytes, then runs batched segment once with N identical sets; asserts each batch slot equals the reference byte-exact. Also runs with N *different* sets to catch accidental batch-slot aliasing.
- **New: `tests/test_batched_ops.c`** — unit tests for the individual batched building blocks: `cpu_box_refine_batched`, `sam3_decoder_compute_rpb` with B=2, scorer mean pooling with B=2, SDPA with `[B, 1, nq, hd]` reshape. Uses small synthetic tensors so it can run without a model.

Guiding principles:
1. **Test B=1 equivalence at every step.** After every change, the single-shot path (now implemented as `batch with B=1`) must produce byte-identical output to the pre-refactor commit on main. A scripted parity test pins this.
2. **Every new unit test must run on Metal.** The deployment backend is Metal. Per-task tests loop over both backends: `for (enum sam3_backend_type bt : {SAM3_BACKEND_CPU, SAM3_BACKEND_METAL}) { … }` — CPU first so failures are easier to diagnose, Metal second because that's what ships. A Metal failure is a blocker; a CPU failure usually points to a logic bug before backend divergence. Use the `run_both_backends` helper introduced in Task 2.
3. **Introduce B by adding a new leading dim, not by reshaping the existing inner dim.** This minimizes the diff at call sites and keeps 2D/3D reshape-view ops intact.
4. **For per-head SDPA, wrap the batched `[B, nq, hd]` slice into a 4D `[B, 1, nq, hd]`** via `gh_reshape` and unwrap the output the same way, to reuse the existing 4D SDPA path in the Metal backend.
5. **Never mutate shared state across batch slots.** In the decoder loop, ref_boxes/query_pos/queries all become `[B, 200, *]` on the persist arena; the per-layer CPU box refinement iterates over B internally.

---

## Phase 0 — Prep

### Task 0: Worktree + branch

**Files:** none

- [ ] **Step 1: Create a worktree for this refactor**

```bash
cd /Users/rbisri/Documents/sam3
git worktree add .worktrees/batched-decoder -b feat/batched-decoder main
cd .worktrees/batched-decoder
```

- [ ] **Step 2: Sanity-build on the worktree**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target sam3 -j
```

Expected: `[100%] Built target sam3` with no new errors/warnings beyond the pre-existing `cblas_sgemm`/`conv2d_nhwc_matmul_fn`/`matmul_parallel_fn` warnings noted in the convenience-API session.

- [ ] **Step 3: Commit the empty branch marker**

```bash
git commit --allow-empty -m "feat/batched-decoder: branch start"
```

---

### Task 1: Lock down the single-shot reference with a parity harness

**Files:**
- Create: `tests/test_segment_batch_parity.c`
- Modify: `CMakeLists.txt` (register `SAM3_SOURCE_DIR` for the new test)

- [ ] **Step 1: Write the parity harness test (fails until Phase 7 because `sam3_segment_batch` today serializes per-set segment calls — we lock in the *reference bytes* now so subsequent phases can't drift)**

```c
/* tests/test_segment_batch_parity.c */
#include "test_helpers.h"
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include "sam3/sam3.h"

#ifndef SAM3_SOURCE_DIR
#define SAM3_SOURCE_DIR "."
#endif
#define MODEL_PATH SAM3_SOURCE_DIR "/models/sam3.sam3"

static int model_available(void)
{
	return access(MODEL_PATH, F_OK) == 0;
}

static void fill_deterministic(uint8_t *pix, int sz)
{
	/* Deterministic, non-degenerate pattern. Using index-based RGB so
	 * the encoded features are non-uniform and exercise all stages. */
	for (int i = 0; i < sz * sz * 3; i++)
		pix[i] = (uint8_t)((i * 31 + 17) & 0xff);
}

/* Parity: batch(N identical sets) == N × single-shot. Byte-exact on
 * masks / iou_scores / boxes / best_mask. */
static void test_batch_vs_single_identical_sets(void)
{
	if (!model_available()) { printf("  skip: no model\n"); return; }

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pix = malloc((size_t)sz * sz * 3);
	fill_deterministic(pix, sz);
	ASSERT_EQ(sam3_set_image(ctx, pix, sz, sz), SAM3_OK);

	struct sam3_prompt p = {.type = SAM3_PROMPT_TEXT, .text = "cat"};

	/* Reference: 3 serial single-shot segments */
	struct sam3_result refs[3] = {0};
	for (int i = 0; i < 3; i++)
		ASSERT_EQ(sam3_segment(ctx, &p, 1, &refs[i]), SAM3_OK);

	/* Under test: batch of 3 identical sets */
	struct sam3_prompt_set sets[3] = {
		{.prompts = &p, .n_prompts = 1},
		{.prompts = &p, .n_prompts = 1},
		{.prompts = &p, .n_prompts = 1},
	};
	struct sam3_result batch[3] = {0};
	ASSERT_EQ(sam3_segment_batch(ctx, sets, 3, batch), SAM3_OK);

	for (int i = 0; i < 3; i++) {
		ASSERT_EQ(refs[i].n_masks,      batch[i].n_masks);
		ASSERT_EQ(refs[i].mask_height,  batch[i].mask_height);
		ASSERT_EQ(refs[i].mask_width,   batch[i].mask_width);
		size_t mn = (size_t)refs[i].n_masks * refs[i].mask_height * refs[i].mask_width;
		ASSERT_EQ(memcmp(refs[i].masks, batch[i].masks, mn * 4), 0);
		ASSERT_EQ(memcmp(refs[i].iou_scores, batch[i].iou_scores,
				 (size_t)refs[i].n_masks * 4), 0);
	}

	for (int i = 0; i < 3; i++) { sam3_result_free(&refs[i]); sam3_result_free(&batch[i]); }
	free(pix);
	sam3_free(ctx);
}

/* Parity: batch(N different sets) produces the correct per-set masks.
 * Catches accidental batch-slot aliasing in the decoder state. */
static void test_batch_vs_single_different_sets(void)
{
	if (!model_available()) { printf("  skip: no model\n"); return; }

	sam3_ctx *ctx = sam3_init();
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(sam3_load_model(ctx, MODEL_PATH), SAM3_OK);

	int sz = sam3_get_image_size(ctx);
	uint8_t *pix = malloc((size_t)sz * sz * 3);
	fill_deterministic(pix, sz);
	ASSERT_EQ(sam3_set_image(ctx, pix, sz, sz), SAM3_OK);

	struct sam3_prompt pa = {.type = SAM3_PROMPT_TEXT, .text = "cat"};
	struct sam3_prompt pb = {.type = SAM3_PROMPT_TEXT, .text = "dog"};

	struct sam3_result ref_a = {0}, ref_b = {0};
	ASSERT_EQ(sam3_segment(ctx, &pa, 1, &ref_a), SAM3_OK);
	ASSERT_EQ(sam3_segment(ctx, &pb, 1, &ref_b), SAM3_OK);

	struct sam3_prompt_set sets[2] = {
		{.prompts = &pa, .n_prompts = 1},
		{.prompts = &pb, .n_prompts = 1},
	};
	struct sam3_result batch[2] = {0};
	ASSERT_EQ(sam3_segment_batch(ctx, sets, 2, batch), SAM3_OK);

	size_t mn = (size_t)ref_a.n_masks * ref_a.mask_height * ref_a.mask_width;
	ASSERT_EQ(memcmp(ref_a.masks, batch[0].masks, mn * 4), 0);
	ASSERT_EQ(memcmp(ref_b.masks, batch[1].masks, mn * 4), 0);

	sam3_result_free(&ref_a); sam3_result_free(&ref_b);
	sam3_result_free(&batch[0]); sam3_result_free(&batch[1]);
	free(pix);
	sam3_free(ctx);
}

int main(void)
{
	test_batch_vs_single_identical_sets();
	test_batch_vs_single_different_sets();
	TEST_REPORT();
}
```

- [ ] **Step 2: Register the test with SAM3_SOURCE_DIR**

Modify `CMakeLists.txt` — add alongside the existing `test_segment_batch` block:

```cmake
# test_segment_batch_parity opens models/sam3.sam3
if(TARGET test_segment_batch_parity)
	target_compile_definitions(test_segment_batch_parity PRIVATE
		SAM3_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endif()
```

- [ ] **Step 3: Build and run — must PASS today (batched impl currently loops single-shot)**

```bash
cmake --build build --target test_segment_batch_parity -j
./build/test_segment_batch_parity
```

Expected: `4 tests, 0 failures` (2 tests × 2 assertion groups). This is the invariant we must preserve through every subsequent task.

- [ ] **Step 4: Commit**

```bash
git add tests/test_segment_batch_parity.c CMakeLists.txt
git commit -m "tests/batched: parity harness for segment_batch vs per-set segment"
```

---

## Phase 1 — Batched scorer (smallest end-stage)

The scorer produces `[200, 1]` from `queries[200, d]` + `prompt[seq, d]`. Batching it first is low-risk: the MLP and projections are 2D, `gh_linear` already threads batch dims, only the mean-pool weight matrix and input reshapes need per-batch handling.

### Task 2: Add synthetic unit test for batched scorer

**Files:**
- Create: `tests/test_batched_ops.c`
- Modify: `CMakeLists.txt` (register)

- [ ] **Step 1: Write the failing test scaffold + the `run_both_backends` helper**

```c
/* tests/test_batched_ops.c */
#include "test_helpers.h"
#include "model/model_misc.h"
#include "model/sam3_processor.h"
#include "backend/backend.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "model/graph_helpers.h"

/*
 * Helper: every batched-op unit test is run on both CPU and Metal.
 * CPU first to localize logic bugs; Metal second because that's the
 * deployment backend. A Metal-only failure indicates a backend
 * divergence worth flagging to the user.
 */
typedef void (*backend_test_fn)(struct sam3_backend *be, const char *name);

static void run_both_backends(backend_test_fn fn)
{
	{
		struct sam3_backend *cpu = sam3_backend_init(SAM3_BACKEND_CPU);
		ASSERT_NOT_NULL(cpu);
		fn(cpu, "CPU");
		sam3_backend_free(cpu);
	}
	{
		struct sam3_backend *mtl = sam3_backend_init(SAM3_BACKEND_METAL);
		if (!mtl) {
			printf("  skip Metal: backend unavailable\n");
			return;
		}
		fn(mtl, "Metal");
		sam3_backend_free(mtl);
	}
}

/* When I wire batched scorer:
 * Input:  queries[B=2, 200, 256], prompt[B=2, seq=3, 256]
 * Output: scores[B=2, 200, 1]
 * Check:  result[0] == scorer(queries[0], prompt[0])  — within fp tolerance
 *         result[1] == scorer(queries[1], prompt[1])
 */
static void scorer_batched_case(struct sam3_backend *be, const char *name);

static void test_scorer_batched_equals_per_slot(void)
{
	run_both_backends(scorer_batched_case);
}

int main(void)
{
	test_scorer_batched_equals_per_slot();
	TEST_REPORT();
}
```

`scorer_batched_case` body is filled in by Task 3. Note: on Metal, byte-exact equality does NOT hold across different graphs; use `ASSERT_NEAR` with `rtol=1e-4, atol=1e-5` for Metal comparisons, and `memcmp`-exact only for CPU. The `name` parameter lets tests pick the tolerance.

- [ ] **Step 2: Register**

```cmake
# test_batched_ops — no model needed, uses synthetic tensors
# (no SAM3_SOURCE_DIR definition required)
```

- [ ] **Step 3: Build (stub main compiles with no tests yet)**

```bash
cmake --build build --target test_batched_ops -j
./build/test_batched_ops
```

Expected: `0 tests, 0 failures`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_batched_ops.c CMakeLists.txt
git commit -m "tests/batched: scaffold unit test harness for batched ops"
```

### Task 3: Batch the scorer

**Files:**
- Modify: `src/model/model_misc.h` — declare batched builder
- Modify: `src/model/model_misc.c` — implement
- Modify: `tests/test_batched_ops.c` — fill test body
- Modify: `src/model/sam3_image.c` (Stage 4e call site at line 2967+) — pass `[B, nq, d]`, `[B, seq, d]`

- [ ] **Step 1: Fill in the scorer-batched case**

Replace the forward-declared `scorer_batched_case` stub with the real body. It must run correctly on CPU and Metal — tolerance is per-backend.

```c
static void scorer_batched_case(struct sam3_backend *be, const char *name)
{
	const int d = 16;       /* small d_model for unit test */
	const int seq = 3;
	const int nq = 8;
	const int B = 2;
	const int d_ffn = 32;
	const float rtol = (be->type == SAM3_BACKEND_METAL) ? 1e-4f : 1e-6f;
	const float atol = (be->type == SAM3_BACKEND_METAL) ? 1e-5f : 1e-6f;

	struct sam3_arena ar;
	sam3_arena_init(&ar, 1 << 22);
	struct sam3_dot_scorer sc = {0};
	sam3_dot_scorer_init(&sc, d, d_ffn);
	ASSERT_EQ(sam3_dot_scorer_alloc_synthetic(&sc, &ar), 0);

	int qb_dims[] = {B, nq, d};
	int pb_dims[] = {B, seq, d};
	struct sam3_tensor *qb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3, qb_dims);
	struct sam3_tensor *pb = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 3, pb_dims);
	for (int i = 0; i < B * nq * d; i++) ((float *)qb->data)[i] = (float)(i % 13) * 0.1f;
	for (int i = 0; i < B * seq * d; i++) ((float *)pb->data)[i] = (float)(i % 17) * 0.2f;

	/* Run batched. */
	struct sam3_graph g; sam3_graph_init(&g);
	struct sam3_tensor *sb = sam3_dot_scorer_build_batched(&sc, &g, qb, pb, &ar);
	ASSERT_NOT_NULL(sb);
	ASSERT_EQ(be->ops->graph_eval(be, &g), SAM3_OK);
	ASSERT_EQ(sb->n_dims, 3);
	ASSERT_EQ(sb->dims[0], B);

	/* Run per-slot; verify each batch slot matches the unbatched reference
	 * within the backend-appropriate tolerance. */
	for (int b = 0; b < B; b++) {
		int q1_dims[] = {nq, d};
		int p1_dims[] = {seq, d};
		struct sam3_tensor *q1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 2, q1_dims);
		struct sam3_tensor *p1 = gh_alloc_tensor(&ar, SAM3_DTYPE_F32, 2, p1_dims);
		memcpy(q1->data, (char *)qb->data + (size_t)b * nq * d * 4,
		       (size_t)nq * d * 4);
		memcpy(p1->data, (char *)pb->data + (size_t)b * seq * d * 4,
		       (size_t)seq * d * 4);

		struct sam3_graph g1; sam3_graph_init(&g1);
		struct sam3_tensor *s1 = sam3_dot_scorer_build(&sc, &g1, q1, p1, &ar);
		ASSERT_NOT_NULL(s1);
		ASSERT_EQ(be->ops->graph_eval(be, &g1), SAM3_OK);

		const float *bptr = (const float *)sb->data + (size_t)b * nq;
		const float *sptr = (const float *)s1->data;
		for (int i = 0; i < nq; i++) {
			float tol = atol + rtol * fabsf(sptr[i]);
			if (fabsf(bptr[i] - sptr[i]) > tol) {
				fprintf(stderr, "FAIL [%s] batch %d query %d: "
					"batched=%.6f per-slot=%.6f tol=%g\n",
					name, b, i, bptr[i], sptr[i], tol);
				tests_failed++;
			}
			tests_run++;
		}
	}

	sam3_arena_free(&ar);
}
```

(Helper `sam3_dot_scorer_alloc_synthetic` fills weights deterministically so the result is nonzero; add it in `model_misc.c` as a static test helper exposed via `model_misc_internal.h` or inline in the test.)

- [ ] **Step 2: Run test — must FAIL (`sam3_dot_scorer_build_batched` not defined)**

```bash
cmake --build build --target test_batched_ops -j 2>&1 | tail -5
```

Expected: link or compile error for `sam3_dot_scorer_build_batched`.

- [ ] **Step 3: Implement `sam3_dot_scorer_build_batched` in `model_misc.c`**

Signature (add to `model_misc.h`):

```c
struct sam3_tensor *sam3_dot_scorer_build_batched(
	struct sam3_dot_scorer *sc,
	struct sam3_graph *g,
	struct sam3_tensor *queries,   /* [B, nq, d] */
	struct sam3_tensor *prompt,    /* [B, seq, d] */
	struct sam3_arena *arena);
```

The existing `sam3_dot_scorer_build` does (see survey): prompt MLP → mean pool with uniform weight `[1, seq]` → prompt_proj → hs_proj → transpose → matmul. For batched:

- Prompt MLP: `gh_linear/gh_relu/gh_linear/gh_layernorm` all thread batch dims; output stays `[B, seq, d]`.
- Mean pool becomes a per-batch operation. Replace the `[1, seq]` weight with a batched matmul `[B, 1, seq] @ [B, seq, d] → [B, 1, d]`. Allocate a `[B, 1, seq]` tensor with all entries `1/seq` (using `gh_alloc_tensor` then `memset` the data pointer's B×1×seq floats).
- `prompt_proj` applied to `[B, 1, d]` → `[B, 1, d]`, then multiply by scale — `gh_linear` + `gh_mul`.
- `hs_proj` applied to `[B, nq, d]` → `[B, nq, d]`.
- Transpose prompt to `[B, d, 1]` via `gh_transpose` swapping last 2 dims.
- `gh_matmul([B, nq, d], [B, d, 1])` → `[B, nq, 1]`.

Full implementation (~80 lines) follows the existing `sam3_dot_scorer_build` structure but with 3D inputs throughout.

- [ ] **Step 4: Run — expect PASS**

```bash
./build/test_batched_ops
```

Expected: `1 tests, 0 failures`.

- [ ] **Step 5: Update Stage 4e call site in `src/model/sam3_image.c` to use batched builder when B>1**

Today (around line 2967), the single-shot call path ends at `sam3_dot_scorer_build(&model->scorer, ...)`. Don't change the behavior yet — instead, wrap with a shim:

```c
struct sam3_tensor *scores_out;
if (queries->n_dims == 3) {
	scores_out = sam3_dot_scorer_build_batched(
		&model->scorer, &g, queries, context_prompt, scratch);
} else {
	scores_out = sam3_dot_scorer_build(
		&model->scorer, &g, queries, context_prompt, scratch);
}
```

This dispatch lets later tasks feed a batched queries tensor; existing 2D flow is unchanged.

- [ ] **Step 6: Run parity harness — must still PASS**

```bash
./build/test_segment_batch_parity
```

Expected: `4 tests, 0 failures`. (Segment flow still feeds 2D queries, so the 2D branch is taken.)

- [ ] **Step 7: Commit**

```bash
git add src/model/model_misc.h src/model/model_misc.c \
        src/model/sam3_image.c tests/test_batched_ops.c
git commit -m "model/scorer: add batched builder, dispatch by queries rank"
```

---

## Phase 2 — Batched mask head

Four sub-stages mirror Stages 4a-4e in `src/model/sam3_image.c:2440-2956`.

### Task 4: Batched prompt cross-attn (Stage 4a)

**Files:** `src/model/segmentation.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Add failing unit test using `run_both_backends`**

Synthesize `x[B=2, n_pixels=16, d=8]` and `text[B=2, seq=3, d=8]`. Call `sam3_seg_head_build_cross_attn_batched`, eval, compare each batch slot with a B=1 call. **Both CPU and Metal** — wrap the case fn with `run_both_backends(cross_attn_case)`. Test fails until the batched builder exists.

- [ ] **Step 2: Implement batched cross-attn**

New builder signature:

```c
struct sam3_tensor *sam3_seg_head_build_cross_attn_batched(
	struct sam3_seg_head *sh, struct sam3_graph *g,
	struct sam3_tensor *x,      /* [B, n_pixels, d] */
	struct sam3_tensor *text,   /* [B, seq, d] */
	struct sam3_arena *arena);
```

The existing `sam3_seg_head_build_cross_attn` does per-head SDPA on 2D slices. For batched, each head's sliced tensor is `[B, n_pixels, head_dim]` (3D). To reuse the existing 4D SDPA path in the Metal backend, reshape each head slice to `[B, 1, n_pixels, head_dim]` before `gh_sdpa`, then reshape back. Concrete:

```c
struct sam3_tensor *hq_4d = gh_reshape(g, arena, hq,
	4, (int[]){B, 1, n_pixels, head_dim});
/* similarly hk_4d, hv_4d */
struct sam3_tensor *ho_4d = gh_sdpa(g, arena, hq_4d, hk_4d, hv_4d, NULL, head_dim);
struct sam3_tensor *ho = gh_reshape(g, arena, ho_4d,
	3, (int[]){B, n_pixels, head_dim});
```

Concat heads along the d dim (existing concat handles a leading B dim since it operates on an arbitrary axis).

- [ ] **Step 3: Run test — expect PASS**

```bash
./build/test_batched_ops
```

- [ ] **Step 4: Wire the call site in `sam3_image.c:2544` behind a rank dispatch (as in Task 3)**

- [ ] **Step 5: Run parity harness — must PASS**

```bash
./build/test_segment_batch_parity
```

- [ ] **Step 6: Commit**

```bash
git commit -am "model/segmentation: batched prompt cross-attn"
```

### Task 5: Batched FPN + instance projection (Stages 4b, 4c)

**Files:** `src/model/segmentation.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)**

Synthesize `enc[B=2, 18, 18, 16]` + feat_2x `[B=2, 36, 36, 16]` + feat_4x `[B=2, 72, 72, 16]`. Call `sam3_seg_head_build_fpn_batched(enc, f2, f4)` → `[B, 72, 72, 16]`. Compare vs B=1 per-slot. Use the same per-backend tolerance (`rtol=1e-4/atol=1e-5` on Metal, `1e-6` on CPU).

- [ ] **Step 2: Implement**

The existing FPN (`segmentation.c:251-337`) uses `gh_upsample`, `gh_add`, `gh_conv2d`, `gh_groupnorm`, `gh_relu` — all already N-aware per the survey. So the batched FPN is mostly about giving the caller a way to produce `[B, H, W, C]` input. The feat_2x / feat_4x come from the image encoder which is shared across B — the caller must `gh_tile` or manually broadcast-copy them along B (below).

Since image features are shared across all sets, add a helper `gh_broadcast_batch(tensor, B)` in `graph_helpers.c`: takes `[H, W, C]` or `[1, H, W, C]`, returns `[B, H, W, C]` by tiling in persist arena (memcpy loop).

Instance projection (line 2704+): `gh_conv2d` with 1x1 kernel, OHWI weight. Already N-aware; no change beyond feeding `[B, 288, 288, 256]`.

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: Wire call site by rank dispatch**

- [ ] **Step 5: Parity harness — PASS**

- [ ] **Step 6: Commit**

```bash
git commit -am "model/segmentation: batched FPN + instance projection"
```

### Task 6: Batched mask-embedder MLP (Stage 4d)

**Files:** `src/model/segmentation.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)** — Synthesize `queries[B=2, 8, 16]`. Run `sam3_seg_head_build_mask_mlp_batched`. Compare vs B=1 per-slot on each backend.

- [ ] **Step 2: Implement** — Existing 3-layer MLP uses `gh_linear + gh_relu + gh_linear + gh_relu + gh_linear`. All are batch-transparent. The "batched" version is trivially the same code with the rank assertion relaxed; expose it as a new entry point or update the existing one. Prefer the former to keep diff minimal.

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: Wire call site**

- [ ] **Step 5: Parity harness — PASS**

- [ ] **Step 6: Commit** — `git commit -am "model/segmentation: batched mask MLP (trivial rank relax)"`

### Task 7: Batched dot-product mask logits (Stage 4e)

**Files:** `src/model/sam3_image.c` Stage 4e area, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)**

Synthesize `mask_embed[B=2, 8, 16]` and `inst[B=2, 18, 18, 16]`. Compute batched dot product → `[B, 8, 18, 18]`. Compare vs B=1 on each backend.

- [ ] **Step 2: Implement**

Current path (around `sam3_image.c:2809` per survey):

```
inst[1, H, W, d] → reshape [H*W, d] → transpose [d, H*W] → matmul(mask_embed[nq, d]) → [nq, H*W] → reshape [nq, H, W]
```

Batched:

```
inst[B, H, W, d] → reshape [B, H*W, d] → transpose-last-two [B, d, H*W] → matmul(mask_embed[B, nq, d]) → [B, nq, H*W] → reshape [B, nq, H, W]
```

All ops are batch-transparent; the only change is the reshape sizes.

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: Wire call site (rank dispatch)**

- [ ] **Step 5: Parity harness — PASS**

- [ ] **Step 6: Commit** — `git commit -am "model/seg: batched dot-product mask logits"`

---

## Phase 3 — Batched decoder

### Task 8: Batched CPU box refinement helper

**Files:** `src/model/sam3_image.c` (around line 106 `cpu_box_refine`), `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)**

Synthesize `queries[B=2, 8, 16]` and `ref_boxes[B=2, 8, 4]`. Call `cpu_box_refine_batched` (new). Compare vs calling `cpu_box_refine` per slot on each backend. Note: `cpu_box_refine` uses the backend for the box_head MLP (not purely CPU despite its name), so backend coverage matters here.

- [ ] **Step 2: Implement**

Existing `cpu_box_refine` (in `sam3_image.c:122`) takes `queries[nq, d]` and `ref_boxes[nq, 4]`, runs output_ln, box_head 3-layer MLP (all via graph eval), then applies `sigmoid(inverse_sigmoid(old) + delta)` CPU-side. Batched version:

```c
static enum sam3_error cpu_box_refine_batched(
	const struct sam3_decoder *dec,
	struct sam3_backend *be,
	struct sam3_tensor *queries,    /* [B, nq, d] */
	float *ref_boxes,               /* [B * nq * 4] */
	struct sam3_arena *scratch);
```

The box_head MLP is per-query — run it on the batched tensor (batch-transparent). The CPU sigmoid loop iterates `B * nq * 4` indices instead of `nq * 4`.

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: (No call-site wiring yet — batched decoder loop in Task 13 will invoke this.)**

- [ ] **Step 5: Commit** — `git commit -am "model/decoder: batched cpu_box_refine helper"`

### Task 9: Batched `sam3_decoder_compute_rpb`

**Files:** `src/model/decoder.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)** — Given `ref_boxes[B=2, 8, 4]`, `H=W=6`, assert output `[B, n_heads, 8, 36]` matches 2 calls of the current 2D API on each backend. RPB itself is a CPU computation but the ref_point_head MLP calls inside go through the backend, so run both.

- [ ] **Step 2: Add a batched overload**

```c
void sam3_decoder_compute_rpb_batched(
	const struct sam3_decoder *dec,
	const float *ref_boxes,  /* [B * nq * 4] */
	int B, int H, int W,
	float *out);             /* [B * n_heads * nq * H * W] */
```

Implementation: wrap the existing CPU code in a `for (int b = 0; b < B; b++)` and advance pointers by `nq*4` and `n_heads*nq*H*W` per iteration. The existing MLP calls use malloc temporaries sized for `nq=200`; make the temporaries arena-based so they're reused across batch iterations (or hoist the malloc outside the loop).

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: Commit** — `git commit -am "model/decoder: batched RPB computation"`

### Task 10: Batched `sam3_decoder_compute_query_pos`

**Files:** `src/model/decoder.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)** — Given `ref_boxes[B=2, 8, 4]`, assert batched qpos equals 2× per-slot call on each backend.

- [ ] **Step 2: Add batched overload**

```c
struct sam3_tensor *sam3_decoder_compute_query_pos_batched(
	struct sam3_decoder *dec,
	struct sam3_graph *g,
	struct sam3_arena *arena,
	const float *ref_boxes,  /* [B * nq * 4] */
	int B);
```

Sine position embed over `B * nq * 4` iterations on CPU → packs into a `[B, nq, d]` tensor. Then `ref_point_head` MLP runs on `[B, nq, d]` via `gh_linear` (batch-transparent). Output `[B, nq, d]`.

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: Commit** — `git commit -am "model/decoder: batched query_pos"`

### Task 11: Batched decoder self-attention

**Files:** `src/model/decoder.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test (CPU + Metal via `run_both_backends`)** — `q[B=2, 8, 16]`, `qpos[B=2, 8, 16]`. Run `sam3_decoder_build_sa_batched`. Compare vs B=1 per-slot on each backend. This is the first test that exercises the 4D SDPA path (`[B, 1, nq, hd]`) on Metal; if Metal SDPA rejects `B>1`, this task surfaces it.

- [ ] **Step 2: Implement**

Current `sam3_decoder_build_sa` (in `decoder.c`) wraps `decoder_self_attention_with_pos` which does per-head SDPA on 2D slices. For batched, introduce `decoder_self_attention_with_pos_batched` that:

- Accepts `q[B, nq, d]` and `qpos[B, nq, d]`.
- Builds `q_with_pos = q + qpos` via `gh_add` (batch-transparent).
- Linear projections `sq = gh_linear(q_with_pos, sa_qkv_w, sa_qkv_b)` → `[B, nq, 3*d]`.
- Slice into Q/K/V: `gh_slice` on the last dim (keeps batch dim intact).
- Per-head loop: slice each head on dim `-1`, reshape to `[B, 1, nq, head_dim]` for SDPA, reshape back.
- Concat heads along last dim.
- Output projection.

- [ ] **Step 3: Run test — PASS**

- [ ] **Step 4: Commit** — `git commit -am "model/decoder: batched self-attention"`

### Task 12: Batched decoder text + vision cross-attention

**Files:** `src/model/decoder.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit tests — one for TCA (text), one for CA (vision, with RPB). Both use `run_both_backends`.**

- [ ] **Step 2: Implement**

- `sam3_decoder_build_tca_batched(q[B, nq, d], qpos[B, nq, d], text[B, seq, d])` — same per-head pattern as Task 4, reshape to 4D around `gh_sdpa`.
- `sam3_decoder_build_ca_batched(q[B, nq, d], qpos[B, nq, d], enc_features[B, n_pixels, d], enc_pos[n_pixels, d] broadcast, rpb_mask[B, n_heads, nq, n_pixels])` — same pattern, plus per-head rpb slice retains `[B, nq, n_pixels]` shape (3D, no reshape needed for the mask).

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit** — `git commit -am "model/decoder: batched cross-attention (text + vision)"`

### Task 13: Batched decoder FFN and `build_layer`

**Files:** `src/model/decoder.h/.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Failing unit test for `sam3_decoder_build_ffn_batched` (CPU + Metal via `run_both_backends`)**

- [ ] **Step 2: Implement FFN batched (trivial — `gh_mlp` + `gh_add` + `gh_layernorm` are batch-transparent).**

- [ ] **Step 3: Compose `sam3_decoder_build_layer_batched` from the batched substep builders. Pass through `sam3_decoder_build_final_batched` (just a `gh_layernorm` on `[B, nq, d]`).**

- [ ] **Step 4: Run test — PASS**

- [ ] **Step 5: Commit** — `git commit -am "model/decoder: batched FFN + build_layer wrapper"`

---

## Phase 4 — Integration

### Task 14: Batched image-model pipeline entry point

**Files:** `src/model/sam3_image.h`, `src/model/sam3_image.c`, `tests/test_batched_ops.c`

- [ ] **Step 1: Add declaration**

```c
enum sam3_error sam3_image_model_segment_batched(
	struct sam3_image_model *model,
	struct sam3_backend *be,
	struct sam3_backend *cpu_be,
	struct sam3_tensor *stacked_prompt_tokens,  /* [B, N_prompt+1, d] or NULL */
	struct sam3_tensor *stacked_text_features,  /* [B, seq, d] */
	int B,
	struct sam3_arena *scratch,
	struct sam3_arena *persist,
	struct sam3_tensor **out_masks,   /* [B, 200, H, W] */
	struct sam3_tensor **out_scores,  /* [B, 200, 1] */
	struct sam3_profiler *profiler);
```

- [ ] **Step 2: Implement as a copy of `sam3_image_model_segment` with every stage swapped for its batched version**

Per-stage edits:
- Stage 1 (geometry encoder): keep as CPU, but feed `prompt_tokens[B, N+1, d]` — since the CPU geometry encoder loop is small, wrap the whole thing in a `for (b = 0; b < B; b++)` loop over per-slot tensors. Output stacked to `[B, N+1, d]`.
- Stage 2 (encoder fusion): each fused output `[n_pixels, d]` already exists per-set; the batched entry point accepts a stacked `[B, n_pixels, d]` from the caller. No in-stage change.
- Stage 3 (DETR decoder): replace every `sam3_decoder_build_*` with `_batched`. Queries / ref_boxes / qpos buffers sized `[B * nq * *]`. Per-layer `cpu_box_refine_batched`. Per-layer `sam3_decoder_compute_rpb_batched` into `rpb_buf[B, n_heads, nq, n_pixels]`.
- Stage 4a-e (seg head + scorer): replace with `_batched` entry points from Phase 2.

- [ ] **Step 3: Make single-shot `sam3_image_model_segment` a thin wrapper that unsqueezes inputs and calls the batched version with B=1**

```c
enum sam3_error sam3_image_model_segment(...)
{
	/* unsqueeze prompt_tokens and text_features to [1, ...] via
	 * gh_reshape, call sam3_image_model_segment_batched with B=1,
	 * then squeeze out_masks / out_scores back to [nq, H, W] /
	 * [nq, 1]. */
}
```

- [ ] **Step 4: Run parity harness — MUST PASS**

```bash
./build/test_segment_batch_parity
```

Expected: identical output (bytes) to pre-refactor. This is the critical step; if it fails, bisect per-stage.

- [ ] **Step 5: Commit** — `git commit -am "model/image: add batched pipeline entry point; single-shot delegates"`

### Task 15: Wire `sam3_processor_segment_batch` through the batched pipeline

**Files:** `src/model/sam3_processor.c`

- [ ] **Step 1: Refactor `sam3_processor_segment_batch` to use the batched image-model entry point**

Shape:
1. Drop any pending async text (unchanged).
2. For each set, run fusion+geometry+text serially; persist each set's `[n_pixels, d]` fused output and `[seq, d]` text feature on `proc->model_arena`.
3. Stack fused outputs into `[B, n_pixels, d]` and text features into `[B, seq_max, d]` padded to max seq — OR require all sets to share text length and skip padding in phase 1 (acceptable simplification; document it).
4. Call `sam3_image_model_segment_batched(...)` once.
5. Unpack `[B, 200, H, W]` masks and `[B, 200, 1]` scores into per-set `sam3_result`.
6. On any error, free already-populated results and return (existing pattern).

Important: retain the existing per-set loop as a fallback (`fallback_sequential = 1` env var or build-flag) for debugging. This is a debug knob only, not a user-visible API addition.

- [ ] **Step 2: Run the existing batch convenience test**

```bash
./build/test_segment_batch
```

Expected: `37 tests, 0 failures`.

- [ ] **Step 3: Run parity harness**

```bash
./build/test_segment_batch_parity
```

Expected: `4 tests, 0 failures`.

- [ ] **Step 4: Commit** — `git commit -am "processor/segment_batch: route through batched image-model pipeline"`

---

## Phase 5 — Validation

### Task 16: Performance micro-benchmark

**Files:** `tests/bench_segment_batch.c` (new)

- [ ] **Step 1: Write a timing harness that runs batch sizes 1, 2, 4, 8 and prints latency per set**

```c
/* tests/bench_segment_batch.c — prints wall-clock for batch sizes 1-8 */
/* Uses clock_gettime(CLOCK_MONOTONIC). Compares per-set latency of
 * batched vs serial for the same prompts. */
```

- [ ] **Step 2: Register behind `SAM3_BENCH=ON`** (already gated by the existing `if(NOT SAM3_BENCH)` block in `CMakeLists.txt:236`)

- [ ] **Step 3: Run and capture numbers**

```bash
cmake -B build -DSAM3_BENCH=ON
cmake --build build --target bench_segment_batch -j
./build/bench_segment_batch
```

Expected: batched latency per set ≤ serial latency per set for B≥2. If it's worse, investigate cache thrashing or a regression in a batched op.

- [ ] **Step 4: Commit** — `git commit -am "bench: per-batch-size segment latency"`

### Task 17: Finish the branch

**Files:** none

- [ ] **Step 1: Run the parity and unit tests one more time**

```bash
./build/test_segment_batch
./build/test_segment_batch_parity
./build/test_batched_ops
```

All three must pass.

- [ ] **Step 2: Rebase on main, open PR**

```bash
git fetch origin
git rebase origin/main
gh pr create --title "Graph-level batched decoder / mask head / scorer" --body "<see plan>"
```

---

## Parity invariants to preserve (checked every phase)

At the end of each Task in Phases 1-3, run:

```bash
./build/test_segment_batch_parity
./build/test_segment_batch
```

Both must pass. If either regresses, revert the last task's commit and bisect within it (typical causes: missed rank dispatch at a call site, wrong reshape, mask broadcast).

## Risks and open questions

1. **`gh_sdpa` with 4D `[B, 1, nq, hd]` on Metal** — survey says the Metal backend supports 2D and 4D. The batched decoder reshapes each head slice to `[B, 1, nq, hd]` before `gh_sdpa`. If the MLX/Metal 4D path assumes `H > 1` or a different axis order, Task 11's Metal assertion will fail. Mitigation: Task 11's test uses `run_both_backends` — the CPU path typically passes first (more permissive shape handling); a Metal-only failure localizes the problem to the backend. Fix options: add a leading-1 squeeze-expand in `gh_sdpa`, or wrap heads differently (e.g., pack `B` into the head dim as `[1, B, nq, hd]`).
2. **Variable text sequence length across sets** — if set A has text with 3 tokens and set B has 6 tokens, stacking forces padding. Phase 1 punts: require all sets in a batch to share text length (pad up to max, mask out). This is a soft constraint we document in the docstring; relaxing it is a Phase 2 improvement.
3. **Arena size growth** — `rpb_buf` grows from `n_heads * nq * n_pixels` to `B * n_heads * nq * n_pixels`. For nq=200, n_heads=8, n_pixels=5184, B=8, that's ~265 MiB of persist arena. Check `proc->model_arena` capacity before running (likely need to raise default).
4. **Metal tensor cache invalidation** — if the batched pipeline leaves `[B, ...]` tensors in the model arena past `persist_save` rollback, the backend's tensor-cache could return stale handles. The existing `cache_invalidate` call at `sam3_processor.c:1494` handles this; verify it's still invoked after Task 15.

## Self-review checklist (run once after writing, don't re-run)

- ✅ Spec coverage: every stage of the segment pipeline (decoder, seg head, scorer) has tasks; fusion/geometry/text deliberately out of scope and flagged.
- ✅ No "TBD" / "implement later" / "similar to Task N" without repeating code.
- ✅ Type consistency: `B`, `nq`, `d`, `n_heads`, `n_pixels`, `H`, `W`, `seq` used consistently; batched function names always end `_batched`.
- ✅ Each task has test → run-fail → implement → run-pass → commit.
- ✅ File paths are absolute-within-repo, line numbers reference the current state on `main`.
