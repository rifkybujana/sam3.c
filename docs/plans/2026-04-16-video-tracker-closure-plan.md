# Video Tracker Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Wire the scaffolded SAM3 video tracker into a working end-to-end
pipeline: `add_points` / `add_box` / `propagate` produce real masks; the
`sam3_cli` binary exposes a `track` subcommand; numerical tests
validate against Python reference fixtures; the feature is reachable
from Python and documented.

**Architecture:** Eager backbone feature caching at `video_start`. Prompt
storage in the session struct. Memory bank populated on every frame
(conditioning on prompted frames, non-conditioning otherwise).
Object-pointer extraction via a new optional out-param on the mask
decoder. New `tools/cli_track.c` subcommand. Integration test exercises
an in-process synthetic clip.

**Tech Stack:** C11, CMake, CTest, Python CFFI (existing), pytest.

**Design reference:** `docs/plans/2026-04-16-video-tracker-closure-design.md`

---

## Phase A — Core inference wiring

### Task 1: Extend mask_decoder with out_obj_token parameter

**Files:**
- Modify: `src/model/mask_decoder.h` — signature change
- Modify: `src/model/mask_decoder.c` — emit object-score token
- Test: `tests/test_mask_decoder.c` — add test for token shape

**Step 1: Write the failing test**

Add to `tests/test_mask_decoder.c`:

```c
static void test_mask_decoder_emits_obj_token(void)
{
	/* Same setup as existing test_mask_decoder_build_shapes, but
	 * pass a non-NULL out_obj_token and verify it has shape
	 * [1, d_model]. The token stack is 2-D in this codebase. */
	struct sam3_tensor *obj_token = NULL;
	int err = sam3_mask_decoder_build(&md, &g, feats, H, W,
	                                   NULL, s0, s1, &arena,
	                                   &masks, &iou, &obj_token);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT(obj_token != NULL);
	ASSERT_EQ(obj_token->n_dims, 2);
	ASSERT_EQ(obj_token->dims[0], 1);
	ASSERT_EQ(obj_token->dims[1], md.d_model);
}
```

**Step 2: Run test to verify it fails**

```
cd build && cmake --build . --target test_mask_decoder && \
	ctest -R test_mask_decoder -V
```
Expected: FAIL — compile error (signature mismatch) OR `obj_token` NULL.

**Step 3: Change header signature**

`src/model/mask_decoder.h`:
```c
int sam3_mask_decoder_build(struct sam3_graph *g,
                            struct sam3_arena *arena,
                            const struct sam3_mask_decoder *md,
                            struct sam3_tensor *features,
                            struct sam3_tensor *sparse_embeds,
                            struct sam3_tensor *dense_embeds,
                            struct sam3_tensor *pos_enc,
                            struct sam3_tensor **out_masks,
                            struct sam3_tensor **out_iou,
                            struct sam3_tensor **out_obj_token);
```

Document the new param: "Optional. If non-NULL, receives the
pre-projection object-score token of shape [1, d_model]. Pass NULL to
skip." (Note: this codebase keeps the transformer token stack 2-D as
`[n_tokens, d_model]`, so the slice is 2-D, not 3-D. Downstream Tasks
4 and 7 must index `obj_token->dims[1]` for the feature dim, not
`dims[2]`.)

**Step 4: Implement in mask_decoder.c**

Locate the transformer block that produces output tokens. The first
token slot is typically the IoU token; the object-score token is at a
fixed index (match upstream Python — inspect
`reference/sam3/...mask_decoder.py` to confirm the slot; do not guess).

After the transformer runs, if `out_obj_token` is non-NULL, slice the
corresponding token into a new tensor via
`gh_slice(g, arena, tokens, axis=1, start=obj_idx, end=obj_idx+1)` and
assign to `*out_obj_token`.

**Step 5: Update existing callers to pass NULL**

`grep -rn "sam3_mask_decoder_build" src/ tests/` — update every call
site to add a trailing `NULL` argument. Expected sites: `sam3_image.c`,
`tracker.c`, `test_mask_decoder.c` (existing tests).

**Step 6: Run full mask_decoder test suite**

```
ctest -R test_mask_decoder -V
```
Expected: all existing tests PASS, new `test_mask_decoder_emits_obj_token`
PASS.

**Step 7: Commit**

```
git add src/model/mask_decoder.h src/model/mask_decoder.c \
        src/model/sam3_image.c src/model/tracker.c \
        tests/test_mask_decoder.c
git commit -m "model/mask_decoder: expose optional object-score token output"
```

---

### Task 2: Add prompt storage types and constants

**Files:**
- Modify: `include/sam3/sam3_types.h` — add `SAM3_MAX_POINTS_PER_OBJ`
- Modify: `src/model/video_session.h` — add prompt struct + bitmap field
- Modify: `src/model/video_session.c` — allocate + free new fields
- Test: `tests/test_video_session.c` — prompt append + clear

**Step 1: Write the failing test**

Add to `tests/test_video_session.c`:

```c
static void test_session_prompts_append_and_clear(void)
{
	struct sam3_video_session *s = /* minimal init */;
	struct sam3_video_prompt p = {
		.frame_idx = 3,
		.obj_internal_idx = 0,
		.kind = SAM3_PROMPT_POINTS,
		.points = { .n = 1, .xys = {504.f, 504.f},
		            .labels = {1} }
	};
	ASSERT_EQ(sam3_session_add_prompt(s, &p), SAM3_OK);
	ASSERT_EQ(s->n_prompts, 1);
	ASSERT_TRUE(sam3_session_is_prompted(s, 3));
	ASSERT_FALSE(sam3_session_is_prompted(s, 0));

	sam3_session_clear_prompts(s);
	ASSERT_EQ(s->n_prompts, 0);
	ASSERT_FALSE(sam3_session_is_prompted(s, 3));
}
```

**Step 2: Run test to verify it fails**

```
ctest -R test_video_session -V
```
Expected: FAIL (compile error).

**Step 3: Add constant**

`include/sam3/sam3_types.h`:
```c
#define SAM3_MAX_POINTS_PER_OBJ  16
```

**Step 4: Add prompt struct + session fields**

`src/model/video_session.h`:
```c
enum sam3_prompt_kind {
	SAM3_PROMPT_POINTS = 0,
	SAM3_PROMPT_BOX    = 1,
};

struct sam3_video_prompt {
	int frame_idx;
	int obj_internal_idx;
	enum sam3_prompt_kind kind;
	union {
		struct {
			int   n;
			float xys[SAM3_MAX_POINTS_PER_OBJ * 2];
			int   labels[SAM3_MAX_POINTS_PER_OBJ];
		} points;
		struct sam3_box box;
	} data;
};

/* In struct sam3_video_session: */
struct sam3_video_prompt *prompts;   /* arena-owned */
int                       n_prompts;
int                       cap_prompts;
uint8_t                  *prompted_frames;  /* arena-owned, 1 byte/frame */

/* Helpers */
int  sam3_session_add_prompt(struct sam3_video_session *s,
                             const struct sam3_video_prompt *p);
void sam3_session_clear_prompts(struct sam3_video_session *s);
int  sam3_session_is_prompted(const struct sam3_video_session *s,
                              int frame_idx);
```

**Step 5: Implement helpers in video_session.c**

Straightforward: linear append; `memset(prompted_frames, 0, n_frames)`
on clear; indexed read on is_prompted.

**Step 6: Allocate in sam3_video_start**

`src/model/sam3_video.c` — after `cached_features` allocation, allocate:
- `prompted_frames = sam3_arena_alloc(&session->persist, nf)`
- `prompts = sam3_arena_alloc(&session->persist, SAM3_MAX_OBJECTS * nf * sizeof(*prompts))`
- `cap_prompts = SAM3_MAX_OBJECTS * nf`

Zero both via `memset`.

**Step 7: Run test**

```
ctest -R test_video_session -V
```
Expected: PASS.

**Step 8: Commit**

```
git add include/sam3/sam3_types.h src/model/video_session.h \
        src/model/video_session.c src/model/sam3_video.c \
        tests/test_video_session.c
git commit -m "video/session: add prompt storage and prompted-frame bitmap"
```

---

### Task 3: Eager backbone feature caching in video_start

**Files:**
- Modify: `src/model/sam3_video.c` — encode all frames at start
- Modify: `src/model/video_session.h` — document cache contract
- Test: `tests/test_video_api.c` — add cached_features-populated test

**Step 1: Write the failing test**

Add to `tests/test_video_api.c`:

```c
static void test_video_start_populates_cached_features(void)
{
	/* Use a zero-weight model (shape-only) — full weight load is
	 * not needed to verify caching runs. Use a 2-frame synthetic
	 * directory. */
	struct sam3_ctx *ctx = /* minimal */;
	struct sam3_video_session *s = NULL;
	int err = sam3_video_start(ctx, "tests/data/video2", &s);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_NOT_NULL(s->cached_features);
	for (int i = 0; i < sam3_video_frame_count(s); i++) {
		ASSERT_NOT_NULL(s->cached_features[i]);
	}
	sam3_video_end(s);
}
```

Create `tests/data/video2/frame_000.png` and `frame_001.png`
(small — 32×32 solid colors suffice). Check them into git.

**Step 2: Run test to verify it fails**

Expected: FAIL — `cached_features[i]` is NULL.

**Step 3: Implement eager caching**

In `sam3_video_start`, after the frame-loading block and after
`cached_features` is allocated:

```c
for (int i = 0; i < nf; i++) {
	struct sam3_graph *g = sam3_graph_create(&session->scratch);
	if (!g) { err = SAM3_ENOMEM; goto cleanup; }

	struct sam3_tensor *frame = session->frames.frames[i];
	struct sam3_tensor *features = NULL;
	err = sam3_image_encoder_build(g, &session->scratch,
	                                &ctx->img_enc, frame, &features);
	if (err) goto cleanup;

	err = sam3_graph_eval(ctx, g);
	if (err) goto cleanup;

	/* Copy features into persist arena — scratch will be reset */
	session->cached_features[i] =
		sam3_tensor_clone_persist(&session->persist, features);
	if (!session->cached_features[i]) {
		err = SAM3_ENOMEM;
		goto cleanup;
	}

	sam3_arena_reset(&session->scratch);
	sam3_log_debug("video_start: cached features for frame %d/%d",
	               i + 1, nf);
}
```

Add `sam3_tensor_clone_persist` if not present (`src/core/tensor.c`
— copies tensor data from one arena to another, preserves shape/dtype).

**Step 4: Run test**

```
ctest -R test_video_start -V
```
Expected: PASS.

**Step 5: Commit**

```
git add src/model/sam3_video.c src/model/video_session.h \
        src/core/tensor.h src/core/tensor.c \
        tests/test_video_api.c tests/data/video2/
git commit -m "video/session: eagerly cache backbone features for all frames"
```

---

### Task 4: Implement add_points with full pipeline

**Files:**
- Modify: `src/model/sam3_video.c` — `sam3_video_add_points` body
- Modify: `src/model/tracker.h` — expose helper for prompt→mask flow
- Modify: `src/model/tracker.c` — implement helper
- Test: `tests/test_video_api.c` — add_points returns non-empty mask

**Step 1: Write the failing test**

```c
static void test_video_add_points_produces_nonempty_mask(void)
{
	struct sam3_ctx *ctx = /* with real weights if available */;
	struct sam3_video_session *s = NULL;
	sam3_video_start(ctx, "tests/data/video2", &s);

	struct sam3_point pts[1] = { {16.f, 16.f, 1} };
	struct sam3_result result = {0};
	int err = sam3_video_add_points(s, /*frame=*/0, /*obj=*/0,
	                                 pts, 1, &result);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_GT(result.n_masks, 0);
	ASSERT_NOT_NULL(result.masks);
	/* at least one mask pixel should be non-zero */
	int nonzero = 0;
	for (int i = 0; i < result.masks[0].h * result.masks[0].w; i++)
		if (result.masks[0].data[i] > 0.5f) nonzero++;
	ASSERT_GT(nonzero, 0);

	sam3_video_end(s);
}
```

**Step 2: Add tracker helper**

`src/model/tracker.h`:
```c
/*
 * sam3_tracker_segment_prompted — run prompt encoder + mask decoder +
 * memory encoder on a cached frame with user prompts. Adds the
 * resulting memory entry to the bank with is_cond=true.
 *
 * @features: backbone features for the frame (cached)
 * @points / n_points / box: prompt (one or the other, not both)
 * @out_mask: best mask [H, W] (arena-allocated)
 * Returns SAM3_OK or error code.
 */
int sam3_tracker_segment_prompted(struct sam3_ctx *ctx,
                                   struct sam3_tracker *trk,
                                   struct sam3_arena *arena,
                                   struct sam3_tensor *features,
                                   int frame_idx,
                                   const struct sam3_point *points,
                                   int n_points,
                                   const struct sam3_box *box,
                                   struct sam3_tensor **out_mask);
```

**Step 3: Implement the helper**

Sketch (in `src/model/tracker.c`):

```c
int sam3_tracker_segment_prompted(...)
{
	struct sam3_graph *g = sam3_graph_create(arena);
	if (!g) return SAM3_ENOMEM;

	/* Prompt encoder */
	struct sam3_tensor *sparse = NULL, *dense = NULL;
	int err = sam3_prompt_encoder_build(g, arena, &trk->prompt_enc,
	                                    points, n_points, box,
	                                    &sparse, &dense);
	if (err) return err;

	/* Mask decoder with obj_token */
	struct sam3_tensor *masks = NULL, *iou = NULL, *obj_tok = NULL;
	err = sam3_mask_decoder_build(g, arena, &trk->mask_decoder,
	                               features, sparse, dense,
	                               trk->prompt_enc.pos_enc,
	                               &masks, &iou, &obj_tok);
	if (err) return err;

	err = sam3_graph_eval(ctx, g);
	if (err) return err;

	/* Select best-IoU mask */
	int best = argmax(iou);
	struct sam3_tensor *best_mask = slice(masks, best);

	/* Project obj_token through obj_ptr_proj */
	struct sam3_tensor *obj_ptr = linear(obj_tok,
	                                      trk->obj_ptr_proj_w,
	                                      trk->obj_ptr_proj_b);

	/* Memory encoder */
	struct sam3_tensor *mem_feats = NULL;
	err = sam3_memory_encoder_build(g2, arena, &trk->mem_encoder,
	                                 features, best_mask, &mem_feats);
	/* ... eval ... */

	/* Add to bank */
	sam3_memory_bank_add(&trk->mem_bank, frame_idx,
	                     mem_feats, obj_ptr, /*is_cond=*/1,
	                     iou->data[best]);

	*out_mask = best_mask;
	return SAM3_OK;
}
```

**Step 4: Wire into sam3_video_add_points**

Replace the stub body in `src/model/sam3_video.c`:

```c
sam3_error sam3_video_add_points(struct sam3_video_session *s,
                                  int frame_idx, int obj_id,
                                  const struct sam3_point *points,
                                  int n_points,
                                  struct sam3_result *result)
{
	/* existing validation ... */

	int obj_internal = sam3_session_get_or_add_obj(s, obj_id);
	if (obj_internal < 0) return SAM3_EINVAL;

	struct sam3_video_prompt p = {
		.frame_idx = frame_idx,
		.obj_internal_idx = obj_internal,
		.kind = SAM3_PROMPT_POINTS,
	};
	p.data.points.n = n_points;
	for (int i = 0; i < n_points; i++) {
		p.data.points.xys[2 * i]     = points[i].x;
		p.data.points.xys[2 * i + 1] = points[i].y;
		p.data.points.labels[i]      = points[i].label;
	}
	int err = sam3_session_add_prompt(s, &p);
	if (err) return err;

	struct sam3_tensor *mask = NULL;
	err = sam3_tracker_segment_prompted(s->ctx, &s->tracker,
	                                     &s->scratch,
	                                     s->cached_features[frame_idx],
	                                     frame_idx, points, n_points,
	                                     NULL, &mask);
	if (err) return err;

	/* Copy mask into result */
	err = sam3_result_fill_from_tensor(result, mask);
	sam3_arena_reset(&s->scratch);
	return err;
}
```

Add `sam3_result_fill_from_tensor` helper (or reuse an existing one
from `sam3_image.c` — check first; DRY).

**Step 5: Run test**

Expected: PASS. If weights aren't available in CI, gate the test with
a model-file-exists check (do NOT silently skip — use the existing
`tests/data/*` pattern or generate a tiny synthetic model).

**Step 6: Commit**

```
git add src/model/sam3_video.c src/model/tracker.h \
        src/model/tracker.c tests/test_video_api.c
git commit -m "video/api: implement add_points with full prompt pipeline"
```

---

### Task 5: Implement add_box

**Files:**
- Modify: `src/model/sam3_video.c` — `sam3_video_add_box` body
- Test: `tests/test_video_api.c` — add_box non-empty mask

Mirror Task 4 but pass `box` instead of points to
`sam3_tracker_segment_prompted`. Most of the code is shared.

**Step 1-5:** Same TDD structure.

**Commit:**
```
git commit -m "video/api: implement add_box with full prompt pipeline"
```

---

### Task 6: Implement full memory collection in tracker_track_frame

**Files:**
- Modify: `src/model/tracker.c` — replace the TODO/no-memory fallback

**Step 1: Write the failing test**

Add to `tests/test_tracker.c`:

```c
static void test_tracker_track_frame_uses_populated_bank(void)
{
	struct sam3_tracker trk;
	sam3_tracker_init(&trk, NULL);
	/* manually add a fake memory entry */
	struct sam3_tensor *fake_spatial = /* [5184, 64] zeros */;
	struct sam3_tensor *fake_obj = /* [1, 256] zeros */;
	sam3_memory_bank_add(&trk.mem_bank, /*frame=*/0, fake_spatial,
	                     fake_obj, /*is_cond=*/1, /*score=*/1.f);

	/* Run track_frame on frame 1 — should take the populated-bank
	 * branch, not the no-memory fallback */
	/* Use a debug counter or log probe to verify the branch taken */
	struct sam3_tensor *masks = NULL, *iou = NULL;
	int err = sam3_tracker_track_frame(ctx, &trk, arena, g,
	                                    fake_backbone, /*frame=*/1,
	                                    /*is_cond=*/0, &masks, &iou,
	                                    NULL, NULL);
	ASSERT_EQ(err, SAM3_OK);
	ASSERT_NOT_NULL(masks);
	/* sanity: masks shape */
	ASSERT_EQ(masks->ne[0], 4);   /* 4 mask candidates */
}
```

**Step 2: Replace the TODO block**

In `src/model/tracker.c` around line 260, replace the
"fall back to no-memory path" with real memory collection:

```c
int total = sam3_memory_bank_total(&trk->mem_bank);
if (total == 0) {
	memory   = /* existing no-mem-embed branch */;
	mem_pos  = /* existing no-mem-pos-enc branch */;
} else {
	/* Collect spatial features from bank:
	 *   spatial_list: stack of [5184, 64] tensors
	 *   concatenated to [total * 5184, 64]
	 */
	struct sam3_tensor *spatial_cat = gh_concat_mem(
		g, arena, &trk->mem_bank, total);

	/* Apply temporal position encoding:
	 *   for i in 0..total: spatial_cat[i*5184:(i+1)*5184] +=
	 *       maskmem_tpos_enc[relative_frame_distance(i)]
	 */
	struct sam3_tensor *mem_pos_spatial = gh_tpos_enc_mem(
		g, arena, &trk->mem_bank, trk->maskmem_tpos_enc,
		current_frame_idx, total);

	/* Collect obj_pointers — concatenate [n_obj * total, 256] */
	struct sam3_tensor *obj_ptrs_cat = gh_concat_obj_ptrs(
		g, arena, &trk->mem_bank, trk->max_obj_ptrs);

	/* Memory attention needs:
	 *   memory: [total_spatial + n_obj_ptrs, 256]
	 *   mem_pos: [total_spatial + n_obj_ptrs, 256]  (obj_ptrs use zeros)
	 */
	memory  = gh_concat_rows(g, arena, spatial_cat, obj_ptrs_cat);
	mem_pos = gh_concat_rows(g, arena, mem_pos_spatial,
	                         gh_zeros_like(g, arena, obj_ptrs_cat));
}
```

Implement the helpers `gh_concat_mem`, `gh_tpos_enc_mem`,
`gh_concat_obj_ptrs`, `gh_concat_rows` in `src/model/graph_helpers.{h,c}`
if not already present.

**Step 3: Run test**

Expected: PASS.

**Step 4: Commit**

```
git add src/model/tracker.c src/model/graph_helpers.h \
        src/model/graph_helpers.c tests/test_tracker.c
git commit -m "video/tracker: collect spatial + obj_ptr memory for memory attention"
```

---

### Task 7: Wire obj_ptr extraction in tracker_track_frame

**Files:**
- Modify: `src/model/tracker.c` — replace `out_obj_ptr = NULL` stub

**Step 1: Extend existing test**

Modify `test_tracker_track_frame_uses_populated_bank` to request
`out_obj_ptr` and assert non-NULL + correct shape.

**Step 2: Implement**

In `sam3_tracker_track_frame`, change the mask decoder call to pass
`&obj_tok`, and after eval:

```c
if (out_obj_ptr) {
	struct sam3_tensor *obj_ptr = gh_linear(g, arena, obj_tok,
	                                         trk->obj_ptr_proj_w,
	                                         trk->obj_ptr_proj_b);
	*out_obj_ptr = obj_ptr;
}
if (out_score) {
	/* Best IoU as existence score */
	*out_score = iou;  /* or a scalar slice */
}
```

**Step 3: Commit**

```
git commit -m "video/tracker: extract object pointer and existence score"
```

---

### Task 8: Implement propagate

**Files:**
- Modify: `src/model/sam3_video.c` — replace stub propagate
- Test: `tests/test_video_api.c` — propagate produces masks

**Step 1: Write the failing test**

```c
static void test_video_propagate_tracks_across_frames(void)
{
	/* 2-frame video, prompt on frame 0 */
	struct sam3_video_session *s = /* setup */;
	sam3_video_add_points(s, 0, 0, pts, 1, &r0);

	int frame_mask_counts[2] = {0};
	sam3_video_propagate(s, SAM3_PROPAGATE_FORWARD,
	                     count_cb, frame_mask_counts);

	ASSERT_GT(frame_mask_counts[0], 0);
	ASSERT_GT(frame_mask_counts[1], 0);
}
```

Where `count_cb` counts non-zero mask pixels per frame.

**Step 2: Implement**

Replace the propagate body in `src/model/sam3_video.c`. Structure:

```c
/* Helper: run one frame */
static int propagate_one(struct sam3_video_session *s, int f,
                         struct sam3_result *out)
{
	if (sam3_session_is_prompted(s, f)) {
		/* Re-run the prompt pipeline for every prompt on this
		 * frame, merging masks per object. */
		for (int i = 0; i < s->n_prompts; i++) {
			if (s->prompts[i].frame_idx != f) continue;
			/* call sam3_tracker_segment_prompted */
		}
	} else {
		/* Pure tracking */
		struct sam3_tensor *masks = NULL, *iou = NULL;
		struct sam3_tensor *obj_ptr = NULL;
		int err = sam3_tracker_track_frame(
			s->ctx, &s->tracker, &s->scratch, g,
			s->cached_features[f], f, /*is_cond=*/0,
			&masks, &iou, &obj_ptr, NULL);
		if (err) return err;
		/* Memory encode + bank add (is_cond=false) */
		struct sam3_tensor *mem = NULL;
		sam3_memory_encoder_build(g, arena, &s->tracker.mem_encoder,
		                           s->cached_features[f], /*best*/masks,
		                           &mem);
		sam3_memory_bank_add(&s->tracker.mem_bank, f, mem, obj_ptr,
		                     0, /*score=*/...);
	}
	/* Fill out->masks */
	return SAM3_OK;
}

/* Main loop */
int nf = s->frames.n_frames;
int start, end, step;
if (direction == SAM3_PROPAGATE_FORWARD ||
    direction == SAM3_PROPAGATE_BOTH) {
	for (int f = 0; f < nf; f++) {
		struct sam3_result r = {0};
		int err = propagate_one(s, f, &r);
		if (err) return err;
		if (callback(f, &r, s->n_objects, s->obj_ids, user_data))
			return SAM3_OK;
		sam3_arena_reset(&s->scratch);
	}
}
if (direction == SAM3_PROPAGATE_BACKWARD ||
    direction == SAM3_PROPAGATE_BOTH) {
	/* Reset bank before backward pass so memory is rebuilt */
	sam3_memory_bank_clear(&s->tracker.mem_bank);
	for (int f = nf - 1; f >= 0; f--) {
		/* same as above */
	}
}
```

**Step 3: Commit**

```
git commit -m "video/api: implement propagate with conditioning and pure-tracking frames"
```

---

### Task 9: Fix session reset

**Files:**
- Modify: `src/model/sam3_video.c` — `sam3_video_reset`
- Test: `tests/test_video_api.c` — reset clears all state

**Step 1: Test**

```c
static void test_video_reset_clears_all_state(void)
{
	/* add prompts, propagate, then reset */
	sam3_video_add_points(s, 0, 0, pts, 1, &r);
	sam3_video_reset(s);

	ASSERT_EQ(s->n_objects, 0);
	ASSERT_EQ(s->n_prompts, 0);
	ASSERT_FALSE(sam3_session_is_prompted(s, 0));
	ASSERT_EQ(sam3_memory_bank_total(&s->tracker.mem_bank), 0);
}
```

**Step 2: Implement**

```c
void sam3_video_reset(struct sam3_video_session *s)
{
	if (!s) return;
	sam3_tracker_reset(&s->tracker);
	sam3_session_clear_prompts(s);
	memset(s->frames_tracked, 0, s->frames.n_frames);
	memset(s->obj_ids, 0, sizeof(s->obj_ids));
	s->n_objects = 0;
	sam3_arena_reset(&s->scratch);
}
```

**Step 3: Commit**

```
git commit -m "video/api: reset clears prompts, bitmap, and scratch arena"
```

---

## Phase B — CLI

### Task 10: Create cli_track.c subcommand

**Files:**
- Create: `tools/cli_track.c`
- Create: `tools/cli_track.h`
- Modify: `tools/sam3_cli.c` — register `track`
- Modify: `CMakeLists.txt` — add cli_track.c
- Test: `tests/test_cli_track.c` — argument parsing

**Step 1: Write the failing test**

Parsing test (no model inference):

```c
static void test_cli_track_parses_args(void)
{
	char *argv[] = {"sam3_cli", "track", "-m", "model.sam3",
	                 "-v", "clip.mp4", "-p", "100,100,1", "--frame", "0",
	                 "-o", "out"};
	struct cli_track_args args = {0};
	int err = cli_track_parse(12, argv, &args);
	ASSERT_EQ(err, 0);
	ASSERT_STR_EQ(args.model_path, "model.sam3");
	ASSERT_STR_EQ(args.video_path, "clip.mp4");
	ASSERT_EQ(args.n_points, 1);
	ASSERT_EQ(args.frame_idx, 0);
}
```

**Step 2: Implement**

Port the 179 lines from `tools/sam3_main.c` (`--video` block), adapting
to the cli_segment pattern: separate `cli_track_parse`, `cli_track_run`.
Use `sam3_video_start/add_points/add_box/propagate/end`.

Per-frame callback writes `<outdir>/frame_NNNNN.png` via
`sam3_image_overlay_write` (reuse the existing helper from cli_segment).

**Step 3: Register in sam3_cli.c**

```c
if (strcmp(argv[1], "track") == 0) {
	return cli_track_main(argc - 1, argv + 1);
}
```

**Step 4: Add to CMakeLists.txt**

```cmake
add_executable(sam3_cli
	tools/sam3_cli.c
	tools/cli_common.c
	tools/cli_segment.c
	tools/cli_track.c       # NEW
	tools/cli_convert.c
	tools/cli_info.c
	tools/weight_rename.c
	tools/weight_conv_perm.c)
```

**Step 5: Commit**

```
git commit -m "cli: add track subcommand for video object tracking"
```

---

### Task 11: Delete orphaned sam3_main.c

**Files:**
- Delete: `tools/sam3_main.c`
- Modify: `tools/cli_common.h` — remove stale `Used by:` line

**Step 1:** Verify nothing references it:

```
grep -rn "sam3_main" .  # excluding build artifacts
```
Expected: only the `cli_common.h` comment.

**Step 2: Delete**

```
git rm tools/sam3_main.c
```

**Step 3: Fix the stale comment in cli_common.h**

Update the header's `Used by:` line.

**Step 4: Commit**

```
git commit -m "tools: remove orphaned sam3_main.c (superseded by cli_track)"
```

---

## Phase C — Tests and fixtures

### Task 12: Add assert_tensor_close helper

**Files:**
- Modify: `tests/test_helpers.h`
- Test: `tests/test_helpers_test.c` (if exists) or a self-test macro

**Step 1: Write**

```c
#define ASSERT_TENSOR_CLOSE(tensor, expected_path, rtol, atol) do { \
	struct sam3_tensor *__exp = load_safetensor(expected_path); \
	ASSERT_NOT_NULL(__exp); \
	ASSERT_EQ((tensor)->n_dims, __exp->n_dims); \
	for (int __i = 0; __i < (tensor)->n_dims; __i++) \
		ASSERT_EQ((tensor)->ne[__i], __exp->ne[__i]); \
	size_t __n = sam3_tensor_nelem((tensor)); \
	for (size_t __k = 0; __k < __n; __k++) { \
		float __a = ((float *)(tensor)->data)[__k]; \
		float __e = ((float *)__exp->data)[__k]; \
		float __tol = (atol) + (rtol) * fabsf(__e); \
		if (fabsf(__a - __e) > __tol) { \
			fprintf(stderr, "%s: mismatch at %zu: " \
			        "actual=%f expected=%f tol=%f\n", \
			        expected_path, __k, __a, __e, __tol); \
			exit(1); \
		} \
	} \
} while (0)
```

**Step 2: Commit**

```
git commit -m "test/helpers: add tolerance-based tensor comparison"
```

---

### Task 13: Replace silent skip in test_tracker_fixtures

**Files:**
- Modify: `tests/test_tracker_fixtures.c` — remove skip, add numerical asserts
- Modify: `CMakeLists.txt` — add `SAM3_FIXTURE_TESTS` option

**Step 1: Add CMake option**

```cmake
option(SAM3_FIXTURE_TESTS
	"Register tests that require Python-generated fixtures" OFF)

# In the test glob block:
if(SAM3_FIXTURE_TESTS)
	# register test_tracker_fixtures
else()
	list(REMOVE_ITEM TEST_SOURCES
		${CMAKE_SOURCE_DIR}/tests/test_tracker_fixtures.c)
endif()
```

**Step 2: Rewrite test to use real asserts**

Replace every `ASSERT(data != NULL)` + shape check with
`ASSERT_TENSOR_CLOSE(actual_tensor, fixture_path, 1e-3, 1e-4)` or
appropriate tolerance from the design.

Remove the silent-skip block entirely — absence of fixtures becomes a
hard failure.

**Step 3: Commit**

```
git commit -m "test/tracker: real numerical validation gated on SAM3_FIXTURE_TESTS"
```

---

### Task 14: Add moving-square clip to fixture generator

**Files:**
- Modify: `scripts/gen_tracker_fixtures.py`

**Step 1: Add synthetic clip mode**

Extend the script with `--clip-type square` flag. Generate 8 frames of
a 32×32 white square on noisy gray background, sliding diagonally
8 px/frame. Seed NumPy with 0 for determinism. Save to
`tests/fixtures/tracker/moving_square/`.

**Step 2: Run fixture generator**

```
python scripts/gen_tracker_fixtures.py \
	--checkpoint path/to/sam3_video.pth \
	--clip-type square \
	--output tests/fixtures/tracker/moving_square
```

**Step 3: Commit fixture metadata (not large tensors — add to .gitignore if needed)**

```
git add scripts/gen_tracker_fixtures.py
# Fixtures themselves stay out of git unless small
git commit -m "scripts: add moving-square clip to fixture generator"
```

---

### Task 15: Write test_video_e2e integration test

**Files:**
- Create: `tests/test_video_e2e.c`

**Step 1: Write**

```c
/* Generates 8-frame moving-square clip in-process (no fixture
 * dependency), loads it via a temp directory, runs full add_points +
 * propagate, asserts centroid tracking. */

static void generate_moving_square_clip(const char *dir, int n);

static int collect_cb(int frame_idx, const struct sam3_result *r,
                      int n_obj, const int *obj_ids, void *ud)
{
	struct centroid_tracker *t = ud;
	t->centroids[frame_idx] = compute_centroid(&r->masks[0]);
	return 0;
}

static void test_video_e2e_tracks_moving_square(void)
{
	char tmpdir[] = "/tmp/sam3_e2e_XXXXXX";
	mkdtemp(tmpdir);
	generate_moving_square_clip(tmpdir, 8);

	struct sam3_ctx *ctx = sam3_ctx_create(/* ... */);
	load_real_model(ctx);  /* requires a test model */

	struct sam3_video_session *s = NULL;
	ASSERT_EQ(sam3_video_start(ctx, tmpdir, &s), SAM3_OK);

	/* Prompt at center of square in frame 0: (64, 64) */
	struct sam3_point pts[1] = { {64.f, 64.f, 1} };
	struct sam3_result r = {0};
	ASSERT_EQ(sam3_video_add_points(s, 0, 0, pts, 1, &r), SAM3_OK);

	struct centroid_tracker tracker = {0};
	sam3_video_propagate(s, SAM3_PROPAGATE_FORWARD, collect_cb, &tracker);

	/* Assert centroids follow the diagonal motion */
	for (int f = 0; f < 8; f++) {
		float exp_x = 64.f + 8.f * f;
		float exp_y = 64.f + 8.f * f;
		ASSERT_NEAR(tracker.centroids[f].x, exp_x, 8.f);
		ASSERT_NEAR(tracker.centroids[f].y, exp_y, 8.f);
	}

	sam3_video_end(s);
	rmtree(tmpdir);
}
```

**Step 2: Gate on model availability**

Guard with an env var (`SAM3_TEST_MODEL=path/to/model.sam3`). If not
set, test is not compiled into CTest (similar to fixture tests).

**Step 3: Commit**

```
git commit -m "test: add end-to-end video tracking integration test"
```

---

## Phase D — Polish

### Task 16: Python VideoSession binding

**Files:**
- Create: `python/sam3/video.py`
- Modify: `python/sam3/__init__.py` — export `VideoSession`
- Create: `python/tests/test_video.py`

**Step 1: Write pytest**

```python
def test_video_session_tracks_moving_square(tmp_path, synthetic_model):
	generate_moving_square_clip(tmp_path, n=8)
	with sam3.VideoSession(synthetic_model, str(tmp_path)) as s:
		mask0 = s.add_points(frame=0, obj_id=0,
		                     points=[(64, 64, 1)])
		assert mask0.sum() > 0
		centroids = []
		for frame, masks in s.propagate(direction="forward"):
			centroids.append(compute_centroid(masks[0]))
		# Assert diagonal motion
		for i, (cx, cy) in enumerate(centroids):
			assert abs(cx - (64 + 8*i)) < 8
			assert abs(cy - (64 + 8*i)) < 8
```

**Step 2: Implement VideoSession**

Thin CFFI wrapper. Key piece: `propagate` creates a CFFI callback
bridge that pushes results onto a queue; the Python iterator yields
from the queue.

Sketch:
```python
class VideoSession:
	def __init__(self, model, video_path):
		self._session = _lib.sam3_video_start(model._ctx,
		                                       video_path.encode())
	def add_points(self, *, frame, obj_id, points):
		arr = np.array(points, dtype=...)
		result = _ffi.new("sam3_result*")
		err = _lib.sam3_video_add_points(self._session, frame, obj_id,
		                                  arr, len(points), result)
		_check(err)
		return _result_to_numpy(result)
	def propagate(self, *, direction="forward"):
		import queue
		q = queue.Queue()
		@_ffi.callback("int(int, const sam3_result*, int, const int*, void*)")
		def cb(f, r, nobj, ids, ud):
			q.put((f, _result_to_numpy(r)))
			return 0
		dir_ = {"forward": 0, "backward": 1, "both": 2}[direction]
		_lib.sam3_video_propagate(self._session, dir_, cb, _ffi.NULL)
		while not q.empty():
			yield q.get()
	def close(self):
		_lib.sam3_video_end(self._session)
	def __enter__(self): return self
	def __exit__(self, *a): self.close()
```

**Step 3: Commit**

```
git commit -m "python: add VideoSession wrapper for video tracking"
```

---

### Task 17: Docs (README + architecture.md)

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Step 1: README**

Add to Features list:
- "Video object tracking with memory-based frame propagation (MPEG and
  frame-directory input)"

Add Quick Start section after the image example:
````markdown
### Video tracking

Track an object across frames of a video:

```bash
sam3_cli track -m sam3.sam3 -v clip.mp4 \
	-p 504,504,1 --frame 0 -o out/
```

Output: `out/frame_NNNNN.png` overlays per frame.
````

**Step 2: architecture.md**

Add a Section "Video Tracker" with the data flow diagram, memory bank
policy, and limits per the design doc. Remove the stale line 1901.

**Step 3: Commit**

```
git commit -m "docs: document video tracker feature"
```

---

### Task 18: Benchmarks

**Files:**
- Create: `src/bench/bench_video_frame.c`
- Create: `src/bench/bench_video_end_to_end.c`
- Modify: `src/bench/bench_suite.c` (or equivalent registration point)

**Step 1: Write bench_video_frame**

Mirror the pattern of existing pipeline benchmarks. Pre-populate the
memory bank with one entry, then time `sam3_tracker_track_frame` in a
loop.

**Step 2: Write bench_video_end_to_end**

Run the full `video_start → add_points → propagate → end` on the
synthetic 8-frame clip in a timed loop.

**Step 3: Register both in the bench harness**

**Step 4: Commit**

```
git commit -m "bench: add video per-frame and end-to-end benchmarks"
```

---

## Verification checklist

After every task, before commit:

1. `cmake --build build` — no warnings, no errors
2. `ctest --output-on-failure` — all green
3. `git diff` — review changes; no drive-by modifications outside scope

After Phase A: `test_video_e2e` passes (if model available).
After Phase B: `sam3_cli track --help` prints; basic invocation works.
After Phase C: `SAM3_FIXTURE_TESTS=ON` test passes with real fixtures.
After Phase D: `pytest python/tests/test_video.py` passes; README
renders; benchmarks run.

## Rollback points

Each phase ends in a working state. If Phase A commits compile and
pass existing tests but the new integration test fails, keep Phase A,
skip Phase B-D until the issue is understood.

## Commit discipline

- One logical change per commit (per plan step).
- CLAUDE.md commit format: `subsystem: imperative description`.
- Never bundle unrelated changes.
