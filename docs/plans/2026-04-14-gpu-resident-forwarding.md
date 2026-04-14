# GPU-Resident Forwarding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate GPU-to-host readback between consecutive graph_eval calls by keeping intermediate output mlx_arrays in the tensor map for zero-copy forwarding.

**Architecture:** Add a `no_readback` flag to `sam3_graph`. When set, `metal_graph_eval` skips Phase 3 (readback) and keeps final output mlx_arrays in the tensor map. The next `graph_eval` finds them via `metal_wrap_tensor` without data transfer. Requires three supporting changes: (1) `metal_map_put` must handle updates, (2) Phase 1.5 intermediate detection must only count forward edges, (3) ViT builder must reuse a persistent forwarding tensor.

**Tech Stack:** C11, MLX-C, Metal backend

---

### Task 1: Add `no_readback` flag to `sam3_graph`

**Files:**
- Modify: `src/core/graph.h:74-77`

**Step 1: Add bool field and include**

In `src/core/graph.h`, add `#include <stdbool.h>` after the existing `#include "tensor.h"`, then add the `no_readback` field to the struct:

```c
#include "tensor.h"
#include <stdbool.h>

/* ... */

struct sam3_graph {
	struct sam3_node nodes[SAM3_GRAPH_MAX_NODES];
	int              n_nodes;
	bool             no_readback; /* Skip Phase 3; keep outputs GPU-resident */
};
```

No change to `sam3_graph_init` — `memset(g, 0, sizeof(*g))` already sets `no_readback = false`.

**Step 2: Verify build**

Run: `cd /Users/rbisri/Documents/sam3/build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)`
Expected: Clean compile, no warnings about the new field.

**Step 3: Commit**

```
core/graph: add no_readback flag for GPU-resident forwarding
```

---

### Task 2: Make `metal_map_put` handle updates

**Files:**
- Modify: `src/backend/metal/metal_backend.c:164-193`

**Context:** `metal_map_put` currently only inserts into empty/tombstone slots. With `no_readback`, a tensor may already be in the map from a previous graph_eval. When `metal_dispatch_node` stores a new result for the same tensor, `metal_map_put` must update the existing entry instead of creating a duplicate.

**Step 1: Write the failing test**

In `tests/test_metal.c`, add before the `#endif /* SAM3_HAS_METAL */` line (line 955):

```c
/*
 * Test that metal_map_put handles updates: when a tensor's mlx_array
 * is already in the map (from a previous no_readback eval), a second
 * graph_eval that outputs to the same tensor must overwrite the entry,
 * not create a duplicate.
 */
static void test_metal_map_put_update(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(metal != NULL);
	if (!metal)
		return;

	/*
	 * Tensors: a=[1,2,3,4], b=[10,20,30,40], c=forwarding,
	 *          d=[100,200,300,400]
	 */
	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
	float c_data[4] = {0};
	float d_data[] = {100.0f, 200.0f, 300.0f, 400.0f};

	int dims[] = {4};
	struct sam3_tensor a = make_tensor(SAM3_DTYPE_F32, 1, dims);
	a.data = a_data; a.nbytes = sizeof(a_data);
	sam3_tensor_compute_strides(&a);

	struct sam3_tensor b = make_tensor(SAM3_DTYPE_F32, 1, dims);
	b.data = b_data; b.nbytes = sizeof(b_data);
	sam3_tensor_compute_strides(&b);

	struct sam3_tensor c = make_tensor(SAM3_DTYPE_F32, 1, dims);
	c.data = c_data; c.nbytes = sizeof(c_data);
	sam3_tensor_compute_strides(&c);

	struct sam3_tensor d = make_tensor(SAM3_DTYPE_F32, 1, dims);
	d.data = d_data; d.nbytes = sizeof(d_data);
	sam3_tensor_compute_strides(&d);

	/*
	 * Graph 1: c = a + b (no_readback)
	 * c's mlx_array stays in the tensor map.
	 */
	struct sam3_graph g1;
	sam3_graph_init(&g1);
	g1.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &c,
	};
	g1.n_nodes = 1;
	g1.no_readback = true;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g1), SAM3_OK);

	/*
	 * Graph 2: c = c + d (readback)
	 * c is BOTH input (from map) and output (overwrite).
	 * metal_map_put must update, not duplicate.
	 */
	struct sam3_graph g2;
	sam3_graph_init(&g2);
	g2.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&c, &d}, .output = &c,
	};
	g2.n_nodes = 1;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g2), SAM3_OK);

	/* c should be (a + b) + d = {111, 222, 333, 444} */
	float expected[] = {111.0f, 222.0f, 333.0f, 444.0f};
	ASSERT(float_arrays_match((float *)c.data, expected, 4, 1e-6f));

	sam3_backend_free(metal);
}
```

Register it in `main()` inside the `#ifdef SAM3_HAS_CPU` block:
```c
	test_metal_map_put_update();
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rbisri/Documents/sam3/build && make -j$(nproc) && ctest -R test_metal --output-on-failure`
Expected: FAIL — `metal_map_put` creates a duplicate entry, or the no_readback path doesn't exist yet, causing c to be evicted after graph 1.

**Step 3: Implement metal_map_put update logic**

In `src/backend/metal/metal_backend.c`, replace `metal_map_put` (lines 164-193) with:

```c
static int metal_map_put(struct sam3_metal_backend *mtl,
			 const struct sam3_tensor *key, mlx_array val)
{
	/* Rehash at 75% load factor to keep probe chains short */
	if (mtl->map_count >= mtl->map_capacity * 3 / 4) {
		if (metal_map_rehash(mtl) < 0) {
			sam3_log_error("metal: map rehash failed at %d/%d",
				       mtl->map_count, mtl->map_capacity);
			return -1;
		}
	}

	int cap = mtl->map_capacity;
	unsigned mask = (unsigned)(cap - 1);
	unsigned idx = metal_map_slot(key, cap);
	int first_empty = -1;

	for (int i = 0; i < cap; i++) {
		unsigned slot = (idx + (unsigned)i) & mask;
		if (mtl->map_keys[slot] == key) {
			/* Update existing entry in-place */
			mlx_array_free(mtl->map_vals[slot]);
			mtl->map_vals[slot] = val;
			return 0;
		}
		if (!mtl->map_keys[slot]) {
			if (first_empty < 0)
				first_empty = (int)slot;
			break;
		}
		if (mtl->map_keys[slot] == METAL_MAP_TOMBSTONE
		    && first_empty < 0) {
			first_empty = (int)slot;
		}
	}

	if (first_empty >= 0) {
		mtl->map_keys[first_empty] = key;
		mtl->map_vals[first_empty] = val;
		mtl->map_count++;
		return 0;
	}
	sam3_log_error("metal: tensor map full (%d entries)",
		       mtl->map_count);
	return -1;
}
```

**Step 4: Run build to verify compilation**

Run: `cd /Users/rbisri/Documents/sam3/build && make -j$(nproc)`
Expected: Clean compile. Test still fails (no_readback not implemented yet).

**Step 5: Commit**

```
metal: make map_put handle updates for GPU-resident forwarding
```

---

### Task 3: Implement `no_readback` in `metal_graph_eval`

**Files:**
- Modify: `src/backend/metal/metal_backend.c:1580-1595` (Phase 1.5 fix)
- Modify: `src/backend/metal/metal_backend.c:1630-1795` (Phase 3 no_readback path)

**Step 1: Fix Phase 1.5 forward-edge detection**

In `metal_graph_eval`, Phase 1.5 marks a node as intermediate if its output tensor matches any other node's input. This breaks with GPU-resident forwarding when the same tensor (x_fwd) is both input to node 0 (from previous graph) and output of the last node (current graph). The fix: only count forward edges (producer before consumer).

In the Phase 1.5 inner loop (around line 1588), change:

```c
				if (imap[s].key == inp) {
					is_intermediate[imap[s].idx]
						= true;
					break;
				}
```

to:

```c
				if (imap[s].key == inp
				    && imap[s].idx < i) {
					is_intermediate[imap[s].idx]
						= true;
					break;
				}
```

This ensures only forward dependencies (producer index < consumer index) are counted. A tensor used as output of the last node AND input of the first node is a cross-graph dependency, not a within-graph intermediate.

**Step 2: Add no_readback path in Phase 3**

Replace the Phase 3 block (lines 1630-1795) by wrapping it with the no_readback check. The existing code becomes the `else` branch:

```c
	/*
	 * Phase 3: readback or GPU-resident forwarding.
	 *
	 * When no_readback is set, skip host readback entirely.
	 * Evict within-graph intermediates and data-less outputs
	 * from the tensor map, but keep final outputs with host
	 * buffers resident for the next graph_eval to consume
	 * via metal_wrap_tensor.
	 */
	if (g->no_readback) {
		for (int i = 0; i < g->n_nodes; i++) {
			struct sam3_tensor *out_t = g->nodes[i].output;
			if (is_intermediate[i] || !out_t->data)
				metal_map_evict(mtl, out_t);
		}
		sam3_log_debug("metal_eval: no_readback, kept %d "
			"final outputs GPU-resident", g->n_nodes);
	} else {
		/* existing Phase 3 code unchanged */
		...
	}
```

Wrap the entire existing Phase 3 block (from `{` after the comment on line 1637 through the closing `}` on line 1795) inside the `else` branch.

**Step 3: Run test to verify it passes**

Run: `cd /Users/rbisri/Documents/sam3/build && make -j$(nproc) && ctest -R test_metal --output-on-failure`
Expected: `test_metal_map_put_update` PASSES.

**Step 4: Commit**

```
metal: implement no_readback GPU-resident forwarding in graph_eval
```

---

### Task 4: Write GPU-resident forwarding correctness tests

**Files:**
- Modify: `tests/test_metal.c`

**Step 1: Add basic forwarding test**

Tests the core scenario: graph 1 outputs with no_readback, graph 2 consumes the GPU-resident tensor.

```c
/*
 * Test GPU-resident forwarding: graph 1 skips readback, graph 2
 * finds the output in the tensor map and uses it directly.
 */
static void test_metal_no_readback_forward(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(metal != NULL);
	if (!metal)
		return;

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
	float d_data[] = {100.0f, 200.0f, 300.0f, 400.0f};

	int dims[] = {4};
	struct sam3_tensor a = make_tensor(SAM3_DTYPE_F32, 1, dims);
	a.data = a_data; a.nbytes = sizeof(a_data);
	sam3_tensor_compute_strides(&a);

	struct sam3_tensor b = make_tensor(SAM3_DTYPE_F32, 1, dims);
	b.data = b_data; b.nbytes = sizeof(b_data);
	sam3_tensor_compute_strides(&b);

	/* Forwarding tensor: persists across both graphs */
	float c_data[4] = {0};
	struct sam3_tensor c = make_tensor(SAM3_DTYPE_F32, 1, dims);
	c.data = c_data; c.nbytes = sizeof(c_data);
	sam3_tensor_compute_strides(&c);

	struct sam3_tensor d = make_tensor(SAM3_DTYPE_F32, 1, dims);
	d.data = d_data; d.nbytes = sizeof(d_data);
	sam3_tensor_compute_strides(&d);

	float e_data[4] = {0};
	struct sam3_tensor e = make_tensor(SAM3_DTYPE_F32, 1, dims);
	e.data = e_data; e.nbytes = sizeof(e_data);
	sam3_tensor_compute_strides(&e);

	/* Graph 1: c = a + b (no_readback) */
	struct sam3_graph g1;
	sam3_graph_init(&g1);
	g1.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &c,
	};
	g1.n_nodes = 1;
	g1.no_readback = true;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g1), SAM3_OK);

	/* c host data should be unchanged (no readback) */
	ASSERT(c_data[0] == 0.0f);

	/* Graph 2: e = c + d (readback) */
	struct sam3_graph g2;
	sam3_graph_init(&g2);
	g2.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&c, &d}, .output = &e,
	};
	g2.n_nodes = 1;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g2), SAM3_OK);

	/* e = (1+10)+100, (2+20)+200, ... = {111, 222, 333, 444} */
	float expected[] = {111.0f, 222.0f, 333.0f, 444.0f};
	ASSERT(float_arrays_match((float *)e.data, expected, 4, 1e-6f));

	sam3_backend_free(metal);
}
```

**Step 2: Add multi-hop forwarding test**

Tests 3 sequential graph_evals: g1 (no_readback) -> g2 (no_readback) -> g3 (readback). Simulates the ViT pattern of multiple batches.

```c
/*
 * Test multi-hop GPU-resident forwarding (simulates ViT batch chain).
 * g1: c = a + b (no_readback)
 * g2: c = c + d (no_readback, c reused as input+output)
 * g3: e = c + a (readback)
 * Expected: e = ((a+b)+d) + a = {112, 224, 336, 448}
 */
static void test_metal_no_readback_chain(void)
{
	struct sam3_backend *metal = sam3_backend_init(SAM3_BACKEND_METAL);
	ASSERT(metal != NULL);
	if (!metal)
		return;

	float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
	float b_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
	float d_data[] = {100.0f, 200.0f, 300.0f, 400.0f};

	int dims[] = {4};
	struct sam3_tensor a = make_tensor(SAM3_DTYPE_F32, 1, dims);
	a.data = a_data; a.nbytes = sizeof(a_data);
	sam3_tensor_compute_strides(&a);

	struct sam3_tensor b = make_tensor(SAM3_DTYPE_F32, 1, dims);
	b.data = b_data; b.nbytes = sizeof(b_data);
	sam3_tensor_compute_strides(&b);

	float c_data[4] = {0};
	struct sam3_tensor c = make_tensor(SAM3_DTYPE_F32, 1, dims);
	c.data = c_data; c.nbytes = sizeof(c_data);
	sam3_tensor_compute_strides(&c);

	struct sam3_tensor d = make_tensor(SAM3_DTYPE_F32, 1, dims);
	d.data = d_data; d.nbytes = sizeof(d_data);
	sam3_tensor_compute_strides(&d);

	float e_data[4] = {0};
	struct sam3_tensor e = make_tensor(SAM3_DTYPE_F32, 1, dims);
	e.data = e_data; e.nbytes = sizeof(e_data);
	sam3_tensor_compute_strides(&e);

	/* g1: c = a + b (no_readback) */
	struct sam3_graph g1;
	sam3_graph_init(&g1);
	g1.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&a, &b}, .output = &c,
	};
	g1.n_nodes = 1;
	g1.no_readback = true;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g1), SAM3_OK);

	/* g2: c = c + d (no_readback, c is input+output) */
	struct sam3_graph g2;
	sam3_graph_init(&g2);
	g2.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&c, &d}, .output = &c,
	};
	g2.n_nodes = 1;
	g2.no_readback = true;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g2), SAM3_OK);

	/* g3: e = c + a (readback) */
	struct sam3_graph g3;
	sam3_graph_init(&g3);
	g3.nodes[0] = (struct sam3_node){
		.op = SAM3_OP_ADD, .n_inputs = 2,
		.inputs = {&c, &a}, .output = &e,
	};
	g3.n_nodes = 1;

	ASSERT_EQ(metal->ops->graph_eval(metal, &g3), SAM3_OK);

	/* e = ((1+10)+100)+1 = 112, etc. */
	float expected[] = {112.0f, 224.0f, 336.0f, 448.0f};
	ASSERT(float_arrays_match((float *)e.data, expected, 4, 1e-6f));

	sam3_backend_free(metal);
}
```

**Step 3: Register tests in main()**

Inside the `#ifdef SAM3_HAS_CPU` block in `main()`, add:

```c
	test_metal_no_readback_forward();
	test_metal_no_readback_chain();
```

**Step 4: Run tests**

Run: `cd /Users/rbisri/Documents/sam3/build && make -j$(nproc) && ctest -R test_metal --output-on-failure`
Expected: ALL tests pass including the three new ones.

**Step 5: Commit**

```
tests: add GPU-resident forwarding correctness tests
```

---

### Task 5: Integrate GPU-resident forwarding into ViT builder

**Files:**
- Modify: `src/model/image_encoder.c:608-787`

**Context:** The ViT evaluates 32 blocks in batches of 4 on Metal (8 graph_eval calls). Currently each batch reads back the residual to `x_buf` (in persist arena), then the next batch wraps `x_buf` into a new tensor and re-uploads. With GPU-resident forwarding, batches 0-6 skip readback. The residual's mlx_array stays in the tensor map. Batch 7 reads back normally.

**Step 1: Allocate persistent forwarding tensor before the batch loop**

After line 611 (`int batch = skip_data ? 4 : 2;`), add:

```c
		/*
		 * GPU-resident forwarding tensor. Allocated in
		 * persist arena so it survives scratch resets.
		 * On non-last batches, the output mlx_array stays
		 * in the tensor map; the next batch finds it via
		 * metal_wrap_tensor without data transfer.
		 */
		struct sam3_tensor *x_fwd = NULL;
		if (skip_data) {
			int fwd_dims[] = {np, e};
			x_fwd = gh_alloc_tensor(persist, SAM3_DTYPE_F32,
						 2, fwd_dims);
			if (!x_fwd)
				return NULL;
			x_fwd->data = x_buf;
		}
```

**Step 2: Use x_fwd as input on subsequent batches**

Replace the input wrapping (lines 623-627):

```c
			int x_dims[] = {np, e};
			x = gh_tensor_wrap(scratch, SAM3_DTYPE_F32,
					    2, x_dims, x_buf);
			if (!x)
				return NULL;
```

with:

```c
			if (skip_data && base > 0) {
				/* GPU-resident: x_fwd in tensor map */
				x = x_fwd;
			} else {
				int x_dims[] = {np, e};
				x = gh_tensor_wrap(scratch,
						    SAM3_DTYPE_F32,
						    2, x_dims, x_buf);
				if (!x)
					return NULL;
			}
```

**Step 3: Redirect final output and set no_readback**

Replace the readback section (lines 776-786):

```c
			/* Assign host buffer for Phase 3 readback */
			if (skip_data)
				x->data = x_buf;

			err = be->ops->graph_eval(be, &g);
			scratch->skip_data = 0;
			if (err != SAM3_OK)
				return NULL;

			if (!skip_data)
				memcpy(x_buf, x->data, x_bytes);
```

with:

```c
			if (skip_data) {
				bool last = (end >= vit->depth);
				/*
				 * Redirect last node output to the
				 * persistent forwarding tensor so its
				 * mlx_array key survives scratch reset.
				 */
				g.nodes[g.n_nodes - 1].output = x_fwd;
				x = x_fwd;
				g.no_readback = !last;
			}

			err = be->ops->graph_eval(be, &g);
			scratch->skip_data = 0;
			if (err != SAM3_OK)
				return NULL;

			if (!skip_data)
				memcpy(x_buf, x->data, x_bytes);
```

**Step 4: Build and run tests**

Run: `cd /Users/rbisri/Documents/sam3/build && make -j$(nproc) && ctest --output-on-failure`
Expected: All tests pass including existing ViT tests.

**Step 5: Commit**

```
metal: enable GPU-resident forwarding for ViT block batches
```

---

### Task 6: Update TODO.md

**Files:**
- Modify: `docs/TODO.md`

**Step 1: Mark item 5 as done**

Update the heading and add resolution text:

```markdown
## 5. ~~Multi-stream pipelining~~ [x] DONE (GPU-resident forwarding)

**Resolution:** Replaced the multi-stream approach with GPU-resident
forwarding via `no_readback` flag on `sam3_graph`. When set, Phase 3
readback is skipped entirely — output mlx_arrays stay in the tensor
map. The next `graph_eval` finds them via `metal_wrap_tensor` without
data transfer (no GPU→host readback, no host→GPU re-upload).

Applied to the ViT block batch loop: 7 of 8 batches skip readback,
eliminating ~74 MB of round-trip data transfer (7 × 10.6 MB) for
the [5184, 1024] residual stream tensor.

**Files changed:**
- `src/core/graph.h` — `no_readback` field on `sam3_graph`
- `src/backend/metal/metal_backend.c` — no_readback path in Phase 3,
  forward-edge fix in Phase 1.5, update-aware `metal_map_put`
- `src/model/image_encoder.c` — persistent forwarding tensor, skip
  readback for non-last ViT batches
- `tests/test_metal.c` — 3 correctness tests (forward, chain, map update)
```

Update the priority list entry and remove from "Next steps" tasks.

**Step 2: Commit**

```
docs: mark GPU-resident forwarding as done in TODO
```
