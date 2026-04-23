/*
 * tests/test_batched_ops.c - Unit tests for batched building blocks.
 *
 * Each batched op (scorer, seg head, decoder substeps, CPU helpers)
 * is verified here with small synthetic tensors so the tests run
 * without loading a model. Every test loops over both the CPU and
 * Metal backends via run_both_backends(): CPU first (tight tolerance,
 * easier failure localization), Metal second (loose tolerance, the
 * deployment backend). A Metal-only failure indicates backend
 * divergence worth escalating.
 *
 * Key types:  sam3_backend, sam3_arena, sam3_graph, sam3_tensor
 * Depends on: test_helpers.h, backend/backend.h, core/alloc.h,
 *             core/graph.h, model/graph_helpers.h
 * Used by:    CTest
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include "test_helpers.h"

#include <stdio.h>

#include "backend/backend.h"
#include "core/alloc.h"
#include "core/graph.h"
#include "model/graph_helpers.h"

/*
 * Helper: run a backend-parameterized test case on both CPU and Metal.
 * The callback receives the backend handle and a short human name
 * ("CPU" / "Metal") so failure messages can identify which backend
 * diverged.
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

/*
 * Placeholder so the file compiles before Task 3 fills in the scorer
 * test body. Tasks 3-13 will declare more case fns above main() and
 * invoke them via run_both_backends().
 */
static void scorer_batched_case(struct sam3_backend *be, const char *name);

static void test_scorer_batched_equals_per_slot(void)
{
	/* Filled in Task 3; stub for now so the file compiles. */
	(void)scorer_batched_case;
}

static void scorer_batched_case(struct sam3_backend *be, const char *name)
{
	(void)be;
	(void)name;
	/* Filled in Task 3. */
}

int main(void)
{
	test_scorer_batched_equals_per_slot();
	TEST_REPORT();
}
