/*
 * src/core/trace.h - Dtype tracing and numeric diagnostics
 *
 * Compile-time gated tracing system for inspecting tensor values, kernel
 * dispatch, and compute graph execution. When SAM3_HAS_TRACE is not defined,
 * all SAM3_TRACE_* macros expand to no-ops, but the compute functions
 * (sam3_trace_compute_stats, sam3_trace_compute_compare) are always available
 * for use in tests and diagnostics.
 *
 * Key types:  sam3_trace_flags, sam3_numeric_stats, sam3_compare_result
 * Depends on: core/tensor.h, core/graph.h
 * Used by:    backend/ files, tests/test_trace.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_CORE_TRACE_H
#define SAM3_CORE_TRACE_H

#include "tensor.h"
#include "graph.h"

/* Runtime flag bits controlling which categories are logged. */
enum sam3_trace_flags {
	SAM3_TRACE_KERNELS = 1 << 0,
	SAM3_TRACE_NUMERIC = 1 << 1,
	SAM3_TRACE_COMPARE = 1 << 2,
	SAM3_TRACE_GRAPH   = 1 << 3,
	SAM3_TRACE_ALL     = 0xFFFF,
};

/* Statistics computed over all elements of a tensor. */
struct sam3_numeric_stats {
	float min;
	float max;
	float mean;
	int   nan_count;
	int   inf_count;
	int   denormal_count;
	int   total_elems;
};

/* Comparison result between two tensors of the same shape. */
struct sam3_compare_result {
	float max_abs_error;
	float max_rel_error;
	float mean_abs_error;
	int   mismatches;
};

/*
 * sam3_trace_compute_stats - Compute numeric statistics over a tensor.
 *
 * @t:   Tensor to analyse (any dtype).
 * @out: Output struct filled with min, max, mean, nan/inf/denormal counts.
 *
 * Always available regardless of SAM3_HAS_TRACE. Converts each element
 * to f32 internally. NaN elements are excluded from min/max/mean.
 */
void sam3_trace_compute_stats(const struct sam3_tensor *t,
			      struct sam3_numeric_stats *out);

/*
 * sam3_trace_compute_compare - Compare two tensors element-by-element.
 *
 * @actual:    Tensor produced by the implementation under test.
 * @reference: Reference tensor to compare against.
 * @tolerance: Absolute error threshold; elements exceeding this are mismatches.
 * @out:       Output struct filled with error metrics.
 *
 * Always available regardless of SAM3_HAS_TRACE. Both tensors must have
 * the same total element count. Converts elements to f32 internally.
 */
void sam3_trace_compute_compare(const struct sam3_tensor *actual,
				const struct sam3_tensor *reference,
				float tolerance,
				struct sam3_compare_result *out);

/* ── Functions gated on SAM3_HAS_TRACE ─────────────────────────────── */

#ifdef SAM3_HAS_TRACE

/*
 * sam3_trace_set_flags - Set the runtime trace flag mask.
 *
 * @flags: Bitmask of sam3_trace_flags values. Pass SAM3_TRACE_ALL to
 *         enable all categories.
 */
void sam3_trace_set_flags(unsigned flags);

/* sam3_trace_get_flags - Return the current runtime trace flag mask. */
unsigned sam3_trace_get_flags(void);

/*
 * sam3_trace_kernel - Log a kernel dispatch event.
 *
 * @kernel_name: Short name of the kernel (e.g. "matmul_f16_f32").
 * @in_dtype:    Input tensor dtype.
 * @out_dtype:   Output tensor dtype.
 * @variant:     Implementation variant string (e.g. "neon", "scalar").
 *
 * Emits a debug log line when SAM3_TRACE_KERNELS flag is set.
 */
void sam3_trace_kernel(const char *kernel_name, enum sam3_dtype in_dtype,
		       enum sam3_dtype out_dtype, const char *variant);

/*
 * sam3_trace_numeric - Log numeric statistics for a tensor.
 *
 * @label: Human-readable label for the tensor (e.g. "attn_output").
 * @t:     Tensor to inspect.
 *
 * Computes stats and emits a debug log line when SAM3_TRACE_NUMERIC is set.
 */
void sam3_trace_numeric(const char *label, const struct sam3_tensor *t);

/*
 * sam3_trace_compare - Log comparison results between two tensors.
 *
 * @label:     Human-readable label.
 * @actual:    Tensor under test.
 * @reference: Reference tensor.
 * @tolerance: Absolute error threshold for mismatch counting.
 *
 * Computes comparison and emits a debug log line when SAM3_TRACE_COMPARE
 * is set.
 */
void sam3_trace_compare(const char *label, const struct sam3_tensor *actual,
			const struct sam3_tensor *reference, float tolerance);

/*
 * sam3_trace_graph_plan - Log the execution plan for a compute graph.
 *
 * @g: Compute graph to inspect.
 *
 * Logs up to 20 nodes (op name and output dtype) at debug level when
 * SAM3_TRACE_GRAPH flag is set.
 */
void sam3_trace_graph_plan(const struct sam3_graph *g);

/*
 * sam3_trace_graph_done - Log graph execution completion.
 *
 * @g:          The graph that finished executing.
 * @elapsed_ms: Wall-clock time for the execution in milliseconds.
 *
 * Emits a debug log line when SAM3_TRACE_GRAPH flag is set.
 */
void sam3_trace_graph_done(const struct sam3_graph *g, double elapsed_ms);

/* Convenience macros — active only with SAM3_HAS_TRACE. */
#define SAM3_TRACE_KERNEL(name, in_dt, out_dt, var) \
	do { sam3_trace_kernel((name), (in_dt), (out_dt), (var)); } while (0)
#define SAM3_TRACE_NUMERIC(label, tensor) \
	do { sam3_trace_numeric((label), (tensor)); } while (0)

#else /* !SAM3_HAS_TRACE */

#define SAM3_TRACE_KERNEL(name, in_dt, out_dt, var) ((void)0)
#define SAM3_TRACE_NUMERIC(label, tensor)           ((void)0)

#endif /* SAM3_HAS_TRACE */

#endif /* SAM3_CORE_TRACE_H */
