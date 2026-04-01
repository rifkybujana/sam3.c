/*
 * src/core/trace.c - Dtype tracing and numeric diagnostics implementation
 *
 * Implements the trace system declared in trace.h. The compute functions
 * (stats and compare) are unconditionally compiled. The logging functions
 * (kernel, numeric, compare, graph_plan, graph_done) are compiled only when
 * SAM3_HAS_TRACE is defined, and each checks the runtime flag mask before
 * emitting any output.
 *
 * Key types:  sam3_numeric_stats, sam3_compare_result
 * Depends on: core/trace.h, core/half.h, util/log.h
 * Used by:    backend/ files, tests/test_trace.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "trace.h"
#include "half.h"
#include "util/log.h"

/* ── Internal helpers ───────────────────────────────────────────────── */

/*
 * get_elem_f32 - Convert the i-th element of a tensor to float.
 *
 * @t: Source tensor (any supported dtype).
 * @i: Zero-based linear element index.
 *
 * Returns the element value as a 32-bit float. For integer dtypes
 * the value is cast to float without scaling.
 */
static float
get_elem_f32(const struct sam3_tensor *t, int i)
{
	switch (t->dtype) {
	case SAM3_DTYPE_F32: {
		const float *p = (const float *)t->data;
		return p[i];
	}
	case SAM3_DTYPE_F16: {
		const uint16_t *p = (const uint16_t *)t->data;
		return fp16_to_f32(p[i]);
	}
	case SAM3_DTYPE_BF16: {
		const uint16_t *p = (const uint16_t *)t->data;
		return bf16_to_f32(p[i]);
	}
	case SAM3_DTYPE_I32: {
		const int32_t *p = (const int32_t *)t->data;
		return (float)p[i];
	}
	case SAM3_DTYPE_I8: {
		const int8_t *p = (const int8_t *)t->data;
		return (float)p[i];
	}
	default:
		return 0.0f;
	}
}

/* ── Always-available compute functions ─────────────────────────────── */

void
sam3_trace_compute_stats(const struct sam3_tensor *t,
			 struct sam3_numeric_stats *out)
{
	int   n     = sam3_tensor_nelems(t);
	float mn    = 0.0f;
	float mx    = 0.0f;
	double sum  = 0.0;
	int   valid = 0;

	memset(out, 0, sizeof(*out));
	out->total_elems = n;

	for (int i = 0; i < n; i++) {
		float v = get_elem_f32(t, i);

		if (isnan(v)) {
			out->nan_count++;
			continue;
		}
		if (isinf(v)) {
			out->inf_count++;
			continue;
		}

		/*
		 * A denormal (subnormal) f32 has exponent bits all zero
		 * and a non-zero mantissa. fpclassify() is the portable test.
		 */
		if (fpclassify(v) == FP_SUBNORMAL)
			out->denormal_count++;

		if (valid == 0) {
			mn = v;
			mx = v;
		} else {
			if (v < mn)
				mn = v;
			if (v > mx)
				mx = v;
		}
		sum += (double)v;
		valid++;
	}

	out->min  = mn;
	out->max  = mx;
	out->mean = (valid > 0) ? (float)(sum / (double)valid) : 0.0f;
}

void
sam3_trace_compute_compare(const struct sam3_tensor *actual,
			   const struct sam3_tensor *reference,
			   float tolerance,
			   struct sam3_compare_result *out)
{
	int    n   = sam3_tensor_nelems(actual);
	int    ref_n = sam3_tensor_nelems(reference);
	double sum_abs = 0.0;

	memset(out, 0, sizeof(*out));

	if (n != ref_n)
		n = (n < ref_n) ? n : ref_n;

	for (int i = 0; i < n; i++) {
		float a   = get_elem_f32(actual, i);
		float r   = get_elem_f32(reference, i);
		float abs_err = fabsf(a - r);
		float denom   = fabsf(r);
		float rel_err = (denom > 1e-8f) ? (abs_err / denom) : abs_err;

		if (abs_err > out->max_abs_error)
			out->max_abs_error = abs_err;
		if (rel_err > out->max_rel_error)
			out->max_rel_error = rel_err;
		sum_abs += (double)abs_err;
		if (abs_err > tolerance)
			out->mismatches++;
	}

	out->mean_abs_error = (n > 0) ? (float)(sum_abs / (double)n) : 0.0f;
}

/* ── Logging functions (compiled only with SAM3_HAS_TRACE) ─────────── */

#ifdef SAM3_HAS_TRACE

/* Module-level runtime flags. */
static unsigned g_trace_flags = 0;

void
sam3_trace_set_flags(unsigned flags)
{
	g_trace_flags = flags;
}

unsigned
sam3_trace_get_flags(void)
{
	return g_trace_flags;
}

void
sam3_trace_kernel(const char *kernel_name, enum sam3_dtype in_dtype,
		  enum sam3_dtype out_dtype, const char *variant)
{
	if (!(g_trace_flags & SAM3_TRACE_KERNELS))
		return;

	sam3_log_debug("TRACE kernel=%s in=%s out=%s variant=%s",
		       kernel_name,
		       sam3_dtype_str(in_dtype),
		       sam3_dtype_str(out_dtype),
		       variant ? variant : "default");
}

void
sam3_trace_numeric(const char *label, const struct sam3_tensor *t)
{
	struct sam3_numeric_stats stats;

	if (!(g_trace_flags & SAM3_TRACE_NUMERIC))
		return;

	sam3_trace_compute_stats(t, &stats);
	sam3_log_debug(
		"TRACE numeric label=%s dtype=%s n=%d "
		"min=%g max=%g mean=%g nan=%d inf=%d denorm=%d",
		label,
		sam3_dtype_str(t->dtype),
		stats.total_elems,
		(double)stats.min,
		(double)stats.max,
		(double)stats.mean,
		stats.nan_count,
		stats.inf_count,
		stats.denormal_count);
}

void
sam3_trace_compare(const char *label, const struct sam3_tensor *actual,
		   const struct sam3_tensor *reference, float tolerance)
{
	struct sam3_compare_result cmp;

	if (!(g_trace_flags & SAM3_TRACE_COMPARE))
		return;

	sam3_trace_compute_compare(actual, reference, tolerance, &cmp);
	sam3_log_debug(
		"TRACE compare label=%s max_abs=%g max_rel=%g "
		"mean_abs=%g mismatches=%d",
		label,
		(double)cmp.max_abs_error,
		(double)cmp.max_rel_error,
		(double)cmp.mean_abs_error,
		cmp.mismatches);
}

void
sam3_trace_graph_plan(const struct sam3_graph *g)
{
	int limit;

	if (!(g_trace_flags & SAM3_TRACE_GRAPH))
		return;

	sam3_log_debug("TRACE graph_plan n_nodes=%d", g->n_nodes);

	limit = (g->n_nodes < 20) ? g->n_nodes : 20;
	for (int i = 0; i < limit; i++) {
		const struct sam3_node *nd = &g->nodes[i];
		const char *dtype_str = (nd->output != NULL)
			? sam3_dtype_str(nd->output->dtype)
			: "none";

		sam3_log_debug("TRACE   [%d] op=%s out_dtype=%s",
			       i, sam3_op_str(nd->op), dtype_str);
	}

	if (g->n_nodes > 20)
		sam3_log_debug("TRACE   ... (%d more nodes)",
			       g->n_nodes - 20);
}

void
sam3_trace_graph_done(const struct sam3_graph *g, double elapsed_ms)
{
	if (!(g_trace_flags & SAM3_TRACE_GRAPH))
		return;

	sam3_log_debug("TRACE graph_done n_nodes=%d elapsed_ms=%.3f",
		       g->n_nodes, elapsed_ms);
}

#endif /* SAM3_HAS_TRACE */
