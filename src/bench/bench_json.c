/*
 * src/bench/bench_json.c - JSON serialization for benchmark results
 *
 * Implements versioned JSON I/O for benchmark results and environment
 * metadata using vendored cJSON. The format includes an env section for
 * hardware info, a config section for run parameters, and an array of
 * result objects with timing and throughput data. Also provides a tabular
 * printer for human-readable stderr output.
 *
 * Key types:  sam3_bench_result, sam3_bench_env
 * Depends on: bench/bench_json.h, core/json/cJSON.h, util/log.h
 * Used by:    bench_compare.c, cli_bench.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench/bench_json.h"
#include "core/json/cJSON.h"
#include "util/log.h"

/* JSON format version. Bump when schema changes. */
#define BENCH_JSON_VERSION 1

/* Read entire file into a malloc'd buffer. Caller frees. */
static char *read_file(const char *path, long *out_len)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return NULL;

	fseek(f, 0, SEEK_END);
	long len = ftell(f);
	fseek(f, 0, SEEK_SET);

	if (len <= 0) {
		fclose(f);
		return NULL;
	}

	char *buf = malloc((size_t)len + 1);
	if (!buf) {
		fclose(f);
		return NULL;
	}

	size_t nread = fread(buf, 1, (size_t)len, f);
	fclose(f);

	if ((long)nread != len) {
		free(buf);
		return NULL;
	}

	buf[len] = '\0';
	if (out_len)
		*out_len = len;
	return buf;
}

/* Helper to add a string field to a cJSON object. */
static void json_add_str(cJSON *obj, const char *key, const char *val)
{
	cJSON_AddStringToObject(obj, key, val);
}

/* Helper to add a number field to a cJSON object. */
static void json_add_num(cJSON *obj, const char *key, double val)
{
	cJSON_AddNumberToObject(obj, key, val);
}

/* Helper to add an int field to a cJSON object. */
static void json_add_int(cJSON *obj, const char *key, int val)
{
	cJSON_AddNumberToObject(obj, key, (double)val);
}

/* Build a cJSON object from a bench_env struct. */
static cJSON *env_to_json(const struct sam3_bench_env *env)
{
	cJSON *obj = cJSON_CreateObject();
	if (!obj)
		return NULL;

	json_add_str(obj, "chip", env->chip);
	json_add_str(obj, "os", env->os);
	json_add_int(obj, "cpu_cores", env->cpu_cores);
	json_add_int(obj, "gpu_cores", env->gpu_cores);
	json_add_str(obj, "backend", env->backend);
	json_add_str(obj, "commit", env->commit);
	json_add_str(obj, "timestamp", env->timestamp);
	json_add_str(obj, "model_variant", env->model_variant);

	return obj;
}

/* Build a cJSON object from a bench_config struct. */
static cJSON *config_to_json(const struct sam3_bench_config *cfg)
{
	cJSON *obj = cJSON_CreateObject();
	if (!obj)
		return NULL;

	json_add_int(obj, "warmup_iters", cfg->warmup_iters);
	json_add_int(obj, "timed_iters", cfg->timed_iters);
	cJSON_AddBoolToObject(obj, "statistical", cfg->statistical);
	json_add_num(obj, "threshold_pct", cfg->threshold_pct);

	return obj;
}

/* Build a cJSON object from a single bench_result. */
static cJSON *result_to_json(const struct sam3_bench_result *r)
{
	cJSON *obj = cJSON_CreateObject();
	if (!obj)
		return NULL;

	json_add_str(obj, "name", r->name);
	json_add_str(obj, "suite", r->suite);
	json_add_num(obj, "mean_ms", r->mean_ms);
	json_add_num(obj, "min_ms", r->min_ms);
	json_add_num(obj, "max_ms", r->max_ms);
	json_add_num(obj, "stddev_ms", r->stddev_ms);
	json_add_num(obj, "gflops", r->gflops);
	json_add_num(obj, "throughput_mbs", r->throughput_mbs);
	json_add_int(obj, "iterations", r->iterations);

	return obj;
}

/* Parse a cJSON object into a bench_env struct. */
static int json_to_env(const cJSON *obj, struct sam3_bench_env *env)
{
	if (!obj || !env)
		return -1;

	memset(env, 0, sizeof(*env));

	cJSON *v;

	v = cJSON_GetObjectItemCaseSensitive(obj, "chip");
	if (cJSON_IsString(v))
		snprintf(env->chip, sizeof(env->chip), "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "os");
	if (cJSON_IsString(v))
		snprintf(env->os, sizeof(env->os), "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "cpu_cores");
	if (cJSON_IsNumber(v))
		env->cpu_cores = (int)v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "gpu_cores");
	if (cJSON_IsNumber(v))
		env->gpu_cores = (int)v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "backend");
	if (cJSON_IsString(v))
		snprintf(env->backend, sizeof(env->backend),
			 "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "commit");
	if (cJSON_IsString(v))
		snprintf(env->commit, sizeof(env->commit),
			 "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "timestamp");
	if (cJSON_IsString(v))
		snprintf(env->timestamp, sizeof(env->timestamp),
			 "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "model_variant");
	if (cJSON_IsString(v))
		snprintf(env->model_variant, sizeof(env->model_variant),
			 "%s", v->valuestring);

	return 0;
}

/* Parse a cJSON object into a bench_result struct. */
static int json_to_result(const cJSON *obj, struct sam3_bench_result *r)
{
	if (!obj || !r)
		return -1;

	memset(r, 0, sizeof(*r));

	cJSON *v;

	v = cJSON_GetObjectItemCaseSensitive(obj, "name");
	if (cJSON_IsString(v))
		snprintf(r->name, sizeof(r->name), "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "suite");
	if (cJSON_IsString(v))
		snprintf(r->suite, sizeof(r->suite), "%s", v->valuestring);

	v = cJSON_GetObjectItemCaseSensitive(obj, "mean_ms");
	if (cJSON_IsNumber(v))
		r->mean_ms = v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "min_ms");
	if (cJSON_IsNumber(v))
		r->min_ms = v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "max_ms");
	if (cJSON_IsNumber(v))
		r->max_ms = v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "stddev_ms");
	if (cJSON_IsNumber(v))
		r->stddev_ms = v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "gflops");
	if (cJSON_IsNumber(v))
		r->gflops = v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "throughput_mbs");
	if (cJSON_IsNumber(v))
		r->throughput_mbs = v->valuedouble;

	v = cJSON_GetObjectItemCaseSensitive(obj, "iterations");
	if (cJSON_IsNumber(v))
		r->iterations = (int)v->valuedouble;

	return 0;
}

int sam3_bench_write_json(const char *path,
			  const struct sam3_bench_env *env,
			  const struct sam3_bench_config *cfg,
			  const struct sam3_bench_result *results,
			  int n_results)
{
	if (!path || !env || !cfg || !results || n_results < 0) {
		sam3_log_error("bench_write_json: invalid arguments");
		return -1;
	}

	cJSON *root = NULL;
	cJSON *env_obj = NULL;
	cJSON *cfg_obj = NULL;
	cJSON *arr = NULL;
	char *str = NULL;
	int ret = -1;

	root = cJSON_CreateObject();
	if (!root) {
		sam3_log_error("bench_write_json: failed to create root");
		goto cleanup;
	}

	json_add_int(root, "version", BENCH_JSON_VERSION);

	env_obj = env_to_json(env);
	if (!env_obj) {
		sam3_log_error("bench_write_json: failed to create env");
		goto cleanup;
	}
	cJSON_AddItemToObject(root, "env", env_obj);
	env_obj = NULL; /* ownership transferred */

	cfg_obj = config_to_json(cfg);
	if (!cfg_obj) {
		sam3_log_error("bench_write_json: failed to create config");
		goto cleanup;
	}
	cJSON_AddItemToObject(root, "config", cfg_obj);
	cfg_obj = NULL;

	arr = cJSON_CreateArray();
	if (!arr) {
		sam3_log_error("bench_write_json: failed to create array");
		goto cleanup;
	}

	for (int i = 0; i < n_results; i++) {
		cJSON *item = result_to_json(&results[i]);
		if (!item) {
			sam3_log_error("bench_write_json: failed at result %d",
				       i);
			goto cleanup;
		}
		cJSON_AddItemToArray(arr, item);
	}
	cJSON_AddItemToObject(root, "results", arr);
	arr = NULL;

	str = cJSON_Print(root);
	if (!str) {
		sam3_log_error("bench_write_json: failed to serialize");
		goto cleanup;
	}

	FILE *f = fopen(path, "w");
	if (!f) {
		sam3_log_error("bench_write_json: cannot open %s", path);
		goto cleanup;
	}

	size_t len = strlen(str);
	size_t written = fwrite(str, 1, len, f);
	fclose(f);

	if (written != len) {
		sam3_log_error("bench_write_json: short write to %s", path);
		goto cleanup;
	}

	ret = 0;

cleanup:
	free(str);
	cJSON_Delete(env_obj);
	cJSON_Delete(cfg_obj);
	cJSON_Delete(arr);
	cJSON_Delete(root);
	return ret;
}

int sam3_bench_read_json(const char *path,
			 struct sam3_bench_env *env,
			 struct sam3_bench_result *results,
			 int max_results, int *n_results)
{
	if (!path || !results || !n_results || max_results <= 0) {
		sam3_log_error("bench_read_json: invalid arguments");
		return -1;
	}

	*n_results = 0;

	long file_len = 0;
	char *buf = read_file(path, &file_len);
	if (!buf) {
		sam3_log_error("bench_read_json: cannot read %s", path);
		return -1;
	}

	cJSON *root = cJSON_Parse(buf);
	free(buf);

	if (!root) {
		sam3_log_error("bench_read_json: parse error in %s", path);
		return -1;
	}

	int ret = -1;

	/* Check version. */
	cJSON *ver = cJSON_GetObjectItemCaseSensitive(root, "version");
	if (!cJSON_IsNumber(ver) || (int)ver->valuedouble != BENCH_JSON_VERSION) {
		sam3_log_error("bench_read_json: unsupported version in %s",
			       path);
		goto cleanup;
	}

	/* Parse environment if requested. */
	if (env) {
		cJSON *env_obj = cJSON_GetObjectItemCaseSensitive(root, "env");
		if (env_obj) {
			if (json_to_env(env_obj, env) != 0) {
				sam3_log_error("bench_read_json: bad env in %s",
					       path);
				goto cleanup;
			}
		}
	}

	/* Parse results array. */
	cJSON *arr = cJSON_GetObjectItemCaseSensitive(root, "results");
	if (!cJSON_IsArray(arr)) {
		sam3_log_error("bench_read_json: no results array in %s", path);
		goto cleanup;
	}

	int count = 0;
	cJSON *item;
	cJSON_ArrayForEach(item, arr) {
		if (count >= max_results) {
			sam3_log_warn("bench_read_json: truncated at %d results",
				      max_results);
			break;
		}
		if (json_to_result(item, &results[count]) != 0) {
			sam3_log_error("bench_read_json: bad result %d in %s",
				       count, path);
			goto cleanup;
		}
		count++;
	}

	*n_results = count;
	ret = 0;

cleanup:
	cJSON_Delete(root);
	return ret;
}

void sam3_bench_print_results(const struct sam3_bench_result *results,
			      int n_results)
{
	if (!results || n_results <= 0)
		return;

	fprintf(stderr,
		"%-40s %10s %10s %10s %8s %10s\n",
		"Name", "Mean(ms)", "Min(ms)", "Max(ms)",
		"GFLOPS", "MB/s");
	fprintf(stderr,
		"%-40s %10s %10s %10s %8s %10s\n",
		"----------------------------------------",
		"----------", "----------", "----------",
		"--------", "----------");

	for (int i = 0; i < n_results; i++) {
		const struct sam3_bench_result *r = &results[i];
		fprintf(stderr,
			"%-40s %10.3f %10.3f %10.3f %8.2f %10.1f\n",
			r->name, r->mean_ms, r->min_ms, r->max_ms,
			r->gflops, r->throughput_mbs);
	}
}
