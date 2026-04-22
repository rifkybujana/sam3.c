/*
 * src/model/feature_cache_persist.h - Binary persistence of feature
 * cache entries.
 *
 * Serializes image / text feature bundles to a single .sam3cache file
 * so an app can save the result of an expensive encoder run across
 * process restarts. Each file holds exactly one entry keyed by the
 * original FNV-1a 64-bit content hash; loading re-registers the
 * bundle in the in-memory cache. A model signature (image_size,
 * dims, backbone variant) is embedded so files from a different model
 * are rejected rather than producing garbage features.
 *
 * Key types:  sam3_cache_persist_sig
 * Depends on: core/tensor.h, core/alloc.h, model/feature_cache.h,
 *             sam3/sam3_types.h
 * Used by:    src/sam3.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_FEATURE_CACHE_PERSIST_H
#define SAM3_MODEL_FEATURE_CACHE_PERSIST_H

#include <stdint.h>

#include "sam3/sam3_types.h"
#include "feature_cache.h"

/*
 * sam3_cache_persist_sig - Model-identity fingerprint embedded in every
 * saved cache file. Loads reject files whose signature does not match
 * the currently-loaded model.
 */
struct sam3_cache_persist_sig {
	int32_t image_size;
	int32_t encoder_dim;
	int32_t decoder_dim;
	int32_t backbone_type;
	int32_t variant;
	int32_t n_fpn_scales;
	int32_t text_backbone;
	int32_t reserved;
};

/*
 * sam3_image_bundle_save - Write an image bundle to a .sam3cache file.
 *
 * @path:    Destination path.
 * @sig:     Model signature to embed.
 * @hash:    Content hash key.
 * @prefix:  Verification prefix (may be NULL when prefix_len == 0).
 * @prefix_len: Bytes of @prefix to write (0..SAM3_CACHE_PREFIX_BYTES).
 * @bundle:  Bundle to serialize. NULL tensor fields are written as
 *           "not present" and restored as NULL on load.
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_image_bundle_save(const char *path,
	const struct sam3_cache_persist_sig *sig,
	uint64_t hash, const uint8_t *prefix, size_t prefix_len,
	const struct sam3_image_bundle *bundle);

/*
 * sam3_image_bundle_load - Read an image bundle from a .sam3cache file
 * into @arena and return it via @out_bundle. @out_hash, @out_prefix,
 * @out_prefix_len receive the stored key material for re-registration.
 *
 * @expected_sig: Signature of the currently-loaded model. The load
 *                fails with SAM3_EMODEL if the file's signature differs.
 * @out_prefix:   Caller-provided buffer (SAM3_CACHE_PREFIX_BYTES).
 *
 * Returns SAM3_OK on success.
 */
enum sam3_error sam3_image_bundle_load(const char *path,
	const struct sam3_cache_persist_sig *expected_sig,
	struct sam3_arena *arena,
	uint64_t *out_hash,
	uint8_t *out_prefix, size_t *out_prefix_len,
	struct sam3_image_bundle *out_bundle);

/* --- text --- */

enum sam3_error sam3_text_bundle_save(const char *path,
	const struct sam3_cache_persist_sig *sig,
	uint64_t hash,
	const int32_t *prefix_tokens, int prefix_len,
	const struct sam3_text_bundle *bundle);

enum sam3_error sam3_text_bundle_load(const char *path,
	const struct sam3_cache_persist_sig *expected_sig,
	struct sam3_arena *arena,
	uint64_t *out_hash,
	int32_t *out_prefix_tokens, int *out_prefix_len,
	struct sam3_text_bundle *out_bundle);

/* --- disk-backed spill helpers --- */

/*
 * sam3_image_bundle_write_uncompressed - Serialize a bundle to @path
 * without compression. Used as the on-disk format for in-process
 * tiered spill: writes fast (SSD-bandwidth limited) at the cost of a
 * larger file than the .sam3cache compressed format. No magic or
 * signature is written — the file is only valid for the lifetime of
 * the originating process.
 */
enum sam3_error sam3_image_bundle_write_uncompressed(
	const char *path,
	const struct sam3_image_bundle *bundle);

/*
 * sam3_image_bundle_read_uncompressed - Inverse of the above.
 * Restores a bundle from @path into @arena.
 */
enum sam3_error sam3_image_bundle_read_uncompressed(
	const char *path,
	struct sam3_arena *arena,
	struct sam3_image_bundle *out_bundle);

#endif /* SAM3_MODEL_FEATURE_CACHE_PERSIST_H */
