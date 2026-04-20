/*
 * tools/weight_conv_perm.c - Conv weight OHWI permute reader wrapper
 *
 * Implements a weight_reader wrapper that detects conv2d and
 * conv_transpose2d weight tensors by name allowlist and transposes
 * them from OIHW / IOHW to OHWI [OC, KH, KW, IC] before they reach
 * the writer. Sits between the rename reader and the quant reader in
 * the conversion pipeline:
 *
 *   safetensors -> rename -> conv_perm -> quant -> writer
 *
 * Previously the permute happened at load time in the model code
 * (necks.c / segmentation.c / mask_decoder.c / image_encoder.c).
 * Moving it into the converter lets those load paths consume OHWI
 * weights directly and lets Task 13 remove the NCHW fallback.
 *
 * Key types:  conv_perm_impl
 * Depends on: weight_conv_perm.h, core/weight.h, core/tensor.h,
 *             util/log.h
 * Used by:    sam3_convert.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "weight_conv_perm.h"
#include "core/tensor.h"
#include "util/log.h"

/* --- Implementation state ─────────────── --- */

struct conv_perm_impl {
	struct weight_reader *inner;
};

/* Singleton — only one conv_perm reader exists at a time in the
 * converter, matching the pattern used by weight_rename.c. */
static struct conv_perm_impl cp_state;

/* --- Name classification ──────────────── --- */

static int str_ends_with(const char *s, const char *suffix)
{
	size_t ls = strlen(s);
	size_t lt = strlen(suffix);
	if (lt > ls)
		return 0;
	return memcmp(s + ls - lt, suffix, lt) == 0;
}

/*
 * is_conv2d_weight - True if @name is a forward Conv2d weight that
 *                    must be permuted from OIHW to OHWI.
 *
 * The allowlist matches the exact renamed tensor names that the
 * former load-time permute helpers in necks.c, segmentation.c,
 * mask_decoder.c and image_encoder.c used to rewrite.
 */
static int is_conv2d_weight(const char *name)
{
	/* ViT patch embed:
	 * detector_model.vision_encoder.backbone.embeddings
	 *     .patch_embeddings.projection.weight */
	if (str_ends_with(name,
		".embeddings.patch_embeddings.projection.weight"))
		return 1;

	/* FPN neck Conv2d layers: .neck.fpn_layers.{i}.proj1/proj2.weight
	 * (scale_layers.* are ConvTranspose2d — handled separately).
	 * Also match the second neck (.neck.sam2_fpn_layers.) used by the
	 * video tracker's SAM mask decoder path. */
	if ((strstr(name, ".neck.fpn_layers.") ||
	     strstr(name, ".neck.sam2_fpn_layers.")) &&
	    (str_ends_with(name, ".proj1.weight") ||
	     str_ends_with(name, ".proj2.weight")))
		return 1;

	/* Seg head FPN pixel decoder Conv2d:
	 * detector_model.mask_decoder.pixel_decoder.conv_layers.{i}.weight */
	if (strstr(name,
		   "detector_model.mask_decoder.pixel_decoder.conv_layers.") &&
	    str_ends_with(name, ".weight"))
		return 1;

	/* Seg head instance projection 1x1 Conv2d:
	 * detector_model.mask_decoder.instance_projection.weight */
	if (str_ends_with(name,
		".mask_decoder.instance_projection.weight"))
		return 1;

	/* Mask decoder multi-scale skip Conv2d:
	 * tracker_model.mask_decoder.conv_s0/conv_s1.weight  (SAM 3)
	 * tracker_multiplex.sam_mask_decoder.conv_s0/conv_s1.weight (SAM 3.1)
	 * Both are 1x1 so the permute is a no-op byte-wise, but routing
	 * them through the allowlist is harmless and keeps the shape
	 * declarations explicit. */
	if (str_ends_with(name, ".mask_decoder.conv_s0.weight") ||
	    str_ends_with(name, ".mask_decoder.conv_s1.weight") ||
	    str_ends_with(name, ".sam_mask_decoder.conv_s0.weight") ||
	    str_ends_with(name, ".sam_mask_decoder.conv_s1.weight"))
		return 1;

	/* Memory encoder (maskmem_backbone): 4D conv weights used by the
	 * memory encoder's mask_downsampler encoder chain, fuser CXBlocks
	 * (depthwise), final out_proj and pix_feat_proj. Names use the raw
	 * PyTorch suffix ".weight" and originate from
	 *   tracker.maskmem_backbone.mask_downsampler.encoder.{0,3,6,9,12}.weight
	 *   tracker.maskmem_backbone.fuser.layers.{0,1}.dwconv.weight
	 *   tracker.maskmem_backbone.out_proj.weight
	 *   tracker.maskmem_backbone.pix_feat_proj.weight
	 *
	 * After the weight_rename pass the prefix is `tracker_model.` so
	 * we match on the maskmem_backbone substring. */
	if (strstr(name, ".maskmem_backbone.") &&
	    (str_ends_with(name, ".encoder.0.weight") ||
	     str_ends_with(name, ".encoder.3.weight") ||
	     str_ends_with(name, ".encoder.6.weight") ||
	     str_ends_with(name, ".encoder.9.weight") ||
	     str_ends_with(name, ".encoder.12.weight") ||
	     str_ends_with(name, ".dwconv.weight") ||
	     str_ends_with(name, ".out_proj.weight") ||
	     str_ends_with(name, ".pix_feat_proj.weight")))
		return 1;

	/*
	 * EfficientViT backbone Conv2d weights:
	 *   input_stem.*.conv.weight  (stem conv + DSConv blocks)
	 *   stages.*.conv.weight      (MBConv + LiteMLA proj via ConvLayer)
	 *   context.aggreg.0.{0,1}.weight  (LiteMLA DW/PW raw Conv2d)
	 *   context.qkv.conv.weight   (LiteMLA QKV)
	 *   projection.conv1/2.weight (projection head raw Conv2d)
	 *
	 * TinyViT backbone Conv2d weights (Conv2d_BN naming):
	 *   patch_embed.seq.{0,2}.c.weight  (stem convs)
	 *   layers.*.blocks.*.conv{1,2,3}.c.weight  (MBConv + local_conv)
	 *   layers.*.downsample.conv{1,2,3}.c.weight  (PatchMerging)
	 *   projection.conv{1,2}.c.weight  (projection head)
	 */
	if (strstr(name, ".vision_encoder.backbone.") &&
	    (str_ends_with(name, ".conv.weight") ||
	     str_ends_with(name, ".c.weight") ||
	     str_ends_with(name, ".aggreg.0.0.weight") ||
	     str_ends_with(name, ".aggreg.0.1.weight") ||
	     str_ends_with(name, ".projection.conv1.weight") ||
	     str_ends_with(name, ".projection.conv2.weight")))
		return 1;

	return 0;
}

/*
 * is_conv_transpose_weight - True if @name is a ConvTranspose2d
 *                            weight whose checkpoint layout is IOHW.
 *
 * The source layout is [IC, OC, KH, KW] and the destination OHWI
 * layout is [OC, KH, KW, IC], so the permutation is (1, 2, 3, 0) on
 * the source axes, not (0, 2, 3, 1).
 */
static int is_conv_transpose_weight(const char *name)
{
	/* FPN neck scale_layers ConvTranspose2d:
	 * detector_model.vision_encoder.neck.fpn_layers.{i}
	 *     .scale_layers.{j}.weight
	 * Also match the sam2 second neck (.neck.sam2_fpn_layers.). */
	if ((strstr(name, ".neck.fpn_layers.") ||
	     strstr(name, ".neck.sam2_fpn_layers.")) &&
	    strstr(name, ".scale_layers.") &&
	    str_ends_with(name, ".weight"))
		return 1;

	/* Mask decoder pixel upscaling ConvTranspose2d:
	 * tracker_model.mask_decoder.upscale_conv1/2.weight  (SAM 3)
	 * tracker_multiplex.sam_mask_decoder.output_upscaling.{0,3}.weight
	 *   (SAM 3.1 — 2-stage ConvTranspose 256->64, 64->32). */
	if (str_ends_with(name, ".mask_decoder.upscale_conv1.weight") ||
	    str_ends_with(name, ".mask_decoder.upscale_conv2.weight") ||
	    str_ends_with(name, ".sam_mask_decoder.output_upscaling.0.weight") ||
	    str_ends_with(name, ".sam_mask_decoder.output_upscaling.3.weight"))
		return 1;

	return 0;
}

/* --- Permute kernel --- */

/*
 * permute_ohwi - Copy-permute a 4-D conv weight into OHWI.
 *
 * @src:          Source buffer (contiguous OIHW or IOHW depending on
 *                @is_transpose)
 * @dst:          Destination buffer, must be sized for the same element
 *                count
 * @oc, @ic, @kh, @kw: OHWI shape dimensions
 * @esz:          Element size in bytes (from sam3_dtype_size)
 * @is_transpose: 0 for Conv2d (src = OIHW), 1 for ConvTranspose2d
 *                (src = IOHW)
 *
 * Walks the OHWI destination linearly; computes the matching source
 * offset per element. Byte-generic so F32/F16/BF16/I32/I8 all work.
 */
static void permute_ohwi(const void *src, void *dst,
			 int oc, int ic, int kh, int kw,
			 size_t esz, int is_transpose)
{
	const char *sbytes = (const char *)src;
	char *dbytes = (char *)dst;

	for (int o = 0; o < oc; o++) {
		for (int y = 0; y < kh; y++) {
			for (int x = 0; x < kw; x++) {
				for (int c = 0; c < ic; c++) {
					size_t s;
					if (is_transpose) {
						s = (((size_t)c * oc + o)
						     * kh + y) * kw + x;
					} else {
						s = (((size_t)o * ic + c)
						     * kh + y) * kw + x;
					}
					size_t d = (((size_t)o * kh + y)
						    * kw + x) * ic + c;
					memcpy(dbytes + d * esz,
					       sbytes + s * esz,
					       esz);
				}
			}
		}
	}
}

/* --- Vtable callbacks ─ --- */

static enum sam3_error cp_open(struct weight_reader *r, const char *path)
{
	(void)r; (void)path;
	return SAM3_OK; /* inner already opened */
}

static int cp_n_tensors(struct weight_reader *r)
{
	struct conv_perm_impl *s = r->impl;
	return s->inner->ops->n_tensors(s->inner);
}

static enum sam3_error cp_get_tensor_info(struct weight_reader *r, int idx,
					   struct weight_tensor_info *info)
{
	struct conv_perm_impl *s = r->impl;
	enum sam3_error err;

	err = s->inner->ops->get_tensor_info(s->inner, idx, info);
	if (err != SAM3_OK)
		return err;

	if (info->n_dims != 4)
		return SAM3_OK;

	int conv = is_conv2d_weight(info->name);
	int trans = conv ? 0 : is_conv_transpose_weight(info->name);
	if (!conv && !trans)
		return SAM3_OK;

	int oc, ic;
	if (trans) {
		oc = info->dims[1];
		ic = info->dims[0];
	} else {
		oc = info->dims[0];
		ic = info->dims[1];
	}
	int kh = info->dims[2];
	int kw = info->dims[3];

	int old0 = info->dims[0];
	int old1 = info->dims[1];
	int old2 = info->dims[2];
	int old3 = info->dims[3];

	info->dims[0] = oc;
	info->dims[1] = kh;
	info->dims[2] = kw;
	info->dims[3] = ic;

	sam3_log_info("conv_perm: %s [%d,%d,%d,%d] -> OHWI [%d,%d,%d,%d]",
		      info->name, old0, old1, old2, old3,
		      info->dims[0], info->dims[1],
		      info->dims[2], info->dims[3]);

	return SAM3_OK;
}

static enum sam3_error cp_read_tensor_data(struct weight_reader *r, int idx,
					    void *dst, size_t dst_size)
{
	struct conv_perm_impl *s = r->impl;
	struct weight_tensor_info inner_info;
	enum sam3_error err;

	err = s->inner->ops->get_tensor_info(s->inner, idx, &inner_info);
	if (err != SAM3_OK)
		return err;

	/* Non-conv tensors pass through untouched. */
	if (inner_info.n_dims != 4)
		return s->inner->ops->read_tensor_data(s->inner, idx,
						       dst, dst_size);

	int conv = is_conv2d_weight(inner_info.name);
	int trans = conv ? 0 : is_conv_transpose_weight(inner_info.name);
	if (!conv && !trans)
		return s->inner->ops->read_tensor_data(s->inner, idx,
						       dst, dst_size);

	if (inner_info.dtype == SAM3_DTYPE_Q8_0) {
		sam3_log_error("conv_perm: refusing to permute quantized "
			       "conv weight '%s' (Q8_0 blocks cross "
			       "channels)", inner_info.name);
		return SAM3_EINVAL;
	}

	size_t esz = sam3_dtype_size(inner_info.dtype);
	if (esz == 0) {
		sam3_log_error("conv_perm: unsupported dtype %d for '%s'",
			       inner_info.dtype, inner_info.name);
		return SAM3_EINVAL;
	}

	int oc, ic;
	if (trans) {
		oc = inner_info.dims[1];
		ic = inner_info.dims[0];
	} else {
		oc = inner_info.dims[0];
		ic = inner_info.dims[1];
	}
	int kh = inner_info.dims[2];
	int kw = inner_info.dims[3];

	size_t nbytes = inner_info.nbytes;
	if (dst_size < nbytes) {
		sam3_log_error("conv_perm: dst too small for '%s' "
			       "(%zu < %zu)",
			       inner_info.name, dst_size, nbytes);
		return SAM3_EINVAL;
	}

	void *tmp = malloc(nbytes);
	if (!tmp) {
		sam3_log_error("conv_perm: malloc %zu bytes failed for '%s'",
			       nbytes, inner_info.name);
		return SAM3_ENOMEM;
	}

	err = s->inner->ops->read_tensor_data(s->inner, idx, tmp, nbytes);
	if (err != SAM3_OK) {
		free(tmp);
		return err;
	}

	permute_ohwi(tmp, dst, oc, ic, kh, kw, esz, trans);

	free(tmp);
	return SAM3_OK;
}

static void cp_close(struct weight_reader *r)
{
	(void)r; /* inner closed separately; no dynamic state to release */
}

static const struct weight_reader_ops conv_perm_reader_ops = {
	.open             = cp_open,
	.n_tensors        = cp_n_tensors,
	.get_tensor_info  = cp_get_tensor_info,
	.read_tensor_data = cp_read_tensor_data,
	.close            = cp_close,
};

/* --- Public init ─── --- */

enum sam3_error weight_reader_conv_perm_init(struct weight_reader *r,
					      struct weight_reader *inner)
{
	if (!r || !inner)
		return SAM3_EINVAL;

	memset(&cp_state, 0, sizeof(cp_state));
	cp_state.inner = inner;

	r->ops  = &conv_perm_reader_ops;
	r->impl = &cp_state;

	return SAM3_OK;
}
