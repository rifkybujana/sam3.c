/*
 * tools/weight_rename.c - Weight name remapping reader wrapper
 *
 * Implements a weight_reader wrapper that remaps PyTorch .pt checkpoint
 * key names (facebook/sam3) to the C model's expected weight names.
 * Sits between the checkpoint reader and the quant reader in the pipeline:
 *
 *   PT Reader -> Rename Reader -> Quant Reader -> Writer
 *
 * Two-stage remapping: (1) prefix replacement maps PT module hierarchy
 * to C prefixes, (2) per-subsystem attribute renaming handles layer
 * numbering, norm names, MLP renaming, and QKV splitting.
 *
 * For fused QKV tensors (attn.in_proj_weight in PT), the rename reader
 * creates 2-3 output entries from 1 input (Q+K+V or Q+KV), adjusting
 * dims and slicing data on read.
 *
 * Key types:  rename_entry, rename_reader_state
 * Depends on: core/weight.h, util/log.h
 * Used by:    sam3_convert.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "weight_rename.h"
#include "util/log.h"

/* ── Data structures ───────────────────────────────────────────────── */

struct rename_entry {
	int  inner_idx;
	char name[SAM3_WEIGHT_NAME_MAX];
	int  split_total;  /* 0=no split, 3=thirds */
	int  split_start;  /* starting slice index */
	int  split_count;  /* number of slices to take */
};

struct rename_reader_state {
	struct weight_reader *inner;
	struct rename_entry  *entries;
	int                   n_entries;
};

/* ── String helpers ────────────────────────────────────────────────── */

/*
 * strip_prefix - Check if str starts with prefix.
 *
 * Returns pointer past prefix if matched, NULL otherwise.
 */
static const char *strip_prefix(const char *str, const char *prefix)
{
	size_t len = strlen(prefix);
	if (strncmp(str, prefix, len) == 0)
		return str + len;
	return NULL;
}

/* ── Entry helpers ─────────────────────────────────────────────────── */

static int add_entry(struct rename_entry *out, int inner_idx,
		     const char *name)
{
	out->inner_idx = inner_idx;
	snprintf(out->name, SAM3_WEIGHT_NAME_MAX, "%s", name);
	out->split_total = 0;
	out->split_start = 0;
	out->split_count = 0;
	return 1;
}

static const char *qkv_infix[] = {"q_proj.", "k_proj.", "v_proj."};

static int add_qkv_split(struct rename_entry *out, int inner_idx,
			  const char *prefix, const char *suffix)
{
	for (int p = 0; p < 3; p++) {
		out[p].inner_idx = inner_idx;
		snprintf(out[p].name, SAM3_WEIGHT_NAME_MAX,
			 "%s%s%s", prefix, qkv_infix[p], suffix);
		out[p].split_total = 3;
		out[p].split_start = p;
		out[p].split_count = 1;
	}
	return 3;
}

static int add_q_kv_split(struct rename_entry *out, int inner_idx,
			  const char *prefix, const char *q_name,
			  const char *kv_name, const char *suffix)
{
	/* Q: first third */
	out[0].inner_idx = inner_idx;
	snprintf(out[0].name, SAM3_WEIGHT_NAME_MAX,
		 "%s%s%s", prefix, q_name, suffix);
	out[0].split_total = 3;
	out[0].split_start = 0;
	out[0].split_count = 1;

	/* KV: last two thirds */
	out[1].inner_idx = inner_idx;
	snprintf(out[1].name, SAM3_WEIGHT_NAME_MAX,
		 "%s%s%s", prefix, kv_name, suffix);
	out[1].split_total = 3;
	out[1].split_start = 1;
	out[1].split_count = 2;

	return 2;
}


/* ── Per-subsystem handlers (PyTorch .pt format) ──────────────────── */

/*
 * Each handler takes the remainder after prefix stripping and
 * produces 1 or 3 (QKV split) rename entries. Returns count added.
 */

/*
 * EfficientViT backbone: detector.backbone.vision_backbone.trunk.model.*
 * -> detector_model.vision_encoder.backbone.*
 *
 * Maps EfficientViT-B2 module hierarchy (input_stem, stages with MBConv
 * and LiteMLA attention blocks) and projection head to C model names.
 * BN running stats (running_mean, running_var, num_batches_tracked) are
 * passed through since EfficientViT uses BatchNorm at inference.
 */
static int handle_efficientvit(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "detector_model.vision_encoder.backbone.";
	const char *r, *s;

	/* === Backbone trunk: backbone.model.* === */
	r = strip_prefix(rest, "backbone.model.");
	if (r) {
		/* input_stem.op_list.{i}.* -> input_stem.{i}.* */
		s = strip_prefix(r, "input_stem.op_list.");
		if (s) {
			snprintf(buf, sizeof(buf),
				 "%sinput_stem.%s", prefix, s);
			return add_entry(out, inner_idx, buf);
		}

		/* stages.{s}.op_list.{i}.* with sub-rules */
		s = strip_prefix(r, "stages.");
		if (s) {
			int stage, consumed;
			if (sscanf(s, "%d.%n", &stage, &consumed) < 1)
				goto passthrough;
			const char *after_stage = s + consumed;
			const char *after_oplist =
				strip_prefix(after_stage, "op_list.");
			if (!after_oplist)
				goto passthrough;

			int blk, blk_consumed;
			if (sscanf(after_oplist, "%d.%n",
				   &blk, &blk_consumed) < 1)
				goto passthrough;
			const char *attr = after_oplist + blk_consumed;

			/* context_module.main.* -> context.* */
			const char *cm;
			cm = strip_prefix(attr, "context_module.main.");
			if (cm) {
				snprintf(buf, sizeof(buf),
					 "%sstages.%d.blocks.%d."
					 "context.%s",
					 prefix, stage, blk, cm);
				return add_entry(out, inner_idx, buf);
			}

			/* local_module.main.* -> local.* */
			cm = strip_prefix(attr, "local_module.main.");
			if (cm) {
				snprintf(buf, sizeof(buf),
					 "%sstages.%d.blocks.%d."
					 "local.%s",
					 prefix, stage, blk, cm);
				return add_entry(out, inner_idx, buf);
			}

			/* main.* -> strip "main." (MBConv blocks) */
			cm = strip_prefix(attr, "main.");
			if (cm) {
				snprintf(buf, sizeof(buf),
					 "%sstages.%d.blocks.%d.%s",
					 prefix, stage, blk, cm);
				return add_entry(out, inner_idx, buf);
			}

			/* Passthrough within block */
			snprintf(buf, sizeof(buf),
				 "%sstages.%d.blocks.%d.%s",
				 prefix, stage, blk, attr);
			return add_entry(out, inner_idx, buf);
		}

		/* Other backbone.model.* tensors: passthrough */
		snprintf(buf, sizeof(buf), "%s%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* === Projection head: head.* === */
	r = strip_prefix(rest, "head.");
	if (r) {
		/* head.0.* -> projection.conv1.* (1x1 conv) */
		s = strip_prefix(r, "0.");
		if (s) {
			snprintf(buf, sizeof(buf),
				 "%sprojection.conv1.%s", prefix, s);
			return add_entry(out, inner_idx, buf);
		}

		/* head.1.* -> projection.bn.* (BatchNorm) */
		s = strip_prefix(r, "1.");
		if (s) {
			snprintf(buf, sizeof(buf),
				 "%sprojection.bn.%s", prefix, s);
			return add_entry(out, inner_idx, buf);
		}

		/* head.3.* -> projection.conv2.* (3x3 conv) */
		s = strip_prefix(r, "3.");
		if (s) {
			snprintf(buf, sizeof(buf),
				 "%sprojection.conv2.%s", prefix, s);
			return add_entry(out, inner_idx, buf);
		}

		/* Passthrough within head */
		snprintf(buf, sizeof(buf),
			 "%sprojection.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * ViT backbone: detector.backbone.vision_backbone.trunk.*
 * -> detector_model.vision_encoder.backbone.*
 */
static int handle_vit(struct rename_entry *out, int inner_idx,
		      const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "detector_model.vision_encoder.backbone.";
	const char *r;

	/* patch_embed.proj.weight -> embeddings.patch_embeddings.projection.weight */
	r = strip_prefix(rest, "patch_embed.proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sembeddings.patch_embeddings.projection.%s",
			 prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* pos_embed -> embeddings.position_embedding.weight */
	if (strcmp(rest, "pos_embed") == 0) {
		snprintf(buf, sizeof(buf),
			 "%sembeddings.position_embedding.weight", prefix);
		return add_entry(out, inner_idx, buf);
	}

	/* ln_pre.* -> layer_norm.* */
	r = strip_prefix(rest, "ln_pre.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%slayer_norm.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* blocks.{i}.* -> layers.{i}.* with sub-rules */
	r = strip_prefix(rest, "blocks.");
	if (r) {
		int idx;
		int consumed;
		if (sscanf(r, "%d.%n", &idx, &consumed) < 1)
			goto passthrough;
		const char *attr = r + consumed;

		char layer_prefix[SAM3_WEIGHT_NAME_MAX];
		snprintf(layer_prefix, sizeof(layer_prefix),
			 "%slayers.%d.", prefix, idx);

		/* .attn.qkv.X -> SPLIT .attention.{q,k,v}_proj.X */
		r = strip_prefix(attr, "attn.qkv.");
		if (r) {
			char split_prefix[SAM3_WEIGHT_NAME_MAX];
			snprintf(split_prefix, sizeof(split_prefix),
				 "%sattention.", layer_prefix);
			return add_qkv_split(out, inner_idx,
					     split_prefix, r);
		}

		/* .attn.proj.X -> .attention.o_proj.X */
		r = strip_prefix(attr, "attn.proj.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%sattention.o_proj.%s",
				 layer_prefix, r);
			return add_entry(out, inner_idx, buf);
		}

		/* .attn.freqs_cis -> pass through with prefix */
		if (strcmp(attr, "attn.freqs_cis") == 0) {
			snprintf(buf, sizeof(buf),
				 "%sattn.freqs_cis", layer_prefix);
			return add_entry(out, inner_idx, buf);
		}

		/* .norm1.X -> .layer_norm1.X */
		r = strip_prefix(attr, "norm1.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm1.%s", layer_prefix, r);
			return add_entry(out, inner_idx, buf);
		}

		/* .norm2.X -> .layer_norm2.X */
		r = strip_prefix(attr, "norm2.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm2.%s", layer_prefix, r);
			return add_entry(out, inner_idx, buf);
		}

		/* .mlp.fc1.X and .mlp.fc2.X -> unchanged */
		/* Passthrough within layer */
		snprintf(buf, sizeof(buf), "%s%s", layer_prefix, attr);
		return add_entry(out, inner_idx, buf);
	}

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * FPN neck: detector.backbone.vision_backbone.convs.{i}.*
 * -> detector_model.vision_encoder.neck.fpn_layers.*
 */
static int handle_neck(struct rename_entry *out, int inner_idx,
		       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix =
		"detector_model.vision_encoder.neck.fpn_layers.";
	const char *r;
	int idx, consumed;

	/* convs.{i}.X -> fpn_layers.{i}.{renamed}.X */
	if (sscanf(rest, "%d.%n", &idx, &consumed) < 1)
		goto passthrough;

	const char *attr = rest + consumed;

	char fpn_prefix[SAM3_WEIGHT_NAME_MAX];
	snprintf(fpn_prefix, sizeof(fpn_prefix), "%s%d.", prefix, idx);

	/* dconv_2x2_0.X -> scale_layers.0.X */
	r = strip_prefix(attr, "dconv_2x2_0.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sscale_layers.0.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* dconv_2x2_1.X -> scale_layers.2.X (NOT 1!) */
	r = strip_prefix(attr, "dconv_2x2_1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sscale_layers.2.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* dconv_2x2.X -> scale_layers.0.X (single deconv for 2x) */
	r = strip_prefix(attr, "dconv_2x2.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sscale_layers.0.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* conv_1x1.X -> proj1.X */
	r = strip_prefix(attr, "conv_1x1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sproj1.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* conv_3x3.X -> proj2.X */
	r = strip_prefix(attr, "conv_3x3.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sproj2.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Passthrough within FPN layer */
	snprintf(buf, sizeof(buf), "%s%s", fpn_prefix, attr);
	return add_entry(out, inner_idx, buf);

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * SAM2 FPN neck: detector.backbone.vision_backbone.sam2_convs.{i}.*
 * -> detector_model.vision_encoder.neck.sam2_fpn_layers.*
 *
 * Identical structure to handle_neck() (dconv_2x2, conv_1x1, conv_3x3)
 * but writes to sam2_fpn_layers instead of fpn_layers. This duplicate
 * FPN is used by the tracker in EfficientSAM3 checkpoints.
 */
static int handle_sam2_neck(struct rename_entry *out, int inner_idx,
			    const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix =
		"detector_model.vision_encoder.neck.sam2_fpn_layers.";
	const char *r;
	int idx, consumed;

	/* sam2_convs.{i}.X -> sam2_fpn_layers.{i}.{renamed}.X */
	if (sscanf(rest, "%d.%n", &idx, &consumed) < 1)
		goto passthrough;

	const char *attr = rest + consumed;

	char fpn_prefix[SAM3_WEIGHT_NAME_MAX];
	snprintf(fpn_prefix, sizeof(fpn_prefix), "%s%d.", prefix, idx);

	/* dconv_2x2_0.X -> scale_layers.0.X */
	r = strip_prefix(attr, "dconv_2x2_0.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sscale_layers.0.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* dconv_2x2_1.X -> scale_layers.2.X (NOT 1!) */
	r = strip_prefix(attr, "dconv_2x2_1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sscale_layers.2.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* dconv_2x2.X -> scale_layers.0.X (single deconv for 2x) */
	r = strip_prefix(attr, "dconv_2x2.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sscale_layers.0.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* conv_1x1.X -> proj1.X */
	r = strip_prefix(attr, "conv_1x1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sproj1.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* conv_3x3.X -> proj2.X */
	r = strip_prefix(attr, "conv_3x3.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sproj2.%s", fpn_prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Passthrough within FPN layer */
	snprintf(buf, sizeof(buf), "%s%s", fpn_prefix, attr);
	return add_entry(out, inner_idx, buf);

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Text encoder: detector.backbone.language_backbone.*
 * -> detector_model.text_encoder.*
 */
static int handle_text_encoder(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "detector_model.text_encoder.";
	const char *r;

	/* encoder.text_projection [1024, 512] is the CLIP-internal
	 * projection — NOT used in SAM3 inference (VETextEncoder sets
	 * output_dim=None so text_projection=None in Python).
	 * Skip it by giving it an inert name. */
	if (strcmp(rest, "encoder.text_projection") == 0) {
		return add_entry(out, inner_idx,
				 "detector_model.text_encoder._unused_clip_text_projection");
	}

	/* resizer.weight/bias -> detector_model.text_projection.weight/bias
	 * This is the actual VETextEncoder resizer nn.Linear(1024, 256)
	 * that projects CLIP token embeddings to d_model. */
	if (strcmp(rest, "resizer.weight") == 0) {
		return add_entry(out, inner_idx,
				 "detector_model.text_projection.weight");
	}
	if (strcmp(rest, "resizer.bias") == 0) {
		return add_entry(out, inner_idx,
				 "detector_model.text_projection.bias");
	}

	/* encoder.token_embedding.X ->
	 * text_model.embeddings.token_embedding.X */
	r = strip_prefix(rest, "encoder.token_embedding.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%stext_model.embeddings.token_embedding.%s",
			 prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* encoder.positional_embedding ->
	 * text_model.embeddings.position_embedding.weight */
	if (strcmp(rest, "encoder.positional_embedding") == 0) {
		snprintf(buf, sizeof(buf),
			 "%stext_model.embeddings."
			 "position_embedding.weight", prefix);
		return add_entry(out, inner_idx, buf);
	}

	/* encoder.ln_final.X -> text_model.final_layer_norm.X */
	r = strip_prefix(rest, "encoder.ln_final.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%stext_model.final_layer_norm.%s",
			 prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* encoder.transformer.resblocks.{i}.* ->
	 * text_model.encoder.layers.{i}.* */
	r = strip_prefix(rest, "encoder.transformer.resblocks.");
	if (r) {
		int idx, consumed;
		if (sscanf(r, "%d.%n", &idx, &consumed) < 1)
			goto passthrough;
		const char *attr = r + consumed;

		char lp[SAM3_WEIGHT_NAME_MAX];
		snprintf(lp, sizeof(lp),
			 "%stext_model.encoder.layers.%d.",
			 prefix, idx);

		/* attn.in_proj_weight -> SPLIT self_attn.{q,k,v}_proj.weight */
		if (strcmp(attr, "attn.in_proj_weight") == 0) {
			char sp[SAM3_WEIGHT_NAME_MAX];
			snprintf(sp, sizeof(sp), "%sself_attn.", lp);
			return add_qkv_split(out, inner_idx,
					     sp, "weight");
		}

		/* attn.in_proj_bias -> SPLIT self_attn.{q,k,v}_proj.bias */
		if (strcmp(attr, "attn.in_proj_bias") == 0) {
			char sp[SAM3_WEIGHT_NAME_MAX];
			snprintf(sp, sizeof(sp), "%sself_attn.", lp);
			return add_qkv_split(out, inner_idx,
					     sp, "bias");
		}

		/* attn.out_proj.X -> self_attn.out_proj.X (keep out_proj) */
		r = strip_prefix(attr, "attn.out_proj.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%sself_attn.out_proj.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* ln_1.X -> layer_norm1.X */
		r = strip_prefix(attr, "ln_1.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm1.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* ln_2.X -> layer_norm2.X */
		r = strip_prefix(attr, "ln_2.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm2.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* mlp.c_fc.X -> mlp.fc1.X */
		r = strip_prefix(attr, "mlp.c_fc.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%smlp.fc1.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* mlp.c_proj.X -> mlp.fc2.X */
		r = strip_prefix(attr, "mlp.c_proj.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%smlp.fc2.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* Passthrough within resblock layer */
		snprintf(buf, sizeof(buf), "%s%s", lp, attr);
		return add_entry(out, inner_idx, buf);
	}

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * DETR encoder: detector.transformer.encoder.*
 * -> detector_model.detr_encoder.*
 */
static int handle_detr_encoder(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "detector_model.detr_encoder.";
	const char *r;

	/* layers.{i}.* */
	r = strip_prefix(rest, "layers.");
	if (!r)
		goto passthrough;

	int idx, consumed;
	if (sscanf(r, "%d.%n", &idx, &consumed) < 1)
		goto passthrough;
	const char *attr = r + consumed;

	char lp[SAM3_WEIGHT_NAME_MAX];
	snprintf(lp, sizeof(lp), "%slayers.%d.", prefix, idx);

	/* self_attn.in_proj_weight -> SPLIT self_attn.{q,k,v}_proj.weight */
	if (strcmp(attr, "self_attn.in_proj_weight") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%sself_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "weight");
	}

	/* self_attn.in_proj_bias -> SPLIT self_attn.{q,k,v}_proj.bias */
	if (strcmp(attr, "self_attn.in_proj_bias") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%sself_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "bias");
	}

	/* self_attn.out_proj.X -> self_attn.o_proj.X */
	r = strip_prefix(attr, "self_attn.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sself_attn.o_proj.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* cross_attn_image.in_proj_weight -> SPLIT cross_attn.{q,k,v}_proj.weight */
	if (strcmp(attr, "cross_attn_image.in_proj_weight") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%scross_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "weight");
	}

	/* cross_attn_image.in_proj_bias -> SPLIT cross_attn.{q,k,v}_proj.bias */
	if (strcmp(attr, "cross_attn_image.in_proj_bias") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%scross_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "bias");
	}

	/* cross_attn_image.out_proj.X -> cross_attn.o_proj.X */
	r = strip_prefix(attr, "cross_attn_image.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%scross_attn.o_proj.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* norm1.X -> layer_norm1.X */
	r = strip_prefix(attr, "norm1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%slayer_norm1.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* norm2.X -> layer_norm2.X */
	r = strip_prefix(attr, "norm2.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%slayer_norm2.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* norm3.X -> layer_norm3.X */
	r = strip_prefix(attr, "norm3.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%slayer_norm3.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* linear1.X -> mlp.fc1.X */
	r = strip_prefix(attr, "linear1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%smlp.fc1.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* linear2.X -> mlp.fc2.X */
	r = strip_prefix(attr, "linear2.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%smlp.fc2.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Passthrough within layer */
	snprintf(buf, sizeof(buf), "%s%s", lp, attr);
	return add_entry(out, inner_idx, buf);

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * DETR decoder: detector.transformer.decoder.*
 * -> detector_model.detr_decoder.*
 */
static int handle_detr_decoder(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "detector_model.detr_decoder.";
	const char *r;

	/* norm.X (decoder-level, not in layers) -> output_layer_norm.X */
	r = strip_prefix(rest, "norm.");
	if (r && r[0] != '\0') {
		/* Make sure it's not inside layers.{i} */
		const char *layers_check = strip_prefix(rest, "layers.");
		if (!layers_check) {
			snprintf(buf, sizeof(buf),
				 "%soutput_layer_norm.%s", prefix, r);
			return add_entry(out, inner_idx, buf);
		}
	}

	/* bbox_embed.layers.{j}.X -> box_head.layer{j+1}.X (decoder level) */
	r = strip_prefix(rest, "bbox_embed.layers.");
	if (r) {
		int j, jc;
		if (sscanf(r, "%d.%n", &j, &jc) >= 1) {
			snprintf(buf, sizeof(buf),
				 "%sbox_head.layer%d.%s",
				 prefix, j + 1, r + jc);
			return add_entry(out, inner_idx, buf);
		}
	}

	/* ref_point_head.layers.{j}.X -> ref_point_head.layer{j+1}.X (decoder level) */
	r = strip_prefix(rest, "ref_point_head.layers.");
	if (r) {
		int j, jc;
		if (sscanf(r, "%d.%n", &j, &jc) >= 1) {
			snprintf(buf, sizeof(buf),
				 "%sref_point_head.layer%d.%s",
				 prefix, j + 1, r + jc);
			return add_entry(out, inner_idx, buf);
		}
	}

	/* layers.{i}.* */
	r = strip_prefix(rest, "layers.");
	if (!r)
		goto passthrough;

	int idx, consumed;
	if (sscanf(r, "%d.%n", &idx, &consumed) < 1)
		goto passthrough;
	const char *attr = r + consumed;

	char lp[SAM3_WEIGHT_NAME_MAX];
	snprintf(lp, sizeof(lp), "%slayers.%d.", prefix, idx);

	/* self_attn.in_proj_weight -> SPLIT self_attn.{q,k,v}_proj.weight */
	if (strcmp(attr, "self_attn.in_proj_weight") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%sself_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "weight");
	}

	/* self_attn.in_proj_bias -> SPLIT self_attn.{q,k,v}_proj.bias */
	if (strcmp(attr, "self_attn.in_proj_bias") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%sself_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "bias");
	}

	/* self_attn.out_proj.X -> self_attn.o_proj.X */
	r = strip_prefix(attr, "self_attn.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sself_attn.o_proj.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* cross_attn.in_proj_weight -> SPLIT vision_cross_attn.{q,k,v}_proj.weight */
	if (strcmp(attr, "cross_attn.in_proj_weight") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%svision_cross_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "weight");
	}

	/* cross_attn.in_proj_bias -> SPLIT vision_cross_attn.{q,k,v}_proj.bias */
	if (strcmp(attr, "cross_attn.in_proj_bias") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%svision_cross_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "bias");
	}

	/* cross_attn.out_proj.X -> vision_cross_attn.o_proj.X */
	r = strip_prefix(attr, "cross_attn.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%svision_cross_attn.o_proj.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* ca_text.in_proj_weight -> SPLIT text_cross_attn.{q,k,v}_proj.weight */
	if (strcmp(attr, "ca_text.in_proj_weight") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%stext_cross_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "weight");
	}

	/* ca_text.in_proj_bias -> SPLIT text_cross_attn.{q,k,v}_proj.bias */
	if (strcmp(attr, "ca_text.in_proj_bias") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp), "%stext_cross_attn.", lp);
		return add_qkv_split(out, inner_idx, sp, "bias");
	}

	/* ca_text.out_proj.X -> text_cross_attn.o_proj.X */
	r = strip_prefix(attr, "ca_text.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%stext_cross_attn.o_proj.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* norm1.X -> vision_cross_attn_layer_norm.X */
	r = strip_prefix(attr, "norm1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%svision_cross_attn_layer_norm.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* norm2.X -> self_attn_layer_norm.X */
	r = strip_prefix(attr, "norm2.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sself_attn_layer_norm.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* catext_norm.X -> text_cross_attn_layer_norm.X */
	r = strip_prefix(attr, "catext_norm.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%stext_cross_attn_layer_norm.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* norm3.X -> mlp_layer_norm.X */
	r = strip_prefix(attr, "norm3.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%smlp_layer_norm.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* linear1.X -> mlp.fc1.X */
	r = strip_prefix(attr, "linear1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%smlp.fc1.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* linear2.X -> mlp.fc2.X */
	r = strip_prefix(attr, "linear2.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%smlp.fc2.%s", lp, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Passthrough within layer */
	snprintf(buf, sizeof(buf), "%s%s", lp, attr);
	return add_entry(out, inner_idx, buf);

passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Segmentation head: detector.segmentation_head.*
 * -> detector_model.mask_decoder.*
 */
static int handle_seg_head(struct rename_entry *out, int inner_idx,
			   const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "detector_model.mask_decoder.";
	const char *r;

	/* mask_predictor.mask_embed.layers.{i}.X ->
	 * mask_embedder.layers.{i}.X */
	r = strip_prefix(rest, "mask_predictor.mask_embed.layers.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%smask_embedder.layers.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* instance_seg_head.X -> instance_projection.X */
	r = strip_prefix(rest, "instance_seg_head.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sinstance_projection.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* cross_attend_prompt.in_proj_weight ->
	 * SPLIT prompt_cross_attn.{q,k,v}_proj.weight */
	if (strcmp(rest, "cross_attend_prompt.in_proj_weight") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp),
			 "%sprompt_cross_attn.", prefix);
		return add_qkv_split(out, inner_idx, sp, "weight");
	}

	/* cross_attend_prompt.in_proj_bias ->
	 * SPLIT prompt_cross_attn.{q,k,v}_proj.bias */
	if (strcmp(rest, "cross_attend_prompt.in_proj_bias") == 0) {
		char sp[SAM3_WEIGHT_NAME_MAX];
		snprintf(sp, sizeof(sp),
			 "%sprompt_cross_attn.", prefix);
		return add_qkv_split(out, inner_idx, sp, "bias");
	}

	/* cross_attend_prompt.out_proj.X -> prompt_cross_attn.o_proj.X */
	r = strip_prefix(rest, "cross_attend_prompt.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sprompt_cross_attn.o_proj.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* cross_attn_norm.X -> prompt_cross_attn_norm.X */
	r = strip_prefix(rest, "cross_attn_norm.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sprompt_cross_attn_norm.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Everything else: pass through with prefix */
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Geometry encoder: detector.geometry_encoder.*
 * -> geom_enc.*
 *
 * The C model uses a simplified cross-attention-only architecture.
 * Fused in_proj [3d,d] cannot be split into Q [d,d] + KV [2d,d]
 * with our current 3-way split infrastructure. Pass through with
 * prefix and let unmatched keys fall back to zeros at load time.
 */
static int handle_geom_encoder(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "geom_enc.";
	const char *r;

	/* points_direct_project.X -> point_proj.X */
	r = strip_prefix(rest, "points_direct_project.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%spoint_proj.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* boxes_direct_project.X -> box_proj.X */
	r = strip_prefix(rest, "boxes_direct_project.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%sbox_proj.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* cls_embed.weight -> cls_token (bare param) */
	if (strcmp(rest, "cls_embed.weight") == 0) {
		snprintf(buf, sizeof(buf), "%scls_token", prefix);
		return add_entry(out, inner_idx, buf);
	}

	/* encode.{i}.X -> layers.{i}.{mapped_attr} */
	r = strip_prefix(rest, "encode.");
	if (r) {
		int idx, consumed;
		if (sscanf(r, "%d.%n", &idx, &consumed) < 1)
			goto passthrough;
		const char *attr = r + consumed;

		char lp[SAM3_WEIGHT_NAME_MAX];
		snprintf(lp, sizeof(lp), "%slayers.%d.", prefix, idx);

		/* cross_attn_image.in_proj_weight -> Q+KV split */
		if (strcmp(attr, "cross_attn_image.in_proj_weight") == 0)
			return add_q_kv_split(out, inner_idx, lp,
					      "ca_q.", "ca_kv.", "weight");

		/* cross_attn_image.in_proj_bias -> Q+KV split */
		if (strcmp(attr, "cross_attn_image.in_proj_bias") == 0)
			return add_q_kv_split(out, inner_idx, lp,
					      "ca_q.", "ca_kv.", "bias");

		/* cross_attn_image.out_proj.X -> ca_out.X */
		const char *s;
		s = strip_prefix(attr, "cross_attn_image.out_proj.");
		if (s) {
			snprintf(buf, sizeof(buf), "%sca_out.%s", lp, s);
			return add_entry(out, inner_idx, buf);
		}

		/* norm2.X -> ca_ln.X (norm2 = cross-attn norm in PT) */
		s = strip_prefix(attr, "norm2.");
		if (s) {
			snprintf(buf, sizeof(buf), "%sca_ln.%s", lp, s);
			return add_entry(out, inner_idx, buf);
		}

		/* Everything else (self_attn, norm2, norm3, linear1, linear2):
		 * pass through - C model doesn't use these */
		snprintf(buf, sizeof(buf), "%s%s", lp, attr);
		return add_entry(out, inner_idx, buf);
	}

	/* final_proj.X -> post_proj.X */
	r = strip_prefix(rest, "final_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%spost_proj.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Everything else: pass through with prefix */
passthrough:
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Dot product scorer: detector.dot_prod_scoring.*
 * -> tracker_model.mask_decoder.pred_obj_score_head.*
 */
static int handle_scorer(struct rename_entry *out, int inner_idx,
			 const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix =
		"tracker_model.mask_decoder.pred_obj_score_head.";
	const char *r;

	/* prompt_mlp.layers.0.X -> prompt_mlp.fc1.X */
	r = strip_prefix(rest, "prompt_mlp.layers.0.");
	if (r) {
		snprintf(buf, sizeof(buf), "%sprompt_mlp.fc1.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* prompt_mlp.layers.1.X -> prompt_mlp.fc2.X */
	r = strip_prefix(rest, "prompt_mlp.layers.1.");
	if (r) {
		snprintf(buf, sizeof(buf), "%sprompt_mlp.fc2.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* Everything else: pass through with prefix */
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Mask decoder: tracker.sam_mask_decoder.*
 * -> tracker_model.mask_decoder.*
 *
 * PT names need attribute renaming for upscaling, norms, MLP layers,
 * out_proj->o_proj, and hypernetwork MLP structure.
 */
static int handle_mask_decoder(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "tracker_model.mask_decoder.";
	const char *r;

	/* output_upscaling.0.X -> upscale_conv1.X */
	r = strip_prefix(rest, "output_upscaling.0.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%supscale_conv1.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* output_upscaling.1.X -> upscale_layer_norm.X */
	r = strip_prefix(rest, "output_upscaling.1.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%supscale_layer_norm.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* output_upscaling.3.X -> upscale_conv2.X */
	r = strip_prefix(rest, "output_upscaling.3.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%supscale_conv2.%s", prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* transformer.final_attn_token_to_image.out_proj.X ->
	 * transformer.final_attn_token_to_image.o_proj.X */
	r = strip_prefix(rest,
			 "transformer.final_attn_token_to_image.out_proj.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%stransformer."
			 "final_attn_token_to_image.o_proj.%s",
			 prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* transformer.norm_final_attn.X ->
	 * transformer.layer_norm_final_attn.X */
	r = strip_prefix(rest, "transformer.norm_final_attn.");
	if (r) {
		snprintf(buf, sizeof(buf),
			 "%stransformer.layer_norm_final_attn.%s",
			 prefix, r);
		return add_entry(out, inner_idx, buf);
	}

	/* transformer.layers.{i}.* with sub-rules */
	r = strip_prefix(rest, "transformer.layers.");
	if (r) {
		int idx, consumed;
		if (sscanf(r, "%d.%n", &idx, &consumed) < 1)
			goto check_hypernetworks;
		const char *attr = r + consumed;

		char lp[SAM3_WEIGHT_NAME_MAX];
		snprintf(lp, sizeof(lp),
			 "%stransformer.layers.%d.", prefix, idx);

		/* self_attn.out_proj.X -> self_attn.o_proj.X */
		r = strip_prefix(attr, "self_attn.out_proj.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%sself_attn.o_proj.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* cross_attn_token_to_image.out_proj.X ->
		 * cross_attn_token_to_image.o_proj.X */
		r = strip_prefix(attr,
				 "cross_attn_token_to_image.out_proj.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%scross_attn_token_to_image."
				 "o_proj.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* cross_attn_image_to_token.out_proj.X ->
		 * cross_attn_image_to_token.o_proj.X */
		r = strip_prefix(attr,
				 "cross_attn_image_to_token.out_proj.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%scross_attn_image_to_token."
				 "o_proj.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* norm1.X -> layer_norm1.X */
		r = strip_prefix(attr, "norm1.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm1.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* norm2.X -> layer_norm2.X */
		r = strip_prefix(attr, "norm2.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm2.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* norm3.X -> layer_norm3.X */
		r = strip_prefix(attr, "norm3.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm3.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* norm4.X -> layer_norm4.X */
		r = strip_prefix(attr, "norm4.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%slayer_norm4.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* mlp.lin1.X -> mlp.proj_in.X */
		r = strip_prefix(attr, "mlp.lin1.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%smlp.proj_in.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* mlp.lin2.X -> mlp.proj_out.X */
		r = strip_prefix(attr, "mlp.lin2.");
		if (r) {
			snprintf(buf, sizeof(buf),
				 "%smlp.proj_out.%s", lp, r);
			return add_entry(out, inner_idx, buf);
		}

		/* Passthrough within transformer layer */
		snprintf(buf, sizeof(buf), "%s%s", lp, attr);
		return add_entry(out, inner_idx, buf);
	}

check_hypernetworks:
	/* output_hypernetworks_mlps.{i}.layers.0.X ->
	 * output_hypernetworks_mlps.{i}.proj_in.X */
	r = strip_prefix(rest, "output_hypernetworks_mlps.");
	if (r) {
		int idx, consumed;
		if (sscanf(r, "%d.%n", &idx, &consumed) >= 1) {
			const char *mlp_attr = r + consumed;
			const char *s;

			char hp[SAM3_WEIGHT_NAME_MAX];
			snprintf(hp, sizeof(hp),
				 "%soutput_hypernetworks_mlps.%d.",
				 prefix, idx);

			/* layers.0.X -> proj_in.X */
			s = strip_prefix(mlp_attr, "layers.0.");
			if (s) {
				snprintf(buf, sizeof(buf),
					 "%sproj_in.%s", hp, s);
				return add_entry(out, inner_idx, buf);
			}

			/* layers.1.X -> layers.0.X */
			s = strip_prefix(mlp_attr, "layers.1.");
			if (s) {
				snprintf(buf, sizeof(buf),
					 "%slayers.0.%s", hp, s);
				return add_entry(out, inner_idx, buf);
			}

			/* layers.2.X -> proj_out.X */
			s = strip_prefix(mlp_attr, "layers.2.");
			if (s) {
				snprintf(buf, sizeof(buf),
					 "%sproj_out.%s", hp, s);
				return add_entry(out, inner_idx, buf);
			}

			/* Other sub-attrs: passthrough */
			snprintf(buf, sizeof(buf),
				 "%s%s", hp, mlp_attr);
			return add_entry(out, inner_idx, buf);
		}
	}

	/* iou_prediction_head.layers.{j}.X -> renumbered */
	r = strip_prefix(rest, "iou_prediction_head.layers.");
	if (r) {
		int j, jc;
		if (sscanf(r, "%d.%n", &j, &jc) >= 1) {
			const char *s = r + jc;
			char hp[SAM3_WEIGHT_NAME_MAX];
			snprintf(hp, sizeof(hp),
				 "%siou_prediction_head.", prefix);

			if (j == 0) {
				snprintf(buf, sizeof(buf),
					 "%sproj_in.%s", hp, s);
				return add_entry(out, inner_idx, buf);
			}
			if (j == 1) {
				snprintf(buf, sizeof(buf),
					 "%slayers.0.%s", hp, s);
				return add_entry(out, inner_idx, buf);
			}
			if (j == 2) {
				snprintf(buf, sizeof(buf),
					 "%sproj_out.%s", hp, s);
				return add_entry(out, inner_idx, buf);
			}
		}
	}

	/* Everything else: pass through with prefix */
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Prompt encoder: tracker.sam_prompt_encoder.*
 * -> tracker_model.prompt_encoder.*
 */
static int handle_prompt_encoder(struct rename_entry *out, int inner_idx,
				 const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	const char *prefix = "tracker_model.prompt_encoder.";

	/* pe_layer.positional_encoding_gaussian_matrix ->
	 * shared_embedding.positional_embedding */
	if (strcmp(rest,
		  "pe_layer.positional_encoding_gaussian_matrix") == 0) {
		snprintf(buf, sizeof(buf),
			 "%sshared_embedding.positional_embedding",
			 prefix);
		return add_entry(out, inner_idx, buf);
	}

	/* Everything else: pass through with prefix */
	snprintf(buf, sizeof(buf), "%s%s", prefix, rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Memory attention: tracker.transformer.encoder.*
 * -> tracker_model.transformer.encoder.*
 *
 * The C memory_attn module loads weights using the PyTorch attribute
 * names directly (out_proj, cross_attn_image, norm1, linear1, etc.),
 * so no attribute renaming is needed — just a prefix change.
 */
static int handle_memory_attention(struct rename_entry *out, int inner_idx,
				   const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	snprintf(buf, sizeof(buf),
		 "tracker_model.transformer.encoder.%s", rest);
	return add_entry(out, inner_idx, buf);
}

/* Memory encoder: tracker.maskmem_backbone.* -> tracker_model.maskmem_backbone.* */
static int handle_memory_encoder(struct rename_entry *out, int inner_idx,
				 const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	snprintf(buf, sizeof(buf),
		 "tracker_model.maskmem_backbone.%s", rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * Tracker misc: fallback for tracker.* keys not matched by more
 * specific prefixes (mask_downsample, no_mem_embed, etc.).
 * Maps to tracker_model.* which tracker.c loads with WP "tracker_model.".
 */
static int handle_tracker_misc(struct rename_entry *out, int inner_idx,
			       const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	snprintf(buf, sizeof(buf), "tracker_model.%s", rest);
	return add_entry(out, inner_idx, buf);
}

/*
 * SAM 3.1 multiplex tracker — strip `tracker.model.` and emit keys
 * under the `tracker_multiplex.` namespace so C modules can load them without
 * colliding with SAM 3's `tracker_model.` names (the architectures
 * differ; same names would mean two incompatible layouts).
 *
 * The upstream multiplex checkpoint already stores weights with
 * pre-split q_proj / k_proj / v_proj / out_proj names — no QKV
 * splitting work needed here. The handler just rewrites the prefix.
 */
static int handle_tracker_multiplex(struct rename_entry *out, int inner_idx,
			     const char *rest)
{
	char buf[SAM3_WEIGHT_NAME_MAX];
	snprintf(buf, sizeof(buf), "tracker_multiplex.%s", rest);
	return add_entry(out, inner_idx, buf);
}

/* ── Prefix table (longest-match-wins) ─────────────────────────────── */

struct prefix_rule {
	const char *py_prefix;
	int (*handler)(struct rename_entry *out, int inner_idx,
		       const char *rest);
};

/*
 * Sorted longest-first so the first match is the most specific.
 * Maps PyTorch .pt key prefixes to C model names.
 */
static const struct prefix_rule prefix_table[] = {
	{"detector.backbone.vision_backbone.trunk.model.",
		handle_efficientvit},
	{"detector.backbone.vision_backbone.trunk.",
		handle_vit},
	{"detector.backbone.vision_backbone.sam2_convs.",
		handle_sam2_neck},
	{"detector.backbone.vision_backbone.convs.",
		handle_neck},
	{"detector.backbone.language_backbone.",
		handle_text_encoder},
	{"detector.transformer.encoder.",
		handle_detr_encoder},
	{"detector.transformer.decoder.",
		handle_detr_decoder},
	{"detector.segmentation_head.",
		handle_seg_head},
	{"detector.geometry_encoder.",
		handle_geom_encoder},
	{"detector.dot_prod_scoring.",
		handle_scorer},
	{"tracker.sam_mask_decoder.",
		handle_mask_decoder},
	{"tracker.sam_prompt_encoder.",
		handle_prompt_encoder},
	{"tracker.transformer.encoder.",
		handle_memory_attention},
	{"tracker.maskmem_backbone.",
		handle_memory_encoder},
	/* SAM 3.1 multiplex — all under tracker.model.* */
	{"tracker.model.",
		handle_tracker_multiplex},
	{"tracker.",
		handle_tracker_misc},
};

#define N_PREFIX_RULES \
	(int)(sizeof(prefix_table) / sizeof(prefix_table[0]))

/* ── Core rename logic ─────────────────────────────────────────────── */

/*
 * rename_tensor - Apply prefix + attribute renaming to a single tensor.
 *
 * Returns count of entries added (1 for normal, 3 for QKV split).
 */
static int rename_tensor(const char *name, int inner_idx,
			 struct rename_entry *out)
{
	for (int i = 0; i < N_PREFIX_RULES; i++) {
		const char *rest = strip_prefix(name,
						prefix_table[i].py_prefix);
		if (rest)
			return prefix_table[i].handler(out, inner_idx, rest);
	}

	/* No prefix matched: pass through unchanged */
	return add_entry(out, inner_idx, name);
}

static enum sam3_error build_rename_table(struct rename_reader_state *s)
{
	int inner_n = s->inner->ops->n_tensors(s->inner);

	/* Worst case: every tensor splits into 3 */
	s->entries = calloc((size_t)inner_n * 3, sizeof(*s->entries));
	if (!s->entries)
		return SAM3_ENOMEM;

	s->n_entries = 0;
	for (int i = 0; i < inner_n; i++) {
		struct weight_tensor_info info;
		enum sam3_error err;

		err = s->inner->ops->get_tensor_info(s->inner, i, &info);
		if (err != SAM3_OK) {
			free(s->entries);
			s->entries = NULL;
			return err;
		}

		int added = rename_tensor(info.name, i,
					  s->entries + s->n_entries);
		s->n_entries += added;
	}

	sam3_log_info("rename: %d inner tensors -> %d output entries",
		       inner_n, s->n_entries);

	return SAM3_OK;
}

/* ── Vtable callbacks ──────────────────────────────────────────────── */

static enum sam3_error rr_open(struct weight_reader *r, const char *path)
{
	(void)r; (void)path;
	return SAM3_OK; /* inner already opened */
}

static int rr_n_tensors(struct weight_reader *r)
{
	struct rename_reader_state *s = r->impl;
	return s->n_entries;
}

static enum sam3_error rr_get_tensor_info(struct weight_reader *r, int idx,
					  struct weight_tensor_info *info)
{
	struct rename_reader_state *s = r->impl;

	if (idx < 0 || idx >= s->n_entries)
		return SAM3_EINVAL;

	struct rename_entry *e = &s->entries[idx];

	/* Get inner tensor info */
	enum sam3_error err;
	err = s->inner->ops->get_tensor_info(s->inner, e->inner_idx, info);
	if (err != SAM3_OK)
		return err;

	/* Override name */
	info->name = e->name;

	/* Adjust dims/nbytes for splits */
	if (e->split_total > 0) {
		info->dims[0] = info->dims[0] * e->split_count / e->split_total;
		info->nbytes = info->nbytes * e->split_count / e->split_total;
	}

	return SAM3_OK;
}

static enum sam3_error rr_read_tensor_data(struct weight_reader *r, int idx,
					   void *dst, size_t dst_size)
{
	struct rename_reader_state *s = r->impl;

	if (idx < 0 || idx >= s->n_entries)
		return SAM3_EINVAL;

	struct rename_entry *e = &s->entries[idx];

	/* Non-split: pass through directly */
	if (e->split_total == 0)
		return s->inner->ops->read_tensor_data(s->inner,
						       e->inner_idx,
						       dst, dst_size);

	/* QKV split: read full inner tensor, copy the right slice */
	struct weight_tensor_info inner_info;
	enum sam3_error err;

	err = s->inner->ops->get_tensor_info(s->inner, e->inner_idx,
					     &inner_info);
	if (err != SAM3_OK)
		return err;

	size_t inner_nbytes = inner_info.nbytes;
	size_t slice_size = inner_nbytes / (size_t)e->split_total;
	size_t part_size = slice_size * (size_t)e->split_count;

	if (dst_size < part_size)
		return SAM3_EINVAL;

	void *full_buf = malloc(inner_nbytes);
	if (!full_buf)
		return SAM3_ENOMEM;

	err = s->inner->ops->read_tensor_data(s->inner, e->inner_idx,
					      full_buf, inner_nbytes);
	if (err != SAM3_OK) {
		free(full_buf);
		return err;
	}

	size_t offset = slice_size * (size_t)e->split_start;
	memcpy(dst, (const char *)full_buf + offset, part_size);

	free(full_buf);
	return SAM3_OK;
}

static void rr_close(struct weight_reader *r)
{
	struct rename_reader_state *s = r->impl;
	free(s->entries);
	s->entries = NULL;
	s->n_entries = 0;
	/* Do NOT close inner; caller owns it */
}

static const struct weight_reader_ops rename_reader_ops = {
	.open             = rr_open,
	.n_tensors        = rr_n_tensors,
	.get_tensor_info  = rr_get_tensor_info,
	.read_tensor_data = rr_read_tensor_data,
	.close            = rr_close,
};

/* ── Public init ───────────────────────────────────────────────────── */

/*
 * Persistent state — only one rename reader exists at a time in
 * the converter, so a static instance is fine.
 */
static struct rename_reader_state rr_state;

enum sam3_error weight_reader_rename_init(struct weight_reader *r,
					  struct weight_reader *inner)
{
	enum sam3_error err;

	memset(&rr_state, 0, sizeof(rr_state));
	rr_state.inner = inner;

	err = build_rename_table(&rr_state);
	if (err != SAM3_OK)
		return err;

	r->ops = &rename_reader_ops;
	r->impl = &rr_state;

	return SAM3_OK;
}
