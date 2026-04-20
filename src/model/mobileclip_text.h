/*
 * src/model/mobileclip_text.h - MobileCLIP text encoder (S0/S1/L)
 *
 * Pre-norm transformer text encoder with non-causal attention. Covers
 * three variants from one code path, parameterized by config (n_layers,
 * width, n_heads, mlp_dim, n_repmixer_blocks). The S0 variant has
 * RepMixer blocks at indices listed in cfg.repmixer_block_indices
 * (verified in the .pt audit at indices 0 and 5); S1 and L have no
 * RepMixer blocks. Each layer slot uses either the std or repmixer
 * sub-struct depending on the per-block flag.
 *
 * Output contract matches sam3_text_encoder: per-token [ctx_len, 256]
 * plus pooled [256] from the EOT position.
 *
 * Key types:  sam3_mobileclip_text_encoder, sam3_mobileclip_config
 * Depends on: text_encoder_iface.h, core/tensor.h, core/graph.h
 * Used by:    src/model/text_encoder_iface.c, tests/test_mobileclip_text.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_MOBILECLIP_TEXT_H
#define SAM3_MODEL_MOBILECLIP_TEXT_H

#include "text_encoder_iface.h"
#include "core/tensor.h"
#include "core/graph.h"
#include "core/alloc.h"
#include "core/weight.h"
#include "backend/backend.h"

#define SAM3_MOBILECLIP_MAX_LAYERS 12
#define SAM3_MOBILECLIP_MAX_REPMIXER_BLOCKS 4

struct sam3_mobileclip_config {
	int  text_backbone;             /* enum sam3_text_backbone */
	int  n_layers;                  /* total transformer block count */
	int  width;                     /* embedding/transformer dim */
	int  n_heads;
	int  mlp_dim;
	int  ctx_len;
	int  out_dim;                   /* always 256 for SAM3 */
	int  vocab_size;                /* always 49408 */
	int  pos_embed_table_len;       /* always 77 */
	int  n_repmixer_blocks;         /* 0 for S1/L; 2 for S0 (indices 0,5) */
	int  repmixer_block_indices[SAM3_MOBILECLIP_MAX_REPMIXER_BLOCKS];
};

struct sam3_mobileclip_layer_std {
	struct sam3_tensor *ln1_w, *ln1_b;
	struct sam3_tensor *qkv_w, *qkv_b;
	struct sam3_tensor *out_w, *out_b;
	struct sam3_tensor *ln2_w, *ln2_b;
	struct sam3_tensor *fc1_w, *fc1_b;
	struct sam3_tensor *fc2_w, *fc2_b;
};

/*
 * RepMixer block weights (S0 blocks 0 and 5). MobileOne reparameterization
 * is NOT collapsed at export, so we ship parallel BN/conv branches that
 * the build code sums at inference. eps=1e-5 for every BN.
 *
 * See spec section "RepMixer block (S0 blocks 0 and 5) — verified from
 * reference" for the exact arithmetic.
 */
struct sam3_mobileclip_layer_repmixer {
	/* token_mixer.norm — single standalone BN. */
	struct sam3_tensor *norm_skip_w, *norm_skip_b;
	struct sam3_tensor *norm_skip_rm, *norm_skip_rv;

	/* token_mixer.mixer.rbr_skip — BN-only branch. */
	struct sam3_tensor *mixer_skip_w, *mixer_skip_b;
	struct sam3_tensor *mixer_skip_rm, *mixer_skip_rv;

	/* token_mixer.mixer.rbr_conv[0] — depthwise conv + BN branch. */
	struct sam3_tensor *mixer_conv_w;                    /* [C,1,1,11] */
	struct sam3_tensor *mixer_conv_bn_w, *mixer_conv_bn_b;
	struct sam3_tensor *mixer_conv_bn_rm, *mixer_conv_bn_rv;

	/* (Optional) token_mixer.mixer.rbr_scale — 1×1 conv + BN branch.
	 * Absent in the audited S0 checkpoint; loader leaves NULL when
	 * keys are missing and build_repmixer_block skips the branch. */
	struct sam3_tensor *mixer_scale_w;                   /* [C,C,1,1] or NULL */
	struct sam3_tensor *mixer_scale_bn_w, *mixer_scale_bn_b;
	struct sam3_tensor *mixer_scale_bn_rm, *mixer_scale_bn_rv;

	/* token_mixer residual scale: x + tm_layer_scale * (mixer_out - norm_out) */
	struct sam3_tensor *tm_layer_scale;                  /* [C,1,1] */

	/* convffn.conv — depthwise 1×11 conv + BN. */
	struct sam3_tensor *convffn_dw_w;                    /* [C,1,1,11] */
	struct sam3_tensor *convffn_dw_bn_w, *convffn_dw_bn_b;
	struct sam3_tensor *convffn_dw_bn_rm, *convffn_dw_bn_rv;

	/* convffn.fc1 — 1×1 conv (with bias). */
	struct sam3_tensor *convffn_fc1_w, *convffn_fc1_b;

	/* convffn.fc2 — 1×1 conv (with bias). */
	struct sam3_tensor *convffn_fc2_w, *convffn_fc2_b;

	/* outer block residual scale: x + outer_layer_scale * convffn(x) */
	struct sam3_tensor *outer_layer_scale;               /* [C,1,1] */
};

/*
 * Tagged per-block layer slot. is_repmixer=1 selects the repmixer union
 * arm; is_repmixer=0 selects std. Set by the loader from
 * cfg.repmixer_block_indices.
 */
struct sam3_mobileclip_layer {
	int is_repmixer;
	union {
		struct sam3_mobileclip_layer_std       std;
		struct sam3_mobileclip_layer_repmixer  repmixer;
	} u;
};

struct sam3_mobileclip_text_encoder {
	struct sam3_mobileclip_config cfg;

	/* Embeddings */
	struct sam3_tensor *token_embedding;   /* [vocab_size, width] */
	struct sam3_tensor *pos_embed_full;    /* [1,1,77,width] sliced at build */

	/* Final norm + projection */
	struct sam3_tensor *ln_final_w;
	struct sam3_tensor *ln_final_b;
	struct sam3_tensor *projection_layer;  /* [width, width] (raw tensor, no .weight suffix) */

	/* External 256-dim projector (TextStudentEncoder.projector) */
	struct sam3_tensor *out_proj_w;        /* [256, width] */
	struct sam3_tensor *out_proj_b;        /* [256] */

	/* Per-block tagged layers. cfg.n_layers entries used. */
	struct sam3_mobileclip_layer layers[SAM3_MOBILECLIP_MAX_LAYERS];
};

enum sam3_error sam3_mobileclip_text_load(
	struct sam3_mobileclip_text_encoder *enc,
	const struct sam3_weight_file *wf,
	struct sam3_arena *arena);

struct sam3_tensor *sam3_mobileclip_text_build(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_graph *g,
	struct sam3_tensor *token_ids,
	struct sam3_tensor **pooled_out,
	struct sam3_arena *arena);

struct sam3_tensor *sam3_mobileclip_text_build_perblock(
	struct sam3_mobileclip_text_encoder *enc,
	struct sam3_backend *be,
	struct sam3_tensor *token_ids,
	struct sam3_arena *scratch,
	struct sam3_arena *persist);

/*
 * sam3_mobileclip_config_for - Return the static const config for a variant.
 *
 * @text_backbone: SAM3_TEXT_MOBILECLIP_S0 / _S1 / _L
 *
 * Returns NULL on unknown variant.
 */
const struct sam3_mobileclip_config *sam3_mobileclip_config_for(int text_backbone);

#endif /* SAM3_MODEL_MOBILECLIP_TEXT_H */
