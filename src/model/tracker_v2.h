/*
 * src/model/tracker_v2.h - SAM 3.1 multiplex tracker (scaffold)
 *
 * The SAM 3.1 video tracker. The memory-attention transformer and mask
 * decoder are different architectures from SAM 3's (8-head decoupled
 * vs 1-head RoPE; separate per-frame decoder with high-res conv_s0/s1
 * feeds vs the shared image-path SAM decoder), so SAM 3 and SAM 3.1
 * keep parallel tracker structs rather than bifurcating every field of
 * sam3_tracker.
 *
 * Phase 2.1 loads the small sub-modules only: singleton embedding
 * parameters, obj_ptr projections, maskmem backbone. The transformer,
 * sam_mask_decoder, and interactive pathways are stubbed in the struct
 * and filled in by phases 2.2-2.5 (see
 * docs/superpowers/specs/2026-04-19-sam3-1-multiplex-tracker-design.md).
 *
 * Key types:  sam3_tracker_v2
 * Depends on: core/tensor.h, core/weight.h, core/alloc.h
 * Used by:    model/sam3_video.c (variant dispatch), tests/test_tracker_v2_load.c
 *
 * Copyright (c) 2026 Rifky Bujana Bisri
 * SPDX-License-Identifier: MIT
 */

#ifndef SAM3_MODEL_TRACKER_V2_H
#define SAM3_MODEL_TRACKER_V2_H

#include "core/tensor.h"
#include "core/weight.h"
#include "core/alloc.h"
#include "core/graph.h"

/*
 * Multiplex bucket size. The SAM 3.1 checkpoint bakes 16 into several
 * tensors (no_obj_embed_spatial, output_{valid,invalid}_embed,
 * maskmem downsampler input channels = 16*2 = 32).
 */
#define SAM3_V2_MULTIPLEX_COUNT       16
#define SAM3_V2_MULTIPLEX_IN_CHANNEL_MULT  2
#define SAM3_V2_MULTIPLEX_IN_CHANNELS \
	(SAM3_V2_MULTIPLEX_COUNT * SAM3_V2_MULTIPLEX_IN_CHANNEL_MULT)  /* 32 */

#define SAM3_V2_NUM_MASKMEM   7
#define SAM3_V2_HIDDEN_DIM    256

/*
 * CXBlock (ConvNeXt-style fuser layer). Two of these live in
 * maskmem_backbone.fuser. Each one owns depthwise conv + layer norm +
 * pointwise conv1 + pointwise conv2 + a layer-scale gamma.
 */
struct sam3_v2_cxblock {
	struct sam3_tensor *dwconv_w;   /* [256, 1, 7, 7] depthwise */
	struct sam3_tensor *dwconv_b;   /* [256] */
	struct sam3_tensor *norm_w;     /* [256] */
	struct sam3_tensor *norm_b;     /* [256] */
	struct sam3_tensor *pwconv1_w;  /* [1024, 256] */
	struct sam3_tensor *pwconv1_b;  /* [1024] */
	struct sam3_tensor *pwconv2_w;  /* [256, 1024] */
	struct sam3_tensor *pwconv2_b;  /* [256] */
	struct sam3_tensor *gamma;      /* [256] layer scale */
};

/*
 * Progressive mask downsampler. Alternating Conv2d + LayerNorm2d +
 * GELU × 4 stages, then a 1×1 projection.
 *
 * channels:  32 -> 16 -> 64 -> 256 -> 1024 -> 256
 */
struct sam3_v2_mask_downsampler {
	/* conv/norm/(gelu) stack, 4 stages */
	struct sam3_tensor *conv_w[4];  /* OHWI [out, 3, 3, in] */
	struct sam3_tensor *conv_b[4];
	struct sam3_tensor *norm_w[4];  /* LayerNorm2d weight [out] */
	struct sam3_tensor *norm_b[4];  /* LayerNorm2d bias   [out] */
	/* Final 1×1 projection to hidden_dim */
	struct sam3_tensor *proj_w;     /* OHWI [256, 1, 1, 1024] */
	struct sam3_tensor *proj_b;     /* [256] */
};

/*
 * Mask-memory backbone (multiplex-aware SimpleMaskEncoder).
 *   masks [B, 32, 1152, 1152] -> sigmoid -> mask_downsampler
 *     -> [B, 256, 72, 72]
 *   pix_feat -> pix_feat_proj -> + masks -> fuser (2 CXBlocks) -> out
 */
struct sam3_v2_maskmem {
	struct sam3_v2_mask_downsampler mask_downsampler;
	struct sam3_tensor             *pix_feat_proj_w; /* OHWI [256,1,1,256] */
	struct sam3_tensor             *pix_feat_proj_b; /* [256] */
	struct sam3_v2_cxblock          fuser[2];
};

/*
 * 3-layer MLP used for object pointer projection (both the primary
 * obj_ptr_proj and the interactive path's own copy). 256->256->256->256
 * with biases.
 */
struct sam3_v2_mlp3 {
	struct sam3_tensor *fc_w[3];
	struct sam3_tensor *fc_b[3];
};

/*
 * Decoupled memory-attention layer (one of 4 in `transformer.encoder`).
 *
 * Python: DecoupledTransformerDecoderLayerv2 — three attention blocks
 * share the FFN but each has its own q/k/v/out projections:
 *
 *   self_attn         (8-head RoPE): object-query ↔ object-query
 *   cross_attn        (8-head RoPE): object-query ← memory-bank tokens
 *   image_cross_attn  (q/k only):    object-query ↔ image features;
 *     V and out are *shared* with cross_attn (that's what makes it
 *     "decoupled" — the query side has its own projections while the
 *     value/output path feeds the same buffer the memory-attn updates).
 *
 * Plus an FFN (linear1, GELU, linear2) and three LayerNorms
 * (norm1 before self_attn, norm2 before cross_attn, norm3 before FFN).
 */
struct sam3_v2_memory_attn_layer {
	/* self_attn */
	struct sam3_tensor *self_q_w, *self_q_b;
	struct sam3_tensor *self_k_w, *self_k_b;
	struct sam3_tensor *self_v_w, *self_v_b;
	struct sam3_tensor *self_out_w, *self_out_b;

	/* cross_attn (obj-query ← memory) */
	struct sam3_tensor *cross_q_w, *cross_q_b;
	struct sam3_tensor *cross_k_w, *cross_k_b;
	struct sam3_tensor *cross_v_w, *cross_v_b;
	struct sam3_tensor *cross_out_w, *cross_out_b;

	/* image_cross_attn — only q and k projections; shares V+out with
	 * cross_attn. */
	struct sam3_tensor *img_q_w, *img_q_b;
	struct sam3_tensor *img_k_w, *img_k_b;

	/* FFN: 256 -> 2048 -> 256 */
	struct sam3_tensor *lin1_w, *lin1_b;
	struct sam3_tensor *lin2_w, *lin2_b;

	/* 3 LayerNorms */
	struct sam3_tensor *norm1_w, *norm1_b;
	struct sam3_tensor *norm2_w, *norm2_b;
	struct sam3_tensor *norm3_w, *norm3_b;
};

/*
 * Memory-attention encoder: 4 layers + a final LayerNorm.
 */
struct sam3_v2_memory_attn {
	struct sam3_v2_memory_attn_layer layers[4];
	struct sam3_tensor *final_norm_w;
	struct sam3_tensor *final_norm_b;
};

/*
 * Top-level SAM 3.1 tracker. Sub-modules reserved for later phases
 * (sam_mask_decoder, interactive path) are placeholders for now —
 * see the sub-project-2 spec.
 */
struct sam3_tracker_v2 {
	/* --- memory attention transformer (phase 2.3a, 122 tensors) --- */
	struct sam3_v2_memory_attn transformer;

	/* --- maskmem backbone (phase 2.1, 38 tensors) --- */
	struct sam3_v2_maskmem maskmem;

	/* --- object pointer MLPs (phase 2.1, 6 tensors) --- */
	struct sam3_v2_mlp3    obj_ptr_proj;

	/* --- small projection layers (phase 2.1, 4 tensors) --- */
	struct sam3_tensor *obj_ptr_tpos_proj_w;  /* [256, 256] */
	struct sam3_tensor *obj_ptr_tpos_proj_b;  /* [256] */
	struct sam3_tensor *no_obj_ptr_linear_w;  /* [256, 256] */
	struct sam3_tensor *no_obj_ptr_linear_b;  /* [256] */

	/* --- singleton embeddings (phase 2.1, 6 tensors) --- */
	struct sam3_tensor *image_pe_gauss;          /* [2, 128] Gaussian PE basis */
	struct sam3_tensor *maskmem_tpos_enc;        /* [7, 1, 1, 256] */
	struct sam3_tensor *no_obj_embed_spatial;    /* [16, 256] */
	struct sam3_tensor *output_valid_embed;      /* [16, 256] */
	struct sam3_tensor *output_invalid_embed;    /* [16, 256] */
	struct sam3_tensor *interactivity_no_mem_embed; /* [1, 1, 256] */

	/*
	 * --- Deferred to later phases ---
	 *
	 * memory_attn_v2 (phase 2.3): 4-layer 8-head decoupled transformer,
	 *   122 tensors under tracker_v2.transformer.*
	 *
	 * sam_mask_decoder_v2 (phase 2.4): 125 tensors under
	 *   tracker_v2.sam_mask_decoder.*
	 *
	 * interactive_sam_mask_decoder (sub-project 3): 131 tensors under
	 *   tracker_v2.interactive_sam_mask_decoder.*
	 *
	 * interactive_sam_prompt_encoder (sub-project 3): 17 tensors.
	 *
	 * interactive_obj_ptr_proj (sub-project 3): 6 tensors.
	 *
	 * interactive_mask_downsample (sub-project 3): 2 tensors.
	 *
	 * No allocations happen for these in phase 2.1; the corresponding
	 * .sam3 tensors are present in the file but tolerated as unused by
	 * the loader (absence at load time is OK — the unused tensors just
	 * sit in the mmap region).
	 */
};

/*
 * Number of tensors loaded by sam3_tracker_v2_load at phase 2.1. Used
 * by the round-trip test to sanity-check that no plumbing was lost.
 * 38 (maskmem) + 6 (obj_ptr_proj) + 4 (obj_ptr_tpos + no_obj_linear) +
 * 6 (singletons) = 54.
 */
#define SAM3_V2_PHASE_2_1_TENSORS 54
#define SAM3_V2_PHASE_2_3A_TENSORS (54 + 122)  /* + transformer */

/*
 * sam3_tracker_v2_init - Zero the struct and seed config constants.
 *
 * @trk: caller-allocated tracker struct; memset + field defaults
 *       applied.
 *
 * Returns SAM3_OK, or SAM3_EINVAL if trk is NULL. Does not allocate
 * tensors.
 */
enum sam3_error sam3_tracker_v2_init(struct sam3_tracker_v2 *trk);

/*
 * sam3_tracker_v2_load - Populate the struct from a SAM 3.1 .sam3 file.
 *
 * @trk:   Initialized tracker (sam3_tracker_v2_init must have run).
 * @wf:    Weight file opened from sam3_convert --variant sam3.1.
 * @arena: Arena for tensor metadata allocations. Tensor data is
 *         referenced directly from the mmap region.
 *
 * All tensors live under the `tracker_v2.` namespace (see the rename
 * handler added alongside the sub-project-2 design spec). Returns
 * SAM3_OK on success. Missing tensors fail the load — this is a real
 * tracker, not a best-effort zero-fill.
 */
enum sam3_error sam3_tracker_v2_load(struct sam3_tracker_v2 *trk,
				     const struct sam3_weight_file *wf,
				     struct sam3_arena *arena);

/*
 * sam3_v2_maskmem_forward - Build the graph that encodes a
 * (pix_feat, masks) pair into a memory-tokens tensor.
 *
 * @g:         Graph being built.
 * @arena:     Arena for intermediate tensors.
 * @mm:        Loaded maskmem sub-module.
 * @pix_feat:  [1, H, W, 256] NHWC — usually the 1x-scale neck output
 *             (H=W=72 in production).
 * @masks:     [1, H*16, W*16, 32] NHWC — multiplex-packed mask logits
 *             (pre-sigmoid). In production H*16=W*16=1152.
 *
 * Shape:
 *   masks -> sigmoid -> 4-stage Conv+LN+GELU downsampler (stride 2 each)
 *         -> 1x1 proj -> [1, H, W, 256]
 *   pix_feat -> pix_feat_proj (1x1) -> [1, H, W, 256]
 *   sum + 2x CXBlock fuser -> [1, H, W, 256]
 *
 * Returns the output tensor on success. Returns NULL on allocation
 * failure; the caller must have space in the arena for the
 * intermediate activations (the 1152-scale mask path is the peak;
 * budget at least ~170 MiB of scratch for production sizes).
 *
 * The caller is responsible for sigmoid preprocessing if it wants to
 * pass in already-sigmoid-ed masks: pass `skip_mask_sigmoid = 1`.
 */
struct sam3_tensor *sam3_v2_maskmem_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_v2_maskmem *mm,
		struct sam3_tensor *pix_feat,
		struct sam3_tensor *masks,
		int skip_mask_sigmoid);

/*
 * sam3_v2_memory_attn_forward - Build the memory-attention graph for a
 * single tracker frame.
 *
 * PHASE 2.3B STUB. Returns NULL unconditionally; phase 2.3b fills in
 * the 8-head decoupled RoPE attention. The interface below describes
 * what the full implementation will consume.
 *
 * @g:                Graph being built.
 * @arena:            Arena for intermediate tensors.
 * @ma:               Loaded memory-attention transformer (4 layers).
 * @tgt:              Object queries [B, Nq, 256].
 * @tgt_pos:          Positional encoding for object queries [B, Nq, 256]
 *                    (unused when pos_enc_at_attn=false, which matches
 *                    the upstream SAM 3.1 config).
 * @image:            Image features [B, HW, 256] for the current frame.
 * @image_pos:        Image positional encoding [B, HW, 256] (fed as the
 *                    image-side K-projection input, NOT summed into Q).
 * @memory:           Memory-bank tokens from prior frames
 *                    [B, N_mem, 256]. Includes obj_ptr tokens + maskmem
 *                    tokens.
 * @memory_pos:       Memory positional encoding [B, N_mem, 256]
 *                    (summed into the image-side K projection).
 * @num_k_exclude_rope: Number of memory K tokens to skip RoPE on
 *                    (obj_ptr tokens do not receive RoPE; see upstream
 *                    SimpleRoPEAttention kwarg).
 *
 * Semantics per layer (`DecoupledTransformerDecoderLayerv2`):
 *
 *   ## Self-attention on tgt (8-head RoPE)
 *   tgt2 = norm1(tgt)
 *   q = self_attn_q_proj(tgt2)
 *   k = self_attn_k_proj(tgt2)
 *   v = self_attn_v_proj(tgt2)
 *   tgt = tgt + self_attn_out_proj(attn_rope(q, k, v))
 *
 *   ## Decoupled cross-attention (image + memory, 8-head RoPE)
 *   tgt2 = norm2(tgt)
 *   q = image_cross_attn_q_proj(image) + cross_attn_q_proj(tgt2)
 *   k = image_cross_attn_k_proj(memory_image) +
 *         cross_attn_k_proj(memory)
 *   v = cross_attn_v_proj(memory)
 *   tgt = tgt + cross_attn_out_proj(attn_rope(q, k, v))
 *
 *   ## FFN
 *   tgt2 = norm3(tgt)
 *   tgt = tgt + linear2(GELU(linear1(tgt2)))
 *
 * Final: encoder.norm(tgt) after all 4 layers.
 *
 * `memory_image` in the reference is the per-frame image features
 * concatenated across the 7 memory slots with positional offsets; the
 * caller sets it up from the memory bank's stored pix_feat snapshots.
 */
struct sam3_tensor *sam3_v2_memory_attn_forward(
		struct sam3_graph *g,
		struct sam3_arena *arena,
		const struct sam3_v2_memory_attn *ma,
		struct sam3_tensor *tgt,
		struct sam3_tensor *tgt_pos,
		struct sam3_tensor *image,
		struct sam3_tensor *image_pos,
		struct sam3_tensor *memory,
		struct sam3_tensor *memory_pos,
		int num_k_exclude_rope);

#endif /* SAM3_MODEL_TRACKER_V2_H */
