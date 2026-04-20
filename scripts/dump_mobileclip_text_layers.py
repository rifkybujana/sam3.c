#!/usr/bin/env python3
"""
scripts/dump_mobileclip_text_layers.py - Dump per-block activations from
the reference MobileCLIP text encoder for a fixed prompt. Used to build
golden fixtures that the C-side test_mobileclip_text checks against.

Usage:
    python scripts/dump_mobileclip_text_layers.py \
        --variant mobileclip_s1 \
        --ckpt models/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
        --prompt "a person riding a bike" \
        --out tests/fixtures/mobileclip_s1
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Locate the repo root and add reference packages to sys.path.
# The efficientsam3 submodule lives under reference/efficientsam3/sam3/ and
# exposes a top-level package named `sam3`.  The submodule is initialised in
# the main worktree; fall back to the main repo if the current worktree has
# an empty reference dir.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

def _find_efficientsam3_root() -> Path:
    """Return the path that contains the `sam3` package (efficientsam3 fork)."""
    candidate = REPO_ROOT / "reference" / "efficientsam3" / "sam3"
    if (candidate / "sam3" / "__init__.py").exists():
        return candidate
    # Worktree may have the submodule uninitialised; try the main repo.
    main_repo = Path("/Users/rbisri/Documents/sam3")
    fallback = main_repo / "reference" / "efficientsam3" / "sam3"
    if (fallback / "sam3" / "__init__.py").exists():
        return fallback
    raise RuntimeError(
        "Cannot find the efficientsam3 `sam3` package.  "
        "Run: git submodule update --init reference/efficientsam3"
    )

EFFICIENTSAM3_ROOT = _find_efficientsam3_root()
sys.path.insert(0, str(EFFICIENTSAM3_ROOT))

from sam3.model.text_encoder_student import TextStudentEncoder  # noqa: E402
from sam3.model_builder import _create_student_text_encoder     # noqa: E402

# ---------------------------------------------------------------------------
# Variant -> backbone_type mapping (mirrors model_builder.py conventions)
# ---------------------------------------------------------------------------
VARIANT_TO_BACKBONE = {
    "mobileclip_s0": "MobileCLIP-S0",
    "mobileclip_s1": "MobileCLIP-S1",
    "mobileclip_l":  "MobileCLIP2-L",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--variant", required=True,
                    choices=list(VARIANT_TO_BACKBONE.keys()))
    ap.add_argument("--ckpt", required=True,
                    help="Path to the .pt checkpoint")
    ap.add_argument("--prompt", default="a person riding a bike")
    ap.add_argument("--ctx", type=int, default=16,
                    help="Context length (default 16, matches ctx16 checkpoints)")
    ap.add_argument("--out", required=True,
                    help="Output directory for .npy fixtures")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_type = VARIANT_TO_BACKBONE[args.variant]

    # BPE vocabulary is shipped alongside the checkpoint directory.
    bpe_path = str(REPO_ROOT / "models" / "bpe_simple_vocab_16e6.txt.gz")
    if not Path(bpe_path).exists():
        # Fall back to the asset bundled with the reference package.
        bpe_path = str(EFFICIENTSAM3_ROOT / "sam3" / "assets"
                       / "bpe_simple_vocab_16e6.txt.gz")

    print(f"[dump] variant={args.variant}  backbone_type={backbone_type}",
          file=sys.stderr)
    print(f"[dump] ckpt={args.ckpt}", file=sys.stderr)
    print(f"[dump] bpe_path={bpe_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build model with correct config via the reference factory function.
    #
    # pos_embed_table_size must match the checkpoint's stored shape (77),
    # not args.ctx.  After loading weights we call set_context_length() to
    # slice the positional embedding down to ctx=16 — exactly the same
    # policy as _apply_text_context_policy in model_builder.py.
    # ------------------------------------------------------------------
    encoder: TextStudentEncoder = _create_student_text_encoder(
        bpe_path=bpe_path,
        backbone_type=backbone_type,
        context_length=args.ctx,
        pos_embed_table_size=77,       # checkpoint always stores ctx=77
    )

    # ------------------------------------------------------------------
    # Load weights from checkpoint, stripping the full module prefix.
    # Key prefix in .pt: detector.backbone.language_backbone.*
    # ------------------------------------------------------------------
    print(f"[dump] Loading checkpoint …", file=sys.stderr)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    PREFIX = "detector.backbone.language_backbone."
    text_sd = {
        k[len(PREFIX):]: v
        for k, v in ckpt.items()
        if k.startswith(PREFIX)
    }
    if not text_sd:
        print("error: no keys matched prefix "
              f"'{PREFIX}' — check the checkpoint", file=sys.stderr)
        return 1

    missing, unexpected = encoder.load_state_dict(text_sd, strict=False)
    if missing:
        print(f"[dump] warn: missing keys ({len(missing)}): {missing[:5]}",
              file=sys.stderr)
    if unexpected:
        print(f"[dump] warn: unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}", file=sys.stderr)

    # Slice positional embeddings from ctx=77 down to the requested ctx.
    encoder.set_context_length(args.ctx)
    encoder.eval()

    # ------------------------------------------------------------------
    # Tokenize the prompt independently so we can save raw token ids.
    # encoder.tokenizer is the SimpleTokenizer attached at construction.
    # ------------------------------------------------------------------
    tokenizer = encoder.tokenizer
    # tokenizer returns [1, ctx_len] (always batched); squeeze to [ctx_len].
    tokens = tokenizer(args.prompt, context_length=args.ctx)
    if tokens.dim() == 2 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)  # [ctx_len]
    np.save(out_dir / "tokens.npy",
            tokens.numpy().astype(np.int32))
    print(f"[dump] tokens shape={list(tokens.shape)}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Register forward hooks on each transformer block and final LN.
    # encoder.encoder is the MobileCLIPTextTransformer.
    # encoder.encoder.transformer is a ModuleList of blocks.
    # ------------------------------------------------------------------
    activations: dict = {}

    def make_hook(name: str):
        def hook(_mod, _inputs, output):
            x = output if isinstance(output, torch.Tensor) else output[0]
            activations[name] = x.detach().cpu().float().numpy()
        return hook

    mc_transformer = encoder.encoder  # MobileCLIPTextTransformer
    for i, blk in enumerate(mc_transformer.transformer):
        blk.register_forward_hook(make_hook(f"block_{i:02d}_out"))
    mc_transformer.final_layer_norm.register_forward_hook(
        make_hook("final_ln_out")
    )

    # ------------------------------------------------------------------
    # Forward pass.  TextStudentEncoder.forward expects a raw string and
    # tokenizes internally; we pass the prompt and capture outputs.
    # Returns: (text_attention_mask, text_memory[seq,B,dim],
    #           input_embeds[seq,B,dim])
    # ------------------------------------------------------------------
    with torch.no_grad():
        _, text_memory_T, _ = encoder(
            args.prompt, device=torch.device("cpu")
        )
    # text_memory_T shape: [ctx_len, 1, 256]  ->  out_tokens: [ctx_len, 256]
    out_tokens = text_memory_T[:, 0, :].cpu().float().numpy()

    # Pooled embedding at the EOT (highest token id) position.
    eot_idx = int(tokens.argmax(dim=-1).item())
    pooled = out_tokens[eot_idx]  # [256]

    np.save(out_dir / "out_tokens.npy", out_tokens)
    np.save(out_dir / "pooled.npy", pooled)

    for name, arr in activations.items():
        # Hook output shape is [1, ctx_len, dim] — drop the batch dim.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        np.save(out_dir / f"{name}.npy", arr)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    n_blocks = len(mc_transformer.transformer)
    print(f"[dump] n_transformer_blocks={n_blocks}", file=sys.stderr)
    print(f"[dump] out_tokens shape={list(out_tokens.shape)}", file=sys.stderr)
    print(f"[dump] pooled shape={list(pooled.shape)}", file=sys.stderr)
    print(f"[dump] eot_idx={eot_idx}", file=sys.stderr)
    print(f"[dump] activation files: {sorted(activations.keys())}",
          file=sys.stderr)
    print(f"[dump] Wrote fixtures to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
