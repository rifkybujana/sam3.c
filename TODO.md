# EfficientSAM3 Inference TODO

Track what's needed to run EfficientSAM3 (EfficientViT-B2 backbone) end-to-end.

## Done

- [x] Python export script (`tools/export_efficientsam3.py`)
- [x] EfficientViT weight rename handler (`tools/weight_rename.c`)
- [x] sam2_convs FPN neck rename handler
- [x] `--backbone efficientvit` CLI flag in both converters
- [x] `backbone_type` in `sam3_model_config` and `.sam3` header
- [x] Text encoder, decoder, geometry encoder (backbone-independent)
- [x] CLI `sam3 segment -t` text prompt support

## Weight Preparation

- [x] Export: `python tools/export_efficientsam3.py models/efficient_sam3_efficientvit_l.pt models/efficient_sam3_efficientvit_l.safetensors` (1407 tensors)
- [x] Convert: `sam3_cli convert -i ...safetensors -o models/efficient.sam3 --backbone efficientvit` (1586 tensors after rename/split, 1.67 GB)
      Note: `num_batches_tracked` (I64) tensors skipped — training-only, not needed for inference

## BatchNorm Op

EfficientViT uses BatchNorm (not LayerNorm like Hiera).

- [x] Add `SAM3_OP_BATCHNORM` to `enum sam3_op` in `src/core/graph.h`
- [x] CPU kernel `src/backend/cpu/kernels/cpu_batchnorm.c` (scalar/NEON/AVX2)
- [x] Metal dispatch via MLX-C (`rsqrt`+arithmetic) in `metal_backend.c`
- [x] Backend dispatch for new op in CPU backend
- [x] `gh_batchnorm()` graph helper

## Grouped / Depthwise Convolution

MBConv blocks use depthwise conv (`groups=C`). Current conv2d has no groups param.

- [ ] Extend `SAM3_OP_CONV2D` params to carry `groups` count
- [ ] Update `src/backend/cpu/kernels/cpu_conv2d.c` for grouped conv
- [ ] Update Metal conv2d kernel for grouped conv

## EfficientViT Encoder Graph Builder

The entire image encoder is missing (~500-800 LOC).

- [ ] `src/model/image_encoder_efficientvit.h` — struct + API
- [ ] `src/model/image_encoder_efficientvit.c` — graph construction:
  - [ ] Input stem (ConvLayer + DSConv residual blocks)
  - [ ] MBConv blocks (inverted_conv -> depth_conv -> point_conv + skip)
  - [ ] LiteMLA attention blocks (qkv -> aggreg -> proj in context_module)
  - [ ] Projection head (1x1 conv -> BN -> 3x3 conv)
- [ ] Register in CMakeLists.txt

## Backbone Type Dispatch

Model builder is hardcoded to Hiera. Need runtime backbone routing.

- [ ] `src/model/vl_combiner.c`: check `backbone_type` in init/load/build
- [ ] `struct sam3_vl_backbone`: support both encoder types (union or separate fields)
- [ ] Propagate `backbone_type` through `sam3_processor_load` -> `sam3_image_model_load` -> `sam3_vl_backbone_load`

## FPN Neck Dimension Fix

`vl_combiner.c` hardcodes `backbone_dim=1024` (Hiera). EfficientViT-B2 outputs 384.

- [ ] Make `backbone_dim` configurable per backbone type in `sam3_vl_backbone_init`
- [ ] Verify sam2_fpn_layers tensors are loaded when present

## Tests

- [ ] `tests/test_efficientvit.c` — encoder graph build + forward
- [ ] `tests/test_batchnorm.c` — BN kernel correctness
- [ ] `tests/test_grouped_conv.c` — depthwise / grouped conv correctness
- [ ] Integration test: full EfficientSAM3 inference vs Python reference
