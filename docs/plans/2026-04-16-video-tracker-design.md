# SAM3 Video Tracker Design

## Overview

Add video object tracking to the SAM3 C inference engine. Implements the
`Sam3TrackerPredictor` from the reference Python implementation — a memory-based
tracker that propagates masks across video frames using cross-attention to a bank
of past frame features.

## Scope

- **In scope:** Tracker module, memory encoder, memory attention (RoPE
  cross-attention), memory bank with SAM2Long selection, multi-object tracking,
  video file loading (bundled mini decoder + frame directory), public C API.
- **Out of scope:** Full video inference orchestrator
  (`Sam3VideoInference` / `Sam3VideoInferenceWithInstanceInteractivity`),
  automatic detection, temporal disambiguation, hotstart logic, multi-GPU.

## Architecture

```
Frame N pixels
    |
    v
+-------------------+     Reused: existing sam3_image_model backbone
| Image Encoder     | <-- ViT + Neck (encode per-frame, cache features)
| (ViT + Neck)      |
+--------+----------+
         | backbone features [72x72, 256]
         v
+-------------------+     NEW: 4-layer RoPE cross-attention
| Memory Attention  | <-- between current features and memory bank
| (4-layer xformer) |
+--------+----------+
         | memory-conditioned features
         v
+-------------------+     Reused: SAM prompt encoder + mask decoder
| SAM Prompt Enc    | <-- from existing mask_decoder module
| + Mask Decoder    |
+--------+----------+
         | mask logits + obj pointers
         v
+-------------------+     NEW: encode mask + pixel features
| Memory Encoder    |     into memory tokens
| (ConvNeXt fuser)  |
+--------+----------+
         | memory tokens + obj pointers
         v
+-------------------+     NEW: FIFO ring buffer of N frame
| Memory Bank       |     memories with temporal pos encoding
| (ring buffer)     |     + SAM2Long selection
+---------+---------+
```

The image backbone is run per-frame during tracking (matching
`forward_backbone_per_frame_for_eval=True`). All objects share backbone
features; masks are stacked per-object through the memory encoder.

## Public API

New functions in `sam3.h`:

```c
typedef struct sam3_video_session sam3_video_session;

/* Session lifecycle */
enum sam3_error sam3_video_start(sam3_ctx *ctx,
                                const char *resource_path,
                                sam3_video_session **out_session);
void sam3_video_end(sam3_video_session *session);
enum sam3_error sam3_video_reset(sam3_video_session *session);

/* Prompting on specific frames */
enum sam3_error sam3_video_add_points(sam3_video_session *session,
                                     int frame_idx, int obj_id,
                                     const struct sam3_point *points,
                                     int n_points,
                                     struct sam3_result *result);
enum sam3_error sam3_video_add_box(sam3_video_session *session,
                                  int frame_idx, int obj_id,
                                  const struct sam3_box *box,
                                  struct sam3_result *result);

/* Propagation with callback */
typedef int (*sam3_video_frame_cb)(int frame_idx,
                                  const struct sam3_result *result,
                                  int n_objects,
                                  const int *obj_ids,
                                  void *user_data);
enum sam3_error sam3_video_propagate(sam3_video_session *session,
                                    int direction,
                                    sam3_video_frame_cb callback,
                                    void *user_data);

/* Object management */
enum sam3_error sam3_video_remove_object(sam3_video_session *session,
                                        int obj_id);
int sam3_video_frame_count(const sam3_video_session *session);
```

Direction constants: `SAM3_PROPAGATE_BOTH=0`, `SAM3_PROPAGATE_FORWARD=1`,
`SAM3_PROPAGATE_BACKWARD=2`.

Callback returns non-zero to stop propagation early.

## New Internal Components

### Memory Encoder (`src/model/memory_encoder.h/.c`)

Matches Python `SimpleMaskEncoder` from `memory.py`:

1. **Mask Downsampler**: 2-layer Conv2d cascade (kernel=3, stride=2, pad=1)
   with LayerNorm2d + GELU. Downsamples masks to 72x72.
   - Optional interpolation to `interpol_size=[1152, 1152]` before conv layers
2. **Pixel Feature Projection**: 1x1 conv on backbone features
3. **Fuse**: Add downsampled mask + projected features, run 2-layer ConvNeXt
   (CXBlock: depthwise conv-7x7, LayerNorm, Linear, GELU, Linear, layer_scale,
   residual)
4. **Output Projection**: 1x1 conv (256 -> 64)
5. **Position Encoding**: Sinusoidal 2D (`num_pos_feats=64`)

Weight prefix: `tracker_model.maskmem_backbone.*`

### Memory Attention (`src/model/memory_attn.h/.c`)

Fill existing stub. 4-layer transformer encoder with:

- **Self-attention**: RoPE attention (d=256, 1 head, dropout=0.1,
  theta=10000, feat_sizes=[72,72])
- **Cross-attention**: RoPE attention (d=256, 1 head, kv_dim=64,
  rope_k_repeat=True)
- Pre-norm LayerNorm, FFN dim=2048
- `TransformerDecoderLayerv2` equivalent: cross_attention_first=False

Weight prefix: `tracker_model.transformer.encoder.*`

### Memory Bank (`src/model/memory_bank.h/.c`)

Ring buffer holding recent frame memories:

```c
struct sam3_memory_entry {
    struct sam3_tensor *spatial_features; /* [HW, mem_dim] */
    struct sam3_tensor *obj_pointers;     /* [n_obj, hidden_dim] */
    int frame_idx;
    int is_conditioning;
    float obj_score; /* for SAM2Long selection */
};

struct sam3_memory_bank {
    struct sam3_memory_entry entries[SAM3_MAX_MEMORY_FRAMES];
    int count;
    int capacity; /* default 7 */
    int temporal_stride; /* default 1 */
    float mf_threshold; /* 0.01 for SAM2Long selection */
};
```

- Conditioning frames: stored separately, max `max_cond_frames=4`
- Non-conditioning: ring buffer, evicts oldest when full
- Temporal stride: when stride > 1, keep every r-th frame + most recent
- SAM2Long selection: skip frames where all objects have score < threshold

### Tracker Core (`src/model/tracker.h/.c`)

```c
struct sam3_tracker {
    struct sam3_mask_decoder sam_decoder;
    struct sam3_memory_encoder mem_encoder;
    struct sam3_memory_attn mem_attention;
    struct sam3_memory_bank mem_bank;

    /* Learned parameters */
    struct sam3_tensor *maskmem_tpos_enc;  /* [7, 1, 1, 64] */
    struct sam3_tensor *no_mem_embed;      /* [1, 1, 256] */
    struct sam3_tensor *no_mem_pos_enc;    /* [1, 1, 256] */
    struct sam3_tensor *no_obj_ptr;        /* [1, 256] */
    struct sam3_tensor *no_obj_embed_spatial; /* [1, 64] */
    struct sam3_tensor *mask_downsample_w; /* [1, 1, 4, 4] conv */

    /* Config matching Sam3TrackerBase */
    int num_maskmem;           /* 7 */
    int max_cond_frames;       /* 4 */
    int image_size;            /* 1008 */
    int backbone_stride;       /* 14 */
    int max_obj_ptrs;          /* 16 */
    float sigmoid_scale;       /* 20.0 */
    float sigmoid_bias;        /* -10.0 */
    float mf_threshold;        /* 0.01 */

    int multimask_output;      /* 1 */
    int multimask_min_pt_num;  /* 0 */
    int multimask_max_pt_num;  /* 1 */
};
```

Per-frame tracking workflow:
1. Get backbone features (from image encoder, cached per frame)
2. Select closest conditioning frames from memory bank
3. Assemble memory tokens (spatial + temporal pos enc + obj pointers)
4. Cross-attend current features to memory via memory_attn
5. Run SAM mask decoder with conditioned features + prompts
6. Encode output masks into memory via memory_encoder
7. Store result in memory bank (with SAM2Long selection)

### Video I/O (`src/util/video.h/.c`)

- Bundle `pl_mpeg` (MIT, MPEG1/2 decoder) for video files
- Support frame directories (JPEG/PNG via stb_image)
- Auto-detect: file extension check
- Frame loading:
  - Decode to RGB uint8
  - Resize to model input size
  - Normalize to [-1, 1] with mean=0.5, std=0.5
  - Store as F16 tensors (matching reference)

### Video Session (`src/model/video_session.h/.c`)

Internal session state:

```c
struct sam3_video_session {
    sam3_ctx *ctx;
    struct sam3_tracker tracker;

    /* Frame storage */
    struct sam3_tensor **frames;  /* preprocessed frame tensors */
    int num_frames;
    int orig_width, orig_height;

    /* Object tracking */
    int obj_ids[SAM3_MAX_OBJECTS];
    int n_objects;

    /* Per-frame state */
    struct sam3_tensor **cached_features; /* backbone features per frame */
    int *frames_already_tracked;

    /* Per-object per-frame prompts */
    struct { /* point_inputs[obj_idx][frame_idx] */ } point_inputs;
    struct { /* mask_inputs[obj_idx][frame_idx] */ } mask_inputs;

    /* Output storage */
    struct {
        struct sam3_tensor *cond_outputs;     /* conditioning frame outputs */
        struct sam3_tensor *non_cond_outputs; /* propagated frame outputs */
    } output_dict;

    /* Arenas */
    struct sam3_arena persist; /* session-lifetime: frames, features, memory */
    struct sam3_arena scratch; /* per-frame: reset between frames */
};
```

## Error Handling

- New error code: `SAM3_EVIDEO = -7` (frame OOB, invalid session, etc.)
- Object limit: 64 simultaneous objects, `SAM3_EINVAL` if exceeded
- Video I/O: `SAM3_EIO` for corrupt/unsupported files
- All allocations through arena (persist for memory bank, scratch per-frame)

## Weight Loading

Tracker weights live in the same `.sam3` weight file with prefix
`tracker_model.*`:

- `tracker_model.maskmem_backbone.*` -> memory encoder
- `tracker_model.transformer.*` -> memory attention
- `tracker_model.maskmem_tpos_enc` -> temporal position encoding
- `tracker_model.no_mem_embed`, `no_mem_pos_enc` -> no-memory tokens
- `tracker_model.no_obj_ptr` -> no-object pointer
- `tracker_model.no_obj_embed_spatial` -> no-object spatial embed
- `tracker_model.mask_downsample.*` -> GT mask downsample conv
- `tracker_model.sam_prompt_encoder.*` -> SAM prompt encoder
- `tracker_model.sam_mask_decoder.*` -> SAM mask decoder

The weight converter (`sam3_convert`) must be updated to include these
tensors when converting from PyTorch safetensors.

## Testing

| Test File | What it validates |
|-----------|-------------------|
| `tests/test_memory_encoder.c` | Mask downsampler shapes, fuser output, full encode vs reference |
| `tests/test_memory_attn.c` | RoPE frequencies, single-layer, 4-layer vs Python dump |
| `tests/test_memory_bank.c` | Ring buffer insert/evict, temporal stride, SAM2Long filter |
| `tests/test_video_io.c` | Frame directory loading, pl_mpeg decode, normalization |
| `tests/test_tracker.c` | Single-frame track, 2-frame propagation, multi-object |
| `tests/test_video_session.c` | API lifecycle: start/add_points/propagate/end |

Reference fixtures: tensor dumps from Python reference for numerical
comparison at each component boundary.

## Reference Fixture List

Fixtures to generate from the Python reference implementation for
numerical validation. Each fixture is a binary `.bin` file containing
raw tensor data with a small header (shape + dtype).

### Memory Encoder Fixtures
1. `mem_enc_mask_input.bin` - Input mask [1, 1, H, W] float32
2. `mem_enc_pix_feat.bin` - Input pixel features [1, 256, 72, 72] float32
3. `mem_enc_mask_downsampled.bin` - After mask downsampler [1, 256, 72, 72]
4. `mem_enc_fused.bin` - After CXBlock fuser [1, 256, 72, 72]
5. `mem_enc_vision_features.bin` - Final output [1, 64, 72, 72]
6. `mem_enc_vision_pos_enc.bin` - Position encoding [1, 64, 72, 72]

### Memory Attention Fixtures
7. `mem_attn_current_features.bin` - Current frame features [5184, 256]
8. `mem_attn_memory_tokens.bin` - Memory bank tokens [N*5184, 64]
9. `mem_attn_memory_pos_enc.bin` - Memory position encoding [N*5184, 64]
10. `mem_attn_output.bin` - Cross-attended output [5184, 256]

### SAM Mask Decoder Fixtures (tracker context)
11. `tracker_sam_image_embed.bin` - Conditioned image embedding [1, 256, 72, 72]
12. `tracker_sam_point_coords.bin` - Point prompt coordinates [1, N, 2]
13. `tracker_sam_point_labels.bin` - Point prompt labels [1, N]
14. `tracker_sam_mask_logits.bin` - Output mask logits [1, M, 288, 288]
15. `tracker_sam_obj_ptr.bin` - Object pointer token [1, 256]

### Tracker Integration Fixtures
16. `tracker_frame0_backbone.bin` - Frame 0 backbone features [72, 72, 256]
17. `tracker_frame0_masks.bin` - Frame 0 output masks (after prompting)
18. `tracker_frame1_backbone.bin` - Frame 1 backbone features
19. `tracker_frame1_masks.bin` - Frame 1 output masks (after propagation)
20. `tracker_frame1_memory_bank.bin` - Memory bank state after frame 1

### Memory Selection Fixtures
21. `mem_select_obj_scores.bin` - Object existence scores per frame
22. `mem_select_filtered_indices.bin` - Which frames pass threshold

## Build Integration

- New cmake option: `SAM3_VIDEO` (default ON)
- New source files added to `SAM3_MODEL_SOURCES` glob
- `pl_mpeg.h` goes in `src/util/vendor/` alongside stb headers
- Tests gated behind `SAM3_TESTS AND SAM3_VIDEO`
