"""
cffi type declarations for the sam3 C API.

Manually maintained subset of include/sam3/sam3.h and sam3_types.h.
Update this file when the C API changes.
"""
from sam3._lib import ffi

ffi.cdef("""
    /* Error codes */
    enum sam3_error {
        SAM3_OK       =  0,
        SAM3_EINVAL   = -1,
        SAM3_ENOMEM   = -2,
        SAM3_EIO      = -3,
        SAM3_EBACKEND = -4,
        SAM3_EMODEL   = -5,
        SAM3_EDTYPE   = -6,
        SAM3_EVIDEO   = -7,
    };

    /* Prompt types */
    enum sam3_prompt_type {
        SAM3_PROMPT_POINT = 0,
        SAM3_PROMPT_BOX   = 1,
        SAM3_PROMPT_MASK  = 2,
        SAM3_PROMPT_TEXT  = 3,
    };

    /* Log levels */
    enum sam3_log_level {
        SAM3_LOG_DEBUG = 0,
        SAM3_LOG_INFO  = 1,
        SAM3_LOG_WARN  = 2,
        SAM3_LOG_ERROR = 3,
    };

    /* Structs */
    struct sam3_point {
        float x;
        float y;
        int   label;
    };

    struct sam3_box {
        float x1;
        float y1;
        float x2;
        float y2;
    };

    struct sam3_prompt {
        enum sam3_prompt_type type;
        union {
            struct sam3_point point;
            struct sam3_box   box;
            struct {
                const float *data;
                int          width;
                int          height;
            } mask;
            const char *text;
        };
    };

    struct sam3_result {
        float *masks;
        float *iou_scores;
        int    n_masks;
        int    mask_height;
        int    mask_width;
        int    iou_valid;
        float *boxes;
        int    boxes_valid;
        int    best_mask;
    };

    /* Opaque context */
    typedef struct sam3_ctx sam3_ctx;

    /* Lifecycle */
    sam3_ctx *sam3_init(void);
    void sam3_free(sam3_ctx *ctx);

    /* Model loading */
    enum sam3_error sam3_load_model(sam3_ctx *ctx, const char *path);
    enum sam3_error sam3_load_bpe(sam3_ctx *ctx, const char *path);

    /* Image input */
    enum sam3_error sam3_set_image(sam3_ctx *ctx, const uint8_t *pixels,
                                   int width, int height);
    enum sam3_error sam3_set_image_file(sam3_ctx *ctx, const char *path);
    void sam3_set_prompt_space(sam3_ctx *ctx, int width, int height);

    /* Text prompt */
    enum sam3_error sam3_set_text(sam3_ctx *ctx, const char *text);

    /* Inference */
    enum sam3_error sam3_segment(sam3_ctx *ctx,
                                 const struct sam3_prompt *prompts,
                                 int n_prompts,
                                 struct sam3_result *result);
    void sam3_result_free(struct sam3_result *result);

    /* Queries */
    int sam3_get_image_size(const sam3_ctx *ctx);
    const char *sam3_version(void);

    /* Utilities */
    const char *sam3_error_str(enum sam3_error err);
    void sam3_log_set_level(enum sam3_log_level level);

    /* Profiling */
    enum sam3_error sam3_profile_enable(sam3_ctx *ctx);
    void sam3_profile_disable(sam3_ctx *ctx);
    void sam3_profile_report(sam3_ctx *ctx);
    void sam3_profile_reset(sam3_ctx *ctx);

    /* Debug */
    enum sam3_error sam3_dump_tensors(sam3_ctx *ctx, const char *out_dir);

    /* Video tracking */
    enum sam3_propagate_dir {
        SAM3_PROPAGATE_BOTH     = 0,
        SAM3_PROPAGATE_FORWARD  = 1,
        SAM3_PROPAGATE_BACKWARD = 2,
    };

    typedef struct sam3_video_session sam3_video_session;

    typedef int (*sam3_video_frame_cb)(int frame_idx,
                                       const struct sam3_result *result,
                                       int n_objects,
                                       const int *obj_ids,
                                       void *user_data);

    enum sam3_error sam3_video_start(sam3_ctx *ctx,
                                     const char *resource_path,
                                     sam3_video_session **out_session);
    enum sam3_error sam3_video_add_points(sam3_video_session *session,
                                          int frame_idx, int obj_id,
                                          const struct sam3_point *points,
                                          int n_points,
                                          struct sam3_result *result);
    enum sam3_error sam3_video_add_box(sam3_video_session *session,
                                       int frame_idx, int obj_id,
                                       const struct sam3_box *box,
                                       struct sam3_result *result);
    enum sam3_error sam3_video_propagate(sam3_video_session *session,
                                         int direction,
                                         sam3_video_frame_cb callback,
                                         void *user_data);
    enum sam3_error sam3_video_remove_object(sam3_video_session *session,
                                             int obj_id);
    enum sam3_error sam3_video_reset(sam3_video_session *session);
    void sam3_video_end(sam3_video_session *session);
    int sam3_video_frame_count(const sam3_video_session *session);
""")
