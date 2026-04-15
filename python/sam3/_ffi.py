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
""")
