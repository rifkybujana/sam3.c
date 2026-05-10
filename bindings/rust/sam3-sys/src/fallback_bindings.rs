// Fallback bindings used when bindgen cannot run locally (for example,
// a Windows MSVC machine without libclang.dll installed). Keep this file in
// sync with include/sam3/sam3.h and include/sam3/sam3_types.h.

use std::os::raw::{c_char, c_float, c_int, c_uchar, c_uint, c_void};

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct sam3_error(pub c_int);

impl sam3_error {
    pub const SAM3_OK: sam3_error = sam3_error(0);
    pub const SAM3_EINVAL: sam3_error = sam3_error(-1);
    pub const SAM3_ENOMEM: sam3_error = sam3_error(-2);
    pub const SAM3_EIO: sam3_error = sam3_error(-3);
    pub const SAM3_EBACKEND: sam3_error = sam3_error(-4);
    pub const SAM3_EMODEL: sam3_error = sam3_error(-5);
    pub const SAM3_EDTYPE: sam3_error = sam3_error(-6);
    pub const SAM3_EVIDEO: sam3_error = sam3_error(-7);
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum sam3_log_level {
    SAM3_LOG_DEBUG = 0,
    SAM3_LOG_INFO = 1,
    SAM3_LOG_WARN = 2,
    SAM3_LOG_ERROR = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum sam3_dtype {
    SAM3_DTYPE_F32 = 0,
    SAM3_DTYPE_F16 = 1,
    SAM3_DTYPE_BF16 = 2,
    SAM3_DTYPE_I32 = 3,
    SAM3_DTYPE_I8 = 4,
    SAM3_DTYPE_Q8_0 = 5,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum sam3_prompt_type {
    SAM3_PROMPT_POINT = 0,
    SAM3_PROMPT_BOX = 1,
    SAM3_PROMPT_MASK = 2,
    SAM3_PROMPT_TEXT = 3,
}

pub const SAM3_PROPAGATE_BOTH: c_uint = 0;
pub const SAM3_PROPAGATE_FORWARD: c_uint = 1;
pub const SAM3_PROPAGATE_BACKWARD: c_uint = 2;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_ctx {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_video_session {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_cache_opts {
    pub n_image_slots: c_int,
    pub n_text_slots: c_int,
    pub image_mem_budget_bytes: usize,
    pub image_spill_dir: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct sam3_cache_stats {
    pub image_hits: u64,
    pub image_misses: u64,
    pub image_evictions: u64,
    pub text_hits: u64,
    pub text_misses: u64,
    pub text_evictions: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct sam3_point {
    pub x: c_float,
    pub y: c_float,
    pub label: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct sam3_box {
    pub x1: c_float,
    pub y1: c_float,
    pub x2: c_float,
    pub y2: c_float,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct sam3_prompt__bindgen_ty_1__bindgen_ty_1 {
    pub data: *const c_float,
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union sam3_prompt__bindgen_ty_1 {
    pub point: sam3_point,
    pub box_: sam3_box,
    pub mask: sam3_prompt__bindgen_ty_1__bindgen_ty_1,
    pub text: *const c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct sam3_prompt {
    pub type_: sam3_prompt_type,
    pub __bindgen_anon_1: sam3_prompt__bindgen_ty_1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_prompt_set {
    pub prompts: *const sam3_prompt,
    pub n_prompts: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_result {
    pub masks: *mut c_float,
    pub iou_scores: *mut c_float,
    pub n_masks: c_int,
    pub mask_height: c_int,
    pub mask_width: c_int,
    pub iou_valid: c_int,
    pub boxes: *mut c_float,
    pub boxes_valid: c_int,
    pub best_mask: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_video_object_mask {
    pub obj_id: c_int,
    pub mask: *mut c_float,
    pub mask_h: c_int,
    pub mask_w: c_int,
    pub iou_score: c_float,
    pub obj_score_logit: c_float,
    pub is_occluded: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct sam3_video_frame_result {
    pub frame_idx: c_int,
    pub n_objects: c_int,
    pub objects: *mut sam3_video_object_mask,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct sam3_video_start_opts {
    pub frame_cache_backend_budget: usize,
    pub frame_cache_spill_budget: usize,
    pub clear_non_cond_window: c_int,
    pub iter_use_prev_mask_pred: c_int,
    pub multimask_via_stability: c_int,
    pub multimask_stability_delta: c_float,
    pub multimask_stability_thresh: c_float,
}

pub type sam3_video_frame_cb = Option<
    unsafe extern "C" fn(result: *const sam3_video_frame_result, user_data: *mut c_void) -> c_int,
>;

extern "C" {
    pub fn sam3_init() -> *mut sam3_ctx;
    pub fn sam3_init_ex(opts: *const sam3_cache_opts) -> *mut sam3_ctx;
    pub fn sam3_free(ctx: *mut sam3_ctx);
    pub fn sam3_load_model(ctx: *mut sam3_ctx, path: *const c_char) -> sam3_error;
    pub fn sam3_load_bpe(ctx: *mut sam3_ctx, path: *const c_char) -> sam3_error;
    pub fn sam3_set_image(
        ctx: *mut sam3_ctx,
        pixels: *const c_uchar,
        width: c_int,
        height: c_int,
    ) -> sam3_error;
    pub fn sam3_set_image_file(ctx: *mut sam3_ctx, path: *const c_char) -> sam3_error;
    pub fn sam3_set_prompt_space(ctx: *mut sam3_ctx, width: c_int, height: c_int);
    pub fn sam3_set_text(ctx: *mut sam3_ctx, text: *const c_char) -> sam3_error;
    pub fn sam3_precache_image(
        ctx: *mut sam3_ctx,
        pixels: *const c_uchar,
        width: c_int,
        height: c_int,
    ) -> sam3_error;
    pub fn sam3_cache_save_image(
        ctx: *mut sam3_ctx,
        pixels: *const c_uchar,
        width: c_int,
        height: c_int,
        path: *const c_char,
    ) -> sam3_error;
    pub fn sam3_cache_load_image(ctx: *mut sam3_ctx, path: *const c_char) -> sam3_error;
    pub fn sam3_cache_clear(ctx: *mut sam3_ctx, which: c_uint);
    pub fn sam3_cache_stats(ctx: *const sam3_ctx, out: *mut sam3_cache_stats);
    pub fn sam3_segment(
        ctx: *mut sam3_ctx,
        prompts: *const sam3_prompt,
        n_prompts: c_int,
        result: *mut sam3_result,
    ) -> sam3_error;
    pub fn sam3_result_free(result: *mut sam3_result);
    pub fn sam3_get_image_size(ctx: *const sam3_ctx) -> c_int;
    pub fn sam3_version() -> *const c_char;
    pub fn sam3_error_str(err: sam3_error) -> *const c_char;
    pub fn sam3_log_set_level(level: sam3_log_level);

    pub fn sam3_mask_nms(
        masks: *const c_float,
        scores: *const c_float,
        n_masks: c_int,
        h: c_int,
        w: c_int,
        prob_thresh: c_float,
        iou_thresh: c_float,
        min_quality: c_float,
        kept_out: *mut c_int,
    ) -> c_int;

    pub fn sam3_video_frame_result_free(result: *mut sam3_video_frame_result);
    pub fn sam3_video_start(
        ctx: *mut sam3_ctx,
        resource_path: *const c_char,
        out_session: *mut *mut sam3_video_session,
    ) -> sam3_error;
    pub fn sam3_video_start_ex(
        ctx: *mut sam3_ctx,
        resource_path: *const c_char,
        opts: *const sam3_video_start_opts,
        out_session: *mut *mut sam3_video_session,
    ) -> sam3_error;
    pub fn sam3_video_add_points(
        session: *mut sam3_video_session,
        frame_idx: c_int,
        obj_id: c_int,
        points: *const sam3_point,
        n_points: c_int,
        result: *mut sam3_video_frame_result,
    ) -> sam3_error;
    pub fn sam3_video_add_box(
        session: *mut sam3_video_session,
        frame_idx: c_int,
        obj_id: c_int,
        box_: *const sam3_box,
        result: *mut sam3_video_frame_result,
    ) -> sam3_error;
    pub fn sam3_video_add_mask(
        session: *mut sam3_video_session,
        frame_idx: c_int,
        obj_id: c_int,
        mask: *const c_uchar,
        mask_h: c_int,
        mask_w: c_int,
        result: *mut sam3_video_frame_result,
    ) -> sam3_error;
    pub fn sam3_video_propagate(
        session: *mut sam3_video_session,
        direction: c_int,
        callback: sam3_video_frame_cb,
        user_data: *mut c_void,
    ) -> sam3_error;
    pub fn sam3_video_remove_object(session: *mut sam3_video_session, obj_id: c_int) -> sam3_error;
    pub fn sam3_video_reset(session: *mut sam3_video_session) -> sam3_error;
    pub fn sam3_video_end(session: *mut sam3_video_session);
    pub fn sam3_video_frame_count(session: *const sam3_video_session) -> c_int;
}
