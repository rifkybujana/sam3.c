//! Video tracking session wrapping sam3's multi-object propagation API.
//!
//! A [`VideoSession`] borrows a loaded [`Ctx`] and wraps `sam3_video_*`
//! entry points. Per-frame output (from `add_points`, `add_box`,
//! `add_mask`, or the `propagate` callback) is copied out of the C
//! arena into owned [`FrameResult`]/[`ObjectMask`] values so consumers
//! never have to juggle engine-owned pointers.
//!
//! See `docs/video.md` in the upstream repo for a conceptual overview.

use std::ffi::CString;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;

use sam3_sys as sys;

use crate::ctx::Ctx;
use crate::error::{check, Error, Result};
use crate::prompt::{Box as PromptBox, Point};

/// Direction of propagation through the clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Propagate both forward and backward from each prompted frame.
    Both,
    /// Propagate forward only.
    Forward,
    /// Propagate backward only.
    Backward,
}

impl Direction {
    #[inline]
    fn to_raw(self) -> i32 {
        match self {
            Direction::Both => sys::SAM3_PROPAGATE_BOTH as i32,
            Direction::Forward => sys::SAM3_PROPAGATE_FORWARD as i32,
            Direction::Backward => sys::SAM3_PROPAGATE_BACKWARD as i32,
        }
    }
}

/// Tunables for [`VideoSession::start_with_opts`].
///
/// `None` on any field selects the libsam3 default. See
/// `include/sam3/sam3.h` for the exact defaults.
#[derive(Debug, Clone, Copy, Default)]
pub struct StartOpts {
    /// Resident frame-cache budget in bytes.
    pub frame_cache_backend_budget: Option<usize>,
    /// Spill-to-disk cache budget in bytes (`usize::MAX` disables spill).
    pub frame_cache_spill_budget: Option<usize>,
    /// Non-conditioning frame retention window.
    pub clear_non_cond_window: Option<i32>,
    /// Whether the next-iteration step uses the previous mask prediction.
    pub iter_use_prev_mask_pred: Option<bool>,
    /// Whether multimask selection uses stability filtering.
    pub multimask_via_stability: Option<bool>,
    /// Stability delta used when `multimask_via_stability` is on.
    pub multimask_stability_delta: Option<f32>,
    /// Stability acceptance threshold used when `multimask_via_stability` is on.
    pub multimask_stability_thresh: Option<f32>,
}

impl StartOpts {
    fn to_raw(self) -> sys::sam3_video_start_opts {
        sys::sam3_video_start_opts {
            frame_cache_backend_budget: self.frame_cache_backend_budget.unwrap_or(0),
            frame_cache_spill_budget: self.frame_cache_spill_budget.unwrap_or(0),
            clear_non_cond_window: self.clear_non_cond_window.unwrap_or(0),
            iter_use_prev_mask_pred: self
                .iter_use_prev_mask_pred
                .map(|b| if b { 1 } else { 0 })
                .unwrap_or(-1),
            multimask_via_stability: self
                .multimask_via_stability
                .map(|b| if b { 1 } else { 0 })
                .unwrap_or(-1),
            multimask_stability_delta: self.multimask_stability_delta.unwrap_or(0.0),
            multimask_stability_thresh: self.multimask_stability_thresh.unwrap_or(0.0),
        }
    }
}

/// Per-object segmentation output for one frame.
#[derive(Debug, Clone)]
pub struct ObjectMask {
    /// User-supplied object identifier.
    pub obj_id: i32,
    /// Mask logits, `mask_height * mask_width` f32s in row-major order.
    /// Threshold at `0.0` for a binary mask.
    pub mask: Vec<f32>,
    /// Mask height in pixels.
    pub mask_height: u32,
    /// Mask width in pixels.
    pub mask_width: u32,
    /// Predicted IoU score for this mask.
    pub iou_score: f32,
    /// Object presence logit; `> 0` visible, `<= 0` occluded.
    pub obj_score_logit: f32,
    /// Convenience: `obj_score_logit <= 0`.
    pub is_occluded: bool,
}

impl ObjectMask {
    /// Borrow the mask as a `(height, width)` slice.
    pub fn mask_slice(&self) -> &[f32] {
        &self.mask
    }

    /// Copy a single `sam3_video_object_mask` into an owned `ObjectMask`.
    fn from_raw(raw: &sys::sam3_video_object_mask) -> Result<Self> {
        let h = raw.mask_h.max(0) as usize;
        let w = raw.mask_w.max(0) as usize;
        let total = h.checked_mul(w).ok_or(Error::Invalid)?;
        let mask = if total == 0 {
            Vec::new()
        } else if raw.mask.is_null() {
            return Err(Error::Invalid);
        } else {
            // SAFETY: libsam3 guarantees raw.mask points to h*w f32s while
            // the enclosing frame_result is live.
            unsafe { std::slice::from_raw_parts(raw.mask, total) }.to_vec()
        };
        Ok(ObjectMask {
            obj_id: raw.obj_id,
            mask,
            mask_height: h as u32,
            mask_width: w as u32,
            iou_score: raw.iou_score,
            obj_score_logit: raw.obj_score_logit,
            is_occluded: raw.is_occluded != 0,
        })
    }
}

/// Multi-object segmentation result for one frame.
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Zero-based frame index.
    pub frame_idx: i32,
    /// Per-object masks for this frame.
    pub objects: Vec<ObjectMask>,
}

impl FrameResult {
    /// Return the mask for `obj_id`, if the frame carries one.
    pub fn by_obj_id(&self, obj_id: i32) -> Option<&ObjectMask> {
        self.objects.iter().find(|o| o.obj_id == obj_id)
    }

    fn from_raw(raw: &sys::sam3_video_frame_result) -> Result<Self> {
        let n = raw.n_objects.max(0) as usize;
        let objects = if n == 0 {
            Vec::new()
        } else if raw.objects.is_null() {
            return Err(Error::Invalid);
        } else {
            // SAFETY: libsam3 guarantees raw.objects points to n contiguous
            // sam3_video_object_mask entries while raw is live.
            let slice = unsafe { std::slice::from_raw_parts(raw.objects, n) };
            slice
                .iter()
                .map(ObjectMask::from_raw)
                .collect::<Result<Vec<_>>>()?
        };
        Ok(FrameResult {
            frame_idx: raw.frame_idx,
            objects,
        })
    }
}

/// RAII guard: always call sam3_video_frame_result_free on drop.
///
/// Used for prompt-entry-point results (`add_points`, `add_box`,
/// `add_mask`) whose object buffers are heap-allocated and must be
/// freed even on error / panic before we return.
struct FrameResultGuard(*mut sys::sam3_video_frame_result);

impl Drop for FrameResultGuard {
    fn drop(&mut self) {
        // SAFETY: the guarded struct was zeroed on construction and then
        // (possibly) populated by a sam3_video_* entry point; the free
        // function tolerates partially-filled results.
        unsafe { sys::sam3_video_frame_result_free(self.0) }
    }
}

/// Active video tracking session.
///
/// Lifetime is tied to the backing [`Ctx`]: a session cannot outlive the
/// context it was started from. `VideoSession` is `!Send` + `!Sync` for
/// the same reason [`Ctx`] is: libsam3 stores mutable state inside the
/// session and dereferences the context from background workers.
pub struct VideoSession<'ctx> {
    raw: NonNull<sys::sam3_video_session>,
    _ctx: PhantomData<&'ctx mut Ctx>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'ctx> VideoSession<'ctx> {
    /// Begin a video session over a file or frame directory.
    ///
    /// `resource_path` may be a video file (decoded via the configured
    /// backend) or a directory of `.png`/`.jpg` frames (loaded in sorted
    /// order).
    pub fn start<P: AsRef<Path>>(ctx: &'ctx mut Ctx, resource_path: P) -> Result<Self> {
        Self::start_impl(ctx, resource_path.as_ref(), None)
    }

    /// Begin a video session with explicit [`StartOpts`].
    pub fn start_with_opts<P: AsRef<Path>>(
        ctx: &'ctx mut Ctx,
        resource_path: P,
        opts: StartOpts,
    ) -> Result<Self> {
        Self::start_impl(ctx, resource_path.as_ref(), Some(opts))
    }

    fn start_impl(ctx: &'ctx mut Ctx, path: &Path, opts: Option<StartOpts>) -> Result<Self> {
        let c = path_to_cstring(path)?;
        let mut out: *mut sys::sam3_video_session = std::ptr::null_mut();
        // SAFETY: ctx.raw() is a valid sam3_ctx; &mut ctx guarantees no
        // concurrent use; c is a NUL-terminated path not retained past the
        // call; &mut out points to writable storage for a single pointer.
        let code = unsafe {
            match opts {
                None => sys::sam3_video_start(ctx.raw(), c.as_ptr(), &mut out),
                Some(o) => {
                    let raw_opts = o.to_raw();
                    sys::sam3_video_start_ex(ctx.raw(), c.as_ptr(), &raw_opts, &mut out)
                }
            }
        };
        check(code)?;
        let raw = NonNull::new(out).ok_or(Error::Invalid)?;
        Ok(VideoSession {
            raw,
            _ctx: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Return the number of frames decoded for this session.
    pub fn frame_count(&self) -> u32 {
        // SAFETY: self.raw is valid for the session's lifetime.
        let n = unsafe { sys::sam3_video_frame_count(self.raw.as_ptr()) };
        n.max(0) as u32
    }

    /// Add point prompts for `obj_id` on `frame_idx`.
    ///
    /// Returns the per-frame result containing the prompted object's
    /// mask (`objects.len() == 1`).
    pub fn add_points(
        &mut self,
        frame_idx: i32,
        obj_id: i32,
        points: &[Point],
    ) -> Result<FrameResult> {
        if points.is_empty() || points.len() > i32::MAX as usize {
            return Err(Error::Invalid);
        }
        let raw_points: Vec<sys::sam3_point> = points
            .iter()
            .map(|p| sys::sam3_point {
                x: p.x,
                y: p.y,
                label: p.label.to_raw(),
            })
            .collect();

        self.with_result(|session, result| {
            // SAFETY: session is non-null; raw_points lives for the call;
            // result is zeroed and writable.
            unsafe {
                sys::sam3_video_add_points(
                    session,
                    frame_idx,
                    obj_id,
                    raw_points.as_ptr(),
                    raw_points.len() as i32,
                    result,
                )
            }
        })
    }

    /// Add a bounding box prompt for `obj_id` on `frame_idx`.
    pub fn add_box(&mut self, frame_idx: i32, obj_id: i32, b: PromptBox) -> Result<FrameResult> {
        let raw = sys::sam3_box {
            x1: b.x1,
            y1: b.y1,
            x2: b.x2,
            y2: b.y2,
        };
        self.with_result(|session, result| {
            // SAFETY: session is non-null; raw is live for the call;
            // result is zeroed and writable.
            unsafe { sys::sam3_video_add_box(session, frame_idx, obj_id, &raw, result) }
        })
    }

    /// Add a binary mask prompt for `obj_id` on `frame_idx`.
    ///
    /// `mask` is row-major `height * width` bytes; any non-zero pixel is
    /// treated as foreground. The mask is resized by the engine to the
    /// session's internal high-res size (1152×1152) via nearest neighbor.
    pub fn add_mask(
        &mut self,
        frame_idx: i32,
        obj_id: i32,
        mask: &[u8],
        height: u32,
        width: u32,
    ) -> Result<FrameResult> {
        let need = (height as usize)
            .checked_mul(width as usize)
            .ok_or(Error::Invalid)?;
        if need == 0 || mask.len() < need {
            return Err(Error::Invalid);
        }
        self.with_result(|session, result| {
            // SAFETY: session is non-null; mask has at least height*width
            // bytes (checked above); dimensions fit in i32 per libsam3
            // contract (bounds checked by the engine).
            unsafe {
                sys::sam3_video_add_mask(
                    session,
                    frame_idx,
                    obj_id,
                    mask.as_ptr(),
                    height as i32,
                    width as i32,
                    result,
                )
            }
        })
    }

    /// Remove `obj_id` from the tracked object set.
    pub fn remove_object(&mut self, obj_id: i32) -> Result<()> {
        // SAFETY: self.raw is a valid session pointer; no concurrent use
        // via &mut self.
        unsafe { check(sys::sam3_video_remove_object(self.raw.as_ptr(), obj_id)) }
    }

    /// Clear all tracked objects and prompts (keep encoded features).
    pub fn reset(&mut self) -> Result<()> {
        // SAFETY: self.raw is a valid session pointer; no concurrent use.
        unsafe { check(sys::sam3_video_reset(self.raw.as_ptr())) }
    }

    /// Run propagation and return the per-frame results in emission order.
    ///
    /// The C call is synchronous: propagation runs to completion (or to
    /// first callback failure) before this function returns. Memory is
    /// bounded by one `FrameResult` per visited frame.
    pub fn propagate(&mut self, direction: Direction) -> Result<Vec<FrameResult>> {
        // State the C trampoline writes into. Must live until after
        // sam3_video_propagate returns.
        struct CallbackState {
            frames: Vec<FrameResult>,
            err: Option<Error>,
        }
        let mut state = CallbackState {
            frames: Vec::new(),
            err: None,
        };

        unsafe extern "C" fn trampoline(
            result: *const sys::sam3_video_frame_result,
            user_data: *mut std::ffi::c_void,
        ) -> i32 {
            // SAFETY: libsam3 passes the same user_data we set below, and
            // guarantees result is non-null and live for the duration of
            // this call.
            if result.is_null() || user_data.is_null() {
                return 1;
            }
            let state = unsafe { &mut *(user_data as *mut CallbackState) };
            let raw = unsafe { &*result };
            match FrameResult::from_raw(raw) {
                Ok(fr) => {
                    state.frames.push(fr);
                    0
                }
                Err(e) => {
                    state.err = Some(e);
                    1
                }
            }
        }

        // SAFETY: self.raw is a valid session pointer; state outlives the
        // call; trampoline is a plain extern "C" fn with matching signature.
        let code = unsafe {
            sys::sam3_video_propagate(
                self.raw.as_ptr(),
                direction.to_raw(),
                Some(trampoline),
                &mut state as *mut CallbackState as *mut std::ffi::c_void,
            )
        };
        check(code)?;
        if let Some(e) = state.err {
            return Err(e);
        }
        Ok(state.frames)
    }

    /// Run an FFI entry point that fills a `sam3_video_frame_result` and
    /// return the copied-out `FrameResult`. The raw struct is freed via
    /// `sam3_video_frame_result_free` on all paths.
    fn with_result(
        &mut self,
        call: impl FnOnce(
            *mut sys::sam3_video_session,
            *mut sys::sam3_video_frame_result,
        ) -> sys::sam3_error,
    ) -> Result<FrameResult> {
        // Zero-init is the documented precondition for the prompt entry
        // points (they memset(result, 0) on entry anyway, but we must
        // still start with valid memory).
        let mut raw: MaybeUninit<sys::sam3_video_frame_result> = MaybeUninit::zeroed();
        let raw_ptr = raw.as_mut_ptr();
        let _guard = FrameResultGuard(raw_ptr);
        let code = call(self.raw.as_ptr(), raw_ptr);
        check(code)?;
        // SAFETY: check(code) == Ok(()) implies the entry point fully
        // populated the struct; `raw` has been written.
        let populated = unsafe { raw.assume_init_ref() };
        FrameResult::from_raw(populated)
    }
}

impl Drop for VideoSession<'_> {
    fn drop(&mut self) {
        // SAFETY: self.raw is the pointer returned by sam3_video_start(_ex);
        // sam3_video_end tolerates any valid session handle.
        unsafe { sys::sam3_video_end(self.raw.as_ptr()) }
    }
}

/// Convert a filesystem path to a `CString`.
fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        CString::new(path.as_os_str().as_bytes()).map_err(|_| Error::Invalid)
    }
    #[cfg(not(unix))]
    {
        let s = path.to_str().ok_or(Error::Invalid)?;
        CString::new(s).map_err(|_| Error::Invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_raw_matches_c_constants() {
        assert_eq!(Direction::Both.to_raw(), sys::SAM3_PROPAGATE_BOTH as i32);
        assert_eq!(
            Direction::Forward.to_raw(),
            sys::SAM3_PROPAGATE_FORWARD as i32
        );
        assert_eq!(
            Direction::Backward.to_raw(),
            sys::SAM3_PROPAGATE_BACKWARD as i32
        );
    }

    #[test]
    fn start_opts_defaults_map_to_sentinels() {
        let raw = StartOpts::default().to_raw();
        assert_eq!(raw.frame_cache_backend_budget, 0);
        assert_eq!(raw.frame_cache_spill_budget, 0);
        assert_eq!(raw.clear_non_cond_window, 0);
        // iter_use_prev_mask_pred / multimask_via_stability default to -1
        // (i.e. "use libsam3's compiled-in default").
        assert_eq!(raw.iter_use_prev_mask_pred, -1);
        assert_eq!(raw.multimask_via_stability, -1);
    }

    #[test]
    fn start_opts_bools_map_to_zero_one() {
        let raw = StartOpts {
            iter_use_prev_mask_pred: Some(true),
            multimask_via_stability: Some(false),
            ..StartOpts::default()
        }
        .to_raw();
        assert_eq!(raw.iter_use_prev_mask_pred, 1);
        assert_eq!(raw.multimask_via_stability, 0);
    }
}
