//! SAM3 inference context.

use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;

use sam3_sys as sys;

use crate::error::{Error, Result};

/// Convert a filesystem path to a `CString` suitable for libsam3.
///
/// Returns `Error::Invalid` on paths containing interior NULs or non-UTF-8
/// bytes on non-Unix platforms (libsam3 is Unix-focused; Windows paths are
/// best-effort via the UTF-8 representation).
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

/// An owned SAM3 inference context.
///
/// All state lives inside `Ctx`. Drop frees the context and joins any
/// background text-encoder worker.
///
/// `Ctx` is neither [`Send`] nor [`Sync`]: the context holds internal mutable
/// state and a worker thread, and libsam3 does not document cross-thread
/// safety. For concurrency, use separate contexts per thread.
//
// The `_not_send_sync: PhantomData<*mut ()>` field is load-bearing: it makes
// `Ctx` !Send + !Sync via auto-trait negative reasoning. Removing or changing
// that field requires rethinking thread-safety end-to-end. The compile-time
// `assert_not_send_not_sync` check below catches regressions.
pub struct Ctx {
    raw: NonNull<sys::sam3_ctx>,
    _not_send_sync: PhantomData<*mut ()>,
}

// Compile-time assertion that `Ctx` is neither `Send` nor `Sync`. This uses
// the ambiguous-impl trick: two blanket impls of `AmbiguousIfSend<_>` apply to
// `T: Send`, which would make the method call below ambiguous. Only one impl
// applies when `T: !Send`, so `Ctx` resolving unambiguously here is the proof.
// A future change that accidentally makes `Ctx: Send` breaks compilation.
#[allow(dead_code)]
const _: fn() = || {
    trait AmbiguousIfSend<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
    <Ctx as AmbiguousIfSend<_>>::some_item();

    trait AmbiguousIfSync<A> {
        fn some_item() {}
    }
    impl<T: ?Sized> AmbiguousIfSync<()> for T {}
    impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}
    <Ctx as AmbiguousIfSync<_>>::some_item();
};

impl Ctx {
    /// Create a new SAM3 context.
    pub fn new() -> Result<Self> {
        // SAFETY: sam3_init has no preconditions and returns NULL on failure.
        let raw = unsafe { sys::sam3_init() };
        NonNull::new(raw)
            .map(|raw| Ctx {
                raw,
                _not_send_sync: PhantomData,
            })
            .ok_or(Error::NoMemory)
    }

    /// Create a new SAM3 context with non-default cache slot counts.
    ///
    /// `n_image_slots` is the upper bound on cached images (default is 8).
    /// `image_mem_budget_bytes = 0` keeps all slots resident in RAM (no
    /// spill-to-disk), trading memory for predictability — useful when the
    /// process tolerates the working set and disk-spill failures (e.g. EIO,
    /// permission errors) would silently drop encoded image features and
    /// force re-encoding on the next visit.
    pub fn new_with_cache(n_image_slots: i32, n_text_slots: i32) -> Result<Self> {
        let opts = sys::sam3_cache_opts {
            n_image_slots,
            n_text_slots,
            image_mem_budget_bytes: 0,
            image_spill_dir: std::ptr::null(),
        };
        // SAFETY: sam3_init_ex reads the opts struct synchronously and copies
        // out any retained fields; the local goes out of scope safely after
        // the call returns. NULL on failure.
        let raw = unsafe { sys::sam3_init_ex(&opts as *const _) };
        NonNull::new(raw)
            .map(|raw| Ctx {
                raw,
                _not_send_sync: PhantomData,
            })
            .ok_or(Error::NoMemory)
    }

    /// Raw pointer for intra-crate FFI (e.g. video session construction).
    ///
    /// Not part of the public API. Returned pointer is live for the lifetime
    /// of `&self`; callers must not alias the exclusive access expected by
    /// other `&mut self` methods.
    pub(crate) fn raw(&self) -> *mut sys::sam3_ctx {
        self.raw.as_ptr()
    }

    /// Return the model's expected input image size, or 0 if no model is loaded.
    pub fn image_size(&self) -> u32 {
        // SAFETY: raw is valid; sam3_get_image_size is const and safe.
        let sz = unsafe { sys::sam3_get_image_size(self.raw.as_ptr()) };
        sz.max(0) as u32
    }

    /// Load a `.sam3` weight file.
    ///
    /// Must be called before [`set_image`](Self::set_image) or
    /// [`segment`](Self::segment).
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. c is a NUL-terminated path string not
        // retained beyond this call.
        unsafe { crate::error::check(sys::sam3_load_model(self.raw.as_ptr(), c.as_ptr())) }
    }

    /// Load a BPE vocabulary file (required for text prompts).
    pub fn load_bpe<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. c is a NUL-terminated path string not
        // retained beyond this call.
        unsafe { crate::error::check(sys::sam3_load_bpe(self.raw.as_ptr(), c.as_ptr())) }
    }

    /// Set the input image from a raw RGB byte buffer (`W * H * 3` bytes).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `pixels.len() < width * height * 3` or
    /// if `width * height * 3` overflows `usize`. Oversized buffers are
    /// accepted.
    pub fn set_image_rgb(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<()> {
        let need = crate::ImageData {
            pixels,
            width,
            height,
        }
        .required_len()
        .ok_or(Error::Invalid)?;
        if pixels.len() < need {
            return Err(Error::Invalid);
        }
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. pixels has at least width*height*3
        // bytes (verified above) and is not retained past the call.
        unsafe {
            crate::error::check(sys::sam3_set_image(
                self.raw.as_ptr(),
                pixels.as_ptr(),
                width as i32,
                height as i32,
            ))
        }
    }

    /// Set the input image from an [`ImageData`](crate::ImageData).
    ///
    /// # Errors
    ///
    /// Same as [`set_image_rgb`](Self::set_image_rgb).
    pub fn set_image(&mut self, img: &crate::ImageData<'_>) -> Result<()> {
        self.set_image_rgb(img.pixels, img.width, img.height)
    }

    /// Set the input image by loading a PNG/JPEG/BMP file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the path contains interior NULs (or is
    /// non-UTF-8 on non-Unix), [`Error::Io`] if the file cannot be read, or
    /// other variants as libsam3 reports for decoding failures.
    pub fn set_image_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. c is a NUL-terminated path string not
        // retained beyond this call.
        unsafe { crate::error::check(sys::sam3_set_image_file(self.raw.as_ptr(), c.as_ptr())) }
    }

    /// Set the coordinate space for subsequent point / box prompts.
    ///
    /// libsam3 stores dimensions as `int`; values above `i32::MAX` are
    /// silently truncated when cast.
    pub fn set_prompt_space(&mut self, width: u32, height: u32) {
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. sam3_set_prompt_space has no error
        // path.
        unsafe { sys::sam3_set_prompt_space(self.raw.as_ptr(), width as i32, height as i32) }
    }

    /// Pre-tokenize and asynchronously encode a text prompt.
    ///
    /// Requires a BPE vocab loaded via [`load_bpe`](Self::load_bpe).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `text` contains an interior NUL byte.
    pub fn set_text(&mut self, text: &str) -> Result<()> {
        let c = CString::new(text).map_err(|_| Error::Invalid)?;
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. c is a NUL-terminated string not
        // retained beyond this call.
        unsafe { crate::error::check(sys::sam3_set_text(self.raw.as_ptr(), c.as_ptr())) }
    }

    /// Populate the image feature cache for `(pixels, width, height)` without
    /// changing the current-image state.
    ///
    /// Runs the image encoder (blocking) and stores the resulting features
    /// in the in-memory cache keyed by pixel content. A later
    /// [`set_image_rgb`](Self::set_image_rgb) with the same buffer returns
    /// in microseconds. No-op if the same pixels are already cached.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `pixels.len() < width * height * 3` or
    /// if `width * height * 3` overflows `usize`. Requires a model loaded
    /// via [`load_model`](Self::load_model).
    pub fn precache_image(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<()> {
        let need = crate::ImageData {
            pixels,
            width,
            height,
        }
        .required_len()
        .ok_or(Error::Invalid)?;
        if pixels.len() < need {
            return Err(Error::Invalid);
        }
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. pixels has at least width*height*3
        // bytes (verified above) and is not retained past the call.
        unsafe {
            crate::error::check(sys::sam3_precache_image(
                self.raw.as_ptr(),
                pixels.as_ptr(),
                width as i32,
                height as i32,
            ))
        }
    }

    /// Serialize a cached image entry to disk.
    ///
    /// Looks up the in-memory cache entry keyed by `(pixels, width, height)`
    /// and writes it to `path`. Call [`precache_image`](Self::precache_image)
    /// first with the same buffer. The saved file includes a model signature,
    /// so loading into a different model will be rejected.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if the buffer is too small, no matching
    /// cache entry exists, or the path cannot be converted to a C string.
    pub fn cache_save_image<P: AsRef<Path>>(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        path: P,
    ) -> Result<()> {
        let need = crate::ImageData {
            pixels,
            width,
            height,
        }
        .required_len()
        .ok_or(Error::Invalid)?;
        if pixels.len() < need {
            return Err(Error::Invalid);
        }
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. pixels has at least width*height*3
        // bytes (verified above) and neither the buffer nor the path string
        // are retained past the call.
        unsafe {
            crate::error::check(sys::sam3_cache_save_image(
                self.raw.as_ptr(),
                pixels.as_ptr(),
                width as i32,
                height as i32,
                c.as_ptr(),
            ))
        }
    }

    /// Restore a previously-saved image cache entry from disk.
    ///
    /// After this call, a [`set_image_rgb`](Self::set_image_rgb) or
    /// [`precache_image`](Self::precache_image) with the pixels used at save
    /// time will hit the cache. Returns [`Error::Model`] if the file's model
    /// signature does not match the currently loaded model.
    pub fn cache_load_image<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let c = path_to_cstring(path.as_ref())?;
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. c is a NUL-terminated path string
        // not retained past the call.
        unsafe { crate::error::check(sys::sam3_cache_load_image(self.raw.as_ptr(), c.as_ptr())) }
    }

    /// Flush in-memory image and/or text feature caches.
    ///
    /// `which` is a bitmask: `1` = image, `2` = text, `0` = both. After this
    /// call the next `set_image_rgb` / `set_text` will re-encode from scratch.
    pub fn cache_clear(&mut self, which: u32) {
        // SAFETY: self.raw is a non-null sam3_ctx from sam3_init(); &mut self
        // guarantees no concurrent use. sam3_cache_clear has no return value
        // and tolerates any `which` value.
        unsafe { sys::sam3_cache_clear(self.raw.as_ptr(), which) }
    }

    /// Run segmentation with the given prompts against the current image.
    ///
    /// `prompts` may mix points, boxes, masks, and (if a BPE vocab is
    /// loaded) text. Borrows inside `Prompt` must outlive this call.
    pub fn segment(&mut self, prompts: &[crate::Prompt<'_>]) -> Result<crate::SegmentResult> {
        let scratch = crate::prompt::Prompt::lower_all(prompts)?;

        // RAII guard: always call sam3_result_free, even on early return or
        // panic. Holds a raw pointer rather than `&mut` so the result can be
        // simultaneously read via `&raw` below.
        struct Guard(*mut sys::sam3_result);
        impl Drop for Guard {
            fn drop(&mut self) {
                // SAFETY: the guarded struct was (attempted to be) filled by
                // sam3_segment; sam3_result_free tolerates partially-filled
                // results (it null-checks internal pointers).
                unsafe { sys::sam3_result_free(self.0) }
            }
        }

        // SAFETY: zero-initialized sam3_result is valid for sam3_segment to
        // fill in; all pointer fields will be set by the callee.
        let mut raw = unsafe { std::mem::zeroed::<sys::sam3_result>() };
        // Named binding (not bare `_`) so the guard lives to scope end and
        // drops *after* the subsequent `from_raw` read completes.
        let _guard = Guard(&mut raw as *mut _);
        let err_code = unsafe {
            sys::sam3_segment(
                self.raw.as_ptr(),
                scratch.lowered.as_ptr(),
                scratch.lowered.len() as i32,
                &mut raw,
            )
        };

        crate::error::check(err_code)?;
        crate::SegmentResult::from_raw(&raw)
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        // SAFETY: raw is a valid pointer obtained from sam3_init().
        // sam3_free joins the async worker and releases all resources.
        unsafe { sys::sam3_free(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_succeeds_and_drop_does_not_crash() {
        let ctx = Ctx::new().expect("sam3_init should succeed on a fresh process");
        drop(ctx);
    }

    #[test]
    fn multiple_contexts_can_coexist_sequentially() {
        for _ in 0..4 {
            let _ctx = Ctx::new().unwrap();
        }
    }

    #[test]
    fn image_size_before_load_is_zero_or_positive() {
        let ctx = Ctx::new().unwrap();
        // Before loading a model, the returned size is either 0 or the
        // compiled-in default; both are valid. Just verify the call succeeds.
        let _ = ctx.image_size();
    }

    #[test]
    fn load_model_missing_file_returns_error() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx
            .load_model("/nonexistent/path/to/model.sam3")
            .unwrap_err();
        // libsam3 maps file-not-found to SAM3_EIO (src/core/weight.c:314).
        assert!(matches!(err, Error::Io), "unexpected error: {err:?}");
    }

    #[test]
    fn set_image_rgb_rejects_short_buffer() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.set_image_rgb(&[0; 10], 10, 10).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn set_image_rgb_rejects_dimension_overflow() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.set_image_rgb(&[0; 10], u32::MAX, u32::MAX).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn set_text_rejects_interior_nul() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.set_text("hello\0world").unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn set_prompt_space_is_infallible() {
        let mut ctx = Ctx::new().unwrap();
        ctx.set_prompt_space(1024, 1024);
    }

    #[test]
    fn precache_image_rejects_short_buffer() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.precache_image(&[0; 10], 10, 10).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn precache_image_rejects_dimension_overflow() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx.precache_image(&[0; 10], u32::MAX, u32::MAX).unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn cache_save_image_rejects_short_buffer() {
        let mut ctx = Ctx::new().unwrap();
        let err = ctx
            .cache_save_image(&[0; 10], 10, 10, "/tmp/unused.sam3cache")
            .unwrap_err();
        assert!(matches!(err, Error::Invalid));
    }

    #[test]
    fn cache_load_image_missing_file_errors() {
        let mut ctx = Ctx::new().unwrap();
        // No model loaded, so proc is not ready — libsam3 returns SAM3_EINVAL
        // before touching the filesystem. Either Invalid or Io is acceptable.
        let err = ctx.cache_load_image("/nonexistent/path.sam3cache").unwrap_err();
        assert!(matches!(err, Error::Invalid | Error::Io));
    }
}
