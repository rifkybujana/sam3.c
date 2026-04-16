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
pub struct Ctx {
    raw: NonNull<sys::sam3_ctx>,
    _not_send_sync: PhantomData<*mut ()>,
}

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
}
