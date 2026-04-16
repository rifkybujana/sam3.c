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
}
