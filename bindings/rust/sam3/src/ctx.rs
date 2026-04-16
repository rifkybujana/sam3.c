//! SAM3 inference context.

use std::marker::PhantomData;
use std::ptr::NonNull;

use sam3_sys as sys;

use crate::error::{Error, Result};

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
}
