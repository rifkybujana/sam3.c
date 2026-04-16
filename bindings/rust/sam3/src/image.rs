//! Image data helpers.

/// A borrowed RGB image buffer with explicit dimensions.
///
/// Row-major interleaved RGB (`R, G, B, R, G, B, ...`). `pixels.len()`
/// must be exactly `width * height * 3`.
#[derive(Debug, Clone, Copy)]
pub struct ImageData<'a> {
    /// Raw pixel bytes.
    pub pixels: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl<'a> ImageData<'a> {
    /// Return the expected buffer length for `width * height * 3 bytes`.
    #[allow(dead_code)] // TODO(task-3.10): remove when set_image calls it.
    pub(crate) fn required_len(&self) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_len_computes_width_times_height_times_3() {
        let img = ImageData {
            pixels: &[],
            width: 4,
            height: 5,
        };
        assert_eq!(img.required_len(), Some(60));
    }

    #[test]
    fn required_len_overflow_returns_none() {
        let img = ImageData {
            pixels: &[],
            width: u32::MAX,
            height: u32::MAX,
        };
        assert!(img.required_len().is_none());
    }
}
