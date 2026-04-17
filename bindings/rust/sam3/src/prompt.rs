//! Prompt types consumed by [`Ctx::segment`](crate::Ctx::segment).

use std::ffi::CString;

use sam3_sys as sys;

use crate::error::{Error, Result};

/// Foreground vs. background label attached to a [`Point`] prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLabel {
    /// Include the surrounding region in the output mask.
    Foreground,
    /// Exclude the surrounding region from the output mask.
    Background,
}

impl PointLabel {
    #[inline]
    pub(crate) fn to_raw(self) -> i32 {
        match self {
            PointLabel::Foreground => 1,
            PointLabel::Background => 0,
        }
    }
}

/// A (x, y, label) point prompt in the prompt-coordinate space.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Foreground / background label.
    pub label: PointLabel,
}

/// An axis-aligned bounding box prompt (xyxy).
#[derive(Debug, Clone, Copy)]
pub struct Box {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge (exclusive).
    pub x2: f32,
    /// Bottom edge (exclusive).
    pub y2: f32,
}

/// A dense `H*W` mask prompt borrowed from the caller.
///
/// `data.len()` must equal `width * height`; mismatches return
/// [`Error::Invalid`](crate::Error::Invalid) from [`Ctx::segment`](crate::Ctx::segment).
#[derive(Debug, Clone, Copy)]
pub struct MaskPrompt<'a> {
    /// Row-major `H*W` f32 values.
    pub data: &'a [f32],
    /// Mask width in pixels.
    pub width: u32,
    /// Mask height in pixels.
    pub height: u32,
}

/// An input prompt passed to [`Ctx::segment`](crate::Ctx::segment).
#[derive(Debug, Clone, Copy)]
pub enum Prompt<'a> {
    /// Point prompt.
    Point(Point),
    /// Bounding-box prompt.
    Box(Box),
    /// Dense mask prompt.
    Mask(MaskPrompt<'a>),
    /// UTF-8 text prompt (requires a BPE vocab loaded via `Ctx::load_bpe`).
    Text(&'a str),
}

/// Scratch storage needed to keep prompt arguments alive across the FFI call.
pub(crate) struct PromptScratch {
    /// Raw prompts passed to `sam3_segment`.
    pub lowered: Vec<sys::sam3_prompt>,
    /// Owned CStrings for text prompts; referenced by `lowered`.
    #[allow(dead_code)]
    text_keepalive: Vec<CString>,
}

impl<'a> Prompt<'a> {
    /// Lower a slice of `Prompt`s into FFI-ready storage.
    ///
    /// The returned `PromptScratch` must outlive the `sam3_segment` call.
    pub(crate) fn lower_all(prompts: &[Prompt<'a>]) -> Result<PromptScratch> {
        let mut text_keepalive: Vec<CString> = Vec::new();
        let mut lowered: Vec<sys::sam3_prompt> = Vec::with_capacity(prompts.len());

        for p in prompts {
            // SAFETY: zeroing a POD-with-union is valid; we overwrite the
            // active variant immediately below.
            let mut raw: sys::sam3_prompt = unsafe { std::mem::zeroed() };
            // Invariant maintained by each arm: set `raw.type_` to the tag
            // and write exactly one matching union variant. Union-field
            // *writes* are safe in Rust (only reads require `unsafe`), so
            // no `unsafe` block is needed here despite the union access.
            match p {
                Prompt::Point(pt) => {
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_POINT;
                    raw.__bindgen_anon_1.point = sys::sam3_point {
                        x: pt.x,
                        y: pt.y,
                        label: pt.label.to_raw(),
                    };
                }
                Prompt::Box(bx) => {
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_BOX;
                    raw.__bindgen_anon_1.box_ = sys::sam3_box {
                        x1: bx.x1,
                        y1: bx.y1,
                        x2: bx.x2,
                        y2: bx.y2,
                    };
                }
                Prompt::Mask(m) => {
                    let need = (m.width as usize)
                        .checked_mul(m.height as usize)
                        .ok_or(Error::Invalid)?;
                    if m.data.len() < need {
                        return Err(Error::Invalid);
                    }
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_MASK;
                    raw.__bindgen_anon_1.mask.data = m.data.as_ptr();
                    raw.__bindgen_anon_1.mask.width = m.width as i32;
                    raw.__bindgen_anon_1.mask.height = m.height as i32;
                }
                Prompt::Text(s) => {
                    let c = CString::new(*s).map_err(|_| Error::Invalid)?;
                    raw.type_ = sys::sam3_prompt_type::SAM3_PROMPT_TEXT;
                    raw.__bindgen_anon_1.text = c.as_ptr();
                    text_keepalive.push(c);
                }
            }
            lowered.push(raw);
        }

        Ok(PromptScratch {
            lowered,
            text_keepalive,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_label_raw_values_match_c_convention() {
        assert_eq!(PointLabel::Foreground.to_raw(), 1);
        assert_eq!(PointLabel::Background.to_raw(), 0);
    }

    #[test]
    fn lower_point_prompt_tags_correctly() {
        let prompts = [Prompt::Point(Point {
            x: 1.0,
            y: 2.0,
            label: PointLabel::Foreground,
        })];
        let scratch = Prompt::lower_all(&prompts).unwrap();
        assert_eq!(scratch.lowered.len(), 1);
        assert_eq!(
            scratch.lowered[0].type_,
            sys::sam3_prompt_type::SAM3_PROMPT_POINT
        );
        // SAFETY: tag is POINT, so the point variant is active.
        let pt = unsafe { scratch.lowered[0].__bindgen_anon_1.point };
        assert_eq!(pt.x, 1.0);
        assert_eq!(pt.y, 2.0);
        assert_eq!(pt.label, 1);
    }

    #[test]
    fn lower_text_prompt_rejects_interior_nul() {
        let prompts = [Prompt::Text("bad\0string")];
        assert!(matches!(Prompt::lower_all(&prompts), Err(Error::Invalid)));
    }

    #[test]
    fn lower_mask_rejects_short_buffer() {
        let data = [0.0_f32; 3];
        let prompts = [Prompt::Mask(MaskPrompt {
            data: &data,
            width: 4,
            height: 4,
        })];
        assert!(matches!(Prompt::lower_all(&prompts), Err(Error::Invalid)));
    }
}
