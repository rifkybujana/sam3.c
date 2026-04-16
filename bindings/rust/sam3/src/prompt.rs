//! Prompt types consumed by [`Ctx::segment`](crate::Ctx::segment).

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
    #[allow(dead_code)] // Used by Ctx::segment lowering in Task 3.11.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_label_raw_values_match_c_convention() {
        assert_eq!(PointLabel::Foreground.to_raw(), 1);
        assert_eq!(PointLabel::Background.to_raw(), 0);
    }
}
