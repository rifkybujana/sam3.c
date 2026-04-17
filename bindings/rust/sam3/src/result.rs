//! Segmentation result returned by [`Ctx::segment`](crate::Ctx::segment).
//!
//! Mask buffers contain raw f32 logits; threshold at 0.0 for a binary mask.

use sam3_sys as sys;

use crate::error::{Error, Result};

/// A segmentation result: one or more mask logits plus scores.
///
/// All buffers are owned `Vec<f32>` — copied out of the SAM3 arenas before
/// the next segmentation call reuses them.
#[derive(Debug)]
pub struct SegmentResult {
    pub(crate) masks: Vec<f32>,
    pub(crate) iou_scores: Vec<f32>,
    pub(crate) boxes: Option<Vec<[f32; 4]>>,
    pub(crate) n_masks: usize,
    pub(crate) mask_height: usize,
    pub(crate) mask_width: usize,
    pub(crate) iou_valid: bool,
    pub(crate) best_mask: Option<usize>,
}

impl SegmentResult {
    /// Number of masks returned.
    pub fn n_masks(&self) -> usize {
        self.n_masks
    }

    /// Mask height in pixels.
    pub fn mask_height(&self) -> usize {
        self.mask_height
    }

    /// Mask width in pixels.
    pub fn mask_width(&self) -> usize {
        self.mask_width
    }

    /// Whether [`iou_scores`](Self::iou_scores) are model-predicted (`true`)
    /// or placeholder zeros (`false`).
    pub fn iou_valid(&self) -> bool {
        self.iou_valid
    }

    /// Per-mask IoU scores; `len() == n_masks` regardless of [`iou_valid`](Self::iou_valid).
    /// When `!iou_valid()` the scores are placeholder zeros.
    pub fn iou_scores(&self) -> &[f32] {
        &self.iou_scores
    }

    /// Per-mask axis-aligned bounding boxes (xyxy), when boxes were computed.
    pub fn boxes(&self) -> Option<&[[f32; 4]]> {
        self.boxes.as_deref()
    }

    /// Stability-selected mask index, if the model emitted one.
    pub fn best_mask(&self) -> Option<usize> {
        self.best_mask
    }

    /// Flat view of all masks, row-major.
    ///
    /// Index element `(m, y, x)` via
    /// `m * mask_height() * mask_width() + y * mask_width() + x`.
    pub fn masks(&self) -> &[f32] {
        &self.masks
    }

    /// Return the `i`-th mask as an `H*W` slice, or `None` if out of range.
    pub fn mask(&self, index: usize) -> Option<&[f32]> {
        if index >= self.n_masks {
            return None;
        }
        let stride = self.mask_height * self.mask_width;
        let start = index.checked_mul(stride)?;
        let end = start.checked_add(stride)?;
        self.masks.get(start..end)
    }

    /// Return the stability-selected mask, if any.
    pub fn best(&self) -> Option<&[f32]> {
        self.best_mask.and_then(|i| self.mask(i))
    }

    /// Copy a C-owned `sam3_result` into an owned `SegmentResult`.
    ///
    /// The caller must free the source `sam3_result` via `sam3_result_free`
    /// regardless of the outcome of this function.
    pub(crate) fn from_raw(raw: &sys::sam3_result) -> Result<Self> {
        let n = raw.n_masks.max(0) as usize;
        let h = raw.mask_height.max(0) as usize;
        let w = raw.mask_width.max(0) as usize;
        let total = n
            .checked_mul(h)
            .and_then(|x| x.checked_mul(w))
            .ok_or(Error::Invalid)?;

        let masks = if total == 0 || raw.masks.is_null() {
            Vec::new()
        } else {
            // SAFETY: libsam3 guarantees raw.masks points to `total` f32s
            // while `raw` lives.
            unsafe { std::slice::from_raw_parts(raw.masks, total) }.to_vec()
        };

        let iou_scores = if n == 0 || raw.iou_scores.is_null() {
            Vec::new()
        } else {
            // SAFETY: libsam3 guarantees raw.iou_scores points to `n` f32s
            // while `raw` lives.
            unsafe { std::slice::from_raw_parts(raw.iou_scores, n) }.to_vec()
        };

        let boxes = if raw.boxes_valid != 0 && !raw.boxes.is_null() && n > 0 {
            // SAFETY: boxes_valid != 0 ⇒ raw.boxes holds n*4 f32s.
            let flat = unsafe { std::slice::from_raw_parts(raw.boxes, n * 4) };
            Some(
                flat.chunks_exact(4)
                    .map(|c| [c[0], c[1], c[2], c[3]])
                    .collect(),
            )
        } else {
            None
        };

        Ok(SegmentResult {
            masks,
            iou_scores,
            boxes,
            n_masks: n,
            mask_height: h,
            mask_width: w,
            iou_valid: raw.iou_valid != 0,
            best_mask: if raw.best_mask >= 0 {
                Some(raw.best_mask as usize)
            } else {
                None
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(n: usize, h: usize, w: usize, best: Option<usize>) -> SegmentResult {
        SegmentResult {
            masks: vec![0.0; n * h * w],
            iou_scores: vec![0.5; n],
            boxes: None,
            n_masks: n,
            mask_height: h,
            mask_width: w,
            iou_valid: true,
            best_mask: best,
        }
    }

    #[test]
    fn accessors_report_shape() {
        let r = synthetic(3, 4, 5, Some(1));
        assert_eq!(r.n_masks(), 3);
        assert_eq!(r.mask_height(), 4);
        assert_eq!(r.mask_width(), 5);
        assert!(r.iou_valid());
        assert_eq!(r.iou_scores().len(), 3);
        assert_eq!(r.masks().len(), 60);
    }

    #[test]
    fn mask_slice_is_h_times_w() {
        let r = synthetic(3, 4, 5, None);
        assert_eq!(r.mask(0).unwrap().len(), 20);
        assert_eq!(r.mask(2).unwrap().len(), 20);
        assert!(r.mask(3).is_none(), "out-of-range should be None");
    }

    #[test]
    fn best_returns_selected_mask() {
        let mut r = synthetic(2, 2, 2, Some(1));
        for v in r.masks[4..8].iter_mut() {
            *v = 9.0;
        }
        assert_eq!(r.best().unwrap(), &[9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn no_best_mask_is_none() {
        let r = synthetic(2, 2, 2, None);
        assert!(r.best().is_none());
    }

    #[test]
    fn zero_masks_is_valid() {
        let r = synthetic(0, 0, 0, None);
        assert_eq!(r.n_masks(), 0);
        assert!(r.masks().is_empty());
        assert!(r.mask(0).is_none());
    }
}
