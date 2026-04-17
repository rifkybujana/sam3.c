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

    /// Apply greedy mask non-maximum suppression, mirroring
    /// `sam3_cli segment`'s post-processing.
    ///
    /// Prefilters by per-mask score (`prob_thresh`), then greedily keeps
    /// the highest-scoring mask whose binarised (logit > 0) IoU with all
    /// previously-kept masks is ≤ `iou_thresh`. `min_quality` rejects
    /// masks below a stability threshold after NMS (pass `0.0` to disable).
    ///
    /// CLI defaults are `prob_thresh = 0.5`, `iou_thresh = 0.5`,
    /// `min_quality = 0.0`.
    ///
    /// Returns a new `SegmentResult` with only the kept masks, ordered by
    /// descending score. [`best_mask`](Self::best_mask) is cleared because
    /// NMS re-indexes the output; boxes are compacted when present.
    ///
    /// # Errors
    ///
    /// - [`Error::Invalid`] if [`iou_valid`](Self::iou_valid) is `false`
    ///   (NMS needs real scores), if `n_masks > 512` (libsam3 limit), or if
    ///   the underlying `sam3_mask_nms` rejects its inputs.
    pub fn nms(&self, prob_thresh: f32, iou_thresh: f32, min_quality: f32) -> Result<Self> {
        if !self.iou_valid {
            return Err(Error::Invalid);
        }
        if self.n_masks == 0 {
            return Ok(SegmentResult {
                masks: Vec::new(),
                iou_scores: Vec::new(),
                boxes: self.boxes.as_ref().map(|_| Vec::new()),
                n_masks: 0,
                mask_height: self.mask_height,
                mask_width: self.mask_width,
                iou_valid: true,
                best_mask: None,
            });
        }
        if self.n_masks > 512 {
            return Err(Error::Invalid);
        }

        let mut kept = vec![0_i32; self.n_masks];
        // SAFETY: masks has n_masks*h*w f32s and iou_scores has n_masks
        // f32s (invariants held by `from_raw`). `kept` has n_masks slots,
        // which is the maximum NMS can return.
        let n_kept = unsafe {
            sys::sam3_mask_nms(
                self.masks.as_ptr(),
                self.iou_scores.as_ptr(),
                self.n_masks as i32,
                self.mask_height as i32,
                self.mask_width as i32,
                prob_thresh,
                iou_thresh,
                min_quality,
                kept.as_mut_ptr(),
            )
        };
        if n_kept < 0 {
            return Err(Error::Invalid);
        }
        let n_kept = n_kept as usize;
        kept.truncate(n_kept);

        let stride = self.mask_height * self.mask_width;
        let mut masks = Vec::with_capacity(n_kept * stride);
        let mut iou_scores = Vec::with_capacity(n_kept);
        let mut boxes_out: Option<Vec<[f32; 4]>> =
            self.boxes.as_ref().map(|_| Vec::with_capacity(n_kept));
        for &i in &kept {
            let i = i as usize;
            let start = i * stride;
            masks.extend_from_slice(&self.masks[start..start + stride]);
            iou_scores.push(self.iou_scores[i]);
            if let (Some(out), Some(src)) = (boxes_out.as_mut(), self.boxes.as_ref()) {
                out.push(src[i]);
            }
        }

        Ok(SegmentResult {
            masks,
            iou_scores,
            boxes: boxes_out,
            n_masks: n_kept,
            mask_height: self.mask_height,
            mask_width: self.mask_width,
            iou_valid: true,
            best_mask: None,
        })
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

        // libsam3 contract (include/sam3/sam3_types.h): `masks` points to
        // n*H*W f32s. A null masks pointer with n_masks > 0 is a contract
        // violation we surface as Error::Invalid rather than a silently
        // inconsistent result.
        let masks = if total == 0 {
            Vec::new()
        } else if raw.masks.is_null() {
            return Err(Error::Invalid);
        } else {
            // SAFETY: libsam3 guarantees raw.masks points to `total` f32s
            // while `raw` lives.
            unsafe { std::slice::from_raw_parts(raw.masks, total) }.to_vec()
        };

        let iou_scores = if n == 0 {
            Vec::new()
        } else if raw.iou_scores.is_null() {
            return Err(Error::Invalid);
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

    #[test]
    fn nms_on_empty_result_is_empty_result() {
        let r = synthetic(0, 0, 0, None);
        let out = r.nms(0.5, 0.5, 0.0).unwrap();
        assert_eq!(out.n_masks(), 0);
        assert!(out.masks().is_empty());
        assert!(out.best_mask().is_none());
    }

    #[test]
    fn nms_rejects_invalid_iou_scores() {
        let mut r = synthetic(2, 2, 2, None);
        r.iou_valid = false;
        assert!(matches!(r.nms(0.5, 0.5, 0.0), Err(Error::Invalid)));
    }

    #[test]
    fn nms_drops_low_score_masks() {
        // Two masks: one with score 0.9 (kept), one with 0.1 (dropped by
        // prob_thresh=0.5). Masks don't overlap so IoU wouldn't eliminate
        // either — only the score prefilter drops the weak one.
        let mut r = synthetic(2, 4, 4, None);
        r.iou_scores = vec![0.9, 0.1];
        for v in r.masks[0..16].iter_mut() {
            *v = 5.0;
        }
        for v in r.masks[16..32].iter_mut() {
            *v = 5.0;
        }
        let out = r.nms(0.5, 0.5, 0.0).unwrap();
        assert_eq!(out.n_masks(), 1);
        assert!((out.iou_scores()[0] - 0.9).abs() < 1e-6);
        assert!(out.best_mask().is_none());
        assert_eq!(out.mask_height(), 4);
        assert_eq!(out.mask_width(), 4);
    }

    #[test]
    fn nms_suppresses_overlapping_masks() {
        // Two identical masks; higher-score one wins, the other is
        // suppressed by IoU >= iou_thresh.
        let mut r = synthetic(2, 4, 4, None);
        r.iou_scores = vec![0.9, 0.8];
        for v in r.masks.iter_mut() {
            *v = 5.0;
        }
        let out = r.nms(0.5, 0.5, 0.0).unwrap();
        assert_eq!(out.n_masks(), 1);
        assert!((out.iou_scores()[0] - 0.9).abs() < 1e-6);
    }
}
