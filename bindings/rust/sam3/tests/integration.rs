//! End-to-end integration tests against a real SAM3 model.
//!
//! Gated by environment variables (matching `python/tests/conftest.py`):
//!   - SAM3_MODEL_PATH — required
//!   - SAM3_TEST_IMAGE — required
//!   - SAM3_BPE_PATH   — optional (required for text-prompt tests)
//!
//! Tests are silently skipped (not failed) when env vars are unset.

use sam3::{Ctx, Point, PointLabel, Prompt};

fn env_or_skip(var: &str) -> Option<String> {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("skipping: {var} not set");
            None
        }
    }
}

#[test]
fn load_and_segment_with_point() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let image = match env_or_skip("SAM3_TEST_IMAGE") {
        Some(v) => v,
        None => return,
    };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.set_image_file(&image).unwrap();

    let prompts = [Prompt::Point(Point {
        x: 100.0,
        y: 100.0,
        label: PointLabel::Foreground,
    })];
    let result = ctx.segment(&prompts).unwrap();

    assert!(result.n_masks() > 0, "expected at least one mask");
    assert!(result.mask_height() > 0);
    assert!(result.mask_width() > 0);
    assert_eq!(result.iou_scores().len(), result.n_masks());
}

#[test]
fn load_and_segment_with_box() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let image = match env_or_skip("SAM3_TEST_IMAGE") {
        Some(v) => v,
        None => return,
    };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.set_image_file(&image).unwrap();

    let prompts = [Prompt::Box(sam3::Box {
        x1: 50.0,
        y1: 50.0,
        x2: 200.0,
        y2: 200.0,
    })];
    let result = ctx.segment(&prompts).unwrap();

    assert!(result.n_masks() > 0);
}

#[test]
fn segment_with_text_requires_bpe() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };
    let image = match env_or_skip("SAM3_TEST_IMAGE") {
        Some(v) => v,
        None => return,
    };
    let bpe = match env_or_skip("SAM3_BPE_PATH") {
        Some(v) => v,
        None => return,
    };

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();
    ctx.load_bpe(&bpe).unwrap();
    ctx.set_image_file(&image).unwrap();
    ctx.set_text("object").unwrap();

    let prompts = [Prompt::Text("object")];
    let result = ctx.segment(&prompts).unwrap();
    assert!(result.n_masks() > 0);
}

#[test]
fn multiple_contexts_are_independent() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };

    let a = Ctx::new().unwrap();
    drop(a);
    let mut b = Ctx::new().unwrap();
    b.load_model(&model).unwrap();
}
