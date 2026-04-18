//! End-to-end integration tests against a real SAM3 model.
//!
//! Gated by environment variables (matching `python/tests/conftest.py`):
//!   - SAM3_MODEL_PATH — required
//!   - SAM3_TEST_IMAGE — required
//!   - SAM3_BPE_PATH   — optional (required for text-prompt tests)
//!
//! Tests are silently skipped (not failed) when env vars are unset.

use sam3::{Ctx, Direction, Point, PointLabel, Prompt, VideoSession};

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

// --- Video tracking ----------------------------------------------------

// Mirrors the parameters used by tests/test_video_e2e.c and the Python
// test_video.py fixture: 8 frames of 256x256 with a 32x32 white square
// moving 8 px diagonally per step.
const VIDEO_IMG: usize = 256;
const VIDEO_N_FRAMES: usize = 8;
const VIDEO_SQUARE: usize = 32;
const VIDEO_START: usize = 100;
const VIDEO_STEP: usize = 8;

fn write_ppm(path: &std::path::Path, pixels: &[u8], width: usize, height: usize) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "P6\n{width} {height}\n255").unwrap();
    f.write_all(pixels).unwrap();
}

fn generate_moving_square_clip(dir: &std::path::Path) {
    // Deterministic gray-noise background via a tiny LCG so the test
    // does not depend on a PRNG crate.
    let mut state: u64 = 0xC0FFEE;
    for i in 0..VIDEO_N_FRAMES {
        let mut frame = vec![0u8; VIDEO_IMG * VIDEO_IMG * 3];
        for px in frame.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Background: uniform in [100, 156).
            *px = 100 + ((state >> 32) as u32 % 56) as u8;
        }
        let x0 = VIDEO_START + i * VIDEO_STEP;
        let y0 = x0;
        for y in y0..y0 + VIDEO_SQUARE {
            for x in x0..x0 + VIDEO_SQUARE {
                let off = (y * VIDEO_IMG + x) * 3;
                frame[off] = 255;
                frame[off + 1] = 255;
                frame[off + 2] = 255;
            }
        }
        let path = dir.join(format!("frame_{i:04}.ppm"));
        write_ppm(&path, &frame, VIDEO_IMG, VIDEO_IMG);
    }
}

fn square_center(i: usize) -> (f32, f32) {
    let c = VIDEO_START as f32 + (i * VIDEO_STEP) as f32 + VIDEO_SQUARE as f32 * 0.5;
    (c, c)
}

/// Centroid of the `>= 0` region of a single mask plane.
fn mask_centroid(mask: &[f32], h: u32, w: u32) -> Option<(f32, f32)> {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut n = 0u32;
    for y in 0..h {
        for x in 0..w {
            if mask[(y * w + x) as usize] >= 0.0 {
                sx += x as f32;
                sy += y as f32;
                n += 1;
            }
        }
    }
    if n == 0 {
        None
    } else {
        Some((sx / n as f32, sy / n as f32))
    }
}

#[test]
fn video_session_tracks_moving_square() {
    let model = match env_or_skip("SAM3_MODEL_PATH") {
        Some(v) => v,
        None => return,
    };

    let tmp = std::env::temp_dir().join(format!(
        "sam3-rust-video-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    // Best-effort cleanup: the test body owns the directory for its run.
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(tmp.clone());

    generate_moving_square_clip(&tmp);

    let mut ctx = Ctx::new().unwrap();
    ctx.load_model(&model).unwrap();

    let img_size = ctx.image_size();
    assert!(img_size > 0, "model must report a positive input size");
    let scale = img_size as f32 / VIDEO_IMG as f32;

    let mut sess = VideoSession::start(&mut ctx, &tmp).unwrap();
    assert_eq!(sess.frame_count() as usize, VIDEO_N_FRAMES);

    let (gx, gy) = square_center(0);
    let r0 = sess
        .add_points(
            0,
            0,
            &[Point {
                x: gx * scale,
                y: gy * scale,
                label: PointLabel::Foreground,
            }],
        )
        .unwrap();
    assert_eq!(r0.objects.len(), 1);
    assert_eq!(r0.objects[0].obj_id, 0);
    assert!(r0.objects[0].mask.len() > 0);

    let frames = sess.propagate(Direction::Forward).unwrap();
    assert_eq!(frames.len(), VIDEO_N_FRAMES);

    // Track the centroid on each frame and compare to ground truth in
    // PNG-pixel space. The 16 px tolerance matches the Python test.
    let tol = 16.0_f32;
    for fr in &frames {
        let om = fr
            .by_obj_id(0)
            .unwrap_or_else(|| panic!("frame {} missing obj 0", fr.frame_idx));
        let (cx_m, cy_m) =
            mask_centroid(&om.mask, om.mask_height, om.mask_width).expect("empty mask");
        let px_per_mask_x = VIDEO_IMG as f32 / om.mask_width as f32;
        let px_per_mask_y = VIDEO_IMG as f32 / om.mask_height as f32;
        let cx_png = cx_m * px_per_mask_x;
        let cy_png = cy_m * px_per_mask_y;
        let (ex, ey) = square_center(fr.frame_idx as usize);
        assert!(
            (cx_png - ex).abs() < tol,
            "frame {}: cx {cx_png:.1} vs expected {ex:.1}",
            fr.frame_idx
        );
        assert!(
            (cy_png - ey).abs() < tol,
            "frame {}: cy {cy_png:.1} vs expected {ey:.1}",
            fr.frame_idx
        );
    }
}
