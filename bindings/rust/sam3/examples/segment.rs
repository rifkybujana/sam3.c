//! Minimal end-to-end segmentation example.
//!
//! Usage:
//!
//! ```sh
//! DYLD_LIBRARY_PATH=../../build cargo run --example segment -- \
//!     --model /path/to/model.sam3 \
//!     --image /path/to/image.jpg \
//!     --point 400,300,1
//! ```
//!
//! To exercise text prompts, also pass `--bpe /path/to/vocab.bpe --text "a cat"`.

use std::env;
use std::process::ExitCode;

use sam3::{Ctx, Point, PointLabel, Prompt};

fn main() -> ExitCode {
    let mut model: Option<String> = None;
    let mut image: Option<String> = None;
    let mut bpe: Option<String> = None;
    let mut point: Option<(f32, f32, i32)> = None;
    let mut text: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--image" => image = args.next(),
            "--bpe" => bpe = args.next(),
            "--point" => {
                let s = args.next().expect("--point x,y,label");
                let parts: Vec<&str> = s.split(',').collect();
                assert_eq!(parts.len(), 3, "--point x,y,label");
                point = Some((
                    parts[0].parse().expect("x"),
                    parts[1].parse().expect("y"),
                    parts[2].parse().expect("label"),
                ));
            }
            "--text" => text = args.next(),
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let model_path = match model {
        Some(m) => m,
        None => {
            eprintln!("--model is required");
            return ExitCode::from(2);
        }
    };
    let image_path = match image {
        Some(i) => i,
        None => {
            eprintln!("--image is required");
            return ExitCode::from(2);
        }
    };

    let mut ctx = Ctx::new().expect("sam3_init");
    ctx.load_model(&model_path).expect("load_model");
    if let Some(ref b) = bpe {
        ctx.load_bpe(b).expect("load_bpe");
    }
    ctx.set_image_file(&image_path).expect("set_image_file");

    let mut prompts: Vec<Prompt<'_>> = Vec::new();
    if let Some((x, y, lab)) = point {
        prompts.push(Prompt::Point(Point {
            x,
            y,
            label: if lab == 0 {
                PointLabel::Background
            } else {
                PointLabel::Foreground
            },
        }));
    }
    if let Some(ref t) = text {
        if bpe.is_none() {
            eprintln!("--text requires --bpe");
            return ExitCode::from(2);
        }
        ctx.set_text(t).expect("set_text");
        prompts.push(Prompt::Text(t));
    }

    let result = ctx.segment(&prompts).expect("segment");
    println!(
        "segmented: n_masks={} H={} W={} best={:?} iou_valid={}",
        result.n_masks(),
        result.mask_height(),
        result.mask_width(),
        result.best_mask(),
        result.iou_valid(),
    );

    ExitCode::SUCCESS
}
