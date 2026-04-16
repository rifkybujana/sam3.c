//! Safe Rust bindings to the SAM3 inference engine.

#![warn(missing_docs)]

mod error;

pub use error::{Error, Result};

mod log;

pub use log::{set_log_level, version, LogLevel};

mod prompt;

pub use prompt::{Box, MaskPrompt, Point, PointLabel, Prompt};

mod image;

pub use image::ImageData;

mod result;

pub use result::SegmentResult;

mod ctx;

pub use ctx::Ctx;
