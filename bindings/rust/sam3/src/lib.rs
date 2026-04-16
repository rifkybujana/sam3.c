//! Safe Rust bindings to the SAM3 inference engine.

#![warn(missing_docs)]

mod error;

pub use error::{Error, Result};

mod log;

pub use log::{set_log_level, version, LogLevel};
