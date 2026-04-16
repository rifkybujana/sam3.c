//! Logging and runtime-info helpers.

use sam3_sys as sys;

/// Log severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Detailed tracing (suppressed by default).
    Debug,
    /// Operational milestones (default).
    Info,
    /// Non-fatal issues.
    Warn,
    /// Failures affecting correctness.
    Error,
}

impl From<LogLevel> for sys::sam3_log_level {
    fn from(l: LogLevel) -> Self {
        match l {
            LogLevel::Debug => sys::sam3_log_level::SAM3_LOG_DEBUG,
            LogLevel::Info => sys::sam3_log_level::SAM3_LOG_INFO,
            LogLevel::Warn => sys::sam3_log_level::SAM3_LOG_WARN,
            LogLevel::Error => sys::sam3_log_level::SAM3_LOG_ERROR,
        }
    }
}

/// Set the process-global minimum log level.
///
/// Messages below `level` are suppressed. Default is [`LogLevel::Info`].
pub fn set_log_level(level: LogLevel) {
    // SAFETY: sam3_log_set_level takes a simple enum, has no preconditions.
    unsafe { sys::sam3_log_set_level(level.into()) }
}

/// Return the libsam3 version string.
///
/// The pointer has `'static` lifetime (it lives in the loaded library's
/// read-only data segment).
pub fn version() -> &'static str {
    // SAFETY: sam3_version returns a pointer to a static, NUL-terminated
    // ASCII string. Never returns NULL.
    unsafe {
        let c = sys::sam3_version();
        debug_assert!(!c.is_null());
        std::ffi::CStr::from_ptr(c)
            .to_str()
            .expect("sam3_version must be valid UTF-8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn set_log_level_does_not_panic() {
        // All four levels should be accepted without crashing.
        set_log_level(LogLevel::Debug);
        set_log_level(LogLevel::Info);
        set_log_level(LogLevel::Warn);
        set_log_level(LogLevel::Error);
    }
}
