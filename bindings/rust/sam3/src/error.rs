//! SAM3 error type and code conversion.

use sam3_sys::sam3_error as sys_err;

/// Errors returned by the SAM3 runtime.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid argument passed to a SAM3 function.
    #[error("invalid argument")]
    Invalid,
    /// Allocation failure inside the SAM3 runtime.
    #[error("out of memory")]
    NoMemory,
    /// I/O error reading a weight file, image, or shader.
    #[error("I/O error")]
    Io,
    /// Backend (Metal/CPU) initialization failed.
    #[error("backend initialization failed")]
    Backend,
    /// Model file format error (wrong magic, unsupported version).
    #[error("model format error")]
    Model,
    /// Unsupported or mismatched tensor dtype.
    #[error("unsupported or mismatched dtype")]
    Dtype,
    /// Unrecognized error code (future-proofing).
    #[error("unknown SAM3 error ({0})")]
    Unknown(i32),
}

/// Convenience alias for `std::result::Result<T, sam3::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Convert a raw `sam3_error` code into a `Result`.
#[allow(dead_code)] // TODO(task-3.9): remove when load_model/load_bpe call check().
pub(crate) fn check(code: sys_err) -> Result<()> {
    match code {
        sys_err::SAM3_OK => Ok(()),
        sys_err::SAM3_EINVAL => Err(Error::Invalid),
        sys_err::SAM3_ENOMEM => Err(Error::NoMemory),
        sys_err::SAM3_EIO => Err(Error::Io),
        sys_err::SAM3_EBACKEND => Err(Error::Backend),
        sys_err::SAM3_EMODEL => Err(Error::Model),
        sys_err::SAM3_EDTYPE => Err(Error::Dtype),
        other => Err(Error::Unknown(other.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_maps_to_ok() {
        assert!(check(sys_err::SAM3_OK).is_ok());
    }

    #[test]
    fn every_error_variant_is_distinguished() {
        assert!(matches!(check(sys_err::SAM3_EINVAL), Err(Error::Invalid)));
        assert!(matches!(check(sys_err::SAM3_ENOMEM), Err(Error::NoMemory)));
        assert!(matches!(check(sys_err::SAM3_EIO), Err(Error::Io)));
        assert!(matches!(check(sys_err::SAM3_EBACKEND), Err(Error::Backend)));
        assert!(matches!(check(sys_err::SAM3_EMODEL), Err(Error::Model)));
        assert!(matches!(check(sys_err::SAM3_EDTYPE), Err(Error::Dtype)));
    }

    #[test]
    fn unknown_code_maps_to_unknown_variant() {
        // A value outside the documented SAM3_E* set must map to Error::Unknown.
        let bogus = sys_err(-999);
        assert!(matches!(check(bogus), Err(Error::Unknown(-999))));
    }

    #[test]
    fn display_renders_human_message() {
        assert_eq!(format!("{}", Error::Invalid), "invalid argument");
        assert_eq!(format!("{}", Error::Unknown(42)), "unknown SAM3 error (42)");
    }
}
