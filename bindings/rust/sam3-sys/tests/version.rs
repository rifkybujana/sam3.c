//! Verify that `sam3-sys` links and the simplest FFI call works.

#[test]
fn version_returns_non_null_c_string() {
    // SAFETY: sam3_version returns a pointer to a static C string.
    let ptr = unsafe { sam3_sys::sam3_version() };
    assert!(!ptr.is_null(), "sam3_version returned NULL");
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .expect("version must be valid UTF-8");
    assert!(!s.is_empty(), "sam3_version returned empty string");
}
