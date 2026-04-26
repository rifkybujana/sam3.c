//! Build script for `sam3-sys`.
//!
//! Locates `libsam3` (installed or from a sibling CMake build directory),
//! runs `bindgen` against `sam3/sam3.h`, and emits native-link directives.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=SAM3_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SAM3_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SAM3_INCLUDE_DIR");

    let (lib_dir, include_dir) = resolve_paths();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=sam3");

    // Bake an rpath so binaries find a co-located libsam3 after install.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    // Windows has no rpath — `libsam3.dll` must sit next to the .exe.
    // The Tauri host crate's build.rs copies it into the target dir.

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

    bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("sam3_.*")
        .allowlist_type("sam3_.*")
        .allowlist_var("SAM3_.*")
        .prepend_enum_name(false)
        .newtype_enum("sam3_error")
        .rustified_enum("sam3_log_level")
        .rustified_enum("sam3_dtype")
        .rustified_enum("sam3_prompt_type")
        .rustified_enum("sam3_backbone_type")
        .derive_default(true)
        .derive_debug(true)
        .layout_tests(true)
        .generate()
        .expect("bindgen failed to generate sam3 bindings")
        .write_to_file(out_path)
        .expect("failed to write bindings.rs");
}

/// Resolve `(lib_dir, include_dir)` via env vars or auto-detection.
fn resolve_paths() -> (PathBuf, PathBuf) {
    let lib_env = env::var("SAM3_LIB_DIR").ok();
    let inc_env = env::var("SAM3_INCLUDE_DIR").ok();
    if lib_env.is_some() ^ inc_env.is_some() {
        println!(
            "cargo:warning=sam3-sys: SAM3_LIB_DIR and SAM3_INCLUDE_DIR must both be set; \
             ignoring the single-variable override and falling through to auto-detect."
        );
    }

    // 1. Explicit override.
    if let (Some(lib), Some(inc)) = (lib_env, inc_env) {
        return (PathBuf::from(lib), PathBuf::from(inc));
    }

    // 2. SAM3_BUILD_DIR with inferred include dir (explicit request — do not silently fall through).
    if let Ok(build_dir) = env::var("SAM3_BUILD_DIR") {
        let build = PathBuf::from(&build_dir);
        let include = build
            .parent()
            .map(|p| p.join("include"))
            .unwrap_or_else(|| PathBuf::from("include"));
        if !has_lib(&build) {
            panic!(
                "sam3-sys: SAM3_BUILD_DIR={} has no libsam3.{{dylib,so,dll.a}}",
                build.display()
            );
        }
        if !include.join("sam3").join("sam3.h").is_file() {
            panic!(
                "sam3-sys: SAM3_BUILD_DIR={} but inferred include dir {} has no sam3/sam3.h",
                build.display(),
                include.display()
            );
        }
        return (build, include);
    }

    // 3. Auto-detect a sibling build/ by walking up from CARGO_MANIFEST_DIR.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut cur: &Path = &manifest;
    loop {
        let candidate_build = cur.join("build");
        let candidate_include = cur.join("include");
        if has_lib(&candidate_build) && candidate_include.join("sam3").join("sam3.h").is_file() {
            return (candidate_build, candidate_include);
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => break,
        }
    }

    panic!(
        "sam3-sys: unable to locate libsam3. Set SAM3_LIB_DIR and SAM3_INCLUDE_DIR, \
         or SAM3_BUILD_DIR, or ensure `build/libsam3.{{dylib,so,dll.a}}` exists under \
         the repository root (run `cmake -S . -B build -DSAM3_SHARED=ON && cmake --build build`)."
    );
}

fn has_lib(dir: &Path) -> bool {
    dir.join("libsam3.dylib").is_file()
        || dir.join("libsam3.so").is_file()
        // MinGW-w64 import library — the .dll itself sits next to it
        // and is loaded at runtime from the same directory or PATH.
        || dir.join("libsam3.dll.a").is_file()
}
