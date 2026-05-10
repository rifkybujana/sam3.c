//! Build script for `sam3-sys`.
//!
//! Locates `libsam3` (installed or from a sibling CMake build directory),
//! runs `bindgen` against `sam3/sam3.h`, and emits native-link directives.

use std::env;
use std::fs;
use std::panic;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=SAM3_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SAM3_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SAM3_INCLUDE_DIR");

    let (lib_dir, include_dir) = resolve_paths();

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=sam3");
    copy_runtime_dlls(&lib_dir);

    // Bake an rpath so binaries find a co-located libsam3 after install.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

    let generated = panic::catch_unwind(|| {
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
    });

    match generated {
        Ok(Ok(bindings)) => bindings
            .write_to_file(out_path)
            .expect("failed to write bindings.rs"),
        Ok(Err(err)) => {
            if target_os() == "windows" {
                write_fallback_bindings(&out_path, &format!("bindgen failed: {err}"));
            } else {
                panic!("bindgen failed to generate sam3 bindings: {err}");
            }
        }
        Err(_) => {
            if target_os() == "windows" {
                write_fallback_bindings(&out_path, "bindgen panicked (libclang may be missing)");
            } else {
                panic::resume_unwind(Box::new("bindgen panicked"));
            }
        }
    }
}

fn write_fallback_bindings(out_path: &Path, reason: &str) {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let fallback = manifest.join("src").join("fallback_bindings.rs");
    println!(
        "cargo:warning=sam3-sys: using checked-in fallback bindings because {}",
        reason
    );
    println!("cargo:rerun-if-changed={}", fallback.display());
    fs::copy(&fallback, out_path).expect("failed to copy fallback bindings.rs");
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
        if let Some(paths) = resolve_build_dir(&build_dir) {
            return paths;
        }
        panic!(
            "sam3-sys: SAM3_BUILD_DIR={} has no supported SAM3 library artifact. \
             Expected sam3.dll plus sam3.lib under Release/, Debug/, or the build dir \
             on Windows, or libsam3.{{dylib,so}} on Unix.",
            build_dir
        );
    }

    // 3. Auto-detect a sibling build/ by walking up from CARGO_MANIFEST_DIR.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut cur: &Path = &manifest;
    loop {
        let candidate_build = cur.join("build");
        let candidate_include = cur.join("include");
        if let Some(lib_dir) = find_lib_dir(&candidate_build) {
            if candidate_include.join("sam3").join("sam3.h").is_file() {
                return (lib_dir, candidate_include);
            }
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => break,
        }
    }

    panic!(
        "sam3-sys: unable to locate libsam3. Set SAM3_LIB_DIR and SAM3_INCLUDE_DIR, \
         or SAM3_BUILD_DIR, or ensure a supported CMake build artifact exists under \
         the repository root (run `cmake -S . -B build -DSAM3_SHARED=ON && cmake --build build`)."
    );
}

fn resolve_build_dir(build_dir: &str) -> Option<(PathBuf, PathBuf)> {
    for build in build_dir_candidates(build_dir) {
        let include = build
            .parent()
            .map(|p| p.join("include"))
            .unwrap_or_else(|| PathBuf::from("include"));
        let Some(lib_dir) = find_lib_dir(&build) else {
            continue;
        };
        if include.join("sam3").join("sam3.h").is_file() {
            return Some((lib_dir, include));
        }
        println!(
            "cargo:warning=sam3-sys: candidate SAM3_BUILD_DIR={} has library artifacts \
             but inferred include dir {} has no sam3/sam3.h",
            build.display(),
            include.display()
        );
    }
    None
}

fn build_dir_candidates(build_dir: &str) -> Vec<PathBuf> {
    let path = PathBuf::from(build_dir);
    if path.is_absolute() {
        return vec![path];
    }

    let mut candidates = Vec::new();
    candidates.push(path.clone());

    if let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") {
        let manifest = PathBuf::from(manifest);
        candidates.push(manifest.join(&path));
        if let Some(workspace) = manifest.parent() {
            candidates.push(workspace.join(&path));
        }
    }

    candidates
}

fn has_lib(dir: &Path) -> bool {
    dir.join("sam3.lib").is_file()
        || dir.join("libsam3.dll.a").is_file()
        || dir.join("sam3.dll").is_file()
        || dir.join("libsam3.dylib").is_file()
        || dir.join("libsam3.so").is_file()
}

fn find_lib_dir(build: &Path) -> Option<PathBuf> {
    for dir in [build.join("Release"), build.join("Debug"), build.to_path_buf()] {
        if has_lib(&dir) && is_linkable_dir(&dir) {
            return Some(dir);
        }
    }
    None
}

fn is_linkable_dir(dir: &Path) -> bool {
    if target_os() == "windows" {
        let has_import_lib = dir.join("sam3.lib").is_file() || dir.join("libsam3.dll.a").is_file();
        return has_import_lib && dir.join("sam3.dll").is_file();
    }
    dir.join("libsam3.dylib").is_file() || dir.join("libsam3.so").is_file()
}

fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| env::consts::OS.to_string())
}

fn copy_runtime_dlls(lib_dir: &Path) {
    if target_os() != "windows" {
        return;
    }

    let Some(profile_dir) = cargo_profile_dir() else {
        println!("cargo:warning=sam3-sys: could not derive Cargo profile dir from OUT_DIR");
        return;
    };

    copy_dlls_to(lib_dir, &profile_dir);
    copy_dlls_to(lib_dir, &profile_dir.join("deps"));
}

fn cargo_profile_dir() -> Option<PathBuf> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").ok()?);
    out_dir.ancestors().nth(3).map(Path::to_path_buf)
}

fn copy_dlls_to(lib_dir: &Path, dst_dir: &Path) {
    let Ok(entries) = fs::read_dir(lib_dir) else {
        return;
    };
    if let Err(err) = fs::create_dir_all(dst_dir) {
        println!(
            "cargo:warning=sam3-sys: could not create runtime dir {}: {}",
            dst_dir.display(),
            err
        );
        return;
    }

    for entry in entries.flatten() {
        let src = entry.path();
        let is_dll = src
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("dll"))
            .unwrap_or(false);
        if !is_dll {
            continue;
        }
        let Some(name) = src.file_name() else {
            continue;
        };
        let dst = dst_dir.join(name);
        println!("cargo:rerun-if-changed={}", src.display());
        if let Err(err) = fs::copy(&src, &dst) {
            println!(
                "cargo:warning=sam3-sys: could not copy {} to {}: {}",
                src.display(),
                dst.display(),
                err
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_lib_recognizes_windows_and_unix_names() {
        let dir = env::temp_dir().join(format!(
            "sam3-sys-build-rs-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        for name in [
            "sam3.lib",
            "libsam3.dll.a",
            "sam3.dll",
            "libsam3.dylib",
            "libsam3.so",
        ] {
            for entry in fs::read_dir(&dir).unwrap().flatten() {
                let _ = fs::remove_file(entry.path());
            }
            fs::write(dir.join(name), b"").unwrap();
            assert!(has_lib(&dir), "{name} should be recognized");
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
