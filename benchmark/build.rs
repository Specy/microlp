//! With the `scip` feature, the bundled SCIP links dynamically and its
//! shared libraries live deep inside scip-sys's build output — the produced
//! binary would silently fail to start without them. Copy them next to the
//! executable (Windows searches the executable's directory) and embed an
//! rpath on unix, so `cargo run -p microlp-benchmark` works with no manual
//! PATH surgery.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_SCIP").is_none() {
        return;
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    // OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out; the executable
    // directory is three levels up.
    let exe_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .to_path_buf();
    let install = find_scip_install(&exe_dir.join("build")).unwrap_or_else(|| {
        panic!(
            "the scip feature is enabled but no scip-sys-*/out/scip_install exists under {} — \
             did the scip-sys build change its layout?",
            exe_dir.join("build").display()
        )
    });

    let bin = install.join("bin");
    if bin.is_dir() {
        for entry in std::fs::read_dir(&bin).into_iter().flatten().flatten() {
            let path = entry.path();
            let is_shared_lib = path
                .extension()
                .is_some_and(|e| e == "dll" || e == "so" || e == "dylib");
            if !is_shared_lib {
                continue;
            }
            let dest = exe_dir.join(path.file_name().expect("file name"));
            if let Err(e) = std::fs::copy(&path, &dest) {
                // A locked destination usually means a previous (identical)
                // copy is currently loaded by a running benchmark binary.
                println!(
                    "cargo:warning=could not copy {} next to the executable: {}",
                    path.display(),
                    e
                );
            }
        }
    }
    if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        let lib = install.join("lib");
        if lib.is_dir() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
        }
    }
}

/// The scip_install directory of the most recently built scip-sys, found by
/// scanning the profile's build directory (there is no DEP_ env var for it:
/// scip-sys is not a direct dependency of this crate).
fn find_scip_install(build_dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(build_dir).ok()?.flatten() {
        if !entry.file_name().to_string_lossy().starts_with("scip-sys-") {
            continue;
        }
        let install = entry.path().join("out").join("scip_install");
        if !install.is_dir() {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(t, _)| modified > *t) {
            best = Some((modified, install));
        }
    }
    best.map(|(_, path)| path)
}
