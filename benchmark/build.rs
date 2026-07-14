//! Make the bundled SCIP runtime discoverable by benchmark executables. Cargo
//! supplies its library directory through `scip-sys` link metadata: Windows
//! DLLs are copied next to the executable, while Unix targets receive an rpath.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_SCIP").is_none() {
        return;
    }

    let lib_dir = scip_lib_dir();
    let windows = std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("windows");
    if windows {
        copy_windows_runtime(&lib_dir);
    } else {
        let linux_target = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
        configure_unix_rpath(&lib_dir, linux_target);
    }
}

fn scip_lib_dir() -> PathBuf {
    println!("cargo:rerun-if-env-changed=DEP_SCIP_LIBDIR");
    PathBuf::from(
        std::env::var_os("DEP_SCIP_LIBDIR")
            .expect("scip feature enabled but scip-sys did not provide DEP_SCIP_LIBDIR"),
    )
}

fn copy_windows_runtime(lib_dir: &Path) {
    let install = lib_dir.parent().unwrap_or_else(|| {
        panic!(
            "DEP_SCIP_LIBDIR has no installation parent: {}",
            lib_dir.display()
        )
    });
    let bin_dir = install.join("bin");
    let dlls: Vec<PathBuf> = std::fs::read_dir(&bin_dir)
        .unwrap_or_else(|error| {
            panic!(
                "cannot read SCIP runtime directory {}: {}",
                bin_dir.display(),
                error
            )
        })
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "dll"))
        .collect();
    assert!(
        !dlls.is_empty(),
        "DEP_SCIP_LIBDIR points to an installation without Windows DLLs: {}",
        lib_dir.display()
    );

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let exe_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .to_path_buf();
    for path in dlls {
        let destination = exe_dir.join(path.file_name().expect("DLL has a file name"));
        if let Err(error) = std::fs::copy(&path, &destination) {
            // The destination can be locked while an identical benchmark binary
            // is running; keep the existing runtime and surface the copy failure.
            println!(
                "cargo:warning=could not copy {} next to the executable: {}",
                path.display(),
                error
            );
        }
    }
}

fn configure_unix_rpath(lib_dir: &Path, linux_target: bool) {
    let has_scip = std::fs::read_dir(lib_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .any(|name| {
            name.starts_with("libscip.so")
                || (name.starts_with("libscip") && name.contains(".dylib"))
        });
    assert!(
        has_scip,
        "DEP_SCIP_LIBDIR contains no SCIP shared library: {}",
        lib_dir.display()
    );

    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    if linux_target {
        // SCIP's shared library has runtime dependencies in the same directory.
        // Classic DT_RPATH applies transitively, unlike DT_RUNPATH.
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    }
}
