//! Corpus discovery: the benchmark reuses the correctness suite's vendored
//! instance files (`tests/suite/data/<tier>/<family>/…`), read through the
//! same adapter modules the suite uses, so both harnesses agree on instance
//! semantics. Only file-backed instances are benchmarked — generated unit
//! cases are all sub-millisecond and say nothing about performance.

use crate::model::{Domain, ModelSpec};
use microlp::OptimizationDirection;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq)]
pub enum Format {
    Mps,
    Lp,
}

#[derive(Clone)]
pub struct InstanceMeta {
    /// `<tier>/<family>/<file stem>` — the stable handle used on the child
    /// process command line and in the report.
    pub name: String,
    pub format: Format,
    pub path: PathBuf,
}

pub struct Instance {
    pub spec: ModelSpec,
    pub direction: OptimizationDirection,
    pub is_mip: bool,
    pub rows: usize,
    pub cols: usize,
    pub ints: usize,
    pub nnz: usize,
}

/// The repository root: `CARGO_MANIFEST_DIR` is `benchmark/`, one level down.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("benchmark crate sits inside the repository")
        .to_path_buf()
}

fn data_root() -> PathBuf {
    repo_root().join("tests").join("suite").join("data")
}

/// All benchmark instances, sorted by name. Tier folders are walked
/// recursively; every `.mps`/`.lp` file (optionally gzip-compressed as
/// `*.gz`) is one instance.
pub fn discover() -> Vec<InstanceMeta> {
    let mut out: Vec<InstanceMeta> = vec![];
    for tier in ["easy", "medium", "hard", "xhard"] {
        let dir = data_root().join(tier);
        if !dir.is_dir() {
            continue;
        }
        let mut files = vec![];
        collect_files(&dir, &mut files);
        for path in files {
            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let logical = file_name.strip_suffix(".gz").unwrap_or(&file_name);
            let (stem, format) = if let Some(s) = logical.strip_suffix(".mps") {
                (s, Format::Mps)
            } else if let Some(s) = logical.strip_suffix(".lp") {
                (s, Format::Lp)
            } else {
                continue; // READMEs and other non-instance files
            };
            let family = path
                .strip_prefix(&dir)
                .ok()
                .and_then(|rel| rel.components().next())
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".into());
            out.push(InstanceMeta {
                name: format!("{}/{}/{}", tier, family, stem),
                format,
                path: path.clone(),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    for w in out.windows(2) {
        assert_ne!(
            w[0].name, w[1].name,
            "duplicate instance name {}",
            w[0].name
        );
    }
    out
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Mirrors `tests/suite/cases/mod.rs::read_instance` (kept in lockstep):
/// gzipped storage is decompressed in memory, nothing touches the disk.
fn read_text(path: &Path) -> Result<String, String> {
    let err = |e: &dyn std::fmt::Display| format!("cannot read {}: {}", path.display(), e);
    if path.extension().is_some_and(|ext| ext == "gz") {
        let file = std::fs::File::open(path).map_err(|e| err(&e))?;
        let mut text = String::new();
        std::io::Read::read_to_string(
            &mut flate2::read::GzDecoder::new(std::io::BufReader::new(file)),
            &mut text,
        )
        .map_err(|e| err(&e))?;
        Ok(text)
    } else {
        std::fs::read_to_string(path).map_err(|e| err(&e))
    }
}

/// Parse an instance into the neutral shadow model every contender builds
/// its native representation from. Parsing happens once, outside any timed
/// region.
pub fn load(meta: &InstanceMeta) -> Result<Instance, String> {
    let text = read_text(&meta.path)?;
    let (spec, direction) = match meta.format {
        // Every vendored MPS instance (netlib, MIPLIB 3) is a minimization
        // problem; the adapter rejects a contradicting in-file OBJSENSE.
        Format::Mps => {
            let parsed = crate::mps_milp::parse(&text, OptimizationDirection::Minimize, false)?;
            (parsed.spec, OptimizationDirection::Minimize)
        }
        Format::Lp => {
            let parsed = crate::lp_format::parse(&text, false)?;
            (parsed.spec, parsed.direction)
        }
    };
    let ints = spec
        .vars
        .iter()
        .filter(|v| v.domain == Domain::Integer)
        .count();
    let nnz = spec.constraints.iter().map(|c| c.terms.len()).sum();
    Ok(Instance {
        rows: spec.constraints.len(),
        cols: spec.vars.len(),
        ints,
        nnz,
        is_mip: ints > 0,
        direction,
        spec,
    })
}
