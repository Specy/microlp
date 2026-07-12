//! Case registry: every family contributes `Case`s with a stable unique name.

use crate::model::{Expected, ModelSpec};
use microlp::Problem;
use std::time::Duration;

mod incremental;
mod lp_random;
mod lp_unit;
mod milp_families;
mod milp_random;
mod milp_unit;
mod milpbench;
mod milpbench_xhard;
mod miplib;
mod netlib;
mod warm_restart;

/// Difficulty tiers, cumulative: a tier flag on the runner selects that tier
/// *and everything below it* (`--hard` runs easy + medium + hard). The
/// variant order is the selection order, so `Ord` is derived.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Tier {
    /// Milliseconds each; runs everywhere, including debug CI.
    Easy,
    /// Laptop-friendly; the release-mode default runs easy + medium.
    Medium,
    /// From `--hard` up: instances a branch & bound may take minutes on.
    Hard,
    /// Only under `--xhard`: instances with 10-minute default budgets that
    /// microlp is not expected to finish (they assert clean interrupts and
    /// bound sanity against externally certified optima).
    XHard,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::Easy => "easy",
            Tier::Medium => "medium",
            Tier::Hard => "hard",
            Tier::XHard => "xhard",
        }
    }

    fn all() -> [Tier; 4] {
        [Tier::Easy, Tier::Medium, Tier::Hard, Tier::XHard]
    }
}

/// Locate a benchmark data file and derive its tier from the folder it lives
/// in: `tests/suite/data/<tier>/<sub>/<file>`. Moving a file between tier
/// folders re-tiers its cases with no code change.
///
/// Large instances may be stored gzipped as `<file>.gz` to keep the repo
/// small; callers always ask for the logical name and the storage form is
/// resolved here (read the result with [`read_instance`], which decompresses
/// in memory). Compressing or un-compressing a file is therefore a pure
/// data-directory change, like moving it between tiers.
///
/// When the file is absent from every tier folder (e.g. a source checkout
/// without the data directory, which is excluded from the crates.io package),
/// this falls back to the medium-tier path so registration never panics; the
/// case itself then fails with a "cannot read" error if selected.
pub fn locate(sub: &str, file: &str) -> (std::path::PathBuf, Tier) {
    let data = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("suite")
        .join("data");
    for tier in Tier::all() {
        let dir = data.join(tier.label()).join(sub);
        let path = dir.join(file);
        if path.is_file() {
            return (path, tier);
        }
        let gz_path = dir.join(format!("{}.gz", file));
        if gz_path.is_file() {
            return (gz_path, tier);
        }
    }
    (data.join("medium").join(sub).join(file), Tier::Medium)
}

/// Read a benchmark data file located by [`locate`] into a string,
/// decompressing gzipped (`*.gz`) storage in memory — nothing is ever
/// written to disk.
pub fn read_instance(path: &std::path::Path) -> Result<String, String> {
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

pub enum CaseRun {
    /// Build (shadow model, problem, expectation); the runner solves+verifies.
    #[allow(clippy::type_complexity)]
    Solve(Box<dyn Fn() -> Result<(ModelSpec, Problem, Expected), String> + Send + Sync>),
    /// Self-contained check (e.g. metamorphic invariants). Receives the time
    /// budget so it can set per-solve limits; returns Err(reason) on failure.
    Custom(Box<dyn Fn(Duration) -> Result<(), String> + Send + Sync>),
}

pub struct Case {
    pub name: String,
    pub tier: Tier,
    pub budget: Duration,
    pub run: CaseRun,
}

impl Case {
    pub fn solve(
        name: impl Into<String>,
        tier: Tier,
        budget_secs: u64,
        build: impl Fn() -> Result<(ModelSpec, Problem, Expected), String> + Send + Sync + 'static,
    ) -> Case {
        Case {
            name: name.into(),
            tier,
            budget: Duration::from_secs(budget_secs),
            run: CaseRun::Solve(Box::new(build)),
        }
    }

    pub fn custom(
        name: impl Into<String>,
        tier: Tier,
        budget_secs: u64,
        run: impl Fn(Duration) -> Result<(), String> + Send + Sync + 'static,
    ) -> Case {
        Case {
            name: name.into(),
            tier,
            budget: Duration::from_secs(budget_secs),
            run: CaseRun::Custom(Box::new(run)),
        }
    }
}

pub fn all() -> Vec<Case> {
    let mut cases = vec![];
    lp_unit::register(&mut cases);
    lp_random::register(&mut cases);
    milp_unit::register(&mut cases);
    milp_random::register(&mut cases);
    milp_families::register(&mut cases);
    incremental::register(&mut cases);
    netlib::register(&mut cases);
    miplib::register(&mut cases);
    milpbench::register(&mut cases);
    milpbench_xhard::register(&mut cases);
    warm_restart::register(&mut cases);

    // Names must be unique: they are the stable handle for --filter and repro.
    let mut seen = std::collections::BTreeSet::new();
    for c in &cases {
        assert!(
            seen.insert(c.name.clone()),
            "duplicate case name {}",
            c.name
        );
    }
    cases
}
