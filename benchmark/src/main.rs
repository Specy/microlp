//! Benchmark harness: solves the correctness suite's vendored LP/MILP
//! instances with microlp and rival open-source solvers, then writes a
//! Markdown report (`BENCHMARK.md` at the repository root by default).
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p microlp-benchmark --release
//! ```
//!
//! Architecture: the orchestrator process enumerates (instance × solver)
//! pairs and re-invokes this same binary once per pair (`run-one` mode).
//! One process per measurement keeps solvers from interfering with each
//! other (allocator state, caches, FFI globals) and turns a crash, hang or
//! out-of-memory in any single solve into a recorded failure instead of a
//! dead benchmark run.
//!
//! Every solver is timed over the same work: building its native model from
//! the shared instance data plus solving it. File parsing is excluded for
//! everyone. Fast solves are repeated and the fastest run is kept (the
//! corpus is deterministic; repeats only shave scheduler noise).

#[allow(dead_code)]
#[path = "../../tests/suite/lp_format.rs"]
mod lp_format;
#[allow(dead_code)]
#[path = "../../tests/suite/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "../../tests/suite/mps_milp.rs"]
mod mps_milp;

mod contenders;
mod corpus;
mod report;

use contenders::{RunOutcome, RunStatus, SolveTask};
use corpus::{Instance, InstanceMeta};
use microlp::ComparisonOp;
use report::{CaseResult, InstanceInfo, RunData};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const DEFAULT_TIME_LIMIT_SECS: f64 = 300.0;
const DEFAULT_MIN_MS: f64 = 1.0;

const USAGE: &str = "\
microlp benchmark harness — writes BENCHMARK.md

USAGE:
  cargo run -p microlp-benchmark --release [--] [OPTIONS]

OPTIONS:
  --time-limit <secs>   per-solver budget for every instance (default 300)
  --mip-gap <fraction>  relative MIP gap at which every solver may stop and
                        report the optimum (default 0 = exact; e.g. 0.01
                        accepts anything proven within 1%). Reference solves
                        stay exact regardless.
  --solvers <a,b,...>   run only these solvers (default: all compiled in)
  --filter <substr>     run only instances whose name contains this
                        (repeatable; matches are OR-ed)
  --out <path>          report path (default: <repo root>/BENCHMARK.md)
  --min-ms <float>      hide instances every solver finishes faster than
                        this many milliseconds (default 1.0)
  --list                print the selected instances and solvers, then exit
  --help                this text

Solvers are cargo features (highs, scip, clarabel — all on by default);
pick a subset at runtime with --solvers, or at compile time with
  cargo run -p microlp-benchmark --release --no-default-features --features highs
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("run-one") {
        run_one(&args[1..]);
        return;
    }
    orchestrate(Config::parse(&args));
}

struct Config {
    time_limit: f64,
    mip_gap: f64,
    min_ms: f64,
    filters: Vec<String>,
    solvers: Vec<String>,
    out: PathBuf,
    list: bool,
}

impl Config {
    fn parse(args: &[String]) -> Config {
        let mut cfg = Config {
            time_limit: DEFAULT_TIME_LIMIT_SECS,
            mip_gap: 0.0,
            min_ms: DEFAULT_MIN_MS,
            filters: vec![],
            solvers: contenders::all()
                .iter()
                .map(|c| c.name().to_string())
                .collect(),
            out: corpus::repo_root().join("BENCHMARK.md"),
            list: false,
        };
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            let mut value = |what: &str| {
                it.next()
                    .unwrap_or_else(|| die(&format!("{} needs a value", what)))
                    .clone()
            };
            match arg.as_str() {
                "--time-limit" => {
                    cfg.time_limit = value("--time-limit")
                        .parse()
                        .unwrap_or_else(|_| die("--time-limit expects seconds"));
                }
                "--mip-gap" => {
                    cfg.mip_gap = value("--mip-gap")
                        .parse()
                        .unwrap_or_else(|_| die("--mip-gap expects a fraction, e.g. 0.01"));
                }
                "--min-ms" => {
                    cfg.min_ms = value("--min-ms")
                        .parse()
                        .unwrap_or_else(|_| die("--min-ms expects a float"));
                }
                "--filter" => {
                    let v = value("--filter");
                    cfg.filters.push(v);
                }
                "--solvers" => {
                    cfg.solvers = value("--solvers")
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "--out" => cfg.out = PathBuf::from(value("--out")),
                "--list" => cfg.list = true,
                "--help" | "-h" => {
                    print!("{}", USAGE);
                    std::process::exit(0);
                }
                other => die(&format!("unknown argument {} (see --help)", other)),
            }
        }
        if !(cfg.time_limit.is_finite() && cfg.time_limit > 0.0) {
            die("--time-limit must be positive");
        }
        if !(0.0..1.0).contains(&cfg.mip_gap) {
            die("--mip-gap must be in [0, 1)");
        }
        cfg
    }
}

fn die(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(2);
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

fn orchestrate(cfg: Config) {
    let available: Vec<String> = contenders::all()
        .iter()
        .map(|c| c.name().to_string())
        .collect();
    for s in &cfg.solvers {
        if !available.contains(s) {
            die(&format!(
                "unknown solver {} — compiled-in solvers: {} (others are cargo features of the benchmark crate)",
                s,
                available.join(", ")
            ));
        }
    }
    let metas: Vec<InstanceMeta> = corpus::discover()
        .into_iter()
        .filter(|m| {
            cfg.filters.is_empty() || cfg.filters.iter().any(|f| m.name.contains(f.as_str()))
        })
        .collect();
    if cfg.list {
        for m in &metas {
            println!("{}", m.name);
        }
        println!(
            "({} instances; solvers: {})",
            metas.len(),
            cfg.solvers.join(", ")
        );
        return;
    }
    if metas.is_empty() {
        die("no instances matched the filters");
    }

    let exe = std::env::current_exe().expect("cannot locate the benchmark executable");
    let total = metas.len() * cfg.solvers.len();
    eprintln!(
        "benchmarking {} instances x {} solvers, {} s budget per solve ({} solves; runs are sequential so timings stay clean)",
        metas.len(),
        cfg.solvers.len(),
        cfg.time_limit,
        total,
    );

    let mut data = RunData {
        instances: vec![],
        references: vec![],
        solvers: cfg.solvers.clone(),
        time_limit_secs: cfg.time_limit,
        mip_gap: cfg.mip_gap,
        min_ms: cfg.min_ms,
        partial: !cfg.filters.is_empty() || cfg.solvers != available,
    };
    let started = Instant::now();
    let mut done = 0usize;
    for meta in &metas {
        let mut sizes: Option<Sizes> = None;
        let mut results: Vec<CaseResult> = vec![];
        for solver in &cfg.solvers {
            done += 1;
            // One line when a solve starts, one when it ends — a 300 s
            // timeout would otherwise be five silent minutes with no way to
            // tell what the run is doing.
            eprintln!(
                "[{:>3}/{}] {:<45} {:<10} started · elapsed {} · worst case {} of solving left",
                done,
                total,
                meta.name,
                solver,
                fmt_span(started.elapsed().as_secs_f64()),
                fmt_span((total - done + 1) as f64 * cfg.time_limit),
            );
            let (s, res) = run_child(&exe, meta, solver, cfg.time_limit, cfg.mip_gap, false);
            if sizes.is_none() {
                sizes = s;
            }
            eprintln!(
                "[{:>3}/{}] {:<45} {:<10} {}",
                done,
                total,
                meta.name,
                solver,
                progress_summary(&res)
            );
            results.push(res);
        }
        let s = sizes.unwrap_or(Sizes {
            rows: 0,
            cols: 0,
            ints: 0,
            nnz: 0,
            is_mip: false,
            maximize: false,
        });
        data.instances.push((
            InstanceInfo {
                name: meta.name.clone(),
                rows: s.rows,
                cols: s.cols,
                ints: s.ints,
                nnz: s.nnz,
                is_mip: s.is_mip,
                maximize: s.maximize,
            },
            results,
        ));
    }
    eprintln!(
        "all solves done in {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );

    reference_pass(&cfg, &exe, &metas, &mut data);

    let md = report::render(&data);
    std::fs::write(&cfg.out, md)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", cfg.out.display(), e)));
    eprintln!("wrote {}", cfg.out.display());
}

/// For every instance microlp left unfinished and *no* solver proved inside
/// the shared budget, give the primary rival one longer, untimed
/// certification solve (all threads, still exact) so the correction-gap
/// column has an independent reference optimum.
fn reference_pass(cfg: &Config, exe: &Path, metas: &[InstanceMeta], data: &mut RunData) {
    let Some(rival) = cfg.solvers.iter().find(|s| *s != "microlp").cloned() else {
        return;
    };
    let ref_budget = (cfg.time_limit * 12.0).clamp(120.0, 600.0);
    let needing: Vec<&InstanceMeta> = metas
        .iter()
        .filter(|meta| {
            data.instances.iter().any(|(info, results)| {
                info.name == meta.name
                    && results
                        .iter()
                        .any(|r| r.solver == "microlp" && r.budget_limited())
                    && !results.iter().any(|r| r.optimal())
            })
        })
        .collect();
    if needing.is_empty() {
        return;
    }
    eprintln!(
        "reference pass: {} unproven instances x {} at {} s (untimed certification solves)",
        needing.len(),
        rival,
        ref_budget
    );
    for (i, meta) in needing.iter().enumerate() {
        eprintln!(
            "[ref {:>2}/{}] {:<45} started · up to {} each",
            i + 1,
            needing.len(),
            meta.name,
            fmt_span(ref_budget),
        );
        // References must stay exact whatever --mip-gap says: correction
        // gaps are measured against them.
        let (_, res) = run_child(exe, meta, &rival, ref_budget, 0.0, true);
        eprintln!(
            "[ref {:>2}/{}] {:<45} {}",
            i + 1,
            needing.len(),
            meta.name,
            progress_summary(&res)
        );
        if res.optimal() {
            if let Some(objective) = res.objective {
                data.references.push(report::ReferenceResult {
                    instance: meta.name.clone(),
                    solver: rival.clone(),
                    objective,
                    budget_secs: ref_budget,
                });
            }
        }
    }
}

/// Compact duration for progress lines ("42 s", "18 min", "2.3 h").
fn fmt_span(secs: f64) -> String {
    if secs < 90.0 {
        format!("{:.0} s", secs)
    } else if secs < 5400.0 {
        format!("{:.0} min", secs / 60.0)
    } else {
        format!("{:.1} h", secs / 3600.0)
    }
}

fn progress_summary(r: &CaseResult) -> String {
    let status: String = r.status.chars().take(70).collect();
    match r.ms {
        Some(ms) => format!("{:>10}  {}", report::fmt_time(ms), status),
        None => status,
    }
}

struct Sizes {
    rows: usize,
    cols: usize,
    ints: usize,
    nnz: usize,
    is_mip: bool,
    maximize: bool,
}

/// Run one (instance, solver) measurement in a child process and parse its
/// protocol lines. The hard deadline is only a net for solves that hang or
/// ignore their budget — parsing the largest instances alone takes a while,
/// so it is deliberately generous.
fn run_child(
    exe: &Path,
    meta: &InstanceMeta,
    solver: &str,
    limit: f64,
    mip_gap: f64,
    reference: bool,
) -> (Option<Sizes>, CaseResult) {
    let mut args = vec![
        "run-one".to_string(),
        "--instance".to_string(),
        meta.name.clone(),
        "--solver".to_string(),
        solver.to_string(),
        "--time-limit".to_string(),
        limit.to_string(),
        "--mip-gap".to_string(),
        mip_gap.to_string(),
    ];
    if reference {
        args.push("--reference".to_string());
    }
    let child = Command::new(exe)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return (
                None,
                CaseResult::failed(solver, format!("spawn failed: {}", e)),
            )
        }
    };

    // Drain both pipes on threads so a chatty child can never fill a pipe
    // buffer and deadlock against our wait loop.
    let mut stdout_pipe = child.stdout.take().expect("piped stdout");
    let stdout_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stdout_pipe.read_to_string(&mut s);
        s
    });
    let mut stderr_pipe = child.stderr.take().expect("piped stderr");
    let stderr_thread = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr_pipe.read_to_string(&mut s);
        s
    });

    let hard_deadline = Duration::from_secs_f64(limit * 2.0 + 120.0);
    let started = Instant::now();
    let exit = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if started.elapsed() > hard_deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();

    let sizes = parse_sizes_line(&stdout);
    let result = match exit {
        None => CaseResult::failed(
            solver,
            format!(
                "killed: exceeded the {:.0} s hard deadline",
                hard_deadline.as_secs_f64()
            ),
        ),
        Some(status) => match parse_result_line(&stdout, solver) {
            Some(r) => r,
            None => CaseResult::failed(
                solver,
                format!(
                    "crash: exit {:?}; stderr: {}",
                    status.code(),
                    tail(&stderr, 300)
                ),
            ),
        },
    };
    (sizes, result)
}

fn tail(s: &str, n: usize) -> String {
    let t: String = s
        .chars()
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    t.replace(['\n', '\r', '\t'], " ").trim().to_string()
}

fn parse_sizes_line(stdout: &str) -> Option<Sizes> {
    let line = stdout.lines().find(|l| l.starts_with("SIZES\t"))?;
    let f: Vec<&str> = line.split('\t').collect();
    if f.len() != 7 {
        return None;
    }
    Some(Sizes {
        rows: f[1].parse().ok()?,
        cols: f[2].parse().ok()?,
        ints: f[3].parse().ok()?,
        nnz: f[4].parse().ok()?,
        is_mip: f[5] == "1",
        maximize: f[6] == "max",
    })
}

fn parse_result_line(stdout: &str, solver: &str) -> Option<CaseResult> {
    let line = stdout.lines().rfind(|l| l.starts_with("RESULT\t"))?;
    let f: Vec<&str> = line.splitn(8, '\t').collect();
    if f.len() != 8 {
        return None;
    }
    let opt_f64 = |s: &str| {
        if s == "-" {
            None
        } else {
            s.parse::<f64>().ok()
        }
    };
    // Fields 5 and 6 (microlp's node and simplex-iteration counts) are still
    // emitted by children but not used by the report.
    Some(CaseResult {
        solver: solver.to_string(),
        ms: opt_f64(f[1]),
        objective: opt_f64(f[2]),
        bound: opt_f64(f[3]),
        gap: opt_f64(f[4]),
        status: f[7].to_string(),
    })
}

// ---------------------------------------------------------------------------
// Child mode: one measurement, protocol lines on stdout
// ---------------------------------------------------------------------------

fn run_one(args: &[String]) {
    let mut instance = None;
    let mut solver = None;
    let mut limit = DEFAULT_TIME_LIMIT_SECS;
    let mut mip_gap = 0.0;
    let mut reference = false;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--instance" => instance = it.next().cloned(),
            "--solver" => solver = it.next().cloned(),
            "--time-limit" => {
                limit = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_TIME_LIMIT_SECS)
            }
            "--mip-gap" => mip_gap = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            "--reference" => reference = true,
            _ => die("run-one expects --instance, --solver, --time-limit, --mip-gap, --reference"),
        }
    }
    let (Some(instance), Some(solver)) = (instance, solver) else {
        die("run-one expects --instance and --solver");
    };

    let metas = corpus::discover();
    let Some(meta) = metas.iter().find(|m| m.name == instance) else {
        emit_result(
            None,
            &RunOutcome::bare(RunStatus::Error(format!("unknown instance {}", instance))),
        );
        return;
    };
    let inst = match corpus::load(meta) {
        Ok(i) => i,
        Err(e) => {
            emit_result(
                None,
                &RunOutcome::bare(RunStatus::Error(format!("parse: {}", e))),
            );
            return;
        }
    };
    println!(
        "SIZES\t{}\t{}\t{}\t{}\t{}\t{}",
        inst.rows,
        inst.cols,
        inst.ints,
        inst.nnz,
        inst.is_mip as u8,
        match inst.direction {
            microlp::OptimizationDirection::Maximize => "max",
            microlp::OptimizationDirection::Minimize => "min",
        }
    );

    let Some(contender) = contenders::by_name(&solver) else {
        emit_result(
            None,
            &RunOutcome::bare(RunStatus::Error(format!("unknown solver {}", solver))),
        );
        return;
    };
    if inst.is_mip && !contender.supports_mip() {
        emit_result(
            None,
            &RunOutcome::bare(RunStatus::Error(
                "unsupported: LP-only solver on a MILP instance".into(),
            )),
        );
        return;
    }

    let task = SolveTask {
        budget: Duration::from_secs_f64(limit),
        reference,
        // Correction gaps are measured against reference optima, so those
        // stay exact whatever gap the timed run uses.
        mip_gap: if reference { 0.0 } else { mip_gap },
    };
    let mut best: Option<(f64, RunOutcome)> = None;
    let loop_started = Instant::now();
    for run in 0.. {
        let t0 = Instant::now();
        let out = contender.run(&inst, &task);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if best.as_ref().is_none_or(|(b, _)| ms < *b) {
            best = Some((ms, out));
        }
        // Reference solves are untimed certifications: one run is enough.
        if reference || ms >= 200.0 || run >= 4 || loop_started.elapsed() >= Duration::from_secs(2)
        {
            break;
        }
    }
    let (ms, mut out) = best.expect("at least one run");

    // Re-check any returned solution against the shadow model. A conversion
    // bug in a contender would otherwise poison every downstream comparison,
    // so a failure here is loud and counts as that solver failing the case.
    if let Some(values) = &out.values {
        if let Err(e) = validate(&inst, values, out.objective) {
            out = RunOutcome::bare(RunStatus::Error(format!("validation failed: {}", e)));
        }
    }
    emit_result(Some(ms), &out);
}

fn emit_result(ms: Option<f64>, out: &RunOutcome) {
    let f = |x: Option<f64>| x.map(|v| format!("{:?}", v)).unwrap_or_else(|| "-".into());
    let u = |x: Option<u64>| x.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
    // The status is the last field and must stay a single line.
    let status = out.status.label().replace(['\t', '\n', '\r'], " ");
    println!(
        "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        f(ms),
        f(out.objective),
        f(out.bound),
        f(out.gap),
        u(out.nodes),
        u(out.simplex_iters),
        status
    );
}

/// Independent solution check: bounds, integrality, constraint satisfaction
/// and objective recomputation against the shadow model. Tolerances are
/// scale-aware and generous relative to every solver's defaults — the point
/// is to catch model-conversion bugs (magnitude ~1), not to relitigate
/// solver feasibility tolerances.
fn validate(inst: &Instance, values: &[f64], reported_obj: Option<f64>) -> Result<(), String> {
    let spec = &inst.spec;
    if values.len() != spec.vars.len() {
        return Err(format!(
            "{} values for {} variables",
            values.len(),
            spec.vars.len()
        ));
    }
    for (i, (v, &x)) in spec.vars.iter().zip(values).enumerate() {
        if !x.is_finite() {
            return Err(format!("variable {} is {}", i, x));
        }
        let slack = 1e-5 * (1.0 + x.abs());
        if x < v.min - slack || x > v.max + slack {
            return Err(format!(
                "variable {} = {} outside [{}, {}]",
                i, x, v.min, v.max
            ));
        }
        if v.domain == crate::model::Domain::Integer && (x - x.round()).abs() > 1e-4 {
            return Err(format!("integer variable {} = {} is fractional", i, x));
        }
    }
    for (ci, c) in spec.constraints.iter().enumerate() {
        let mut activity = 0.0f64;
        let mut scale = 1.0f64 + c.rhs.abs();
        for &(vi, coeff) in &c.terms {
            let t = coeff * values[vi];
            activity += t;
            scale = scale.max(t.abs());
        }
        let slack = 1e-5 * scale;
        let ok = match c.op {
            ComparisonOp::Le => activity <= c.rhs + slack,
            ComparisonOp::Ge => activity >= c.rhs - slack,
            ComparisonOp::Eq => (activity - c.rhs).abs() <= slack,
        };
        if !ok {
            return Err(format!(
                "constraint {} violated: activity {} vs rhs {} ({:?})",
                ci, activity, c.rhs, c.op
            ));
        }
    }
    if let Some(r) = reported_obj {
        let mut obj = 0.0f64;
        let mut scale = 1.0f64 + r.abs();
        for (v, &x) in spec.vars.iter().zip(values) {
            let t = v.obj_coeff * x;
            obj += t;
            scale = scale.max(t.abs());
        }
        if (obj - r).abs() > 1e-6 * scale {
            return Err(format!(
                "reported objective {} but the values give {}",
                r, obj
            ));
        }
    }
    Ok(())
}
