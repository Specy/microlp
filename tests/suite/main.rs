//! Correctness suite runner for microlp (a `harness = false` test target).
//!
//! Every case is an LP/MILP instance whose true answer is independently known
//! (published benchmark value, mathematical construction, or an exact oracle
//! computed at build time). Solutions are re-validated against a shadow model
//! — feasibility, bounds, integrality and objective consistency — before the
//! objective is compared with the expected value.
//!
//! Cases are tiered easy/medium/hard/xhard; a tier flag is a cumulative upper
//! limit (`--hard` runs easy + medium + hard). The default is `--medium`.
//! File-based cases derive their tier from the folder the instance lives in:
//! `data/<tier>/<source>/<file>` (see `cases::locate`).
//!
//! Usage (args go after `--`):
//!   cargo test --release --test suite                          default run (easy+medium)
//!   cargo test --release --test suite -- --limit 25 --seed 42  stable random subset
//!   cargo test --release --test suite -- knapsack netlib       name filters
//!   cargo test --release --test suite -- --hard                + hard tier
//!   cargo test --release --test suite -- --xhard               everything
//!   cargo test --release --test suite -- --list                list case names
//!
//! Exit code is nonzero if any selected case fails, times out or panics.

mod cases;
mod lp_format;
mod model;
mod mps_milp;
mod oracles;
mod rng;
mod verify;

use cases::{Case, CaseRun, Tier};
use model::Expected;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

struct Options {
    limit: Option<usize>,
    seed: Option<u64>,
    filters: Vec<String>,
    /// Highest tier to run; tiers are cumulative (`--hard` runs easy, medium
    /// and hard). `None` means the default: medium in release builds, easy in
    /// debug builds without `--full`.
    max_tier: Option<Tier>,
    full: bool,
    list: bool,
    timeout_scale: f64,
    /// Hard upper bound, in seconds, on any single case's time budget,
    /// applied AFTER `timeout_scale`. Lets CI run the full suite with a
    /// guarantee that no single instance exceeds the cap.
    max_case_seconds: Option<f64>,
    parallel: usize,
}

fn parse_args() -> Options {
    let mut opts = Options {
        limit: None,
        seed: None,
        filters: vec![],
        max_tier: None,
        full: false,
        list: false,
        timeout_scale: 1.0,
        max_case_seconds: None,
        parallel: 1,
    };
    // If several tier flags are given, the highest wins.
    fn raise_tier(opts: &mut Options, tier: Tier) {
        opts.max_tier = Some(opts.max_tier.map_or(tier, |cur| cur.max(tier)));
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" | "-l" => {
                let v = take_value(&args, &mut i, "--limit");
                opts.limit = Some(
                    v.parse()
                        .unwrap_or_else(|_| die("--limit must be a number")),
                );
            }
            "--seed" | "-s" => {
                let v = take_value(&args, &mut i, "--seed");
                opts.seed = Some(v.parse().unwrap_or_else(|_| die("--seed must be a u64")));
            }
            "--parallel" | "-p" => {
                let v = take_value(&args, &mut i, "--parallel");
                opts.parallel = v
                    .parse()
                    .unwrap_or_else(|_| die("--parallel must be a number"));
            }
            "--timeout-scale" => {
                let v = take_value(&args, &mut i, "--timeout-scale");
                opts.timeout_scale = v
                    .parse()
                    .unwrap_or_else(|_| die("--timeout-scale must be a number"));
            }
            "--max-case-seconds" => {
                let v = take_value(&args, &mut i, "--max-case-seconds");
                let secs: f64 = v
                    .parse()
                    .unwrap_or_else(|_| die("--max-case-seconds must be a number"));
                // NaN must fail this check too, so test for validity directly.
                let valid = secs.is_finite() && secs > 0.0;
                if !valid {
                    die("--max-case-seconds must be positive");
                }
                opts.max_case_seconds = Some(secs);
            }
            "--easy" => raise_tier(&mut opts, Tier::Easy),
            "--medium" => raise_tier(&mut opts, Tier::Medium),
            "--hard" => raise_tier(&mut opts, Tier::Hard),
            "--xhard" => raise_tier(&mut opts, Tier::XHard),
            "--bench" => die(
                "--bench was removed: the MILPBench instances now live in the \
                 medium and xhard tiers (--xhard runs everything)",
            ),
            "--full" => opts.full = true,
            "--list" => opts.list = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            // Tolerate common libtest flags so `cargo test -- --nocapture`
            // style invocations don't break this target.
            "--nocapture" | "--show-output" | "--quiet" | "-q" | "--exact" | "--ignored"
            | "--include-ignored" => {}
            s if s.starts_with("--test-threads")
                || s.starts_with("--format")
                || s.starts_with("--color") =>
            {
                if !s.contains('=') {
                    i += 1; // skip the "--flag value" form's value
                }
            }
            s if s.starts_with('-') => {
                eprintln!("warning: ignoring unknown flag {}", s);
            }
            // Positional arguments act as name filters, like libtest.
            s => opts.filters.push(s.to_string()),
        }
        i += 1;
    }
    opts
}

fn take_value(args: &[String], i: &mut usize, name: &str) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| die(&format!("{} requires a value", name)))
}

fn die(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    eprintln!();
    print_help();
    std::process::exit(2);
}

fn print_help() {
    println!(
        "microlp correctness suite

USAGE:
    cargo test --release --test suite -- [FILTERS] [OPTIONS]

ARGS:
    [FILTERS]...        substrings matched against case names (OR)

OPTIONS:
    -l, --limit <N>     run at most N cases (after shuffling, if any)
    -s, --seed <SEED>   shuffle case order deterministically with this seed
    -p, --parallel <N>  run tests in parallel with N worker threads (default: 1)
        --easy          run only the easy tier
        --medium        run easy + medium (the release-mode default)
        --hard          also include the hard tier (long-running benchmark
                        MILPs, minutes each)
        --xhard         run everything, including the 10-minute-budget
                        instances (tier flags are cumulative upper limits;
                        the highest given wins)
        --full          in debug builds, run the default set anyway
        --timeout-scale <F>  multiply every per-case time budget by F
        --max-case-seconds <S>  cap every per-case budget at S seconds
                        (applied after --timeout-scale; used by CI to bound
                        any single instance's runtime)
        --list          print the selected case names and exit
    -h, --help          show this help

Notes:
    * Case generation is deterministic and independent of --seed; the seed
      only shuffles which subset --limit picks and in what order.
    * Pass a failing case's full name as a filter to reproduce it alone."
    );
}

#[derive(Debug, Clone)]
struct SolveDetails {
    found: f64,
    expected: f64,
}

#[derive(Debug, Clone)]
enum Status {
    Pass(Option<SolveDetails>),
    Fail(String),
    Timeout,
    Panic(String),
}

fn main() {
    let opts = parse_args();
    let debug_build = cfg!(debug_assertions);

    let mut all = cases::all();

    // Tier selection is cumulative: run everything up to the requested tier.
    // The default is medium; a debug build without --full drops to easy (the
    // heavier tiers are far too slow without optimizations). An explicit tier
    // flag is honored even in debug builds.
    let tier_limit = opts.max_tier.unwrap_or(if debug_build && !opts.full {
        Tier::Easy
    } else {
        Tier::Medium
    });
    all.retain(|c| c.tier <= tier_limit);

    if !opts.filters.is_empty() {
        all.retain(|c| opts.filters.iter().any(|f| c.name.contains(f.as_str())));
    }

    if let Some(seed) = opts.seed {
        rng::Rng::new(seed).shuffle(&mut all);
    }

    let total_known = all.len();
    if let Some(limit) = opts.limit {
        all.truncate(limit);
    }

    if opts.list {
        for c in &all {
            println!("{}  [{}]", c.name, c.tier.label());
        }
        println!("{} cases selected", all.len());
        return;
    }

    if all.is_empty() {
        println!("no cases selected (of {} eligible)", total_known);
        std::process::exit(2);
    }

    println!(
        "microlp correctness suite: running {} of {} eligible cases [up to {} tier]{}{}{}",
        all.len(),
        total_known,
        tier_limit.label(),
        if debug_build && opts.max_tier.is_none() && !opts.full {
            " [debug build: easy tier only, use --full for the default set]"
        } else {
            ""
        },
        match opts.seed {
            Some(s) => format!(" (seed {})", s),
            None => String::new(),
        },
        match opts.limit {
            Some(l) => format!(" (limit {})", l),
            None => String::new(),
        },
    );
    println!();

    // Capture panic messages instead of letting the default hook spam stderr.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|info| {
        LAST_PANIC.with(|slot| {
            *slot.borrow_mut() = Some(info.to_string());
        });
    }));

    let suite_start = Instant::now();
    let mut results: Vec<(String, Status, Duration)> = vec![];
    let width = all.iter().map(|c| c.name.len()).max().unwrap_or(0);

    let mut completed = 0;
    if opts.parallel <= 1 {
        for case in &all {
            print!("{:width$}  ", case.name, width = width);
            std::io::stdout().flush().ok();
            let started = Instant::now();
            let status = run_case(case, opts.timeout_scale, opts.max_case_seconds);
            let elapsed = started.elapsed();
            completed += 1;
            let left = all.len() - completed;
            match &status {
                Status::Pass(details) => {
                    let details_str = match details {
                        Some(d) => format!(" {}", fmt_solve_details(d.found, d.expected)),
                        None => String::new(),
                    };
                    println!(
                        "ok       ({}) [{} left]{}",
                        fmt_duration(elapsed),
                        left,
                        details_str
                    );
                }
                Status::Fail(msg) => println!(
                    "FAIL     ({}) [{} left]\n    {}",
                    fmt_duration(elapsed),
                    left,
                    msg
                ),
                Status::Timeout => println!(
                    "TIMEOUT  (budget {}) [{} left]",
                    fmt_duration(effective_budget(
                        case.budget,
                        opts.timeout_scale,
                        opts.max_case_seconds
                    )),
                    left
                ),
                Status::Panic(msg) => println!(
                    "PANIC    ({}) [{} left]\n    {}",
                    fmt_duration(elapsed),
                    left,
                    msg.lines().collect::<Vec<_>>().join(" | ")
                ),
            }
            results.push((case.name.clone(), status, elapsed));
        }
    } else {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::thread;

        let next_index = AtomicUsize::new(0);
        let all_cases = &all;
        let (tx, rx) = mpsc::channel();

        thread::scope(|s| {
            // Spawn worker threads
            for _ in 0..opts.parallel {
                let tx = tx.clone();
                let next_index = &next_index;
                s.spawn(move || loop {
                    let idx = next_index.fetch_add(1, Ordering::SeqCst);
                    if idx >= all_cases.len() {
                        break;
                    }
                    let case = &all_cases[idx];
                    let started = Instant::now();
                    let status = run_case(case, opts.timeout_scale, opts.max_case_seconds);
                    let elapsed = started.elapsed();
                    if tx.send((idx, status, elapsed)).is_err() {
                        break;
                    }
                });
            }

            // Drop our master sender so the channel closes when all threads are done
            drop(tx);

            // Print results in real time as they complete
            let mut completed = 0;
            while let Ok((idx, status, elapsed)) = rx.recv() {
                completed += 1;
                let left = all_cases.len() - completed;
                let case = &all_cases[idx];
                match &status {
                    Status::Pass(details) => {
                        let details_str = match details {
                            Some(d) => format!(" {}", fmt_solve_details(d.found, d.expected)),
                            None => String::new(),
                        };
                        println!(
                            "{:width$}  ok       ({}) [{} left]{}",
                            case.name,
                            fmt_duration(elapsed),
                            left,
                            details_str,
                            width = width
                        );
                    }
                    Status::Fail(msg) => {
                        println!(
                            "{:width$}  FAIL     ({}) [{} left]\n    {}",
                            case.name,
                            fmt_duration(elapsed),
                            left,
                            msg,
                            width = width
                        );
                    }
                    Status::Timeout => {
                        println!(
                            "{:width$}  TIMEOUT  (budget {}) [{} left]",
                            case.name,
                            fmt_duration(effective_budget(
                                case.budget,
                                opts.timeout_scale,
                                opts.max_case_seconds
                            )),
                            left,
                            width = width
                        );
                    }
                    Status::Panic(msg) => {
                        println!(
                            "{:width$}  PANIC    ({}) [{} left]\n    {}",
                            case.name,
                            fmt_duration(elapsed),
                            left,
                            msg.lines().collect::<Vec<_>>().join(" | "),
                            width = width
                        );
                    }
                }
                results.push((case.name.clone(), status, elapsed));
            }
        });
    }

    std::panic::set_hook(default_hook);

    // ---- Summary ----
    let total = results.len();
    let passed = results
        .iter()
        .filter(|(_, s, _)| matches!(s, Status::Pass(_)))
        .count();
    let failed: Vec<_> = results
        .iter()
        .filter(|(_, s, _)| matches!(s, Status::Fail(_)))
        .collect();
    let timeouts: Vec<_> = results
        .iter()
        .filter(|(_, s, _)| matches!(s, Status::Timeout))
        .collect();
    let panics: Vec<_> = results
        .iter()
        .filter(|(_, s, _)| matches!(s, Status::Panic(_)))
        .collect();

    println!();
    if !failed.is_empty() || !timeouts.is_empty() || !panics.is_empty() {
        println!("failing cases (rerun one with: cargo test --release --test suite -- <name>):");
        for (name, status, _) in results.iter() {
            match status {
                Status::Fail(msg) => println!("  FAIL    {}\n          {}", name, msg),
                Status::Timeout => println!("  TIMEOUT {}", name),
                Status::Panic(msg) => println!(
                    "  PANIC   {}\n          {}",
                    name,
                    msg.lines().collect::<Vec<_>>().join(" | ")
                ),
                Status::Pass(_) => {}
            }
        }
        println!();
    }

    let mut slowest: Vec<_> = results.iter().collect();
    slowest.sort_by_key(|(_, _, d)| std::cmp::Reverse(*d));
    let slow_list = slowest
        .iter()
        .take(5)
        .map(|(n, _, d)| format!("{} {}", n, fmt_duration(*d)))
        .collect::<Vec<_>>()
        .join(", ");

    let pct = 100.0 * passed as f64 / total as f64;
    println!("════════════════════════════════════════════════════════");
    println!(
        "  passed {}/{} ({:.1}%) · failed {} · timeout {} · panicked {}",
        passed,
        total,
        pct,
        failed.len(),
        timeouts.len(),
        panics.len()
    );
    println!(
        "  total time {} · slowest: {}",
        fmt_duration(suite_start.elapsed()),
        slow_list
    );
    println!("════════════════════════════════════════════════════════");

    if passed != total {
        std::process::exit(1);
    }
}

thread_local! {
    static LAST_PANIC: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    pub static LAST_SOLVE: std::cell::RefCell<Option<(f64, f64)>> = const { std::cell::RefCell::new(None) };
}

/// A case's time budget after applying `--timeout-scale` and the
/// `--max-case-seconds` cap.
fn effective_budget(base: Duration, timeout_scale: f64, max_case_seconds: Option<f64>) -> Duration {
    let scaled = mul_duration(base, timeout_scale);
    match max_case_seconds {
        Some(cap) => scaled.min(Duration::from_secs_f64(cap)),
        None => scaled,
    }
}

fn run_case(case: &Case, timeout_scale: f64, max_case_seconds: Option<f64>) -> Status {
    let budget = effective_budget(case.budget, timeout_scale, max_case_seconds);
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        LAST_SOLVE.with(|slot| *slot.borrow_mut() = None);
        match &case.run {
            CaseRun::Solve(build) => {
                let (spec, mut problem, expected) = match build() {
                    Ok(parts) => parts,
                    Err(msg) => return Status::Fail(format!("case build failed: {}", msg)),
                };
                problem.set_time_limit(budget);
                let solve_result = problem.solve();
                let details = match (&solve_result, &expected) {
                    (Ok(sol), Expected::Objective { value, .. }) => Some(SolveDetails {
                        found: sol.objective(),
                        expected: *value,
                    }),
                    _ => None,
                };
                match verify::check(&spec, &expected, solve_result) {
                    verify::Outcome::Pass => Status::Pass(details),
                    verify::Outcome::Fail(msg) => Status::Fail(msg),
                    verify::Outcome::Timeout => Status::Timeout,
                }
            }
            CaseRun::Custom(run) => match run(budget) {
                Ok(()) => {
                    let details = LAST_SOLVE.with(|slot| {
                        slot.borrow_mut()
                            .take()
                            .map(|(found, expected)| SolveDetails { found, expected })
                    });
                    Status::Pass(details)
                }
                Err(msg) => {
                    if msg.contains("hit time limit") {
                        Status::Timeout
                    } else {
                        Status::Fail(msg)
                    }
                }
            },
        }
    }));
    match outcome {
        Ok(status) => status,
        Err(payload) => {
            let mut msg = LAST_PANIC
                .with(|slot| slot.borrow_mut().take())
                .unwrap_or_default();
            if msg.is_empty() {
                msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with non-string payload".to_string());
            }
            Status::Panic(msg)
        }
    }
}

fn mul_duration(d: Duration, scale: f64) -> Duration {
    Duration::from_secs_f64((d.as_secs_f64() * scale).max(0.001))
}

fn fmt_duration(d: Duration) -> String {
    let ms = d.as_millis();
    if ms < 1000 {
        format!("{} ms", ms)
    } else if ms < 60_000 {
        format!("{:.1} s", d.as_secs_f64())
    } else {
        format!("{:.1} min", d.as_secs_f64() / 60.0)
    }
}

fn fmt_solve_details(found: f64, expected: f64) -> String {
    let f_str = format_float(found);
    let e_str = format_float(expected);
    if (found - expected).abs() < 1e-9 {
        format!("(found: {}, expected: {}, diff: 0.00%)", f_str, e_str)
    } else if expected.abs() > 1e-9 {
        let pct_diff = (found - expected).abs() / expected.abs() * 100.0;
        format!(
            "(found: {}, expected: {}, diff: {:.2}%)",
            f_str, e_str, pct_diff
        )
    } else {
        format!(
            "(found: {}, expected: {}, diff: {:.2} abs)",
            f_str,
            e_str,
            (found - expected).abs()
        )
    }
}

fn format_float(val: f64) -> String {
    let s = format!("{:.4}", val);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}
