//! Correctness suite runner for microlp (a `harness = false` test target).
//!
//! Every case is an LP/MILP instance whose true answer is independently known
//! (published benchmark value, mathematical construction, or an exact oracle
//! computed at build time). Solutions are re-validated against a shadow model
//! — feasibility, bounds, integrality and objective consistency — before the
//! objective is compared with the expected value.
//!
//! Usage (args go after `--`):
//!   cargo test --release --test suite                          full default run
//!   cargo test --release --test suite -- --limit 25 --seed 42  stable random subset
//!   cargo test --release --test suite -- knapsack netlib       name filters
//!   cargo test --release --test suite -- --hard                include hard tier
//!   cargo test --release --test suite -- --list                list case names
//!
//! Exit code is nonzero if any selected case fails, times out or panics.

mod cases;
mod model;
mod mps_milp;
mod oracles;
mod rng;
mod verify;

use cases::{Case, CaseRun, Tier};
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

struct Options {
    limit: Option<usize>,
    seed: Option<u64>,
    filters: Vec<String>,
    hard: bool,
    full: bool,
    list: bool,
    timeout_scale: f64,
}

fn parse_args() -> Options {
    let mut opts = Options {
        limit: None,
        seed: None,
        filters: vec![],
        hard: false,
        full: false,
        list: false,
        timeout_scale: 1.0,
    };
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
            "--timeout-scale" => {
                let v = take_value(&args, &mut i, "--timeout-scale");
                opts.timeout_scale = v
                    .parse()
                    .unwrap_or_else(|_| die("--timeout-scale must be a number"));
            }
            "--hard" => opts.hard = true,
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
        --hard          include the hard tier (long-running benchmark MILPs)
        --full          in debug builds, run the full suite anyway
        --timeout-scale <F>  multiply every per-case time budget by F
        --list          print the selected case names and exit
    -h, --help          show this help

Notes:
    * Case generation is deterministic and independent of --seed; the seed
      only shuffles which subset --limit picks and in what order.
    * Pass a failing case's full name as a filter to reproduce it alone."
    );
}

enum Status {
    Pass,
    Fail(String),
    Timeout,
    Panic(String),
}

fn main() {
    let opts = parse_args();
    let debug_build = cfg!(debug_assertions);

    let mut all = cases::all();

    // Tier selection: quick always runs; standard unless this is a debug
    // build without --full; hard only with --hard.
    let run_standard = !debug_build || opts.full || opts.hard;
    all.retain(|c| match c.tier {
        Tier::Quick => true,
        Tier::Standard => run_standard,
        Tier::Hard => opts.hard,
    });

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
        "microlp correctness suite: running {} of {} eligible cases{}{}{}{}",
        all.len(),
        total_known,
        if debug_build && !opts.full && !opts.hard {
            " [debug build: quick tier only, use --full for everything]"
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
        if opts.hard { " (+hard tier)" } else { "" },
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

    for case in &all {
        print!("{:width$}  ", case.name, width = width);
        std::io::stdout().flush().ok();
        let started = Instant::now();
        let status = run_case(case, opts.timeout_scale);
        let elapsed = started.elapsed();
        match &status {
            Status::Pass => println!("ok       ({})", fmt_duration(elapsed)),
            Status::Fail(msg) => println!("FAIL     ({})\n    {}", fmt_duration(elapsed), msg),
            Status::Timeout => println!(
                "TIMEOUT  (budget {})",
                fmt_duration(mul_duration(case.budget, opts.timeout_scale))
            ),
            Status::Panic(msg) => println!(
                "PANIC    ({})\n    {}",
                fmt_duration(elapsed),
                msg.lines().collect::<Vec<_>>().join(" | ")
            ),
        }
        results.push((case.name.clone(), status, elapsed));
    }

    std::panic::set_hook(default_hook);

    // ---- Summary ----
    let total = results.len();
    let passed = results
        .iter()
        .filter(|(_, s, _)| matches!(s, Status::Pass))
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
                Status::Pass => {}
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
}

fn run_case(case: &Case, timeout_scale: f64) -> Status {
    let budget = mul_duration(case.budget, timeout_scale);
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| match &case.run {
        CaseRun::Solve(build) => {
            let (spec, mut problem, expected) = match build() {
                Ok(parts) => parts,
                Err(msg) => return Status::Fail(format!("case build failed: {}", msg)),
            };
            problem.set_time_limit(budget);
            match verify::check(&spec, &expected, problem.solve()) {
                verify::Outcome::Pass => Status::Pass,
                verify::Outcome::Fail(msg) => Status::Fail(msg),
                verify::Outcome::Timeout => Status::Timeout,
            }
        }
        CaseRun::Custom(run) => match run(budget) {
            Ok(()) => Status::Pass,
            Err(msg) => {
                if msg.contains("hit time limit") {
                    Status::Timeout
                } else {
                    Status::Fail(msg)
                }
            }
        },
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
