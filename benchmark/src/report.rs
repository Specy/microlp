//! Aggregation of the per-(instance, solver) measurements and rendering of
//! the Markdown report (BENCHMARK.md).

pub struct InstanceInfo {
    /// `<tier>/<family>/<stem>` — the tier and family are part of the name
    /// on purpose.
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub ints: usize,
    pub nnz: usize,
    pub is_mip: bool,
    pub maximize: bool,
}

impl InstanceInfo {
    fn class(&self) -> &'static str {
        if self.is_mip {
            "MILP"
        } else {
            "LP"
        }
    }
}

pub struct CaseResult {
    pub solver: String,
    pub ms: Option<f64>,
    pub objective: Option<f64>,
    pub bound: Option<f64>,
    pub gap: Option<f64>,
    pub status: String,
}

impl CaseResult {
    pub fn failed(solver: &str, msg: String) -> CaseResult {
        CaseResult {
            solver: solver.to_string(),
            ms: None,
            objective: None,
            bound: None,
            gap: None,
            status: msg,
        }
    }

    pub fn optimal(&self) -> bool {
        self.status == "optimal"
    }

    /// The solver proved something: an optimum, infeasibility or unboundedness.
    fn conclusive(&self) -> bool {
        matches!(self.status.as_str(), "optimal" | "infeasible" | "unbounded")
    }

    /// The solve ended because of the budget, not because of a defect.
    pub fn budget_limited(&self) -> bool {
        self.status == "feasible"
            || self.status == "interrupted"
            || self.status.starts_with("killed")
    }

    fn not_applicable(&self) -> bool {
        self.status.starts_with("error: unsupported")
    }
}

/// A proven optimum obtained by the reference pass (longer, untimed
/// certification solve), used only for the correction-gap column.
pub struct ReferenceResult {
    pub instance: String,
    pub solver: String,
    pub objective: f64,
    pub budget_secs: f64,
}

pub struct RunData {
    pub instances: Vec<(InstanceInfo, Vec<CaseResult>)>,
    pub references: Vec<ReferenceResult>,
    pub solvers: Vec<String>,
    pub time_limit_secs: f64,
    /// Relative MIP gap every solver ran with (0 = exact proofs required).
    pub mip_gap: f64,
    pub min_ms: f64,
    /// True when --filter/--solvers restricted the run; the report then
    /// carries a partial-run warning.
    pub partial: bool,
}

type Row = (InstanceInfo, Vec<CaseResult>);

fn result_of<'a>(row: &'a Row, solver: &str) -> Option<&'a CaseResult> {
    row.1.iter().find(|r| r.solver == solver)
}

/// Objective values that both count as a proven optimum must agree within
/// this tolerance; anything beyond is reported as a disagreement.
fn objectives_agree(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-5 * a.abs().max(b.abs())
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub fn render(data: &RunData) -> String {
    let (kept, trivial): (Vec<&Row>, Vec<&Row>) =
        data.instances.iter().partition(|(_, results)| {
            !results
                .iter()
                .all(|r| r.conclusive() && r.ms.is_some_and(|ms| ms < data.min_ms))
        });

    let mut md = String::new();
    header(&mut md, data, kept.len(), trivial.len());
    outcome_counts(&mut md, data, &kept);
    alerts(&mut md, data, &kept);

    if data.solvers.iter().any(|s| s == "microlp") {
        if let Some(primary) = data.solvers.iter().find(|s| *s != "microlp") {
            slowest_relative(&mut md, &kept, primary);
        }
        unsolved(&mut md, data, &kept);
    }
    best_incumbents(&mut md, data, &kept);
    trivial_section(&mut md, data, &trivial);
    full_results(&mut md, data, &kept);
    md
}

fn header(md: &mut String, data: &RunData, kept: usize, trivial: usize) {
    md.push_str("# microlp benchmark\n\n");
    if data.partial {
        md.push_str(
            "> **Partial run** — this report was generated with instance or solver \
             filters and does not cover the full corpus.\n\n",
        );
    }
    md.push_str(&format!(
        "microlp measured against other open-source solvers on the {} LP/MILP \
         instances vendored for the [correctness suite](tests/suite/README.md) \
         (netlib LPs, MIPLIB 3 and MILPBench MILPs). Regenerate with:\n\n\
         ```bash\ncargo run -p microlp-benchmark --release\n```\n\n",
        kept + trivial,
    ));
    let gap_clause = if data.mip_gap == 0.0 {
        "with relative MIP gap 0, i.e. every solver (microlp included) must \
         prove exact optimality"
            .to_string()
    } else {
        format!(
            "and every solver (microlp included) may stop once its solution is \
             proven within a relative MIP gap of {} — \"proved optimum\" then \
             means proven within that tolerance",
            data.mip_gap
        )
    };
    md.push_str("**Method.** ");
    md.push_str(&format!(
        "Each (instance, solver) pair runs in a fresh process with a {} s budget. \
         The measured time covers building the solver's native model from the parsed \
         instance plus solving it; file parsing is excluded. Solves finishing under \
         200 ms are repeated (up to 5 times) and the fastest run is kept. Rival \
         solvers run single-threaded, {}. A solver that times out reports the \
         incumbent it was holding, so timed-out instances still compare solution \
         quality. Every returned solution or incumbent is independently re-checked \
         against the instance (bounds, integrality, constraints, objective); a \
         check failure counts as a failed solve. Instances that every solver \
         finishes in under {} ms are excluded as trivial ({} excluded, listed at \
         the end). When no solver proves an optimum inside the shared budget on \
         an instance microlp left unfinished, the primary rival gets one longer \
         certification solve (untimed, all threads, always exact); its value \
         feeds only the correction-gap column.\n\n",
        data.time_limit_secs, gap_clause, data.min_ms, trivial,
    ));
    md.push_str(&format!(
        "| | |\n| --- | --- |\n| Date | {} |\n| Machine | {} ({} threads available) |\n| OS | {} {} |\n| microlp | {} |\n| Budget | {} s per (instance, solver) |\n| MIP gap | {} |\n\n",
        today_utc(),
        cpu_description(),
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        std::env::consts::OS,
        std::env::consts::ARCH,
        git_describe(),
        data.time_limit_secs,
        if data.mip_gap == 0.0 {
            "0 (exact)".to_string()
        } else {
            format!("{} (proofs within this tolerance count as optimal)", data.mip_gap)
        },
    ));
}

fn outcome_counts(md: &mut String, data: &RunData, kept: &[&Row]) {
    md.push_str("## Outcomes\n\n");
    md.push_str("| solver | proved optimum | timed out | failed |\n");
    md.push_str("| --- | ---: | ---: | ---: |\n");
    for solver in &data.solvers {
        let mut proved = 0;
        let mut timed_out = 0;
        let mut failed = 0;
        for row in kept {
            let Some(r) = result_of(row, solver) else {
                continue;
            };
            // The rare infeasible/unbounded proof counts as "proved" — it is
            // the exact answer for such an instance (the full-results cell
            // spells it out). An LP-only solver on a MILP counts nowhere.
            if r.conclusive() {
                proved += 1;
            } else if r.budget_limited() {
                timed_out += 1;
            } else if !r.not_applicable() {
                failed += 1;
            }
        }
        md.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            solver, proved, timed_out, failed
        ));
    }
    md.push('\n');
}

/// (microlp ms, rival ms) pairs over instances both solved to proven optimality.
fn both_optimal_pairs<'a>(kept: &[&'a Row], rival: &str) -> Vec<(&'a InstanceInfo, f64, f64)> {
    let mut out = vec![];
    for row in kept {
        let (Some(m), Some(r)) = (result_of(row, "microlp"), result_of(row, rival)) else {
            continue;
        };
        if m.optimal() && r.optimal() {
            if let (Some(mm), Some(rm)) = (m.ms, r.ms) {
                out.push((&row.0, mm, rm));
            }
        }
    }
    out
}

fn slowest_relative(md: &mut String, kept: &[&Row], rival: &str) {
    let mut pairs = both_optimal_pairs(kept, rival);
    pairs.sort_by(|a, b| ((b.1 + 1.0) / (b.2 + 1.0)).total_cmp(&((a.1 + 1.0) / (a.2 + 1.0))));
    if pairs.is_empty() {
        return;
    }
    md.push_str(&format!("## Largest slowdowns vs {}\n\n", rival));
    md.push_str(&format!(
        "The instances where microlp is furthest behind {rival} — the most \
         concrete list of what to profile next. The time ratio is microlp's \
         time over {rival}'s (with a 1 ms shift on both); below 1× microlp \
         is faster.\n\n",
    ));
    md.push_str(&format!(
        "| instance | class | microlp | {} | time ratio |\n",
        rival
    ));
    md.push_str("| --- | --- | ---: | ---: | ---: |\n");
    for (info, m_ms, r_ms) in pairs.iter().take(10) {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            info.name,
            info.class(),
            fmt_time(*m_ms),
            fmt_time(*r_ms),
            fmt_ratio((m_ms + 1.0) / (r_ms + 1.0)),
        ));
    }
    md.push('\n');
}

/// The reference optimum for an instance: a value some solver *proved* in the
/// timed run (rivals preferred, so microlp's correction gap is measured
/// against an independent answer), else one certified by the reference pass.
/// The label says where the value came from.
fn reference_optimum(data: &RunData, row: &Row) -> Option<(String, f64)> {
    let mut best: Option<(&str, f64)> = None;
    for r in &row.1 {
        if r.optimal() {
            if let Some(obj) = r.objective {
                match best {
                    Some(_) if r.solver != "microlp" => best = Some((&r.solver, obj)),
                    None => best = Some((&r.solver, obj)),
                    _ => {}
                }
            }
        }
    }
    if let Some((solver, obj)) = best {
        return Some((solver.to_string(), obj));
    }
    data.references
        .iter()
        .find(|r| r.instance == row.0.name)
        .map(|r| {
            (
                format!("{}, {:.0} s reference solve", r.solver, r.budget_secs),
                r.objective,
            )
        })
}

fn unsolved(md: &mut String, data: &RunData, kept: &[&Row]) {
    let rows: Vec<&&Row> = kept
        .iter()
        .filter(|row| result_of(row, "microlp").is_some_and(|m| m.budget_limited()))
        .collect();
    md.push_str("## Not solved by microlp within the budget\n\n");
    if rows.is_empty() {
        md.push_str(&format!(
            "microlp proved every instance it was given within the {} s budget.\n\n",
            data.time_limit_secs
        ));
        return;
    }
    md.push_str(&format!(
        "What microlp had in hand when the {} s budget ran out. The *correction \
         gap* is how far its incumbent is from the reference optimum, relative to \
         the reference: 0% means the right answer was found but not yet proven \
         optimal; \"no incumbent\" means the search had no integer-feasible \
         solution yet. The *self-reported gap* is microlp's own bound-based \
         estimate at the same moment. Track this table release over release: \
         entries should move up (smaller gaps) and eventually leave the table.\n\n",
        data.time_limit_secs
    ));
    md.push_str("| instance | class | microlp status | incumbent | self-reported gap | reference optimum | correction gap |\n");
    md.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: |\n");
    for row in rows {
        let m = result_of(row, "microlp").unwrap();
        let reference = reference_optimum(data, row);
        let correction = match (m.objective, &reference) {
            (Some(inc), Some((_, opt))) => fmt_pct(correction_gap(row.0.maximize, inc, *opt)),
            (None, Some(_)) => "no incumbent".into(),
            _ => "no reference proved in this run".into(),
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            row.0.name,
            row.0.class(),
            m.status.replace('|', "\\|"),
            m.objective.map(fmt_obj).unwrap_or_else(|| "—".into()),
            m.gap.map(fmt_pct).unwrap_or_else(|| "—".into()),
            reference
                .map(|(s, v)| format!("{} ({})", fmt_obj(v), s))
                .unwrap_or_else(|| "—".into()),
            correction,
        ));
    }
    md.push_str(
        "\nInstances with no reference optimum were not proved by any solver in \
         this run at this budget; the correctness suite holds externally \
         certified bounds for them (see tests/suite/data/xhard/milpbench/README.md).\n\n",
    );
}

/// Instances nobody proved anything about: compare the incumbents each
/// solver was holding when time ran out — without this table those rows
/// carry no comparative signal at all.
fn best_incumbents(md: &mut String, data: &RunData, kept: &[&Row]) {
    let rows: Vec<&&Row> = kept
        .iter()
        .filter(|row| !row.1.iter().any(|r| r.conclusive()))
        .collect();
    if rows.is_empty() {
        return;
    }
    md.push_str("## Best solution found where nothing was proved\n\n");
    md.push_str(
        "No solver proved these instances within the budget, so the incumbents \
         they were holding when time ran out compare solution quality instead. \
         Every value shown passed the independent feasibility check; **bold** \
         marks the best incumbent, percentages are the distance behind it, and \
         \"—\" means the solver found no feasible solution at all.\n\n",
    );
    md.push_str("| instance | class |");
    for s in &data.solvers {
        md.push_str(&format!(" {} |", s));
    }
    md.push('\n');
    md.push_str("| --- | --- |");
    for _ in &data.solvers {
        md.push_str(" ---: |");
    }
    md.push('\n');
    for row in rows {
        let best = row
            .1
            .iter()
            .filter(|r| r.budget_limited())
            .filter_map(|r| r.objective)
            .reduce(|a, b| if row.0.maximize { a.max(b) } else { a.min(b) });
        md.push_str(&format!("| {} | {} |", row.0.name, row.0.class()));
        for s in &data.solvers {
            let cell = match result_of(row, s) {
                Some(r) if r.not_applicable() => "n/a".into(),
                Some(r) => match (r.objective, best) {
                    (Some(obj), Some(best)) => {
                        let behind = correction_gap(row.0.maximize, obj, best);
                        if behind <= 1e-9 {
                            format!("**{}**", fmt_obj(obj))
                        } else {
                            format!("{} (+{})", fmt_obj(obj), fmt_pct(behind))
                        }
                    }
                    _ => "—".into(),
                },
                None => "—".into(),
            };
            md.push_str(&format!(" {} |", cell));
        }
        md.push('\n');
    }
    md.push('\n');
}

fn correction_gap(maximize: bool, incumbent: f64, reference: f64) -> f64 {
    let shortfall = if maximize {
        reference - incumbent
    } else {
        incumbent - reference
    };
    (shortfall / reference.abs().max(1.0)).max(0.0)
}

/// Red flags only — completely silent when everything is consistent. A
/// conflicting claim (two proven optima that differ, or a bound that cuts
/// off another solver's proven optimum) means a solver bug, so it renders
/// loudly whenever present; outright failures get their detail listed.
fn alerts(md: &mut String, data: &RunData, kept: &[&Row]) {
    let mut disagreements: Vec<String> = vec![];
    for row in kept {
        let optimal: Vec<&CaseResult> = row.1.iter().filter(|r| r.optimal()).collect();
        for a in 0..optimal.len() {
            for b in a + 1..optimal.len() {
                if let (Some(x), Some(y)) = (optimal[a].objective, optimal[b].objective) {
                    if !objectives_agree(x, y) {
                        disagreements.push(format!(
                            "| {} | {} = {} | {} = {} |",
                            row.0.name,
                            optimal[a].solver,
                            fmt_obj(x),
                            optimal[b].solver,
                            fmt_obj(y),
                        ));
                    }
                }
            }
        }
        if let Some((ref_solver, opt)) = reference_optimum(data, row) {
            for r in &row.1 {
                let Some(bound) = r.bound else { continue };
                if r.solver == ref_solver {
                    continue;
                }
                let slack = 1e-6 + 1e-5 * opt.abs();
                let unsound = if row.0.maximize {
                    bound < opt - slack // an upper bound below the true optimum
                } else {
                    bound > opt + slack // a lower bound above the true optimum
                };
                if unsound {
                    disagreements.push(format!(
                        "| {} | {} bound = {} | {} optimum = {} |",
                        row.0.name,
                        r.solver,
                        fmt_obj(bound),
                        ref_solver,
                        fmt_obj(opt),
                    ));
                }
            }
        }
    }
    if !disagreements.is_empty() {
        md.push_str("**⚠ Conflicting claims between solvers:**\n\n");
        md.push_str("| instance | claim A | claim B |\n| --- | --- | --- |\n");
        for d in &disagreements {
            md.push_str(d);
            md.push('\n');
        }
        md.push('\n');
    }

    let mut failures: Vec<String> = vec![];
    for row in kept {
        for r in &row.1 {
            if !r.conclusive() && !r.budget_limited() && !r.not_applicable() {
                failures.push(format!(
                    "| {} | {} | {} |",
                    row.0.name,
                    r.solver,
                    r.status.replace('|', "\\|")
                ));
            }
        }
    }
    if !failures.is_empty() {
        md.push_str("**Failures:**\n\n| instance | solver | detail |\n| --- | --- | --- |\n");
        for f in &failures {
            md.push_str(f);
            md.push('\n');
        }
        md.push('\n');
    }
}

fn trivial_section(md: &mut String, data: &RunData, trivial: &[&Row]) {
    if trivial.is_empty() {
        return;
    }
    md.push_str(&format!(
        "## Excluded as trivial\n\nEvery solver finished these in under {} ms; \
         they carry no performance signal: ",
        data.min_ms
    ));
    let names: Vec<&str> = trivial.iter().map(|r| r.0.name.as_str()).collect();
    md.push_str(&names.join(", "));
    md.push_str(".\n\n");
}

fn full_results(md: &mut String, data: &RunData, kept: &[&Row]) {
    md.push_str("## Full results\n\n");
    md.push_str("Cells show wall time; non-optimal outcomes are spelled out.\n\n");
    md.push_str("| instance | class | rows | cols | int | nnz |");
    for s in &data.solvers {
        md.push_str(&format!(" {} |", s));
    }
    md.push('\n');
    md.push_str("| --- | --- | ---: | ---: | ---: | ---: |");
    for _ in &data.solvers {
        md.push_str(" ---: |");
    }
    md.push('\n');
    for row in kept {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |",
            row.0.name,
            row.0.class(),
            row.0.rows,
            row.0.cols,
            row.0.ints,
            row.0.nnz,
        ));
        for s in &data.solvers {
            let cell = match result_of(row, s) {
                Some(r) => result_cell(r),
                None => "—".into(),
            };
            md.push_str(&format!(" {} |", cell));
        }
        md.push('\n');
    }
    md.push('\n');
}

fn result_cell(r: &CaseResult) -> String {
    match r.status.as_str() {
        "optimal" => r.ms.map(fmt_time).unwrap_or_else(|| "?".into()),
        "feasible" => format!(
            "feasible{}",
            r.gap
                .map(|g| format!(", gap {}", fmt_pct(g)))
                .unwrap_or_default()
        ),
        "interrupted" => "timed out, no solution".into(),
        "infeasible" | "unbounded" => format!(
            "{} ({})",
            r.status,
            r.ms.map(fmt_time).unwrap_or_else(|| "?".into())
        ),
        s if s.starts_with("error: unsupported") => "n/a".into(),
        _ => "failed".into(),
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

pub fn fmt_time(ms: f64) -> String {
    if ms < 999.5 {
        format!("{:.1} ms", ms)
    } else {
        format!("{:.2} s", ms / 1000.0)
    }
}

fn fmt_ratio(r: f64) -> String {
    if r >= 99.5 {
        format!("{:.0}×", r)
    } else if r >= 9.95 {
        format!("{:.1}×", r)
    } else {
        format!("{:.2}×", r)
    }
}

fn fmt_pct(x: f64) -> String {
    let pct = x * 100.0;
    if pct != 0.0 && pct.abs() < 0.01 {
        format!("{:.1e}%", pct)
    } else {
        format!("{:.2}%", pct)
    }
}

fn fmt_obj(x: f64) -> String {
    let a = x.abs();
    if a != 0.0 && !(1e-3..1e7).contains(&a) {
        format!("{:.6e}", x)
    } else {
        let mut s = format!("{:.4}", x);
        while s.contains('.') && (s.ends_with('0') || s.ends_with('.')) {
            s.pop();
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Environment description (best effort, for the report header)
// ---------------------------------------------------------------------------

/// Civil date from the system clock (Howard Hinnant's days-to-date algorithm),
/// so the report can carry a date without pulling in a chrono dependency.
fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let z = secs.div_euclid(86400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02} (UTC)", y, m, d)
}

fn cpu_description() -> String {
    // Windows: the marketing name lives in the registry.
    if cfg!(windows) {
        if let Ok(out) = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "/v",
                "ProcessorNameString",
            ])
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = text.lines().find(|l| l.contains("ProcessorNameString")) {
                if let Some(idx) = line.find("REG_SZ") {
                    return line[idx + "REG_SZ".len()..].trim().to_string();
                }
            }
        }
    }
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = text.lines().find(|l| l.starts_with("model name")) {
            if let Some((_, name)) = line.split_once(':') {
                return name.trim().to_string();
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown CPU".into())
}

fn git_describe() -> String {
    let root = crate::corpus::repo_root();
    let rev = std::process::Command::new("git")
        .args(["-C"])
        .arg(&root)
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    match rev {
        Some(rev) => {
            let dirty = std::process::Command::new("git")
                .args(["-C"])
                .arg(&root)
                .args(["status", "--porcelain"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false);
            format!(
                "commit {}{}",
                rev,
                if dirty { " + uncommitted changes" } else { "" }
            )
        }
        None => "unversioned checkout".into(),
    }
}
