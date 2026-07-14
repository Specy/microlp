# microlp correctness suite

A problem-based correctness suite: 230+ LP/MILP instances whose true answers
are **independently known** — published benchmark values, mathematical
constructions, exact combinatorial oracles computed at build time, or
verdicts certified with an external reference solver. It exists so you can
change the solver and trust that a green run means the solver still produces
right answers, not merely the same answers as before.

## Running

```bash
cargo test --release --test suite                          # default: easy + medium (~2 s)
cargo test --release --test suite -- --hard                # + long-running MILPs (CI runs this)
cargo test --release --test suite -- --xhard               # + 10-minute-budget instances
cargo test --release --test suite -- --limit 25 --seed 42  # stable shuffled subset
cargo test --release --test suite -- knapsack netlib       # substring name filters
cargo test --release --test suite -- --list                # show selected cases
cargo test --release --test suite -- --parallel 8          # worker threads
cargo test --release --test suite -- --help
```

The runner prints one line per case and a final summary with pass/fail counts,
then exits nonzero if anything failed (CI-friendly). Under a debug build only
the `easy` tier runs by default (the heavier tiers are far too slow without
optimizations); `--full` or an explicit tier flag overrides.

`--seed` shuffles *which cases run and in what order* (deterministic
Fisher-Yates). Instance generation itself is seeded by constants baked into
case names, so a given case is byte-identical across runs regardless of
`--seed` — a failure seen in a subset run reproduces alone via its name:

```bash
cargo test --release --test suite -- "oracle/ilp-box/s07"
```

## Tiers

Tiers are **cumulative upper limits**: a tier flag runs that tier and
everything below it (`--hard` = easy + medium + hard). If several flags are
given, the highest wins.

* `easy` — milliseconds each; runs everywhere, including debug builds.
* `medium` — the release-mode default (with `easy`); laptop-friendly.
* `hard` — long-running MILPs, minutes each. CI runs
  `--hard --parallel 4 --max-case-seconds 300`, clamping the solve budget
  supplied to each case to 5 minutes after `--timeout-scale`. Custom cases
  cooperatively enforce that budget by passing it to their solves.
* `xhard` — instances beyond the solver's current ceiling, on 10-minute
  budgets, with externally certified verdicts (see below). Run manually.

**File-based cases derive their tier from the folder their instance lives
in** — `data/<tier>/<source>/<file>` (see `cases::locate`) — so moving a file
between tier folders re-tiers its cases with no code change. The MIPLIB
instances are one file with two cases: the integer solve's tier follows the
folder, while the LP-relaxation case always runs in the medium tier (every
relaxation is fast and doubles as a parser check). Large instances are
stored gzipped (`<file>.lp.gz`); `locate` resolves either storage form for
the same logical name and `read_instance` decompresses in memory at run
time, so compressing a file is as transparent as moving it.

## What a case checks

Every solved case goes through two layers:

1. **Independent solution validation** (`verify.rs`): variable values are
   pulled out of the `Solution` and re-checked against a *shadow model*
   (`model.rs`) recorded outside the solver — constraint satisfaction, bound
   compliance, integrality of integer/boolean variables, and that the reported
   objective equals the objective recomputed from the values. This catches
   "right objective, infeasible point" and "claims optimal, violates bounds"
   bug classes regardless of the expected value.
2. **Expected-answer comparison**: objective vs the known optimum within a
   per-case tolerance, or an expected `Infeasible` / `Unbounded` error.
   Variable values are asserted only where the optimum is provably unique.

Interrupt-tolerant cases (instances that may not finish in budget) assert the
solver ends in a **clean status** and hold whatever it returns to soundness
bars: a claimed optimum must match the reference value, and a feasible
incumbent must validate against the shadow model and never be *better* than a
proven optimum or bound.

Panics are caught per case and reported as `PANIC` without aborting the run.
Standard `CaseRun::Solve` cases map a limit without an accepted result to
`TIMEOUT`. Custom cases decide whether an interrupted solve is a pass, timeout,
or failure. The runner has no process-level watchdog; custom cases must forward
their supplied budget to every potentially long-running solve.

## Where the answers come from

| family | source of truth |
| ------ | --------------- |
| `lp/*` unit | hand-computed unique vertices; Klee-Minty optimum is 5^n by construction (Klee & Minty 1972); Beale's cycling example (Beale 1955); infeasible/unbounded variants |
| `lp/certified/*` | random LPs with an optimality certificate built in: integer matrix A (diagonally dominant), b = A·x*, c = Aᵀ·1, so y = 1 certifies x* by strong duality |
| `lp/metamorphic/*` | solver vs itself under invariants: objective scaling, duplicated constraints, min/max negation |
| `milp/*` unit | rounding traps, diophantine (in)feasibility, boolean gates, big-M implications, coin change vs DP, magic squares, n-queens, interruption/restart behaviors |
| `oracle/*` | exact oracles computed per run: knapsack/subset-sum DP, assignment brute force, tiny-ILP box enumeration |
| `netlib/*` | official netlib optimal values (`lp/data/readme`), 11 significant digits |
| `miplib/*` | MIPLIB 3 official catalog (`miplib3.cat`): LP relaxation *and* integer optimum per instance |
| `milpbench/*` | MILPBench instances. The medium-tier CFL instances are solver-proven optima (integral LP relaxation, so the LP bound certifies the answer); the xhard families carry HiGHS-certified verdicts — a proven optimum, a proven `[dual bound, incumbent]` envelope, or a proven unboundedness — see `data/xhard/milpbench/README.md` |
| `incr/*` | the incremental Solution APIs (`add_constraint`, `fix_var`/`unfix_var`, `resume`) vs certified optima, oracles and from-scratch solves |
| `milp/warm-restart-*`, `milp/nodelimit-steps-*` | restart-with-hint loops on real instances with monotone-improvement assertions |

See [data/README.md](data/README.md) for per-instance attribution,
contributors, citations and the licensing status of the vendored benchmark
files (kept out of the published crate on purpose).

## Readers

Benchmark files are read through thin adapters over external
dev-dependencies: the `mps` crate for MPS files (`mps_milp.rs`, which also
recovers INTORG/INTEND integer markers and owns the MPS bound conventions)
and `lp_parser_rs` for CPLEX-LP files (`lp_format.rs`, which consumes the
crate's raw grammar output and owns the variable-domain semantics). Both
adapters are loud on anything they cannot faithfully represent — a misparse
would poison the suite, so unsupported constructs are errors, never guesses.

## Known failures

`KNOWN_INFEASIBLE_BUG` in `cases/netlib.rs` marks instances that are explicitly
expected to return `Err(Infeasible)` despite a published feasible optimum. A
marked case rejects every other outcome so the exception cannot silently skip
normal validation. The list is currently **empty** — every case asserts correct
behavior and passes.

## Adding a case

Add a closure to the matching family file under `cases/` (or a new file wired
into `cases/mod.rs`). Build the instance through `model::Builder` (it records
the shadow model), return an `Expected`, and give the case a stable unique
name. If the optimum has alternate solutions, assert the objective only —
never variable values. For file-based instances, drop the file into the tier
folder that matches its cost; the tier follows the folder.
