# microlp correctness suite

A problem-based correctness suite: ~190 LP/MILP instances whose true answers
are **independently known** — published benchmark values, mathematical
constructions, or exact combinatorial oracles computed at build time. It
exists so you can refactor the solver and trust that a green run means the
solver still produces right answers, not merely the same answers as before.

## Running

```bash
cargo test --release --test suite                          # full default run (~2 s)
cargo test --release --test suite -- --limit 25 --seed 42  # stable shuffled subset
cargo test --release --test suite -- knapsack netlib       # substring name filters
cargo test --release --test suite -- --hard                # + long-running MILPs
cargo test --release --test suite -- --list                # show selected cases
cargo test --release --test suite -- --help
```

The runner prints one line per case and a final summary with pass/fail counts
and a percentage, then exits nonzero if anything failed (CI-friendly). Under a
debug build (`cargo test` without `--release`) only the `quick` tier runs,
mirroring the repo's convention of skipping slow tests in debug; `--full`
overrides.

CI runs the no-argument release invocation (`cargo test --release --test
suite`) as its own step — the quick + standard tiers, which stay green. The
`hard` tier is never run in CI (see Tiers and Known failures below).

`--seed` shuffles *which cases run and in what order* (deterministic
Fisher-Yates). Instance generation itself is seeded by constants baked into
case names, so a given case is byte-identical across runs regardless of
`--seed` — a failure seen in a subset run reproduces alone via its name:

```bash
cargo test --release --test suite -- "oracle/ilp-box/s07"
```

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

Panics are caught per case and reported as `PANIC` without aborting the run.
Each case runs under `Problem::set_time_limit`, so a cycling or exploding
solve becomes a `TIMEOUT` result instead of a hung suite.

## Where the answers come from

| family | cases | source of truth |
| ------ | ----- | --------------- |
| `lp/*` unit | 31 | hand-computed unique vertices; Klee-Minty optimum is 5^n by construction (Klee & Minty 1972); Beale's cycling example (-0.05, Beale 1955); 5 infeasible + 4 unbounded variants |
| `lp/certified/*` | 15 | random LPs with an optimality certificate built in: integer matrix A (diagonally dominant), b = A·x*, c = Aᵀ·1, so y = 1 certifies x* by strong duality — unique optimum, exact integer data |
| `lp/metamorphic/*` | 5 | solver vs itself under invariants: objective scaling, duplicated constraints, min/max negation |
| `milp/*` unit | 32 | rounding traps, diophantine (in)feasibility, boolean gate truth tables, big-M implications, coin change vs DP, magic-square center = 5, n-queens = n, time-limit interruption |
| `oracle/*` | 66 | exact oracles computed per run: 0/1 and bounded knapsack DP, subset-sum DP, assignment brute force, tiny-ILP box enumeration (expected result may be `Infeasible` — whatever enumeration says), mixed ILP+continuous with constructed optimum |
| `netlib/*` | 14 | official netlib optimal values (`lp/data/readme`), 11 significant digits; parsed with microlp's `MpsFile` and cross-validated against an independent reader |
| `miplib/*` | 26 | MIPLIB 3 official catalog (`miplib3.cat`): LP relaxation *and* integer optimum per instance |
| `incr/*` | 24 | the incremental Solution APIs: `add_constraint` step chains vs certified optima and a brute-force-checked MILP conflict chain, `fix_var`/`unfix_var` vs from-scratch solves and knapsack DP (including the MILP roundtrip and rejecting fractional fixes of integer variables), `add_gomory_cut` validity vs an ILP enumeration oracle (a cut must never push the LP objective past the true integer optimum) plus the exact textbook cut sequence, and `resume` after `Duration::ZERO` and mid-simplex interruptions. The LP-level cases pass today; the `*-milp*` edit cases assert behavior the solver gets wrong (see Known failures) and sit in the hard tier. |

In total 213 cases (195 in the default release run, 18 more with `--hard`):
78 are pure-LP solves and 16 more exercise LP-level incremental editing and
cutting planes; the rest exercise branch & bound. 17 cases expect
`Infeasible` (11 by construction plus 6 of the random box-ILPs, proven
infeasible by their enumeration oracle) and 5 expect `Unbounded`. See
[data/README.md](data/README.md) for per-instance attribution, contributors,
citations and the licensing status of the vendored benchmark files (kept out
of the published crate on purpose).

## Tiers

* `quick` — milliseconds each; runs in debug builds and in CI's debug job.
* `standard` — the default release-mode suite; the whole run takes ~10 s. This
  plus `quick` is what CI runs, and it is always fully green.
* `hard` — `--hard` only: long-running MIPLIB instances (a naive branch &
  bound needs minutes) together with the cases that pin the known solver bugs
  below. Not run in CI, both because of runtime and because these cases fail
  on purpose. Useful as a stress/progress meter for solver work.

## Known failures (real solver bugs, hard tier)

The suite found three solver bugs. Each has case(s) that assert the correct
behavior and therefore fail until the bug is fixed; all are quarantined in the
hard tier so the default (CI) run stays green. Fixing these is separate work.

1. **`netlib/brandy` — feasible LP reported infeasible.** microlp returns
   `Infeasible` for a feasible netlib instance (official optimum
   1518.5098965). The parse is verified correct (249 columns / 220 rows,
   matching the official statistics, and the suite's independent reader
   agrees), pointing at phase-1 simplex.

2. **Time-limit interrupt panics branch & bound.** Whenever a time limit fires
   while branch & bound is still running, the solver panics
   (`assertion failed: self.is_primal_feasible`, solver.rs:787) instead of
   returning `StopReason::Limit` with the best solution so far. Reproduces on
   any MILP interrupted mid-search, e.g.
   `-- "miplib/stein27/int" --timeout-scale 0.003`. Cases: `miplib/{mod008,
   vpm1,gt2,bell3a}/int` (overrun their budgets and so `PANIC` instead of
   `TIMEOUT`) and `incr/resume-midway-milp` (panics before `resume` can run).

3. **Incremental edits on MILP solutions.** After a MILP solve,
   `Solution::add_constraint` and `Solution::fix_var` act on the branch &
   bound incumbent leaf — whose solver still carries the branch path's
   bound-rows and variable fixings — instead of the original problem, so
   feasible edits return a false `Infeasible` (or a wrong/fractional value).
   The LP behavior of the same APIs is correct and covered by the default-tier
   `incr/*-cert`, `incr/gomory-*` and `incr/resume-*-lp` cases. Cases:
   `incr/add-constraint-milp`, `incr/fix-var-milp`,
   `incr/add-constraint-milp-chain`, `incr/fix-unfix-milp-roundtrip`,
   `incr/fix-var-milp-fractional`, `incr/add-constraint-mixed`.

## Adding a case

Add a closure to the matching family file under `cases/` (or a new file wired
into `cases/mod.rs`). Build the instance through `model::Builder` (it records
the shadow model), return an `Expected`, and give the case a stable unique
name. If the optimum has alternate solutions, assert the objective only —
never variable values.
