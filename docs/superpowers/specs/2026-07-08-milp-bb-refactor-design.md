# MILP Branch & Bound Refactor — Design

Date: 2026-07-08
Status: approved (design), not yet implemented
Scope: the integer (branch & bound) layer of microlp. The LP simplex core is not redesigned.

## 1. Motivation

The current B&B (`solve_integer`, src/solver.rs) drives the search through the public
incremental API (`Solution::add_constraint` / `fix_var`) instead of purpose-built machinery.
Consequences, confirmed by the correctness suite (`tests/suite`):

**Correctness**
- C1. Deadline firing inside a child's LP re-solve leaves the solver primal-infeasible with
  half-pivoted values; the B&B loop ignores the child's stop reason, prunes/branches on garbage,
  and the next `add_constraint` panics on `assert!(self.is_primal_feasible)` (solver.rs:787).
  Suite: PANIC on mod008/vpm1/gt2/bell3a budgets, `milp/time-limit-interrupt`.
- C2. Post-MILP incremental edits act on the B&B incumbent leaf (branch bound-rows and fixings
  still applied) → false `Infeasible` on feasible edits. Suite: `incr/*-milp*` known failures.
- C3. Any `Err` from a child solve — including `InternalError` from LU failure — is treated as
  "infeasible, prune", so numerical failure can silently produce a wrong answer.
- C4. Time limit with no incumbent returns the fractional LP relaxation as a `Solution` with no
  way to detect it; `var_value()` panics on fractional values.
- C5. Integrality tolerance is `EPS = 1e-10` in branching but `1e-5` in `var_value()`.

**Performance**
- P1. Branching adds a constraint row (`x ≤ ⌊v⌋` / `x ≥ ⌈v⌉`): full matrix rebuild, a new slack
  var per depth level, LU refactorization from scratch at every node.
- P2. Every tree node stores a complete `Solution` containing a complete `Solver` clone
  (matrices in CSR + CSC, LU factors, eta matrices); nodes are cloned again when popped.
- P3. Most-fractional branch variable selection (performs ≈ random).
- P4. Pure DFS, no global dual bound → no MIP gap, no gap-based stopping, no early incumbents.

**Design**
- D1. Type cycle `Solver ⊃ bb_state ⊃ Step ⊃ Solution ⊃ Solver`; tree, LP engine, and
  user-facing result are one interleaved thing. Pause/resume, warm start, and problem editing
  are all painful because of this coupling.

## 2. Decisions taken (with user)

- Public API may break freely; next release bumps version (v0.5).
- Resumable state is in-memory only (no serde requirement now; nodes are plain data anyway,
  so serialization can be added later).
- Single-threaded, WASM-friendly (web-time stays). No parallel tree search.
- SOS1 and similar structure-exploiting optimizations are **deferred entirely**. When they come,
  the preference is *implicit detection* (presolve recognizes structure, e.g. `Σ binaries ≤ 1`
  rows, and enables SOS branching) so the modeling user does nothing. The node representation
  below keeps that door open (a branch = an arbitrary set of bound changes).
- Resume and post-solve editing live on `Solution` (one type), not a separate handle.
- Nothing is committed to git without explicit request; work stays local.

## 3. Architecture

```
src/solver.rs        LP simplex engine. Core untouched; gains set_var_bounds and
                     basis snapshot/load. Loses solve_integer, Step, BranchAndBoundState,
                     bb_state, choose_branch_var, new_steps, is_solution_better.
src/mip/mod.rs       B&B driver: main loop, MipState, interruption, incumbent management.
src/mip/node.rs      Node type, open-node storage, bound-change application.
src/mip/branching.rs Branch variable selection (fractional in phase 1, pseudocost in phase 2).
```

The MIP driver owns exactly one `Solver` for the whole search. The LP matrix never grows during
B&B: branching only changes variable bounds.

## 4. LP engine additions (solver.rs)

### 4.1 `set_var_bounds(&mut self, var: usize, lb: f64, ub: f64) -> Result<(), Error>`

- If `lb > ub` (beyond tolerance): return `Err(Infeasible)` (caller prunes without LP work).
- Updates `orig_var_mins/maxs[var]`.
- Var non-basic: if its current value falls outside `[lb, ub]`, shift it to the nearest bound and
  propagate the delta into `basic_var_vals` incrementally via the var's column (same mechanism as
  the non-basic arm of `fix_var`); recompute `at_min`/`at_max` flags against the new bounds;
  mark primal-infeasible if any basic value left its range.
- Var basic: update the `basic_var_mins/maxs` mirrors; mark primal-infeasible if its value is
  now out of range.
- Does NOT run simplex; the driver calls reoptimization explicitly.
- Rationale: tightening a bound keeps the basis dual-feasible (reduced costs untouched), so the
  follow-up is a short dual-simplex run warm-started from the parent basis.
- Loosening is also supported (needed when jumping between tree nodes); correctness there is
  guaranteed because every jump is followed by `load_basis`, which recomputes non-basic values
  and flags from statuses + current bounds. Dives (parent → child) only ever tighten.

### 4.2 Basis snapshot / load

```rust
enum VarStatus { Basic, AtLower, AtUpper, Free }   // Free: non-basic free var, value 0
struct Basis(Vec<VarStatus>);                       // one entry per total var (structural+slack)
fn snapshot_basis(&self) -> Basis;
fn load_basis(&mut self, b: &Basis) -> Result<(), Error>;
```

- `load_basis`: assign basic vars to rows in ascending var order (row assignment is arbitrary,
  any permutation denotes the same basis), set non-basic values from statuses + current bounds
  (`AtLower` → lb, `AtUpper` → ub, `Free` → 0; a stored status pointing at an infinite bound maps
  to the nearest finite bound or 0), rebuild `var_states`, recompute basic values
  (`recalc_basic_var_vals`) and reduced costs (`recalc_obj_coeffs`), refactorize LU
  (`basis_solver.reset`).
- Validation: number of `Basic` statuses must equal the constraint count; on mismatch or LU
  failure (singular basis from numerics), **fall back to the all-slack basis and solve the node
  from scratch** instead of erroring. Robustness valve; log at debug level.
- A cumulative LP iteration counter (`u64`) is added to the pivot loops so the driver can report
  per-node and total simplex work in `Stats`.

### 4.3 Kept as-is

`add_constraint`, `fix_var`, `unfix_var`, `add_gomory_cut` remain for the LP incremental API;
B&B no longer calls them. Deadline checks inside `optimize()`/`restore_feasibility()` remain
(every 1000 iterations) and still return `StopReason::Limit` early — the MIP driver treats that
as "node unsolved", never as a result.

## 5. MIP driver

### 5.1 Data

```rust
struct Node {
    bound_changes: Vec<(usize, f64, f64)>, // cumulative from root: (var, lb, ub); later entries win
    basis: Basis,                          // parent's optimal basis (warm start)
    lp_bound: f64,                         // parent's LP objective (internal minimize space)
    depth: u32,
}

struct MipState {
    solver: Solver,                    // the single LP instance
    root_bounds: Vec<(f64, f64)>,      // original var bounds (to reset when jumping)
    applied: Vec<(usize, f64, f64)>,   // bound changes currently applied to `solver`
    open: Vec<Node>,
    incumbent: Option<Incumbent>,      // values: Vec<f64> (structural vars), obj: f64 (internal space)
    best_bound: f64,                   // min over open nodes' lp_bound and the current path
    pseudocosts: PseudoCosts,          // phase 2
    stats: Stats,                      // nodes, LP iterations, elapsed, gap
    options: SolveOptions,
}
```

Everything except `solver` is plain data. All objectives inside the driver are in the solver's
internal minimize space; `OptimizationDirection` is applied only at the `Solution` boundary.

### 5.2 Main loop (per node)

1. Check limits (deadline, node limit). If hit → return `Interrupted`/`Feasible` (see §5.3).
2. Pop node. Phase 1: plain DFS (last pushed). Phase 2: DFS while plunging, and when a dive
   ends (prune/infeasible/integral) jump to the best-bound open node (linear scan). The pop
   policy is a single function; only it changes between phases.
3. Bound-prune with stored `node.lp_bound` vs incumbent (no LP work).
4. Apply node state: diff `node.bound_changes` against `applied`, reset vars that are no longer
   changed to `root_bounds`, apply new bounds (`set_var_bounds`). Parent→child dive: exactly one
   change, no basis load. Jump: also `load_basis(&node.basis)`.
5. Reoptimize (dual simplex). Outcomes:
   - `Infeasible` → prune, continue.
   - `Limit` → push the node back **unsolved** and return (see §5.3).
   - `InternalError` → propagate as `Err` (never silently prune). C3 fixed.
   - solved → fresh objective `z`.
6. Bound-prune again with `z`.
7. Find fractional integer vars (|v − round(v)| > `int_tol`, default 1e-6). None → candidate
   incumbent: store values (structural vars, as-is; rounding happens at the `var_value()`
   boundary like today), update incumbent/best bound, and if gap ≤ `mip_gap` → finish
   (gap-based finish needs the phase-2 global bound; in phase 1 the loop simply runs until
   `open` is empty).
8. Else choose branch var (phase 1: most-fractional for behavioral parity; phase 2: pseudocost),
   create two children `{bound_changes: parent's + new bound, basis: snapshot, lp_bound: z}`,
   push with the preferred direction last (so it's dived first).

Termination: `open` empty → `Optimal` (or `Err(Infeasible)` if no incumbent was ever found —
same semantics as today). Boolean vars need no special casing: branching on binaries produces
`ub=0` / `lb=1` bound changes naturally.

### 5.3 Interruption (fixes C1, C4 structurally)

- Limits are checked between nodes; additionally the LP itself gives up via its internal
  deadline checks. In that case the current node is **requeued unsolved** — its pre-solve
  description (bounds + parent basis + parent bound) is still in hand, and the solver's
  half-pivoted state is never consulted: the next visit re-applies bounds and reloads a basis.
  Cost of an interruption ≤ one node's LP work.
- Return value on interruption: `Feasible` if an incumbent exists (values = incumbent, gap
  available), else `Interrupted` (no values; accessors panic with a clear message, documented).
- `resume(new_limit)`: sets the deadline and re-enters the loop. `Optimal` resumes are no-ops.
- Pruning tolerance: a node is pruned when `node_bound ≥ incumbent − max(1e-9, 1e-9·|incumbent|)`
  (internal space). Default `mip_gap = 0.0` so the suite's exact-optimum assertions keep holding.

## 6. Public API (v0.5, breaking)

`Problem` construction is unchanged (`add_var`, `add_integer_var`, `add_binary_var`,
`add_constraint`, MPS). `StopReason` is removed; `Error` keeps `Infeasible`/`Unbounded`/
`InternalError`. Time limits are a **status, not an error**.

```rust
#[non_exhaustive]
pub struct SolveOptions {
    pub time_limit: Option<Duration>,   // None = unlimited
    pub node_limit: Option<u64>,        // deterministic interruption (tests!)
    pub mip_gap: f64,                   // relative; default 0.0
    pub int_tol: f64,                   // default 1e-6
    pub warm_start: Option<Vec<(Variable, f64)>>,  // phase 3
}

pub enum Status {
    Optimal,      // proven within mip_gap
    Feasible,     // limit hit; incumbent available; gap() reports quality
    Interrupted,  // limit hit; no incumbent yet; no values available
}

impl Problem {
    pub fn solve(&self) -> Result<Solution, Error>;                 // default options
    pub fn solve_with(&self, opts: &SolveOptions) -> Result<Solution, Error>;
    // set_time_limit(...) kept as convenience, feeds default options
}

impl Solution {
    pub fn status(&self) -> Status;
    pub fn objective(&self) -> f64;                 // panics if Interrupted (documented)
    pub fn var_value(&self, v: Variable) -> f64;    // rounds int vars; panics if Interrupted
    pub fn var_value_raw(&self, v: Variable) -> f64;
    pub fn gap(&self) -> Option<f64>;               // None until an incumbent exists
    pub fn stats(&self) -> &Stats;                  // nodes, LP iters, best bound, elapsed
    pub fn iter(&self) -> impl Iterator<Item = (Variable, f64)>;
    pub fn resume(self, time_limit: Option<Duration>) -> Result<Solution, Error>;
    // post-solve edits — MILP-correct semantics (§7):
    pub fn add_constraint(self, expr, op, rhs) -> Result<Solution, Error>;
    pub fn fix_var(self, v: Variable, val: f64) -> Result<Solution, Error>;
    pub fn unfix_var(self, v: Variable) -> (Solution, bool);
    pub fn add_gomory_cut(self, v: Variable) -> Result<Solution, Error>;  // LP solutions only
}
```

Internally:

```rust
enum SolutionKind {
    Lp(Box<Solver>),                       // live solver: true incremental edits + LP resume
    Mip { state: Box<MipState>, base: Problem },  // plain values + resumable search state
}
```

- MILP `Solution` exposes plain copied values; no solver machinery is reachable from accessors.
- `base` is a copy of the user's `Problem` (needed for edit semantics). Memory note: one copy,
  ~nnz-sized; acceptable. `MipState` is kept even after `Optimal` (open list empty) so edits and
  warm-started re-solves work; users who only read values pay the memory until drop, same as
  today but without the per-node clones.
- LP solutions keep today's behavior: live-basis incremental `add_constraint`/`fix_var`
  (dual-simplex warm start) and mid-simplex resume via `initial_solve` continuation.

## 7. Warm start and post-solve editing

### 7.1 Warm start (`SolveOptions::warm_start`)

Validation before the tree starts: clamp/round the hinted values for integer vars, fix all
hinted vars to those values, solve the LP. Feasible → the result is the initial incumbent
(immediate pruning cutoff). **Hints are advisory**: an infeasible hint must not kill the solve —
log at debug level, ignore the hint, proceed cold. Partial hints (a subset of vars) are allowed;
unhinted vars come from the LP completion.

### 7.2 Post-solve / post-pause edits (fixes C2)

For MILP solutions, `add_constraint`/`fix_var`/`unfix_var`:
1. Apply the edit to `base` (the stored clean copy of the problem).
2. Drop the open tree (existing node bounds may exclude newly-optimal regions only when the
   edit *relaxes*; a conservative and always-sound rule is: any edit invalidates the tree).
3. Revalidate the old incumbent against the edited problem (cheap constraint check). Still
   feasible → it seeds the new solve as warm start / cutoff.
4. Re-run the MIP solve (fresh root LP on base+edits).

This is the clean version of the previously reverted `clean_solver` fix. `add_gomory_cut`
remains LP-only (it reads simplex tableau rows; on a MILP solution it returns
`Err(InternalError)` with a clear message rather than panicking).

Editing after a pause is the same path: `Feasible`/`Interrupted` solutions expose the same edit
methods; the open tree is dropped, incumbent revalidated, search restarts warm.

## 8. Error handling & tolerances (summary)

| Situation | Behavior |
|---|---|
| Child LP infeasible | prune (correct) |
| Child LP `InternalError` | propagate `Err` up (C3) |
| Child LP unbounded | impossible in a bounded node → `InternalError` |
| Basis load fails / singular LU | fall back to all-slack basis, re-solve node from scratch |
| Deadline mid-LP | requeue node unsolved; return with status (C1) |
| Limit, no incumbent | `Status::Interrupted`; accessors panic, documented (C4) |
| Integrality | `int_tol = 1e-6` default, option (C5) |
| Pruning | `bound ≥ incumbent − max(1e-9, 1e-9·|incumbent|)` |
| `lb > ub` after branch | prune without LP work |

## 9. Testing & acceptance

- `tests/suite` default tier stays green at the end of every phase
  (`cargo test --release --test suite`).
- Phase 1 acceptance: hard-tier `milp/time-limit-interrupt` passes; the PANIC-at-budget cases
  (mod008, vpm1, gt2, bell3a) no longer panic (they may still time out, but return
  `Feasible`/`Interrupted` cleanly).
- Phase 3 acceptance: hard-tier `incr/*-milp*` cases pass.
- `netlib/brandy` stays a known failure (phase-1 simplex LP bug — explicitly out of scope).
- New unit tests: `set_var_bounds` (non-basic in/out of range, basic, tighten/loosen, crossing
  bounds), basis snapshot/load round-trip (solve → snapshot → perturb → load → same objective),
  interrupt-requeue determinism via `node_limit`, warm-start accepted/ignored, edit-after-solve
  and edit-after-pause equivalence to from-scratch solves.
- `src/tests/resume.rs` is updated to the new API; the knapsack resume test doubles as a
  performance canary (expected to speed up materially).

## 10. Phasing (each phase lands green and independently useful)

1. **Core refactor.** `set_var_bounds`, basis snapshot/load, `mip/` module with DFS +
   most-fractional (behavioral parity), interrupt-safe requeue semantics, plain-data `Solution`,
   `Status`, error propagation. Deletes old B&B. Fixes C1, C3, C4, C5, P1, P2, D1.
2. **Search quality.** Pseudocost branching, plunging + best-bound jumps, global dual bound,
   `gap()`, `node_limit`, `Stats`. Fixes P3, P4.
3. **Warm start + editing.** `SolveOptions::warm_start`, MILP-correct `Solution` edits,
   edit-after-pause. Fixes C2.
4. **Deferred (future work, not in this refactor).** Implicit SOS1 detection as presolve
   (recognize `Σ binaries ≤ 1` rows, branch on the set — `bound_changes` being a list already
   supports multi-var branches), rounding/diving heuristics, integer presolve, MPS INTORG
   support, brandy phase-1 LP fix, serde for `MipState`.

## 11. Out of scope

The LP simplex core (pricing, ratio tests, LU), the `netlib/brandy` phase-1 bug, parallelism,
serialization, cut generation inside B&B (Gomory cuts stay a user-driven API), and any
structure-exploiting optimization (SOS1/SOS2, cliques, covers).
