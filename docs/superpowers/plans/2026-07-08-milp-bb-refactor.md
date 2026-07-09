# MILP Branch & Bound Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the constraint-row-based, solver-cloning branch & bound in `src/solver.rs` with a bound-change-based B&B in a new `src/mip/` module, with structurally safe interruption/resume, then add pseudocost branching + gap reporting (phase 2) and warm start + MILP-correct post-solve editing (phase 3).

**Architecture:** One persistent `Solver` per search; branching = variable bound changes (`set_var_bounds`) + dual-simplex warm starts; tree nodes are plain data (cumulative bound changes + basis snapshot + parent LP bound); a deadline firing mid-LP requeues the node unsolved so half-pivoted state is never consulted. `Solution` becomes plain data for MILP (values + `Status` + boxed resumable `MipState`).

**Tech Stack:** Rust 2021, no new dependencies. Existing crates: `sprs`, `log`, `web-time`.

**Spec:** `docs/superpowers/specs/2026-07-08-milp-bb-refactor-design.md` (approved). Bug labels C1–C5, P1–P4, D1 below refer to the spec's §1.

## Global Constraints

- **NEVER run `git commit`, `git push`, or any history-mutating git command.** The user curates commits themselves. All work stays in the working tree. Where a workflow would normally commit, run the task's verification commands instead.
- Rust edition 2021. `src/lib.rs` has `#![deny(missing_debug_implementations, missing_docs)]` — every `pub` item you add needs a `///` doc comment and every type reachable from a `Debug` impl needs `Debug`.
- No new dependencies in `Cargo.toml`.
- In `src/`, only use `web_time::Instant` (WASM target), never `std::time::Instant`. (Test-only code may use std.)
- Determinism: no `HashMap` iteration anywhere in solve paths — use `BTreeMap` or sorted `Vec`s.
- Full-test command: `cargo test --release` (also runs doctests). Correctness suite: `cargo test --release --test suite` (default tier, ~195 cases, must be green at the end of Tasks 5, 6, 7, 8, 9, 10, 11, 12, 13). Hard tier: `cargo test --release --test suite -- --hard`.
- Run `cargo fmt` at the end of every task.
- The internal objective space is always MINIMIZE (`Problem` negates coefficients for `Maximize` at variable-creation time). All `mip::` code works in internal space; sign flips happen only in `Solution` accessors and user-facing `Stats`.
- `EPS` (1e-10) stays the simplex pivot tolerance. Integrality decisions use `SolveOptions::int_tol` (default 1e-6) — never `EPS`.

## File Structure

```
src/solver.rs         MODIFY  +get_var_bounds/set_var_bounds/reoptimize/lp_iterations (Task 1)
                              +VarStatus/Basis/snapshot_basis/slack_basis/load_basis (Task 2)
                              -solve_integer, Step, BranchKind, BranchAndBoundState, bb_state,
                               choose_branch_var, get_branch_min_max, new_steps,
                               is_solution_better (Task 5)
src/mip/node.rs       CREATE  Node, effective_bounds (Task 3)
src/mip/branching.rs  CREATE  is_integral, choose_branch_var (Task 3); pseudocosts (Task 10)
src/mip/mod.rs        CREATE  SolveOptions, Status, Stats, Incumbent, MipState, MipOutcome,
                              MipRun, run, resume_run, status_of (Tasks 3–4);
                              gap/best-bound (8), plunge pop (9), warm start (11), edits (12)
src/lib.rs            MODIFY  Solution rebuild, solve_with, re-exports, StopReason→pub(crate) (Task 5)
src/tests/mip_api.rs  CREATE  public-API MILP tests (Task 5)
src/tests/*.rs        MODIFY  API migration (Task 5)
examples/*.rs         MODIFY  API migration if needed (Task 5)
tests/suite/**        MODIFY  API migration (Task 6), flip known-failures (Tasks 6, 13)
```

## API migration mapping (used by Tasks 5 and 6)

| Old | New |
|---|---|
| `StopReason` (public) | `Status` — `pub(crate) StopReason` remains internal-only |
| `sol.stop_reason() == &StopReason::Finished` | `sol.status() == Status::Optimal` |
| `sol.stop_reason() == &StopReason::Limit` | `matches!(sol.status(), Status::Feasible \| Status::Interrupted)` |
| `sol.var_value_raw(v) -> &f64` | `sol.var_value_raw(v) -> f64` (drop the deref) |
| `sol.iter() -> (Variable, &f64)` | `sol.iter() -> (Variable, f64)` (drop `&` in patterns) |
| `sol[v] -> &f64` via Index | unchanged |
| `sol.objective()`, `sol.var_value(v)` | unchanged signature; now panic on `Status::Interrupted` |
| `sol.resume(limit)` | unchanged signature |
| `sol.add_constraint/fix_var/unfix_var/add_gomory_cut` | unchanged signatures; MILP semantics change (Task 12); LP path unchanged |

Never weaken a test assertion during migration — translate it 1:1 via this table.

---

### Task 1: `Solver::set_var_bounds` + `reoptimize` + LP iteration counter

**Files:**
- Modify: `src/solver.rs` (impl Solver, around the existing `fix_var` at ~line 443; tests module at the bottom)

**Interfaces:**
- Consumes: existing `Solver` internals (`var_states`, `nb_var_vals`, `basic_var_vals`, `calc_col_coeffs`, `restore_feasibility`, `optimize`, `calc_*_infeasibility`).
- Produces (for Tasks 2–4):
  - `pub(crate) fn get_var_bounds(&self, var: usize) -> (f64, f64)`
  - `pub(crate) fn set_var_bounds(&mut self, var: usize, min: f64, max: f64) -> Result<(), Error>`
  - `pub(crate) fn reoptimize(&mut self) -> Result<StopReason, Error>`
  - `pub(crate) lp_iterations: u64` field on `Solver`
  - `check_deadline` becomes `pub(crate)`

- [ ] **Step 1: Write the failing tests** (append inside `mod tests` in `src/solver.rs`)

```rust
#[test]
fn set_var_bounds_tighten_matches_fresh_solve() {
    init();
    // minimize 2x + 3y s.t. x + y >= 4, 0 <= x,y <= 10. Optimum: x=4, y=0, obj 8.
    let coeffs = [2.0, 3.0];
    let mins = [0.0, 0.0];
    let maxs = [10.0, 10.0];
    let cons = [(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 4.0)];
    let domains = [VarDomain::Real, VarDomain::Real];

    let mut warm = Solver::try_new(&coeffs, &mins, &maxs, &cons, &domains, None).unwrap();
    warm.initial_solve().unwrap();
    assert!(float_eq(warm.cur_obj_val, 8.0));

    // Tighten x to [0, 2] and re-solve warm: optimum becomes x=2, y=2, obj 10.
    warm.set_var_bounds(0, 0.0, 2.0).unwrap();
    assert_eq!(warm.reoptimize().unwrap(), StopReason::Finished);
    assert!(warm.is_primal_feasible && warm.is_dual_feasible);
    assert!(float_eq(warm.cur_obj_val, 10.0));
    assert!(float_eq(*warm.get_value(0), 2.0));
    assert!(float_eq(*warm.get_value(1), 2.0));

    // Fresh solve of the tightened problem must agree.
    let mut fresh = Solver::try_new(&coeffs, &mins, &[2.0, 10.0], &cons, &domains, None).unwrap();
    fresh.initial_solve().unwrap();
    assert!(float_eq(fresh.cur_obj_val, warm.cur_obj_val));
}

#[test]
fn set_var_bounds_loosen_and_retighten() {
    init();
    // maximize x + y (internally minimize -x - y) s.t. x + y <= 4, 0 <= x,y <= 3.
    let mut solver = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[3.0, 3.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    assert!(float_eq(solver.cur_obj_val, -4.0));

    // Tighten x to [0, 0.5]: optimum x=0.5, y=3, obj -3.5.
    solver.set_var_bounds(0, 0.0, 0.5).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, -3.5));

    // Loosen x back to [0, 3]: optimum returns to -4.
    solver.set_var_bounds(0, 0.0, 3.0).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, -4.0));

    assert!(solver.lp_iterations > 0);
}

#[test]
fn set_var_bounds_crossing_is_infeasible_and_leaves_state_untouched() {
    init();
    let mut solver = Solver::try_new(
        &[1.0],
        &[0.0],
        &[10.0],
        &[(to_sparse(&[1.0]), ComparisonOp::Ge, 1.0)],
        &[VarDomain::Real],
        None,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    let obj_before = solver.cur_obj_val;
    assert_eq!(solver.set_var_bounds(0, 2.0, 1.0).unwrap_err(), Error::Infeasible);
    assert_eq!(solver.get_var_bounds(0), (0.0, 10.0)); // untouched
    assert!(float_eq(solver.cur_obj_val, obj_before));
}
```

Note on loosening: after loosening a bound the current point can become dual-infeasible (a
non-basic var no longer sits at a bound that justifies its reduced cost), which is exactly why
`reoptimize` must fall back to primal simplex (`optimize`) when `is_dual_feasible` is false.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release set_var_bounds`
Expected: compile error `no method named 'set_var_bounds' found for struct 'Solver'`.

- [ ] **Step 3: Implement** (in `impl Solver`, next to `fix_var`)

```rust
pub(crate) fn get_var_bounds(&self, var: usize) -> (f64, f64) {
    (self.orig_var_mins[var], self.orig_var_maxs[var])
}

/// Change a variable's bounds in place. Records the new bounds and repairs the
/// invariants that depend on them; does NOT run simplex — call [`Self::reoptimize`]
/// afterwards. Returns `Err(Infeasible)` (state untouched) if `min > max`.
pub(crate) fn set_var_bounds(&mut self, var: usize, min: f64, max: f64) -> Result<(), Error> {
    if min > max {
        return Err(Error::Infeasible);
    }
    self.orig_var_mins[var] = min;
    self.orig_var_maxs[var] = max;
    match self.var_states[var] {
        VarState::Basic(row) => {
            self.basic_var_mins[row] = min;
            self.basic_var_maxs[row] = max;
            let val = self.basic_var_vals[row];
            if val < min - EPS || val > max + EPS {
                self.is_primal_feasible = false;
            }
        }
        VarState::NonBasic(col) => {
            let cur = self.nb_var_vals[col];
            let new_val = cur.clamp(min, max);
            if new_val != cur {
                // Shift the non-basic var to the nearest bound and propagate the
                // delta into basic values (same mechanism as fix_var's non-basic arm).
                self.calc_col_coeffs(col);
                let diff = new_val - cur;
                for (r, coeff) in self.col_coeffs.iter() {
                    self.basic_var_vals[r] -= diff * coeff;
                }
                self.cur_obj_val += diff * self.nb_var_obj_coeffs[col];
                self.nb_var_vals[col] = new_val;
                self.is_primal_feasible = false;
            }
            self.nb_var_states[col] = NonBasicVarState {
                at_min: float_eq(new_val, min),
                at_max: float_eq(new_val, max),
            };
            // A var at a loosened bound may no longer justify its reduced cost.
            self.is_dual_feasible = self.is_dual_feasible
                && (self.nb_var_states[col].at_min && self.nb_var_obj_coeffs[col] > -EPS
                    || self.nb_var_states[col].at_max && self.nb_var_obj_coeffs[col] < EPS
                    || self.nb_var_obj_coeffs[col].abs() < EPS);
        }
    }
    Ok(())
}

/// Re-solve after bound changes or a basis load: dual simplex to restore primal
/// feasibility, then primal simplex if reduced costs became dual-infeasible
/// (only happens after loosening bounds or a numerically imperfect basis load).
pub(crate) fn reoptimize(&mut self) -> Result<StopReason, Error> {
    if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
        return Ok(StopReason::Limit);
    }
    if !self.is_dual_feasible {
        self.recalc_obj_coeffs()?;
        if self.optimize()? == StopReason::Limit {
            return Ok(StopReason::Limit);
        }
        // Primal simplex may have moved through vertices; make sure primal holds too.
        if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
            return Ok(StopReason::Limit);
        }
    }
    Ok(StopReason::Finished)
}
```

Also in this step:
1. Add field `pub(crate) lp_iterations: u64` to `struct Solver` and `lp_iterations: 0` in `Solver::try_new`'s `Self { ... }` literal.
2. In `optimize()` and `restore_feasibility()`, increment it once per pivot: add `self.lp_iterations += 1;` as the first line inside each `for iter in 0..` loop body.
3. Change `fn check_deadline` to `pub(crate) fn check_deadline` (Task 4 uses it from `mip`).
4. `optimize()` sets `self.is_dual_feasible = true` after its loop and `restore_feasibility()` sets `self.is_primal_feasible = true` — verify both lines exist (they do today); do not remove them.

Note: `recalc_obj_coeffs` before `optimize()` mirrors `initial_solve`'s sequence and clears
accumulated float drift in reduced costs before a primal run.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release set_var_bounds`
Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Full check + fmt**

Run: `cargo test --release && cargo fmt`
Expected: all existing tests still pass (nothing else changed behavior).

---

### Task 2: Basis snapshot / load / slack fallback

**Files:**
- Modify: `src/solver.rs` (new types near `VarState`; methods in `impl Solver`; tests at bottom)

**Interfaces:**
- Consumes: Task 1's `reoptimize`; existing `recalc_basic_var_vals` (remove its `#[allow(dead_code)]`), `recalc_obj_coeffs`, `basis_solver.reset`, `calc_primal_infeasibility`, `calc_dual_infeasibility`.
- Produces (for Tasks 3–4):
  - `#[derive(Clone, Debug, PartialEq, Eq)] pub(crate) enum VarStatus { Basic, AtLower, AtUpper, Free }`
  - `#[derive(Clone, Debug)] pub(crate) struct Basis(pub(crate) Vec<VarStatus>)`
  - `pub(crate) fn snapshot_basis(&self) -> Basis`
  - `pub(crate) fn slack_basis(&self) -> Basis`
  - `pub(crate) fn load_basis(&mut self, basis: &Basis) -> Result<(), Error>`

- [ ] **Step 1: Write the failing tests** (append inside `mod tests`)

```rust
#[test]
fn basis_snapshot_load_roundtrip() {
    init();
    let mut solver = Solver::try_new(
        &[2.0, 1.0],
        &[f64::NEG_INFINITY, 5.0],
        &[0.0, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 6.0),
            (to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 8.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    let obj = solver.cur_obj_val;
    let vals: Vec<f64> = (0..2).map(|v| *solver.get_value(v)).collect();
    let basis = solver.snapshot_basis();

    // Wreck the state by loading the all-slack basis…
    let slack = solver.slack_basis();
    solver.load_basis(&slack).unwrap();

    // …then reload the optimal basis: objective and values must round-trip.
    solver.load_basis(&basis).unwrap();
    assert!(solver.is_primal_feasible && solver.is_dual_feasible);
    assert!(float_eq(solver.cur_obj_val, obj));
    for v in 0..2 {
        assert!(float_eq(*solver.get_value(v), vals[v]));
    }
}

#[test]
fn slack_basis_load_then_reoptimize_reaches_optimum() {
    init();
    // minimize 2x + 3y s.t. x + y >= 4, 0 <= x,y <= 10 → obj 8.
    let mut solver = Solver::try_new(
        &[2.0, 3.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 4.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    assert!(float_eq(solver.cur_obj_val, 8.0));

    let slack = solver.slack_basis();
    solver.load_basis(&slack).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, 8.0));
}

#[test]
fn load_basis_rejects_wrong_shape() {
    init();
    let mut solver = Solver::try_new(
        &[1.0],
        &[0.0],
        &[1.0],
        &[(to_sparse(&[1.0]), ComparisonOp::Le, 1.0)],
        &[VarDomain::Real],
        None,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    // 2 total vars (1 structural + 1 slack); a basis with zero Basic entries is invalid.
    let bad = Basis(vec![VarStatus::AtLower, VarStatus::AtLower]);
    assert!(solver.load_basis(&bad).is_err());
    // Solver must still be usable via the slack-basis fallback path.
    let slack = solver.slack_basis();
    solver.load_basis(&slack).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release basis_`
Expected: compile error `cannot find struct 'Basis'` / `no method named 'snapshot_basis'`.

- [ ] **Step 3: Implement**

Add near `enum VarState` (top of `src/solver.rs`):

```rust
/// Status of one variable in a simplex basis snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum VarStatus {
    Basic,
    AtLower,
    AtUpper,
    /// Non-basic free variable (both bounds infinite), pinned at 0.
    Free,
}

/// A compact simplex basis: one status per total var (structural + slack).
/// Together with the current variable bounds it fully determines a vertex.
#[derive(Clone, Debug)]
pub(crate) struct Basis(pub(crate) Vec<VarStatus>);
```

Add in `impl Solver`:

```rust
pub(crate) fn snapshot_basis(&self) -> Basis {
    let mut statuses = Vec::with_capacity(self.num_total_vars());
    for var in 0..self.num_total_vars() {
        statuses.push(match self.var_states[var] {
            VarState::Basic(_) => VarStatus::Basic,
            VarState::NonBasic(col) => {
                let s = &self.nb_var_states[col];
                if s.at_min {
                    VarStatus::AtLower
                } else if s.at_max {
                    VarStatus::AtUpper
                } else {
                    VarStatus::Free
                }
            }
        });
    }
    Basis(statuses)
}

/// The all-slack basis (identity basis matrix). Loading it cannot fail with a
/// singular factorization, so it is the universal fallback.
pub(crate) fn slack_basis(&self) -> Basis {
    let mut statuses = Vec::with_capacity(self.num_total_vars());
    for var in 0..self.num_vars {
        let min = self.orig_var_mins[var];
        let max = self.orig_var_maxs[var];
        statuses.push(if min.is_finite() {
            VarStatus::AtLower
        } else if max.is_finite() {
            VarStatus::AtUpper
        } else {
            VarStatus::Free
        });
    }
    for _ in 0..self.num_constraints() {
        statuses.push(VarStatus::Basic);
    }
    Basis(statuses)
}

/// Rebuild the solver state from a basis snapshot and the CURRENT variable bounds:
/// non-basic values come from statuses + bounds, basic values and reduced costs are
/// recomputed from scratch, and the LU factorization is rebuilt. Feasibility flags
/// are recomputed honestly, so half-pivoted pre-load state is fully discarded.
pub(crate) fn load_basis(&mut self, basis: &Basis) -> Result<(), Error> {
    let n = self.num_total_vars();
    let m = self.num_constraints();
    if basis.0.len() != n || basis.0.iter().filter(|s| **s == VarStatus::Basic).count() != m {
        return Err(Error::InternalError("basis shape mismatch".to_string()));
    }

    self.basic_vars.clear();
    self.basic_var_mins.clear();
    self.basic_var_maxs.clear();
    self.nb_vars.clear();
    self.nb_var_vals.clear();
    self.nb_var_states.clear();
    self.nb_var_is_fixed.clear();

    for var in 0..n {
        match basis.0[var] {
            VarStatus::Basic => {
                self.var_states[var] = VarState::Basic(self.basic_vars.len());
                self.basic_vars.push(var);
                self.basic_var_mins.push(self.orig_var_mins[var]);
                self.basic_var_maxs.push(self.orig_var_maxs[var]);
            }
            ref status => {
                let min = self.orig_var_mins[var];
                let max = self.orig_var_maxs[var];
                let val = match status {
                    VarStatus::AtLower => {
                        if min.is_finite() {
                            min
                        } else if max.is_finite() {
                            max
                        } else {
                            0.0
                        }
                    }
                    VarStatus::AtUpper => {
                        if max.is_finite() {
                            max
                        } else if min.is_finite() {
                            min
                        } else {
                            0.0
                        }
                    }
                    VarStatus::Free => {
                        if min.is_finite() {
                            min
                        } else if max.is_finite() {
                            max
                        } else {
                            0.0
                        }
                    }
                    VarStatus::Basic => unreachable!(),
                };
                self.var_states[var] = VarState::NonBasic(self.nb_vars.len());
                self.nb_vars.push(var);
                self.nb_var_vals.push(val);
                self.nb_var_states.push(NonBasicVarState {
                    at_min: float_eq(val, min),
                    at_max: float_eq(val, max),
                });
                self.nb_var_is_fixed.push(false);
            }
        }
    }

    self.basis_solver
        .reset(&self.orig_constraints_csc, &self.basic_vars)?;

    // Steepest-edge reference reset (standard practice after a warm-start load;
    // only affects pivot ordering quality, not correctness).
    if self.enable_dual_steepest_edge {
        self.dual_edge_sq_norms = vec![1.0; self.basic_vars.len()];
    }

    self.recalc_basic_var_vals()?;
    self.recalc_obj_coeffs()?;

    self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
    self.is_dual_feasible = self.calc_dual_infeasibility().0 == 0;
    Ok(())
}
```

Also: remove `#[allow(dead_code)]` from `recalc_basic_var_vals`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release basis_ && cargo test --release slack_basis`
Expected: `3 passed` across the new tests.

- [ ] **Step 5: Full check + fmt**

Run: `cargo test --release && cargo fmt`
Expected: all pass.

---

### Task 3: `mip` module skeleton — types, Node, fractional branching

**Files:**
- Create: `src/mip/mod.rs`, `src/mip/node.rs`, `src/mip/branching.rs`
- Modify: `src/lib.rs` (add `#[allow(dead_code)] mod mip;` next to the other `mod` items — the allow is removed in Task 5)

**Interfaces:**
- Consumes: `crate::solver::{Solver, Basis}` (Task 2), `crate::{VarDomain, Variable}`.
- Produces (for Tasks 4–12):
  - `mip::Status { Optimal, Feasible, Interrupted }` (pub, Copy, PartialEq)
  - `mip::SolveOptions { time_limit, node_limit, mip_gap, int_tol, warm_start }` (pub, `#[non_exhaustive]`, `Default`)
  - `mip::Stats { nodes_solved: u64, lp_iterations: u64, elapsed: Duration, best_bound: Option<f64>, gap: Option<f64> }` (pub, Copy, `#[non_exhaustive]`, `Default`)
  - `mip::node::Node { bound_changes: Vec<(usize, f64, f64)>, basis: Basis, lp_bound: f64, depth: u32, parent_id: u64 }`
  - `mip::node::effective_bounds(&[(usize, f64, f64)]) -> Vec<(usize, f64, f64)>` (later entries win, sorted by var, deduped)
  - `mip::branching::is_integral(solver: &Solver, domains: &[VarDomain], int_tol: f64) -> bool`
  - `mip::branching::choose_branch_var(solver: &Solver, domains: &[VarDomain], int_tol: f64) -> Option<usize>` (most-fractional)

- [ ] **Step 1: Write the failing tests**

In `src/mip/node.rs` (bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_bounds_later_entries_win() {
        let changes = vec![(3, 0.0, 7.0), (1, 0.0, 1.0), (3, 2.0, 5.0)];
        assert_eq!(effective_bounds(&changes), vec![(1, 0.0, 1.0), (3, 2.0, 5.0)]);
        assert_eq!(effective_bounds(&[]), vec![]);
    }
}
```

In `src/mip/branching.rs` (bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::Solver;
    use crate::{ComparisonOp, VarDomain};

    fn to_sparse(values: &[f64]) -> crate::CsVec {
        let mut indices = vec![];
        let mut data = vec![];
        for (i, &v) in values.iter().enumerate() {
            if v != 0.0 {
                indices.push(i);
                data.push(v);
            }
        }
        crate::CsVec::new(values.len(), indices, data)
    }

    #[test]
    fn most_fractional_var_is_chosen() {
        // minimize -x - y s.t. x + 2y <= 3.2, x <= 1.9; x,y integer-domained.
        // LP optimum: x = 1.9, y = 0.65 → fractional parts 0.9 and 0.65;
        // most-fractional metric |v - round(v)|: x → 0.1, y → 0.35 → picks y (idx 1).
        let mut solver = Solver::try_new(
            &[-1.0, -1.0],
            &[0.0, 0.0],
            &[1.9, 10.0],
            &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 3.2)],
            &[VarDomain::Integer, VarDomain::Integer],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        assert!(!is_integral(&solver, solver.orig_var_domains.clone().as_slice(), 1e-6));
        assert_eq!(choose_branch_var(&solver, solver.orig_var_domains.clone().as_slice(), 1e-6), Some(1));
    }

    #[test]
    fn integral_solution_yields_no_branch_var() {
        // minimize x s.t. x >= 2, x integer in [0, 10] → LP optimum x = 2 (integral).
        let mut solver = Solver::try_new(
            &[1.0],
            &[0.0],
            &[10.0],
            &[(to_sparse(&[2.0]), ComparisonOp::Ge, 4.0)],
            &[VarDomain::Integer],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        assert!(is_integral(&solver, solver.orig_var_domains.clone().as_slice(), 1e-6));
        assert_eq!(choose_branch_var(&solver, solver.orig_var_domains.clone().as_slice(), 1e-6), None);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release mip::`
Expected: compile error (module `mip` does not exist yet).

- [ ] **Step 3: Implement**

`src/mip/node.rs`:

```rust
//! Plain-data branch & bound tree nodes.

use crate::solver::Basis;
use std::collections::BTreeMap;

/// One open node of the search tree. Contains no solver machinery — applying
/// `bound_changes` on top of the root bounds plus loading `basis` fully
/// reconstructs the node's starting state.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    /// Cumulative bound changes from the root, in creation order; for a var that
    /// appears multiple times the LAST entry is its current bounds.
    pub bound_changes: Vec<(usize, f64, f64)>,
    /// Optimal basis of the parent node (dual-feasible warm start after tightening).
    pub basis: Basis,
    /// Parent's LP objective in internal (minimize) space — a valid lower bound
    /// for this node, used for pruning before any LP work.
    pub lp_bound: f64,
    pub depth: u32,
    /// Sequence number of the branching that created this node; used to detect
    /// "the solver is already at my parent's optimum" (warm dive).
    pub parent_id: u64,
}

/// Collapse a bound-change list to one entry per var (later entries win),
/// sorted by var index for deterministic application order.
pub(crate) fn effective_bounds(changes: &[(usize, f64, f64)]) -> Vec<(usize, f64, f64)> {
    let mut map: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for &(v, lo, hi) in changes {
        map.insert(v, (lo, hi));
    }
    map.into_iter().map(|(v, (lo, hi))| (v, lo, hi)).collect()
}
```

`src/mip/branching.rs`:

```rust
//! Branch variable selection.

use crate::solver::Solver;
use crate::VarDomain;

fn fractionality(val: f64) -> f64 {
    (val - val.round()).abs()
}

fn is_int_domain(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

/// True if every integer-domained structural var is within `int_tol` of an integer.
pub(crate) fn is_integral(solver: &Solver, domains: &[VarDomain], int_tol: f64) -> bool {
    domains
        .iter()
        .enumerate()
        .filter(|(_, d)| is_int_domain(d))
        .all(|(v, _)| fractionality(*solver.get_value(v)) <= int_tol)
}

/// Most-fractional rule (phase-1 parity; replaced by pseudocosts in phase 2).
pub(crate) fn choose_branch_var(
    solver: &Solver,
    domains: &[VarDomain],
    int_tol: f64,
) -> Option<usize> {
    let mut best = None;
    let mut best_frac = int_tol;
    for (v, d) in domains.iter().enumerate() {
        if !is_int_domain(d) {
            continue;
        }
        let frac = fractionality(*solver.get_value(v));
        if frac > best_frac {
            best_frac = frac;
            best = Some(v);
        }
    }
    best
}
```

`src/mip/mod.rs`:

```rust
//! Branch & bound driver for mixed-integer problems.
//!
//! Owns exactly one [`Solver`] per search. Branching changes variable bounds in
//! place (never adds constraint rows), so the LP never grows during the search.

pub(crate) mod branching;
pub(crate) mod node;

use crate::Variable;
use core::time::Duration;

/// The outcome class of a finished or interrupted solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Proven optimal (within the configured `mip_gap`, which defaults to exact).
    Optimal,
    /// A limit was hit; a feasible solution is available but optimality is unproven.
    Feasible,
    /// A limit was hit before any usable solution was found. Value accessors
    /// ([`crate::Solution::objective`] etc.) panic on such solutions; call
    /// [`crate::Solution::resume`] to continue the search.
    Interrupted,
}

/// Options controlling a solve. Construct with [`SolveOptions::default`] and
/// mutate the fields you need.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolveOptions {
    /// Wall-clock budget for this call (`None` = unlimited). On expiry the search
    /// stops cleanly and can be resumed.
    pub time_limit: Option<Duration>,
    /// Maximum number of branch & bound nodes to solve in this call
    /// (`None` = unlimited). Deterministic alternative to `time_limit`; the
    /// budget applies per call, so each `resume` gets a fresh budget.
    /// The root relaxation does not count as a node.
    pub node_limit: Option<u64>,
    /// Relative MIP gap at which the search stops and reports [`Status::Optimal`].
    /// Default `0.0` (prove exact optimality).
    pub mip_gap: f64,
    /// Integrality tolerance: a value within this distance of an integer counts
    /// as integral. Default `1e-6`.
    pub int_tol: f64,
    /// Optional (partial) starting assignment used to seed the incumbent.
    /// Advisory: an infeasible or incomplete hint is ignored. Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            time_limit: None,
            node_limit: None,
            mip_gap: 0.0,
            int_tol: 1e-6,
            warm_start: None,
        }
    }
}

/// Statistics of a solve, available via [`crate::Solution::stats`].
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Stats {
    /// Branch & bound nodes whose LP was solved (0 for pure-LP problems).
    pub nodes_solved: u64,
    /// Total simplex pivots across the whole solve (including the root LP).
    pub lp_iterations: u64,
    /// Wall-clock time spent inside the solver, accumulated across resumes.
    pub elapsed: Duration,
    /// Best proven bound on the objective, in user space (filled in phase 2).
    pub best_bound: Option<f64>,
    /// Relative gap between incumbent and best bound (filled in phase 2).
    pub gap: Option<f64>,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release mip::`
Expected: `test result: ok. 3 passed` (2 branching + 1 node test).

- [ ] **Step 5: Full check + fmt**

Run: `cargo test --release && cargo fmt`
Expected: all pass. (The `#[allow(dead_code)]` on `mod mip` silences unused warnings until Task 5 wires it in.)

---

### Task 4: The B&B driver — `MipState`, `run`, `resume_run`

**Files:**
- Modify: `src/mip/mod.rs` (driver + tests)

**Interfaces:**
- Consumes: Tasks 1–3 (`set_var_bounds`, `get_var_bounds`, `reoptimize`, `snapshot_basis`, `slack_basis`, `load_basis`, `check_deadline`, `lp_iterations`, `Node`, `effective_bounds`, `is_integral`, `choose_branch_var`), `crate::Problem` private fields (accessible: `mip` is a descendant module of the crate root), `crate::solver::{Deadline, StopReason}`.
- Produces (for Task 5):
  - `pub(crate) struct Incumbent { pub values: Vec<f64>, pub objective: f64 }` (internal space)
  - `pub(crate) struct MipState { solver, root_bounds, applied, open, incumbent, node_seq, last_solved_id, root_solved, stats, options, deadline, direction }` (all fields `pub(crate)`)
  - `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub(crate) enum MipOutcome { Optimal, Interrupted }`
  - `pub(crate) struct MipRun { pub outcome: MipOutcome, pub state: MipState }`
  - `pub(crate) fn run(problem: &Problem, options: SolveOptions) -> Result<MipRun, Error>`
  - `pub(crate) fn resume_run(state: &mut MipState, time_limit: Option<Duration>) -> Result<MipOutcome, Error>`
  - `pub(crate) fn status_of(outcome: MipOutcome, state: &MipState) -> Status`

- [ ] **Step 1: Write the failing tests** (append `#[cfg(test)] mod tests` at the bottom of `src/mip/mod.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    fn int_2var_problem() -> Problem {
        // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b integer in [0,10].
        // LP relaxation: a=0.6, b=2.2, obj 10.6. Integer optimum: a=1, b=2, obj 11.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let a = p.add_integer_var(3.0, (0, 10));
        let b = p.add_integer_var(4.0, (0, 10));
        p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
        p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
        p
    }

    fn binary_knapsack() -> Problem {
        // maximize 8x + 11y + 6z + 4w s.t. 5x + 7y + 4z + 3w <= 14, binaries.
        // Optimum: y + z + w = 21 (weight 14).
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_binary_var(8.0);
        let y = p.add_binary_var(11.0);
        let z = p.add_binary_var(6.0);
        let w = p.add_binary_var(4.0);
        p.add_constraint(
            &[(x, 5.0), (y, 7.0), (z, 4.0), (w, 3.0)],
            ComparisonOp::Le,
            14.0,
        );
        p
    }

    fn incumbent_obj(state: &MipState) -> f64 {
        state.incumbent.as_ref().unwrap().objective
    }

    #[test]
    fn driver_finds_integer_optimum() {
        let run = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert_eq!(run.outcome, MipOutcome::Optimal);
        // Internal space == user space for Minimize.
        assert!((incumbent_obj(&run.state) - 11.0).abs() < 1e-6);
        let inc = run.state.incumbent.as_ref().unwrap();
        assert!((inc.values[0] - 1.0).abs() < 1e-6);
        assert!((inc.values[1] - 2.0).abs() < 1e-6);
        assert!(run.state.stats.nodes_solved > 0);
    }

    #[test]
    fn driver_binary_knapsack_maximize() {
        let run = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert_eq!(run.outcome, MipOutcome::Optimal);
        // Maximize is negated internally: internal optimum is -21.
        assert!((incumbent_obj(&run.state) + 21.0).abs() < 1e-6);
    }

    #[test]
    fn driver_no_integer_point_is_infeasible() {
        // 2x == 1 with x integer in [0,10]: LP-feasible (x=0.5), integer-infeasible.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        assert_eq!(
            run(&p, SolveOptions::default()).unwrap_err(),
            crate::Error::Infeasible
        );
    }

    #[test]
    fn driver_node_limit_interrupts_and_resumes_to_same_optimum() {
        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut r = run(&int_2var_problem(), options).unwrap();
        let mut guard = 0;
        while r.outcome == MipOutcome::Interrupted {
            guard += 1;
            assert!(guard < 10_000, "resume loop did not terminate");
            r.outcome = resume_run(&mut r.state, None).unwrap();
        }
        assert!(guard >= 1, "node_limit=1 should interrupt at least once");
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn driver_zero_time_limit_interrupts_cleanly_then_resumes() {
        let mut options = SolveOptions::default();
        options.time_limit = Some(Duration::ZERO);
        let mut r = run(&binary_knapsack(), options).unwrap();
        assert_eq!(r.outcome, MipOutcome::Interrupted);
        assert!(r.state.incumbent.is_none());
        assert_eq!(status_of(r.outcome, &r.state), Status::Interrupted);
        let outcome = resume_run(&mut r.state, None).unwrap();
        assert_eq!(outcome, MipOutcome::Optimal);
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release mip::`
Expected: compile error (`run`, `MipState`, … not defined).

- [ ] **Step 3: Implement the driver** (in `src/mip/mod.rs`, above the tests)

```rust
use crate::solver::{check_deadline, Deadline, Solver};
use crate::{Error, OptimizationDirection, Problem, StopReason};
use node::{effective_bounds, Node};
use web_time::Instant;

/// A feasible integer assignment, in internal (minimize) objective space.
#[derive(Clone, Debug)]
pub(crate) struct Incumbent {
    /// Values of the structural variables (length = Problem var count).
    pub values: Vec<f64>,
    pub objective: f64,
}

/// The complete, resumable state of a branch & bound search.
#[derive(Clone)]
pub(crate) struct MipState {
    pub solver: Solver,
    /// Original bounds of the structural vars (to reset when jumping between nodes).
    pub root_bounds: Vec<(f64, f64)>,
    /// Bound changes currently applied to `solver` (collapsed, sorted by var).
    pub applied: Vec<(usize, f64, f64)>,
    pub open: Vec<Node>,
    pub incumbent: Option<Incumbent>,
    /// Sequence counter for branchings; children carry it as `parent_id`.
    pub node_seq: u64,
    /// `Some(id)` iff `solver` currently holds the optimal basis + bounds of the
    /// branching with that id — its children can skip the basis load (warm dive).
    pub last_solved_id: Option<u64>,
    pub root_solved: bool,
    pub stats: Stats,
    pub options: SolveOptions,
    pub deadline: Deadline,
    pub direction: OptimizationDirection,
}

impl std::fmt::Debug for MipState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipState")
            .field("open_nodes", &self.open.len())
            .field("has_incumbent", &self.incumbent.is_some())
            .field("stats", &self.stats)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MipOutcome {
    Optimal,
    Interrupted,
}

pub(crate) struct MipRun {
    pub outcome: MipOutcome,
    pub state: MipState,
}

pub(crate) fn status_of(outcome: MipOutcome, state: &MipState) -> Status {
    match outcome {
        MipOutcome::Optimal => Status::Optimal,
        MipOutcome::Interrupted => {
            if state.incumbent.is_some() {
                Status::Feasible
            } else {
                Status::Interrupted
            }
        }
    }
}

/// Build the search state for `problem` and run it under `options`.
pub(crate) fn run(problem: &Problem, options: SolveOptions) -> Result<MipRun, Error> {
    let deadline = options.time_limit.map(|d| Instant::now() + d);
    let solver = Solver::try_new(
        &problem.obj_coeffs,
        &problem.var_mins,
        &problem.var_maxs,
        &problem.constraints,
        &problem.var_domains,
        deadline,
    )?;
    let root_bounds = problem
        .var_mins
        .iter()
        .zip(&problem.var_maxs)
        .map(|(&lo, &hi)| (lo, hi))
        .collect();
    let mut state = MipState {
        solver,
        root_bounds,
        applied: Vec::new(),
        open: Vec::new(),
        incumbent: None,
        node_seq: 0,
        last_solved_id: None,
        root_solved: false,
        stats: Stats::default(),
        options,
        deadline,
        direction: problem.direction,
    };
    let outcome = resume_run_with_deadline(&mut state)?;
    Ok(MipRun { outcome, state })
}

/// Continue a paused search with a fresh time budget.
pub(crate) fn resume_run(
    state: &mut MipState,
    time_limit: Option<Duration>,
) -> Result<MipOutcome, Error> {
    state.deadline = time_limit.map(|d| Instant::now() + d);
    resume_run_with_deadline(state)
}

fn resume_run_with_deadline(state: &mut MipState) -> Result<MipOutcome, Error> {
    let started = Instant::now();
    let res = search_loop(state);
    state.stats.elapsed += started.elapsed();
    state.stats.lp_iterations = state.solver.lp_iterations;
    res
}

/// Prune threshold: a node whose lower bound is ≥ this cannot improve the incumbent.
fn cutoff(incumbent_obj: f64) -> f64 {
    incumbent_obj - f64::max(1e-9, 1e-9 * incumbent_obj.abs())
}

fn maybe_update_incumbent(state: &mut MipState) {
    let obj = state.solver.cur_obj_val;
    let better = match &state.incumbent {
        Some(inc) => obj < inc.objective,
        None => true,
    };
    if better {
        let values = (0..state.solver.num_vars)
            .map(|v| *state.solver.get_value(v))
            .collect();
        debug!("new incumbent, internal obj: {:.6}", obj);
        state.incumbent = Some(Incumbent {
            values,
            objective: obj,
        });
    }
}

/// Apply `node`'s bounds to the solver, diffing against what is currently applied.
/// Returns false (node pruned, solver untouched) if the node's bounds cross.
fn apply_node_bounds(state: &mut MipState, node: &Node) -> bool {
    let target = effective_bounds(&node.bound_changes);
    if target.iter().any(|&(_, lo, hi)| lo > hi) {
        return false;
    }
    // Reset vars that are currently changed but absent from the target.
    for &(v, _, _) in &state.applied {
        if target.binary_search_by_key(&v, |t| t.0).is_err() {
            let (rlo, rhi) = state.root_bounds[v];
            state
                .solver
                .set_var_bounds(v, rlo, rhi)
                .expect("root bounds cannot cross");
        }
    }
    // Apply the target bounds (validated above, cannot fail).
    for &(v, lo, hi) in &target {
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("validated bounds cannot cross");
    }
    state.applied = target;
    true
}

/// Branch on `var` at the solver's current (just solved) optimum: push the two
/// children carrying the parent's basis and objective bound.
fn branch(state: &mut MipState, parent: &Node, var: usize) {
    let z = state.solver.cur_obj_val;
    let val = *state.solver.get_value(var);
    let (lo, hi) = state.solver.get_var_bounds(var);
    let floor = val.floor();

    state.node_seq += 1;
    let id = state.node_seq;
    state.last_solved_id = Some(id);
    let basis = state.solver.snapshot_basis();

    let mut down_changes = parent.bound_changes.clone();
    down_changes.push((var, lo, floor));
    let mut up_changes = parent.bound_changes.clone();
    up_changes.push((var, floor + 1.0, hi));

    // Phase-1 order parity with the old solver: the up (ceil) child is dived first.
    state.open.push(Node {
        bound_changes: down_changes,
        basis: basis.clone(),
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
    });
    state.open.push(Node {
        bound_changes: up_changes,
        basis,
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
    });
}

fn search_loop(state: &mut MipState) -> Result<MipOutcome, Error> {
    let domains = state.solver.orig_var_domains.clone();
    let int_tol = state.options.int_tol;
    state.solver.deadline = state.deadline;

    // Root relaxation (also the resume path if the root was interrupted mid-LP:
    // initial_solve continues from the solver's feasibility flags).
    if !state.root_solved {
        match state.solver.initial_solve()? {
            StopReason::Limit => return Ok(MipOutcome::Interrupted),
            StopReason::Finished => {}
        }
        state.root_solved = true;
        if branching::is_integral(&state.solver, &domains, int_tol) {
            maybe_update_incumbent(state);
            return Ok(MipOutcome::Optimal);
        }
        let root = Node {
            bound_changes: Vec::new(),
            basis: state.solver.snapshot_basis(),
            lp_bound: state.solver.cur_obj_val,
            depth: 0,
            parent_id: 0,
        };
        let var = branching::choose_branch_var(&state.solver, &domains, int_tol)
            .expect("non-integral relaxation must have a fractional int var");
        branch(state, &root, var);
    }

    let mut nodes_this_run: u64 = 0;

    loop {
        // Limits are checked BETWEEN nodes; nothing half-solved is ever consulted.
        if check_deadline(&state.deadline) == StopReason::Limit {
            return Ok(MipOutcome::Interrupted);
        }
        if let Some(nl) = state.options.node_limit {
            if nodes_this_run >= nl {
                return Ok(MipOutcome::Interrupted);
            }
        }

        let node = match state.open.pop() {
            Some(n) => n,
            None => break,
        };

        // Prune with the stored parent bound before any LP work.
        if let Some(inc) = &state.incumbent {
            if node.lp_bound >= cutoff(inc.objective) {
                continue;
            }
        }

        // Load the node into the solver: bounds first (values derive from them),
        // then the basis if the solver isn't already at this node's parent optimum.
        if !apply_node_bounds(state, &node) {
            continue; // crossing bounds — pruned without touching the solver
        }
        let warm = state.last_solved_id == Some(node.parent_id);
        if !warm && state.solver.load_basis(&node.basis).is_err() {
            // Robustness valve: a numerically singular basis falls back to the
            // all-slack basis and the node is solved from scratch.
            debug!("basis load failed; falling back to slack basis");
            let slack = state.solver.slack_basis();
            state
                .solver
                .load_basis(&slack)
                .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
        }

        state.solver.deadline = state.deadline;
        match state.solver.reoptimize() {
            Err(Error::Infeasible) => {
                state.stats.nodes_solved += 1;
                nodes_this_run += 1;
                state.last_solved_id = None;
                continue;
            }
            Err(Error::Unbounded) => {
                return Err(Error::InternalError(
                    "bounded B&B node reported unbounded".to_string(),
                ))
            }
            Err(e) => return Err(e),
            Ok(StopReason::Limit) => {
                // Requeue UNSOLVED: the node's plain data is intact, the solver's
                // half-pivoted state will be discarded by the next basis load.
                state.open.push(node);
                state.last_solved_id = None;
                return Ok(MipOutcome::Interrupted);
            }
            Ok(StopReason::Finished) => {}
        }
        state.stats.nodes_solved += 1;
        nodes_this_run += 1;

        let z = state.solver.cur_obj_val;
        if let Some(inc) = &state.incumbent {
            if z >= cutoff(inc.objective) {
                state.last_solved_id = None;
                continue;
            }
        }

        if branching::is_integral(&state.solver, &domains, int_tol) {
            maybe_update_incumbent(state);
            state.last_solved_id = None;
            continue;
        }

        let var = branching::choose_branch_var(&state.solver, &domains, int_tol)
            .expect("non-integral node must have a fractional int var");
        branch(state, &node, var);
    }

    if state.incumbent.is_some() {
        Ok(MipOutcome::Optimal)
    } else {
        Err(Error::Infeasible)
    }
}
```

Notes for the implementer:
- `debug!` comes from the `log` crate, already imported crate-wide via `#[macro_use]` in `lib.rs`.
- `effective_bounds` returns a var-sorted vec, so `binary_search_by_key` in `apply_node_bounds` is valid.
- `Error` needs `PartialEq` for the tests' `unwrap_err()` comparisons — it already derives it.
- The two `expect("non-integral … fractional int var")` calls are safe: `is_integral` returned false with the same `int_tol`, so `choose_branch_var`'s `frac > int_tol` filter matches at least one var.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release mip::`
Expected: `test result: ok. 9 passed` (3 from Task 3 + 6 new).

- [ ] **Step 5: Full check + fmt**

Run: `cargo test --release && cargo fmt`
Expected: all pass; old B&B still active for the public API (unchanged until Task 5).

---

### Task 5: The switch — rewire `lib.rs`, delete the old B&B, migrate in-crate tests

This is the one atomic task: the public API flips from `StopReason` to `Status` and MILP
solving flips to the new driver. Everything in-repo must compile and pass at the end.

**Files:**
- Modify: `src/lib.rs` (Solution rebuild, `solve_with`, re-exports, `StopReason` → `pub(crate)`)
- Modify: `src/solver.rs` (delete old B&B: `solve_integer`, `Step`, `BranchKind`, `BranchAndBoundState`, `bb_state` field + its init in `try_new`, `choose_branch_var`, `get_branch_min_max`, `new_steps`, `is_solution_better`)
- Create: `src/tests/mip_api.rs`; Modify: `src/tests/mod.rs` (add `mod mip_api;`)
- Modify: `src/tests/resume.rs`, `src/tests/general.rs`, `src/tests/broken_tests.rs`, `src/tests/aoc.rs`, `src/tests/aoc2.rs`, `src/tests/tsp/*` — whatever the compiler flags, per the **API migration mapping** table at the top
- Modify: `examples/solve_mps.rs`, `examples/tsp.rs`, `src/problems_solvers/tsp.rs` if flagged

**Interfaces:**
- Consumes: `mip::{run, resume_run, status_of, MipState, MipOutcome, SolveOptions, Status, Stats}` (Tasks 3–4).
- Produces (public API v0.5, used by Tasks 6–13):
  - `pub use mip::{SolveOptions, Stats, Status};` from the crate root
  - `Problem::solve(&self) -> Result<Solution, Error>` (unchanged signature)
  - `Problem::solve_with(&self, options: SolveOptions) -> Result<Solution, Error>` —
    deliberate deviation from the spec's `&SolveOptions` sketch: options are taken by value
    because they are stored inside the search state (and `warm_start` is consumed)
  - `Solution::{status, objective, var_value, var_value_raw, gap, stats, iter, resume, add_constraint, fix_var, unfix_var, add_gomory_cut}` with signatures exactly as written in Step 2

- [ ] **Step 1: Delete the old B&B from `src/solver.rs`**

Remove: `struct BranchAndBoundState`, `enum BranchKind`, `struct Step`, `fn solve_integer`, `fn choose_branch_var`, `fn get_branch_min_max`, `fn new_steps`, `fn is_solution_better`, the `pub(crate) bb_state` field on `Solver` and its `bb_state: None` init in `try_new`, and the now-unused `use crate::{... Solution, OptimizationDirection, Variable ...}` imports (keep the ones still used). Keep everything else (including `add_constraint`, `fix_var`, `unfix_var`, `add_gomory_cut` — the LP incremental path). Also remove the commented-out `orig_int_vars` leftovers while there.

Run: `cargo build 2>&1 | head -40` — expect errors only in `lib.rs` (`solve`, `resume` referencing deleted items). That's the work of the next step.

- [ ] **Step 2: Rebuild the Solution/solve layer in `src/lib.rs`**

Replace the `StopReason` enum, `Problem::solve`, the whole `Solution` struct + impl, and `SolutionIter` with:

```rust
pub use mip::{SolveOptions, Stats, Status};

/// Internal signal for "a limit fired mid-simplex" (public API uses [`Status`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopReason {
    Limit,
    Finished,
}

impl Problem {
    /// Solve with default options (respecting [`Problem::set_time_limit`] if set).
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if no feasible (integer) point exists,
    /// [`Error::Unbounded`] if the objective is unbounded.
    pub fn solve(&self) -> Result<Solution, Error> {
        let mut options = SolveOptions::default();
        options.time_limit = self.time_limit;
        self.solve_with(options)
    }

    /// Solve with explicit [`SolveOptions`].
    ///
    /// Hitting a limit is NOT an error: the returned [`Solution`] has status
    /// [`Status::Feasible`] (an incumbent exists) or [`Status::Interrupted`]
    /// (none yet) and can be continued with [`Solution::resume`].
    ///
    /// # Errors
    ///
    /// See [`Problem::solve`].
    pub fn solve_with(&self, options: SolveOptions) -> Result<Solution, Error> {
        let num_vars = self.obj_coeffs.len();
        if self.has_integer_vars() {
            let run = mip::run(self, options)?;
            Ok(Solution {
                direction: self.direction,
                num_vars,
                status: mip::status_of(run.outcome, &run.state),
                kind: SolutionKind::Mip(Box::new(run.state)),
            })
        } else {
            let deadline = options.time_limit.map(|d| Instant::now() + d);
            let mut solver = Solver::try_new(
                &self.obj_coeffs,
                &self.var_mins,
                &self.var_maxs,
                &self.constraints,
                &self.var_domains,
                deadline,
            )?;
            let status = match solver.initial_solve()? {
                StopReason::Finished => Status::Optimal,
                StopReason::Limit => Status::Interrupted,
            };
            Ok(Solution {
                direction: self.direction,
                num_vars,
                status,
                kind: SolutionKind::Lp(Box::new(solver)),
            })
        }
    }
}

/// A solution of a problem.
///
/// For problems with integer variables this is plain data (values + status) plus
/// an opaque resumable search state; for pure-LP problems it keeps the live
/// simplex basis so constraints can be added incrementally.
#[derive(Clone)]
pub struct Solution {
    direction: OptimizationDirection,
    num_vars: usize,
    status: Status,
    kind: SolutionKind,
}

#[derive(Clone)]
enum SolutionKind {
    Lp(Box<solver::Solver>),
    Mip(Box<mip::MipState>),
}

impl std::fmt::Debug for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Solution");
        d.field("direction", &self.direction)
            .field("num_vars", &self.num_vars)
            .field("status", &self.status);
        if self.status != Status::Interrupted {
            d.field("objective", &self.objective());
        }
        d.finish()
    }
}

impl Solution {
    /// The outcome of the solve: proven optimal, feasible-but-unproven, or interrupted.
    pub fn status(&self) -> Status {
        self.status
    }

    fn assert_has_values(&self) {
        assert!(
            self.status != Status::Interrupted,
            "the solver was interrupted before finding a usable solution; \
             call resume() to continue the search"
        );
    }

    /// Objective value of the best known solution.
    ///
    /// # Panics
    ///
    /// Panics if `status()` is [`Status::Interrupted`].
    pub fn objective(&self) -> f64 {
        self.assert_has_values();
        let internal = match &self.kind {
            SolutionKind::Lp(solver) => solver.cur_obj_val,
            SolutionKind::Mip(state) => state.incumbent.as_ref().unwrap().objective,
        };
        match self.direction {
            OptimizationDirection::Minimize => internal,
            OptimizationDirection::Maximize => -internal,
        }
    }

    /// Raw value of the variable (no integer rounding).
    ///
    /// # Panics
    ///
    /// Panics if `status()` is [`Status::Interrupted`].
    pub fn var_value_raw(&self, var: Variable) -> f64 {
        assert!(var.0 < self.num_vars);
        self.assert_has_values();
        match &self.kind {
            SolutionKind::Lp(solver) => *solver.get_value(var.0),
            SolutionKind::Mip(state) => state.incumbent.as_ref().unwrap().values[var.0],
        }
    }

    /// Value of the variable, rounded to an exact integer for integer/boolean vars.
    ///
    /// # Panics
    ///
    /// Panics if `status()` is [`Status::Interrupted`], or if an integer variable's
    /// value is further than 1e-5 from an integer (indicates a solver bug).
    pub fn var_value(&self, var: Variable) -> f64 {
        let val = self.var_value_raw(var);
        let domain = match &self.kind {
            SolutionKind::Lp(solver) => &solver.orig_var_domains[var.0],
            SolutionKind::Mip(state) => &state.solver.orig_var_domains[var.0],
        };
        if *domain == VarDomain::Integer || *domain == VarDomain::Boolean {
            let rounded = val.round();
            assert!(
                f64::abs(rounded - val) < 1e-5,
                "Variable was expected to be an integer, got {}",
                val
            );
            rounded
        } else {
            val
        }
    }

    /// Relative MIP gap of the best known solution (`None` until an incumbent and
    /// a bound exist; always `Some(0.0)` for optimal LP solutions).
    pub fn gap(&self) -> Option<f64> {
        match &self.kind {
            SolutionKind::Lp(_) => (self.status == Status::Optimal).then_some(0.0),
            SolutionKind::Mip(state) => state.stats.gap,
        }
    }

    /// Solve statistics (nodes, simplex iterations, elapsed time).
    pub fn stats(&self) -> Stats {
        match &self.kind {
            SolutionKind::Lp(solver) => {
                let mut s = Stats::default();
                s.lp_iterations = solver.lp_iterations;
                s
            }
            SolutionKind::Mip(state) => state.stats,
        }
    }

    /// Iterate over variable/value pairs (raw values, like [`Solution::var_value_raw`]).
    ///
    /// # Panics
    ///
    /// Panics if `status()` is [`Status::Interrupted`].
    pub fn iter(&self) -> SolutionIter<'_> {
        SolutionIter {
            solution: self,
            var_idx: 0,
        }
    }

    /// Continue an interrupted solve with a fresh time budget (`None` = unlimited).
    /// A no-op on solutions whose status is already [`Status::Optimal`].
    ///
    /// # Errors
    ///
    /// Same as [`Problem::solve`].
    pub fn resume(mut self, time_limit: Option<Duration>) -> Result<Self, Error> {
        if self.status == Status::Optimal {
            return Ok(self);
        }
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                solver.deadline = time_limit.map(|d| Instant::now() + d);
                self.status = match solver.initial_solve()? {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                };
            }
            SolutionKind::Mip(state) => {
                let outcome = mip::resume_run(state, time_limit)?;
                self.status = mip::status_of(outcome, state);
            }
        }
        Ok(self)
    }

    /// Add a constraint to the solved problem and re-solve.
    ///
    /// LP solutions re-solve incrementally from the live basis (dual simplex).
    /// MILP solutions are re-solved on the ORIGINAL problem plus the edit
    /// (implemented in phase 3; until then this returns an internal error).
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the edit makes the problem infeasible.
    pub fn add_constraint(
        mut self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<Self, Error> {
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                if self.status == Status::Interrupted {
                    return Err(Error::InternalError(
                        "cannot edit an interrupted solution; resume() it first".to_string(),
                    ));
                }
                let expr = expr.into();
                let sr = solver.add_constraint(
                    CsVec::new_from_unsorted(self.num_vars, expr.vars, expr.coeffs)
                        .map_err(|v| Error::InternalError(v.2.to_string()))?,
                    cmp_op,
                    rhs,
                )?;
                self.status = match sr {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                };
                Ok(self)
            }
            SolutionKind::Mip(_) => Err(Error::InternalError(
                "editing MILP solutions is implemented in phase 3 of the refactor".to_string(),
            )),
        }
    }

    /// Fix the variable to `val` and re-solve. See [`Solution::add_constraint`]
    /// for LP/MILP semantics.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the fix makes the problem infeasible.
    pub fn fix_var(mut self, var: Variable, val: f64) -> Result<Self, Error> {
        assert!(var.0 < self.num_vars);
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                if self.status == Status::Interrupted {
                    return Err(Error::InternalError(
                        "cannot edit an interrupted solution; resume() it first".to_string(),
                    ));
                }
                let sr = solver.fix_var(var.0, val)?;
                self.status = match sr {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                };
                Ok(self)
            }
            SolutionKind::Mip(_) => Err(Error::InternalError(
                "editing MILP solutions is implemented in phase 3 of the refactor".to_string(),
            )),
        }
    }

    /// Undo a previous [`Solution::fix_var`]. The boolean reports whether the
    /// variable was actually fixed.
    pub fn unfix_var(mut self, var: Variable) -> (Self, bool) {
        assert!(var.0 < self.num_vars);
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                let res = solver.unfix_var(var.0);
                (self, res)
            }
            // Interim until phase 3 (Task 12): report "was not fixed".
            SolutionKind::Mip(_) => (self, false),
        }
    }

    /// Add a Gomory cut for `var`. Only available on pure-LP solutions.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the cut makes the problem infeasible; an internal
    /// error on MILP solutions (the cut reads the live simplex tableau).
    pub fn add_gomory_cut(mut self, var: Variable) -> Result<Self, Error> {
        assert!(var.0 < self.num_vars);
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                let sr = solver.add_gomory_cut(var.0)?;
                self.status = match sr {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                };
                Ok(self)
            }
            SolutionKind::Mip(_) => Err(Error::InternalError(
                "Gomory cuts require a pure-LP solution".to_string(),
            )),
        }
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    fn index(&self, var: Variable) -> &Self::Output {
        assert!(var.0 < self.num_vars);
        self.assert_has_values();
        match &self.kind {
            SolutionKind::Lp(solver) => solver.get_value(var.0),
            SolutionKind::Mip(state) => &state.incumbent.as_ref().unwrap().values[var.0],
        }
    }
}

/// An iterator over the variable-value pairs of a [`Solution`].
#[derive(Debug, Clone)]
pub struct SolutionIter<'a> {
    solution: &'a Solution,
    var_idx: usize,
}

impl<'a> Iterator for SolutionIter<'a> {
    type Item = (Variable, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.var_idx < self.solution.num_vars {
            let var = Variable(self.var_idx);
            self.var_idx += 1;
            Some((var, self.solution.var_value_raw(var)))
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Solution {
    type Item = (Variable, f64);
    type IntoIter = SolutionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
```

Additional `lib.rs` edits in this step:
1. Change `mod mip;` — remove the `#[allow(dead_code)]` from Task 3.
2. Update `set_time_limit`'s doc comment: replace the `StopReason::Limit` sentence with
   "If the solver exceeds this duration, the returned solution's status will be
   [`Status::Feasible`] or [`Status::Interrupted`] and can be continued with
   [`Solution::resume`]."
3. Delete the old `resume` implementation's `bb_state` handling (gone with the rewrite above).
4. Keep `Problem`, `LinearExpr`, `Variable`, `ComparisonOp`, `Error`, `VarDomain`,
   `OptimizationDirection`, `MpsFile` exactly as they are.

- [ ] **Step 3: Fix everything the compiler flags**

Run: `cargo build --all-targets 2>&1 | head -60` repeatedly, fixing per the **API migration
mapping** table. Expected touch points: `src/tests/resume.rs` (`StopReason` → `Status`,
`stop_reason()` → `status()`), `src/tests/general.rs`, `src/tests/broken_tests.rs`,
`src/tests/tsp/*`, `src/problems_solvers/tsp.rs`, `examples/*.rs`, and the two doc examples in
`lib.rs` (`Problem` docs). Translate assertions 1:1 — do not weaken or delete any.

In `src/tests/resume.rs` specifically, `while *sol_limited.stop_reason() == StopReason::Limit`
becomes `while sol_limited.status() != Status::Optimal`.

- [ ] **Step 4: Write the new public-API tests**

Create `src/tests/mip_api.rs` and add `mod mip_api;` to `src/tests/mod.rs`:

```rust
#[cfg(test)]
mod tests_mip_api {
    use crate::*;
    use core::time::Duration;

    fn int_2var_problem() -> (Problem, Variable, Variable) {
        // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b int in [0,10] → a=1,b=2, obj 11.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let a = p.add_integer_var(3.0, (0, 10));
        let b = p.add_integer_var(4.0, (0, 10));
        p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
        p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
        (p, a, b)
    }

    #[test]
    fn milp_solve_reports_optimal_and_rounds_values() {
        let (p, a, b) = int_2var_problem();
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
        assert_eq!(sol.var_value(a), 1.0);
        assert_eq!(sol.var_value(b), 2.0);
        assert!(sol.stats().nodes_solved > 0);
        assert!(sol.stats().lp_iterations > 0);
    }

    #[test]
    fn milp_maximize_sign_is_correct() {
        // maximize 8x + 11y + 6z + 4w, 5x + 7y + 4z + 3w <= 14, binaries → 21.
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_binary_var(8.0);
        let y = p.add_binary_var(11.0);
        let z = p.add_binary_var(6.0);
        let w = p.add_binary_var(4.0);
        p.add_constraint(
            &[(x, 5.0), (y, 7.0), (z, 4.0), (w, 3.0)],
            ComparisonOp::Le,
            14.0,
        );
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 21.0).abs() < 1e-6);
        assert_eq!(sol.var_value(x), 0.0);
        assert_eq!(sol.var_value(y), 1.0);
    }

    #[test]
    fn zero_time_limit_is_interrupted_then_resume_finishes() {
        let (mut p, _, _) = int_2var_problem();
        p.set_time_limit(Duration::ZERO);
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Interrupted);
        assert!(sol.gap().is_none());
        let sol = sol.resume(None).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "interrupted before finding a usable solution")]
    fn interrupted_objective_panics_with_clear_message() {
        let (mut p, _, _) = int_2var_problem();
        p.set_time_limit(Duration::ZERO);
        let sol = p.solve().unwrap();
        let _ = sol.objective(); // must panic, not return garbage
    }

    #[test]
    fn node_limit_interrupts_deterministically_and_resumes() {
        let (p, _, _) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut sol = p.solve_with(options).unwrap();
        let mut resumes = 0;
        while sol.status() != Status::Optimal {
            resumes += 1;
            assert!(resumes < 10_000);
            sol = sol.resume(None).unwrap();
        }
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn milp_infeasible_is_an_error() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        assert_eq!(p.solve().unwrap_err(), Error::Infeasible);
    }

    #[test]
    fn lp_path_still_solves_and_edits_incrementally() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 4.0));
        let y = p.add_var(2.0, (0.0, 3.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 5.0);
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 8.0).abs() < 1e-6);
        // Live-basis incremental add on the LP path.
        let sol = sol.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1.0).unwrap();
        assert!((sol.objective() - 7.0).abs() < 1e-6);
        assert!((sol[x] - 1.0).abs() < 1e-6);
    }
}
```

- [ ] **Step 5: Run the full test suite**

Run: `cargo test --release`
Expected: everything passes, including the migrated `resume` tests and both doctests.
The `solve_integer_singular_var` / `solve_powers_integer` tests in `src/solver.rs` now
exercise the NEW driver through `Problem::solve` — they must pass unchanged.

- [ ] **Step 6: Run the correctness suite (default tier)**

Run: `cargo test --release --test suite`
Expected: green, same case count as before (~195). If `incr/*` LP cases fail, the LP edit
path regressed — fix before proceeding (it must be byte-for-byte the old behavior).

- [ ] **Step 7: fmt + examples**

Run: `cargo build --examples && cargo fmt`
Expected: clean build.

---

### Task 6: Migrate `tests/suite` and flip the interrupt known-failures

**Files:**
- Modify: `tests/suite/**/*.rs` (compiler-driven migration per the mapping table)
- Modify: `tests/suite/cases/incremental.rs`, `tests/suite/cases/miplib.rs` (or wherever the
  interrupt known-failures are registered — read first)

**Interfaces:**
- Consumes: the Task 5 public API.
- Produces: a green default tier; hard tier where the ONLY remaining failures are
  `netlib/brandy` (out of scope) and `incr/*-milp*` (until Task 12).

- [ ] **Step 1: Read the harness before touching it**

Read `tests/suite/main.rs`, `tests/suite/cases/mod.rs`, and `tests/suite/verify.rs` to learn:
(a) the `Case` registration shape, (b) how known-failures/hard-tier cases are marked,
(c) what `verify.rs` asserts about solutions. Do not guess — the flip in Step 4 must use the
harness's own mechanism.

- [ ] **Step 2: Migrate compile errors**

Run: `cargo build --release --test suite 2>&1 | head -60` repeatedly; fix per the
**API migration mapping** table. Assertions translate 1:1; `stop_reason()==Limit` checks in
resume-style cases become `status() != Status::Optimal` (or the tighter
`matches!(..., Status::Feasible | Status::Interrupted)` where the case distinguishes them).

- [ ] **Step 3: Default tier green**

Run: `cargo test --release --test suite`
Expected: all default-tier cases pass.

- [ ] **Step 4: Flip the interrupt known-failures**

The B&B interrupt panic (C1) is structurally fixed, so these hard-tier cases must now pass
and be promoted out of known-failure status, using the mechanism learned in Step 1:
- `milp/time-limit-interrupt` — a mid-search deadline now returns a clean
  `Feasible`/`Interrupted` status instead of panicking. Update the case to assert exactly
  that (no panic; if `Feasible`, the incumbent must verify against the shadow model;
  resume(None) afterwards must reach the known optimum).
- The PANIC-at-budget MIPLIB cases (`mod008`, `vpm1`, `gt2`, `bell3a` at their budgets) —
  same semantics: clean status instead of panic. They may still be slow; keep their budgets.

Leave `netlib/brandy` (LP phase-1 bug, out of scope) and `incr/*-milp*` (fixed in Task 12)
as known failures.

- [ ] **Step 5: Hard tier inventory**

Run: `cargo test --release --test suite -- --hard`
Expected: the flipped cases pass; remaining failures are exactly `netlib/brandy` and the
`incr/*-milp*` family. Record the actual list in the task notes for Task 12's acceptance.

- [ ] **Step 6: fmt**

Run: `cargo fmt`

---

### Task 7 (phase 2): Global dual bound, `gap()`, `mip_gap` stopping

**Files:**
- Modify: `src/mip/mod.rs`

**Interfaces:**
- Consumes: Task 4 driver.
- Produces: `fn global_bound_internal(state: &MipState) -> Option<f64>`; `Stats.best_bound`
  and `Stats.gap` filled on every run exit (user-space sign); `Solution::gap()` now returns
  `Some(...)` for MILP solutions with an incumbent; `mip_gap > 0` stops early with
  `MipOutcome::Optimal`.

- [ ] **Step 1: Write the failing tests** (append to `mip::tests`)

```rust
#[test]
fn optimal_solve_reports_zero_gap_and_matching_bound() {
    let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
    assert_eq!(r.outcome, MipOutcome::Optimal);
    assert_eq!(r.state.stats.gap, Some(0.0));
    // User space == internal for Minimize.
    assert!((r.state.stats.best_bound.unwrap() - 11.0).abs() < 1e-6);
}

#[test]
fn maximize_bound_is_in_user_space() {
    let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
    assert_eq!(r.outcome, MipOutcome::Optimal);
    // Internally -21; user-facing bound must be +21.
    assert!((r.state.stats.best_bound.unwrap() - 21.0).abs() < 1e-6);
}

#[test]
fn mip_gap_stops_early_with_consistent_bound() {
    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;
    let r = run(&binary_knapsack(), options).unwrap();
    assert_eq!(r.outcome, MipOutcome::Optimal); // optimal within the configured gap
    let inc = -incumbent_obj(&r.state); // user space (Maximize)
    let bound = r.state.stats.best_bound.unwrap();
    // Incumbent within 50% of the proven bound, and never better than it.
    assert!(inc <= bound + 1e-9);
    assert!((bound - inc) / bound.abs().max(1e-10) <= 0.5 + 1e-9);
}

#[test]
fn feasible_interrupt_reports_gap() {
    let mut options = SolveOptions::default();
    options.node_limit = Some(2);
    let mut r = run(&binary_knapsack(), options).unwrap();
    // Resume with node budget until an incumbent exists but the search isn't done.
    let mut guard = 0;
    while r.outcome == MipOutcome::Interrupted && r.state.incumbent.is_none() {
        guard += 1;
        assert!(guard < 10_000);
        r.outcome = resume_run(&mut r.state, None).unwrap();
    }
    if r.outcome == MipOutcome::Interrupted {
        // Feasible-but-unproven: a gap must be reported.
        assert!(r.state.stats.gap.unwrap() >= 0.0);
        assert!(r.state.stats.best_bound.is_some());
    }
}
```

Note: `resume_run(&mut r.state, None)` keeps `options.node_limit = Some(2)` — the budget is
per call, so the loop makes progress 2 nodes at a time.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release mip::`
Expected: the four new tests fail (`stats.gap` is `None`).

- [ ] **Step 3: Implement**

Add to `src/mip/mod.rs`:

```rust
/// Best proven lower bound (internal space) on the optimum: the min over open-node
/// bounds and the incumbent. `None` while nothing is known (no nodes, no incumbent).
/// Only valid BETWEEN nodes (a popped node's subtree is otherwise unaccounted).
fn global_bound_internal(state: &MipState) -> Option<f64> {
    let open_min = state
        .open
        .iter()
        .map(|n| n.lp_bound)
        .fold(f64::INFINITY, f64::min);
    match (&state.incumbent, state.open.is_empty()) {
        (Some(inc), true) => Some(inc.objective), // proof complete
        (Some(inc), false) => Some(open_min.min(inc.objective)),
        (None, false) => Some(open_min),
        (None, true) => None,
    }
}

/// Relative gap between incumbent and bound, internal space (0.0 when they meet).
fn relative_gap(incumbent_obj: f64, bound: f64) -> f64 {
    (incumbent_obj - bound).max(0.0) / incumbent_obj.abs().max(1e-10)
}

fn to_user_space(direction: OptimizationDirection, internal: f64) -> f64 {
    match direction {
        OptimizationDirection::Minimize => internal,
        OptimizationDirection::Maximize => -internal,
    }
}

fn fill_bound_stats(state: &mut MipState) {
    let bound = global_bound_internal(state);
    state.stats.best_bound = bound.map(|b| to_user_space(state.direction, b));
    state.stats.gap = match (&state.incumbent, bound) {
        (Some(inc), Some(b)) => Some(relative_gap(inc.objective, b)),
        _ => None,
    };
}
```

Wire into the driver:
1. In `resume_run_with_deadline`, after `let res = search_loop(state);` add
   `fill_bound_stats(state);` (before returning).
2. In `search_loop`, at the TOP of the `loop` (before the deadline check), add the
   gap-based stop:

```rust
if state.options.mip_gap > 0.0 {
    if let (Some(inc), Some(bound)) = (&state.incumbent, global_bound_internal(state)) {
        if relative_gap(inc.objective, bound) <= state.options.mip_gap {
            return Ok(MipOutcome::Optimal);
        }
    }
}
```

3. When the search ends with `open` empty and an incumbent, `global_bound_internal`
   returns the incumbent → gap `Some(0.0)` automatically. No extra code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release mip:: && cargo test --release mip_api`
Expected: all pass, including the Task 5 API test asserting `gap().is_none()` on
`Interrupted` (no incumbent → `stats.gap` stays `None`).

- [ ] **Step 5: Full check + fmt**

Run: `cargo test --release && cargo test --release --test suite && cargo fmt`
Expected: green. `mip_gap` default is 0.0, so suite exactness is unaffected.

---

### Task 8 (phase 2): Node selection — DFS plunging with best-bound jumps

**Files:**
- Modify: `src/mip/mod.rs`

**Interfaces:**
- Consumes: Tasks 4 & 7.
- Produces: `MipState.diving: bool` field; `fn pop_node(state: &mut MipState) -> Option<Node>`
  replacing the bare `state.open.pop()` in `search_loop`.

- [ ] **Step 1: Write the failing test** (append to `mip::tests`)

```rust
#[test]
fn plunge_and_jump_selection_preserves_optima() {
    // Same optima as plain DFS on all driver test problems, plus interrupt/resume.
    let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
    assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);

    let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
    assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);

    let mut options = SolveOptions::default();
    options.node_limit = Some(1);
    let mut r = run(&binary_knapsack(), options).unwrap();
    let mut guard = 0;
    while r.outcome == MipOutcome::Interrupted {
        guard += 1;
        assert!(guard < 10_000);
        r.outcome = resume_run(&mut r.state, None).unwrap();
    }
    assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
    // After a best-bound jump the pop is NOT the last-pushed node at least once
    // on this instance; correctness above is the real assertion.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release plunge_and_jump`
Expected: compile error (`diving` / `pop_node` not defined — the test itself passes trivially
once they exist; it exists to pin behavior while the pop policy changes).

Add `diving: bool` to `MipState` first WITHOUT using it, initialized `false` in `run()`;
then the test compiles and passes against plain DFS — run it once to confirm, then change
the policy in Step 3 and run again.

- [ ] **Step 3: Implement**

```rust
/// Pop policy: keep diving (DFS) while the last processed node produced children;
/// when a dive dies out, jump to the open node with the best (lowest) bound.
fn pop_node(state: &mut MipState) -> Option<Node> {
    if state.open.is_empty() {
        return None;
    }
    if state.diving {
        state.open.pop()
    } else {
        let mut best = 0;
        for (i, n) in state.open.iter().enumerate() {
            if n.lp_bound < state.open[best].lp_bound {
                best = i;
            }
        }
        Some(state.open.swap_remove(best))
    }
}
```

In `search_loop`:
- Replace `let node = match state.open.pop() { ... }` with `let node = match pop_node(state) { ... }`.
- Set `state.diving = false;` on every `continue` path where no children were pushed
  (bound-prune, crossing-bounds prune, infeasible node, fresh-z prune, integral node).
- Set `state.diving = true;` at the end of `branch(...)` (children pushed).
- In `run()`, initialize `diving: false`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release mip::`
Expected: all pass (including all earlier driver tests — same optima, different visit order).

- [ ] **Step 5: Full check + fmt**

Run: `cargo test --release && cargo test --release --test suite && cargo fmt`
Expected: green.

---

### Task 9 (phase 2): Pseudocost branching

**Files:**
- Modify: `src/mip/branching.rs` (PseudoCosts), `src/mip/node.rs` (branch metadata),
  `src/mip/mod.rs` (record + selection call sites)

**Interfaces:**
- Consumes: Tasks 4, 7, 8.
- Produces:
  - `Node` gains `pub branch_var: usize, pub branch_up: bool, pub branch_frac: f64`
  - `branching::PseudoCosts` with `fn new(obj_coeffs: &[f64], num_vars: usize) -> Self`,
    `fn record(&mut self, var: usize, up: bool, degradation_per_unit: f64)`,
    `fn estimate(&self, var: usize, up: bool) -> f64`
  - `branching::choose_branch_var(solver, domains, int_tol, pc: &PseudoCosts) -> Option<usize>`
    (signature CHANGES: gains the `pc` parameter; most-fractional body replaced)
  - `MipState.pseudocosts: PseudoCosts` field

- [ ] **Step 1: Write the failing tests**

Append to `branching::tests`:

```rust
#[test]
fn pseudocosts_average_and_fall_back_to_init() {
    // 2 vars, obj coeffs 3 and 0 → init estimates 3+1e-6 and 1e-6... clamped by new().
    let mut pc = PseudoCosts::new(&[3.0, 0.0], 2);
    assert!((pc.estimate(0, true) - 3.0).abs() < 1e-3);
    pc.record(0, true, 10.0);
    pc.record(0, true, 20.0);
    assert!((pc.estimate(0, true) - 15.0).abs() < 1e-9); // average of observations
    assert!((pc.estimate(0, false) - 3.0).abs() < 1e-3); // down side still init
}

#[test]
fn pseudocost_selection_prefers_high_degradation_var() {
    // Two fractional int vars; var 1 has recorded huge degradations → must be chosen.
    let mut solver = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[1.5, 10.0],
        &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 3.0)],
        &[VarDomain::Integer, VarDomain::Integer],
        None,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    // LP optimum: x=1.5, y=0.75 → both fractional.
    let mut pc = PseudoCosts::new(&[-1.0, -1.0], 2);
    pc.record(1, true, 100.0);
    pc.record(1, false, 100.0);
    let domains = solver.orig_var_domains.clone();
    assert_eq!(choose_branch_var(&solver, &domains, 1e-6, &pc), Some(1));
}
```

Update the two existing Task 3 branching tests to pass a fresh `PseudoCosts::new(...)`
(with obj coeffs matching each test's solver) as the 4th argument — with no recorded data
and equal init estimates, the product score `est_down·f_down · est_up·f_up` is maximized by
the most-fractional var, so `most_fractional_var_is_chosen` keeps its `Some(1)` expectation
(var 1: 0.65·0.35 ≈ 0.23 beats var 0: 0.9·0.1 = 0.09) and
`integral_solution_yields_no_branch_var` keeps `None`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release branching`
Expected: compile error (`PseudoCosts` not defined).

- [ ] **Step 3: Implement**

In `src/mip/branching.rs`:

```rust
/// Per-variable average objective degradation per unit of fractionality, per
/// branching direction. Falls back to |obj coeff| before any observation.
#[derive(Clone, Debug)]
pub(crate) struct PseudoCosts {
    up_sum: Vec<f64>,
    up_n: Vec<u32>,
    down_sum: Vec<f64>,
    down_n: Vec<u32>,
    init: Vec<f64>,
}

impl PseudoCosts {
    pub(crate) fn new(obj_coeffs: &[f64], num_vars: usize) -> Self {
        let init = (0..num_vars)
            .map(|v| obj_coeffs.get(v).copied().unwrap_or(0.0).abs() + 1e-6)
            .collect();
        Self {
            up_sum: vec![0.0; num_vars],
            up_n: vec![0; num_vars],
            down_sum: vec![0.0; num_vars],
            down_n: vec![0; num_vars],
            init,
        }
    }

    pub(crate) fn record(&mut self, var: usize, up: bool, degradation_per_unit: f64) {
        if up {
            self.up_sum[var] += degradation_per_unit;
            self.up_n[var] += 1;
        } else {
            self.down_sum[var] += degradation_per_unit;
            self.down_n[var] += 1;
        }
    }

    pub(crate) fn estimate(&self, var: usize, up: bool) -> f64 {
        let (sum, n) = if up {
            (self.up_sum[var], self.up_n[var])
        } else {
            (self.down_sum[var], self.down_n[var])
        };
        if n > 0 {
            sum / n as f64
        } else {
            self.init[var]
        }
    }
}
```

Replace `choose_branch_var` with the pseudocost product rule:

```rust
/// Pseudocost product rule: pick the fractional int var maximizing
/// max(est_down·f_down, ε) · max(est_up·f_up, ε).
pub(crate) fn choose_branch_var(
    solver: &Solver,
    domains: &[VarDomain],
    int_tol: f64,
    pc: &PseudoCosts,
) -> Option<usize> {
    const SCORE_EPS: f64 = 1e-6;
    let mut best: Option<(usize, f64)> = None;
    for (v, d) in domains.iter().enumerate() {
        if !is_int_domain(d) {
            continue;
        }
        let val = *solver.get_value(v);
        if fractionality(val) <= int_tol {
            continue;
        }
        let f_down = val - val.floor();
        let f_up = 1.0 - f_down;
        let score = (pc.estimate(v, false) * f_down).max(SCORE_EPS)
            * (pc.estimate(v, true) * f_up).max(SCORE_EPS);
        if best.map_or(true, |(_, s)| score > s) {
            best = Some((v, score));
        }
    }
    best.map(|(v, _)| v)
}
```

In `src/mip/node.rs`, add to `Node`:

```rust
    /// Which var this node's creating branch changed, in which direction, and the
    /// parent fractionality — feeds pseudocost updates when this node's LP solves.
    pub branch_var: usize,
    pub branch_up: bool,
    pub branch_frac: f64,
```

In `src/mip/mod.rs`:
1. Add `pub pseudocosts: branching::PseudoCosts` to `MipState`; initialize in `run()` with
   `branching::PseudoCosts::new(&problem.obj_coeffs, problem.obj_coeffs.len())`.
2. In `branch(...)`, compute `let f_down = val - floor;` and set on the children:
   down child `branch_var: var, branch_up: false, branch_frac: f_down`;
   up child `branch_var: var, branch_up: true, branch_frac: 1.0 - f_down`.
   The synthetic root `Node` literal in `search_loop` gets `branch_var: 0, branch_up: false,
   branch_frac: 1.0` (never recorded: see next point — record only for `depth > 0`… the root
   is never in `open`, so no guard is actually needed; note it and move on).
3. In `search_loop`, right after `Ok(StopReason::Finished) => {}` and the
   `stats.nodes_solved` increment, record the observation:

```rust
let z = state.solver.cur_obj_val;
state.pseudocosts.record(
    node.branch_var,
    node.branch_up,
    (z - node.lp_bound).max(0.0) / node.branch_frac.max(1e-6),
);
```

   (This replaces the previous bare `let z = ...` line.)
4. Update both `choose_branch_var` call sites to pass `&state.pseudocosts`.
   Borrow note: `choose_branch_var(&state.solver, &domains, int_tol, &state.pseudocosts)`
   takes two shared borrows of `state` fields — fine. The `branch(state, ...)` call that
   follows takes `&mut state` after the returned `Option<usize>` ends the borrows.
5. Diving order in `branch(...)`: build both `Node` literals first (bind them as
   `down_node` and `up_node`), then push the child with the LARGER estimated degradation
   first, so the cheaper (more promising) direction is popped/dived first:

```rust
let est_down = state.pseudocosts.estimate(var, false) * f_down;
let est_up = state.pseudocosts.estimate(var, true) * (1.0 - f_down);
if est_down > est_up {
    // up is cheaper → push it last so it is dived first
    state.open.push(down_node);
    state.open.push(up_node);
} else {
    state.open.push(up_node);
    state.open.push(down_node);
}
```

   (This replaces Task 4's fixed "up child last" parity order.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release mip:: && cargo test --release branching`
Expected: all driver tests keep their optima (selection changes visit order, never
correctness); new pseudocost tests pass.

- [ ] **Step 5: Full check + suite + fmt**

Run: `cargo test --release && cargo test --release --test suite && cargo fmt`
Expected: green. Optionally note the suite wall-clock before/after this task — pseudocosts
should not be slower overall; if a specific case regresses badly, record it (do not tune yet).

---

### Task 10 (phase 3): Warm start from a known solution

**Files:**
- Modify: `src/mip/mod.rs` (hint evaluation after the root solve)
- Modify: `src/tests/mip_api.rs` (public-API tests)

**Interfaces:**
- Consumes: Tasks 1–9. `SolveOptions.warm_start` already exists (Task 3).
- Produces: `fn try_warm_start(state: &mut MipState, hints: &[(Variable, f64)]) -> Result<(), Error>`
  called from `search_loop` right after the root LP solves and before root branching.

- [ ] **Step 1: Write the failing tests** (append to `src/tests/mip_api.rs`)

```rust
#[test]
fn warm_start_with_optimal_hint_is_accepted() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let sol = p.solve_with(options).unwrap();
    assert_eq!(sol.status(), Status::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_with_infeasible_hint_is_ignored() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    // a=0, b=0 violates both constraints — hint must be dropped, solve still exact.
    options.warm_start = Some(vec![(a, 0.0), (b, 0.0)]);
    let sol = p.solve_with(options).unwrap();
    assert_eq!(sol.status(), Status::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_out_of_bounds_hint_is_ignored() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 99.0), (b, 2.0)]); // 99 > upper bound 10
    let sol = p.solve_with(options).unwrap();
    assert_eq!(sol.status(), Status::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_partial_hint_completes_via_lp() {
    // minimize 3a + 4b, a + 2b >= 5, 3a + b >= 4; hint only a=1 → LP completes b,
    // and if the completion is integral (b=2) it seeds the incumbent.
    let (p, a, _) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0)]);
    let sol = p.solve_with(options).unwrap();
    assert_eq!(sol.status(), Status::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_prunes_immediately_when_hint_is_optimal() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let with_hint = p.solve_with(options).unwrap();
    let without = p.solve().unwrap();
    // Correctness identical; the hinted run must not explore MORE nodes.
    assert!(with_hint.stats().nodes_solved <= without.stats().nodes_solved);
    assert_eq!(with_hint.objective(), without.objective());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release warm_start`
Expected: all 5 fail or the hint is silently unused (tests pass trivially except
`prunes_immediately`? No — they fail because `warm_start` is currently never read; the
first four DO pass trivially since the solve is exact anyway. That is the point of
`warm_start_prunes_immediately_when_hint_is_optimal`, which fails while the hint is ignored
only if the cold run explores ≥1 node more than the hinted run — on this instance the cold
run branches at least once while a seeded run prunes at the root bound. Run and confirm
exactly this one fails; the others are regression guards.)

- [ ] **Step 3: Implement** (in `src/mip/mod.rs`)

```rust
/// Evaluate a warm-start hint: fix hinted vars, LP-complete the rest, and if the
/// completion is feasible and integral adopt it as the initial incumbent.
/// Advisory by design — every failure path just drops the hint. Always restores
/// the solver to the root optimum before returning.
fn try_warm_start(state: &mut MipState, hints: &[(crate::Variable, f64)]) -> Result<(), Error> {
    let domains = state.solver.orig_var_domains.clone();
    let root_basis = state.solver.snapshot_basis();
    let mut applied: Vec<usize> = Vec::new();
    let mut ok = true;

    for &(var, val) in hints {
        let v = var.idx();
        if v >= state.solver.num_vars {
            ok = false;
            break;
        }
        let val = if matches!(domains.get(v), Some(VarDomain::Integer | VarDomain::Boolean)) {
            val.round()
        } else {
            val
        };
        let (lo, hi) = state.root_bounds[v];
        if val < lo - 1e-9 || val > hi + 1e-9 {
            ok = false;
            break;
        }
        state
            .solver
            .set_var_bounds(v, val, val)
            .expect("hint value validated against root bounds");
        applied.push(v);
    }

    if ok {
        match state.solver.reoptimize() {
            Ok(StopReason::Finished) => {
                if branching::is_integral(&state.solver, &domains, state.options.int_tol) {
                    maybe_update_incumbent(state);
                } else {
                    debug!("warm-start hint LP-completed fractionally; ignored");
                }
            }
            Ok(StopReason::Limit) | Err(Error::Infeasible) => {
                debug!("warm-start hint infeasible or out of time; ignored");
            }
            Err(e) => return Err(e),
        }
    } else {
        debug!("warm-start hint invalid (unknown var or out of bounds); ignored");
    }

    // Restore the root state exactly: bounds back, then the optimal root basis
    // (load_basis recomputes everything, discarding the hint solve).
    for v in applied {
        let (lo, hi) = state.root_bounds[v];
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("root bounds cannot cross");
    }
    if state.solver.load_basis(&root_basis).is_err() {
        let slack = state.solver.slack_basis();
        state
            .solver
            .load_basis(&slack)
            .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
        if state.solver.reoptimize()? == StopReason::Limit {
            return Ok(()); // deadline will be caught by the main loop
        }
    }
    Ok(())
}
```

Wire in `search_loop`, inside the `if !state.root_solved { ... }` block, immediately after
`state.root_solved = true;` and BEFORE the `is_integral` root check:

```rust
if let Some(hints) = state.options.warm_start.take() {
    try_warm_start(state, &hints)?;
}
```

(`take()` ensures the hint is evaluated once, not on every resume.)

Borrow/type notes: `use crate::VarDomain;` is needed in `mod.rs` if not already imported;
`Variable::idx()` is public. After `try_warm_start` the solver is back at the root optimum,
so the existing root `is_integral` check and branching read correct values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --release warm_start`
Expected: `5 passed`.

- [ ] **Step 5: Full check + suite + fmt**

Run: `cargo test --release && cargo test --release --test suite && cargo fmt`
Expected: green.

---

### Task 11 (phase 3): MILP-correct post-solve edits (fixes C2)

**Files:**
- Modify: `src/mip/mod.rs` (`MipState` gains `base` + `fixed`; helpers)
- Modify: `src/lib.rs` (`Solution::{add_constraint, fix_var, unfix_var}` MILP arms)
- Modify: `src/tests/mip_api.rs` (tests)

**Interfaces:**
- Consumes: everything prior, especially `try_warm_start` (Task 10) via `SolveOptions.warm_start`.
- Produces:
  - `MipState.base: Problem` (clean copy of the user's problem incl. later edits) and
    `MipState.fixed: std::collections::BTreeMap<usize, f64>` (user fixes overlay)
  - `mip::effective_problem(base: &Problem, fixed: &BTreeMap<usize, f64>) -> Problem`
  - `mip::incumbent_feasible(base: &Problem, fixed: &BTreeMap<usize, f64>, values: &[f64]) -> bool`
  - `mip::reedit_and_resolve(state: Box<MipState>) -> Result<MipRun, Error>` — drops the tree,
    revalidates the incumbent as a warm-start hint, re-runs the search on base+fixes
  - Working `Solution::{add_constraint, fix_var, unfix_var}` on MILP solutions (including
    paused ones: `Feasible`/`Interrupted` solutions may be edited; the open tree is dropped)

- [ ] **Step 1: Write the failing tests** (append to `src/tests/mip_api.rs`)

```rust
#[test]
fn milp_add_constraint_resolves_on_base_problem() {
    let (p, a, b) = int_2var_problem();
    let sol = p.solve().unwrap();
    assert!((sol.objective() - 11.0).abs() < 1e-6); // a=1, b=2

    // Cut off the incumbent: a + b >= 4. From-scratch optimum of the edited
    // problem: candidates (a=0,b=4): cons1 8>=5 ok, cons2 4>=4 ok, obj 16;
    // (a=1,b=3): 7>=5, 6>=4, sum 4 ok, obj 13; (a=2,b=2): 6>=5, 8>=4, obj 14.
    // → optimum 13 at (1,3).
    let sol = sol
        .add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
        .unwrap();
    assert_eq!(sol.status(), Status::Optimal);
    assert!((sol.objective() - 13.0).abs() < 1e-6);
    assert_eq!(sol.var_value(a), 1.0);
    assert_eq!(sol.var_value(b), 3.0);

    // Must equal a from-scratch solve of the edited problem.
    let (mut p2, a2, b2) = (int_2var_problem().0, Variable(0), Variable(1));
    p2.add_constraint(&[(a2, 1.0), (b2, 1.0)], ComparisonOp::Ge, 4.0);
    let fresh = p2.solve().unwrap();
    assert!((fresh.objective() - sol.objective()).abs() < 1e-6);
}

#[test]
fn milp_fix_and_unfix_var_roundtrip() {
    let (p, a, b) = int_2var_problem();
    let sol = p.solve().unwrap();

    // Fix a=3: then b >= 1 (cons1: 3+2b>=5) → obj 9+4=13 at (3,1).
    let sol = sol.fix_var(a, 3.0).unwrap();
    assert_eq!(sol.status(), Status::Optimal);
    assert!((sol.objective() - 13.0).abs() < 1e-6);
    assert_eq!(sol.var_value(a), 3.0);
    assert_eq!(sol.var_value(b), 1.0);

    // Unfix restores the original optimum and reports it was fixed.
    let (sol, was_fixed) = sol.unfix_var(a);
    assert!(was_fixed);
    assert!((sol.objective() - 11.0).abs() < 1e-6);

    // Unfixing a never-fixed var is a no-op with `false`.
    let (sol, was_fixed) = sol.unfix_var(b);
    assert!(!was_fixed);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn milp_fix_var_outside_bounds_is_infeasible_error() {
    let (p, a, _) = int_2var_problem();
    let sol = p.solve().unwrap();
    assert!(matches!(sol.fix_var(a, 99.0), Err(Error::Infeasible)));
}

#[test]
fn milp_edit_after_pause_completes_correctly() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.node_limit = Some(1); // pause almost immediately
    let sol = p.solve_with(options).unwrap();
    // Whatever the paused status, editing must work on the ORIGINAL problem + edit.
    let sol = sol
        .add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
        .unwrap();
    let sol = if sol.status() == Status::Optimal {
        sol
    } else {
        sol.resume(None).unwrap()
    };
    assert!((sol.objective() - 13.0).abs() < 1e-6);
}

#[test]
fn milp_infeasible_edit_is_an_error() {
    let (p, a, _) = int_2var_problem();
    let sol = p.solve().unwrap();
    // a <= -1 crosses a's [0,10] bounds → infeasible.
    assert!(matches!(
        sol.add_constraint(&[(a, 1.0)], ComparisonOp::Le, -1.0),
        Err(Error::Infeasible)
    ));
}

#[test]
fn milp_gomory_cut_is_rejected_with_internal_error() {
    let (p, a, _) = int_2var_problem();
    let sol = p.solve().unwrap();
    assert!(matches!(
        sol.add_gomory_cut(a),
        Err(Error::InternalError(_))
    ));
}
```

Note for `milp_add_constraint_resolves_on_base_problem`: `Variable(0)` is not constructible
outside the crate, but `src/tests/` is inside the crate — fine here. (The suite uses its own
handles.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --release milp_`
Expected: the edit tests fail with the interim `InternalError("editing MILP solutions is
implemented in phase 3 …")` from Task 5.

- [ ] **Step 3: Implement the mip-side machinery** (in `src/mip/mod.rs`)

1. Add fields to `MipState`:

```rust
    /// Clean copy of the user's problem, including post-solve edits — never
    /// contains branching artifacts. Post-solve edits re-solve from this.
    pub base: Problem,
    /// User-level fix_var overlay on `base` (var → fixed value).
    pub fixed: std::collections::BTreeMap<usize, f64>,
```

   In `run(problem, options)`, initialize `base: problem.clone(), fixed: BTreeMap::new()`.

2. Add the helpers:

```rust
use std::collections::BTreeMap;

/// `base` with the fix_var overlay applied to the variable bounds.
pub(crate) fn effective_problem(base: &Problem, fixed: &BTreeMap<usize, f64>) -> Problem {
    let mut p = base.clone();
    for (&v, &val) in fixed {
        p.var_mins[v] = val;
        p.var_maxs[v] = val;
    }
    p
}

/// Cheap feasibility check of a value vector against base + fixes (bounds,
/// integrality to 1e-5, every constraint to a relative 1e-7 tolerance).
pub(crate) fn incumbent_feasible(
    base: &Problem,
    fixed: &BTreeMap<usize, f64>,
    values: &[f64],
) -> bool {
    for v in 0..values.len() {
        let (mut lo, mut hi) = (base.var_mins[v], base.var_maxs[v]);
        if let Some(&f) = fixed.get(&v) {
            lo = f;
            hi = f;
        }
        if values[v] < lo - 1e-7 || values[v] > hi + 1e-7 {
            return false;
        }
        if matches!(base.var_domains[v], VarDomain::Integer | VarDomain::Boolean)
            && (values[v] - values[v].round()).abs() > 1e-5
        {
            return false;
        }
    }
    for (coeffs, op, rhs) in &base.constraints {
        let lhs: f64 = coeffs.iter().map(|(i, c)| c * values[i]).sum();
        let tol = 1e-7 * rhs.abs().max(1.0);
        let ok = match op {
            ComparisonOp::Eq => (lhs - rhs).abs() <= tol,
            ComparisonOp::Le => lhs <= rhs + tol,
            ComparisonOp::Ge => lhs >= *rhs - tol,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// After a user edit: drop the open tree, carry the incumbent as a warm-start
/// hint when it survives the edit, and re-run the search on base + fixes.
/// The fresh run gets the state's original options (incl. a fresh time budget).
pub(crate) fn reedit_and_resolve(state: Box<MipState>) -> Result<MipRun, Error> {
    let MipState {
        base,
        fixed,
        incumbent,
        mut options,
        ..
    } = *state;

    options.warm_start = incumbent
        .filter(|inc| incumbent_feasible(&base, &fixed, &inc.values))
        .map(|inc| {
            inc.values
                .iter()
                .enumerate()
                .map(|(v, &val)| (Variable(v), val))
                .collect()
        });

    let effective = effective_problem(&base, &fixed);
    let mut run = run(&effective, options)?;
    // `run` cloned `effective` as its base; restore the true base/fixed split so
    // later edits keep composing against the user's problem.
    run.state.base = base;
    run.state.fixed = fixed;
    Ok(run)
}
```

   Imports needed: `use crate::{ComparisonOp, Variable};` (extend the existing `use crate::…` line).

3. `Problem` needs `Clone` (it already derives it) and `mip` accesses its private fields —
   already the case since Task 4.

- [ ] **Step 4: Implement the Solution-side arms** (in `src/lib.rs`)

Replace the three interim `SolutionKind::Mip(_) => Err(...)` arms:

These arms consume `self.kind` by value, so first restructure each of the three methods to
destructure `self` up front (the LP arms then rebuild `Solution` the same way — keep their
`Status::Interrupted` guard and their behavior identical). Full shape for `add_constraint`;
`fix_var` and `unfix_var` follow the same skeleton with their arms below:

```rust
pub fn add_constraint(
    self,
    expr: impl Into<LinearExpr>,
    cmp_op: ComparisonOp,
    rhs: f64,
) -> Result<Self, Error> {
    let Solution {
        direction,
        num_vars,
        status,
        kind,
    } = self;
    match kind {
        SolutionKind::Lp(mut solver) => {
            if status == Status::Interrupted {
                return Err(Error::InternalError(
                    "cannot edit an interrupted solution; resume() it first".to_string(),
                ));
            }
            let expr = expr.into();
            let sr = solver.add_constraint(
                CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                    .map_err(|v| Error::InternalError(v.2.to_string()))?,
                cmp_op,
                rhs,
            )?;
            Ok(Solution {
                direction,
                num_vars,
                status: match sr {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                },
                kind: SolutionKind::Lp(solver),
            })
        }
        SolutionKind::Mip(mut state) => {
            // Edit-after-pause is allowed by design: any status is editable here.
            let expr = expr.into();
            let coeffs = CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                .map_err(|v| Error::InternalError(v.2.to_string()))?;
            state.base.constraints.push((coeffs, cmp_op, rhs));
            let run = mip::reedit_and_resolve(state)?;
            Ok(Solution {
                direction,
                num_vars,
                status: mip::status_of(run.outcome, &run.state),
                kind: SolutionKind::Mip(Box::new(run.state)),
            })
        }
    }
}
```

`fix_var`'s MILP arm (LP arm unchanged from Task 5, rebuilt via the same destructuring):

```rust
SolutionKind::Mip(mut state) => {
    if val < state.base.var_mins[var.0] || val > state.base.var_maxs[var.0] {
        return Err(Error::Infeasible);
    }
    state.fixed.insert(var.0, val);
    let run = mip::reedit_and_resolve(state)?;
    Ok(Solution {
        direction,
        num_vars,
        status: mip::status_of(run.outcome, &run.state),
        kind: SolutionKind::Mip(Box::new(run.state)),
    })
}
```

`unfix_var`'s MILP arm (returns `(Self, bool)`):

```rust
SolutionKind::Mip(mut state) => {
    if state.fixed.remove(&var.0).is_none() {
        return (
            Solution {
                direction,
                num_vars,
                status,
                kind: SolutionKind::Mip(state),
            },
            false,
        );
    }
    // Relaxing a fix cannot make the problem infeasible; internal failures are
    // solver bugs — panic loudly rather than losing them (matches the LP arm's
    // historical unwrap in Solver::unfix_var).
    let run = mip::reedit_and_resolve(state).expect("re-solve after unfix_var failed");
    (
        Solution {
            direction,
            num_vars,
            status: mip::status_of(run.outcome, &run.state),
            kind: SolutionKind::Mip(Box::new(run.state)),
        },
        true,
    )
}
```

Also update the `add_constraint`/`fix_var` doc comments to drop the "implemented in phase 3"
sentence and document the MILP semantics: "MILP solutions are re-solved on the original
problem plus all edits; the previous incumbent is kept as a warm start when it remains
feasible. Editing a paused (`Feasible`/`Interrupted`) MILP solution is allowed — the open
search tree is discarded and the search restarts from the edited problem."

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --release milp_`
Expected: all 6 new tests pass; Task 5's `lp_path_still_solves_and_edits_incrementally`
still passes (LP arm untouched semantically).

- [ ] **Step 6: Full check + suite + fmt**

Run: `cargo test --release && cargo test --release --test suite && cargo fmt`
Expected: green.

---

### Task 12 (phase 3): Flip `incr/*-milp*` known-failures; final sweep

**Files:**
- Modify: `tests/suite/cases/incremental.rs` (promote the MILP edit cases out of
  known-failure status, using the harness mechanism learned in Task 6 Step 1)

**Interfaces:**
- Consumes: Task 11 semantics (edits act on base problem, incumbent carried as warm start).
- Produces: hard tier where the ONLY remaining failure is `netlib/brandy`.

- [ ] **Step 1: Flip the cases**

In `tests/suite/cases/incremental.rs`, the `milp_known_failures` registrations assert the
CORRECT documented behavior already (they were written as executable bug reports). Promote
them to regular cases (default tier if their runtime is comparable to the other `incr/*`
cases, else keep hard tier without the known-failure marker). Do not change what they assert.

- [ ] **Step 2: Run both tiers**

Run: `cargo test --release --test suite && cargo test --release --test suite -- --hard`
Expected: default tier green including any promoted cases; hard tier's only failure is
`netlib/brandy` (pre-existing LP phase-1 bug, documented out of scope in the spec §11).

- [ ] **Step 3: Full-repo verification**

Run: `cargo test --release && cargo build --examples && cargo clippy --all-targets --release 2>&1 | tail -20 && cargo fmt --check`
Expected: tests green; clippy introduces no NEW warnings vs. the pre-refactor baseline
(run `git stash && cargo clippy --all-targets --release 2>&1 | tail -20 && git stash pop`
first if a baseline is needed); fmt clean.

- [ ] **Step 4: Update the crate docs**

In `src/lib.rs`'s crate-level docs (`//!` block): the feature bullet
"Incremental: add constraints to an existing solution…" gets a second bullet:
"Interruptible: time/node limits with clean resume, warm starts from known solutions,
and MIP gap reporting for integer problems." Verify `cargo test --release --doc` passes.

- [ ] **Step 5: Leave the tree dirty**

Per the global constraints: do NOT commit. Report completion with the working tree as-is
and the verification outputs from Steps 2–4.

---

## Deferred (explicitly NOT in this plan)

Per spec §10 phase 4 / §11: implicit SOS1 detection, rounding/diving heuristics, integer
presolve, MPS INTORG support, the `netlib/brandy` phase-1 LP fix, parallelism, serde for
`MipState`, cut generation inside B&B. The `Node.bound_changes` list representation already
supports multi-var branches, which is the hook future SOS branching needs.
