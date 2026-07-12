//! Node-level bound propagation ("implications"): the activity-based domain
//! deduction the presolve runs at the root, replayed at every branch & bound
//! node on the node's own bounds. Design doc:
//! docs/superpowers/specs/2026-07-11-bb-improvements-design.md §3.
//!
//! After a node's bound changes are applied to the solver and BEFORE its LP
//! is solved, a worklist pass over the rows containing changed variables
//! derives what those changes force elsewhere: on a fixed-charge row
//! `x − M·y ≤ 0`, branching `y = 0` instantly pins `x = 0`; on general
//! integers, a tightened sibling bound often rounds a whole unit inward.
//! Deduced bounds are appended to the node's `bound_changes`, so children
//! inherit them for free, and a contradiction prunes the node without any
//! LP work.
//!
//! Everything deduced here is a feasibility-preserving consequence of the
//! node's subproblem, so the node LP's optimum (the bound used for pruning)
//! is unchanged — the LP just starts closer to it. Node bounds never
//! surface through the public API and post-solve edits re-solve from the
//! untouched base problem, so no emitted/working distinction is needed (the
//! subtlety that presolve has to care about).
//!
//! Bounds are read from and written to the [`Solver`] directly: the
//! solver's current variable bounds ARE the node's bounds, which avoids
//! copying two `num_vars`-sized arrays per node (on 40k-variable instances
//! with thousands of nodes that would dominate the search).

use crate::presolve::{FEAS_TOL, IMPROVE_REL, NOISE_REL};
use crate::solver::Solver;
use crate::{ComparisonOp, CsVec, Error, VarDomain};

/// Waves per node: each wave processes the rows made dirty by the previous
/// one. Chains longer than this are rare and simply left for the LP; the cap
/// keeps worst-case node overhead bounded and deterministic.
const WAVE_CAP: usize = 4;

/// One row in propagation form (`lo <= a·x <= hi`, one side finite for
/// Le/Ge). Slack variables never appear — these are the presolved rows as
/// handed to the solver, in plain data.
#[derive(Clone, Debug)]
struct PropRow {
    vars: Vec<usize>,
    coeffs: Vec<f64>,
    lo: f64,
    hi: f64,
}

/// The propagation view of the search's rows plus reusable scratch. Built
/// once per search in `mip::run`, cloned with `MipState` (plain data).
#[derive(Clone, Debug)]
pub(crate) struct Propagator {
    rows: Vec<PropRow>,
    /// var -> rows adjacency, flat CSR.
    adj_start: Vec<u32>,
    adj_rows: Vec<u32>,
    /// Scratch (cleared after every call): dirty flag + wave queues.
    dirty: Vec<bool>,
    queue: Vec<u32>,
    next: Vec<u32>,
}

fn is_int_domain(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

impl Propagator {
    /// Build from the (presolved) constraints the search hands to the solver.
    pub(crate) fn new(constraints: &[(CsVec, ComparisonOp, f64)], num_vars: usize) -> Self {
        let mut rows = Vec::with_capacity(constraints.len());
        for (coeffs, op, rhs) in constraints {
            let (lo, hi) = match op {
                ComparisonOp::Le => (f64::NEG_INFINITY, *rhs),
                ComparisonOp::Ge => (*rhs, f64::INFINITY),
                ComparisonOp::Eq => (*rhs, *rhs),
            };
            rows.push(PropRow {
                vars: coeffs.indices().to_vec(),
                coeffs: coeffs.data().to_vec(),
                lo,
                hi,
            });
        }
        let mut prop = Propagator {
            rows,
            adj_start: Vec::new(),
            adj_rows: Vec::new(),
            dirty: Vec::new(),
            queue: Vec::new(),
            next: Vec::new(),
        };
        prop.rebuild_adj(num_vars);
        prop
    }

    /// (Re)build the var → rows adjacency and the dirty scratch from
    /// `self.rows`.
    fn rebuild_adj(&mut self, num_vars: usize) {
        let mut adj_start = vec![0u32; num_vars + 1];
        for row in &self.rows {
            for &v in &row.vars {
                adj_start[v + 1] += 1;
            }
        }
        for v in 0..num_vars {
            adj_start[v + 1] += adj_start[v];
        }
        let mut adj_rows = vec![0u32; adj_start[num_vars] as usize];
        let mut fill = adj_start.clone();
        for (r, row) in self.rows.iter().enumerate() {
            for &v in &row.vars {
                adj_rows[fill[v] as usize] = r as u32;
                fill[v] += 1;
            }
        }
        self.adj_start = adj_start;
        self.adj_rows = adj_rows;
        self.dirty = vec![false; self.rows.len()];
    }

    /// Read-only view of the rows (activity form `lo ≤ Σ coeffs·x ≤ hi`),
    /// consumed by the root cut separator.
    pub(crate) fn rows(&self) -> impl Iterator<Item = (&[usize], &[f64], f64, f64)> {
        self.rows
            .iter()
            .map(|r| (r.vars.as_slice(), r.coeffs.as_slice(), r.lo, r.hi))
    }

    /// Append `Σ coeffs·x ≤ rhs` rows (root cuts) and rebuild the adjacency
    /// so node propagation deduces from them like any other row. Root-only:
    /// the search-time invariant that rows never change once nodes exist is
    /// the caller's to hold (`run_root_cuts` runs before the root node is
    /// built).
    pub(crate) fn add_le_rows(&mut self, cuts: &[(Vec<usize>, Vec<f64>, f64)], num_vars: usize) {
        for (vars, coeffs, rhs) in cuts {
            self.rows.push(PropRow {
                vars: vars.clone(),
                coeffs: coeffs.clone(),
                lo: f64::NEG_INFINITY,
                hi: *rhs,
            });
        }
        self.rebuild_adj(num_vars);
    }

    fn mark_var(&mut self, v: usize) {
        let (s, e) = (self.adj_start[v] as usize, self.adj_start[v + 1] as usize);
        for i in s..e {
            let r = self.adj_rows[i] as usize;
            if !self.dirty[r] {
                self.dirty[r] = true;
                self.next.push(r as u32);
            }
        }
    }

    /// Propagate the consequences of this node's changed variables.
    ///
    /// `seed_vars` are the vars whose bounds differ from the root (the
    /// node's `effective_bounds` — a few entries, one per branching along
    /// the path). Every deduced tightening is applied to the solver AND
    /// pushed onto `out_changes` (the node's `bound_changes`), so the caller
    /// can rebuild `applied` and children inherit the deductions.
    ///
    /// Returns `Err(Infeasible)` when a row's activity proves the node
    /// empty; the caller prunes without LP work. Scratch state is fully
    /// cleared before returning on every path.
    pub(crate) fn propagate(
        &mut self,
        solver: &mut Solver,
        seed_vars: impl Iterator<Item = usize>,
        domains: &[VarDomain],
        int_tol: f64,
        out_changes: &mut Vec<(usize, f64, f64)>,
    ) -> Result<(), Error> {
        debug_assert!(self.queue.is_empty() && self.next.is_empty());
        for v in seed_vars {
            self.mark_var(v);
        }
        let mut result = Ok(());
        'waves: for _ in 0..WAVE_CAP {
            if self.next.is_empty() {
                break;
            }
            std::mem::swap(&mut self.queue, &mut self.next);
            for qi in 0..self.queue.len() {
                let r = self.queue[qi] as usize;
                self.dirty[r] = false;
                if let Err(e) = process_row(self, r, solver, domains, int_tol, out_changes) {
                    result = Err(e);
                    break 'waves;
                }
            }
            self.queue.clear();
        }
        // Clear scratch on every path (incl. infeasible early-out and the
        // wave cap): whatever remains queued is simply abandoned — every
        // deduction is optional.
        for &r in self.queue.iter().chain(self.next.iter()) {
            self.dirty[r as usize] = false;
        }
        self.queue.clear();
        self.next.clear();
        result
    }
}

/// Process one row: infeasibility check + per-term implied bounds, reading
/// and writing the solver's current bounds. Free function to keep borrows of
/// the propagator's parts disjoint (rows are read, scratch is written via
/// `mark_var` after each tightening).
fn process_row(
    prop: &mut Propagator,
    r: usize,
    solver: &mut Solver,
    domains: &[VarDomain],
    int_tol: f64,
    out_changes: &mut Vec<(usize, f64, f64)>,
) -> Result<(), Error> {
    // Activity bounds of the row under the node's current bounds.
    let (row_lo, row_hi) = (prop.rows[r].lo, prop.rows[r].hi);
    let mut act_l = 0.0;
    let mut act_u = 0.0;
    let mut ninf_l = 0u32;
    let mut ninf_u = 0u32;
    let mut abs = 0.0;
    for i in 0..prop.rows[r].vars.len() {
        let (v, a) = (prop.rows[r].vars[i], prop.rows[r].coeffs[i]);
        let (blo, bhi) = solver.get_var_bounds(v);
        let (cmin, cmax) = if a > 0.0 {
            (a * blo, a * bhi)
        } else {
            (a * bhi, a * blo)
        };
        if cmin.is_finite() {
            act_l += cmin;
            abs += cmin.abs();
        } else {
            ninf_l += 1;
        }
        if cmax.is_finite() {
            act_u += cmax;
            abs += cmax.abs();
        } else {
            ninf_u += 1;
        }
    }
    if row_lo.is_finite() {
        abs += row_lo.abs();
    }
    if row_hi.is_finite() {
        abs += row_hi.abs();
    }

    // Infeasibility: same generous tolerance as presolve — a false positive
    // here prunes a feasible subtree, i.e. a wrong answer.
    let inf_tol = FEAS_TOL * abs.max(1.0);
    if row_hi.is_finite() && ninf_l == 0 && act_l > row_hi + inf_tol {
        return Err(Error::Infeasible);
    }
    if row_lo.is_finite() && ninf_u == 0 && act_u < row_lo - inf_tol {
        return Err(Error::Infeasible);
    }

    let nz = NOISE_REL * abs;
    for i in 0..prop.rows[r].vars.len() {
        let (v, a) = (prop.rows[r].vars[i], prop.rows[r].coeffs[i]);
        if a == 0.0 {
            continue;
        }
        let (blo, bhi) = solver.get_var_bounds(v);
        let (cmin, cmax) = if a > 0.0 {
            (a * blo, a * bhi)
        } else {
            (a * bhi, a * blo)
        };
        let rest_l = if cmin.is_finite() {
            (ninf_l == 0).then_some(act_l - cmin)
        } else {
            (ninf_l == 1).then_some(act_l)
        };
        let rest_u = if cmax.is_finite() {
            (ninf_u == 0).then_some(act_u - cmax)
        } else {
            (ninf_u == 1).then_some(act_u)
        };
        let amp = nz / a.abs();
        let is_int = is_int_domain(&domains[v]);
        let mut new_lo = blo;
        let mut new_hi = bhi;
        if row_hi.is_finite() {
            if let Some(rl) = rest_l {
                let q = (row_hi - rl) / a;
                if a > 0.0 {
                    tighten_upper(q, amp, is_int, int_tol, new_lo, &mut new_hi)?;
                } else {
                    tighten_lower(q, amp, is_int, int_tol, &mut new_lo, new_hi)?;
                }
            }
        }
        if row_lo.is_finite() {
            if let Some(ru) = rest_u {
                let q = (row_lo - ru) / a;
                if a > 0.0 {
                    tighten_lower(q, amp, is_int, int_tol, &mut new_lo, new_hi)?;
                } else {
                    tighten_upper(q, amp, is_int, int_tol, new_lo, &mut new_hi)?;
                }
            }
        }
        if new_lo > blo || new_hi < bhi {
            solver
                .set_var_bounds(v, new_lo, new_hi)
                .expect("tighten helpers never produce crossing bounds");
            out_changes.push((v, new_lo, new_hi));
            prop.mark_var(v);
        }
    }
    Ok(())
}

/// Fold the implied upper bound `q` (trustworthy to `amp_noise`) into
/// `*hi`. Same math and tolerance policy as the presolve's working-bound
/// tightening: integers round with the noise-aware slack, continuous bounds
/// are relaxed outward so rounding noise can never cut a feasible point, a
/// beyond-tolerance crossing is a proof of infeasibility and a
/// within-tolerance sliver is left alone.
fn tighten_upper(
    q: f64,
    amp_noise: f64,
    is_int: bool,
    int_tol: f64,
    lo: f64,
    hi: &mut f64,
) -> Result<(), Error> {
    if !q.is_finite() {
        return Ok(());
    }
    if is_int {
        let slack = int_tol.max(amp_noise);
        if slack > 0.1 {
            return Ok(());
        }
        let qi = (q + slack).floor();
        if qi < *hi - 0.5 {
            if qi < lo - 0.5 {
                return Err(Error::Infeasible);
            }
            *hi = qi;
        }
    } else {
        let relax = IMPROVE_REL * q.abs().max(1.0) + amp_noise;
        let nq = q + relax;
        if nq < *hi {
            if nq < lo {
                let scale = nq.abs().max(lo.abs()).max(1.0);
                if lo - nq > FEAS_TOL * scale {
                    return Err(Error::Infeasible);
                }
                return Ok(()); // sub-tolerance sliver
            }
            *hi = nq;
        }
    }
    Ok(())
}

/// Mirror of [`tighten_upper`] for the lower bound.
fn tighten_lower(
    q: f64,
    amp_noise: f64,
    is_int: bool,
    int_tol: f64,
    lo: &mut f64,
    hi: f64,
) -> Result<(), Error> {
    if !q.is_finite() {
        return Ok(());
    }
    if is_int {
        let slack = int_tol.max(amp_noise);
        if slack > 0.1 {
            return Ok(());
        }
        let qi = (q - slack).ceil();
        if qi > *lo + 0.5 {
            if qi > hi + 0.5 {
                return Err(Error::Infeasible);
            }
            *lo = qi;
        }
    } else {
        let relax = IMPROVE_REL * q.abs().max(1.0) + amp_noise;
        let nq = q - relax;
        if nq > *lo {
            if nq > hi {
                let scale = nq.abs().max(hi.abs()).max(1.0);
                if nq - hi > FEAS_TOL * scale {
                    return Err(Error::Infeasible);
                }
                return Ok(());
            }
            *lo = nq;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::Solver;

    fn csvec(n: usize, terms: &[(usize, f64)]) -> CsVec {
        CsVec::new_from_unsorted(
            n,
            terms.iter().map(|t| t.0).collect(),
            terms.iter().map(|t| t.1).collect(),
        )
        .unwrap()
    }

    /// Build a solver + propagator for the given problem pieces.
    fn setup(
        mins: &[f64],
        maxs: &[f64],
        cons: Vec<(CsVec, ComparisonOp, f64)>,
        domains: Vec<VarDomain>,
    ) -> (Solver, Propagator, Vec<VarDomain>) {
        let n = mins.len();
        let solver = Solver::try_new(&vec![0.0; n], mins, maxs, &cons, &domains, None).unwrap();
        let prop = Propagator::new(&cons, n);
        (solver, prop, domains)
    }

    #[test]
    fn fixed_charge_implication_pins_continuous_var() {
        // x - 10 y <= 0 with binary y, x in [0, 10]: fixing y = 0 must force
        // x <= (about) 0.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, -10.0)]), ComparisonOp::Le, 0.0)];
        let (mut solver, mut prop, domains) = setup(
            &[0.0, 0.0],
            &[10.0, 1.0],
            cons,
            vec![VarDomain::Real, VarDomain::Boolean],
        );
        solver.set_var_bounds(1, 0.0, 0.0).unwrap(); // branch: y = 0
        let mut changes = vec![];
        prop.propagate(
            &mut solver,
            [1usize].into_iter(),
            &domains,
            1e-6,
            &mut changes,
        )
        .unwrap();
        let (_, x_hi) = solver.get_var_bounds(0);
        assert!(x_hi < 1e-6, "y = 0 must pin x to ~0, got hi = {}", x_hi);
        assert!(changes.iter().any(|&(v, _, _)| v == 0));
    }

    #[test]
    fn integer_chain_propagates_through_two_rows() {
        // a + b >= 5 (ints in [0, 10]); b + c <= 4. Branch a <= 1 =>
        // b >= 4 => c <= 0.
        let cons = vec![
            (csvec(3, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 5.0),
            (csvec(3, &[(1, 1.0), (2, 1.0)]), ComparisonOp::Le, 4.0),
        ];
        let (mut solver, mut prop, domains) =
            setup(&[0.0; 3], &[10.0; 3], cons, vec![VarDomain::Integer; 3]);
        solver.set_var_bounds(0, 0.0, 1.0).unwrap(); // branch: a <= 1
        let mut changes = vec![];
        prop.propagate(
            &mut solver,
            [0usize].into_iter(),
            &domains,
            1e-6,
            &mut changes,
        )
        .unwrap();
        assert_eq!(solver.get_var_bounds(1).0, 4.0, "b >= 4 must be deduced");
        assert_eq!(solver.get_var_bounds(2).1, 0.0, "c <= 0 must follow");
    }

    #[test]
    fn contradiction_is_reported_infeasible() {
        // x + y >= 8 with x, y in [0, 10]: branch x <= 1 AND y <= 2 is empty.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 8.0)];
        let (mut solver, mut prop, domains) =
            setup(&[0.0; 2], &[10.0; 2], cons, vec![VarDomain::Integer; 2]);
        solver.set_var_bounds(0, 0.0, 1.0).unwrap();
        solver.set_var_bounds(1, 0.0, 2.0).unwrap();
        let mut changes = vec![];
        let res = prop.propagate(
            &mut solver,
            [0usize, 1].into_iter(),
            &domains,
            1e-6,
            &mut changes,
        );
        assert_eq!(res.unwrap_err(), Error::Infeasible);
    }

    #[test]
    fn scratch_is_clean_after_infeasible_and_reusable() {
        // Same contradiction as above, then a benign propagation on the same
        // propagator must still work (scratch fully cleared).
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 8.0)];
        let (mut solver, mut prop, domains) =
            setup(&[0.0; 2], &[10.0; 2], cons, vec![VarDomain::Integer; 2]);
        solver.set_var_bounds(0, 0.0, 1.0).unwrap();
        solver.set_var_bounds(1, 0.0, 2.0).unwrap();
        let mut changes = vec![];
        assert!(prop
            .propagate(
                &mut solver,
                [0usize, 1].into_iter(),
                &domains,
                1e-6,
                &mut changes
            )
            .is_err());
        assert!(prop.queue.is_empty() && prop.next.is_empty());
        assert!(prop.dirty.iter().all(|&d| !d));

        // Loosen back; a normal propagation must succeed.
        solver.set_var_bounds(0, 0.0, 10.0).unwrap();
        solver.set_var_bounds(1, 0.0, 6.0).unwrap();
        let mut changes = vec![];
        prop.propagate(
            &mut solver,
            [1usize].into_iter(),
            &domains,
            1e-6,
            &mut changes,
        )
        .unwrap();
        // x >= 8 - 6 = 2 must be deduced.
        assert_eq!(solver.get_var_bounds(0).0, 2.0);
    }

    #[test]
    fn no_seed_no_work() {
        let cons = vec![(csvec(1, &[(0, 1.0)]), ComparisonOp::Le, 5.0)];
        let (mut solver, mut prop, domains) =
            setup(&[0.0], &[10.0], cons, vec![VarDomain::Integer]);
        let mut changes = vec![];
        prop.propagate(
            &mut solver,
            std::iter::empty(),
            &domains,
            1e-6,
            &mut changes,
        )
        .unwrap();
        assert!(changes.is_empty());
    }
}
