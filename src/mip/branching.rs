//! Branch variable selection.

use super::params::{PSEUDOCOST_INIT_EPS, SCORE_EPS};
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
            .map(|v| obj_coeffs.get(v).copied().unwrap_or(0.0).abs() + PSEUDOCOST_INIT_EPS)
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

    /// Total number of `record` calls folded in so far, across every variable
    /// and direction. A cheap summary for diagnostics (e.g. `Debug` impls)
    /// that avoids printing the full per-variable vectors.
    pub(crate) fn observation_count(&self) -> u64 {
        self.up_n.iter().map(|&n| u64::from(n)).sum::<u64>()
            + self.down_n.iter().map(|&n| u64::from(n)).sum::<u64>()
    }
}

/// Pseudocost product rule: pick the fractional int var maximizing
/// max(est_down·f_down, ε) · max(est_up·f_up, ε).
pub(crate) fn choose_branch_var(
    solver: &Solver,
    domains: &[VarDomain],
    int_tol: f64,
    pc: &PseudoCosts,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (v, d) in domains.iter().enumerate() {
        if !is_int_domain(d) {
            continue;
        }
        // A fixed var (lo == hi; integral bounds make hi - lo < 0.5 the
        // robust test) cannot move: branching on it reproduces the parent
        // node verbatim. Its LP value can still carry sub-EPS basic noise
        // (e.g. -5e-16 on a var fixed to 0), including when this function is
        // called with int_tol = 0 after a rounded candidate is rejected.
        let (lo, hi) = solver.get_var_bounds(v);
        if hi - lo < 0.5 {
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
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((v, score));
        }
    }
    best.map(|(v, _)| v)
}

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
        // With a fresh PseudoCosts (uniform init, no recorded data), the product score
        // est_down·f_down · est_up·f_up is maximized by the most-fractional var too:
        // var 1: 0.65·0.35 ≈ 0.23 beats var 0: 0.9·0.1 = 0.09.
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
        assert!(!is_integral(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6
        ));
        let pc = PseudoCosts::new(&[-1.0, -1.0], 2);
        assert_eq!(
            choose_branch_var(
                &solver,
                solver.orig_var_domains.clone().as_slice(),
                1e-6,
                &pc
            ),
            Some(1)
        );
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
        assert!(is_integral(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6
        ));
        let pc = PseudoCosts::new(&[1.0], 1);
        assert_eq!(
            choose_branch_var(
                &solver,
                solver.orig_var_domains.clone().as_slice(),
                1e-6,
                &pc
            ),
            None
        );
    }

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
}
