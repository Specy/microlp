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
        .all(|(v, _)| fractionality(solver.get_value(v)) <= int_tol)
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
        let val = solver.get_value(v);
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
mod tests;
