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
        assert!(!is_integral(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6
        ));
        assert_eq!(
            choose_branch_var(&solver, solver.orig_var_domains.clone().as_slice(), 1e-6),
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
        assert_eq!(
            choose_branch_var(&solver, solver.orig_var_domains.clone().as_slice(), 1e-6),
            None
        );
    }
}
