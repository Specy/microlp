//! Post-solve edits on a MILP solution: `add_constraint`, `fix_var` and
//! `unfix_var` re-solve against the untouched base problem.

use super::common::{int_2var_problem, solution};
use crate::*;

#[test]
fn milp_add_constraint_resolves_on_base_problem() {
    let (p, a, b) = int_2var_problem();
    let sol = solution(p.solve().unwrap());
    assert!((sol.objective() - 11.0).abs() < 1e-6); // a=1, b=2

    // Cut off the incumbent: a + b >= 4. From-scratch optimum of the edited
    // problem, over points on the binding line a+b=4: (a=0,b=4): cons1 8>=5,
    // cons2 4>=4, obj 16; (a=1,b=3): 7>=5, 6>=4, obj 15; (a=2,b=2): 6>=5,
    // 8>=4, obj 14; (a=3,b=1): cons1 5>=5, cons2 10>=4, obj 13; (a=4,b=0)
    // violates a+2b>=5. → unique optimum 13 at (3,1).
    let sol = solution(
        sol.add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
            .unwrap(),
    );
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 13.0).abs() < 1e-6);
    assert_eq!(sol.var_value(a), 3.0);
    assert_eq!(sol.var_value(b), 1.0);

    // Must equal a from-scratch solve of the edited problem.
    let (mut p2, a2, b2) = (int_2var_problem().0, Variable(0), Variable(1));
    p2.add_constraint(&[(a2, 1.0), (b2, 1.0)], ComparisonOp::Ge, 4.0);
    let fresh = solution(p2.solve().unwrap());
    assert!((fresh.objective() - sol.objective()).abs() < 1e-6);
}

#[test]
fn milp_fix_and_unfix_var_roundtrip() {
    let (p, a, b) = int_2var_problem();
    let sol = solution(p.solve().unwrap());

    // Fix a=3: then b >= 1 (cons1: 3+2b>=5) → obj 9+4=13 at (3,1).
    let sol = solution(sol.fix_var(a, 3.0).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 13.0).abs() < 1e-6);
    assert_eq!(sol.var_value(a), 3.0);
    assert_eq!(sol.var_value(b), 1.0);

    // Unfix restores the original optimum and reports it was fixed.
    let (sol, was_fixed) = sol.unfix_var(a).unwrap();
    let sol = solution(sol);
    assert!(was_fixed);
    assert!((sol.objective() - 11.0).abs() < 1e-6);

    // Unfixing a never-fixed var is a no-op with `false`.
    let (sol, was_fixed) = sol.unfix_var(b).unwrap();
    let sol = solution(sol);
    assert!(!was_fixed);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn milp_fix_var_outside_bounds_is_infeasible_error() {
    let (p, a, _) = int_2var_problem();
    let sol = solution(p.solve().unwrap());
    assert!(matches!(sol.fix_var(a, 99.0), Err(Error::Infeasible)));
}

#[test]
fn milp_fix_var_non_finite_is_infeasible_error() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (p, a, _) = int_2var_problem();
        let sol = solution(p.solve().unwrap());
        assert!(
            matches!(sol.fix_var(a, invalid), Err(Error::Infeasible)),
            "non-finite fix {invalid} was not rejected"
        );
    }
}

#[test]
fn milp_edit_after_pause_completes_correctly() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.node_limit = Some(1); // pause almost immediately
    let mut outcome = p.solve_with(options).unwrap();
    // Interrupted searches are deliberately not editable. Resume to a
    // validated solution before changing the original model.
    while outcome.solution().is_none() {
        outcome = outcome.resume().unwrap();
    }
    let sol = solution(outcome);
    let edited = sol
        .add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
        .unwrap();
    let sol = solution(if edited.is_optimal() {
        edited
    } else {
        edited.resume().unwrap()
    });
    assert!((sol.objective() - 13.0).abs() < 1e-6);
}

#[test]
fn milp_infeasible_edit_is_an_error() {
    let (p, a, _) = int_2var_problem();
    let sol = solution(p.solve().unwrap());
    // a <= -1 crosses a's [0,10] bounds → infeasible.
    assert!(matches!(
        sol.add_constraint(&[(a, 1.0)], ComparisonOp::Le, -1.0),
        Err(Error::Infeasible)
    ));
}
