use super::*;

fn csvec(n: usize, terms: &[(usize, f64)]) -> CsVec {
    CsVec::new_from_unsorted(
        n,
        terms.iter().map(|t| t.0).collect(),
        terms.iter().map(|t| t.1).collect(),
    )
    .unwrap()
}

fn real(n: usize) -> Vec<VarDomain> {
    vec![VarDomain::Real; n]
}

/// Whether an emitted row is the given input row, to the last bit.
fn same_row(a: &(CsVec, ComparisonOp, f64), b: &(CsVec, ComparisonOp, f64)) -> bool {
    a.0.indices() == b.0.indices()
        && a.0.data() == b.0.data()
        && std::mem::discriminant(&a.1) == std::mem::discriminant(&b.1)
        && a.2 == b.2
}

fn run_lp(
    mins: &[f64],
    maxs: &[f64],
    cons: &[(CsVec, ComparisonOp, f64)],
) -> Result<Presolved, Error> {
    let n = mins.len();
    presolve(
        &vec![0.0; n],
        mins,
        maxs,
        cons,
        &real(n),
        Mode::Lp,
        1e-7,
        1e-6,
        false,
    )
}

#[test]
fn kept_rows_do_not_leak_working_bounds() {
    // x + y <= 4 with x, y >= 0: presolve DEDUCES x, y <= 4 internally
    // but the row stays, so the emitted bounds must be untouched (an
    // emitted redundant bound would only perturb pivot order).
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 4.0)];
    let pre = run_lp(&[0.0, 0.0], &[f64::INFINITY, f64::INFINITY], &cons).unwrap();
    assert_eq!(pre.constraints.len(), 1, "row is binding, must be kept");
    assert_eq!(pre.var_maxs, vec![f64::INFINITY, f64::INFINITY]);
}

#[test]
fn integer_bounds_are_rounded_in_mip_mode() {
    let cons: Vec<(CsVec, ComparisonOp, f64)> = vec![];
    let pre = presolve(
        &[1.0],
        &[0.3],
        &[2.7],
        &cons,
        &[VarDomain::Integer],
        Mode::Mip,
        1e-7,
        1e-6,
        false,
    )
    .unwrap();
    assert_eq!(pre.var_mins[0], 1.0);
    assert_eq!(pre.var_maxs[0], 2.0);
}

#[test]
fn a_model_presolve_leaves_as_written_is_not_reduced() {
    // x + y <= 1 is binding over binaries and over [0, 1] reals: kept as
    // written, with every bound, so the solve takes the model as written. A
    // binary's lower bound rounds to -0, the same value.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 1.0)];
    let lp = run_lp(&[0.0, 0.0], &[1.0, 1.0], &cons).unwrap();
    assert!(!lp.reduced);
    assert!(reduced_model(Ok(lp)).unwrap().is_none());
    let binaries = [VarDomain::Boolean, VarDomain::Boolean];
    let mip = |maxs: &[f64], cons: &[(CsVec, ComparisonOp, f64)]| {
        presolve(
            &[-1.0, -1.0],
            &[0.0, 0.0],
            maxs,
            cons,
            &binaries,
            Mode::Mip,
            1e-7,
            1e-6,
            false,
        )
        .unwrap()
    };
    assert!(!mip(&[1.0, 1.0], &cons).reduced);
    // A rounded bound or a dropped row is a reduction.
    assert!(mip(&[1.5, 1.0], &cons).reduced);
    let loose = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 3.0)];
    let dropped = mip(&[1.0, 1.0], &loose);
    assert!(dropped.reduced && dropped.constraints.is_empty());
}

#[test]
fn presolve_leaves_every_continuous_var_its_bounds() {
    // x + y <= 0 is forcing over [0, 1] reals, 2 z = 1 is a singleton
    // equality, and w <= 5 a singleton row w's bounds imply: only the row the
    // bounds imply goes, and every bound stays as written.
    let cons = vec![
        (csvec(4, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 0.0),
        (csvec(4, &[(2, 2.0)]), ComparisonOp::Eq, 1.0),
        (csvec(4, &[(3, 1.0)]), ComparisonOp::Le, 5.0),
    ];
    let pre = run_lp(&[0.0; 4], &[1.0; 4], &cons).unwrap();
    assert!(pre.reduced);
    assert_eq!(pre.var_mins, vec![0.0; 4]);
    assert_eq!(pre.var_maxs, vec![1.0; 4]);
    assert_eq!(pre.constraints.len(), 2);
    assert!(same_row(&pre.constraints[0], &cons[0]));
    assert!(same_row(&pre.constraints[1], &cons[1]));
}

#[test]
fn presolve_keeps_the_integer_reductions() {
    // Vars c (real, cost 1), k (integer, cost 1), x (real, cost -1), y
    // (binary, cost 1), m (integer, cost -1). Decreasing c or k never hurts
    // c + k + x <= 5, so dual fixing puts k at 0, and c, continuous, keeps
    // its bounds; x - 10 y <= 0 is a big-M row whose coefficient tightens,
    // and 2 m + x <= 7 implies m <= 3.
    let cons = vec![
        (
            csvec(5, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
            ComparisonOp::Le,
            5.0,
        ),
        (csvec(5, &[(2, 1.0), (3, -10.0)]), ComparisonOp::Le, 0.0),
        (csvec(5, &[(4, 2.0), (2, 1.0)]), ComparisonOp::Le, 7.0),
    ];
    let domains = [
        VarDomain::Real,
        VarDomain::Integer,
        VarDomain::Real,
        VarDomain::Boolean,
        VarDomain::Integer,
    ];
    let pre = presolve(
        &[1.0, 1.0, -1.0, 1.0, -1.0],
        &[0.0; 5],
        &[4.0, 9.0, 1.0, 1.0, 9.0],
        &cons,
        &domains,
        Mode::Mip,
        1e-7,
        1e-6,
        true,
    )
    .unwrap();
    assert_eq!(pre.var_mins, vec![0.0; 5]);
    assert_eq!(pre.var_maxs, vec![4.0, 0.0, 1.0, 1.0, 3.0]);
    assert_eq!(pre.stats.coeffs_tightened, 1);
}

#[test]
fn integer_bound_from_kept_row_is_emitted() {
    // 2x + y <= 7 with y in [0, 10]: x <= 3.5 -> x <= 3 must be emitted
    // even though the row stays (integer rounding genuinely tightens).
    let cons = vec![(csvec(2, &[(0, 2.0), (1, 1.0)]), ComparisonOp::Le, 7.0)];
    let pre = presolve(
        &[-1.0, 0.0],
        &[0.0, 0.0],
        &[100.0, 10.0],
        &cons,
        &[VarDomain::Integer, VarDomain::Real],
        Mode::Mip,
        1e-7,
        1e-6,
        false,
    )
    .unwrap();
    assert_eq!(pre.var_maxs[0], 3.0);
    assert_eq!(pre.var_maxs[1], 10.0, "continuous bound must not persist");
    assert_eq!(pre.constraints.len(), 1);
}

#[test]
fn fractional_integer_range_is_infeasible() {
    // No integer in [0.4, 0.6].
    let cons: Vec<(CsVec, ComparisonOp, f64)> = vec![];
    let err = presolve(
        &[1.0],
        &[0.4],
        &[0.6],
        &cons,
        &[VarDomain::Integer],
        Mode::Mip,
        1e-7,
        1e-6,
        false,
    )
    .unwrap_err();
    assert_eq!(err, Error::Infeasible);
}

#[test]
fn singleton_row_becomes_bound_and_is_dropped() {
    // 0.5x <= 2.5 over an integer x >= 0 -> x <= 5, row gone.
    let cons = vec![(csvec(1, &[(0, 0.5)]), ComparisonOp::Le, 2.5)];
    let pre = presolve_mip(&[0.0], &[f64::INFINITY], &[VarDomain::Integer], &cons).unwrap();
    assert!(pre.constraints.is_empty());
    assert_eq!(pre.var_maxs[0], 5.0);
}

#[test]
fn singleton_row_stays_when_a_bound_would_hold_it_looser() {
    // A continuous x keeps its bounds, so 2x <= 10 stays a row over x >= 0,
    // and over [0, 5] too, where x may pass 5 by its tolerance, 1e-7, off
    // the row by 2e-7, beyond the contract. Over [0, 4.9] the bounds imply
    // the row, which goes.
    let cons = vec![(csvec(1, &[(0, 2.0)]), ComparisonOp::Le, 10.0)];
    for hi in [f64::INFINITY, 5.0] {
        let pre = run_lp(&[0.0], &[hi], &cons).unwrap();
        assert_eq!(pre.constraints.len(), 1);
        assert_eq!(pre.var_maxs[0], hi);
    }
    let pre = run_lp(&[0.0], &[4.9], &cons).unwrap();
    assert!(pre.constraints.is_empty());
    assert_eq!(pre.var_maxs[0], 4.9);
}

#[test]
fn singleton_eq_fixes_var_and_substitutes() {
    // x == 3 over an integer x; x + y <= 10 -> x fixed; the second row
    // (`y <= 7` to presolve) stays, as the continuous y keeps its bounds, and
    // is emitted as written, fixed term included.
    let cons = vec![
        (csvec(2, &[(0, 1.0)]), ComparisonOp::Eq, 3.0),
        (csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 10.0),
    ];
    let pre = presolve_mip(
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[VarDomain::Integer, VarDomain::Real],
        &cons,
    )
    .unwrap();
    assert_eq!(pre.var_mins[0], 3.0);
    assert_eq!(pre.var_maxs[0], 3.0);
    assert_eq!(pre.constraints.len(), 1);
    assert!(same_row(&pre.constraints[0], &cons[1]));
    assert_eq!(pre.var_maxs[1], f64::INFINITY);
}

#[test]
fn redundant_row_is_dropped() {
    // x + y <= 10 with both vars in [0, 3]: implied by bounds, dropped.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 10.0)];
    let pre = run_lp(&[0.0, 0.0], &[3.0, 3.0], &cons).unwrap();
    assert!(pre.constraints.is_empty());
}

#[test]
fn forcing_ge_row_fixes_all_vars() {
    // x + y >= 10 with integers x, y <= 5: only x = y = 5 works.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 10.0)];
    let pre = presolve_mip(
        &[0.0, 0.0],
        &[5.0, 5.0],
        &[VarDomain::Integer, VarDomain::Integer],
        &cons,
    )
    .unwrap();
    assert!(pre.constraints.is_empty());
    assert_eq!((pre.var_mins[0], pre.var_maxs[0]), (5.0, 5.0));
    assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (5.0, 5.0));
}

#[test]
fn activity_infeasibility_is_detected() {
    // x + y >= 11 with x, y <= 5: max activity 10 < 11.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 11.0)];
    assert_eq!(
        run_lp(&[0.0, 0.0], &[5.0, 5.0], &cons).unwrap_err(),
        Error::Infeasible
    );
}

#[test]
fn crossed_input_bounds_are_infeasible() {
    let cons: Vec<(CsVec, ComparisonOp, f64)> = vec![];
    assert_eq!(
        run_lp(&[1.0], &[0.0], &cons).unwrap_err(),
        Error::Infeasible
    );
}

#[test]
fn user_empty_rows_keep_solver_semantics() {
    // 0 <= 1 tautological, dropped; 0 >= 1 infeasible.
    let taut = vec![(csvec(1, &[]), ComparisonOp::Le, 1.0)];
    let pre = run_lp(&[0.0], &[1.0], &taut).unwrap();
    assert!(pre.constraints.is_empty());
    let bad = vec![(csvec(1, &[]), ComparisonOp::Ge, 1.0)];
    assert_eq!(run_lp(&[0.0], &[1.0], &bad).unwrap_err(), Error::Infeasible);
}

#[test]
fn dual_fixing_pins_costly_var_to_lower_bound() {
    // minimize x, x in [2, 9], only occurrence x + y <= 100 (a > 0 in a
    // <=-row never blocks decreasing) -> fixed at 2. Gated on mode+flag.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 100.0)];
    let doms = vec![VarDomain::Integer, VarDomain::Integer];
    let fixed = presolve(
        &[1.0, 0.0],
        &[2.0, 0.0],
        &[9.0, 5.0],
        &cons,
        &doms,
        Mode::Mip,
        1e-7,
        1e-6,
        true,
    )
    .unwrap();
    assert_eq!((fixed.var_mins[0], fixed.var_maxs[0]), (2.0, 2.0));
    // y has c == 0 and is only in a <=-row: dual-fixable at its lower bound too.
    assert_eq!((fixed.var_mins[1], fixed.var_maxs[1]), (0.0, 0.0));

    let hinted = presolve(
        &[1.0, 0.0],
        &[2.0, 0.0],
        &[9.0, 5.0],
        &cons,
        &doms,
        Mode::Mip,
        1e-7,
        1e-6,
        false, // warm hint present -> no dual fixing
    )
    .unwrap();
    assert_eq!((hinted.var_mins[0], hinted.var_maxs[0]), (2.0, 9.0));
}

#[test]
fn dual_fixing_blocked_by_eq_row_and_infinite_bound() {
    // x in an Eq row: both directions blocked; nothing may be fixed here.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Eq, 5.0)];
    let pre = presolve(
        &[1.0, -1.0],
        &[0.0, 0.0],
        &[9.0, 9.0],
        &cons,
        &[VarDomain::Integer, VarDomain::Integer],
        Mode::Mip,
        1e-7,
        1e-6,
        true,
    )
    .unwrap();
    assert!(pre.var_mins[0] < pre.var_maxs[0]);
    assert!(pre.var_mins[1] < pre.var_maxs[1]);

    // minimize x with lo = -inf and no rows: possibly unbounded, must be
    // left for the simplex (no fixing, no error).
    let none: Vec<(CsVec, ComparisonOp, f64)> = vec![];
    let pre = presolve(
        &[1.0],
        &[f64::NEG_INFINITY],
        &[f64::INFINITY],
        &none,
        &[VarDomain::Integer],
        Mode::Mip,
        1e-7,
        1e-6,
        true,
    )
    .unwrap();
    assert_eq!(pre.var_mins[0], f64::NEG_INFINITY);
}

#[test]
fn binary_coefficient_tightening_le_row() {
    // 10 x + y <= 12, x binary, y in [0, 5]: x=1 forces y <= 2 and x=0 is
    // vacuous (5 <= 12) -> becomes (about) 3x + y <= 5. Integer content
    // identical, LP relaxation strictly tighter.
    let cons = vec![(csvec(2, &[(0, 10.0), (1, 1.0)]), ComparisonOp::Le, 12.0)];
    let pre = presolve(
        &[0.0, -1.0],
        &[0.0, 0.0],
        &[1.0, 5.0],
        &cons,
        &[VarDomain::Boolean, VarDomain::Real],
        Mode::Mip,
        1e-7,
        1e-6,
        false,
    )
    .unwrap();
    assert_eq!(pre.constraints.len(), 1);
    let (coeffs, op, rhs) = &pre.constraints[0];
    assert!(matches!(op, ComparisonOp::Le));
    let a_x = coeffs.get(0).copied().unwrap();
    let a_y = coeffs.get(1).copied().unwrap();
    assert_eq!(a_y, 1.0);
    assert!((a_x - 3.0).abs() < 1e-6, "10 -> ~3, got {}", a_x);
    assert!((rhs - 5.0).abs() < 1e-6, "12 -> ~5, got {}", rhs);
    // The binding branch is preserved exactly: rhs - a_x == 12 - 10.
    assert!((rhs - a_x - 2.0).abs() < 1e-9);
    assert_eq!(pre.stats.coeffs_tightened, 1);
}

#[test]
fn binary_coefficient_tightening_ge_row() {
    // -10 x + y >= -8, x binary, y in [0, 5]: x=1 forces y >= 2, x=0
    // vacuous (0 >= -8) -> lo' = rest_l = 0, a' = a + (rest_l - lo) =
    // -10 + 8 = -2: becomes -2x + y >= 0.
    let cons = vec![(csvec(2, &[(0, -10.0), (1, 1.0)]), ComparisonOp::Ge, -8.0)];
    let pre = presolve(
        &[0.0, 1.0],
        &[0.0, 0.0],
        &[1.0, 5.0],
        &cons,
        &[VarDomain::Boolean, VarDomain::Real],
        Mode::Mip,
        1e-7,
        1e-6,
        false,
    )
    .unwrap();
    assert_eq!(pre.constraints.len(), 1);
    let (coeffs, op, rhs) = &pre.constraints[0];
    assert!(matches!(op, ComparisonOp::Ge));
    let a_x = coeffs.get(0).copied().unwrap();
    assert!((a_x + 2.0).abs() < 1e-6, "-10 -> ~-2, got {}", a_x);
    assert!(rhs.abs() < 1e-6, "-8 -> ~0, got {}", rhs);
    assert!(
        (rhs - a_x - 2.0).abs() < 1e-9,
        "x=1 branch must stay y >= 2"
    );
}

#[test]
fn knapsack_coefficient_tightening_shrinks_loose_coeff() {
    // 5x + 7y + 4z + 3w <= 14, binaries. y's coefficient is loose: when
    // y = 0 the rest can reach at most 12 < 14, so the row is equivalent
    // to 5x + 5y + 4z + 3w <= 12 on binary points (y = 1 still forces
    // rest <= 7). x/z/w after the update: rest_u == b, so they stay.
    let cons = vec![(
        csvec(4, &[(0, 5.0), (1, 7.0), (2, 4.0), (3, 3.0)]),
        ComparisonOp::Le,
        14.0,
    )];
    let pre = presolve(
        &[-8.0, -11.0, -6.0, -4.0],
        &[0.0; 4],
        &[1.0; 4],
        &cons,
        &vec![VarDomain::Boolean; 4],
        Mode::Mip,
        1e-7,
        1e-6,
        true,
    )
    .unwrap();
    assert_eq!(pre.stats.coeffs_tightened, 1);
    assert_eq!(pre.constraints.len(), 1);
    let (coeffs, _, rhs) = &pre.constraints[0];
    assert!((rhs - 12.0).abs() < 1e-6, "14 -> ~12, got {}", rhs);
    assert_eq!(coeffs.get(0).copied().unwrap(), 5.0);
    let a_y = coeffs.get(1).copied().unwrap();
    assert!((a_y - 5.0).abs() < 1e-6, "7 -> ~5, got {}", a_y);
    assert_eq!(coeffs.get(2).copied().unwrap(), 4.0);
    assert_eq!(coeffs.get(3).copied().unwrap(), 3.0);
    // y = 1 branch preserved exactly: rhs - a_y == 14 - 7.
    assert!((rhs - a_y - 7.0).abs() < 1e-9);
}

#[test]
fn a_big_m_rewrite_keeps_the_contract_of_its_binding_branch() {
    // 100 x + 5e7 y <= 5e7 + 500 over x in [0, 10] and a binary y: at y = 1
    // the row's contract is its round-off at a magnitude of 1e8, about 1e-6,
    // and the rewritten row, near 100 x + 500 y <= 1000, holds to the
    // absolute 1e-7. At y = 1 the rewritten row must accept an x the row as
    // written accepts within its contract, and reject one beyond it: an
    // excess 100 x - 500 of 0.95e-6 and of 1.05e-6.
    use crate::solver::row_violation;
    let cons = vec![(
        csvec(2, &[(0, 100.0), (1, 5e7)]),
        ComparisonOp::Le,
        5e7 + 500.0,
    )];
    let pre = presolve_mip(
        &[0.0, 0.0],
        &[10.0, 1.0],
        &[VarDomain::Real, VarDomain::Boolean],
        &cons,
    )
    .unwrap();
    assert_eq!(pre.stats.coeffs_tightened, 1);
    let accepts = |(coeffs, op, rhs): &(CsVec, ComparisonOp, f64), excess: f64| {
        let x = 5.0 + excess / 100.0;
        row_violation(coeffs, *op, *rhs, |v| [x, 1.0][v], 1e-7).is_none()
    };
    let rewritten = &pre.constraints[0];
    assert!(rewritten.0.get(1).is_some_and(|&a| a < 1e3));
    assert!(accepts(&cons[0], 0.95e-6) && accepts(rewritten, 0.95e-6));
    assert!(!accepts(&cons[0], 1.05e-6) && !accepts(rewritten, 1.05e-6));
}

#[test]
fn a_big_m_rewrite_keeps_the_contract_its_fixed_terms_give_its_binding_branch() {
    // 100 x + 5e7 w1 - 5e7 w2 + 2.5e6 y <= 2.5e6 + 500 over x in [0, 10],
    // w1 and w2 fixed at 1 and a binary y: at y = 1 the row's contract is the
    // round-off of a magnitude its fixed terms make 1.05e8, where its
    // rewritten terms alone, as written or rewritten, are held to the
    // absolute 1e-7. At y = 1 the rewritten row, fixed terms restored, must
    // accept an excess 100 x - 500 of 1.03e-6, as the row as written does,
    // and reject one of 1.07e-6.
    use crate::solver::row_violation;
    let cons = vec![(
        csvec(4, &[(0, 100.0), (1, 5e7), (2, -5e7), (3, 2.5e6)]),
        ComparisonOp::Le,
        2.5e6 + 500.0,
    )];
    let domains = [
        VarDomain::Real,
        VarDomain::Real,
        VarDomain::Real,
        VarDomain::Boolean,
    ];
    let pre = presolve_mip(
        &[0.0, 1.0, 1.0, 0.0],
        &[10.0, 1.0, 1.0, 1.0],
        &domains,
        &cons,
    )
    .unwrap();
    assert_eq!(pre.stats.coeffs_tightened, 1);
    let accepts = |(coeffs, op, rhs): &(CsVec, ComparisonOp, f64), excess: f64| {
        let x = 5.0 + excess / 100.0;
        row_violation(coeffs, *op, *rhs, |v| [x, 1.0, 1.0, 1.0][v], 1e-7).is_none()
    };
    let rewritten = &pre.constraints[0];
    assert!(rewritten.0.get(3).is_some_and(|&a| a < 1e3));
    assert!(accepts(&cons[0], 1.03e-6) && accepts(rewritten, 1.03e-6));
    assert!(!accepts(&cons[0], 1.07e-6) && !accepts(rewritten, 1.07e-6));
}

/// Presolve the big-M row `100 x + Σ m_i y_i + k w <= Σ m_i + 500 + k` over
/// `x ∈ [0, 10]`, binaries `y_i` of cost -1 and an integer `w ∈ [1, 2]` of
/// cost 1, and check it at `y = w = 1` with `x` at the excess `100 x - 500`
/// of `share` of the row's contract there, for `share` `0.45` and `1.05`:
/// the reduced model keeps the first within the engine's budget, as the row
/// as written does, and its rows reject the second within their contract.
/// Returns the reduction counters.
fn big_m_rewrites_keep_the_budget(ms: &[f64], k: f64) -> PresolveStats {
    use crate::solver::row_violation;
    let n = ms.len() + 2;
    let mut terms = vec![(0, 100.0), (n - 1, k)];
    terms.extend(ms.iter().enumerate().map(|(i, &m)| (i + 1, m)));
    let total: f64 = ms.iter().sum();
    let cons = vec![(csvec(n, &terms), ComparisonOp::Le, total + 500.0 + k)];
    let mut obj = vec![-1.0; n];
    (obj[0], obj[n - 1]) = (0.0, 1.0);
    let mut domains = vec![VarDomain::Boolean; n];
    (domains[0], domains[n - 1]) = (VarDomain::Real, VarDomain::Integer);
    let (mut mins, mut maxs) = (vec![0.0; n], vec![1.0; n]);
    (mins[n - 1], maxs[0], maxs[n - 1]) = (1.0, 10.0, 2.0);
    let pre = presolve(
        &obj,
        &mins,
        &maxs,
        &cons,
        &domains,
        Mode::Mip,
        1e-7,
        1e-6,
        true,
    )
    .unwrap();
    let contract = 1e-14 * (2.0 * (total + k) + 1000.0);
    let at = |share: f64| {
        let mut x = vec![1.0; n];
        x[0] = 5.0 + share * contract / 100.0;
        x
    };
    assert!(keeps(&pre, &cons, &at(0.45)), "fixture: within the budget");
    assert!(keeps(&pre, &pre.constraints, &at(0.45)));
    let beyond = at(1.05);
    for (coeffs, op, rhs) in cons.iter().chain(&pre.constraints) {
        assert!(row_violation(coeffs, *op, *rhs, |v| beyond[v], 1e-7).is_some());
    }
    pre.stats
}

#[test]
fn a_big_m_rewrite_counts_the_terms_fixed_after_it() {
    // With m = 3.75e6 and k = 2.5e6: the row's contract at y = w = 1 is
    // 1.25e-7 at a magnitude of 1.25e7. y's coefficient is rewritten, then
    // dual fixing fixes w at 1, and y's is rewritten again with w's term
    // fixed, which alone makes the contract more than the absolute 1e-7.
    let stats = big_m_rewrites_keep_the_budget(&[3.75e6], 2.5e6);
    assert_eq!((stats.coeffs_tightened, stats.vars_fixed), (2, 1));
}

#[test]
fn a_big_m_rewrite_of_two_binaries_counts_both() {
    // With two binaries at m = 3.5e6 and w's term k = 1: the row's contract
    // at y1 = y2 = w = 1 is 1.4e-7, where each binary's rewrite alone sees
    // the magnitudes of 1.05e7 and 7e6, of contracts 1.05e-7 and 1e-7.
    let stats = big_m_rewrites_keep_the_budget(&[3.5e6, 3.5e6], 1.0);
    assert_eq!(stats.coeffs_tightened, 2);
}

#[test]
fn a_big_m_rewrite_keeps_its_binding_branch_to_the_bit() {
    // 100 x + m y + k1 w1 + k2 w2 <= m + 500 + k1 + k2, and the same row
    // negated as a Ge row, over x in [0, 10], a binary y and w1, w2 fixed at
    // 1. At y = 1 the rewritten row leaves the rest `rhs' - a'`, which must
    // be the row as written's `rhs - m` relaxed by what the rewrite keeps,
    // the bound rounded outward: exactly, so never tighter. With the first
    // fixed terms both contracts are the absolute 1e-7 and the rewrite keeps
    // nothing, so it is also looser by less than an ulp of `rhs'`; with the
    // second the contract at y = 1 is the round-off of 1.05e8, and the
    // rewrite keeps what the rewritten magnitude loses of it, in either form.
    use crate::solver::CDouble;
    let domains = [
        VarDomain::Real,
        VarDomain::Boolean,
        VarDomain::Real,
        VarDomain::Real,
    ];
    let cases = [
        (
            1805051.3787215224,
            [469042.9001025199, -4706346.9724488305],
            true,
        ),
        (2.5e6, [5e7, -5e7], false),
    ];
    for (m, [k1, k2], keeps_nothing) in cases {
        for (sign, op) in [(1.0, ComparisonOp::Le), (-1.0, ComparisonOp::Ge)] {
            let written = [(0, 100.0), (1, m), (2, k1), (3, k2)].map(|(v, a)| (v, sign * a));
            let rhs = sign * (m + 500.0 + (k1 + k2));
            let cons = vec![(csvec(4, &written), op, rhs)];
            let pre = presolve_mip(
                &[0.0, 0.0, 1.0, 1.0],
                &[10.0, 1.0, 1.0, 1.0],
                &domains,
                &cons,
            )
            .unwrap();
            assert_eq!(pre.stats.coeffs_tightened, 1);
            let (coeffs, _, emitted) = &pre.constraints[0];
            let rewritten = *coeffs.get(1).unwrap();
            assert!(rewritten.abs() < 1e3, "m -> about 500, got {rewritten}");
            // How much looser the rewritten branch is, `(rhs' - a') - (rhs - m)`
            // on the side the row bounds.
            let mut looser = CDouble::from(sign * emitted);
            looser.add(-sign * rewritten);
            looser.add(-sign * rhs);
            looser.add(m);
            assert!(
                looser.rounded(false) >= 0.0,
                "{op:?}: {emitted} tightens the branch"
            );
            let ulp = emitted.abs().next_up() - emitted.abs();
            assert!(
                !keeps_nothing || looser.rounded(true) <= ulp,
                "{op:?}: {emitted} is more than an ulp looser"
            );
        }
    }
}

#[test]
fn a_rewritten_row_whose_bound_its_rewrites_keep_is_emitted_with_its_bound_as_written() {
    // x - 10 y + k1 w1 + k2 w2 <= 5 + k1 + k2 over x in [0, 10], a binary y
    // and w1, w2 fixed at 1: at y = 1 the row is vacuous, so tightening
    // shrinks y's coefficient and keeps the bound. Folded past the fixed
    // terms and back in `f64`, that bound comes out an ulp below
    // `5 + k1 + k2`; the rewritten row keeps it as written.
    let (k1, k2) = (4151527.32, 4323263.19);
    let written = 5.0 + k1 + k2;
    let cons = vec![(
        csvec(4, &[(0, 1.0), (1, -10.0), (2, k1), (3, k2)]),
        ComparisonOp::Le,
        written,
    )];
    let domains = [
        VarDomain::Real,
        VarDomain::Boolean,
        VarDomain::Real,
        VarDomain::Real,
    ];
    let pre = presolve_mip(
        &[0.0, 0.0, 1.0, 1.0],
        &[10.0, 1.0, 1.0, 1.0],
        &domains,
        &cons,
    )
    .unwrap();
    assert_eq!(pre.stats.coeffs_tightened, 1);
    let (coeffs, op, rhs) = &pre.constraints[0];
    assert!(coeffs.get(1).is_some_and(|&a| a > -10.0 && a < 0.0));
    assert!(matches!(op, ComparisonOp::Le));
    assert_eq!(rhs.to_bits(), written.to_bits(), "{rhs} against {written}");
}

#[test]
fn lp_mode_never_rounds_or_dual_fixes() {
    // Same shape as the dual-fixing test but Mode::Lp: bounds must come
    // out exactly as they went in.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 100.0)];
    let pre = presolve(
        &[1.0, 0.0],
        &[0.3, 0.0],
        &[9.7, 5.0],
        &cons,
        &real(2),
        Mode::Lp,
        1e-7,
        1e-6,
        true, // even with the flag on, Lp mode must not dual-fix
    )
    .unwrap();
    assert_eq!(pre.var_mins, vec![0.3, 0.0]);
    assert_eq!(pre.var_maxs, vec![9.7, 5.0]);
}

#[test]
fn cascading_substitution_reaches_fixpoint() {
    // Integers x, y and a continuous z: x == 2 (singleton Eq), x + y == 5
    // -> y == 3, y + z <= 4 -> `z <= 1` to presolve, kept as the row it was
    // (z keeps its bounds).
    let cons = vec![
        (csvec(3, &[(0, 1.0)]), ComparisonOp::Eq, 2.0),
        (csvec(3, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Eq, 5.0),
        (csvec(3, &[(1, 1.0), (2, 1.0)]), ComparisonOp::Le, 4.0),
    ];
    let domains = [VarDomain::Integer, VarDomain::Integer, VarDomain::Real];
    let pre = presolve_mip(&[0.0; 3], &[f64::INFINITY; 3], &domains, &cons).unwrap();
    assert_eq!((pre.var_mins[0], pre.var_maxs[0]), (2.0, 2.0));
    assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (3.0, 3.0));
    assert_eq!(pre.constraints.len(), 1);
    assert!(same_row(&pre.constraints[0], &cons[2]));
    assert_eq!(pre.var_maxs[2], f64::INFINITY);
}

#[test]
fn all_substituted_row_is_judged_at_the_full_contract() {
    // A row whose vars are all fixed is judged as the boundary judges it,
    // against the full contract, not the engine's half: at x0 = -2^-30 and
    // x1 = 83887, `-100 x0 + x1 / 1000 = 83.887` has a residual of 9.3e-8,
    // within 1e-7 but not 5e-8.
    let x0 = -(2.0f64.powi(-30));
    let cons = vec![
        (
            csvec(2, &[(0, -1.0), (1, -100.0)]),
            ComparisonOp::Eq,
            -8388700.0,
        ),
        (
            csvec(2, &[(0, -100.0), (1, 1e-3)]),
            ComparisonOp::Eq,
            83.887,
        ),
    ];
    let pre = run_lp(&[x0, 83887.0], &[x0, 83887.0], &cons).unwrap();
    assert!(pre.constraints.is_empty());
}

#[test]
fn inconsistent_substitution_is_infeasible() {
    // x == 2 and x == 3 over an integer x via two singleton Eq rows.
    let cons = vec![
        (csvec(1, &[(0, 1.0)]), ComparisonOp::Eq, 2.0),
        (csvec(1, &[(0, 1.0)]), ComparisonOp::Eq, 3.0),
    ];
    assert_eq!(
        presolve_mip(&[0.0], &[f64::INFINITY], &[VarDomain::Integer], &cons).unwrap_err(),
        Error::Infeasible
    );
}

#[test]
fn infinite_rhs_le_row_is_dropped_not_crashed() {
    let cons = vec![(
        csvec(2, &[(0, 1.0), (1, 1.0)]),
        ComparisonOp::Le,
        f64::INFINITY,
    )];
    let pre = run_lp(&[0.0, 0.0], &[1.0, 1.0], &cons).unwrap();
    assert!(pre.constraints.is_empty());
}

/// Whether the reduced model keeps a point the engine accepts at `xstar`:
/// `xstar`, with each var `pre` fixed moved to its value, lies within every
/// emitted bound's tolerance and meets every input row within the engine's
/// budget. A fixing whose error another row amplifies past its budget, or a
/// bound that cuts `xstar`, fails it.
fn keeps(pre: &Presolved, cons: &[(CsVec, ComparisonOp, f64)], xstar: &[f64]) -> bool {
    use crate::solver::{outside, outside_bounds, row_tolerance, slack_bounds, ROW_BUDGET_SHARE};
    let (mins, maxs) = (&pre.var_mins, &pre.var_maxs);
    let x: Vec<f64> = (0..xstar.len())
        .map(|v| {
            if mins[v] == maxs[v] {
                mins[v]
            } else {
                xstar[v]
            }
        })
        .collect();
    let in_bounds = (0..x.len()).all(|v| !outside_bounds(x[v], mins[v], maxs[v], 1e-7));
    let rows_hold = cons.iter().all(|(coeffs, op, rhs)| {
        let (mut activity, mut magnitude) = (0.0, rhs.abs());
        for (v, &a) in coeffs.iter() {
            activity += a * x[v];
            magnitude += (a * x[v]).abs();
        }
        let (lo, hi) = slack_bounds(*op);
        let budget = ROW_BUDGET_SHARE * row_tolerance(1e-7, magnitude);
        !outside(rhs - activity, lo, hi, budget)
    });
    in_bounds && rows_hold
}

fn presolve_mip(
    mins: &[f64],
    maxs: &[f64],
    domains: &[VarDomain],
    cons: &[(CsVec, ComparisonOp, f64)],
) -> Result<Presolved, Error> {
    presolve(
        &vec![0.0; mins.len()],
        mins,
        maxs,
        cons,
        domains,
        Mode::Mip,
        1e-7,
        1e-6,
        false,
    )
}

#[test]
fn a_fixed_values_error_is_not_amplified_into_a_bound() {
    // Row 2 pins x2 at -1.5e-12, the round-off of solving it at the fixed
    // x1 and x4, where `-x2 - 1e-5 x3 <= 0` gives `x3 >= 1.5e-7`, beyond the
    // contract of `-x2 + x3 = 0`: as a fixing and a bound, the two rows
    // would cut x*, which presolve, taking no continuous var's room, keeps.
    let cons = vec![
        (csvec(5, &[(2, -1.0), (3, -1e-5)]), ComparisonOp::Le, 0.0),
        (csvec(5, &[(2, -1.0), (3, 1.0)]), ComparisonOp::Eq, 0.0),
        (
            csvec(5, &[(1, 1.0), (2, -1.0), (4, -0.1)]),
            ComparisonOp::Eq,
            -16384.1,
        ),
    ];
    let pre = run_lp(
        &[0.0, -16384.0, -1.0, -1.0, 1.0],
        &[0.0, -16384.0, 0.0, 1.0, 1.0],
        &cons,
    )
    .unwrap();
    assert!(keeps(&pre, &cons, &[0.0, -16384.0, 0.0, 0.0, 1.0]));
}

#[test]
fn a_forcing_row_fixes_no_var_another_row_resolves_better() {
    // x1 - x3 = -524288 is forcing at x3's bound of 2^-34, and so is
    // -x0 - 100 x1 = -8388700 at x0's of -2^-30, but the corner misses the
    // point (x3 = 0, x0 = 0) where every row holds, by more than the var's
    // coefficient of 1000 in its other row lets that row absorb: presolve
    // fixes neither continuous var.
    let cons = vec![
        (
            csvec(6, &[(1, 1.0), (3, -1.0)]),
            ComparisonOp::Eq,
            -524288.0,
        ),
        (csvec(6, &[(3, -1000.0), (4, 1.0)]), ComparisonOp::Eq, 0.0),
    ];
    let pre = run_lp(
        &[0.0, -524288.0, 0.0, -1.0, -1.0, 0.0],
        &[0.0, -524287.0, 0.0, 2.0f64.powi(-34), 0.0, 0.0],
        &cons,
    )
    .unwrap();
    assert!(keeps(&pre, &cons, &[0.0, -524288.0, 0.0, 0.0, 0.0, 0.0]));

    let cons = vec![
        (
            csvec(2, &[(0, -1.0), (1, -100.0)]),
            ComparisonOp::Eq,
            -8388700.0,
        ),
        (
            csvec(2, &[(0, -1000.0), (1, 1e-3)]),
            ComparisonOp::Eq,
            83.887,
        ),
    ];
    let pre = run_lp(&[-(2.0f64.powi(-30)), 83887.0], &[1.0, 83888.0], &cons).unwrap();
    assert!(keeps(&pre, &cons, &[0.0, 83887.0]));
}

#[test]
fn a_forcing_row_counts_its_round_off_as_room() {
    // Forcing rows whose round-off at their magnitude leaves a var room the
    // corner does not show: `1e-7 x4` at 1e9 and `2e-8 x3` at 5e8 (rooms of
    // about a hundred units), `x2`'s `7.45e-9` at 6.7e7 (beyond its
    // tolerance, and twenty times that in the other row), and a unit of
    // `1.375e-4 x1` at 1.4e10, where the contract accepts the next integer.
    // Fixed at the corner, each var would contradict the model's other row.
    // The third model is all integer, so its room rule alone keeps `x1`
    // unfixed; the others' forcing rows hold a continuous var, so presolve
    // fixes nothing through them.
    use VarDomain::{Integer as I, Real as R};
    let cases: Vec<(Vec<f64>, Vec<f64>, Vec<VarDomain>, Vec<_>, Vec<f64>)> = vec![
        (
            vec![
                0.0,
                -1.0,
                -3.5762786865234375e-7,
                562122509.5000763,
                -1.0,
                0.0,
            ],
            vec![
                0.0,
                0.0,
                -3.5762786865234375e-7,
                562122509.5000763,
                214.0,
                0.0,
            ],
            vec![I, I, R, R, R, R],
            vec![
                (
                    csvec(6, &[(3, -1.0), (4, -1.0)]),
                    ComparisonOp::Eq,
                    -562122509.5000763,
                ),
                (
                    csvec(6, &[(1, -1.0), (2, -1.0), (3, -1.91015625), (4, 1e-7)]),
                    ComparisonOp::Eq,
                    -1073741824.7872548,
                ),
            ],
            vec![
                0.0,
                0.0,
                -3.5762786865234375e-7,
                562122509.5000763,
                0.0,
                0.0,
            ],
        ),
        (
            vec![536871.0, 0.0, -1.0, -532.0, 0.0],
            vec![536871.0, 0.0, 0.0, 5.0, 0.0],
            vec![I, R, R, R, R],
            vec![
                (csvec(5, &[(2, -1.0), (3, -1.0)]), ComparisonOp::Eq, -2.0),
                (
                    csvec(5, &[(0, -1000.0), (2, -0.001), (3, 2e-8)]),
                    ComparisonOp::Eq,
                    -536870999.9989998,
                ),
            ],
            vec![536871.0, 0.0, -1.0, 3.0, 0.0],
        ),
        (
            vec![-2.0, -1.0, 681878.0],
            vec![3.0, 2.0, 681879.0],
            vec![I, I, I],
            vec![
                (csvec(3, &[(0, -1.0), (1, 1.5)]), ComparisonOp::Eq, 0.0),
                (
                    csvec(3, &[(1, 0.0001375), (2, 20000.0)]),
                    ComparisonOp::Eq,
                    13637560000.0,
                ),
            ],
            vec![0.0, 0.0, 681878.0],
        ),
        (
            vec![0.0, 3355444.0, -7.450580596923828e-9],
            vec![0.0, 3355445.0, 1.0],
            vec![I, R, R],
            vec![
                (
                    csvec(3, &[(1, -20.0), (2, -1.0)]),
                    ComparisonOp::Eq,
                    -67108880.0,
                ),
                (
                    csvec(3, &[(1, 1.0), (2, -20.0)]),
                    ComparisonOp::Eq,
                    3355444.0,
                ),
            ],
            vec![0.0, 3355444.0, 0.0],
        ),
    ];
    for (i, (mins, maxs, domains, cons, xstar)) in cases.iter().enumerate() {
        let pre = presolve_mip(mins, maxs, domains, cons)
            .unwrap_or_else(|err| panic!("model {i}: presolve returned {err:?}"));
        assert!(keeps(&pre, cons, xstar), "model {i}");
    }
}
