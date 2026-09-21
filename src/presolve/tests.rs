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
    // 0.5x <= 2.5  ->  x <= 5, row gone: the engine may leave the bound
    // by its tolerance (1e-7), and with a coefficient of 0.5 the row,
    // held to half of 1e-7, still holds there.
    let cons = vec![(csvec(1, &[(0, 0.5)]), ComparisonOp::Le, 2.5)];
    let pre = run_lp(&[0.0], &[f64::INFINITY], &cons).unwrap();
    assert!(pre.constraints.is_empty());
    assert_eq!(pre.var_maxs[0], 5.0);
}

#[test]
fn singleton_row_stays_when_a_bound_would_hold_it_looser() {
    // 2x <= 10 as the bound x <= 5 would let the engine return
    // x = 5 + 1e-7, off the row by 2e-7: beyond the contract. The row
    // stays and the bound is not emitted.
    let cons = vec![(csvec(1, &[(0, 2.0)]), ComparisonOp::Le, 10.0)];
    let pre = run_lp(&[0.0], &[f64::INFINITY], &cons).unwrap();
    assert_eq!(pre.constraints.len(), 1);
    assert_eq!(pre.var_maxs[0], f64::INFINITY);
}

#[test]
fn singleton_row_stays_when_its_round_off_would_break_another_row() {
    // With x1 fixed at 1, `x1 - 1e-7 x2 = 0.9999999` pins x2 at
    // 1 - 5e-10: exact to the row's own round-off (5e-17), but then
    // `-200 x2 = -200` misses by 1e-7. The tiny row is kept, the -200
    // row fixes x2 = 1 and the tiny row folds to a 5e-17 residual: the
    // point the simplex finds when it balances both rows.
    let cons = vec![
        (
            csvec(2, &[(0, 1.0), (1, -1e-7)]),
            ComparisonOp::Eq,
            0.9999999,
        ),
        (csvec(2, &[(1, -200.0)]), ComparisonOp::Eq, -200.0),
    ];
    let pre = run_lp(&[1.0, 0.0], &[1.0, 1.0], &cons).unwrap();
    assert!(pre.constraints.is_empty());
    assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (1.0, 1.0));
}

#[test]
fn singleton_eq_fixes_var_and_substitutes() {
    // x == 3; x + y <= 10  ->  x fixed; the second row, `y <= 7` to
    // presolve, stays (a unit coefficient cannot become a bound, see
    // `singleton_row_stays_when_a_bound_would_hold_it_looser`) and is
    // emitted as written, fixed term included.
    let cons = vec![
        (csvec(2, &[(0, 1.0)]), ComparisonOp::Eq, 3.0),
        (csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 10.0),
    ];
    let pre = run_lp(&[0.0, 0.0], &[f64::INFINITY, f64::INFINITY], &cons).unwrap();
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
    // x + y >= 10 with x, y <= 5: only x = y = 5 works.
    let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 10.0)];
    let pre = run_lp(&[0.0, 0.0], &[5.0, 5.0], &cons).unwrap();
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
        &real(2),
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
        &real(1),
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
    // x == 2 (singleton Eq), x + y == 5 -> y == 3, y + z <= 4 -> `z <= 1`
    // to presolve, kept as the row it was (a fixing is exact, a
    // unit-coefficient bound is not).
    let cons = vec![
        (csvec(3, &[(0, 1.0)]), ComparisonOp::Eq, 2.0),
        (csvec(3, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Eq, 5.0),
        (csvec(3, &[(1, 1.0), (2, 1.0)]), ComparisonOp::Le, 4.0),
    ];
    let pre = run_lp(&[0.0; 3], &[f64::INFINITY; 3], &cons).unwrap();
    assert_eq!((pre.var_mins[0], pre.var_maxs[0]), (2.0, 2.0));
    assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (3.0, 3.0));
    assert_eq!(pre.constraints.len(), 1);
    assert!(same_row(&pre.constraints[0], &cons[2]));
    assert_eq!(pre.var_maxs[2], f64::INFINITY);
}

#[test]
fn inconsistent_substitution_is_infeasible() {
    // x == 2 and x == 3 via two singleton Eq rows.
    let cons = vec![
        (csvec(1, &[(0, 1.0)]), ComparisonOp::Eq, 2.0),
        (csvec(1, &[(0, 1.0)]), ComparisonOp::Eq, 3.0),
    ];
    assert_eq!(
        run_lp(&[0.0], &[f64::INFINITY], &cons).unwrap_err(),
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

#[test]
fn forcing_row_is_not_applied_to_a_term_below_its_roundoff() {
    // -x0 - 1e-7·x1 = -2^28 with x0 in [2^28 - 1, 2^28] and x1 in [0, 0.25]:
    // the minimum activity equals the bound to round-off (the row's round-off
    // is 5e-6, x1's whole term range is 2.5e-8), so the row looks forcing
    // wherever x1 sits. It forces nothing about x1, and fixing x1 at its
    // extreme 0.25 made the second row (-x1 = 0) inconsistent: a feasible
    // model declared Infeasible by presolve alone.
    let big = 268435456.0; // 2^28
    let cons = vec![
        (csvec(2, &[(0, -1.0), (1, -1e-7)]), ComparisonOp::Eq, -big),
        (csvec(2, &[(1, -1.0)]), ComparisonOp::Eq, -0.0),
    ];
    let pre = run_lp(&[big - 1.0, 0.0], &[big, 0.25], &cons).unwrap();
    assert_eq!(
        (pre.var_mins[1], pre.var_maxs[1]),
        (0.0, 0.0),
        "x1 is fixed by its own row"
    );
    assert_eq!(
        (pre.var_mins[0], pre.var_maxs[0]),
        (big, big),
        "x0 follows from the equality"
    );

    // Control: the same shape with a resolvable second term is forcing and
    // fixes both vars at the corner.
    let cons = vec![(
        csvec(2, &[(0, -1.0), (1, -1.0)]),
        ComparisonOp::Eq,
        -(big + 0.25),
    )];
    let pre = run_lp(&[big - 1.0, 0.0], &[big, 0.25], &cons).unwrap();
    assert_eq!((pre.var_mins[0], pre.var_maxs[0]), (big, big));
    assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (0.25, 0.25));
    assert!(pre.constraints.is_empty(), "a forcing row is dropped");
}
