//! Tests reproducing reported issues, each guarding against its bug
//! returning.

#[cfg(test)]
mod regression_tests {
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    /// <https://github.com/Specy/microlp/issues/3>: a huge but *finite* variable
    /// bound (`f64::MAX`, `f32::MAX`, `i64::MAX`, …) must behave like
    /// `f64::INFINITY`. Such a bound used to be seeded into the simplex tableau
    /// as a literal value, which swamped the problem data (the rhs and
    /// coefficients lost all significance against it) and returned a wrong vertex
    /// — e.g. `[2, 0, 0]` (or a NaN objective) instead of `[2, 6.2, 1.6]`.
    #[test]
    fn issue_3_huge_upper_bound_behaves_like_infinity() {
        // Every one of these upper bounds must give the same answer as infinity.
        for upper in [f64::MAX, f32::MAX as f64, i64::MAX as f64, f64::INFINITY] {
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x = problem.add_var(50.0, (2.0, f64::INFINITY));
            let y = problem.add_var(40.0, (0.0, 7.0));
            let z = problem.add_var(45.0, (0.0, upper));
            problem.add_constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);
            problem.add_constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);

            let sol = problem
                .solve()
                .unwrap()
                .into_solution()
                .expect("an unlimited bounded solve must return a solution");

            assert!(
                (sol.var_value(x) - 2.0).abs() < 1e-6,
                "x wrong for upper={upper:e}"
            );
            assert!(
                (sol.var_value(y) - 6.2).abs() < 1e-6,
                "y wrong for upper={upper:e}"
            );
            assert!(
                (sol.var_value(z) - 1.6).abs() < 1e-6,
                "z wrong for upper={upper:e}"
            );
            assert!(
                (sol.objective() - 420.0).abs() < 1e-6,
                "objective wrong for upper={upper:e}: got {}",
                sol.objective()
            );
        }
    }

    /// Issue #3 from the other side: a huge upper bound must not loosen the
    /// LOWER bound. The round-off floor of a var's bound tolerance was taken
    /// from the larger of its bounds, so `x ∈ [0, f64::MAX]` could sit at
    /// `-5` (a tolerance of 1e294 at zero) and an infeasible model came back
    /// Optimal; with an upper bound of infinity it was correctly Infeasible.
    /// Each bound now has the round-off floor of its own magnitude.
    #[test]
    fn issue_3_huge_upper_bound_does_not_loosen_the_lower_bound() {
        for upper in [f64::MAX, f32::MAX as f64, 1e10, f64::INFINITY] {
            for presolve in [true, false] {
                let mut problem = Problem::new(OptimizationDirection::Minimize);
                let x = problem.add_var(1.0, (0.0, upper));
                let y = problem.add_var(1.0, (0.0, 10.0));
                problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Eq, -5.0);
                let got = problem.solve_with(crate::SolveOptions {
                    presolve,
                    ..crate::SolveOptions::default()
                });
                assert_eq!(
                    got.map(|_| "solved"),
                    Err(crate::Error::Infeasible),
                    "upper {upper:e}, presolve {presolve}"
                );
            }
        }
        // And a lower bound of zero is held to within the absolute tolerance,
        // not to the round-off of the upper bound: min y with y ≥ 1 + 1e6 x
        // wants x as low as it may go, and x = -1e-6 (inside a tolerance of
        // 1e-2, the round-off of 1e12) would give y = 0 instead of 1.
        for upper in [f64::MAX, 1e12, f64::INFINITY] {
            for presolve in [true, false] {
                let mut problem = Problem::new(OptimizationDirection::Minimize);
                let x = problem.add_var(0.0, (0.0, upper));
                let y = problem.add_var(1.0, (0.0, 10.0));
                problem.add_constraint([(y, 1.0), (x, -1e6)], ComparisonOp::Ge, 1.0);
                let sol = problem
                    .solve_with(crate::SolveOptions {
                        presolve,
                        ..crate::SolveOptions::default()
                    })
                    .unwrap()
                    .into_solution()
                    .unwrap();
                assert!(
                    sol[x] >= -1e-7 && (sol.objective() - 1.0).abs() < 1e-6,
                    "upper {upper:e}, presolve {presolve}: x = {:e}, objective {}",
                    sol[x],
                    sol.objective()
                );
            }
        }
    }

    /// Issue #3 in the ratio test (C3-16 of the solver review): `max x` s.t.
    /// `-3x + y <= 1`, `0 <= x <= f64::MAX`, `0 <= y <= 1` is unbounded, as
    /// HiGHS reports it. The bound of `f64::MAX` was only kept out of the
    /// starting vertex, so the primal simplex moved `x` onto it, the row's
    /// activity overflowed, and the boundary reported
    /// `InternalError("... violates constraint row 0 by inf")`. A bound at or
    /// beyond 2^52 on its side of infinity is now infinite for the whole
    /// solve.
    #[test]
    fn huge_bound_on_an_improving_ray_is_unbounded() {
        for upper in [f64::MAX, 1e20, f64::INFINITY] {
            for presolve in [true, false] {
                let mut problem = Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(1.0, (0.0, upper));
                let y = problem.add_var(0.0, (0.0, 1.0));
                problem.add_constraint([(x, -3.0), (y, 1.0)], ComparisonOp::Le, 1.0);
                let got = problem.solve_with(crate::SolveOptions {
                    presolve,
                    ..crate::SolveOptions::default()
                });
                assert_eq!(
                    got.map(|_| "solved"),
                    Err(crate::Error::Unbounded),
                    "upper {upper:e}, presolve {presolve}"
                );
            }
        }
    }

    /// The bound infinity is 2^52 (`INFINITE_BOUND`), not HiGHS's default
    /// of 1e20, and this pins what that means on both sides of it: a bound
    /// from 2^52 on is no bound at all, so `max x` with `x ≤ 5e15` is
    /// unbounded (HiGHS at its default returns 5e15) and a solution can lie
    /// beyond such a bound, while a bound below it binds. The reason for the
    /// choice is the next test.
    #[test]
    fn a_bound_from_2_pow_52_is_infinite_and_one_below_binds() {
        let two_52 = 2.0f64.powi(52);
        for presolve in [true, false] {
            let options = || crate::SolveOptions {
                presolve,
                ..crate::SolveOptions::default()
            };
            for upper in [
                two_52,
                5e15,
                1e19,
                i64::MAX as f64,
                two_52 - 1.0,
                4e15,
                1e15,
            ] {
                let mut problem = Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(1.0, (0.0, upper));
                let y = problem.add_var(0.0, (0.0, 1.0));
                problem.add_constraint([(x, -3.0), (y, 1.0)], ComparisonOp::Le, 1.0);
                let got = problem
                    .solve_with(options())
                    .map(|outcome| outcome.into_solution().unwrap()[x]);
                let want = if upper >= two_52 {
                    Err(crate::Error::Unbounded)
                } else {
                    Ok(upper)
                };
                assert_eq!(got, want, "upper {upper:e}, presolve {presolve}");
            }
            // The bound of 2^53 is infinite, so the row alone stops `x`.
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x = problem.add_var(1.0, (0.0, 2.0 * two_52));
            problem.add_constraint([(x, 1.0)], ComparisonOp::Le, 1e16);
            let solution = problem
                .solve_with(options())
                .unwrap()
                .into_solution()
                .unwrap();
            assert_eq!(solution[x], 1e16, "presolve {presolve}");
        }
    }

    /// Why the bound infinity is 2^52 and not 1e20: a bound in between that
    /// stands for infinity, as `i64::MAX` does in issue #3, must not change
    /// the answer, and it does once a var sits at it, since a row beside such
    /// a value is held to the round-off of its magnitude, units at 2^52
    /// (`INFINITE_BOUND` gives the measurement). `min x0 + x1 + 2 x2` s.t.
    /// `x0 + 2 x1 + 2 x2 >= 9`, `x1 ∈ [4, 10]` has the optimum -1 with `x0`
    /// and `x2` free; written with bounds of `±H` it came back -2, 16 or 0
    /// with a threshold of 1e20, and issue #3's model with `z ≤ 5e15` or
    /// `1e16` an `InternalError` (a row violated by 1 or 2 units).
    #[test]
    fn a_bound_from_2_pow_52_standing_for_infinity_keeps_the_answer() {
        for h in [5e15, 1e16, 1e17, 1e19, i64::MAX as f64] {
            for presolve in [true, false] {
                let options = crate::SolveOptions {
                    presolve,
                    ..crate::SolveOptions::default()
                };
                let mut problem = Problem::new(OptimizationDirection::Minimize);
                let x0 = problem.add_var(1.0, (-h, h));
                let x1 = problem.add_var(1.0, (4.0, 10.0));
                let x2 = problem.add_var(2.0, (-h, h));
                problem.add_constraint([(x0, 1.0), (x1, 2.0), (x2, 2.0)], ComparisonOp::Ge, 9.0);
                let objective = problem
                    .solve_with(options.clone())
                    .unwrap_or_else(|e| panic!("H {h:e}, presolve {presolve}: {e:?}"))
                    .into_solution()
                    .unwrap()
                    .objective();
                assert!(
                    (objective + 1.0).abs() < 1e-9,
                    "H {h:e}, presolve {presolve}: objective {objective}"
                );

                let mut problem = Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(50.0, (2.0, f64::INFINITY));
                let y = problem.add_var(40.0, (0.0, 7.0));
                let z = problem.add_var(45.0, (0.0, h));
                problem.add_constraint([(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);
                problem.add_constraint([(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);
                let objective = problem
                    .solve_with(options)
                    .unwrap_or_else(|e| panic!("issue 3, z <= {h:e}, presolve {presolve}: {e:?}"))
                    .into_solution()
                    .unwrap()
                    .objective();
                assert!(
                    (objective - 420.0).abs() < 1e-6,
                    "issue 3, z <= {h:e}, presolve {presolve}: objective {objective}"
                );
            }
        }
    }

    /// A var whose range lies beyond 2^52 (a lower bound of 5e15, or 2^52
    /// itself) with an upper bound standing for infinity (`f64::MAX`,
    /// `i64::MAX`, 1e20): `max x + y - 2z` s.t. `x + 2y - z <= 7`,
    /// `-3x + y + z <= 1`, `y ∈ [0, 10]`, `z` free, has its optimum at `x`
    /// on its lower bound, `y = 0` and `z = x - 7` (objective `14 - x`), and
    /// its mirror (`min` with `x ∈ [-f64::MAX, -5e15]`) is infeasible. The
    /// far bound stayed as written because the var's whole range lay beyond
    /// the threshold, so the engine started `x` on it: at `f64::MAX` the
    /// rows overflowed (a panic on a NaN in the primal ratio test, and
    /// `InternalError("basis too ill-conditioned ...")` for the mirror), and
    /// at `i64::MAX` or 1e20 the answer came back 14 short of the optimum
    /// (found by the third review of R0; 1dcd664 solved all of them). Each
    /// side of a range is now judged on its own, so the far bound is
    /// infinite and the model is the one written with `f64::INFINITY`, to
    /// the bit.
    #[test]
    fn a_var_beyond_2_pow_52_takes_its_far_bound_as_infinite() {
        let two_52 = 2.0f64.powi(52);
        let solve = |direction, (lo, hi): (f64, f64), milp: bool, presolve: bool| {
            let mut problem = Problem::new(direction);
            let x = problem.add_var(1.0, (lo, hi));
            let y = problem.add_var(1.0, (0.0, 10.0));
            let z = problem.add_var(-2.0, (f64::NEG_INFINITY, f64::INFINITY));
            if milp {
                let n = problem.add_integer_var(0.0, (0, 1));
                problem.add_constraint([(n, 1.0)], ComparisonOp::Le, 1.0);
            }
            problem.add_constraint([(x, 1.0), (y, 2.0), (z, -1.0)], ComparisonOp::Le, 7.0);
            problem.add_constraint([(x, -3.0), (y, 1.0), (z, 1.0)], ComparisonOp::Le, 1.0);
            problem
                .solve_with(crate::SolveOptions {
                    presolve,
                    ..crate::SolveOptions::default()
                })
                .map(|outcome| {
                    let solution = outcome.into_solution().unwrap();
                    (solution.objective(), solution[x], solution[y], solution[z])
                })
        };
        for milp in [false, true] {
            for presolve in [true, false] {
                let case = |near: f64| format!("x >= {near:e}, milp {milp}, presolve {presolve}");
                for near in [5e15, two_52] {
                    let want = solve(
                        OptimizationDirection::Maximize,
                        (near, f64::INFINITY),
                        milp,
                        presolve,
                    )
                    .unwrap_or_else(|e| panic!("{}: {e:?}", case(near)));
                    let (objective, x, y, z) = want;
                    assert!(
                        x == near
                            && y == 0.0
                            && (z - (near - 7.0)).abs() <= 1e-12 * near
                            && (objective - (14.0 - near)).abs() <= 1e-12 * near,
                        "{}: {want:?}",
                        case(near)
                    );
                    for far in [f64::MAX, i64::MAX as f64, 1e20, f64::INFINITY] {
                        let got =
                            solve(OptimizationDirection::Maximize, (near, far), milp, presolve);
                        assert_eq!(got, Ok(want), "{}, x <= {far:e}", case(near));
                        let got = solve(
                            OptimizationDirection::Minimize,
                            (-far, -near),
                            milp,
                            presolve,
                        );
                        assert_eq!(
                            got,
                            Err(crate::Error::Infeasible),
                            "mirror of {}, x >= {:e}",
                            case(near),
                            -far
                        );
                    }
                }
            }
        }
    }

    /// The other side of the per-side rule of the last test: only a bound
    /// that leaves the var room toward it can stand for infinity. A var
    /// fixed beyond 2^52 stays fixed (as `fix_var` fixes a var at a value
    /// the solve reported there), and a range that is empty beyond it stays
    /// infeasible, where making its far bound infinite would have turned
    /// `x ∈ [5e15, 5e15]` into `x ≥ 5e15` and `x ∈ [6e15, 5e15]` into a
    /// feasible var. Any other bound from 2^52 on its side of infinity is
    /// infinite, also beside a bound out there: `max x` with
    /// `x ∈ [5e15, 1e16]` is `Unbounded` as with `x ∈ [0, 1e16]` (the band
    /// trade-off of `a_bound_from_2_pow_52_is_infinite_and_one_below_binds`;
    /// 1dcd664 returned 1e16). Each case is also run in its mirror image
    /// (`x → -x`: the bounds negated and swapped, `x`'s coefficients
    /// negated), where a lower bound is the one that could be taken for
    /// infinity.
    #[test]
    fn a_var_fixed_or_empty_beyond_2_pow_52_keeps_its_bounds() {
        use crate::Error::{Infeasible, Unbounded};
        use OptimizationDirection::{Maximize, Minimize};
        let two_52 = 2.0f64.powi(52);
        for milp in [false, true] {
            for presolve in [true, false] {
                // `x + y` s.t. `x + y >= 1`, `y ∈ [0, 10]`, `x ∈ [lo, hi]`,
                // and with `mirror` the same model in `x' = -x`: the answer
                // is `x' = -x`, and an outcome without one the same.
                let solve = |direction, (lo, hi): (f64, f64), mirror: bool| {
                    let (sign, lo, hi) = if mirror {
                        (-1.0, -hi, -lo)
                    } else {
                        (1.0, lo, hi)
                    };
                    let mut problem = Problem::new(direction);
                    let x = problem.add_var(sign, (lo, hi));
                    let y = problem.add_var(1.0, (0.0, 10.0));
                    if milp {
                        let n = problem.add_integer_var(0.0, (0, 1));
                        problem.add_constraint([(n, 1.0)], ComparisonOp::Le, 1.0);
                    }
                    problem.add_constraint([(x, sign), (y, 1.0)], ComparisonOp::Ge, 1.0);
                    problem
                        .solve_with(crate::SolveOptions {
                            presolve,
                            ..crate::SolveOptions::default()
                        })
                        .map(|outcome| sign * outcome.into_solution().unwrap()[x])
                };
                for mirror in [false, true] {
                    let case = format!("milp {milp}, presolve {presolve}, mirror {mirror}");
                    for value in [two_52, 5e15, 1e20] {
                        for direction in [Maximize, Minimize] {
                            assert_eq!(
                                solve(direction, (value, value), mirror),
                                Ok(value),
                                "{case}: {direction:?}, x fixed at {value:e}"
                            );
                        }
                    }
                    for (lo, hi) in [(6e15, 5e15), (f64::MAX, two_52)] {
                        assert_eq!(
                            solve(Maximize, (lo, hi), mirror),
                            Err(Infeasible),
                            "{case}: x in [{lo:e}, {hi:e}]"
                        );
                    }
                    assert_eq!(
                        solve(Maximize, (5e15, 1e16), mirror),
                        Err(Unbounded),
                        "{case}: max, x in [5e15, 1e16]"
                    );
                    assert_eq!(
                        solve(Minimize, (5e15, 1e16), mirror),
                        Ok(5e15),
                        "{case}: min, x in [5e15, 1e16]"
                    );
                }
            }
        }
    }

    /// Issue #3 through a basis reload (C4-16 of the solver review). Two
    /// free vars written with `±f64::MAX` bounds were seeded at zero, but a
    /// reload of a basis in which they are non-basic put them back at their
    /// finite lower bound, `-f64::MAX`: here the reload of the root basis
    /// after the warm-start hint is evaluated, after which `x + z` overflows
    /// to `-inf` in `x + z <= 5`. The bounds are infinite for the whole solve
    /// now, so every load puts the vars at zero.
    #[test]
    fn huge_bounds_stay_infinite_across_a_basis_reload() {
        for presolve in [true, false] {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let n = problem.add_integer_var(1.0, (0, 10));
            let x = problem.add_var(0.0, (-f64::MAX, f64::MAX));
            let z = problem.add_var(0.0, (-f64::MAX, f64::MAX));
            problem.add_constraint([(n, 1.0)], ComparisonOp::Ge, 1.5);
            problem.add_constraint([(x, 1.0), (z, 1.0)], ComparisonOp::Le, 5.0);
            let options = crate::SolveOptions {
                presolve,
                warm_start: Some(vec![(n, 2.0)]),
                ..crate::SolveOptions::default()
            };
            let sol = problem
                .solve_with(options)
                .unwrap_or_else(|e| panic!("presolve {presolve}: {e:?}"))
                .into_solution()
                .unwrap();
            assert_eq!(sol.objective(), 2.0, "presolve {presolve}");
            assert!(
                sol[x].abs() < 1e10 && sol[z].abs() < 1e10 && sol[x] + sol[z] <= 5.0 + 1e-7,
                "presolve {presolve}: x = {:e}, z = {:e}",
                sol[x],
                sol[z]
            );
        }
    }

    /// `fix_var` accepts a value the solve reported (issue #47's contract)
    /// when it lies beyond a bound from 2^52 on, which the solve takes as
    /// infinite: `x ≤ 5e15` is no bound, the row stops `x` at 1e16, and
    /// fixing `x` there restates the answer. The MIP path checked the fix
    /// against the bound as written and refused it as `Infeasible`, where
    /// the LP path, checking against the engine's bounds, accepted it (found
    /// by the second review of R0). The bound is now made infinite where the
    /// var is added, so no check can see it as written.
    #[test]
    fn fix_var_accepts_a_reported_value_beyond_a_bound_from_2_pow_52() {
        for milp in [false, true] {
            for presolve in [true, false] {
                let mut problem = Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(1.0, (0.0, 5e15));
                if milp {
                    let n = problem.add_integer_var(0.0, (0, 1));
                    problem.add_constraint([(n, 1.0)], ComparisonOp::Le, 1.0);
                }
                problem.add_constraint([(x, 1.0)], ComparisonOp::Le, 1e16);
                let solution = problem
                    .solve_with(crate::SolveOptions {
                        presolve,
                        ..crate::SolveOptions::default()
                    })
                    .unwrap()
                    .into_solution()
                    .unwrap();
                let value = solution[x];
                assert_eq!(value, 1e16, "milp {milp}, presolve {presolve}");
                let refixed = solution
                    .fix_var(x, value)
                    .unwrap_or_else(|e| {
                        panic!("milp {milp}, presolve {presolve}: fix_var(x, {value:e}) gave {e:?}")
                    })
                    .into_solution()
                    .unwrap();
                assert_eq!(refixed[x], 1e16, "milp {milp}, presolve {presolve}");
            }
        }
    }

    /// <https://github.com/Specy/microlp/issues/42>: an always-feasible integer
    /// model must solve no matter how large `x`'s upper bound is. The only
    /// feasible point of `3x + 2y = 5` with `x` integer and `y ∈ [0, 1]` is
    /// `x = 1, y = 1` (objective 1). In 0.4.0, upper bounds of the form `2^k + 2`
    /// made `solve()` return `Err` for even `k` (a bound-magnitude rounding
    /// interaction), even though the feasible point never changes.
    #[test]
    fn issue_42_large_integer_bound_stays_feasible() {
        // The reported bounds are 2^k + 2 for k = 20..=30 (which alternately
        // failed); sweep those and their immediate neighbours as a guard.
        for k in 20..=30u32 {
            for max in [(1i64 << k), (1i64 << k) + 1, (1i64 << k) + 2] {
                let max = max as i32;
                let mut problem = Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_integer_var(1.0, (0, max));
                let y = problem.add_var(0.0, (0.0, 1.0));
                problem.add_constraint([(x, 3.0), (y, 2.0)], ComparisonOp::Eq, 5.0);

                let sol = problem
                    .solve()
                    .unwrap_or_else(|e| panic!("max={max} (~2^{k}) must be feasible, got {e:?}"))
                    .into_solution()
                    .unwrap_or_else(|interrupted| {
                        panic!("max={max} (~2^{k}) must finish without limits, got {interrupted:?}")
                    });

                assert!(
                    (sol.var_value(x) - 1.0).abs() < 1e-6,
                    "x wrong for max={max}"
                );
                assert!(
                    (sol.var_value(y) - 1.0).abs() < 1e-6,
                    "y wrong for max={max}"
                );
                assert!(
                    (sol.objective() - 1.0).abs() < 1e-6,
                    "objective wrong for max={max}: got {}",
                    sol.objective()
                );
            }
        }
    }

    /// One variant of the model reported in issue #44: maximise `-x1` subject
    /// to `8.510167104926385 x2 == 0` (when `first_eq`) and `-x1 + tiny x2 ==
    /// 0`, with `x1` in `[-1, 0]`, `x2` in `[0, 1]` and, when `integer_var`, an
    /// integer variable fixed at zero by its own bounds that appears in neither
    /// the objective nor a row.
    struct Issue44 {
        tiny: f64,
        integer_var: bool,
        first_eq: bool,
    }

    /// Solves an issue-44 variant and returns `(objective, x1, x2)`.
    fn solve_issue_44(variant: Issue44) -> (f64, f64, f64) {
        let Issue44 {
            tiny,
            integer_var,
            first_eq,
        } = variant;
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x1 = problem.add_var(-1.0, (-1.0, 0.0));
        let x2 = problem.add_var(0.0, (0.0, 1.0));
        if integer_var {
            problem.add_integer_var(0.0, (0, 0));
        }
        if first_eq {
            problem.add_constraint([(x2, 8.510167104926385)], ComparisonOp::Eq, 0.0);
        }
        problem.add_constraint([(x1, -1.0), (x2, tiny)], ComparisonOp::Eq, 0.0);
        let sol = problem
            .solve()
            .unwrap_or_else(|e| panic!("tiny={tiny:e}: must solve, got {e:?}"))
            .into_solution()
            .unwrap_or_else(|i| panic!("tiny={tiny:e}: must finish without limits, got {i:?}"));
        (sol.objective(), sol.var_value(x1), sol.var_value(x2))
    }

    /// <https://github.com/Specy/microlp/issues/44>: `solve()` returns
    /// `InternalError("exactly integral solution failed feasibility validation
    /// after slack-basis retry")` on a feasible, bounded MILP. `x1 = x2 = 0`
    /// satisfies both equalities and every bound, and it is the only such
    /// point, so the answer is `Ok` with objective `0`.
    ///
    /// The report rules out a magnitude threshold in the small coefficient: the
    /// same model solves at `-1e-7`, `-1e-8`, `-1e-9` and `-1e-12`, and the
    /// reported coefficient itself solves once either the integer variable or
    /// the first row is dropped.
    #[test]
    fn issue_44_feasible_bounded_milp_solves() {
        let tiny = -1.190834764418229e-8;

        // The reported model, and the coefficients around it that already
        // solve. In each the only feasible point is x1 = x2 = 0.
        for t in [tiny, -1e-7, -1e-8, -1e-9, -1e-12] {
            let (obj, x1, x2) = solve_issue_44(Issue44 {
                tiny: t,
                integer_var: true,
                first_eq: true,
            });
            assert!(
                obj.abs() < 1e-9 && x1.abs() < 1e-9 && x2.abs() < 1e-9,
                "tiny={t:e}: got obj={obj:e} x1={x1:e} x2={x2:e}, expected all zero"
            );
        }

        // The same LP down the pure-LP path, with no integer variable.
        let (obj, x1, x2) = solve_issue_44(Issue44 {
            tiny,
            integer_var: false,
            first_eq: true,
        });
        assert!(
            obj.abs() < 1e-9 && x1.abs() < 1e-9 && x2.abs() < 1e-9,
            "pure LP: got obj={obj:e} x1={x1:e} x2={x2:e}, expected all zero"
        );

        // Without the first row, x2 is free to reach its upper bound and the
        // optimum is -tiny, reached at x1 = tiny, x2 = 1. Asserting it guards
        // against "fixing" the model above by dropping small coefficients from
        // the ratio tests, which would answer 0 here.
        let (obj, x1, x2) = solve_issue_44(Issue44 {
            tiny,
            integer_var: true,
            first_eq: false,
        });
        assert!(
            (obj + tiny).abs() < 1e-14 && (x1 - tiny).abs() < 1e-14 && (x2 - 1.0).abs() < 1e-9,
            "no first row: got obj={obj:e} x1={x1:e} x2={x2:e}, expected obj={:e} x1={tiny:e} x2=1",
            -tiny
        );
    }

    /// <https://github.com/Specy/microlp/issues/47>: `Solution::fix_var`
    /// returns `Infeasible` when a variable that appears in an equality row is
    /// fixed to the value the solve just reported for it. The row pins `x` to
    /// its rhs, `solve` returns that value, and fixing `x` to it only restates
    /// what the row already forces, so the edited model has the same feasible
    /// point as the original.
    ///
    /// The `Le` row is the control: the same value through a row that is not an
    /// equality is accepted. The third case moves the fixed value off the
    /// variable's bounds and away from zero, so neither explains the failure.
    #[test]
    fn issue_47_fix_var_accepts_the_value_solve_returned() {
        for (op, rhs, hi) in [
            (ComparisonOp::Eq, 0.0, 1.0),
            (ComparisonOp::Le, 0.0, 1.0),
            (ComparisonOp::Eq, 1.0, 2.0),
        ] {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(-1.0, (0.0, hi));
            problem.add_constraint([(x, 1.0)], op, rhs);

            let solution = problem
                .solve()
                .unwrap_or_else(|e| panic!("x in [0, {hi}], x {op:?} {rhs}: must solve, got {e:?}"))
                .into_solution()
                .unwrap_or_else(|i| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: must finish without limits, got {i:?}")
                });

            // Minimising -x drives x up to the row, so the reported value is
            // the rhs in all three models.
            let value = solution.var_value(x);
            assert!(
                (value - rhs).abs() < 1e-9,
                "x in [0, {hi}], x {op:?} {rhs}: solve gave x = {value:?}, expected {rhs:?}"
            );

            let refixed = solution
                .fix_var(x, value)
                .unwrap_or_else(|e| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: fix_var(x, {value:?}) gave {e:?}")
                })
                .into_solution()
                .unwrap_or_else(|i| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: fix_var must finish, got {i:?}")
                });

            let fixed_value = refixed.var_value(x);
            assert!(
                (fixed_value - value).abs() < 1e-9,
                "x in [0, {hi}], x {op:?} {rhs}: after fix_var(x, {value:?}) x = {fixed_value:?}"
            );
        }
    }

    /// Issue #47 in general: a value `solve` reports may sit outside the
    /// var's bounds by round-off, within the contract (here
    /// `0.2999999999999998` on `[0.3, 5.3]`), and `fix_var` compared it with
    /// the bounds exactly and refused it as `Infeasible`, on the LP and the
    /// MIP path alike. So did a value within tolerance of a bound presolve
    /// had tightened (`2 x = 6` fixes `x` at 3). A value within the bound
    /// tolerance fixes the var at that bound; one beyond it is refused.
    #[test]
    fn issue_47_fix_var_accepts_a_value_within_the_bound_tolerance() {
        for milp in [false, true] {
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x0 = problem.add_var(0.2, (0.3, 5.3));
            let x1 = problem.add_var(1.3, (-2.0, 0.0));
            if milp {
                problem.add_integer_var(0.0, (0, 1));
            }
            problem.add_constraint([(x0, 0.7), (x1, 0.3)], ComparisonOp::Le, 0.21);
            let solution = problem.solve().unwrap().into_solution().unwrap();
            let value = solution.var_value(x0);
            let refixed = solution
                .clone()
                .fix_var(x0, value)
                .unwrap_or_else(|e| panic!("milp {milp}: fix_var(x0, {value:?}) gave {e:?}"))
                .into_solution()
                .unwrap();
            assert!((refixed.var_value(x0) - 0.3).abs() < 1e-7, "milp {milp}");
            assert!(
                matches!(solution.fix_var(x0, 0.29), Err(crate::Error::Infeasible)),
                "milp {milp}: a value 0.01 below the bound must be refused"
            );
        }
        for presolve in [true, false] {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(1.0, (0.0, 10.0));
            let y = problem.add_var(1.0, (0.0, 10.0));
            problem.add_constraint([(x, 2.0)], ComparisonOp::Eq, 6.0);
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 4.0);
            let solution = problem
                .solve_with(crate::SolveOptions {
                    presolve,
                    ..crate::SolveOptions::default()
                })
                .unwrap()
                .into_solution()
                .unwrap();
            let refixed = solution
                .fix_var(x, 3.000000001)
                .unwrap_or_else(|e| {
                    panic!("presolve {presolve}: fix_var(x, 3.000000001) gave {e:?}")
                })
                .into_solution()
                .unwrap();
            assert!(
                (refixed.objective() - 4.0).abs() < 1e-6,
                "presolve {presolve}"
            );
        }
    }

    /// Fixing an integer var: a value within the integrality tolerance of an
    /// integer fixes it there; any other value is incompatible with its
    /// domain (`Infeasible`, as `fix_var` documents). A fix at `2.0000005`
    /// used to end in an `InternalError` with presolve off (the fixed var is
    /// not integral, and branching skips a var whose range is under half a
    /// unit) and in a silently different value with presolve on.
    #[test]
    fn fix_var_on_an_integer_var_respects_its_domain() {
        for presolve in [true, false] {
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x = problem.add_integer_var(1.0, (0, 10));
            let y = problem.add_var(1.0, (0.0, 1.0));
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 5.5);
            let options = crate::SolveOptions {
                presolve,
                ..crate::SolveOptions::default()
            };
            let solution = problem
                .solve_with(options)
                .unwrap()
                .into_solution()
                .unwrap();
            let fixed = solution
                .clone()
                .fix_var(x, 2.0000005)
                .unwrap_or_else(|e| panic!("presolve {presolve}: fix_var(x, 2.0000005) gave {e:?}"))
                .into_solution()
                .unwrap();
            assert_eq!(fixed.var_value(x), 2.0, "presolve {presolve}");
            assert!(
                matches!(solution.fix_var(x, 2.4), Err(crate::Error::Infeasible)),
                "presolve {presolve}: 2.4 is not an integer"
            );
        }
    }

    /// A DC network with switchable lines, the model of
    /// [`issue_48_switchable_dc_network_solves`]: per line
    /// `(from, to, susceptance, rating)`, a flow within its rating and a
    /// binary `on` whose big-M rows tie the flow to the bus angles (free, but
    /// for the reference bus 0) while the line is on; per bus, generation
    /// within `gen_max` at `gen_cost` less `demand` equals the flow out less
    /// the flow in; at most `max_off` lines off.
    fn switchable_dc_network(
        lines: &[(usize, usize, f64, f64)],
        demand: &[f64],
        gen_max: &[f64],
        gen_cost: &[f64],
        max_off: usize,
    ) -> Problem {
        let n = demand.len();
        let angle_max = 0.52;
        let big = angle_max * n as f64;
        let mut problem = Problem::new(OptimizationDirection::Minimize);

        // Bus angles: free, except the reference bus.
        let theta: Vec<_> = (0..n)
            .map(|i| match i {
                0 => problem.add_var(0.0, (0.0, 0.0)),
                _ => problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY)),
            })
            .collect();
        let gen: Vec<_> = (0..n)
            .map(|i| problem.add_var(gen_cost[i], (0.0, gen_max[i])))
            .collect();
        let flow: Vec<_> = lines
            .iter()
            .map(|&(_, _, _, r)| problem.add_var(0.0, (-r, r)))
            .collect();
        let on: Vec<_> = lines.iter().map(|_| problem.add_binary_var(0.0)).collect();

        for (e, &(i, j, b, r)) in lines.iter().enumerate() {
            let m = b * big;
            // |flow - b (theta_i - theta_j)| <= m (1 - on)
            problem.add_constraint(
                [(flow[e], 1.0), (theta[i], -b), (theta[j], b), (on[e], m)],
                ComparisonOp::Le,
                m,
            );
            problem.add_constraint(
                [(flow[e], 1.0), (theta[i], -b), (theta[j], b), (on[e], -m)],
                ComparisonOp::Ge,
                -m,
            );
            // |flow| <= r on
            problem.add_constraint([(flow[e], 1.0), (on[e], -r)], ComparisonOp::Le, 0.0);
            problem.add_constraint([(flow[e], 1.0), (on[e], r)], ComparisonOp::Ge, 0.0);
            // |theta_i - theta_j| <= angle_max + big (1 - on)
            problem.add_constraint(
                [(theta[i], 1.0), (theta[j], -1.0), (on[e], big)],
                ComparisonOp::Le,
                big + angle_max,
            );
            problem.add_constraint(
                [(theta[i], 1.0), (theta[j], -1.0), (on[e], -big)],
                ComparisonOp::Ge,
                -big - angle_max,
            );
        }
        for i in 0..n {
            // generation - demand = flow out - flow in
            let mut terms = vec![(gen[i], 1.0)];
            for (e, &(from, to, _, _)) in lines.iter().enumerate() {
                if from == i {
                    terms.push((flow[e], -1.0));
                } else if to == i {
                    terms.push((flow[e], 1.0));
                }
            }
            problem.add_constraint(terms, ComparisonOp::Eq, demand[i]);
        }
        let in_service: Vec<_> = on.iter().map(|&z| (z, 1.0)).collect();
        problem.add_constraint(in_service, ComparisonOp::Ge, (lines.len() - max_off) as f64);
        problem
    }

    /// <https://github.com/Specy/microlp/issues/48>: `solve()` returns
    /// `InternalError("Singular matrix")` on a small, feasible, bounded MILP —
    /// a DC network with switchable lines (8 buses, 10 lines, at most one
    /// switched out), whose big-M on/off rows spread the coefficients from
    /// 4e-2 to 1.1e3. HiGHS and SCIP prove the same model optimal at
    /// 23.339329.
    ///
    /// The report traces it to the dual ratio test accepting a pivot of
    /// |alpha| = 2.2e-10 — just above the `1e-10` floor, and zero in exact
    /// arithmetic — after which the refactorization is singular and the
    /// slack-basis retry re-derives a singular basis of its own.
    #[test]
    fn issue_48_switchable_dc_network_solves() {
        // (from, to, susceptance, rating)
        let lines = [
            (0, 1, 0.97, 0.52),
            (2, 4, 0.24, 1.11),
            (3, 5, 431.79, 0.55),
            (2, 6, 54.08, 0.77),
            (3, 7, 0.06, 1.22),
            (6, 1, 63.12, 0.84),
            (4, 1, 1108.6, 1.17),
            (1, 3, 0.34, 0.48),
            (5, 1, 538.81, 0.71),
            (3, 4, 0.04, 0.83),
        ];
        let demand = [0.4, 0.4, 0.85, 0.92, 0.19, 0.15, 0.52, 0.6];
        let gen_max = [0.53, 1.36, 0.95, 0.3, 0.5, 0.92, 0.22, 0.67];
        let gen_cost = [1.57, 8.12, 8.82, 8.04, 3.59, 2.49, 9.28, 8.04];
        let problem = switchable_dc_network(&lines, &demand, &gen_max, &gen_cost, 1);

        let sol = problem
            .solve()
            .unwrap_or_else(|e| panic!("a feasible, bounded MILP must solve, got {e:?}"))
            .into_solution()
            .unwrap_or_else(|i| panic!("must finish without limits, got {i:?}"));

        assert!(
            (sol.objective() - 23.339329).abs() < 1e-4,
            "objective {} does not match the 23.339329 HiGHS and SCIP prove optimal",
            sol.objective()
        );
    }

    /// The minimal objective of the switchable network with these data,
    /// solved with presolve on or off; `Err` with the solve's error.
    fn switchable_dc_network_objective(
        lines: &[(usize, usize, f64, f64)],
        demand: &[f64],
        gen_max: &[f64],
        gen_cost: &[f64],
        max_off: usize,
        presolve: bool,
    ) -> Result<f64, crate::Error> {
        let options = crate::SolveOptions {
            presolve,
            ..crate::SolveOptions::default()
        };
        let solved =
            switchable_dc_network(lines, demand, gen_max, gen_cost, max_off).solve_with(options)?;
        Ok(solved
            .into_solution()
            .unwrap_or_else(|i| panic!("must finish without limits, got {i:?}"))
            .objective())
    }

    /// A switchable network like issue 48's, from a random search over such
    /// networks with susceptances from `1e-3` to `1e6` (HiGHS proves it
    /// optimal at 18.979188), whose lines of `1.05e-3` and `2e5` put big-M
    /// rows of very different scales into its bases. At two of its nodes the
    /// row of the basis inverse that proves the node infeasible gives a free
    /// bus angle, basic, a coefficient whose only term comes from a row whose
    /// multiplier is zero for the exact row: the refinement leaves it as its
    /// own noise, some `1e-46` against multipliers of up to 4, and the
    /// coefficient is that noise times the angle's entry. No floor relative
    /// to the coefficient's terms covers it, those terms being the noise too;
    /// the bound the refinement puts on the multipliers' error, their last
    /// correction (`3.9e-31`), does. Counted only below the `f64` round-off
    /// of its terms, on a row refined once, the coefficient voided those
    /// proofs, and the MILP failed loudly with presolve on and off; counted
    /// only within the round-off of the compensated sums, it failed the same
    /// way.
    #[test]
    fn a_switchable_network_whose_node_proofs_need_the_multipliers_error_solves() {
        let lines = [
            (1, 0, 26.2, 0.353),
            (2, 1, 0.209, 0.445),
            (3, 1, 22.1, 0.511),
            (4, 1, 200000.0, 0.758),
            (5, 4, 51.5, 0.517),
            (6, 2, 0.00105, 0.532),
            (7, 6, 5610.0, 0.689),
            (0, 2, 136.0, 0.562),
            (1, 3, 18.1, 0.706),
            (3, 7, 3.72, 0.93),
        ];
        let demand = [0.683, 1.01, 0.531, 0.274, 1.04, 0.464, 0.303, 0.189];
        let gen_max = [0.828, 0.41, 0.244, 0.451, 1.37, 0.488, 1.24, 0.342];
        let gen_cost = [5.57, 5.72, 8.87, 9.02, 1.21, 2.02, 5.8, 9.19];
        for presolve in [true, false] {
            let objective =
                switchable_dc_network_objective(&lines, &demand, &gen_max, &gen_cost, 2, presolve)
                    .unwrap_or_else(|e| {
                        panic!(
                            "presolve {presolve}: a feasible, bounded MILP must solve, got {e:?}"
                        )
                    });
            assert!(
                (objective - 18.979188).abs() < 1e-4,
                "presolve {presolve}: objective {objective} does not match the 18.979188 HiGHS \
                 proves optimal"
            );
        }
    }

    /// An infeasible switchable network like issue 48's (HiGHS agrees), from
    /// the same search, with lines from `3.15e-3` to `4.13e5`. Its
    /// certificates give free bus angles, basic, coefficients whose terms
    /// come from rows whose multipliers are zero for the exact row of the
    /// basis inverse and which the refinement leaves as its own noise, above
    /// any floor relative to the coefficients' terms, which are that noise
    /// too. The second step of the refinement brings them within the bound it
    /// puts on the multipliers' error, its own correction; refined once, with
    /// that correction as the bound, some lie beyond it, and counted only
    /// within the round-off of the compensated sums, all do: either way the
    /// proofs failed and the solve failed loudly (with presolve on and off).
    /// Counted only below the `f64` round-off of their terms, on a row
    /// refined once, the solve failed loudly without presolve.
    #[test]
    fn an_infeasible_switchable_network_whose_proof_needs_the_twice_refined_row() {
        let lines = [
            (1, 0, 0.17, 0.557),
            (2, 1, 0.00315, 1.15),
            (3, 2, 587.0, 0.374),
            (4, 3, 1.57, 0.519),
            (5, 3, 1150.0, 1.27),
            (6, 3, 31900.0, 0.984),
            (7, 6, 107000.0, 1.15),
            (5, 0, 0.00352, 1.28),
            (5, 6, 10.1, 0.823),
            (0, 1, 413000.0, 1.15),
        ];
        let demand = [0.537, 0.835, 0.245, 1.07, 1.09, 0.823, 0.775, 0.73];
        let gen_max = [0.683, 1.18, 0.38, 1.2, 0.827, 0.318, 1.3, 0.602];
        let gen_cost = [6.7, 8.07, 2.61, 8.44, 4.5, 5.87, 8.26, 4.92];
        for presolve in [true, false] {
            assert_eq!(
                switchable_dc_network_objective(&lines, &demand, &gen_max, &gen_cost, 2, presolve),
                Err(crate::Error::Infeasible),
                "presolve {presolve}"
            );
        }
    }
}
