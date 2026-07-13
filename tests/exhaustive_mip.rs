use microlp::{
    ComparisonOp, Error, OptimizationDirection, Problem, SolveOptions, Status, Variable,
};

#[derive(Clone, Copy, Debug)]
struct Row {
    coeffs: [i32; 4],
    op: ComparisonOp,
    rhs: i32,
    scale: f64,
}

#[derive(Clone, Debug)]
struct Case {
    n: usize,
    lo: [i32; 4],
    hi: [i32; 4],
    objective: [i32; 4],
    rows: Vec<Row>,
}

#[derive(Clone, Debug)]
struct MixedCase {
    n_int: usize,
    lo: [i32; 3],
    hi: [i32; 3],
    real_lo: f64,
    real_hi: f64,
    objective: [i32; 4],
    rows: Vec<Row>,
}

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u32) as i32
    }
}

fn generated_case(seed: u64) -> Case {
    let mut rng = Lcg(seed ^ 0xa076_1d64_78bd_642f);
    let n = rng.range(1, 4) as usize;
    let mut lo = [0; 4];
    let mut hi = [0; 4];
    let mut objective = [0; 4];
    for v in 0..n {
        lo[v] = rng.range(-3, 1);
        hi[v] = rng.range(lo[v], 3);
        objective[v] = rng.range(-9, 9);
    }
    if objective[..n].iter().all(|&c| c == 0) {
        objective[0] = 1;
    }

    let row_count = rng.range(0, 7) as usize;
    let mut rows = Vec::with_capacity(row_count);
    for _ in 0..row_count {
        let mut coeffs = [0; 4];
        for coeff in &mut coeffs[..n] {
            *coeff = rng.range(-7, 7);
        }
        if coeffs[..n].iter().all(|&c| c == 0) {
            coeffs[rng.range(0, n as i32 - 1) as usize] = 1;
        }
        let op = match rng.next() % 5 {
            0 => ComparisonOp::Eq,
            1 | 2 => ComparisonOp::Le,
            _ => ComparisonOp::Ge,
        };
        let scale = match rng.next() % 5 {
            0 => 1e-6,
            1 => 1e6,
            _ => 1.0,
        };
        rows.push(Row {
            coeffs,
            op,
            rhs: rng.range(-20, 20),
            scale,
        });
    }
    Case {
        n,
        lo,
        hi,
        objective,
        rows,
    }
}

fn generated_mixed_case(seed: u64) -> MixedCase {
    let mut rng = Lcg(seed ^ 0xe703_7ed1_a0b4_28db);
    let n_int = rng.range(1, 3) as usize;
    let mut lo = [0; 3];
    let mut hi = [0; 3];
    let mut objective = [0; 4];
    for v in 0..n_int {
        lo[v] = rng.range(-2, 0);
        hi[v] = rng.range(lo[v], 2);
        objective[v] = rng.range(-6, 6);
    }
    objective[n_int] = rng.range(-6, 6);
    if objective[..=n_int].iter().all(|&c| c == 0) {
        objective[n_int] = 1;
    }

    let mut rows = Vec::new();
    for _ in 0..rng.range(0, 6) {
        let mut coeffs = [0; 4];
        for coeff in &mut coeffs[..=n_int] {
            *coeff = rng.range(-6, 6);
        }
        if coeffs[..=n_int].iter().all(|&c| c == 0) {
            coeffs[n_int] = 1;
        }
        rows.push(Row {
            coeffs,
            op: match rng.next() % 5 {
                0 => ComparisonOp::Eq,
                1 | 2 => ComparisonOp::Le,
                _ => ComparisonOp::Ge,
            },
            rhs: rng.range(-15, 15),
            scale: match rng.next() % 5 {
                0 => 1e-6,
                1 => 1e6,
                _ => 1.0,
            },
        });
    }
    MixedCase {
        n_int,
        lo,
        hi,
        real_lo: -3.0,
        real_hi: 3.0,
        objective,
        rows,
    }
}

fn build_problem(
    case: &Case,
    warm_start: Option<&[i32; 4]>,
) -> (Problem, Vec<Variable>, SolveOptions) {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let vars: Vec<_> = (0..case.n)
        .map(|v| problem.add_integer_var(case.objective[v] as f64, (case.lo[v], case.hi[v])))
        .collect();
    for row in &case.rows {
        problem.add_constraint(
            (0..case.n).map(|v| (vars[v], row.coeffs[v] as f64 * row.scale)),
            row.op,
            row.rhs as f64 * row.scale,
        );
    }
    let mut options = SolveOptions::default();
    options.warm_start = warm_start.map(|values| {
        vars.iter()
            .enumerate()
            .map(|(v, &var)| (var, values[v] as f64))
            .collect::<Vec<(Variable, f64)>>()
    });
    (problem, vars, options)
}

fn row_holds(row: &Row, values: &[i32; 4], n: usize) -> bool {
    let lhs: i64 = (0..n)
        .map(|v| i64::from(row.coeffs[v]) * i64::from(values[v]))
        .sum();
    match row.op {
        ComparisonOp::Eq => lhs == i64::from(row.rhs),
        ComparisonOp::Le => lhs <= i64::from(row.rhs),
        ComparisonOp::Ge => lhs >= i64::from(row.rhs),
    }
}

fn brute_force(case: &Case) -> Option<(i64, [i32; 4])> {
    fn visit(case: &Case, v: usize, values: &mut [i32; 4], best: &mut Option<(i64, [i32; 4])>) {
        if v < case.n {
            for value in case.lo[v]..=case.hi[v] {
                values[v] = value;
                visit(case, v + 1, values, best);
            }
            return;
        }
        if !case.rows.iter().all(|row| row_holds(row, values, case.n)) {
            return;
        }
        let objective: i64 = (0..case.n)
            .map(|i| i64::from(case.objective[i]) * i64::from(values[i]))
            .sum();
        if best
            .as_ref()
            .is_none_or(|(incumbent, _)| objective < *incumbent)
        {
            *best = Some((objective, *values));
        }
    }

    let mut best = None;
    visit(case, 0, &mut [0; 4], &mut best);
    best
}

fn mixed_oracle(case: &MixedCase) -> Option<(f64, [i32; 3], f64)> {
    fn visit(
        case: &MixedCase,
        v: usize,
        ints: &mut [i32; 3],
        best: &mut Option<(f64, [i32; 3], f64)>,
    ) {
        if v < case.n_int {
            for value in case.lo[v]..=case.hi[v] {
                ints[v] = value;
                visit(case, v + 1, ints, best);
            }
            return;
        }

        let mut lo = case.real_lo;
        let mut hi = case.real_hi;
        for row in &case.rows {
            let fixed_lhs: i64 = (0..case.n_int)
                .map(|i| i64::from(row.coeffs[i]) * i64::from(ints[i]))
                .sum();
            let rhs = f64::from(row.rhs) - fixed_lhs as f64;
            let coeff = f64::from(row.coeffs[case.n_int]);
            if coeff == 0.0 {
                let feasible = match row.op {
                    ComparisonOp::Eq => rhs == 0.0,
                    ComparisonOp::Le => 0.0 <= rhs,
                    ComparisonOp::Ge => 0.0 >= rhs,
                };
                if !feasible {
                    return;
                }
                continue;
            }
            let boundary = rhs / coeff;
            match row.op {
                ComparisonOp::Eq => {
                    lo = lo.max(boundary);
                    hi = hi.min(boundary);
                }
                ComparisonOp::Le if coeff > 0.0 => hi = hi.min(boundary),
                ComparisonOp::Le => lo = lo.max(boundary),
                ComparisonOp::Ge if coeff > 0.0 => lo = lo.max(boundary),
                ComparisonOp::Ge => hi = hi.min(boundary),
            }
        }
        if lo > hi + 1e-12 {
            return;
        }
        let real = if case.objective[case.n_int] >= 0 {
            lo
        } else {
            hi
        };
        let objective = (0..case.n_int)
            .map(|i| f64::from(case.objective[i]) * f64::from(ints[i]))
            .sum::<f64>()
            + f64::from(case.objective[case.n_int]) * real;
        if best
            .as_ref()
            .is_none_or(|(incumbent, _, _)| objective < *incumbent - 1e-10)
        {
            *best = Some((objective, *ints, real));
        }
    }

    let mut best = None;
    visit(case, 0, &mut [0; 3], &mut best);
    best
}

fn build_mixed_problem(case: &MixedCase) -> (Problem, Vec<Variable>) {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let mut vars: Vec<_> = (0..case.n_int)
        .map(|v| problem.add_integer_var(f64::from(case.objective[v]), (case.lo[v], case.hi[v])))
        .collect();
    vars.push(problem.add_var(
        f64::from(case.objective[case.n_int]),
        (case.real_lo, case.real_hi),
    ));
    for row in &case.rows {
        problem.add_constraint(
            (0..=case.n_int).map(|v| (vars[v], f64::from(row.coeffs[v]) * row.scale)),
            row.op,
            f64::from(row.rhs) * row.scale,
        );
    }
    (problem, vars)
}

fn assert_solution(
    case: &Case,
    vars: &[Variable],
    seed: u64,
    expected: i64,
    solution: &microlp::Solution,
) {
    assert_eq!(solution.status(), Status::Optimal, "seed {seed}: {case:?}");
    assert!(
        (solution.objective() - expected as f64).abs() < 1e-7,
        "seed {seed}: objective {}, expected {expected}; {case:?}",
        solution.objective()
    );
    let mut values = [0; 4];
    for (v, value) in values.iter_mut().enumerate().take(case.n) {
        *value = solution.var_value(vars[v]).round() as i32;
    }
    assert!(
        case.rows.iter().all(|row| row_holds(row, &values, case.n)),
        "seed {seed}: solver returned infeasible values {values:?}; {case:?}"
    );
}

#[test]
fn bounded_integer_models_match_exhaustive_enumeration() {
    for seed in 0..50_000 {
        let case = generated_case(seed);
        let expected = brute_force(&case);
        let (problem, vars, _) = build_problem(&case, None);
        match (expected, problem.solve()) {
            (None, Err(Error::Infeasible)) => {}
            (None, other) => panic!("seed {seed}: expected infeasible, got {other:?}; {case:?}"),
            (Some((objective, values)), Ok(solution)) => {
                assert_solution(&case, &vars, seed, objective, &solution);
                let cold_values: Vec<_> = vars.iter().map(|&var| solution.var_value(var)).collect();

                if seed % 10 == 0 {
                    let (hinted_problem, hinted_vars, options) =
                        build_problem(&case, Some(&values));
                    let hinted = hinted_problem.solve_with(options).unwrap_or_else(|error| {
                        panic!("seed {seed}: valid warm start failed: {error}; {case:?}")
                    });
                    assert_solution(&case, &hinted_vars, seed, objective, &hinted);
                }

                if seed % 25 == 0 {
                    let mut options = SolveOptions::default();
                    options.node_limit = Some(1);
                    let mut resumed = problem.solve_with(options).unwrap_or_else(|error| {
                        panic!("seed {seed}: node-limited solve failed: {error}; {case:?}")
                    });
                    for _ in 0..10_000 {
                        if resumed.status() == Status::Optimal {
                            break;
                        }
                        resumed = resumed.resume(None).unwrap_or_else(|error| {
                            panic!("seed {seed}: resume failed: {error}; {case:?}")
                        });
                    }
                    assert_solution(&case, &vars, seed, objective, &resumed);
                    let resumed_values: Vec<_> =
                        vars.iter().map(|&var| resumed.var_value(var)).collect();
                    assert_eq!(
                        resumed_values, cold_values,
                        "seed {seed}: resumed solve selected a different optimum; {case:?}"
                    );
                }
            }
            (Some(expected), other) => {
                panic!("seed {seed}: expected {expected:?}, got {other:?}; {case:?}")
            }
        }
    }
}

#[test]
fn mixed_integer_models_match_integer_enumeration_and_analytic_lp() {
    for seed in 0..25_000 {
        let case = generated_mixed_case(seed);
        let expected = mixed_oracle(&case);
        let (problem, vars) = build_mixed_problem(&case);
        match (expected, problem.solve()) {
            (None, Err(Error::Infeasible)) => {}
            (None, other) => panic!("seed {seed}: expected infeasible, got {other:?}; {case:?}"),
            (Some((objective, _, _)), Ok(solution)) => {
                assert_eq!(solution.status(), Status::Optimal, "seed {seed}: {case:?}");
                assert!(
                    (solution.objective() - objective).abs() < 1e-6,
                    "seed {seed}: objective {}, expected {objective}; {case:?}",
                    solution.objective()
                );
                for &var in &vars[..case.n_int] {
                    let value = solution.var_value(var);
                    assert_eq!(value, value.round(), "seed {seed}: {case:?}");
                }
            }
            (Some(expected), other) => {
                panic!("seed {seed}: expected {expected:?}, got {other:?}; {case:?}")
            }
        }
    }
}

#[test]
fn integer_edits_match_fresh_exhaustive_models() {
    for seed in 0..10_000 {
        let case = generated_case(seed ^ 0x5bf0_3635);
        let Some((original_objective, _)) = brute_force(&case) else {
            continue;
        };
        let (problem, vars, _) = build_problem(&case, None);
        let solution = problem
            .solve()
            .unwrap_or_else(|error| panic!("seed {seed}: initial solve failed: {error}; {case:?}"));

        let fixed_var = seed as usize % case.n;
        let fixed_value = if seed % 2 == 0 {
            case.lo[fixed_var]
        } else {
            case.hi[fixed_var]
        };
        let mut fixed_case = case.clone();
        fixed_case.lo[fixed_var] = fixed_value;
        fixed_case.hi[fixed_var] = fixed_value;
        let fixed_expected = brute_force(&fixed_case);
        match (
            fixed_expected,
            solution
                .clone()
                .fix_var(vars[fixed_var], f64::from(fixed_value)),
        ) {
            (None, Err(Error::Infeasible)) => {}
            (None, other) => panic!(
                "seed {seed}: fixed model should be infeasible, got {other:?}; {fixed_case:?}"
            ),
            (Some((objective, _)), Ok(fixed)) => {
                assert_solution(&fixed_case, &vars, seed, objective, &fixed);
                let (unfixed, was_fixed) = fixed
                    .unfix_var(vars[fixed_var])
                    .unwrap_or_else(|error| panic!("seed {seed}: unfix failed: {error}"));
                assert!(was_fixed, "seed {seed}: fix was not recorded");
                assert_solution(&case, &vars, seed, original_objective, &unfixed);
            }
            (Some(expected), other) => panic!(
                "seed {seed}: fixed model expected {expected:?}, got {other:?}; {fixed_case:?}"
            ),
        }

        let mut rng = Lcg(seed ^ 0x243f_6a88_85a3_08d3);
        let mut coeffs = [0; 4];
        for coeff in &mut coeffs[..case.n] {
            *coeff = rng.range(-5, 5);
        }
        if coeffs[..case.n].iter().all(|&coeff| coeff == 0) {
            coeffs[0] = 1;
        }
        let added_row = Row {
            coeffs,
            op: if seed % 2 == 0 {
                ComparisonOp::Le
            } else {
                ComparisonOp::Ge
            },
            rhs: rng.range(-10, 10),
            scale: if seed % 3 == 0 { 1e-6 } else { 1e6 },
        };
        let mut edited_case = case.clone();
        edited_case.rows.push(added_row);
        let edited_expected = brute_force(&edited_case);
        let edited = solution.add_constraint(
            (0..case.n).map(|v| (vars[v], f64::from(added_row.coeffs[v]) * added_row.scale)),
            added_row.op,
            f64::from(added_row.rhs) * added_row.scale,
        );
        match (edited_expected, edited) {
            (None, Err(Error::Infeasible)) => {}
            (None, other) => panic!(
                "seed {seed}: edited model should be infeasible, got {other:?}; {edited_case:?}"
            ),
            (Some((objective, _)), Ok(edited)) => {
                assert_solution(&edited_case, &vars, seed, objective, &edited)
            }
            (Some(expected), other) => panic!(
                "seed {seed}: edited model expected {expected:?}, got {other:?}; {edited_case:?}"
            ),
        }
    }
}

#[test]
fn mixed_scale_active_rows_do_not_hide_a_feasible_integer_point() {
    // The second row can only be repaired by moving the first row's slack.
    // Without equilibration that tableau coefficient is 1.2e-12, below the
    // simplex's absolute pivot threshold, and the feasible model was declared
    // infeasible.
    let case = Case {
        n: 2,
        lo: [1, -2, 0, 0],
        hi: [2, 3, 0, 0],
        objective: [-2, 6, 0, 0],
        rows: vec![
            Row {
                coeffs: [4, 5, 0, 0],
                op: ComparisonOp::Ge,
                rhs: 13,
                scale: 1e6,
            },
            Row {
                coeffs: [7, 6, 0, 0],
                op: ComparisonOp::Ge,
                rhs: 26,
                scale: 1e-6,
            },
        ],
    };
    let (expected, values) = brute_force(&case).expect("fixture is feasible");
    assert_eq!((expected, values), (8, [2, 2, 0, 0]));

    let (problem, vars, _) = build_problem(&case, None);
    let solution = problem
        .solve()
        .expect("a feasible bounded ILP must not be classified as infeasible");
    assert_solution(&case, &vars, 5_155, expected, &solution);
}

#[test]
fn mixed_scale_branching_does_not_prune_the_true_optimum() {
    // The LP relaxation starts at (0.9, -2.25). Branching x <= 0 must retain
    // (0, -3), but the unscaled warm reoptimization used to report a worse LP
    // bound and prune away the true integer optimum.
    let case = Case {
        n: 2,
        lo: [-3, -3, 0, 0],
        hi: [1, 3, 0, 0],
        objective: [-4, 2, 0, 0],
        rows: vec![
            Row {
                coeffs: [-5, 6, 0, 0],
                op: ComparisonOp::Ge,
                rhs: -18,
                scale: 1e-6,
            },
            Row {
                coeffs: [-5, -2, 0, 0],
                op: ComparisonOp::Ge,
                rhs: 0,
                scale: 1e6,
            },
            Row {
                coeffs: [0, 1, 0, 0],
                op: ComparisonOp::Ge,
                rhs: -9,
                scale: 1e-6,
            },
            Row {
                coeffs: [-4, 7, 0, 0],
                op: ComparisonOp::Le,
                rhs: -6,
                scale: 1.0,
            },
        ],
    };
    let (expected, values) = brute_force(&case).expect("fixture is feasible");
    assert_eq!((expected, values), (-6, [0, -3, 0, 0]));

    let (problem, vars, _) = build_problem(&case, None);
    let solution = problem.solve().expect("fixture is bounded and feasible");
    assert_solution(&case, &vars, 2_505, expected, &solution);
}
