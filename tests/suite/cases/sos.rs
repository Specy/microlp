//! Special ordered sets against brute-force enumeration: an SOS1 optimum is
//! the best over "one member free, the rest fixed to zero", an SOS2 optimum
//! the best over adjacent member pairs. Every restricted problem is solved
//! without sets, so the oracle exercises only the plain LP / MILP paths.
//!
//! Families: continuous members under random knapsack-like rows, members
//! shared with integer variables, several sets over a common pool, and
//! piecewise-linear λ-models of random nonconvex functions (SOS2).

use super::{Case, Tier};
use crate::rng::Rng;
use microlp::ComparisonOp::{Eq, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};
use microlp::{Error, OptimizationDirection, Problem, Solution, SolveOutcome, Variable};
use std::time::Duration;

const OBJ_TOL: f64 = 1e-6;

fn require_solution(outcome: SolveOutcome, context: &str) -> Result<Solution, String> {
    crate::verify::require_solution(outcome, context)
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Sos1,
    Sos2,
}

/// A random model: `build` fills a problem and returns the members of each
/// set (one or two sets), without declaring the sets themselves.
struct Model {
    direction: OptimizationDirection,
    kind: Kind,
    build: Box<dyn Fn(&mut Problem) -> Vec<Vec<Variable>> + Send + Sync>,
}

fn solve_restricted(
    model: &Model,
    free: &[Vec<usize>],
    budget: Duration,
) -> Result<Option<f64>, String> {
    let mut p = Problem::new(model.direction);
    let sets = (model.build)(&mut p);
    for (set, free) in sets.iter().zip(free) {
        for (i, &m) in set.iter().enumerate() {
            if !free.contains(&i) {
                p.add_constraint([(m, 1.0)], Eq, 0.0);
            }
        }
    }
    p.set_time_limit(budget);
    match p.solve() {
        Ok(outcome) => Ok(Some(require_solution(outcome, "oracle solve")?.objective())),
        Err(Error::Infeasible) => Ok(None),
        Err(e) => Err(format!("oracle solve failed: {e}")),
    }
}

/// Every choice of free positions allowed by `kind` for a set of `n` members.
fn choices(kind: Kind, n: usize) -> Vec<Vec<usize>> {
    match kind {
        Kind::Sos1 => {
            let mut c = vec![vec![]];
            c.extend((0..n).map(|i| vec![i]));
            c
        }
        Kind::Sos2 => (0..n.saturating_sub(1)).map(|i| vec![i, i + 1]).collect(),
    }
}

fn oracle(model: &Model, sizes: &[usize], budget: Duration) -> Result<Option<f64>, String> {
    let per_set: Vec<Vec<Vec<usize>>> = sizes.iter().map(|&n| choices(model.kind, n)).collect();
    let mut best: Option<f64> = None;
    let mut combo = vec![0usize; sizes.len()];
    loop {
        let free: Vec<Vec<usize>> = combo
            .iter()
            .enumerate()
            .map(|(s, &c)| per_set[s][c].clone())
            .collect();
        if let Some(obj) = solve_restricted(model, &free, budget)? {
            best = Some(match (best, model.direction) {
                (None, _) => obj,
                (Some(b), Minimize) => b.min(obj),
                (Some(b), Maximize) => b.max(obj),
            });
        }
        // Next combination (mixed radix).
        let mut s = 0;
        loop {
            if s == combo.len() {
                return Ok(best);
            }
            combo[s] += 1;
            if combo[s] < per_set[s].len() {
                break;
            }
            combo[s] = 0;
            s += 1;
        }
    }
}

fn check_set(kind: Kind, sol: &Solution, members: &[Variable]) -> Result<(), String> {
    let nonzero: Vec<usize> = (0..members.len())
        .filter(|&i| sol.var_value(members[i]).abs() > 1e-7)
        .collect();
    let ok = match kind {
        Kind::Sos1 => nonzero.len() <= 1,
        Kind::Sos2 => nonzero.len() <= 2 && nonzero.windows(2).all(|w| w[1] == w[0] + 1),
    };
    if ok {
        Ok(())
    } else {
        Err(format!("set violated: nonzero positions {nonzero:?}"))
    }
}

fn run_model(model: Model, budget: Duration) -> Result<(), String> {
    let mut p = Problem::new(model.direction);
    let sets = (model.build)(&mut p);
    let sizes: Vec<usize> = sets.iter().map(|s| s.len()).collect();
    for set in &sets {
        let weighted = set.iter().enumerate().map(|(i, &m)| (m, i as f64));
        match model.kind {
            Kind::Sos1 => p.add_sos1_constraint(weighted),
            Kind::Sos2 => p.add_sos2_constraint(weighted),
        }
    }
    p.set_time_limit(budget / 2);
    let want = oracle(&model, &sizes, budget / 8)?;
    match (p.solve(), want) {
        (Err(Error::Infeasible), None) => Ok(()),
        (Err(Error::Infeasible), Some(w)) => Err(format!("Infeasible, oracle found {w}")),
        (Err(e), _) => Err(format!("solve failed: {e}")),
        (Ok(_), None) => Err("solved a model the oracle finds infeasible".into()),
        (Ok(outcome), Some(w)) => {
            let sol = require_solution(outcome, "sos solve")?;
            for set in &sets {
                check_set(model.kind, &sol, set)?;
            }
            let got = sol.objective();
            if (got - w).abs() > OBJ_TOL + OBJ_TOL * w.abs() {
                return Err(format!("objective {got}, oracle {w}"));
            }
            Ok(())
        }
    }
}

fn frac(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (rng.int(0, 1000) as f64 / 1000.0) * (hi - lo)
}

/// Continuous members, one set, random `≤` rows with nonnegative coefficients
/// and a random objective of either sign.
fn continuous_model(seed: u64, kind: Kind) -> Model {
    let mut rng = Rng::new(seed);
    let n = rng.usize(3, 7);
    let rows = rng.usize(1, 3);
    let direction = if rng.int(0, 1) == 0 {
        Minimize
    } else {
        Maximize
    };
    let obj: Vec<f64> = (0..n).map(|_| frac(&mut rng, -5.0, 5.0)).collect();
    let data: Vec<(Vec<f64>, f64)> = (0..rows)
        .map(|_| {
            (
                (0..n).map(|_| frac(&mut rng, 0.0, 3.0)).collect(),
                frac(&mut rng, 2.0, 8.0),
            )
        })
        .collect();
    Model {
        direction,
        kind,
        build: Box::new(move |p| {
            let vars: Vec<Variable> = obj.iter().map(|&c| p.add_var(c, (0.0, 4.0))).collect();
            for (coeffs, rhs) in &data {
                let terms: Vec<(Variable, f64)> =
                    vars.iter().zip(coeffs).map(|(&v, &c)| (v, c)).collect();
                p.add_constraint(terms, Le, *rhs);
            }
            vec![vars]
        }),
    }
}

/// Members coupled to integer variables: each member `m_i` needs `m_i ≤ 2 k_i`
/// with `k_i ∈ {0, 1, 2}` costing something, under a shared capacity.
fn mixed_model(seed: u64, kind: Kind) -> Model {
    let mut rng = Rng::new(seed);
    let n = rng.usize(3, 6);
    let gain: Vec<f64> = (0..n).map(|_| frac(&mut rng, 1.0, 6.0)).collect();
    let cost: Vec<f64> = (0..n).map(|_| frac(&mut rng, 0.5, 3.0)).collect();
    let cap = frac(&mut rng, 3.0, 9.0);
    Model {
        direction: Maximize,
        kind,
        build: Box::new(move |p| {
            let members: Vec<Variable> = gain.iter().map(|&g| p.add_var(g, (0.0, 4.0))).collect();
            let ks: Vec<Variable> = cost
                .iter()
                .map(|&c| p.add_integer_var(-c, (0, 2)))
                .collect();
            for (&m, &k) in members.iter().zip(&ks) {
                p.add_constraint([(m, 1.0), (k, -2.0)], Le, 0.0);
            }
            let terms: Vec<(Variable, f64)> = members.iter().map(|&m| (m, 1.0)).collect();
            p.add_constraint(terms, Le, cap);
            vec![members]
        }),
    }
}

/// Two sets over a common pool of variables, sharing one member.
fn two_sets_model(seed: u64) -> Model {
    let mut rng = Rng::new(seed);
    let obj: Vec<f64> = (0..5).map(|_| frac(&mut rng, 0.5, 5.0)).collect();
    let cap = frac(&mut rng, 2.0, 6.0);
    Model {
        direction: Maximize,
        kind: Kind::Sos1,
        build: Box::new(move |p| {
            let v: Vec<Variable> = obj.iter().map(|&c| p.add_var(c, (0.0, 3.0))).collect();
            let terms: Vec<(Variable, f64)> = v.iter().map(|&x| (x, 1.0)).collect();
            p.add_constraint(terms, Le, cap);
            vec![vec![v[0], v[1], v[2]], vec![v[2], v[3], v[4]]]
        }),
    }
}

/// Piecewise-linear λ-model of a random nonconvex function evaluated at a
/// random abscissa, minimised: the SOS2 answer is the function value on the
/// segment containing the abscissa, the relaxation the convex hull.
fn pwl_model(seed: u64) -> Model {
    let mut rng = Rng::new(seed);
    let k = rng.usize(4, 7);
    let values: Vec<f64> = (0..k).map(|_| frac(&mut rng, 0.0, 10.0)).collect();
    let x = frac(&mut rng, 0.0, (k - 1) as f64);
    Model {
        direction: Minimize,
        kind: Kind::Sos2,
        build: Box::new(move |p| {
            let lam: Vec<Variable> = values.iter().map(|&f| p.add_var(f, (0.0, 1.0))).collect();
            p.add_constraint(lam.iter().map(|&l| (l, 1.0)), Eq, 1.0);
            p.add_constraint(lam.iter().enumerate().map(|(i, &l)| (l, i as f64)), Eq, x);
            vec![lam]
        }),
    }
}

pub fn register(cases: &mut Vec<Case>) {
    for seed in 0..6u64 {
        for (kind, tag) in [(Kind::Sos1, "sos1"), (Kind::Sos2, "sos2")] {
            let name = format!("sos/continuous-{tag}/s{seed:02}");
            cases.push(Case::custom(name, Tier::Easy, 20, move |budget| {
                run_model(continuous_model(1000 + seed, kind), budget)
            }));
            let name = format!("sos/mixed-int-{tag}/s{seed:02}");
            cases.push(Case::custom(name, Tier::Easy, 30, move |budget| {
                run_model(mixed_model(2000 + seed, kind), budget)
            }));
        }
        let name = format!("sos/two-sets/s{seed:02}");
        cases.push(Case::custom(name, Tier::Easy, 20, move |budget| {
            run_model(two_sets_model(3000 + seed), budget)
        }));
        let name = format!("sos/pwl-nonconvex/s{seed:02}");
        cases.push(Case::custom(name, Tier::Easy, 20, move |budget| {
            run_model(pwl_model(4000 + seed), budget)
        }));
    }
}
