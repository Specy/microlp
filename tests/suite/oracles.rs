//! Exact combinatorial oracles. Each computes the true optimum of a small
//! instance by dynamic programming or exhaustive enumeration in integer
//! arithmetic, giving the suite an answer that does not depend on any
//! floating-point solver.

/// 0/1 knapsack: maximize sum(values) with sum(weights) <= capacity.
pub fn knapsack01(values: &[i64], weights: &[i64], capacity: i64) -> i64 {
    let cap = capacity as usize;
    let mut best = vec![0i64; cap + 1];
    for (i, &w) in weights.iter().enumerate() {
        let w = w as usize;
        for c in (w..=cap).rev() {
            best[c] = best[c].max(best[c - w] + values[i]);
        }
    }
    best[cap]
}

/// Bounded knapsack: item i can be picked 0..=counts[i] times.
pub fn knapsack_bounded(values: &[i64], weights: &[i64], counts: &[i64], capacity: i64) -> i64 {
    let cap = capacity as usize;
    let mut best = vec![0i64; cap + 1];
    for i in 0..values.len() {
        // Expand multiplicity naively; instances are tiny.
        for _ in 0..counts[i] {
            let w = weights[i] as usize;
            for c in (w..=cap).rev() {
                best[c] = best[c].max(best[c - w] + values[i]);
            }
        }
    }
    best[cap]
}

/// Subset-sum feasibility: can some subset of `weights` sum exactly to `target`?
pub fn subset_sum(weights: &[i64], target: i64) -> bool {
    let t = target as usize;
    let mut reachable = vec![false; t + 1];
    reachable[0] = true;
    for &w in weights {
        let w = w as usize;
        if w > t {
            continue;
        }
        for c in (w..=t).rev() {
            if reachable[c - w] {
                reachable[c] = true;
            }
        }
    }
    reachable[t]
}

/// Assignment problem: minimize total cost of a perfect matching on an n x n
/// cost matrix, by brute force over all permutations (n <= 8).
pub fn assignment_min(costs: &[Vec<i64>]) -> i64 {
    let n = costs.len();
    assert!(n <= 8, "brute force assignment limited to n <= 8");
    let mut perm: Vec<usize> = (0..n).collect();
    let mut best = i64::MAX;
    permute(&mut perm, 0, &mut |p| {
        let cost: i64 = p.iter().enumerate().map(|(i, &j)| costs[i][j]).sum();
        if cost < best {
            best = cost;
        }
    });
    best
}

fn permute(items: &mut Vec<usize>, k: usize, visit: &mut impl FnMut(&[usize])) {
    if k == items.len() {
        visit(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// Minimum number of coins summing exactly to `target`; None if impossible.
pub fn coin_change_min(denoms: &[i64], target: i64) -> Option<i64> {
    let t = target as usize;
    let mut best = vec![i64::MAX; t + 1];
    best[0] = 0;
    for c in 1..=t {
        for &d in denoms {
            let d = d as usize;
            if d <= c && best[c - d] != i64::MAX {
                best[c] = best[c].min(best[c - d] + 1);
            }
        }
    }
    (best[t] != i64::MAX).then_some(best[t])
}

/// A tiny pure-integer linear program, solved exactly by enumerating the whole
/// bounded box. All data is integer so evaluation is exact in i64.
pub struct TinyIlp {
    /// (lo, hi) inclusive box per variable.
    pub bounds: Vec<(i64, i64)>,
    pub objective: Vec<i64>,
    /// (coefficients, op, rhs); op: -1 => <=, 0 => ==, 1 => >=
    pub constraints: Vec<(Vec<i64>, i8, i64)>,
    pub maximize: bool,
}

impl TinyIlp {
    /// Exhaustively enumerate the box; returns the optimal objective, or None
    /// if no feasible point exists.
    pub fn brute_force(&self) -> Option<i64> {
        let n = self.bounds.len();
        let mut point: Vec<i64> = self.bounds.iter().map(|b| b.0).collect();
        let mut best: Option<i64> = None;
        loop {
            let feasible = self.constraints.iter().all(|(coeffs, op, rhs)| {
                let lhs: i64 = coeffs.iter().zip(&point).map(|(c, x)| c * x).sum();
                match op {
                    -1 => lhs <= *rhs,
                    0 => lhs == *rhs,
                    1 => lhs >= *rhs,
                    _ => unreachable!(),
                }
            });
            if feasible {
                let obj: i64 = self.objective.iter().zip(&point).map(|(c, x)| c * x).sum();
                best = Some(match best {
                    None => obj,
                    Some(b) if self.maximize => b.max(obj),
                    Some(b) => b.min(obj),
                });
            }
            // Mixed-radix increment with carry; done when every digit overflows.
            let mut i = 0;
            loop {
                if i == n {
                    return best;
                }
                if point[i] < self.bounds[i].1 {
                    point[i] += 1;
                    break;
                }
                point[i] = self.bounds[i].0;
                i += 1;
            }
        }
    }
}
