//! CPLEX-LP reader for the suite's MILPBench instances, built on the external
//! [`lp_parser_rs`] crate (dandxy89/lp_parser_rs).
//!
//! Division of labor: the crate's lexer + LALRPOP grammar turn the text into a
//! raw [`ParseResult`] — sense, assembled objectives/constraints, and the
//! Bounds/Generals/Integers/Binaries sections verbatim. This adapter consumes
//! that raw result and owns the *variable-domain semantics*, materializing the
//! instance through the public `Problem` API while recording a shadow
//! [`ModelSpec`] in lockstep.
//!
//! # Why the raw `ParseResult` and not the crate's `LpProblem`
//!
//! `LpProblem`'s section merge keeps a variable's bound-derived type and only
//! upgrades variables that are still `Free` when the Generals/Integers section
//! is applied (`apply_variable_type` in `lp_parser_rs::problem`). A variable
//! declared both in `Bounds` and in `Generals` would therefore silently lose
//! its integrality — the exact kind of misparse this suite exists to catch.
//! Consuming the raw sections lets the adapter apply the CPLEX semantics
//! this suite relies on:
//!
//! * default domain is `[0, +inf)`; `free` means `(-inf, +inf)`;
//! * `Binaries` forces `{0, 1}` integer, overriding explicit bounds (CPLEX);
//! * `Generals`/`Integers` marks the variable integer, keeping its bounds;
//! * bound values with `|v| >= 1e30` are Gurobi's infinity sentinel;
//! * duplicate terms in one expression are merged (sprs `CsVec` needs unique
//!   indices).
//!
//! Ranged constraints (`1 <= x + y <= 3`) are supported: the crate's
//! assembler expands them into two rows (matching the MPS RANGES expansion),
//! which is exactly representable in microlp.
//!
//! # Loud on everything else
//!
//! Constructs microlp cannot represent are rejected with an `Err`, never
//! guessed at: SOS constraints, semi-continuous variables, multiple
//! objectives, constraints with no variables. Syntax the grammar does not
//! accept (including a missing objective sense) surfaces as a parse error.

use crate::model::{Builder, ModelSpec};
use lp_parser_rs::error::LpParseError;
use lp_parser_rs::lexer::{Lexer, ParseResult, RawConstraint};
use lp_parser_rs::lp::LpProblemParser;
use lp_parser_rs::model::{ComparisonOp as LpOp, Sense, VariableType};
use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};
use std::collections::HashMap;

/// Bound values at or beyond this magnitude are treated as infinite
/// (Gurobi emits `1e30` as its infinity sentinel in LP files).
const INF_SENTINEL: f64 = 1e30;

/// A materialization-ready constraint: merged `(var index, coefficient)`
/// terms, its relational operator, and its right-hand side.
type ParsedConstraint = (Vec<(usize, f64)>, ComparisonOp, f64);

pub struct ParsedLp {
    pub problem: Problem,
    pub spec: ModelSpec,
    /// Constant term of the objective. microlp's `Problem` has no objective
    /// constant, so it is reported here and excluded from the solved
    /// objective (see the `objective-offset` case). The structural size
    /// (vars / constraints / nonzeros) is read off `spec`.
    pub obj_offset: f64,
    /// The optimization sense declared by the file (`Maximize`/`Minimize`),
    /// for callers that need direction-aware assertions.
    pub direction: OptimizationDirection,
}

#[derive(Clone)]
struct RawVar {
    obj: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    /// Integer from a `Generals`/`Integers` list (bounds kept).
    general: bool,
    /// Binary {0,1}, from a `Binaries` list (overrides bounds, CPLEX semantics).
    binary: bool,
}

impl RawVar {
    fn new() -> RawVar {
        RawVar {
            obj: 0.0,
            lo: None,
            hi: None,
            general: false,
            binary: false,
        }
    }
}

/// Variables in first-appearance order plus a name index, so the shadow model
/// and the `Problem` are built from one consistent ordering.
struct VarTable<'a> {
    vars: Vec<RawVar>,
    names: Vec<&'a str>,
    index: HashMap<&'a str, usize>,
}

impl<'a> VarTable<'a> {
    fn new() -> Self {
        VarTable {
            vars: vec![],
            names: vec![],
            index: HashMap::new(),
        }
    }

    fn var(&mut self, name: &'a str) -> usize {
        if let Some(&idx) = self.index.get(name) {
            return idx;
        }
        self.vars.push(RawVar::new());
        self.names.push(name);
        self.index.insert(name, self.vars.len() - 1);
        self.vars.len() - 1
    }
}

/// Parse CPLEX LP text. `relax_integers` turns every integer/binary variable
/// into a continuous one.
pub fn parse(text: &str, relax_integers: bool) -> Result<ParsedLp, String> {
    let lexer = Lexer::new(text);
    let parsed: ParseResult = LpProblemParser::new()
        .parse(lexer)
        .map_err(|e| format!("LP parse failed: {}", LpParseError::from(e)))?;

    let direction = match parsed.sense {
        Sense::Minimize => OptimizationDirection::Minimize,
        Sense::Maximize => OptimizationDirection::Maximize,
    };

    if parsed.objectives.len() != 1 {
        return Err(format!(
            "expected exactly one objective, found {}",
            parsed.objectives.len()
        ));
    }
    if !parsed.sos.is_empty() {
        return Err("SOS constraints are not representable in microlp".to_string());
    }
    if !parsed.semi_continuous.is_empty() {
        return Err("semi-continuous variables are not representable in microlp".to_string());
    }

    let mut t = VarTable::new();

    // Objective: merge duplicate mentions of a variable by summing.
    let objective = &parsed.objectives[0];
    let obj_offset = objective.constant;
    for c in &objective.coefficients {
        let idx = t.var(c.name);
        t.vars[idx].obj += c.value;
    }

    // Constraints: merged terms in first-appearance order, one row each.
    // (Ranged rows arrive here already expanded into two Standard entries.)
    let mut constraints: Vec<ParsedConstraint> = vec![];
    for rc in &parsed.constraints {
        match rc {
            RawConstraint::Standard {
                name,
                coefficients,
                operator,
                rhs,
                ..
            } => {
                if coefficients.is_empty() {
                    return Err(format!(
                        "constraint {} has no variables on either side",
                        name
                    ));
                }
                let mut terms: Vec<(usize, f64)> = vec![];
                let mut term_pos: HashMap<usize, usize> = HashMap::new();
                for c in coefficients {
                    let idx = t.var(c.name);
                    match term_pos.get(&idx) {
                        Some(&pos) => terms[pos].1 += c.value,
                        None => {
                            term_pos.insert(idx, terms.len());
                            terms.push((idx, c.value));
                        }
                    }
                }
                // CPLEX treats strict `<`/`>` as `<=`/`>=`.
                let op = match operator {
                    LpOp::LT | LpOp::LTE => ComparisonOp::Le,
                    LpOp::GT | LpOp::GTE => ComparisonOp::Ge,
                    LpOp::EQ => ComparisonOp::Eq,
                };
                constraints.push((terms, op, *rhs));
            }
            RawConstraint::SOS { name, .. } => {
                return Err(format!(
                    "SOS constraint {} is not representable in microlp",
                    name
                ));
            }
        }
    }

    // Bounds section, in declaration order (per-side last-wins). The grammar
    // only emits bound-shaped variable types here; anything else would be a
    // grammar change worth failing loudly on.
    for (name, vt) in &parsed.bounds {
        let idx = t.var(name);
        let v = &mut t.vars[idx];
        match vt {
            VariableType::Free => {
                v.lo = Some(f64::NEG_INFINITY);
                v.hi = Some(f64::INFINITY);
            }
            VariableType::LowerBound(l) => v.lo = Some(desentinel(*l)),
            VariableType::UpperBound(u) => v.hi = Some(desentinel(*u)),
            VariableType::DoubleBound(l, u) => {
                v.lo = Some(desentinel(*l));
                v.hi = Some(desentinel(*u));
            }
            other => {
                return Err(format!(
                    "unexpected variable type {} in Bounds for {}",
                    other, name
                ));
            }
        }
    }

    for name in parsed.generals.iter().chain(parsed.integers.iter()) {
        let idx = t.var(name);
        t.vars[idx].general = true;
    }
    for name in &parsed.binaries {
        let idx = t.var(name);
        t.vars[idx].binary = true;
    }

    // Materialize through the public API, recording the shadow model.
    let mut b = Builder::new(direction);
    for v in &t.vars {
        let (lo, hi) = resolve_bounds(v);
        let integer = (v.binary || v.general) && !relax_integers;
        if integer {
            b.integer(v.obj, clamp_i32(lo), clamp_i32(hi));
        } else {
            b.real(v.obj, lo, hi);
        }
    }
    for (terms, op, rhs) in &constraints {
        let terms: Vec<(Variable, f64)> = terms.iter().map(|&(vi, x)| (b.vars[vi], x)).collect();
        b.constraint(&terms, *op, *rhs);
    }

    // Sanity check on the adapter itself.
    assert_eq!(b.spec.vars.len(), t.vars.len());

    Ok(ParsedLp {
        problem: b.problem,
        spec: b.spec,
        obj_offset,
        direction,
    })
}

/// Map Gurobi's `1e30` infinity sentinel onto a real infinity.
fn desentinel(v: f64) -> f64 {
    if v >= INF_SENTINEL {
        f64::INFINITY
    } else if v <= -INF_SENTINEL {
        f64::NEG_INFINITY
    } else {
        v
    }
}

/// Resolve a variable's effective bounds. Binaries are forced to {0,1} (CPLEX
/// overrides any explicit bound). Otherwise the default lower bound is 0 and the
/// default upper bound is +inf; a `free` var carries explicit -inf/+inf.
fn resolve_bounds(v: &RawVar) -> (f64, f64) {
    if v.binary {
        return (0.0, 1.0);
    }
    let lo = v.lo.unwrap_or(0.0);
    let hi = v.hi.unwrap_or(f64::INFINITY);
    (lo, hi)
}

fn clamp_i32(x: f64) -> i32 {
    if x <= i32::MIN as f64 {
        i32::MIN
    } else if x >= i32::MAX as f64 {
        i32::MAX
    } else {
        x.round() as i32
    }
}
