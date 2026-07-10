//! MPS reader for the suite's vendored benchmark files, built on the external
//! [`mps`] crate (integrated-reasoning/mps).
//!
//! Division of labor: the crate does the *lexical* work — sections, tokens,
//! numbers — and returns the raw `Parser` model (rows, column entries, RHS,
//! bounds lines, verbatim). This adapter owns the *semantics* and materializes
//! the instance through the public `Problem` API while recording a shadow
//! [`ModelSpec`] in lockstep:
//!
//! * the first `N` row is the objective; entries on any other free/unknown row
//!   are ignored (netlib convention);
//! * an RHS entry on the objective row is the *negated* objective constant
//!   (MPS convention), reported as `obj_offset`;
//! * bound resolution, including the classic MPS quirk that a negative `UP`
//!   bound with no explicit `LO` pushes the lower bound to -inf;
//! * integer columns with no BOUNDS entry default to `[0, 1]` (the MPSX-era
//!   convention MIPLIB 3 files assume);
//! * only the first RHS vector and the first BOUNDS vector are honored.
//!
//! # Integer markers are pre-scanned, not taken from the crate
//!
//! `mps` v1.0.1 recognizes `'MARKER'` `'INTORG'`/`'INTEND'` lines in COLUMNS
//! but *consumes them without recording anything* (its `columns` parser
//! filters marker lines out), so integrality would be silently lost — the
//! exact kind of misparse this suite exists to catch. [`integer_columns`]
//! recovers the integer column set with a tiny pre-scan of the COLUMNS
//! section using the same whitespace tokenization. Every MIPLIB case
//! cross-validates the outcome against published LP-relaxation *and* integer
//! optima, so a divergence between the pre-scan and the crate's tokenization
//! would surface as a failed case, not a silent wrong answer.
//!
//! # Loud on everything else
//!
//! Constructs the suite cannot faithfully represent are rejected with an
//! `Err`, never guessed at: RANGES, SOS, quadratic sections, indicator/lazy
//! constraints, cones, branching priorities, semi-continuous bounds (`SC`),
//! and an OBJSENSE that contradicts the caller's direction.

use crate::model::{Builder, ModelSpec};
use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};
use mps::types::{BoundType, ObjectiveSense, RowType};
use std::collections::{HashMap, HashSet};

pub struct ParsedMps {
    pub problem: Problem,
    pub spec: ModelSpec,
    /// Constant added to the objective (from an RHS entry on the objective row).
    pub obj_offset: f64,
}

struct RawVar {
    obj: f64,
    lo: Option<f64>,
    hi: Option<f64>,
    integer: bool,
    had_bound_entry: bool,
}

struct RawConstraint {
    op: ComparisonOp,
    terms: Vec<(usize, f64)>,
    rhs: f64,
}

/// Parse MPS text. `relax_integers` turns every integer column into a
/// continuous one (used for the LP-relaxation cases).
pub fn parse(
    text: &str,
    direction: OptimizationDirection,
    relax_integers: bool,
) -> Result<ParsedMps, String> {
    let parsed = mps::Parser::<f64>::parse(text)
        .map_err(|e| format!("mps crate failed to parse: {}", e.input))?;

    reject_unsupported_sections(&parsed, direction)?;

    let integer_cols = integer_columns(text)?;

    // ROWS: the first N row is the objective; L/G/E rows become constraints
    // in file order. Additional N rows are ignored (netlib convention).
    let mut obj_row: Option<&str> = None;
    let mut row_ops: HashMap<&str, (usize, ComparisonOp)> = HashMap::new();
    let mut constraints: Vec<RawConstraint> = vec![];
    for row in &parsed.rows {
        match row.row_type {
            RowType::Nr => {
                if obj_row.is_none() {
                    obj_row = Some(row.row_name);
                }
            }
            RowType::Leq | RowType::Geq | RowType::Eq => {
                let op = match row.row_type {
                    RowType::Leq => ComparisonOp::Le,
                    RowType::Geq => ComparisonOp::Ge,
                    _ => ComparisonOp::Eq,
                };
                row_ops.insert(row.row_name, (constraints.len(), op));
                constraints.push(RawConstraint {
                    op,
                    terms: vec![],
                    rhs: 0.0,
                });
            }
        }
    }
    let obj_row = obj_row.ok_or_else(|| "no objective (N) row found".to_string())?;

    // COLUMNS: variables in first-appearance order. Entries on the objective
    // row set the objective coefficient; entries on unknown rows (extra free
    // rows) are ignored (netlib convention).
    let mut vars: Vec<RawVar> = vec![];
    let mut var_idx: HashMap<&str, usize> = HashMap::new();
    for line in &parsed.columns {
        let idx = *var_idx.entry(line.name).or_insert_with(|| {
            vars.push(RawVar {
                obj: 0.0,
                lo: None,
                hi: None,
                integer: integer_cols.contains(line.name),
                had_bound_entry: false,
            });
            vars.len() - 1
        });
        let pairs = std::iter::once(&line.first_pair).chain(line.second_pair.as_ref());
        for pair in pairs {
            if pair.row_name == obj_row {
                vars[idx].obj = pair.value;
            } else if let Some(&(ci, _)) = row_ops.get(pair.row_name) {
                constraints[ci].terms.push((idx, pair.value));
            }
        }
    }

    // RHS: only the first RHS vector. An entry on the objective row is the
    // negated objective constant (MPS convention).
    let mut obj_offset = 0.0f64;
    let mut rhs_vector: Option<&str> = None;
    if let Some(rhs) = &parsed.rhs {
        for line in rhs {
            match rhs_vector {
                None => rhs_vector = Some(line.name),
                Some(v) if v != line.name => continue,
                _ => {}
            }
            let pairs = std::iter::once(&line.first_pair).chain(line.second_pair.as_ref());
            for pair in pairs {
                if pair.row_name == obj_row {
                    obj_offset = -pair.value;
                } else if let Some(&(ci, _)) = row_ops.get(pair.row_name) {
                    constraints[ci].rhs = pair.value;
                } else {
                    return Err(format!("RHS for unknown row {}", pair.row_name));
                }
            }
        }
    }

    // BOUNDS: only the first BOUNDS vector; per-side last-wins within it.
    let mut bounds_vector: Option<&str> = None;
    if let Some(bounds) = &parsed.bounds {
        for line in bounds {
            match bounds_vector {
                None => bounds_vector = Some(line.bound_name),
                Some(v) if v != line.bound_name => continue,
                _ => {}
            }
            let idx = *var_idx
                .get(line.column_name)
                .ok_or_else(|| format!("bound for unknown column {}", line.column_name))?;
            let v = &mut vars[idx];
            v.had_bound_entry = true;
            let num = || -> Result<f64, String> {
                line.value.ok_or_else(|| {
                    format!(
                        "missing bound value for column {} ({:?})",
                        line.column_name, line.bound_type
                    )
                })
            };
            match line.bound_type {
                BoundType::Up => v.hi = Some(num()?),
                BoundType::Lo => v.lo = Some(num()?),
                BoundType::Fx => {
                    let x = num()?;
                    v.lo = Some(x);
                    v.hi = Some(x);
                }
                BoundType::Fr => {
                    v.lo = Some(f64::NEG_INFINITY);
                    v.hi = Some(f64::INFINITY);
                }
                BoundType::Mi => v.lo = Some(f64::NEG_INFINITY),
                BoundType::Pl => v.hi = Some(f64::INFINITY),
                BoundType::Bv => {
                    v.lo = Some(0.0);
                    v.hi = Some(1.0);
                    v.integer = true;
                }
                BoundType::Ui => {
                    v.hi = Some(num()?);
                    v.integer = true;
                }
                BoundType::Li => {
                    v.lo = Some(num()?);
                    v.integer = true;
                }
                // Semi-continuous variables are not representable in microlp.
                BoundType::Sc => {
                    return Err(format!(
                        "unsupported bound type SC (semi-continuous) for column {}",
                        line.column_name
                    ))
                }
            }
        }
    }

    // Materialize through the public API, recording the shadow model.
    let mut b = Builder::new(direction);
    for v in &vars {
        let integer = v.integer && !relax_integers;
        let (lo, hi) = resolve_bounds(v);
        if integer {
            let lo_i = lo.max(i32::MIN as f64) as i32;
            let hi_i = hi.min(i32::MAX as f64) as i32;
            b.integer(v.obj, lo_i, hi_i);
        } else {
            b.real(v.obj, lo, hi);
        }
    }
    for c in &constraints {
        let terms: Vec<(Variable, f64)> = c.terms.iter().map(|&(vi, x)| (b.vars[vi], x)).collect();
        b.constraint(&terms, c.op, c.rhs);
    }

    // Sanity check on the adapter itself.
    assert_eq!(b.spec.vars.len(), vars.len());

    Ok(ParsedMps {
        problem: b.problem,
        spec: b.spec,
        obj_offset,
    })
}

/// Reject every parsed section the suite cannot faithfully represent — a loud
/// error beats a silent misparse. RANGES stays unsupported (no vendored file
/// uses it; adding it means implementing the U_i/L_i table for real).
fn reject_unsupported_sections(
    parsed: &mps::Parser<'_, f64>,
    direction: OptimizationDirection,
) -> Result<(), String> {
    if let Some(ranges) = &parsed.ranges {
        if !ranges.is_empty() {
            return Err("RANGES section not supported by suite reader".to_string());
        }
    }
    if parsed
        .special_ordered_sets
        .as_ref()
        .is_some_and(|s| !s.is_empty())
    {
        return Err("SOS section not supported by suite reader".to_string());
    }
    if parsed
        .quadratic_objective
        .as_ref()
        .is_some_and(|q| !q.is_empty())
        || parsed
            .quadratic_constraints
            .as_ref()
            .is_some_and(|q| !q.is_empty())
    {
        return Err("quadratic sections not supported by suite reader".to_string());
    }
    if parsed.indicators.as_ref().is_some_and(|i| !i.is_empty()) {
        return Err("INDICATORS section not supported by suite reader".to_string());
    }
    if parsed
        .lazy_constraints
        .as_ref()
        .is_some_and(|l| !l.is_empty())
    {
        return Err("LAZYCONS section not supported by suite reader".to_string());
    }
    if parsed
        .cone_constraints
        .as_ref()
        .is_some_and(|c| !c.is_empty())
    {
        return Err("CSECTION (cones) not supported by suite reader".to_string());
    }
    if parsed.user_cuts.as_ref().is_some_and(|u| !u.is_empty()) {
        return Err("USERCUTS section not supported by suite reader".to_string());
    }
    if parsed
        .branch_priorities
        .as_ref()
        .is_some_and(|b| !b.is_empty())
    {
        return Err("BRANCH section not supported by suite reader".to_string());
    }
    // OBJSENSE: the caller states the direction for every vendored file; a
    // contradicting in-file sense would flip every expected value, so treat
    // it as an error rather than trusting either side silently.
    if let Some(sense) = parsed.objective_sense {
        let matches = matches!(
            (sense, direction),
            (ObjectiveSense::Min, OptimizationDirection::Minimize)
                | (ObjectiveSense::Max, OptimizationDirection::Maximize)
        );
        if !matches {
            return Err(format!(
                "OBJSENSE {:?} contradicts the caller's direction {:?}",
                sense, direction
            ));
        }
    }
    Ok(())
}

/// Recover the set of integer columns from `'MARKER'` `'INTORG'`/`'INTEND'`
/// blocks in the COLUMNS section. See the module docs for why the `mps` crate
/// cannot provide this (v1.0.1 drops marker lines during parsing).
fn integer_columns(text: &str) -> Result<HashSet<&str>, String> {
    let mut set = HashSet::new();
    let mut in_columns = false;
    let mut in_integer_block = false;
    for (lineno, line) in text.lines().enumerate() {
        let err = |msg: String| format!("line {}: {}", lineno + 1, msg);
        if line.starts_with('*') || line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(' ') {
            in_columns = line.split_whitespace().next() == Some("COLUMNS");
            continue;
        }
        if !in_columns {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let first = tokens.next();
        let second = tokens.next();
        if second == Some("'MARKER'") {
            match line.split_whitespace().last() {
                Some("'INTORG'") => in_integer_block = true,
                Some("'INTEND'") => in_integer_block = false,
                other => return Err(err(format!("unknown marker {:?}", other))),
            }
            continue;
        }
        if in_integer_block {
            if let Some(col) = first {
                set.insert(col);
            }
        }
    }
    if in_integer_block {
        return Err("INTORG block not closed by an INTEND marker".to_string());
    }
    Ok(set)
}

fn resolve_bounds(v: &RawVar) -> (f64, f64) {
    // Integer columns with no BOUNDS entry at all default to [0, 1] (MPSX
    // convention, what MIPLIB 3 assumes). Otherwise missing sides default to
    // lo = 0, hi = +inf — with the classic MPS quirk that a negative UP bound
    // with no explicit LO pushes the lower bound to -inf.
    if v.integer && !v.had_bound_entry {
        return (0.0, 1.0);
    }
    let lo = match (v.lo, v.hi) {
        (Some(lo), _) => lo,
        (None, Some(hi)) if hi < 0.0 => f64::NEG_INFINITY,
        _ => 0.0,
    };
    let hi = v.hi.unwrap_or(f64::INFINITY);
    (lo, hi)
}
