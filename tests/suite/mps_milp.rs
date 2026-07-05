//! Minimal MPS reader used by the suite for the vendored benchmark files.
//!
//! microlp's own `MpsFile` parser has no INTORG/INTEND (integer marker)
//! support, so the MIPLIB instances are read with this independent reader and
//! materialized through the public `Problem` API. It doubles as an independent
//! parse of the netlib LP files: those are additionally parsed with `MpsFile`
//! by the netlib cases, and any disagreement shows up as a failed validation.
//!
//! Supported: ROWS (N/L/G/E), COLUMNS with integer markers, RHS (including an
//! entry on the objective row, which by MPS convention is the *negated*
//! objective constant), BOUNDS (UP, LO, FX, FR, MI, PL, BV, UI, LI).
//! Integer columns with no BOUNDS entry default to [0, 1], the MPSX-era
//! convention MIPLIB 3 files assume; every vendored instance's parse is
//! validated against its published LP-relaxation objective.

use crate::model::{Builder, ModelSpec};
use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};
use std::collections::HashMap;

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
    let mut section = String::new();
    let mut obj_row: Option<String> = None;
    let mut row_ops: HashMap<String, (usize, ComparisonOp)> = HashMap::new();
    let mut constraints: Vec<RawConstraint> = vec![];
    let mut vars: Vec<RawVar> = vec![];
    let mut var_idx: HashMap<String, usize> = HashMap::new();
    let mut in_integer_block = false;
    let mut obj_offset = 0.0f64;
    let mut rhs_vector: Option<String> = None;
    let mut bounds_vector: Option<String> = None;

    for (lineno, raw_line) in text.lines().enumerate() {
        let err = |msg: &str| format!("line {}: {}", lineno + 1, msg);
        if raw_line.starts_with('*') || raw_line.trim().is_empty() {
            continue;
        }
        if !raw_line.starts_with(' ') {
            let mut it = raw_line.split_whitespace();
            section = it.next().unwrap_or("").to_string();
            match section.as_str() {
                "NAME" | "ROWS" | "COLUMNS" | "RHS" | "BOUNDS" | "ENDATA" => continue,
                "RANGES" => return Err(err("RANGES section not supported by suite reader")),
                other => return Err(err(&format!("unsupported section {}", other))),
            }
        }

        let tokens: Vec<&str> = raw_line.split_whitespace().collect();
        match section.as_str() {
            "ROWS" => {
                if tokens.len() != 2 {
                    return Err(err("malformed ROWS line"));
                }
                let (ty, name) = (tokens[0], tokens[1]);
                match ty {
                    "N" => {
                        if obj_row.is_none() {
                            obj_row = Some(name.to_string());
                        }
                        // Additional free rows are ignored (netlib convention).
                    }
                    "L" | "G" | "E" => {
                        let op = match ty {
                            "L" => ComparisonOp::Le,
                            "G" => ComparisonOp::Ge,
                            _ => ComparisonOp::Eq,
                        };
                        row_ops.insert(name.to_string(), (constraints.len(), op));
                        constraints.push(RawConstraint {
                            op,
                            terms: vec![],
                            rhs: 0.0,
                        });
                    }
                    other => return Err(err(&format!("unknown row type {}", other))),
                }
            }
            "COLUMNS" => {
                if tokens.len() >= 3 && tokens[1] == "'MARKER'" {
                    match *tokens.last().unwrap() {
                        "'INTORG'" => in_integer_block = true,
                        "'INTEND'" => in_integer_block = false,
                        other => return Err(err(&format!("unknown marker {}", other))),
                    }
                    continue;
                }
                if tokens.len() < 3 || tokens.len() % 2 == 0 {
                    return Err(err("malformed COLUMNS line"));
                }
                let col = tokens[0];
                let idx = *var_idx.entry(col.to_string()).or_insert_with(|| {
                    vars.push(RawVar {
                        obj: 0.0,
                        lo: None,
                        hi: None,
                        integer: in_integer_block,
                        had_bound_entry: false,
                    });
                    vars.len() - 1
                });
                for pair in tokens[1..].chunks(2) {
                    let (row, val) = (pair[0], pair[1]);
                    let val: f64 = val
                        .parse()
                        .map_err(|_| err(&format!("bad number {}", val)))?;
                    if Some(row) == obj_row.as_deref() {
                        vars[idx].obj = val;
                    } else if let Some(&(ci, _)) = row_ops.get(row) {
                        constraints[ci].terms.push((idx, val));
                    }
                    // Entries on other free rows are ignored.
                }
            }
            "RHS" => {
                if tokens.len() < 3 || tokens.len() % 2 == 0 {
                    return Err(err("malformed RHS line"));
                }
                match &rhs_vector {
                    None => rhs_vector = Some(tokens[0].to_string()),
                    Some(v) if v != tokens[0] => continue, // only first RHS vector
                    _ => {}
                }
                for pair in tokens[1..].chunks(2) {
                    let (row, val) = (pair[0], pair[1]);
                    let val: f64 = val
                        .parse()
                        .map_err(|_| err(&format!("bad number {}", val)))?;
                    if Some(row) == obj_row.as_deref() {
                        // MPS convention: RHS on the objective row negates the constant.
                        obj_offset = -val;
                    } else if let Some(&(ci, _)) = row_ops.get(row) {
                        constraints[ci].rhs = val;
                    } else {
                        return Err(err(&format!("RHS for unknown row {}", row)));
                    }
                }
            }
            "BOUNDS" => {
                if tokens.len() < 3 {
                    return Err(err("malformed BOUNDS line"));
                }
                match &bounds_vector {
                    None => bounds_vector = Some(tokens[1].to_string()),
                    Some(v) if v != tokens[1] => continue, // only first BOUNDS vector
                    _ => {}
                }
                let ty = tokens[0];
                let var = tokens[2];
                let idx = *var_idx
                    .get(var)
                    .ok_or_else(|| err(&format!("bound for unknown column {}", var)))?;
                let v = &mut vars[idx];
                v.had_bound_entry = true;
                let num = || -> Result<f64, String> {
                    tokens
                        .get(3)
                        .ok_or_else(|| err("missing bound value"))?
                        .parse()
                        .map_err(|_| err("bad bound value"))
                };
                match ty {
                    "UP" => v.hi = Some(num()?),
                    "LO" => v.lo = Some(num()?),
                    "FX" => {
                        let x = num()?;
                        v.lo = Some(x);
                        v.hi = Some(x);
                    }
                    "FR" => {
                        v.lo = Some(f64::NEG_INFINITY);
                        v.hi = Some(f64::INFINITY);
                    }
                    "MI" => v.lo = Some(f64::NEG_INFINITY),
                    "PL" => v.hi = Some(f64::INFINITY),
                    "BV" => {
                        v.lo = Some(0.0);
                        v.hi = Some(1.0);
                        v.integer = true;
                    }
                    "UI" => {
                        v.hi = Some(num()?);
                        v.integer = true;
                    }
                    "LI" => {
                        v.lo = Some(num()?);
                        v.integer = true;
                    }
                    other => return Err(err(&format!("unsupported bound type {}", other))),
                }
            }
            "ENDATA" => {}
            other => return Err(err(&format!("data line outside section ({})", other))),
        }
    }

    if obj_row.is_none() {
        return Err("no objective (N) row found".to_string());
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

    // Sanity check on the reader itself.
    assert_eq!(b.spec.vars.len(), vars.len());

    Ok(ParsedMps {
        problem: b.problem,
        spec: b.spec,
        obj_offset,
    })
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
