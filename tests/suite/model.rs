//! A shadow model of an LP/MILP instance, kept alongside the `microlp::Problem`.
//!
//! `microlp::Problem` does not expose its constraint matrix or variable domains,
//! so the suite records its own copy of everything while building. This shadow
//! copy is what the independent verifier checks returned solutions against —
//! deliberately *not* trusting any state inside the solver.

use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};

#[derive(Clone, Copy, PartialEq)]
pub enum Domain {
    Real,
    Integer,
}

#[derive(Clone)]
pub struct VarSpec {
    pub obj_coeff: f64,
    pub min: f64,
    pub max: f64,
    pub domain: Domain,
}

#[derive(Clone)]
pub struct ConstraintSpec {
    pub terms: Vec<(usize, f64)>, // (variable index, coefficient)
    pub op: ComparisonOp,
    pub rhs: f64,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub vars: Vec<VarSpec>,
    pub constraints: Vec<ConstraintSpec>,
}

/// Builds a `microlp::Problem` and the shadow `ModelSpec` in lockstep.
pub struct Builder {
    pub problem: Problem,
    pub spec: ModelSpec,
    pub vars: Vec<Variable>,
}

impl Builder {
    pub fn new(direction: OptimizationDirection) -> Self {
        Builder {
            problem: Problem::new(direction),
            spec: ModelSpec {
                vars: vec![],
                constraints: vec![],
            },
            vars: vec![],
        }
    }

    pub fn real(&mut self, obj_coeff: f64, min: f64, max: f64) -> Variable {
        let v = self.problem.add_var(obj_coeff, (min, max));
        self.push_spec(v, obj_coeff, min, max, Domain::Real);
        v
    }

    pub fn integer(&mut self, obj_coeff: f64, min: i32, max: i32) -> Variable {
        let v = self.problem.add_integer_var(obj_coeff, (min, max));
        self.push_spec(v, obj_coeff, min as f64, max as f64, Domain::Integer);
        v
    }

    pub fn binary(&mut self, obj_coeff: f64) -> Variable {
        let v = self.problem.add_binary_var(obj_coeff);
        self.push_spec(v, obj_coeff, 0.0, 1.0, Domain::Integer);
        v
    }

    fn push_spec(&mut self, v: Variable, obj_coeff: f64, min: f64, max: f64, domain: Domain) {
        assert_eq!(v.idx(), self.spec.vars.len(), "shadow model out of sync");
        self.spec.vars.push(VarSpec {
            obj_coeff,
            min,
            max,
            domain,
        });
        self.vars.push(v);
    }

    pub fn constraint(&mut self, terms: &[(Variable, f64)], op: ComparisonOp, rhs: f64) {
        self.problem.add_constraint(terms, op, rhs);
        self.spec.constraints.push(ConstraintSpec {
            terms: terms.iter().map(|(v, c)| (v.idx(), *c)).collect(),
            op,
            rhs,
        });
    }
}

/// Comparison tolerance for objective values: `|a - b| <= abs + rel * |b|`.
#[derive(Clone, Copy)]
pub struct Tol {
    pub abs: f64,
    pub rel: f64,
}

impl Tol {
    pub const DEFAULT: Tol = Tol {
        abs: 1e-6,
        rel: 1e-6,
    };

    pub fn matches(&self, got: f64, expected: f64) -> bool {
        (got - expected).abs() <= self.abs + self.rel * expected.abs()
    }
}

/// What the suite knows about the true answer of a case.
pub enum Expected {
    Objective {
        value: f64,
        tol: Tol,
        /// Variable values, asserted only when the optimum is known to be unique.
        vars: Option<Vec<(Variable, f64)>>,
    },
    Infeasible,
    Unbounded,
}

impl Expected {
    pub fn objective(value: f64) -> Expected {
        Expected::Objective {
            value,
            tol: Tol::DEFAULT,
            vars: None,
        }
    }

    pub fn objective_tol(value: f64, tol: Tol) -> Expected {
        Expected::Objective {
            value,
            tol,
            vars: None,
        }
    }

    pub fn unique(value: f64, vars: Vec<(Variable, f64)>) -> Expected {
        Expected::Objective {
            value,
            tol: Tol::DEFAULT,
            vars: Some(vars),
        }
    }
}
