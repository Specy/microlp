//! Cycling and stall detection for both simplex phases: a hash of the
//! vertices the phase visits, with a taboo on a basis change that keeps
//! revisiting them or leads to a basis the factorization found singular (see
//! [`CycleGuard`]), and a bound on the iterations without progress
//! (`STALL_ITERATIONS`).
//!
//! Perturbation makes zero steps rare and shifting keeps steps from going
//! backwards, but neither rules out a cycle. In a dual phase a shift moves
//! the objective by the shift times the var's value, and a var that absorbs
//! a violation (see `choose_entering_col_dual`) takes its zero step with a
//! shift of its whole reduced cost (for a fixed var, anything), given back
//! when it leaves, so pivots that pass a tolerance-level violation between
//! two rows can return to a vertex they left
//! (`a_dual_simplex_that_cycled_reaches_the_optimum` in
//! `src/tests/magnitudes.rs`). A primal cleanup phase runs unperturbed, and
//! its zero steps at a degenerate vertex can repeat a basis. The hash
//! catches a repeated vertex however it came about; the stall bound catches
//! a long run whose objective does not move, and fails it loudly.

use std::collections::HashSet;

use super::mix;

/// Iterations of one phase after which the objective (in a dual phase, the
/// dual objective of the working costs) must have improved beyond round-off
/// since the previous check, or the phase fails as stalled (a loud error).
/// With the costs or bounds perturbed a phase progresses at almost every
/// pivot, so ten thousand iterations without progress mean a stuck phase,
/// not a slow one. The measure is the objective alone, so a phase that
/// progresses without improving it fails too: an unperturbed near-optimal
/// start on a degenerate problem (a zero objective never moves), a check that
/// falls while a large shift holds the objective back, or one shortly after a
/// basis repair moved it back (a repair moves a var to a bound, and the
/// objective by any amount).
pub(crate) const STALL_ITERATIONS: u64 = 10_000;

/// Why a basis change is bad. A change found bad twice keeps the greater
/// reason in this order, so one that leads to a singular basis stays one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BadReason {
    /// It leads to a visited vertex right after a change that did too.
    Cycling,
    /// No var can repair its row of the dual simplex, and the row does not
    /// prove the model infeasible: a change with no entering var.
    FailedProof,
    /// It leads to a basis the factorization found singular: the change back
    /// of a basis repair, or a cycling change in a phase that repaired its
    /// basis, which can lead back to that basis by another route.
    Singular,
}

/// A bad basis change: while it is taboo, its row is not chosen to leave
/// (dual simplex) and its entering var not chosen to enter (primal simplex).
#[derive(Clone, Debug)]
struct BadBasisChange {
    row: usize,
    leaving: usize,
    entering: Option<usize>,
    reason: BadReason,
    taboo: bool,
}

/// A change of the vertex the simplex is at, as the term it toggles in the
/// vertex hash.
#[derive(Clone, Copy, Debug)]
pub(crate) enum VertexChange {
    /// `var` joins or leaves the basis.
    Basic(usize),
    /// Non-basic `var` moves to or from its upper bound.
    Upper(usize),
}

impl VertexChange {
    fn term(self) -> u64 {
        match self {
            Self::Basic(var) => mix(var as u64),
            Self::Upper(var) => mix(var as u64 ^ 0xA5A5_A5A5_A5A5_A5A5),
        }
    }
}

/// The state of the cycling detection during one phase, dual or primal.
///
/// What repeats in a cycle is the vertex: the basis and where its non-basic
/// vars sit. The hash is the XOR of a term per basic var and a term per
/// non-basic var at its upper bound, so a pivot or a bound flip toggles a
/// term or two. A hash of the basic set alone would not do: the
/// bound-flipping ratio test and the certificate repair
/// (`Solver::certificate_repair`) move non-basic vars without a basis
/// change, so a basic set can come back at another vertex, and a set hash
/// would report a revisit there, a wrong verdict that nothing recovers from
/// (`vertex_hash_makes_a_repeated_basis_change_taboo` pins the difference).
///
/// A basis change to a visited vertex may still be progress, so cycling is
/// declared only when two consecutive basis changes both lead to visited
/// vertices (`is_bad_change`); the next iteration then chooses another row
/// (dual) or another entering var (primal). A record is dropped once a step
/// moves the value in its row by at least the row's tolerance: the state it
/// was bad in is gone.
///
/// Every rebuild lifts the taboo flags, so that on fresh data a taboo change
/// can be tried again, and the phase fails loudly when fresh data leave
/// nothing but taboo changes. The records survive rebuilds and
/// refactorizations: the eta file outgrows the factors every few pivots on a
/// small basis, and a record dropped that often would never outlive the
/// cycle it records.
///
/// A basis repair within a phase records the change back as bad
/// ([`Self::add_singular_change`]), and a cycling change later in the phase
/// is recorded as singular too, since it can lead back to the singular basis
/// by another route: the loud end then names that cause
/// ([`Self::any_taboo_singular`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct CycleGuard {
    /// Whether a phase is running: only then is the vertex tracked.
    active: bool,
    /// The hash of the current vertex.
    hash: u64,
    /// The hashes of the vertices this phase has reached by a pivot.
    visited: HashSet<u64>,
    /// Whether a basis repair happened in this phase.
    repaired: bool,
    /// Basis changes made in this phase.
    pivots: u64,
    /// The pivot count at which the last basis change leading to a visited
    /// vertex was chosen.
    last_revisit: Option<u64>,
    bad_changes: Vec<BadBasisChange>,
}

impl CycleGuard {
    /// Start tracking a phase from `vertex` (every basic var, every non-basic
    /// var at its upper bound), with no history.
    pub(crate) fn start(&mut self, vertex: impl Iterator<Item = VertexChange>) {
        self.active = true;
        self.hash = 0;
        self.hash = self.hash_after(vertex);
        self.visited.clear();
        self.visited.insert(self.hash);
        self.repaired = false;
        self.pivots = 0;
        self.last_revisit = None;
        self.bad_changes.clear();
    }

    /// Stop tracking: the phase is over.
    pub(crate) fn stop(&mut self) {
        self.active = false;
        self.bad_changes.clear();
    }

    /// The hash of the vertex `changes` lead to from the current one.
    pub(crate) fn hash_after(&self, changes: impl Iterator<Item = VertexChange>) -> u64 {
        changes.fold(self.hash, |h, change| h ^ change.term())
    }

    /// Apply `change` to the current vertex (a flip, or part of a pivot).
    pub(crate) fn apply(&mut self, change: VertexChange) {
        if self.active {
            self.hash ^= change.term();
        }
    }

    /// Record the vertex a pivot has just reached.
    pub(crate) fn record_pivot(&mut self) {
        if self.active {
            self.visited.insert(self.hash);
            self.pivots += 1;
        }
    }

    /// Whether the basis change of `row` (basic var `leaving`) to `entering`,
    /// leading to the vertex hashed `next`, is bad: it is on the list of bad
    /// changes, or it leads to a visited vertex right after the previous
    /// change did too (cycling; singular if the phase has repaired its
    /// basis). Either way the change is made taboo and the caller skips it.
    pub(crate) fn is_bad_change(
        &mut self,
        row: usize,
        leaving: usize,
        entering: usize,
        next: u64,
    ) -> bool {
        if self.visited.contains(&next) {
            if self.last_revisit.is_some_and(|p| self.pivots == p + 1) {
                let reason = if self.repaired {
                    BadReason::Singular
                } else {
                    BadReason::Cycling
                };
                debug!(
                    "simplex cycling: basis change {leaving} out, {entering} in (row {row}) \
                     revisits a vertex again ({reason:?}); taboo"
                );
                self.add_bad_change(row, leaving, Some(entering), reason);
                return true;
            }
            self.last_revisit = Some(self.pivots);
        }
        match self
            .bad_changes
            .iter_mut()
            .find(|c| c.row == row && c.leaving == leaving && c.entering == Some(entering))
        {
            Some(change) => {
                change.taboo = true;
                true
            }
            None => false,
        }
    }

    fn add_bad_change(
        &mut self,
        row: usize,
        leaving: usize,
        entering: Option<usize>,
        reason: BadReason,
    ) {
        match self
            .bad_changes
            .iter_mut()
            .find(|c| c.row == row && c.leaving == leaving && c.entering == entering)
        {
            Some(change) => {
                change.taboo = true;
                change.reason = change.reason.max(reason);
            }
            None => self.bad_changes.push(BadBasisChange {
                row,
                leaving,
                entering,
                reason,
                taboo: true,
            }),
        }
    }

    /// Record that no var can repair `row` (basic var `leaving`) and that
    /// the row does not prove the model infeasible, and make the row taboo,
    /// so that the dual simplex goes on with the other rows.
    pub(crate) fn add_failed_proof(&mut self, row: usize, leaving: usize) {
        self.add_bad_change(row, leaving, None, BadReason::FailedProof);
    }

    /// Note, while a phase runs, that a basis repair happened in it.
    pub(crate) fn note_repair(&mut self) {
        if self.active {
            self.repaired = true;
        }
    }

    /// Record, while a phase runs, that the change of `row` (basic var
    /// `leaving`, a slack a basis repair put there) to `entering` (the var
    /// the repair took out) makes the basis singular, and make it taboo.
    pub(crate) fn add_singular_change(&mut self, row: usize, leaving: usize, entering: usize) {
        if self.active {
            self.add_bad_change(row, leaving, Some(entering), BadReason::Singular);
        }
    }

    /// Whether `row` may not be chosen to leave, in the dual simplex.
    pub(crate) fn is_taboo(&self, row: usize) -> bool {
        self.bad_changes.iter().any(|c| c.taboo && c.row == row)
    }

    /// Whether `row` is taboo because it does not prove the model
    /// infeasible although no var can repair it ([`Self::add_failed_proof`]).
    pub(crate) fn is_unproven(&self, row: usize) -> bool {
        self.bad_changes
            .iter()
            .any(|c| c.taboo && c.row == row && c.reason == BadReason::FailedProof)
    }

    /// Whether `var` may not be chosen to enter, in the primal simplex.
    pub(crate) fn is_taboo_entering(&self, var: usize) -> bool {
        self.bad_changes
            .iter()
            .any(|c| c.taboo && c.entering == Some(var))
    }

    /// Whether any change is taboo.
    pub(crate) fn any_taboo(&self) -> bool {
        self.bad_changes.iter().any(|c| c.taboo)
    }

    /// Whether a taboo change leads to a basis the factorization found
    /// singular: when only taboo changes are left, the cause the loud end
    /// names.
    pub(crate) fn any_taboo_singular(&self) -> bool {
        self.bad_changes
            .iter()
            .any(|c| c.taboo && c.reason == BadReason::Singular)
    }

    /// Lift every taboo, keeping the records (at every rebuild).
    pub(crate) fn clear_taboo(&mut self) {
        for change in &mut self.bad_changes {
            change.taboo = false;
        }
    }

    /// Drop the records whose row a step has moved by at least its
    /// tolerance.
    pub(crate) fn forget_moved_rows(&mut self, moved: impl Fn(usize) -> bool) {
        self.bad_changes.retain(|c| !moved(c.row));
    }
}
