//! Cycling and stall detection for both simplex phases: HiGHS's hash of the
//! visited bases with a taboo on a basis change that keeps revisiting them
//! (`HEkk::updatePivots`, `HEkk::isBadBasisChange`,
//! `HEkk::updateBadBasisChange`, highs/simplex/HEkk.cpp), here over the
//! vertices the phase visits (see [`CycleGuard`] for why), and a stall bound
//! after SoPlex's (`SOPLEX_MAXSTALLS`, src/soplex/spxsolve.hpp). A taboo keeps
//! the dual simplex from choosing the change's leaving row (HiGHS's
//! `applyTabooRowOut`) and the primal simplex from choosing its entering var
//! (HiGHS's `applyTabooVariableIn`).
//!
//! Cost perturbation ([`super::Solver::begin_dual_phase`]) makes a dual step
//! of zero rare, and cost shifting keeps every step from going backwards, so
//! the dual objective of the working costs rises. It is not strictly
//! monotone, though: a shift moves it by the shift times the var's value,
//! and a var that absorbs a violation (see `choose_entering_col_dual`) takes
//! its zero step with a shift of its whole reduced cost, which for a fixed
//! var can be anything and is given back when the var leaves. Pivots that
//! pass a tolerance-level violation between two rows can so return to a
//! vertex they left (the property stress found a model, `milp_wide_2`, whose
//! dual simplex cycled through four pivots this way, forever). Bound
//! perturbation ([`super::Solver::begin_primal_phase`]) does the same for the
//! primal simplex, whose steps are forward or zero, so its objective never
//! rises; but its cleanup phases run unperturbed, and a degenerate vertex
//! there gives zero steps, on which the bases can repeat. The hash catches a
//! repeated vertex however it came about; the stall bound catches the rest, a
//! long run whose objective does not move, and turns what used to be a solve
//! that never returns into a loud error.

use std::collections::HashSet;

use super::mix;

/// Iterations of one phase after which the objective (the dual objective of
/// the working costs in a dual phase, the objective in a primal one) must
/// have improved beyond round-off since the previous check, or the phase is
/// declared stalled (a loud error). SoPlex checks every `SOPLEX_MAXSTALLS =
/// 10000` iterations in the same way, in both of its algorithms, and
/// switches algorithm (up to `SOPLEX_MAXSTALLRECOVERS = 10` times) and
/// finally aborts with `ABORT_CYCLING` when nothing has moved. With the
/// costs (dual) or bounds (primal) perturbed, a phase makes progress at
/// almost every pivot; ten thousand iterations without any is not a slow
/// phase but a stuck one.
///
/// The measure is the objective alone, where SoPlex's adds the reduction of
/// its shifts, and nothing is tried before the error. So a phase that
/// progresses without improving it would be declared stalled too: one that
/// starts near optimal, and so unperturbed, on a degenerate problem (a zero
/// objective never moves), or one whose check falls while a large shift
/// holds the objective back (an absorbing fixed var's shift is its whole
/// reduced cost). Either takes ten thousand iterations of one phase; none
/// was seen in the suite, the differential fuzz or the property stress.
pub(crate) const STALL_ITERATIONS: u64 = 10_000;

/// A basis change found to repeat vertices, as HiGHS's
/// `HighsSimplexBadBasisChangeRecord`: while it is taboo, its row is not
/// chosen to leave (dual simplex) and its entering var not chosen to enter
/// (primal simplex).
#[derive(Clone, Debug)]
struct BadBasisChange {
    row: usize,
    leaving: usize,
    entering: usize,
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
/// non-basic var at its upper bound, so every change of either (a pivot, a
/// bound flip) toggles a term or two. HiGHS hashes the basic set alone
/// (`HighsHashHelpers::sparse_combine` over the basic vars), but a basic set
/// can come back at another vertex: the bound-flipping ratio test and a
/// flip-only step move boxed vars to their other bound without a basis
/// change, so a set hash can take a phase that has moved on for one that
/// revisits, and two such verdicts in a row make a change taboo. HiGHS
/// recovers from a wrong verdict (its `kUnknown` sends the LP to another
/// solve); this engine has no such recovery, so a verdict here must be a
/// real repeat. (A first version hashed the basic set while the correction
/// at a rebuild still moved boxed vars back to their other bound, and so
/// failed the differential fuzz's model `wide_lp-1897`; with that
/// correction shifting instead, see `rebuild_dual`, no model of the fuzz
/// tells the two hashes apart, and the unit test
/// `vertex_hash_makes_a_repeated_basis_change_taboo` pins the difference.)
///
/// Every vertex the phase reaches by a pivot is recorded. A basis change
/// whose vertex was already visited may still be progress, so, as in HiGHS,
/// cycling is declared only when two consecutive basis changes both lead to
/// visited vertices: the change is then recorded as bad and made taboo, and
/// the iteration is skipped, so the next one chooses another row (dual) or
/// another entering var (primal). A bad change found again later is made
/// taboo again at once. A record is dropped once a step moves the value in
/// its row by at least the row's tolerance (HiGHS `updateBadBasisChange`,
/// which HiGHS applies in its dual simplex; here in both): the state it was
/// bad in is gone.
///
/// Every rebuild of the phase lifts the taboo flags and keeps the records,
/// as HiGHS's `HEkkDual::rebuild` does (`clearBadBasisChangeTabooFlag`)
/// when it does not refactorize: HiGHS drops the records only when it
/// refactorizes (`HEkk::computeFactor`), and with its default
/// `no_unnecessary_rebuild_refactor` a rebuild because no row is left to
/// choose (`kRebuildReasonPossiblyOptimal`, the case of only taboo rows
/// violated) or because a candidate is dual infeasible or none is left
/// (`kRebuildReasonPossiblyDualUnbounded`, which also stands for the rebuild
/// before a row's verdict of infeasibility) refactorizes only when its solve
/// error exceeds `1e-8` (`HEkk::rebuildRefactor`). So on fresh data a taboo
/// row is tried again, a recorded change chosen again is taboo again at
/// once, and the phase gives up (a loud error, HiGHS's
/// `kSolvePhaseTabooBasis` returning `kUnknown`) when fresh data leave
/// nothing but taboo changes. Here the records also outlive the rebuild
/// after a pivot disagreement (HiGHS refactorizes there, suspecting a
/// singular basis) and the periodic refactorizations, which HiGHS makes
/// after many updates (at most `simplex_update_limit`, 5000, or when its
/// synthetic clock says a new factorization is cheaper) and this engine as
/// soon as the eta file outgrows the factors, every few pivots on a small
/// basis: a record dropped that often would never outlive the cycle it
/// records.
#[derive(Clone, Debug, Default)]
pub(crate) struct CycleGuard {
    /// Whether a phase is running: only then is the vertex tracked.
    active: bool,
    /// The hash of the current vertex.
    hash: u64,
    /// The hashes of the vertices this phase has reached by a pivot.
    visited: HashSet<u64>,
    /// Basis changes made in this phase.
    pivots: u64,
    /// The pivot count at which the last basis change leading to a visited
    /// vertex was chosen (HiGHS `previous_iteration_cycling_detected`).
    last_revisit: Option<u64>,
    bad_changes: Vec<BadBasisChange>,
}

impl CycleGuard {
    /// Start tracking a phase from the vertex of `changes` (every basic var,
    /// every non-basic var at its upper bound), with no history (HiGHS
    /// clears the visited bases at the start of a solve,
    /// `HEkk::initialiseForSolve`).
    pub(crate) fn start(&mut self, vertex: impl Iterator<Item = VertexChange>) {
        self.active = true;
        self.hash = 0;
        self.hash = self.hash_after(vertex);
        self.visited.clear();
        self.visited.insert(self.hash);
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
    /// leading to the vertex hashed `next`, is bad, as HiGHS's
    /// `HEkk::isBadBasisChange`: it is on the list of bad changes, or it
    /// leads to a visited vertex right after the previous change did too
    /// (cycling). Either way the change is made taboo and the caller skips
    /// it.
    pub(crate) fn is_bad_change(
        &mut self,
        row: usize,
        leaving: usize,
        entering: usize,
        next: u64,
    ) -> bool {
        if self.visited.contains(&next) {
            if self.last_revisit.is_some_and(|p| self.pivots == p + 1) {
                debug!(
                    "simplex cycling: basis change {leaving} out, {entering} in (row {row}) \
                     revisits a vertex again; taboo"
                );
                self.add_bad_change(row, leaving, entering);
                return true;
            }
            self.last_revisit = Some(self.pivots);
        }
        match self
            .bad_changes
            .iter_mut()
            .find(|c| c.row == row && c.leaving == leaving && c.entering == entering)
        {
            Some(change) => {
                change.taboo = true;
                true
            }
            None => false,
        }
    }

    fn add_bad_change(&mut self, row: usize, leaving: usize, entering: usize) {
        match self
            .bad_changes
            .iter_mut()
            .find(|c| c.row == row && c.leaving == leaving && c.entering == entering)
        {
            Some(change) => change.taboo = true,
            None => self.bad_changes.push(BadBasisChange {
                row,
                leaving,
                entering,
                taboo: true,
            }),
        }
    }

    /// Whether `row` may not be chosen to leave, in the dual simplex (HiGHS
    /// `HEkk::applyTabooRowOut`).
    pub(crate) fn is_taboo(&self, row: usize) -> bool {
        self.bad_changes.iter().any(|c| c.taboo && c.row == row)
    }

    /// Whether `var` may not be chosen to enter, in the primal simplex
    /// (HiGHS `HEkk::applyTabooVariableIn`).
    pub(crate) fn is_taboo_entering(&self, var: usize) -> bool {
        self.bad_changes
            .iter()
            .any(|c| c.taboo && c.entering == var)
    }

    /// Whether any change is taboo (HiGHS `HEkk::tabooBadBasisChange`).
    pub(crate) fn any_taboo(&self) -> bool {
        self.bad_changes.iter().any(|c| c.taboo)
    }

    /// Lift every taboo, keeping the records (HiGHS
    /// `HEkk::clearBadBasisChangeTabooFlag`, at every rebuild).
    pub(crate) fn clear_taboo(&mut self) {
        for change in &mut self.bad_changes {
            change.taboo = false;
        }
    }

    /// Drop the records whose row a step has moved by at least its
    /// tolerance (HiGHS `HEkk::updateBadBasisChange`).
    pub(crate) fn forget_moved_rows(&mut self, moved: impl Fn(usize) -> bool) {
        self.bad_changes.retain(|c| !moved(c.row));
    }
}
