# microlp Architecture

This document explains how the solver works: how a problem flows from the public API
through the simplex engine and the branch & bound search, why the pieces are shaped the
way they are, and where to plug in improvements. It is written to be sufficient on its
own — a person (or an agent) who has read this should be able to navigate the codebase,
predict its behavior, and extend it safely.

---

## 1. The big picture

microlp solves **linear programs** (LP: continuous variables, linear constraints, linear
objective) and **mixed-integer linear programs** (MILP: same, but some variables must take
integer values). It is organized as two layers with a deliberately narrow interface between
them:

```mermaid
graph TD
    subgraph API["Public API (src/lib.rs)"]
        P[Problem] -->|solve / solve_with| S[Solution]
        O[SolveOptions + Tolerances]
    end
    subgraph MIP["MIP layer (src/mip/)"]
        D[driver: mod.rs<br/>search loop, MipState] --> B[branching.rs<br/>pseudocosts]
        D --> N[node.rs<br/>plain-data tree nodes]
        D --> PR[params.rs<br/>internal constants]
    end
    subgraph LP["LP engine (src/solver.rs)"]
        SV[Solver: bounded-variable simplex] --> LU[lu.rs: LU factorization]
        SV --> SP[sparse.rs / ordering.rs]
    end
    P --> D
    D -->|one persistent instance| SV
    S -->|LP: live solver<br/>MILP: plain values + MipState| D
```

Three principles shape everything:

1. **One persistent LP solver per search.** The branch & bound tree never clones the solver
   and never grows the LP. Branching is expressed purely as *variable bound changes* on the
   single `Solver` instance, and tree nodes are plain data that describe how to reconstruct
   a state on that instance.
2. **Nothing half-solved is ever consulted.** Time and node limits only take effect *between*
   node LP solves. A node whose LP is interrupted mid-pivot is pushed back **unsolved**; its
   half-pivoted solver state is structurally unreachable (the next visit rebuilds state from
   the node's own data). This is what makes pause/resume safe by construction.
3. **Loud failures over silent wrong answers.** Child-LP errors are never conflated with
   infeasibility; numerical failures either recover through a documented valve or propagate
   as errors; candidate solutions are re-validated before being accepted. Where the code
   cannot do something properly it panics with a message rather than approximating.

### Module map

| File | Responsibility |
|---|---|
| `src/lib.rs` | Public API: `Problem`, `Solution`, `Status`, re-exports of `SolveOptions`/`Tolerances`/`Stats`. Dispatches pure-LP vs MILP solves. |
| `src/solver.rs` | The simplex engine (`Solver`): bounded-variable primal/dual simplex, basis management, and the small contract surface the MIP layer uses. |
| `src/lu.rs`, `src/sparse.rs`, `src/ordering.rs` | LU factorization with eta-file updates, sparse containers, fill-reducing ordering. |
| `src/mip/mod.rs` | The branch & bound driver: `MipState`, the search loop, interruption/resume, warm starts, post-solve edits, bound/gap accounting. |
| `src/mip/node.rs` | `Node` (plain-data tree node) and `effective_bounds` (bound-change collapsing). |
| `src/mip/branching.rs` | Integrality checks and branch-variable selection (pseudocosts). |
| `src/mip/params.rs` | Named, documented internal constants (see §7). |
| `src/mps.rs` | MPS-format reader for the public API. |
| `tests/suite/` | The problem-based correctness suite (see §9) — the safety net for all of this. |

---

## 2. Problem representation

`Problem` stores the model in original user terms: objective coefficients, per-variable
bounds and domains (`Real`, `Integer`, `Boolean`), and constraint rows.

Two normalizations happen at the boundary and hold everywhere inside:

- **Internal objective space is always MINIMIZE.** `Maximize` problems negate their
  objective coefficients at variable-creation time (`internal_add_var`), and the sign is
  flipped back exactly once per read (`Solution::objective`, user-facing `Stats.best_bound`).
  Everything inside the driver — incumbents, bounds, pruning, gaps — is minimize-space.
  When editing driver code, never reason about direction; it does not exist there.
- **Every constraint row gets a slack variable.** A row `a·x ≤ b` becomes `a·x + s = b`
  with `s ∈ [0, +∞)`; `≥` gives `s ∈ (−∞, 0]`; `=` gives `s ∈ [0,0]`. So the `Solver`'s
  variable universe is `num_vars` *structural* variables followed by one slack per row
  ("total vars"), and every constraint is an equality against the basis matrix. A row's
  original sense is recoverable from its slack's bounds — `Solver::check_constraints`
  exploits exactly this.

---

## 3. The LP engine (`src/solver.rs`)

The simplex core is the minilp lineage: a **bounded-variable
revised simplex** with both primal and dual iterations, steepest-edge pricing, the Harris
two-pass ratio test for numerical stability, and an LU-factorized basis updated by eta
matrices (refactorized when the eta file outgrows the factors).

State you need to know when reading it:

- `basic_vars[row]` — which variable is basic in each row; `basic_var_vals` their values.
- `nb_vars[col]` / `nb_var_vals` / `nb_var_states{at_min, at_max}` — non-basic variables
  sit at one of their bounds (or at 0 if free).
- `is_primal_feasible` / `is_dual_feasible` — honest flags; every solve path is a state
  machine over them. `initial_solve` = restore primal feasibility (dual simplex /
  phase-1-style) then optimize (primal simplex).
- `cur_obj_val`, `nb_var_obj_coeffs` (reduced costs), `lp_iterations` (cumulative pivot
  counter for stats).

### 3.1 The contract surface the MIP layer relies on

The engine exposes a small contract surface; everything the B&B does goes through these.

**`set_var_bounds(var, min, max) -> Result<(), Error>`** — change a variable's bounds in
place. Basic variable: update the row's bound mirrors, flag primal-infeasible if its value
fell outside. Non-basic variable: clamp its value to the new range, propagate the delta into
the basic values through the variable's column (same mechanism as `fix_var`), recompute its
at-bound flags, and downgrade `is_dual_feasible` if the move broke the reduced-cost/bound
pairing. Crossing bounds (`min > max`) returns `Err(Infeasible)` with state untouched.
It does **not** run simplex — callers decide when to reoptimize.

*Why this is the branching primitive:* tightening a bound leaves every reduced cost
untouched, so the current basis stays **dual feasible** — re-solving is a short dual-simplex
run warm-started from the parent's optimal basis, typically a handful of pivots. This is the
single biggest performance lever in the design (see §10).

**`reoptimize() -> Result<StopReason, Error>`** — the re-solve entry: dual simplex if primal
feasibility is broken, then (only if needed, e.g. after loosening bounds or a basis load
with drift) recompute reduced costs and run primal simplex. Returns `StopReason::Limit` if
the deadline fires mid-run — leaving the honest feasibility flags so a later call continues
where it left off.

**`snapshot_basis() / load_basis(&Basis)`** — a `Basis` is one status per total variable:
`Basic | AtLower | AtUpper | Free` (~1 byte each). That is the *entire* warm-start state a
tree node needs. `load_basis` rebuilds everything from statuses + **current** bounds:
non-basic values from their status's bound, basic values and reduced costs recomputed from
scratch, LU refactorized, feasibility flags recomputed honestly. Two contracts matter:

- *Statuses are interpreted against the current bounds.* A status pointing at a bound that
  has since moved is remapped (nearest finite bound, else 0) rather than rejected — B&B
  jumps load a parent basis **after** applying different bounds, so this is load-bearing
  by design, not sloppiness.
- *On `Err`, solver state is unspecified* and must be restored by a subsequent successful
  load. `slack_basis()` (all slacks basic = identity basis matrix) always loads successfully
  and is the designated recovery everywhere.

**`check_constraints(values, tol)` / `objective_of(values)`** — evaluate an explicit
structural-variable vector against the *original* rows (sense recovered from slack bounds)
within an **absolute** per-row tolerance, and compute its exact objective. These exist for
the rounded-incumbent guard (§5.4) and are deliberately independent of the simplex state.

---

## 4. The MIP data model (`src/mip/node.rs`, `MipState`)

```rust
// A tree node is PLAIN DATA — no solver machinery anywhere:
Node {
    bound_changes: Vec<(var, lo, hi)>, // cumulative from the root; later entries win
    basis: Basis,                      // the PARENT's optimal basis (warm start)
    lp_bound: f64,                     // parent's LP objective = valid lower bound here
    depth, parent_id,                  // parent_id detects warm dives (§5.2)
    branch_var, branch_up, branch_frac // metadata feeding pseudocost updates (§5.6)
}
```

Reconstructing any node's starting state = apply its `bound_changes` on top of the root
bounds, load its `basis`, reoptimize. That is the whole trick: because nodes carry no live
state, they are trivially storable, resumable, and cheap (a basis is `total_vars` bytes; a
bound-change list is `depth` entries).

`MipState` is the complete, resumable search:

| Field | Role |
|---|---|
| `solver` | the ONE live LP engine |
| `root_bounds`, `applied` | original bounds + which changes are currently applied to the solver (for diffing when switching nodes) |
| `open: Vec<Node>` | the frontier (LIFO tail = current dive; best-bound scan on jumps) |
| `incumbent: Option<Incumbent>` | best integer solution: **rounded** values + `objective = c·x_rounded` |
| `node_seq`, `last_solved_id`, `diving` | warm-dive detection + node-selection mode |
| `pseudocosts`, `stats`, `options`, `deadline`, `direction` | search intelligence and bookkeeping |
| `base: Problem`, `fixed: BTreeMap<var, val>` | the CLEAN user problem + user-level fixes — the substrate for post-solve edits (§5.8) |

`Solution` for a MILP holds `Status` + this `MipState` boxed; user reads come from the
incumbent's plain values, never from the live solver. For a pure LP, `Solution` instead
keeps the live solver (that is what makes LP incremental editing cheap).

---

## 5. The branch & bound search (`src/mip/mod.rs`)

### 5.1 Lifecycle

```mermaid
flowchart TD
    A[run: build Solver from Problem] --> B[initial_solve: root LP relaxation]
    B -->|Limit| I1((return Interrupted<br/>resume re-enters here))
    B -->|Infeasible| E1((Err Infeasible))
    B -->|Unbounded| UF[zero-objective integer-feasibility search]
    UF -->|integer point found| EU((Err Unbounded))
    UF -->|tree exhausted| E1
    UF -->|Limit| I1
    B --> WS{warm_start hint?}
    WS -->|yes| H[try_warm_start §5.7<br/>fix hinted vars → LP-complete →<br/>guarded adopt → restore root exactly]
    WS -->|no| C
    H --> C{root integral<br/>within int_tol?}
    C -->|yes + guard passes + LP point exact| OPT((Optimal))
    C -->|rounded or guard fails| BR0[adopt if feasible;<br/>branch below-tolerance var]
    C -->|no| BR0[branch root: two children pushed]
    BR0 --> L{{search loop}}
    L --> L0{open empty?}
    L0 -->|yes| DONE{incumbent?}
    DONE -->|yes| OPT2((Optimal))
    DONE -->|no| INF((Err Infeasible))
    L0 -->|no| L1{gap ≤ mip_gap?<br/>deadline? node_limit?}
    L1 -->|limit hit| I2((return Interrupted —<br/>everything stays in open))
    L1 --> POP[pop_node: dive LIFO or<br/>best-bound jump §5.5]
    POP --> PR1{stored lp_bound ≥ cutoff?}
    PR1 -->|prune| L0
    PR1 --> APPLY[apply bound diff vs applied<br/>+ load basis unless warm dive §5.2]
    APPLY --> SOLVE[solve_node_lp:<br/>reoptimize, slack retry on singular §5.3]
    SOLVE -->|Infeasible| L0
    SOLVE -->|Limit| REQ[requeue node UNSOLVED] --> I3((return Interrupted))
    SOLVE -->|Solved z| PC[record pseudocost observation §5.6]
    PC --> PR2{z ≥ cutoff?}
    PR2 -->|prune| L0
    PR2 --> INT{integral within int_tol?}
    INT -->|yes| GUARD{rounded values feasible? §5.4}
    GUARD -->|yes| ADOPT[adopt incumbent]
    ADOPT --> EXACT{LP point exactly integral?}
    EXACT -->|yes| L0
    EXACT -->|no| BTOL[branch the below-tolerance var<br/>children fix it exactly] --> L0
    GUARD -->|no| BTOL
    INT -->|no| BRANCH[pseudocost-choose var,<br/>push 2 children, cheaper side on top] --> L0
```

### 5.2 Visiting a node: warm dives vs jumps

When a node is popped, its target bounds are diffed against `applied`: variables no longer
constrained are reset to root bounds, changed ones are set. Then one question decides the
cost of the visit: **is the solver already sitting at this node's parent's optimum?**

- `last_solved_id == node.parent_id` → **warm dive**: the parent was the immediately
  previously solved node, its basis is live in the solver, and the child differs by exactly
  one tightened bound. Skip the basis load entirely; `reoptimize` is a short dual-simplex
  run. This is the common case while diving.
- Otherwise → **jump**: load the node's stored parent basis (one refactorization), then
  reoptimize. `last_solved_id` is cleared on every path where the solver moves away from a
  just-solved optimum (prune-after-solve, infeasible, requeue, incumbent adoption), so the
  warm-dive check can never false-positive: ids are unique per branching.

### 5.3 Node LP solving and the robustness valves

`solve_node_lp` wraps `reoptimize` and owns error discrimination — the fix for the old
design's worst habit (swallowing every child error as "pruned"):

- `Err(Infeasible)` → genuinely infeasible node → prune. Correct and cheap.
- `Err(Unbounded)` → impossible for a bounded node → surfaced as `InternalError`.
- Any other error (singular LU from numerical degradation, discovered in the wild on
  `miplib/enigma`) → **retry once from the slack basis** (identity, cannot fail to load),
  re-solving the node from scratch; a second failure propagates. The retry is per-node-visit
  — it cannot mask a systematic failure.
- `Ok(Limit)` → the deadline fired mid-solve → the node is pushed back **unsolved** and the
  search returns `Interrupted`. Nothing reads the half-pivoted state: the node's next visit
  starts from its own bounds + basis data. (This is the structural fix for the historical
  `assert!(is_primal_feasible)` panic.)

One more valve lives inside the engine itself, in `restore_feasibility` (the dual phase-1):
"no eligible entering column for a violated row" proves infeasibility only in exact
arithmetic. Deep in an eta-file chain, accumulated round-off can promote a phantom bound
violation into a leaving row whose (equally drifted) pivot row blocks every candidate — a
*false* `Infeasible`. This is not hypothetical: `netlib/brandy` was mis-declared infeasible
for the fork's whole history (a fork-era commit had tightened `EPS` from upstream minilp's
`1e-8` to `1e-10` in service of a branch & bound design that no longer exists, putting phase-1 below the noise floor of a
220-row basis). Before an infeasibility declaration is allowed to stand, the engine now
refactorizes the basis and recomputes the basic values from the original data, then
re-examines: a phantom dissolves, a real infeasibility survives and the next declaration
stands. The valve is armed once per stall (any successful pivot re-arms it), so it cannot
loop. Loosening `EPS` instead is NOT an option — the big-M correctness tests explode when
basic integer values may legally sit `1e-8` off their bounds (see the `EPS` docs in
`solver.rs`).

### 5.4 Incumbents and the rounded-feasibility guard

When a node's LP solution is integral within `int_tol` (default `1e-6`), it is a *candidate*
— not yet an incumbent. `try_adopt_incumbent` first rounds every integer variable to the
nearest integer and validates the rounded vector against the **root bounds and every
original row** within the absolute `Tolerances::feasibility` (default `1e-7`).

Why this exists — the **big-M trap**: with `int_tol = 1e-6`, a relaxation value like
`b = 0.999999995` counts as integral. But if `b` multiplies a coefficient of `1e9` somewhere,
rounding it to `1` moves that row by `5.0` — a real violation hiding inside the integrality
tolerance. The guard makes this impossible to adopt:

- Guard **passes** → adopt: store the ROUNDED values with `objective = c·x_rounded`, so what
  the user reads is exactly self-consistent. If rounding changed any integer value, adoption
  does **not** close the node: the relaxation bound can still be strictly better than the
  rounded objective, so the driver branches on that below-tolerance fractionality to finish
  the proof.
- Guard **fails** → do not adopt; **branch on the offending below-tolerance variable**
  (children `⌊v⌋` / `⌊v⌋+1` fix it exactly, and the dive resolves the truth).
- Degenerate fallback: if every integer variable is *exactly* integral yet the check failed,
  retry once from the all-slack basis. This removes eta-chain drift on big-M rows; if the
  independently checked point is still invalid, return an internal error rather than
  force-accepting a potentially infeasible answer.

The tolerance is deliberately **absolute**, never scaled by row magnitude: a relative
tolerance (`1e-7·|rhs|`) evaluates to ~100 on a 1e9-scale row and would swallow exactly the
violations the guard exists to catch. A false *rejection* from the absolute check is benign
(extra exact-fixing branching); a false acceptance would be a wrong answer.

### 5.5 Node selection, bound, and gap

- **Plunging DFS with best-bound jumps** (`pop_node`): while the last processed node
  produced children (`diving == true`), pop LIFO — cheap warm dives, incumbents found fast.
  When a dive dies out (prune/infeasible/leaf), jump to the open node with the **lowest**
  `lp_bound` (linear scan, first-minimum tie-break, `swap_remove`).
- **Global dual bound** = min over open nodes' `lp_bound`, clamped by the incumbent
  (stale nodes may carry looser bounds than a fresher incumbent). Open list empty →
  the bound *is* the incumbent: proof complete. Valid only between nodes — a popped node's
  subtree is otherwise unaccounted.
- **Gap** = `(incumbent − bound) / max(|incumbent|, ε)` in minimize space (sign-free — the
  formula is direction-invariant). `mip_gap > 0` stops the search early with `Optimal`
  (proven within the requested gap); the default `0.0` demands exact proof and adds zero
  overhead (the check short-circuits).
- **Pruning** uses `cutoff(incumbent) = incumbent − max(ε, ε·|incumbent|)` with
  `ε = Tolerances::prune_epsilon` (default 1e-9), applied twice per node: against the stored
  parent bound *before* any LP work, and against the fresh objective after.

### 5.6 Pseudocost branching (`src/mip/branching.rs`)

Which fractional variable to branch on is the single most impactful heuristic in B&B.
Most-fractional selection is empirically ≈ random; the driver instead learns **pseudocosts**:
per variable and direction, the average objective degradation per unit of fractionality
observed across all branchings so far.

- *Recording*: when a node's LP solves, its creating branch (`branch_var`, `branch_up`,
  `branch_frac` on the node) contributes `max(0, z_child − parent_bound) / branch_frac`.
  Only genuinely solved nodes record (never infeasible/interrupted ones).
- *Selection*: maximize the product score
  `max(est_down·f_down, ε) · max(est_up·f_up, ε)` — variables whose BOTH directions hurt
  the relaxation are the ones worth deciding early. Before any observations exist, estimates
  fall back to `|objective coefficient| + ε` — with uniform coefficients this degrades
  gracefully to most-fractional.
- *Dive order*: of the two children, the one with the LOWER estimated degradation is pushed
  last (popped first) — dive toward the side more likely to stay feasible and good.

Measured effect on the correctness suite: total wall-clock halved, with the tree-heavy
MIPLIB instances (`lseu`, `p0201`, `misc03`) shrinking the most.

### 5.7 Warm starts

`SolveOptions::warm_start` accepts a (possibly partial) assignment. Evaluation happens once,
right after the root LP: fix the hinted variables to their (rounded, bounds-checked) values,
LP-complete the rest, and if the completion is integral, adopt it **through the same
feasibility guard as every other incumbent** — hints get no shortcut. Then restore the root
state *exactly* (bounds back, root basis reloaded, everything recomputed) so the search
starts from the true relaxation. Hints are advisory by design: unknown variables,
out-of-range values, infeasible or fractional completions all just drop the hint with a
debug log — a bad hint must never break a solve.

**An important non-obvious fact** (discovered while building the integration tests): a warm
start seeds the *incumbent*, which powers pruning — it does **not** carry any of the search
tree. The search is deterministic, so re-solving the same problem from scratch with the same
budget re-explores the same nodes; even hinting the true optimum leaves most of the
*bound-proving* work intact (measured: 66–99% of a cold solve's nodes). Consequently:

- To **continue** an interrupted solve of an unchanged problem: use `Solution::resume` —
  it keeps the open list and continues where it stopped, budget-for-budget.
- Restart-with-hint is the right tool when the problem **changed** (edits) or the state was
  lost — and callers who loop restart-with-hint must grow the budget between rounds
  (a fixed budget provably never converges; see `tests/suite/cases/warm_restart.rs`).

### 5.8 Post-solve edits

Applying post-solve edits to whatever internal state the search ended in — an incumbent
*leaf*, with branch bound-fixings still applied — would let feasible edits report
`Infeasible`. The edit model makes that impossible:

```mermaid
sequenceDiagram
    participant U as user
    participant S as Solution (MILP)
    participant M as mip::reedit_and_resolve
    U->>S: add_constraint / fix_var / unfix_var
    S->>S: mutate state.base (push row) or state.fixed (insert/remove)
    S->>M: reedit_and_resolve(state)
    M->>M: drop the open tree (its bounds may exclude new optima)
    M->>M: old incumbent still feasible for base+fixed? (cheap check)
    M->>M: yes → seed options.warm_start with it
    M->>M: run(effective_problem(base, fixed), options) — fresh search
    M->>S: new MipState (base/fixed restored onto it)
    S->>U: new Solution
```

`base` is a clean copy of the user's problem that accumulates edits; `fixed` is the
`fix_var` overlay (so `unfix_var` can restore original bounds). Every edit re-solves the
*composed* problem from the root, warm-started by the surviving incumbent. Edits work on
paused (`Feasible`/`Interrupted`) solutions too — that is the "edit after a time limit"
feature. `unfix_var` returns `Result<(Solution, bool), Error>`; a prior interrupted edit can
leave `base+fixed` infeasible-unproven, and the unfix re-solve may be the one to prove it.

### 5.9 Interruption and resume, end to end

Every entry point (`solve_with`, `resume`, edits) computes a **fresh deadline** from
`options.time_limit`. Interruption points, in order of checking per loop iteration:
open-list-empty (a completed proof is never mislabeled as interrupted — checked *first*),
gap target, deadline, node budget (`node_limit` is per-call: each `resume` gets a fresh
budget, which makes deterministic stepped tests possible — `node_limit: Some(0)` + a hint
is how the warm-start liveness test proves the hint path works with zero nodes solved).

`Status` tells the truth about what you have:

| Status | Meaning | Accessors |
|---|---|---|
| `Optimal` | proof complete (within `mip_gap`) | all valid |
| `Feasible` | limit hit; incumbent exists; `gap()` quantifies it | all valid |
| `Interrupted` | limit hit before any usable solution | value accessors expose the **current working point** (possibly fractional/infeasible — checking the status is the caller's job); `resume()` continues |

A time limit is a *status*, never an `Error` — errors are reserved for `Infeasible`,
`Unbounded`, and internal failures.

---

## 6. Public API tour

```rust
let mut problem = Problem::new(OptimizationDirection::Minimize);
let x = problem.add_integer_var(3.0, (0, 10));
let y = problem.add_var(4.0, (0.0, 10.0));
problem.add_constraint(&[(x, 1.0), (y, 2.0)], ComparisonOp::Ge, 5.0);

let mut options = SolveOptions::default();
options.time_limit = Some(Duration::from_secs(10));
options.node_limit = Some(100_000);          // deterministic alternative
options.mip_gap   = 0.01;                    // stop at a proven 1% gap
options.warm_start = Some(vec![(x, 2.0)]);   // advisory hint
options.tolerances.feasibility = 1e-7;       // expert knobs, see §7

let sol = problem.solve_with(options)?;
match sol.status() {
    Status::Optimal | Status::Feasible => {
        let _ = (sol.objective(), sol.var_value(x), sol.gap(), sol.stats());
        // continue searching, or edit and re-solve:
        let sol = sol.resume(Some(Duration::from_secs(10)))?;
        let sol = sol.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 4.0)?;
        let (sol, was_fixed) = sol.fix_var(x, 3.0)?.unfix_var(x)?;
    }
    Status::Interrupted => { let _ = sol.resume(None)?; } // finish the job
}
```

Reading values: `var_value` rounds integer variables (and asserts the stored value was
already integral-clean — a failed assert means a solver bug, not user error);
`var_value_raw`/`iter`/indexing return the stored values — for MILP those are the
incumbent's already-rounded values, for LP the live basis values. `add_gomory_cut` remains
an LP-only tool (it reads live tableau rows).

---

## 7. Numerical policy — every tolerance, in one place

Two homes, by audience:

**User-facing — `SolveOptions` (+ nested `Tolerances`):**

| Knob | Default | Gates |
|---|---|---|
| `int_tol` | `1e-6` | "is this LP value integral?" — a rounded feasible point may be adopted, but branching continues until its LP point is exact. Must be finite and in `[0, 0.5)`. |
| `mip_gap` | `0.0` | early-stop proof quality (relative gap) |
| `tolerances.feasibility` | `1e-7` **absolute** | the rounded-incumbent guard and the post-edit incumbent pre-filter (§5.4 explains why absolute) |
| `tolerances.integrality_rounding` | `1e-5` | integrality check in the edit pre-filter; `var_value`'s sanity assert pins the *default* deliberately |
| `tolerances.prune_epsilon` | `1e-9` | the pruning cutoff slack |

**Internal — `src/mip/params.rs` and `src/solver.rs` consts (each documented at its
definition):** `SCORE_EPS`, `PSEUDOCOST_INIT_EPS`, `BRANCH_FRAC_GUARD` (all `1e-6`),
`GAP_DENOM_GUARD` (`1e-10`), `HINT_BOUNDS_SLACK` (`1e-9`), `DEADLINE_CHECK_INTERVAL`
(`1000` pivots), `LU_STABILITY_THRESHOLD` (`0.1`), and the simplex pivot tolerance
`EPS` (`1e-10`) — the one number the whole engine's float comparisons are built on.

The layering rule: `EPS` decides *simplex* questions (is this coefficient zero, is this
value at its bound); `int_tol` decides *integrality* questions; `feasibility` decides
*solution acceptance*; `prune_epsilon` decides *tree* questions. They are close in
magnitude but must never be conflated — several historical bugs were exactly such
conflations.

---

## 8. Error handling and robustness

| Situation | Behavior |
|---|---|
| Root LP unbounded on a MILP | run a resumable zero-objective integer-feasibility search; any integer point proves `Unbounded`, exhaustion proves `Infeasible` |
| Node LP infeasible | prune (correct) |
| Node LP unbounded | impossible when the node is bounded → `InternalError` |
| Singular LU or an exactly-integral candidate with guard-breaking drift | retry once from the slack basis; then propagate |
| `load_basis` failure on a jump | load the slack basis (infallible) and solve the node from scratch |
| Phase-1 stall (“no entering column”) | refresh the basis (fresh LU + recomputed values) and retry once per stall; declare `Infeasible` only if it survives the refresh |
| Deadline mid-LP | requeue the node unsolved; return `Interrupted` |
| Limit with no incumbent | `Status::Interrupted`; value accessors expose the current working point (inspection only) |
| Search exhausted, no incumbent | `Err(Infeasible)` |
| Warm-start hint bad in any way | hint dropped (debug log), solve proceeds cold |
| Edit makes the problem infeasible | `Err(Infeasible)` from the re-solve — on the *composed base problem*, never on leaf state |

The standing project policy: when something cannot be done properly, fail loudly
(`panic!`/`unreachable!` with a comment) rather than approximate — a solver's silent wrong
answer is strictly worse than its crash.

---

## 9. Testing strategy

Three rings, innermost first:

1. **Unit tests** in each module: the solver primitives (bound changes vs fresh solves,
   basis round-trips, slack-basis recovery), driver behaviors (optimum finding, infeasible
   detection, deterministic node-limit interruption/resume, exact-exhaustion status), and
   pseudocost/selection arithmetic. Written TDD; several encode adversarially-verified facts
   (e.g. the warm-start liveness test *fails if the hint wiring is disconnected*).
2. **Public-API integration tests** (`src/tests/mip_api.rs`, `src/tests/resume.rs`):
   status semantics, panics, sign handling, edit composition, warm starts, sliced resumes
   equal unlimited solves value-for-value.
3. **The correctness suite** (`tests/suite`, `cargo test --release --test suite`) — a
   problem-based harness (parallel runner, ≤ 8 cores) where every answer is independently
   known: netlib/MIPLIB published optima, constructed instances, DP/brute-force oracles,
   plus a shadow model that re-validates every claimed solution (feasibility, integrality,
   objective consistency, and solver-soundness checks like "a feasible incumbent must never
   beat the proven optimum"). Cases are tiered **easy / medium / hard / xhard**; a tier
   flag is a cumulative upper limit (`-- --hard` runs easy + medium + hard). Easy + medium
   is the default run; CI runs the full hard tier with a five-minute per-case cap
   (`-- --hard --max-case-seconds 300`); **xhard** (`-- --xhard`) holds the MILPBench
   families beyond the solver's current ceiling, on 10-minute budgets with externally
   certified (HiGHS) optima — those cases assert clean interrupts and bound sanity rather
   than completion. File-based cases derive their tier from the folder their instance
   lives in (`tests/suite/data/<tier>/<source>/`), so moving a file re-tiers its cases.
   Both benchmark readers are thin adapters over external dev-dependency crates —
   `mps` for MPS files, `lp_parser_rs` for CPLEX-LP files — with the semantics layer
   (integer markers, bound conventions, objective offsets) owned and documented in
   `tests/suite/mps_milp.rs` and `tests/suite/lp_format.rs`.
   The `milp/warm-restart-*` and `milp/nodelimit-steps-*` families exercise the
   restart-with-hint loop on real problems with monotone-improvement assertions.

If you change ANYTHING in the solver, the default suite tier is the first thing to run.

---

## 10. Performance characteristics and current limits

**What bound-change branching buys, structurally:** a row-based B&B adds a constraint row
per branch (matrix rebuild + LU refactorization + a new slack column at every node) and
stores solver clones per tree node. This design's per-node cost is: one bound change + a short
warm-started dual simplex (dive), or one basis refactorization (jump); per-node memory is a
basis snapshot + a bound list. Pseudocost branching then halved the suite's wall-clock again
by shrinking the trees themselves.

**Known, accepted costs:** the best-bound jump and the (only when `mip_gap > 0`) bound scan
are `O(open)` linear scans — fine at current scales, a heap if profiling ever says otherwise.
Basis snapshots per node are `total_vars` bytes; basis *aging* (snapshot every k-th depth)
is the standard next step if memory becomes a concern on deep trees.

**Where the ceiling currently sits** (measured, MILPBench easy tier at 60s/instance):
Capacitated Facility Location instances solve to proven optimality in seconds; the
graph-structured families (MIS, MVC, Set Cover, Combinatorial Auctions, MIKS) at 20k–60k
rows produce clean `Interrupted` — the machinery survives 160k-variable models without
error, but proving optimality there needs the phase-4 items below.

---

## 11. Extension points (rough order of payoff)

- **Cutting planes inside the tree.** Gomory cuts exist as a user API but are not used
  during B&B. Root-node cut rounds (cut-and-branch) are the classic next multiplier on the
  graph-structured instances above. The architecture supports it: cuts are rows added to the
  base problem before the tree starts (NOT during — node bounds assume a fixed row set).
- **Primal heuristics.** Rounding + diving heuristics produce early incumbents → earlier
  pruning. `try_adopt_incumbent(state, true)` is the safe funnel for any candidate a
  heuristic produces.
- **Integer presolve.** Bound rounding is a no-op today (integer bounds come in integral via
  the API), but coefficient tightening / probing on binaries is real value for big-M models.
- **SOS1/SOS2, deferred by design.** The intended route is *implicit detection* (recognize
  `Σ binaries ≤ 1` rows in a presolve pass). `Node.bound_changes` being a list already
  supports multi-variable branches (fix half the set to 0), so SOS branching needs no
  redesign of the tree.
- **Basis aging / node memory** and **heap-based best-bound selection** — see §10.
- **Suspected simplex cycling / tail degeneracy** — one hard-tier problem burns its whole
  10-minute budget while already holding the correct answer (the bound proof never
  finishes). An anti-cycling fallback (Bland's rule) or bound perturbation is the classic
  fix, and pairs naturally with the cutting-plane work above.

---

## 12. History

Until mid-2026 the MILP layer drove branching through the public incremental API
(`add_constraint` per branch on cloned solvers), which coupled the tree, the engine, and
the user-facing result into one recursive tangle (`Solver ⊃ bb_state ⊃ Step ⊃ Solution ⊃
Solver`). Its symptoms — a deadline panic mid-search, edits acting on B&B leaves, errors
swallowed as pruning, no gap/bound, most-fractional branching — are each addressed by name
above by the current design.
