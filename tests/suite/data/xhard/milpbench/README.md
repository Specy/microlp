# MILPBench vendored instances (xhard tier)

One instance per MILPBench "easy" family that is (or was expected to be)
beyond microlp's practical ceiling. These run only under `--xhard`, with a
10-minute default budget per case (`cargo test --release --test suite --
--xhard`). They are read by the `tests/suite/lp_format.rs` reader (an adapter
over the `lp_parser_rs` crate) and registered in
`tests/suite/cases/milpbench_xhard.rs`.

The CFL family — the one microlp solves in well under a second — lives in
`data/medium/milpbench/` instead; see the README there.

## Source & license

- **Repository:** MILPBench — <https://github.com/thuiar/MILPBench>
- **Dataset index:** <https://github.com/thuiar/MILPBench/blob/main/Benchmark%20Datasets/README.md>
- **License:** Apache-2.0 (same as microlp). The instances are redistributed
  unmodified under that license; upstream copyright remains with the
  MILPBench authors (THUIAR).

The `.lp` files are Gurobi-emitted CPLEX LP format, byte-for-byte as they
appear inside the upstream per-family `*_easy_instance.zip` Drive archives
(ids in the medium-tier README's download table). They are **stored gzipped**
(`<file>.lp.gz`) to keep the repository small; the suite resolves the storage
form transparently (`cases::locate`) and decompresses in memory at case run
time (`cases::read_instance`) — nothing is written to disk. The `sha256`
hashes below are of the *decompressed* files, certifying byte-identity with
upstream.

## Certification

MILPBench ships no reference optima, so every vendored instance carries an
**externally certified** verdict produced by HiGHS v1.15.1 (via `highspy`,
2026-07-10) with `mip_rel_gap = 0`, `mip_abs_gap = 0` — i.e. targeting a
*proof*, not a heuristic value. Two certificate strengths exist:

* **optimum** — HiGHS proved optimality (zero gap) within its budget;
* **envelope `[lo, hi]`** — HiGHS could *not* close the gap locally (pass 1:
  1 thread × 1500 s; pass 2: 2 threads × 3600 s), but proved that the true
  optimum lies in the interval: one endpoint is its proven dual bound, the
  other its best feasible incumbent (best across both passes). Weaker than
  an optimum, still with teeth: a claimed optimum outside the envelope, or a
  feasible incumbent beating the proven bound side, is unsound.

| File | vars | cons | sense | certificate | HiGHS effort |
|---|---|---|---|---|---|
| `BIP_easy/BIP_easy_instance_10.lp` | 30000 | 19999 | min | **optimum 80800.0** (proven, gap 0) | 12.9 s |
| `MIKS_easy/MIKS_easy_instance_26.lp` | 50000 | 50000 | max | **unbounded** (see below) | 2.3 s + 0.4 s |
| `MIS_easy/IS_easy_instance_11.lp` | 20000 | 60000 | max | envelope **[523.6264397361941, 5075.512443744508]** (gap 869% at limit) | 1500 s + 3600 s |
| `MVC_easy/MVC_easy_instance_19.lp` | 20000 | 60000 | min | envelope **[4937.689925436303, 9570.635205426428]** (gap 48% at limit) | 1500 s + 3600 s |
| `CAT_easy/CAT_easy_instance_25.lp` | 20000 | 20000 | max | envelope **[979.6483854074107, 4939.24371571862]** (gap 404% at limit) | 1500 s + 3600 s |
| `SC_easy/SC_easy_instance_6.lp` | 40000 | 40000 | min | envelope **[3119.5563850909934, 3379.6673668697686]** (gap 7.7% at limit) | 1500 s + 3600 s |

`sha256`:

```
1ed88377dd5043ede80a2fdee084ec40c3e31592ad0002183195a77c921045fb  BIP_easy/BIP_easy_instance_10.lp
1a8e26bdbc4b0bd089ab138c42f9fb5cb7ff7ed6421f428e73563226c21bf90e  MIKS_easy/MIKS_easy_instance_26.lp
75c4a9e974fb5434e295048388bc80fc1cec48ed64c79fedf7db649158997f46  MIS_easy/IS_easy_instance_11.lp
27aa4b312da5d92e292ac01262639f878a7fd96989e997009484065bd4300df7  MVC_easy/MVC_easy_instance_19.lp
8e8abe2540e71b6a5e0758cc55f3ba4ed44f8eaa6ef9becd019d71abb01936e4  CAT_easy/CAT_easy_instance_25.lp
9150c7fcb363bd88bf5f1cb3581f055f77ff1c4ca713480cd9b5bd114699d6c1  SC_easy/SC_easy_instance_6.lp
```

The four envelope instances are the exact files test-solved in the original
family inventory (the smallest instance of each family's archive).

### The MIKS_easy family is degenerate (unbounded as generated)

Every sampled `MIKS_easy` instance (26, 9, 15, 19 — the four smallest) is
reported "primal infeasible or unbounded" by HiGHS within seconds. The cause
is visible in the files: the `Bounds` section is empty and the `Binaries`
list skips some variables, which therefore default to *continuous*
`[0, +inf)` with positive objective coefficients under `Maximize`. For
instance 26 specifically, HiGHS proves:

- the LP relaxation is **Unbounded** (not infeasible), and
- the instance is **integer-feasible** (a zeroed-objective MILP solve
  returns Optimal),

so the MILP itself is unbounded (feasible point + unbounded continuous ray).
This is a MILPBench generator defect, and it makes a genuinely useful xhard
case of a different flavor: a solver must never claim a finite proven
optimum on this input. The case accepts `Err(Unbounded)` (correct answer), a
clean `Interrupted`, or an honest shadow-validated `Feasible` incumbent —
and fails on `Optimal` or `Err(Infeasible)`.

## Case contract (see `cases/milpbench_xhard.rs`)

For optimum-certified instances: `Optimal` must match the certificate;
`Feasible` must validate against the shadow model and never be *better* than
the certificate (direction-aware); `Interrupted` passes; a solve error
fails. For envelope-certified instances: a claimed `Optimal` must land
inside the envelope, and a `Feasible` incumbent must never beat the proven
dual-bound side. As the solver improves, the certificates bite harder
automatically.

Notably, **microlp already solves the BIP instance to proven optimality in
~15 s**, and its optimum matches HiGHS's certificate exactly — a genuine
two-solver cross-validation. It stays in xhard per the tier's "not expected
to finish" charter; move its file to a lower tier folder to promote it into
the default/CI runs.

## Adding an instance

Drop the `.lp` under `<family>/` here (gzip it as `<file>.lp.gz` if it is
large — the suite reads either form), certify it with a reference solver
(record version, options, wall time), and append a row to `CERTIFIED` in
`tests/suite/cases/milpbench_xhard.rs` plus this README's table — the row
always uses the logical `.lp` name regardless of storage. Downloads go to a
scratch directory, never the repo; the per-family Drive archive ids are in
`data/medium/milpbench/README.md`.

This directory is excluded from the crates.io package via
`exclude = ["tests/suite/data"]` in `Cargo.toml`.
