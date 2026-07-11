# microlp benchmark

microlp measured against other open-source solvers on the 35 LP/MILP instances vendored for the [correctness suite](tests/suite/README.md) (netlib LPs, MIPLIB 3 and MILPBench MILPs). Regenerate with:

```bash
cargo run -p microlp-benchmark --release
```

**Method.** Each (instance, solver) pair runs in a fresh process with a 300 s budget. The measured time covers building the solver's native model from the parsed instance plus solving it; file parsing is excluded. Solves finishing under 200 ms are repeated (up to 5 times) and the fastest run is kept. Rival solvers run single-threaded with relative MIP gap 0, i.e. they must prove exact optimality just like microlp's default. Every returned solution is independently re-checked against the instance (bounds, integrality, constraints, objective); a check failure counts as a failed solve. Instances that every solver finishes in under 1 ms are excluded as trivial (0 excluded, listed at the end). When no solver proves an optimum inside the shared budget on an instance microlp left unfinished, the primary rival gets one longer certification solve (untimed, all threads, still exact); its value feeds only the correction-gap column.

| | |
| --- | --- |
| Date | 2026-07-11 (UTC) |
| Machine | Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (12 threads available) |
| OS | windows x86_64 |
| microlp | commit 825191b + uncommitted changes |
| Budget | 300 s per (instance, solver) |

## Outcomes

| solver | proved optimum | timed out | failed |
| --- | ---: | ---: | ---: |
| microlp | 29 | 6 | 0 |
| highs | 30 | 4 | 1 |
| scip | 31 | 4 | 0 |
| clarabel | 14 | 0 | 0 |

**Failures:**

| instance | solver | detail |
| --- | --- | --- |
| xhard/milpbench/MIKS_easy_instance_26 | highs | error: unbounded-or-infeasible (HiGHS did not separate the two) |

## Largest slowdowns vs highs

The instances where microlp is furthest behind highs — the most concrete list of what to profile next. The time ratio is microlp's time over highs's (with a 1 ms shift on both); below 1× microlp is faster.

| instance | class | microlp | highs | time ratio |
| --- | --- | ---: | ---: | ---: |
| hard/miplib3/gt2 | MILP | 3.03 s | 309.6 ms | 9.76× |
| hard/miplib3/bell3a | MILP | 11.97 s | 3.16 s | 3.79× |
| medium/netlib/scagr7 | LP | 3.4 ms | 6.9 ms | 0.55× |
| medium/miplib3/lseu | MILP | 699.3 ms | 1.52 s | 0.46× |
| hard/netlib/brandy | LP | 9.1 ms | 22.7 ms | 0.42× |
| medium/netlib/sc105 | LP | 1.3 ms | 4.6 ms | 0.41× |
| medium/netlib/share2b | LP | 2.3 ms | 7.1 ms | 0.41× |
| medium/netlib/kb2 | LP | 0.6 ms | 3.0 ms | 0.40× |
| medium/netlib/sc50b | LP | 0.4 ms | 2.7 ms | 0.40× |
| medium/netlib/sc50a | LP | 0.4 ms | 2.5 ms | 0.39× |

## Not solved by microlp within the budget

What microlp had in hand when the 300 s budget ran out. The *correction gap* is how far its incumbent is from the reference optimum, relative to the reference: 0% means the right answer was found but not yet proven optimal; "no incumbent" means the search had no integer-feasible solution yet. The *self-reported gap* is microlp's own bound-based estimate at the same moment. Track this table release over release: entries should move up (smaller gaps) and eventually leave the table.

| instance | class | microlp status | incumbent | self-reported gap | reference optimum | correction gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| hard/miplib3/vpm1 | MILP | feasible | 20 | 8.08% | 20 (scip) | 0.00% |
| xhard/milpbench/CAT_easy_instance_25 | MILP | interrupted | — | — | — | no reference proved in this run |
| xhard/milpbench/IS_easy_instance_11 | MILP | interrupted | — | — | — | no reference proved in this run |
| xhard/milpbench/MIKS_easy_instance_26 | MILP | interrupted | — | — | — | no reference proved in this run |
| xhard/milpbench/MVC_easy_instance_19 | MILP | feasible | 5501.2688 | 10.39% | — | no reference proved in this run |
| xhard/milpbench/SC_easy_instance_6 | MILP | interrupted | — | — | — | no reference proved in this run |

Instances with no reference optimum were not proved by any solver in this run at this budget; the correctness suite holds externally certified bounds for them (see tests/suite/data/xhard/milpbench/README.md).

## Full results

Cells show wall time; non-optimal outcomes are spelled out.

| instance | class | rows | cols | int | nnz | microlp | highs | scip | clarabel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard/miplib3/bell3a | MILP | 123 | 133 | 71 | 347 | 11.97 s | 3.16 s | 1.25 s | n/a |
| hard/miplib3/enigma | MILP | 21 | 100 | 100 | 289 | 130.5 ms | 1.18 s | 263.8 ms | n/a |
| hard/miplib3/gt2 | MILP | 29 | 188 | 188 | 376 | 3.03 s | 309.6 ms | 43.4 ms | n/a |
| hard/miplib3/mod008 | MILP | 6 | 319 | 319 | 1243 | 2.00 s | 6.38 s | 810.5 ms | n/a |
| hard/miplib3/stein45 | MILP | 331 | 45 | 45 | 1034 | 26.79 s | 97.36 s | 21.00 s | n/a |
| hard/miplib3/vpm1 | MILP | 234 | 378 | 168 | 749 | feasible, gap 8.08% | 202.0 ms | 12.5 ms | n/a |
| hard/netlib/beaconfd | LP | 173 | 262 | 0 | 3375 | 1.5 ms | 15.0 ms | 7.6 ms | 5.4 ms |
| hard/netlib/brandy | LP | 220 | 249 | 0 | 2148 | 9.1 ms | 22.7 ms | 24.0 ms | 5.9 ms |
| hard/netlib/israel | LP | 174 | 142 | 0 | 2269 | 5.6 ms | 15.8 ms | 12.2 ms | 4.4 ms |
| hard/netlib/sctap1 | LP | 300 | 480 | 0 | 1692 | 5.3 ms | 22.1 ms | 13.4 ms | 6.8 ms |
| medium/milpbench/CFL_easy_instance_12 | MILP | 800 | 160400 | 148800 | 320400 | 32.0 ms | 17.32 s | 959.0 ms | n/a |
| medium/milpbench/CFL_easy_instance_15 | MILP | 800 | 160400 | 150400 | 320400 | 31.5 ms | 17.26 s | 937.0 ms | n/a |
| medium/miplib3/flugpl | MILP | 18 | 18 | 11 | 46 | 22.3 ms | 535.4 ms | 73.3 ms | n/a |
| medium/miplib3/lseu | MILP | 28 | 89 | 89 | 309 | 699.3 ms | 1.52 s | 494.9 ms | n/a |
| medium/miplib3/misc03 | MILP | 96 | 160 | 159 | 2053 | 268.1 ms | 2.47 s | 1.25 s | n/a |
| medium/miplib3/p0033 | MILP | 16 | 33 | 33 | 98 | 15.1 ms | 195.7 ms | 24.4 ms | n/a |
| medium/miplib3/p0201 | MILP | 133 | 201 | 201 | 1923 | 785.8 ms | 4.33 s | 711.7 ms | n/a |
| medium/miplib3/rgn | MILP | 24 | 180 | 100 | 460 | 244.8 ms | 1.42 s | 365.8 ms | n/a |
| medium/miplib3/stein27 | MILP | 118 | 27 | 27 | 378 | 487.9 ms | 3.27 s | 134.2 ms | n/a |
| medium/netlib/adlittle | LP | 56 | 97 | 0 | 383 | 1.2 ms | 5.2 ms | 6.3 ms | 0.9 ms |
| medium/netlib/afiro | LP | 27 | 32 | 0 | 83 | 0.1 ms | 2.1 ms | 3.8 ms | 0.2 ms |
| medium/netlib/blend | LP | 74 | 83 | 0 | 491 | 1.3 ms | 5.9 ms | 6.2 ms | 1.0 ms |
| medium/netlib/kb2 | LP | 43 | 41 | 0 | 286 | 0.6 ms | 3.0 ms | 4.4 ms | 0.7 ms |
| medium/netlib/sc105 | LP | 105 | 103 | 0 | 280 | 1.3 ms | 4.6 ms | 4.9 ms | 0.8 ms |
| medium/netlib/sc50a | LP | 50 | 48 | 0 | 130 | 0.4 ms | 2.5 ms | 3.9 ms | 0.4 ms |
| medium/netlib/sc50b | LP | 50 | 48 | 0 | 118 | 0.4 ms | 2.7 ms | 3.8 ms | 0.3 ms |
| medium/netlib/scagr7 | LP | 129 | 140 | 0 | 420 | 3.4 ms | 6.9 ms | 5.8 ms | 1.4 ms |
| medium/netlib/share2b | LP | 96 | 79 | 0 | 694 | 2.3 ms | 7.1 ms | 7.6 ms | 1.2 ms |
| medium/netlib/stocfor1 | LP | 117 | 111 | 0 | 447 | 1.2 ms | 5.6 ms | 5.7 ms | 1.4 ms |
| xhard/milpbench/BIP_easy_instance_10 | MILP | 2900 | 40810 | 40000 | 842400 | 5.43 s | 34.60 s | 1.78 s | n/a |
| xhard/milpbench/CAT_easy_instance_25 | MILP | 20000 | 20000 | 20000 | 100000 | no solution in budget | no solution in budget | no solution in budget | n/a |
| xhard/milpbench/IS_easy_instance_11 | MILP | 60000 | 20000 | 20000 | 120000 | no solution in budget | no solution in budget | no solution in budget | n/a |
| xhard/milpbench/MIKS_easy_instance_26 | MILP | 50000 | 50000 | 37298 | 200000 | no solution in budget | failed | unbounded (471.8 ms) | n/a |
| xhard/milpbench/MVC_easy_instance_19 | MILP | 60000 | 20000 | 20000 | 120000 | feasible, gap 10.39% | no solution in budget | no solution in budget | n/a |
| xhard/milpbench/SC_easy_instance_6 | MILP | 40000 | 40000 | 40000 | 160000 | no solution in budget | no solution in budget | no solution in budget | n/a |

