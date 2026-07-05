# Vendored benchmark data — provenance, attribution, licensing

## Licensing summary

Neither the netlib LP collection nor MIPLIB 3 ships a formal (SPDX-style)
license text. Both are de facto freely redistributable benchmark collections:
they have been distributed publicly since the 1980s/90s explicitly so that LP
and MIP solvers can be benchmarked against them, and they are routinely
vendored verbatim by open-source solver projects (SciPy, HiGHS, COIN-OR, and
others). The netlib FAQ states most netlib content has "no restrictions on
their use"; the COIN-OR mirror repositories these copies came from apply their
EPL-2.0 license **only to the build system, explicitly excluding the data
files**, which carry the original collections' terms.

Because there is no formal grant, these files are treated as **test fixtures,
not part of the library**: `Cargo.toml` excludes `tests/suite/data` from the
published crate, so the crates.io artifact contains only Apache-2.0 code. The
files exist only in the git repository, used solely for their intended purpose
(benchmarking a solver). If you redistribute the repository in a context with
stricter requirements, delete this directory — the affected `netlib/*` and
`miplib/*` suite cases will fail with a clear "cannot read" message while all
constructed and oracle-verified cases keep working.

## netlib/ — LP benchmark instances (plain MPS)

Source: the classic netlib LP collection (https://netlib.org/lp/data/),
vendored from the uncompressed mirrors in COIN-OR's Data-Netlib repository
(https://github.com/coin-or-tools/Data-Netlib). Reference: D. M. Gay,
"Electronic mail distribution of linear programming test problems",
COAL Newsletter 13 (1985) 10-12.

Expected optimal objective values used by the suite are the official "BR"
(best reported) values from netlib's `lp/data/readme` (all instances are
minimization). Contributors per the same readme:

| instance | rows | optimal objective | contributed by |
| -------- | ---- | ----------------- | -------------- |
| afiro    |   28 | -4.6475314286E+02 | M. Saunders (Stanford SOL) |
| sc50a    |   51 | -6.4575077059E+01 | M. Saunders (Stanford SOL) |
| sc50b    |   51 | -7.0000000000E+01 | M. Saunders (Stanford SOL) |
| sc105    |  106 | -5.2202061212E+01 | M. Saunders (Stanford SOL) |
| adlittle |   57 |  2.2549496316E+05 | M. Saunders (Stanford SOL) |
| blend    |   75 | -3.0812149846E+01 | M. Saunders (Stanford SOL) |
| kb2      |   44 | -1.7499001299E+03 | M. Saunders (Stanford SOL) |
| share2b  |   97 | -4.1573224074E+02 | M. Saunders (Stanford SOL) |
| stocfor1 |  118 | -4.1131976219E+04 | G. Gassmann |
| scagr7   |  130 | -2.3313892548E+06 | B. Fourer |
| israel   |  175 | -8.9664482186E+05 | B. Fourer |
| brandy   |  221 |  1.5185098965E+03 | B. Fourer |
| beaconfd |  174 |  3.3592485807E+04 | B. Fourer |
| sctap1   |  301 |  1.4122500000E+03 | B. Fourer |

Normalization: `blend.mps` as distributed uses fixed-format MPS with a *blank*
RHS vector name (its RHS lines start directly with row names). Free-format,
whitespace-tokenized readers — including microlp's `MpsFile` — cannot
disambiguate that, so the vendored copy adds the explicit (semantically
irrelevant) vector name `RHS` to those four lines. No numeric content changed.

## miplib3/ — MILP benchmark instances (plain MPS with INTORG/INTEND markers)

Source: MIPLIB 3.0, vendored from COIN-OR's Data-miplib3 repository
(https://github.com/coin-or-tools/Data-miplib3). Reference: R. E. Bixby,
S. Ceria, C. M. McZeal, M. W. P. Savelsbergh, "An updated mixed integer
programming library: MIPLIB 3.0", Optima 58 (1998) 12-15. MIPLIB is a publicly
distributed benchmark library maintained for exactly this purpose (successor
editions live at https://miplib.zib.de).

Expected values are from the official `miplib3.cat` catalog (INT SOLN /
LP SOLN columns; all instances are minimization). Originator / formulator /
donator are from the same catalog's ORIGINS index:

| instance | int solution | LP relaxation | origin (originator; donator) |
| -------- | ------------ | ------------- | ---------------------------- |
| flugpl   | 1201500      | 1167185.73    | H. M. Wagner; E. A. Boyd |
| p0033    | 3089         | 2520.57       | CJP set; E. A. Boyd |
| p0201    | 7615         | 6875.0        | CJP set; E. A. Boyd |
| stein27  | 18           | 13.0          | G. L. Nemhauser; E. A. Boyd |
| stein45  | 30           | 22.0          | G. L. Nemhauser; E. A. Boyd |
| lseu     | 1120         | 834.68        | C. E. Lemke, E. L. Johnson; J. J. Forrest |
| mod008   | 307          | 290.93        | IBM France; J. J. Forrest |
| enigma   | 0.0          | 0.0           | H. Crowder; E. A. Boyd |
| rgn      | 82.1999      | 48.7999       | L. E. Schrage, L. A. Wolsey; M. Savelsbergh |
| vpm1     | 20           | 15.4167       | L. A. Wolsey; M. Savelsbergh |
| misc03   | 3360         | 1910.0        | (unlisted); G. Astfalk |
| gt2      | 21166.000    | 13460.233074  | (unlisted); S. Ceria |
| bell3a   | 878430.32    | 862578.64     | W. Cook |

Note: microlp's own `MpsFile` parser has no integer-marker support, so the
suite uses its own minimal reader (`tests/suite/mps_milp.rs`) for these files.
Integer columns that receive no BOUNDS entry default to bounds [0, 1] (the
MPSX-era convention MIPLIB 3 files were written for); every instance's parse
is validated by checking the LP relaxation objective against the published
value above.

## Constructed cases (no external data)

Everything outside this directory is either original to this suite (oracle
families, random certified LPs, unit cases) or a formulation of a published
mathematical construction, cited here:

* Klee-Minty cubes (`lp/klee-minty-*`): V. Klee, G. J. Minty, "How good is
  the simplex algorithm?", in *Inequalities III*, 1972.
* Beale's cycling example (`lp/beale-cycling`): E. M. L. Beale, "Cycling in
  the dual simplex algorithm", Naval Research Logistics Quarterly 2 (1955).

Mathematical formulations are facts, not copyrightable content; the citations
are attribution, not license requirements.
