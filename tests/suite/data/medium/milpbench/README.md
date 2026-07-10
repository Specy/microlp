# MILPBench vendored instances (medium tier)

Vendored MILPBench instances that microlp **solves to proven optimality** —
they live in the medium tier (part of the default suite run) because they
finish in well under a second each. They are read by the
`tests/suite/lp_format.rs` CPLEX-LP reader (an adapter over the
`lp_parser_rs` crate) and registered in `tests/suite/cases/milpbench.rs`.

The MILPBench families microlp *cannot* finish live in
`data/xhard/milpbench/` instead, with externally certified optima — see the
README there.

## Source & license

- **Repository:** MILPBench — <https://github.com/thuiar/MILPBench>
- **Dataset index:** <https://github.com/thuiar/MILPBench/blob/main/Benchmark%20Datasets/README.md>
- **License:** Apache-2.0 (same as microlp). The instances are redistributed
  unmodified under that license; upstream copyright remains with the MILPBench
  authors (THUIAR).

The `.lp` files are Gurobi-emitted CPLEX LP format, byte-for-byte as they appear
inside the upstream `CFL_easy_instance.zip` archive. They are **stored
gzipped** (`<file>.lp.gz`) to keep the repository small; the suite resolves
the storage form transparently (`cases::locate`) and decompresses in memory
at case run time (`cases::read_instance`). The `sha256` hashes below are of
the *decompressed* files, certifying byte-identity with upstream.

## Vendored files (provenance)

| File | Family / tier | Original path in archive | vars (int) | cons | nnz | microlp optimum |
|---|---|---|---|---|---|---|
| `CFL_easy/CFL_easy_instance_15.lp` | CFL / easy | `CFL_easy_instance/LP/CFL_easy_instance_15.lp` | 160400 (150400) | 800 | 320400 | 9959.569649350798 |
| `CFL_easy/CFL_easy_instance_12.lp` | CFL / easy | `CFL_easy_instance/LP/CFL_easy_instance_12.lp` | 160400 (148800) | 800 | 320400 | 10000.194107917776 |

`sha256`:

```
8b17851c5aeb718606fe601d1a059dc0b844f710cf68efe402a91501aa704ee1  CFL_easy_instance_15.lp
372d7348f2b154f05472f4b45e8c5a9953081591452a5fb3ab1e17c7992019e2  CFL_easy_instance_12.lp
```

Both instances are ~12.3 MB decompressed (~3.4 MB stored; the smallest
instances in `CFL_easy`). The `CFL_easy` Drive archive they came from is
`https://drive.google.com/file/d/1z6oNG1ja6CwlsRYViXIzBj0j8Ch6sxdt/view`.

## Selection criteria

An instance is vendored only if microlp solves it to **proven optimality**
(`Status::Optimal`, `gap() == Some(0)`) within a per-instance budget, and its
returned solution passes the suite's independent shadow-model validation
(feasibility, bounds, integrality, objective consistency).

Of the seven classic MILPBench "easy" families, **only CFL** clears this bar.
CFL's easy instances are *wide* — ≈800 constraints over ≈160 400 variables — with
an integral LP relaxation, so the simplex proves optimality in ~0.03 s. Because
the LP relaxation is integral, the simplex-proven LP optimum is itself an
integer-feasible point and therefore the true MILP optimum; the recorded
objective is asserted as a regression guard.

MILPBench ships no certified optima for these families, so the objective is
**microlp-proven, not externally certified** (see above for why it is a genuine
optimum for these particular instances).

Every other easy family (MIS, MVC, SC, CAT, MIKS, BIP) is 20k–60k
*constraints* and beyond this solve-to-optimality bar; those families are
vendored in `data/xhard/milpbench/` instead, with externally certified
verdicts (HiGHS optima, proven envelopes, or a proven unboundedness) — see
the README there.

## Re-downloading / adding instances

Downloads go to a scratch directory, **never** into the repo. The per-family
Drive archives download directly with the confirm-token form:

```bash
# Example: the CFL_easy archive (115.7 MB). Substitute the file id from the
# MILPBench dataset index for other families. Cap at ~200 MB.
id=1z6oNG1ja6CwlsRYViXIzBj0j8Ch6sxdt
curl -sSL --max-filesize 209715200 \
  "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t" \
  -o CFL_easy.zip
unzip CFL_easy.zip 'CFL_easy_instance/LP/CFL_easy_instance_15.lp'
```

Per-family Drive file ids (easy tier):

| Family | Drive file id | Archive size |
|---|---|---|
| MIS_easy | `1slfuVvma5R5qwoFtIw1I3wLeIzg5EGvM` | 24.8 MB |
| MVC_easy | `10CCgHflKtxO4XOXZZCkD-pLU7vh81GZ0` | 73.5 MB |
| SC_easy | `1Oa9NiP6I1XpOkneLETGfKgTeYMDybVJX` | 94.3 MB |
| BIP_easy | `1u22POZv184KgWvjmJIYdyfwRZDBRX2vb` | 212 MB (over the 200 MB cap — not fetched) |
| CAT_easy | `1sWsUkQdKYi50HYAutunFieRMcmckXqHr` | 19.9 MB |
| CFL_easy | `1z6oNG1ja6CwlsRYViXIzBj0j8Ch6sxdt` | 115.7 MB |
| MIKS_easy | `1YYIAWqxzHAtfQQmKqdWy_UOhx8f3V8ZU` | 46.7 MB |

To add a vendored instance: drop its `.lp` here under `<family>/`, then append a
row to `VENDORED` in `tests/suite/cases/milpbench.rs` (family, file, budget,
microlp optimum, tolerance). Registration is data-driven.

This directory is excluded from the crates.io package via `exclude =
["tests/suite/data"]` in `Cargo.toml`.
