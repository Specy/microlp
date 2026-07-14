// Smoke test: solve an LP and a MILP with microlp compiled to WebAssembly.
//
//   wasm-pack build --target nodejs
//   node test.js
//
// Exits non-zero if either solve returns the wrong optimum.

const solver = require("./pkg/microlp_wasm_example.js");

function approx(a, b, tol = 1e-6) {
  return Math.abs(a - b) <= tol;
}

// solve_lp / solve_milp each return [objective, x, y].
const lp = solver.solve_lp();
const milp = solver.solve_milp();

console.log(`LP   → objective ${lp[0]}  at x=${lp[1]}, y=${lp[2]}   (expected 36 at 2, 6)`);
console.log(`MILP → objective ${milp[0]}  at x=${milp[1]}, y=${milp[2]}   (expected 22 at 8, 2)`);

const ok =
  approx(lp[0], 36) && approx(lp[1], 2) && approx(lp[2], 6) &&
  approx(milp[0], 22) && approx(milp[1], 8) && approx(milp[2], 2);

if (!ok) {
  console.error("FAIL: microlp returned an unexpected solution under wasm");
  process.exit(1);
}
console.log("PASS: microlp solved both problems correctly under wasm");
