# microlp WebAssembly example

A minimal [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen) wrapper that
compiles microlp to WebAssembly and solves an LP and a MILP from JavaScript.

microlp is wasm-ready out of the box: it reads the clock through
[`web-time`](https://crates.io/crates/web-time) instead of `std::time::Instant`
(which panics on `wasm32-unknown-unknown`), so `set_time_limit` and the branch &
bound deadline checks work in the browser and in Node.

This crate is intentionally **not** part of the microlp workspace, so a plain
`cargo build` at the repo root never pulls in `wasm-bindgen`.

## Run under Node

```bash
# from this directory (wasm-example/)
wasm-pack build --target nodejs
node test.js
```

Expected output:

```
LP   → objective 36  at x=2, y=6   (expected 36 at 2, 6)
MILP → objective 22  at x=8, y=2   (expected 22 at 8, 2)
PASS: microlp solved both problems correctly under wasm
```

## Use in the browser

Build for the web target instead and import the generated ES module:

```bash
wasm-pack build --target web
```

```js
import init, { solve_lp, solve_milp } from "./pkg/microlp_wasm_example.js";

await init();
const [objective, x, y] = solve_lp();
console.log(objective, x, y); // 36 2 6
```

Requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) and the
`wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`).
