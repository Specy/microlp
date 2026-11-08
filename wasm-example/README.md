# microlp WebAssembly example

This example builds a [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen)
wrapper around microlp. JavaScript calls it to solve a linear program and a
mixed-integer linear program.

microlp reads the clock through [`web-time`](https://crates.io/crates/web-time)
instead of `std::time::Instant`, which panics on `wasm32-unknown-unknown`, so
`set_time_limit` and the branch & bound deadline checks work in browsers and
Node.

This example is outside the microlp Cargo workspace. Running `cargo build` at
the repository root does not build its `wasm-bindgen` dependency.

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

To use the example in a browser, build for the web target and import the
generated ES module:

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
