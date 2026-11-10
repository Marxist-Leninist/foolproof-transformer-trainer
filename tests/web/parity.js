/* Second half of tests/web/parity.py: load weights exported from the PyTorch model into the
   in-page engine and compare scores, loss and every gradient. Not meant to be run by hand. */
"use strict";
const path = require("path"), fs = require("fs");
const root = path.join(__dirname, "..", "..");
const D = require(path.join(root, "foolproof", "web", "design.js"));
const N = require(path.join(root, "docs", "nn.js"));

(async () => {
  const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  const kernels = await N.compileKernels(fs.readFileSync(path.join(root, "docs", "kernels.wasm")));
  let failed = 0;
  for (const c of cases) {
    const steps = D.expand(D.normalise(c.design)), { B, T, vocab } = c;
    for (const backend of ["js", "wasm"]) {
      if (backend === "wasm" && !kernels) { console.log("  skip wasm: not supported by this Node"); continue; }
      const net = new N.Net(steps, vocab, { batch: B, wasm: backend === "wasm" ? kernels : null });
      const mine = net.params.map(p => p.name).sort().join(","), theirs = Object.keys(c.state).filter(k => k !== "lm_head.weight").sort().join(",");
      if (mine !== theirs) { failed++; console.log(`  FAIL ${c.name}: table names differ\n    page:    ${mine}\n    pytorch: ${theirs}`); break; }
      net.load(Object.fromEntries(net.params.map(p => [p.name, { data: c.state[p.name] }])));
      const loss = net.forward(Int32Array.from(c.idx), B, T, Int32Array.from(c.targets));
      net.backward();
      let top = 0, gap = 0;
      for (let i = 0; i < c.logits.length; i++) { top = Math.max(top, Math.abs(c.logits[i])); gap = Math.max(gap, Math.abs(c.logits[i] - net.logits[i])); }
      let worst = 0, where = "";
      for (const p of net.params) {
        const g = c.grads[p.name];
        let num = 0, den = 0;
        for (let i = 0; i < g.length; i++) { num += (g[i] - p.g[i]) ** 2; den += g[i] ** 2; }
        const rel = Math.sqrt(num / (den + 1e-30));
        if (rel > worst) { worst = rel; where = p.name; }
      }
      const good = gap / top < 2e-4 && Math.abs(loss - c.loss) < 2e-4 * Math.max(1, c.loss) && worst < 2e-3;
      if (!good) failed++;
      console.log(`  ${good ? "ok  " : "FAIL"} ${c.name} [${net.backend}]: scores differ by ${(gap / top).toExponential(1)} of their size, loss ${loss.toFixed(5)} vs PyTorch ${c.loss.toFixed(5)}, worst gradient table off by ${worst.toExponential(1)} (${where})`);
    }
  }
  console.log(failed ? `${failed} comparison(s) FAILED` : "the in-page engine matches PyTorch");
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
