/* Checks the in-page transformer (docs/nn.js) with nothing but Node:
     1. every hand-written gradient against finite differences (double precision)
     2. the design rules (counting, codes, snapping, what the app would refuse)
     3. the WebAssembly loops against the plain JavaScript ones
     4. that it really learns: the loss on Shakespeare must fall a long way
   Run:  node tests/web/test_nn.js */
"use strict";
const path = require("path");
const fs = require("fs");
const root = path.join(__dirname, "..", "..");
const D = require(path.join(root, "foolproof", "web", "design.js"));
const N = require(path.join(root, "docs", "nn.js"));
const { Net, sample, rng32 } = N;

let failed = 0;
function ok(cond, msg) { if (cond) console.log("  ok   " + msg); else { failed++; console.log("  FAIL " + msg); } }
const design = (width, heads, context, letters) => ({ version: 1, width, heads, context, blocks: letters.split("").map(ch => ({ type: { t: "transformer", a: "attention", f: "feed_forward" }[ch] })) });

// ---------------------------------------------------------------- 1. gradients
function gradcheck(steps, vocab, B, T, seed) {
  const net = new Net(steps, vocab, { float64: true, seed, initScale: 12, batch: B });
  const rand = rng32(seed + 1);
  // the gains start at exactly 1; move them so their gradients are tested away from that point
  for (const p of net.params) if (!p.decay) for (let i = 0; i < p.w.length; i++) p.w[i] = 0.6 + rand();
  const idx = new Int32Array(B * T), tg = new Int32Array(B * T);
  for (let i = 0; i < B * T; i++) { idx[i] = Math.floor(rand() * vocab); tg[i] = Math.floor(rand() * vocab); }
  net.forward(idx, B, T, tg);
  net.backward();
  const analytic = net.params.map(p => Float64Array.from(p.g));
  let worst = 0, checked = 0;
  net.params.forEach((p, pi) => {
    const picks = Math.min(p.w.length, 14);
    for (let k = 0; k < picks; k++) {
      const i = Math.floor(rand() * p.w.length), keep = p.w[i], h = 1e-5;
      p.w[i] = keep + h; const up = net.forward(idx, B, T, tg);
      p.w[i] = keep - h; const down = net.forward(idx, B, T, tg);
      p.w[i] = keep;
      const numeric = (up - down) / (2 * h), a = analytic[pi][i];
      const err = Math.abs(numeric - a) / Math.max(1e-7, Math.abs(numeric) + Math.abs(a));
      if (err > worst) worst = err;
      checked++;
    }
  });
  return { worst, checked };
}
async function main() {
  console.log("gradients vs finite differences");
  const cases = [
    ["two transformer layers", D.expand(design(8, 2, 6, "tt")), 11, 3, 6],
    ["attention only, 1 head", D.expand(design(8, 1, 5, "a")), 9, 2, 5],
    ["feed-forward only", D.expand(design(6, 1, 4, "ff")), 7, 2, 4],
    ["own order: f a t a", D.expand(design(16, 4, 7, "fata")), 13, 2, 7],
    ["no blocks at all", D.expand(design(6, 1, 4, "")), 8, 2, 4],
    ["text shorter than its memory", D.expand(design(8, 2, 12, "t")), 10, 2, 5],
    ["mixed head counts and sizes (engine only)", { width: 16, context: 6, layers: 3, blocks: [{ type: "attention", heads: 8 }, { type: "feedforward", mult: 3 }, { type: "attention", heads: 2 }, { type: "feedforward", mult: 1 }] }, 12, 2, 6],
  ];
  cases.forEach(([name, steps, vocab, B, T], i) => {
    const r = gradcheck(steps, vocab, B, T, 100 + i);
    ok(r.worst < 2e-6, `${name}: ${r.checked} numbers checked, worst relative error ${r.worst.toExponential(2)}`);
  });

  // ---------------------------------------------------------------- 2. design rules
  console.log("design rules");
  {
    const d = D.normalise(design(64, 4, 64, "tt"));
    ok(D.countParams(d, 65) === new Net(D.expand(d), 65).numParams(), "parameter count matches the network (" + D.countParams(d, 65).toLocaleString("en") + ")");
    const odd = D.normalise(design(48, 4, 32, "afta"));
    ok(D.countParams(odd, 97) === new Net(D.expand(odd), 97).numParams(), "count matches for a custom order");
    ok(D.expand(odd).blocks.length === 5 && D.expand(odd).layers === 4, "a transformer layer expands to attention + feed-forward");
    const names = new Net(D.expand(odd), 97).params.map(p => p.name).join(" ");
    ok(names.includes("blocks.0.op.qkv.weight") && names.includes("blocks.1.op.fc.weight") && names.includes("blocks.2.attn.proj.weight") && names.includes("blocks.2.mlp.fc.weight") && names.includes("blocks.2.norm2.weight"), "tables carry the same names PyTorch gives them");
    ok(D.code(odd) === "fp1-w48-h4-c32-afta" && JSON.stringify(D.parse(D.code(odd))) === JSON.stringify(odd), "design code round trip (" + D.code(odd) + ")");
    ok(D.parse("fp1-w48-h5-c32-a") === null && D.parse("hello") === null && D.parse("fp1-w48-h4-c32-x") === null, "bad codes are refused");
    ok(D.code(design(32, 2, 32, "")) === "fp1-w32-h2-c32-0" && D.parse("fp1-w32-h2-c32-0").blocks.length === 0, "an empty sheet has a code too");
    let threw = 0;
    const bad = [design(50, 4, 32, "a"), design(48, 16, 32, "a"), { version: 1, width: 48, heads: 4, context: 32, blocks: [{ type: "banana" }] }, null, design(1024, 4, 32, "t"), design(64, 4, 2048, "t"), design(64, 4, 64, "t".repeat(17))];
    for (const b of bad) { try { D.normalise(b); } catch (e) { threw++; } }
    ok(threw === bad.length, "broken designs are refused with a message");
    const snapped = D.fix({ width: 50, heads: 16, context: 1000, blocks: [{ type: "attention" }, { type: "nonsense" }, { type: "feed_forward" }] }, D.WEB);
    ok(D.WEB.width.includes(snapped.width) && snapped.context === 128 && D.headsFor(snapped.width, D.WEB).includes(snapped.heads) && snapped.blocks.length === 2, "fix() snaps anything odd to something that works");
    let allHaveHeads = true;
    for (const lim of [D.WEB, D.APP]) for (const w of lim.width) if (!D.headsFor(w, lim).length) allHaveHeads = false;
    ok(allHaveHeads, "every offered width has at least one usable number of heads");
    ok(D.problem(design(64, 4, 64, "ff"), D.APP) !== "" && D.problem(design(64, 4, 64, ""), D.APP) !== "" && D.problem(design(64, 4, 64, "fa"), D.APP) === "" && D.problem(design(64, 4, 64, "ff"), D.WEB) === "", "the app asks for a block that looks back; the page lets you try without");
    ok(D.problem(design(768, 4, 64, "t".repeat(16)), D.APP, 60000) === "" && D.problem(design(768, 4, 64, "t".repeat(16)), D.APP, 400000) !== "", "the 350 million limit is checked");
    for (const lim of [D.WEB, D.APP]) for (const w of lim.width) for (const c of lim.context) for (const hd of D.headsFor(w, lim)) D.normalise(design(w, hd, c, "t"));
    ok(true, "everything the builder can offer passes the strict check");
  }

  // ---------------------------------------------------------------- 3. WebAssembly vs JavaScript
  console.log("fast loops (WebAssembly) vs plain JavaScript");
  const kernels = await N.compileKernels(fs.readFileSync(path.join(root, "docs", "kernels.wasm")));
  ok(!!kernels, "kernels.wasm compiles here");
  if (kernels) {
    const imports = WebAssembly.Module.imports(kernels);
    ok(imports.length === 1 && imports[0].kind === "memory", "kernels.wasm asks for nothing but memory (no network, no files)");
    for (const [name, d, V, B] of [["t a f, 4 heads", design(48, 4, 48, "taf"), 65, 7], ["odd sizes", design(48, 8, 32, "fat"), 131, 3], ["widest", design(128, 8, 64, "t"), 90, 2]]) {
      const steps = D.expand(D.normalise(d)), T = d.context;
      const a = new Net(steps, V, { seed: 1, batch: B }), b = new Net(steps, V, { seed: 1, batch: B, wasm: kernels });
      const r = rng32(3), idx = new Int32Array(B * T), tg = new Int32Array(B * T);
      for (let i = 0; i < idx.length; i++) { idx[i] = Math.floor(r() * V); tg[i] = Math.floor(r() * V); }
      let worst = 0, lossGap = 0;
      for (let s = 0; s < 4; s++) {
        const la = a.forward(idx, B, T, tg), lb = b.forward(idx, B, T, tg);
        a.backward(); b.backward();
        lossGap = Math.max(lossGap, Math.abs(la - lb));
        a.params.forEach((p, i) => {
          const q = b.params[i];
          let num = 0, den = 0;
          for (let j = 0; j < p.g.length; j++) { num += (p.g[j] - q.g[j]) ** 2; den += p.g[j] ** 2; }
          worst = Math.max(worst, Math.sqrt(num / (den + 1e-30)));
        });
        a.step(3e-3, 0.01, 1); b.step(3e-3, 0.01, 1);
      }
      ok(b.backend === "wasm" && a.backend === "js" && worst < 2e-4 && lossGap < 2e-5, `${name}: 4 training steps agree (gradients differ by ${worst.toExponential(1)}, loss by ${lossGap.toExponential(1)})`);
      const pa = Array.from(a.predict([1, 2, 3, 4, 5])), pb = Array.from(b.predict([1, 2, 3, 4, 5]));
      ok(pa.every((v, i) => Math.abs(v - pb[i]) < 1e-3), `${name}: both give the same next-letter scores`);
    }
  }

  // ---------------------------------------------------------------- 4. it learns
  console.log("learning");
  const bytes = fs.readFileSync(path.join(root, "foolproof", "assets", "tiny_shakespeare.txt"));
  const letters = Array.from(new Set(bytes)).sort((x, y) => x - y), idOf = new Int32Array(256).fill(-1);
  letters.forEach((b, i) => { idOf[b] = i; });
  const toks = Int32Array.from(bytes, b => idOf[b]);
  function train(d, steps, B, lr, seed, wasm) {
    const T = d.context, net = new Net(D.expand(D.normalise(d)), letters.length, { seed, batch: B, wasm });
    const rand = rng32(seed + 7), idx = new Int32Array(B * T), tg = new Int32Array(B * T);
    let first = null, ema = 0;
    const t0 = Date.now();
    for (let s = 0; s < steps; s++) {
      for (let b = 0; b < B; b++) {
        const o = Math.floor(rand() * (toks.length - T - 1));
        for (let t = 0; t < T; t++) { idx[b * T + t] = toks[o + t]; tg[b * T + t] = toks[o + t + 1]; }
      }
      const loss = net.forward(idx, B, T, tg);
      net.backward();
      net.step(lr * Math.min(1, (s + 1) / 40), 0.01, 1.0);
      if (first === null) first = loss;
      ema = 0.95 * ema + 0.05 * loss;
    }
    return { net, first, last: ema / (1 - Math.pow(0.95, steps)), sec: (Date.now() - t0) / 1000 };
  }
  function write(net, n, seed) {
    const rand = rng32(seed), ctx = [idOf[10]], out = [];
    for (let i = 0; i < n; i++) { const tok = sample(net.predict(ctx.slice(-net.T)), 0.8, 40, rand); ctx.push(tok); out.push(letters[tok]); }
    return Buffer.from(out).toString("utf8");
  }
  {
    const r = train(design(48, 4, 48, "t"), 400, 8, 4e-3, 3, kernels);
    ok(Math.abs(r.first - Math.log(letters.length)) < 0.25, `starts at random guessing (loss ${r.first.toFixed(2)}, chance is ${Math.log(letters.length).toFixed(2)})`);
    ok(r.last < 2.6, `one transformer layer, 400 steps: loss fell to ${r.last.toFixed(2)} in ${r.sec.toFixed(1)} s (${r.net.backend})`);
    const text = write(r.net, 70, 5);
    ok(/ /.test(text) && /[a-z]{2,}/.test(text), "it writes letters and spaces: " + JSON.stringify(text.slice(0, 54)));
    const again = new Net(r.net.design, letters.length, { seed: 99, batch: 8 });
    again.load(r.net.state());
    const a = Array.from(r.net.predict([5, 9, 12])), b = Array.from(again.predict([5, 9, 12]));
    ok(a.every((v, i) => Math.abs(v - b[i]) < 1e-3), "saved weights load back and give the same answers");
    const blind = train(design(48, 4, 48, "ff"), 400, 8, 4e-3, 3, kernels);
    ok(blind.last > r.last + 0.05, `without attention it does worse (${blind.last.toFixed(2)} vs ${r.last.toFixed(2)}): it cannot look back`);
  }
  console.log(failed ? `\n${failed} check(s) FAILED` : "\nall checks passed");
  process.exit(failed ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(1); });
