/* Foolproof Trainer - where the in-page network lives and learns.
   A worker runs beside the page, so the page stays smooth while the sums are done here.
   Nothing in this file talks to the internet except to fetch kernels.wasm from this same site:
   the text being learned from never leaves the device. */
"use strict";
importScripts("app/design.js", "nn.js");

const SLICE_MS = 30;          // train this long, then look at messages again
let kernels;                  // compiled kernels.wasm, or null (then plain JavaScript is used)
const kernelsReady = fetch("kernels.wasm").then(r => (r.ok ? r.arrayBuffer() : null))
  .then(buf => (buf ? FPNet.compileKernels(new Uint8Array(buf)) : null)).catch(() => null)
  .then(mod => { kernels = mod; });

let net = null, toks = null, nTrain = 0, vocab = [], idOf = null, base = null;
let B = 8, T = 64, lrMax = 3e-3;
let idx = null, tgt = null, evalSets = null, rand = null, sampleRand = null;
let running = false, step = 0, seconds = 0, ema = 0, lastLoss = null, sliceStart = 0;
let nextTick = 0, nextSample = 0, nextEval = 0, tickStep = 0, tickSeconds = 0, speed = 0;
let job = null;               // a piece of writing somebody asked for

const channel = new MessageChannel();
channel.port1.onmessage = loop;
const again = () => channel.port2.postMessage(0);

onmessage = ev => {
  const m = ev.data || {};
  try {
    if (m.type === "init") kernelsReady.then(() => { try { init(m); } catch (e) { fail(e); } });
    else if (m.type === "run") { if (net && !running) { running = true; again(); } }
    else if (m.type === "pause") { running = false; if (net) tick(true); }
    else if (m.type === "write") { if (net) { job = newJob(m); if (!running) again(); } }
    else if (m.type === "stop_write") job = null;
  } catch (e) { fail(e); }
};
function fail(e) { running = false; postMessage({ type: "error", message: String((e && e.message) || e) }); }

// ------------------------------------------------------------------ setting up
function init(m) {
  running = false; job = null;
  const design = FPDesign.normalise(m.design);          // exactly what the app's trainer accepts
  const bytes = new TextEncoder().encode(String(m.text || ""));
  T = design.context; B = m.batch || 8; lrMax = m.lr || 3e-3;
  if (bytes.length < Math.max(400, 3 * (T + 2))) throw new Error("That is too little text to learn from. Give it at least " + Math.max(400, 3 * (T + 2)) + " letters; a few pages is better.");

  // the tokens are simply the different bytes that turn up in the text (for English: its letters and punctuation)
  const seen = new Uint8Array(256);
  for (let i = 0; i < bytes.length; i++) seen[bytes[i]] = 1;
  vocab = []; idOf = new Int16Array(256).fill(-1);
  for (let b = 0; b < 256; b++) if (seen[b]) { idOf[b] = vocab.length; vocab.push(b); }
  toks = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) toks[i] = idOf[bytes[i]];
  // keep the last twentieth aside: it is never studied, only used to check real learning
  const nVal = bytes.length >= 40 * (T + 2) ? Math.max(4 * (T + 2), Math.floor(bytes.length / 20)) : 0;
  nTrain = bytes.length - nVal;

  rand = FPNet.rng32(m.seed == null ? 7 : m.seed);
  sampleRand = FPNet.rng32(99);
  net = new FPNet.Net(FPDesign.expand(design), vocab.length, { batch: B, seed: m.seed == null ? 20261110 : m.seed + 1, wasm: kernels });
  idx = new Int32Array(B * T); tgt = new Int32Array(B * T);
  const fixed = FPNet.rng32(4242), pick = (lo, hi) => {
    const a = new Int32Array(B * T), b = new Int32Array(B * T);
    for (let r = 0; r < B; r++) fillRow(a, b, r, lo + Math.floor(fixed() * (hi - lo - T - 1)));
    return [a, b];
  };
  evalSets = { train: [0, 1, 2, 3].map(() => pick(0, nTrain)), val: nVal ? [0, 1, 2, 3].map(() => pick(nTrain, bytes.length)) : null };
  base = yardsticks(nVal);
  step = 0; seconds = 0; ema = 0; lastLoss = null; speed = 0;
  nextTick = 0; nextSample = 0; nextEval = 0; tickStep = 0; tickSeconds = 0;
  postMessage({ type: "ready", params: net.numParams(), vocab: vocab.length, letters: bytes.length, backend: net.backend, base, batch: B, context: T });
  postMessage({ type: "sample", step: 0, seconds: 0, text: write("", 210, 0.8) });
  evaluate();
}
function fillRow(a, b, r, o) {
  for (let t = 0; t < T; t++) { a[r * T + t] = toks[o + t]; b[r * T + t] = toks[o + t + 1]; }
}

/* How well could you guess with no brain at all? Three yardsticks, measured on this very text:
   pure chance, knowing which letters are common, and knowing which letter tends to follow which.
   The stage shown on the page comes from where the network's score sits between them. */
function yardsticks(nVal) {
  const V = vocab.length, hi = toks.length, lo = nVal ? nTrain : Math.floor(hi * 0.9), nFit = lo;
  const uni = new Float64Array(V), ctx = new Float64Array(V), bi = new Float64Array(V * V), kinds = new Float64Array(V);
  for (let i = 0; i < nFit; i++) uni[toks[i]]++;
  for (let i = 1; i < nFit; i++) { const a = toks[i - 1]; ctx[a]++; if (bi[a * V + toks[i]]++ === 0) kinds[a]++; }
  let h1 = 0, h2 = 0, n = 0;
  for (let i = lo + 1; i < hi; i++) {
    const a = toks[i - 1], b = toks[i];
    const p1 = (uni[b] + 0.5) / (nFit + 0.5 * V);
    const lam = ctx[a] / Math.max(1, ctx[a] + kinds[a]);          // Witten-Bell smoothing
    const p2 = lam * bi[a * V + b] / Math.max(1, ctx[a]) + (1 - lam) * p1;
    h1 -= Math.log(p1); h2 -= Math.log(Math.max(p2, 1e-12)); n++;
  }
  const unigram = h1 / Math.max(1, n);
  return { uniform: Math.log(V), unigram, bigram: Math.min(unigram, h2 / Math.max(1, n)) };
}

// ------------------------------------------------------------------ learning
function learningRate() {
  return lrMax * Math.min(1, (step + 1) / 40) * Math.sqrt(800 / (800 + step));   // gentle start, then slowly smaller steps
}
function trainStep() {
  for (let r = 0; r < B; r++) fillRow(idx, tgt, r, Math.floor(rand() * (nTrain - T - 1)));
  const loss = net.forward(idx, B, T, tgt);
  if (!isFinite(loss)) throw new Error("The numbers blew up. Press Start again; if it keeps happening, use fewer or smaller blocks.");
  net.backward();
  net.step(learningRate(), 0.01, 1.0);
  step++;
  ema = 0.95 * ema + 0.05 * loss;
  lastLoss = ema / (1 - Math.pow(0.95, step));
}
function evaluate() {
  const mean = sets => { let s = 0; for (const [a, b] of sets) s += net.forward(a, B, T, b); return s / sets.length; };
  const train = mean(evalSets.train), val = evalSets.val ? mean(evalSets.val) : null;
  postMessage({ type: "eval", step, seconds, train, val });
}
function tick(force) {
  if (seconds > tickSeconds) speed = (step - tickStep) / (seconds - tickSeconds);
  tickStep = step; tickSeconds = seconds;
  postMessage({ type: "tick", step, seconds, loss: lastLoss, speed, seen: step * B * T, running: running && !force });
}

function loop() {
  if (!net) return;
  try {
    if (job) writeSome();
    if (running) {
      const t0 = performance.now(), until = t0 + SLICE_MS;
      do { trainStep(); } while (performance.now() < until);
      seconds += (performance.now() - t0) / 1000;
      if (seconds >= nextTick) { tick(false); nextTick = seconds + 0.25; }
      if (seconds >= nextEval) { evaluate(); nextEval = seconds + (seconds < 20 ? 2 : seconds < 120 ? 4 : 8); }
      if (seconds >= nextSample && !job) {
        postMessage({ type: "sample", step, seconds, text: write("", 210, 0.8) });
        nextSample = seconds + (seconds < 12 ? 2 : seconds < 60 ? 4 : 8);
      }
    }
  } catch (e) { fail(e); return; }
  if (running || job) again();
}

// ------------------------------------------------------------------ writing
const decoder = new TextDecoder("utf-8");
function encode(text) {
  const out = [];
  for (const b of new TextEncoder().encode(text)) if (idOf[b] >= 0) out.push(idOf[b]);
  return out;
}
function startTokens(prompt) {
  const ctx = encode(prompt);
  if (!ctx.length) ctx.push(idOf[10] >= 0 ? idOf[10] : idOf[32] >= 0 ? idOf[32] : 0);   // start on a new line
  return ctx;
}
function write(prompt, n, temperature) {
  const ctx = startTokens(prompt), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const tok = FPNet.sample(net.predict(ctx.length > T ? ctx.slice(-T) : ctx), temperature, 40, sampleRand);
    ctx.push(tok); out[i] = vocab[tok];
    if (ctx.length > 4 * T) ctx.splice(0, ctx.length - T);
  }
  return decoder.decode(out);
}
function newJob(m) {
  return { id: m.id, ctx: startTokens(String(m.prompt || "")), left: Math.max(1, Math.min(2000, m.n || 200)),
           temperature: m.temperature == null ? 0.8 : m.temperature, stream: new TextDecoder("utf-8") };
}
function writeSome() {
  const j = job, n = Math.min(j.left, running ? 6 : 24), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const tok = FPNet.sample(net.predict(j.ctx.length > T ? j.ctx.slice(-T) : j.ctx), j.temperature, 40, sampleRand);
    j.ctx.push(tok); out[i] = vocab[tok];
    if (j.ctx.length > 4 * T) j.ctx.splice(0, j.ctx.length - T);
  }
  j.left -= n;
  const done = j.left <= 0;
  postMessage({ type: "wrote", id: j.id, text: j.stream.decode(out, { stream: !done }), done });
  if (done && job === j) job = null;
}
