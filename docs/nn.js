/* Foolproof Trainer - a real transformer in plain JavaScript.

   This is the same network the app trains with PyTorch (foolproof/model.py), written out
   by hand so it can learn inside a web page with nothing to install:

     letters -> numbers (embedding table)
     then each block the person stacked:   stream += block(normalise(stream))
        Attention:     queries/keys/values, positions marked by rotation (RoPE), causal softmax
        Feed-forward:  widen, GELU, narrow
     final normalise -> scores for the next letter (the same table as the embedding, tied)

   No bias terms anywhere. The backward pass is written by hand and checked against
   finite differences and against PyTorch in tests/web/. No dependencies.

   The multiplying of big tables is also available four-numbers-at-a-time in
   kernels.wasm (built from kernels.c). Give Net a compiled module and it uses that;
   without one, the plain JavaScript loops below do the same job, only slower. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.FPNet = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  const EPS = 1e-6;                      // inside RMSNorm, as in the app
  const K0 = Math.sqrt(2 / Math.PI), K1 = 0.044715;   // GELU, tanh form

  function rng32(seed) {                 // small seeded generator, so runs can be repeated
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ------------------------------------------------------------------ the three kernels
  /* Y[N,O] = X[N,I] . W[O,I]^T   (four running sums so the processor can overlap the adds) */
  function linear(X, W, Y, N, I, O) {
    const I4 = I & ~3;
    for (let n = 0; n < N; n++) {
      const xo = n * I, yo = n * O;
      for (let o = 0; o < O; o++) {
        const wo = o * I;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
        for (; i < I4; i += 4) {
          s0 += X[xo + i] * W[wo + i];
          s1 += X[xo + i + 1] * W[wo + i + 1];
          s2 += X[xo + i + 2] * W[wo + i + 2];
          s3 += X[xo + i + 3] * W[wo + i + 3];
        }
        for (; i < I; i++) s0 += X[xo + i] * W[wo + i];
        Y[yo + o] = s0 + s1 + s2 + s3;
      }
    }
  }
  /* Given dY, add into dW and dX. */
  function linearBack(X, W, dY, dX, dW, N, I, O) {
    for (let n = 0; n < N; n++) {
      const xo = n * I, yo = n * O;
      for (let o = 0; o < O; o++) {
        const g = dY[yo + o];
        if (g === 0) continue;
        const wo = o * I;
        for (let i = 0; i < I; i++) {
          dW[wo + i] += g * X[xo + i];
          dX[xo + i] += g * W[wo + i];
        }
      }
    }
  }
  /* RMSNorm: nrm = x / rms(x), h = nrm * gain.  r keeps 1/rms for the way back. */
  function rmsForward(x, gain, nrm, h, r, N, C) {
    for (let n = 0; n < N; n++) {
      const o = n * C;
      let ss = 0;
      for (let c = 0; c < C; c++) { const v = x[o + c]; ss += v * v; }
      const rr = 1 / Math.sqrt(ss / C + EPS);
      r[n] = rr;
      for (let c = 0; c < C; c++) { const v = x[o + c] * rr; nrm[o + c] = v; h[o + c] = v * gain[c]; }
    }
  }
  function rmsBackward(dh, gain, nrm, r, dGain, dx, N, C) {
    for (let n = 0; n < N; n++) {
      const o = n * C;
      let dot = 0;
      for (let c = 0; c < C; c++) { const d = dh[o + c], v = nrm[o + c]; dGain[c] += d * v; dot += d * gain[c] * v; }
      const mean = dot / C, rr = r[n];
      for (let c = 0; c < C; c++) dx[o + c] += rr * (dh[o + c] * gain[c] - nrm[o + c] * mean);
    }
  }

  /* Rotary positions: the same tables as model.py, worked out in single precision like PyTorch. */
  function ropeTables(T, D, cos, sin) {
    const half = D >> 1, f = Math.fround;
    for (let j = 0; j < half; j++) {
      const inv = f(1 / f(Math.pow(10000, f((2 * j) / D))));
      for (let t = 0; t < T; t++) {
        const a = f(t * inv);
        cos[t * half + j] = f(Math.cos(a));
        sin[t * half + j] = f(Math.sin(a));
      }
    }
  }
  /* Turn the query and key parts of a [N, 3C] buffer by their position. sign -1 turns back. */
  function rotate(buf, B, T, C, half, cos, sin, sign) {
    const C3 = 3 * C, pairs = C >> 1;
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
      const base = (b * T + t) * C3, ro = t * half;
      for (let part = 0; part < 2; part++) {
        const po = base + part * C;
        for (let p = 0; p < pairs; p++) {
          const j = p % half, c = cos[ro + j], s = sign * sin[ro + j];
          const i = po + 2 * p, a = buf[i], bb = buf[i + 1];
          buf[i] = a * c - bb * s;
          buf[i + 1] = a * s + bb * c;
        }
      }
    }
  }

  const JS = { linear, linearBack, adamw: null, sumsq: null, attention: null, attentionBack: null, rotate: null, gelu: null, softmaxRows: null };
  /* The same loops from kernels.wasm. Every table lives in that module's memory, so a
     table is handed over as its address (byteOffset). */
  function wasmKernels(ex) {
    return {
      linear: (X, W, Y, N, I, O) => ex.linear(X.byteOffset, W.byteOffset, Y.byteOffset, N, I, O),
      linearBack: (X, W, dY, dX, dW, N, I, O) => ex.linear_back(X.byteOffset, W.byteOffset, dY.byteOffset, dX.byteOffset, dW.byteOffset, N, I, O),
      adamw: (w, g, m, v, n, k, keep, a, b1, b2, invC2) => ex.adamw(w.byteOffset, g.byteOffset, m.byteOffset, v.byteOffset, n, k, keep, a, b1, b2, invC2),
      sumsq: (g, n) => ex.sumsq(g.byteOffset, n),
      rotate: (buf, cos, sin, B, T, C, half, sign) => ex.rotate(buf.byteOffset, cos.byteOffset, sin.byteOffset, B, T, C, half, sign),
      attention: (q, att, o, tmp, B, T, C, H) => ex.attention(q.byteOffset, att.byteOffset, o.byteOffset, tmp.byteOffset, B, T, C, H),
      attentionBack: (q, att, dO, dq, tmp, B, T, C, H) => ex.attention_back(q.byteOffset, att.byteOffset, dO.byteOffset, dq.byteOffset, tmp.byteOffset, B, T, C, H),
      gelu: (act, dact, n) => ex.gelu(act.byteOffset, dact.byteOffset, n),
      softmaxRows: (lg, dl, tmp, N, V, inv) => ex.softmax_rows(lg.byteOffset, dl.byteOffset, tmp.byteOffset, N, V, inv),
    };
  }
  /* bytes of kernels.wasm -> a module to hand to Net, or null if this browser cannot run it. */
  function compileKernels(bytes) {
    try {
      if (typeof WebAssembly !== "object" || !WebAssembly.validate(bytes)) return Promise.resolve(null);
      return WebAssembly.compile(bytes).catch(() => null);
    } catch (e) { return Promise.resolve(null); }
  }

  // ------------------------------------------------------------------ the network
  /* design: the steps from FPDesign.expand(design):
       { width, context, layers, blocks: [ { type: "attention", heads } | { type: "feedforward", mult } ] }
     (each step may carry the PyTorch names of its tables: norm, a, b).
     vocab: how many different tokens.
     opts.batch: the most sequences it will ever be given at once (memory is set aside up front).
     opts.wasm:  a compiled kernels.wasm module, or nothing for plain JavaScript. */
  function Net(design, vocab, opts) {
    opts = opts || {};
    const F = this.F = opts.float64 ? Float64Array : Float32Array;
    const C = this.C = design.width, V = this.V = vocab, T = this.T = design.context;
    const B = this.maxB = opts.batch || 8, N = B * T;
    this.design = design;
    // what each block adds to the stream starts small, so tall stacks stay stable (as in model.py)
    const outStd = 0.02 / Math.sqrt(2 * Math.max(1, design.layers || Math.ceil(design.blocks.length / 2)));

    const make = alloc => {
      this.params = [];
      const P = (name, rows, cols, std) => {
        const n = rows * cols;
        const p = { name, rows, cols, std, w: alloc(n), g: alloc(n), m: alloc(n), v: alloc(n), decay: std !== null };
        this.params.push(p);
        return p;
      };
      this.wte = P("wte.weight", V, C, 0.02);
      let maxM = C;
      this.blocks = design.blocks.map((b, i) => {
        const name = (given, fallback) => given || "steps." + i + "." + fallback;
        const blk = { type: b.type, norm: P(name(b.norm, "norm.weight"), 1, C, null), n: alloc(N * C), h: alloc(N * C), r: alloc(N) };
        if (b.type === "attention") {
          blk.H = b.heads; blk.D = C / b.heads;
          blk.qkv = P(name(b.a, "qkv.weight"), 3 * C, C, 0.02);
          blk.proj = P(name(b.b, "proj.weight"), C, C, outStd);
          blk.q = alloc(N * 3 * C); blk.att = alloc(B * blk.H * T * T); blk.o = alloc(N * C);
          blk.half = blk.D >> 1; blk.cos = alloc(T * blk.half); blk.sin = alloc(T * blk.half);
        } else {
          blk.M = b.mult * C; if (blk.M > maxM) maxM = blk.M;
          blk.fc = P(name(b.a, "fc.weight"), blk.M, C, 0.02);
          blk.proj = P(name(b.b, "proj.weight"), C, blk.M, outStd);
          blk.act = alloc(N * blk.M); blk.dact = alloc(N * blk.M);
        }
        return blk;
      });
      this.normf = P("norm_f.weight", 1, C, null);
      this.x = alloc(N * C); this.br = alloc(N * C); this.dx = alloc(N * C); this.dh = alloc(N * C);
      this.dO = alloc(N * C); this.dq = alloc(N * 3 * C); this.dAct = alloc(N * maxM);
      this.fn = alloc(N * C); this.fh = alloc(N * C); this.fr = alloc(N);
      this.logits = alloc(N * V); this.dlog = alloc(N * V);
      this.tmp = alloc(Math.max(T, V) + 8);
    };

    this.backend = "js"; this.k = JS;
    if (opts.wasm && !opts.float64) {
      try {
        let bytes = 0;
        make(n => { bytes += (n * 4 + 15) & ~15; return null; });
        const memory = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) + 4 });
        const ex = new WebAssembly.Instance(opts.wasm, { env: { memory } }).exports;
        let ptr = (ex.__heap_base.value + 15) & ~15;
        const short = ptr + bytes - memory.buffer.byteLength;
        if (short > 0) memory.grow(Math.ceil(short / 65536));
        const buf = memory.buffer;      // taken after any growing: growing would detach earlier views
        make(n => { const a = new Float32Array(buf, ptr, n); ptr += (n * 4 + 15) & ~15; return a; });
        this.k = wasmKernels(ex); this.backend = "wasm";
      } catch (e) { this.backend = "js"; this.k = JS; make(n => new F(n)); }
    } else make(n => new F(n));

    for (const blk of this.blocks) if (blk.type === "attention") ropeTables(T, blk.D, blk.cos, blk.sin);
    this.sc = new Float64Array(T + 1); this.datt = new Float64Array(T + 1);
    const rand = rng32(opts.seed == null ? 20261110 : opts.seed), scale = opts.initScale || 1;
    const gauss = () => { let u = 0; while (u === 0) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
    for (const p of this.params) {
      if (p.std === null) p.w.fill(1);
      else for (let i = 0; i < p.w.length; i++) p.w[i] = gauss() * p.std * scale;
    }
    this.steps = 0;
  }

  Net.prototype.numParams = function () { let n = 0; for (const p of this.params) n += p.w.length; return n; };

  /* idx: B*T token ids, row after row. With targets (same shape) returns the loss and
     keeps what the backward pass needs. Without, it only runs the stack. */
  Net.prototype.forward = function (idx, B, T, targets) {
    if (T > this.T || B > this.maxB) throw new Error("more text at once than this network was set up for");
    const N = B * T, C = this.C, V = this.V, x = this.x, wte = this.wte.w, br = this.br;
    this.B = B; this.Tc = T; this.idx = idx;
    for (let n = 0; n < N; n++) {
      const src = idx[n] * C, dst = n * C;
      for (let c = 0; c < C; c++) x[dst + c] = wte[src + c];
    }
    for (const blk of this.blocks) {
      rmsForward(x, blk.norm.w, blk.n, blk.h, blk.r, N, C);
      if (blk.type === "attention") this._attention(blk, B, T); else this._feed(blk, N);
      for (let i = 0, e = N * C; i < e; i++) x[i] += br[i];
    }
    rmsForward(x, this.normf.w, this.fn, this.fh, this.fr, N, C);
    if (!targets) return NaN;
    const lg = this.logits, dl = this.dlog, inv = 1 / N;
    this.k.linear(this.fh, wte, lg, N, C, V);
    let loss = 0;
    if (this.k.softmaxRows) {
      this.k.softmaxRows(lg, dl, this.tmp, N, V, inv);
      for (let n = 0; n < N; n++) {
        const o = n * V + targets[n];
        loss -= Math.log(Math.max(dl[o] / inv, 1e-30));
        dl[o] -= inv;
      }
      return loss * inv;
    }
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let mx = -Infinity;
      for (let v = 0; v < V; v++) if (lg[o + v] > mx) mx = lg[o + v];
      let sum = 0;
      for (let v = 0; v < V; v++) { const e = Math.exp(lg[o + v] - mx); dl[o + v] = e; sum += e; }
      const tg = targets[n], k = inv / sum;
      loss -= Math.log(dl[o + tg] / sum);
      for (let v = 0; v < V; v++) dl[o + v] *= k;
      dl[o + tg] -= inv;
    }
    return loss * inv;
  };

  Net.prototype._attention = function (blk, B, T) {
    const C = this.C, H = blk.H, D = blk.D, N = B * T, C3 = 3 * C, C2 = 2 * C;
    const q = blk.q, att = blk.att, o = blk.o, sc = this.sc, scale = 1 / Math.sqrt(D);
    const K = this.k;
    K.linear(blk.h, blk.qkv.w, q, N, C, C3);
    if (K.attention) {
      K.rotate(q, blk.cos, blk.sin, B, T, C, blk.half, 1);
      K.attention(q, att, o, this.tmp, B, T, C, H);
      K.linear(o, blk.proj.w, this.br, N, C, C);
      return;
    }
    rotate(q, B, T, C, blk.half, blk.cos, blk.sin, 1);
    for (let b = 0; b < B; b++) for (let hd = 0; hd < H; hd++) {
      const ho = hd * D, row = b * T;
      for (let t = 0; t < T; t++) {
        const qo = (row + t) * C3 + ho;
        let mx = -Infinity;
        for (let u = 0; u <= t; u++) {           // only earlier letters: it may not peek ahead
          const ko = (row + u) * C3 + C + ho;
          let s = 0;
          for (let d = 0; d < D; d++) s += q[qo + d] * q[ko + d];
          s *= scale; sc[u] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let u = 0; u <= t; u++) { const e = Math.exp(sc[u] - mx); sc[u] = e; sum += e; }
        const ao = ((b * H + hd) * T + t) * T, oo = (row + t) * C + ho;
        for (let d = 0; d < D; d++) o[oo + d] = 0;
        for (let u = 0; u <= t; u++) {
          const a = sc[u] / sum, vo = (row + u) * C3 + C2 + ho;
          att[ao + u] = a;
          for (let d = 0; d < D; d++) o[oo + d] += a * q[vo + d];
        }
      }
    }
    K.linear(o, blk.proj.w, this.br, N, C, C);
  };

  Net.prototype._attentionBack = function (blk, B, T) {
    const C = this.C, H = blk.H, D = blk.D, N = B * T, C3 = 3 * C, C2 = 2 * C;
    const q = blk.q, att = blk.att, dO = this.dO, dq = this.dq, datt = this.datt, scale = 1 / Math.sqrt(D);
    dO.fill(0, 0, N * C);
    this.k.linearBack(blk.o, blk.proj.w, this.dx, dO, blk.proj.g, N, C, C);
    dq.fill(0, 0, N * C3);
    if (this.k.attentionBack) {
      this.k.attentionBack(q, att, dO, dq, this.tmp, B, T, C, H);
      this.k.rotate(dq, blk.cos, blk.sin, B, T, C, blk.half, -1);
      this.k.linearBack(blk.h, blk.qkv.w, dq, this.dh, blk.qkv.g, N, C, C3);
      return;
    }
    for (let b = 0; b < B; b++) for (let hd = 0; hd < H; hd++) {
      const ho = hd * D, row = b * T;
      for (let t = 0; t < T; t++) {
        const qo = (row + t) * C3 + ho, ao = ((b * H + hd) * T + t) * T, oo = (row + t) * C + ho;
        let dot = 0;
        for (let u = 0; u <= t; u++) {
          const vo = (row + u) * C3 + C2 + ho;
          let da = 0;
          for (let d = 0; d < D; d++) da += dO[oo + d] * q[vo + d];
          datt[u] = da;
          dot += att[ao + u] * da;
        }
        for (let u = 0; u <= t; u++) {
          const a = att[ao + u], ds = a * (datt[u] - dot) * scale;
          const ko = (row + u) * C3 + C + ho, vo = ko + C;
          for (let d = 0; d < D; d++) {
            dq[vo + d] += a * dO[oo + d];
            dq[qo + d] += ds * q[ko + d];
            dq[ko + d] += ds * q[qo + d];
          }
        }
      }
    }
    rotate(dq, B, T, C, blk.half, blk.cos, blk.sin, -1);
    this.k.linearBack(blk.h, blk.qkv.w, dq, this.dh, blk.qkv.g, N, C, C3);
  };

  Net.prototype._feed = function (blk, N) {
    const C = this.C, M = blk.M, act = blk.act, dact = blk.dact;
    this.k.linear(blk.h, blk.fc.w, act, N, C, M);
    if (this.k.gelu) this.k.gelu(act, dact, N * M);
    else for (let i = 0, e = N * M; i < e; i++) {
      const v = act[i], t = Math.tanh(K0 * (v + K1 * v * v * v));
      act[i] = 0.5 * v * (1 + t);
      dact[i] = 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * K0 * (1 + 3 * K1 * v * v);
    }
    this.k.linear(act, blk.proj.w, this.br, N, M, C);
  };

  Net.prototype._feedBack = function (blk, N) {
    const C = this.C, M = blk.M, dA = this.dAct, dact = blk.dact;
    dA.fill(0, 0, N * M);
    this.k.linearBack(blk.act, blk.proj.w, this.dx, dA, blk.proj.g, N, M, C);
    for (let i = 0, e = N * M; i < e; i++) dA[i] *= dact[i];
    this.k.linearBack(blk.h, blk.fc.w, dA, this.dh, blk.fc.g, N, C, M);
  };

  /* Work out, for every number in the network, which way to nudge it. Call after forward(…, targets). */
  Net.prototype.backward = function () {
    const B = this.B, T = this.Tc, N = B * T, C = this.C, dx = this.dx, dh = this.dh;
    for (const p of this.params) p.g.fill(0);
    dh.fill(0, 0, N * C);
    this.k.linearBack(this.fh, this.wte.w, this.dlog, dh, this.wte.g, N, C, this.V);
    dx.fill(0, 0, N * C);
    rmsBackward(dh, this.normf.w, this.fn, this.fr, this.normf.g, dx, N, C);
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      const blk = this.blocks[i];
      dh.fill(0, 0, N * C);
      if (blk.type === "attention") this._attentionBack(blk, B, T); else this._feedBack(blk, N);
      rmsBackward(dh, blk.norm.w, blk.n, blk.r, blk.norm.g, dx, N, C);
    }
    const g = this.wte.g, idx = this.idx;
    for (let n = 0; n < N; n++) {
      const dst = idx[n] * C, src = n * C;
      for (let c = 0; c < C; c++) g[dst + c] += dx[src + c];
    }
  };

  /* One AdamW nudge (same rule and constants as the app). Returns the size of the gradient. */
  Net.prototype.step = function (lr, weightDecay, clip) {
    const K = this.k;
    let ss = 0;
    for (const p of this.params) {
      if (K.sumsq) ss += K.sumsq(p.g, p.g.length);
      else { const g = p.g; for (let i = 0; i < g.length; i++) ss += g[i] * g[i]; }
    }
    const norm = Math.sqrt(ss), k = clip && norm > clip ? clip / norm : 1;
    const t = ++this.steps, b1 = 0.9, b2 = 0.95, c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t), a = lr / c1;
    for (const p of this.params) {
      const w = p.w, g = p.g, m = p.m, v = p.v, keep = p.decay ? 1 - lr * (weightDecay || 0) : 1;
      if (K.adamw) { K.adamw(w, g, m, v, w.length, k, keep, a, b1, b2, 1 / c2); continue; }
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * k;
        const mi = m[i] = b1 * m[i] + (1 - b1) * gi;
        const vi = v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        w[i] = w[i] * keep - a * mi / (Math.sqrt(vi / c2) + 1e-8);
      }
    }
    return norm;
  };

  /* Scores for the token that comes after ctx (an array of 1..T token ids). */
  Net.prototype.predict = function (ctx) {
    const T = ctx.length, C = this.C, V = this.V;
    this.forward(ctx, 1, T, null);
    const out = this.out || (this.out = new this.F(V)), fh = this.fh, wte = this.wte.w, xo = (T - 1) * C;
    for (let v = 0; v < V; v++) {
      const wo = v * C;
      let s = 0;
      for (let c = 0; c < C; c++) s += fh[xo + c] * wte[wo + c];
      out[v] = s;
    }
    return out;
  };

  /* Pick the next token from scores: temperature, then keep only the top few. */
  function sample(logits, temperature, topK, rand) {
    const V = logits.length;
    if (!(temperature > 1e-5)) { let best = 0; for (let v = 1; v < V; v++) if (logits[v] > logits[best]) best = v; return best; }
    let floor = -Infinity;
    if (topK && topK < V) {
      const sorted = Array.from(logits).sort((a, b) => b - a);
      floor = sorted[topK - 1];
    }
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    const p = new Float64Array(V);
    let sum = 0;
    for (let v = 0; v < V; v++) if (logits[v] >= floor) { p[v] = Math.exp((logits[v] - mx) / temperature); sum += p[v]; }
    let r = rand() * sum;
    for (let v = 0; v < V; v++) { r -= p[v]; if (r <= 0 && p[v] > 0) return v; }
    for (let v = V - 1; v >= 0; v--) if (p[v] > 0) return v;
    return 0;
  }

  Net.prototype.state = function () {
    const out = {};
    for (const p of this.params) out[p.name] = { rows: p.rows, cols: p.cols, data: Array.from(p.w) };
    return out;
  };
  Net.prototype.load = function (state) {
    for (const p of this.params) {
      const s = state[p.name];
      if (!s || s.data.length !== p.w.length) throw new Error("weights do not fit this design: " + p.name);
      p.w.set(s.data);
    }
  };

  return { Net, sample, rng32, compileKernels, linear, linearBack };
});
