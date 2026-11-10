/* Foolproof Trainer - the two inner loops of the in-browser transformer, in C.

   Almost all the time in training goes on multiplying tables of numbers. A browser's
   JavaScript does that one number at a time; WebAssembly can do four at a time (SIMD).
   This file is only those loops. Everything else stays in nn.js, which also contains a
   plain JavaScript version of the same loops for browsers without WebAssembly SIMD.
   tests/web/test_nn.js checks the two versions against each other.

   Build (see tools/build_kernels.sh):
     clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding ... -o docs/kernels.wasm docs/kernels.c */
#include <wasm_simd128.h>

#define EXPORT(name) __attribute__((export_name(#name)))
typedef v128_t v4;
static inline v4 ld(const float *p) { return wasm_v128_load(p); }
static inline v4 madd(v4 acc, v4 a, v4 b) { return wasm_f32x4_add(acc, wasm_f32x4_mul(a, b)); }
static inline float hsum(v4 v) {
  return (wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)) +
         (wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3));
}

/* Y[N,O] = X[N,I] . W[O,I]^T      four output rows at a time, so each x is loaded once */
EXPORT(linear) void linear(const float *X, const float *W, float *Y, int N, int I, int O) {
  const int I4 = I & ~3, O4 = O & ~3;
  for (int n = 0; n < N; n++) {
    const float *x = X + n * I;
    float *y = Y + n * O;
    int o = 0;
    for (; o < O4; o += 4) {
      const float *w0 = W + o * I, *w1 = w0 + I, *w2 = w1 + I, *w3 = w2 + I;
      v4 a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
      int i = 0;
      for (; i < I4; i += 4) {
        const v4 xv = ld(x + i);
        a0 = madd(a0, xv, ld(w0 + i)); a1 = madd(a1, xv, ld(w1 + i));
        a2 = madd(a2, xv, ld(w2 + i)); a3 = madd(a3, xv, ld(w3 + i));
      }
      float s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
      for (; i < I; i++) { const float xv = x[i]; s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i]; }
      y[o] = s0; y[o + 1] = s1; y[o + 2] = s2; y[o + 3] = s3;
    }
    for (; o < O; o++) {
      const float *w = W + o * I;
      v4 a = wasm_f32x4_splat(0);
      int i = 0;
      for (; i < I4; i += 4) a = madd(a, ld(x + i), ld(w + i));
      float s = hsum(a);
      for (; i < I; i++) s += x[i] * w[i];
      y[o] = s;
    }
  }
}

/* dst[0..L) += g0*r0 + g1*r1 + g2*r2 + g3*r3 */
static inline void add4(float *dst, const float *r0, const float *r1, const float *r2, const float *r3,
                        float g0, float g1, float g2, float g3, int L) {
  const int L4 = L & ~3;
  const v4 v0 = wasm_f32x4_splat(g0), v1 = wasm_f32x4_splat(g1), v2 = wasm_f32x4_splat(g2), v3 = wasm_f32x4_splat(g3);
  int i = 0;
  for (; i < L4; i += 4) {
    v4 d = ld(dst + i);
    d = madd(d, v0, ld(r0 + i)); d = madd(d, v1, ld(r1 + i));
    d = madd(d, v2, ld(r2 + i)); d = madd(d, v3, ld(r3 + i));
    wasm_v128_store(dst + i, d);
  }
  for (; i < L; i++) dst[i] += g0 * r0[i] + g1 * r1[i] + g2 * r2[i] + g3 * r3[i];
}
static inline void add1(float *dst, const float *r, float g, int L) {
  const int L4 = L & ~3;
  const v4 v = wasm_f32x4_splat(g);
  int i = 0;
  for (; i < L4; i += 4) wasm_v128_store(dst + i, madd(ld(dst + i), v, ld(r + i)));
  for (; i < L; i++) dst[i] += g * r[i];
}

/* The way back through the same multiplication:  dX += dY . W   and   dW += dY^T . X */
EXPORT(linear_back) void linear_back(const float *X, const float *W, const float *dY, float *dX, float *dW, int N, int I, int O) {
  const int O4 = O & ~3, N4 = N & ~3;
  for (int n = 0; n < N; n++) {
    const float *g = dY + n * O;
    float *dx = dX + n * I;
    int o = 0;
    for (; o < O4; o += 4) {
      const float *w = W + o * I;
      add4(dx, w, w + I, w + 2 * I, w + 3 * I, g[o], g[o + 1], g[o + 2], g[o + 3], I);
    }
    for (; o < O; o++) add1(dx, W + o * I, g[o], I);
  }
  for (int o = 0; o < O; o++) {
    float *dw = dW + o * I;
    int n = 0;
    for (; n < N4; n += 4) {
      const float *x = X + n * I, *g = dY + n * O + o;
      add4(dw, x, x + I, x + 2 * I, x + 3 * I, g[0], g[O], g[2 * O], g[3 * O], I);
    }
    for (; n < N; n++) add1(dw, X + n * I, dY[n * O + o], I);
  }
}

/* One AdamW nudge over a whole table. hyper-parameters arrive ready-made from nn.js. */
EXPORT(adamw) void adamw(float *w, const float *g, float *m, float *v, int n,
                         float k, float keep, float a, float b1, float b2, float inv_c2) {
  const int n4 = n & ~3;
  const v4 vk = wasm_f32x4_splat(k), vkeep = wasm_f32x4_splat(keep), va = wasm_f32x4_splat(a);
  const v4 vb1 = wasm_f32x4_splat(b1), vb2 = wasm_f32x4_splat(b2), vn1 = wasm_f32x4_splat(1 - b1), vn2 = wasm_f32x4_splat(1 - b2);
  const v4 vc2 = wasm_f32x4_splat(inv_c2), veps = wasm_f32x4_splat(1e-8f);
  int i = 0;
  for (; i < n4; i += 4) {
    const v4 gi = wasm_f32x4_mul(ld(g + i), vk);
    const v4 mi = wasm_f32x4_add(wasm_f32x4_mul(ld(m + i), vb1), wasm_f32x4_mul(gi, vn1));
    const v4 vi = wasm_f32x4_add(wasm_f32x4_mul(ld(v + i), vb2), wasm_f32x4_mul(wasm_f32x4_mul(gi, gi), vn2));
    wasm_v128_store(m + i, mi);
    wasm_v128_store(v + i, vi);
    const v4 den = wasm_f32x4_add(wasm_f32x4_sqrt(wasm_f32x4_mul(vi, vc2)), veps);
    wasm_v128_store(w + i, wasm_f32x4_sub(wasm_f32x4_mul(ld(w + i), vkeep), wasm_f32x4_div(wasm_f32x4_mul(va, mi), den)));
  }
  for (; i < n; i++) {
    const float gi = g[i] * k;
    const float mi = m[i] = b1 * m[i] + (1 - b1) * gi;
    const float vi = v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    w[i] = w[i] * keep - a * mi / (__builtin_sqrtf(vi * inv_c2) + 1e-8f);
  }
}

/* Sum of squares, for measuring the size of the gradient. */
EXPORT(sumsq) float sumsq(const float *g, int n) {
  const int n4 = n & ~3;
  v4 a = wasm_f32x4_splat(0);
  int i = 0;
  for (; i < n4; i += 4) { const v4 x = ld(g + i); a = madd(a, x, x); }
  float s = hsum(a);
  for (; i < n; i++) s += g[i] * g[i];
  return s;
}

/* ------------------------------------------------------------------ e to the power x, four at a time
   The usual recipe (Cephes): split x into a whole number of ln 2 plus a small rest, use a
   short polynomial for the rest, then put the power of two back by writing the exponent bits. */
static inline v4 exp4(v4 x) {
  x = wasm_f32x4_max(x, wasm_f32x4_splat(-87.3f));
  x = wasm_f32x4_min(x, wasm_f32x4_splat(88.3f));
  const v4 kf = wasm_f32x4_nearest(wasm_f32x4_mul(x, wasm_f32x4_splat(1.44269504088896341f)));
  v4 r = wasm_f32x4_sub(x, wasm_f32x4_mul(kf, wasm_f32x4_splat(0.693359375f)));
  r = wasm_f32x4_sub(r, wasm_f32x4_mul(kf, wasm_f32x4_splat(-2.12194440e-4f)));
  v4 p = wasm_f32x4_splat(1.9875691500e-4f);
  p = madd(wasm_f32x4_splat(1.3981999507e-3f), p, r);
  p = madd(wasm_f32x4_splat(8.3334519073e-3f), p, r);
  p = madd(wasm_f32x4_splat(4.1665795894e-2f), p, r);
  p = madd(wasm_f32x4_splat(1.6666665459e-1f), p, r);
  p = madd(wasm_f32x4_splat(5.0000001201e-1f), p, r);
  const v4 y = wasm_f32x4_add(madd(r, p, wasm_f32x4_mul(r, r)), wasm_f32x4_splat(1.0f));
  const v128_t pow2 = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(kf), wasm_i32x4_splat(127)), 23);
  return wasm_f32x4_mul(y, pow2);
}
static inline float dot(const float *a, const float *b, int L) {
  const int L4 = L & ~3;
  v4 acc = wasm_f32x4_splat(0);
  int i = 0;
  for (; i < L4; i += 4) acc = madd(acc, ld(a + i), ld(b + i));
  float s = hsum(acc);
  for (; i < L; i++) s += a[i] * b[i];
  return s;
}
/* tmp[0..n) = exp(tmp - mx); returns the sum. tmp must have room for 3 spare numbers. */
static inline float exp_rows(float *tmp, int n, float mx) {
  const v4 vm = wasm_f32x4_splat(mx);
  tmp[n] = tmp[n + 1] = tmp[n + 2] = -1e30f;
  for (int i = 0; i < n; i += 4) wasm_v128_store(tmp + i, exp4(wasm_f32x4_sub(ld(tmp + i), vm)));
  float sum = 0;
  for (int i = 0; i < n; i++) sum += tmp[i];
  return sum;
}

/* Mark positions by turning the query and key halves of q[N,3C] (sign -1 turns them back). */
EXPORT(rotate) void rotate(float *buf, const float *cs, const float *sn, int B, int T, int C, int half, float sign) {
  const int C3 = 3 * C, pairs = C >> 1;
  for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
    float *base = buf + (b * T + t) * C3;
    const float *c = cs + t * half, *s = sn + t * half;
    for (int part = 0; part < 2; part++) {
      float *p = base + part * C;
      for (int k = 0; k < pairs; k++) {
        const int j = k % half;
        const float cc = c[j], ss = sign * s[j], a = p[2 * k], bb = p[2 * k + 1];
        p[2 * k] = a * cc - bb * ss;
        p[2 * k + 1] = a * ss + bb * cc;
      }
    }
  }
}

/* Attention, the middle part: scores between each letter and the earlier ones, softmax,
   then the weighted sum of values.  q[N,3C] holds queries | keys | values per letter. */
EXPORT(attention) void attention(const float *q, float *att, float *o, float *tmp, int B, int T, int C, int H) {
  const int D = C / H, C3 = 3 * C;
  const float scale = 1.0f / __builtin_sqrtf((float)D);
  for (int b = 0; b < B; b++) for (int hd = 0; hd < H; hd++) {
    const int ho = hd * D, row = b * T;
    for (int t = 0; t < T; t++) {
      const float *qt = q + (row + t) * C3 + ho;
      float mx = -1e30f;
      for (int u = 0; u <= t; u++) {
        const float s = dot(qt, q + (row + u) * C3 + C + ho, D) * scale;
        tmp[u] = s;
        if (s > mx) mx = s;
      }
      const float inv = 1.0f / exp_rows(tmp, t + 1, mx);
      float *a = att + ((b * H + hd) * T + t) * T, *ot = o + (row + t) * C + ho;
      for (int d = 0; d < D; d++) ot[d] = 0;
      for (int u = 0; u <= t; u++) {
        const float w = tmp[u] * inv;
        a[u] = w;
        add1(ot, q + (row + u) * C3 + 2 * C + ho, w, D);
      }
    }
  }
}

EXPORT(attention_back) void attention_back(const float *q, const float *att, const float *dO, float *dq, float *tmp, int B, int T, int C, int H) {
  const int D = C / H, C3 = 3 * C;
  const float scale = 1.0f / __builtin_sqrtf((float)D);
  for (int b = 0; b < B; b++) for (int hd = 0; hd < H; hd++) {
    const int ho = hd * D, row = b * T;
    for (int t = 0; t < T; t++) {
      const float *qt = q + (row + t) * C3 + ho, *a = att + ((b * H + hd) * T + t) * T, *dot_ = dO + (row + t) * C + ho;
      float *dqt = dq + (row + t) * C3 + ho;
      float mean = 0;
      for (int u = 0; u <= t; u++) {
        const float da = dot(dot_, q + (row + u) * C3 + 2 * C + ho, D);
        tmp[u] = da;
        mean += a[u] * da;
      }
      for (int u = 0; u <= t; u++) {
        const float w = a[u], ds = w * (tmp[u] - mean) * scale;
        const float *ku = q + (row + u) * C3 + C + ho;
        float *dku = dq + (row + u) * C3 + C + ho;
        add1(dku + C, dot_, w, D);        /* values */
        add1(dqt, ku, ds, D);             /* query of this letter */
        add1(dku, qt, ds, D);             /* key of the earlier letter */
      }
    }
  }
}

/* GELU (tanh form) in place, and its slope for the way back. */
EXPORT(gelu) void gelu(float *act, float *dact, int n) {
  const v4 k0 = wasm_f32x4_splat(0.7978845608028654f), k1 = wasm_f32x4_splat(0.044715f), k3 = wasm_f32x4_splat(3 * 0.044715f);
  const v4 half = wasm_f32x4_splat(0.5f), one = wasm_f32x4_splat(1.0f), two = wasm_f32x4_splat(2.0f);
  const v4 lo = wasm_f32x4_splat(-9.0f), hi = wasm_f32x4_splat(9.0f);
  float in[4], out[4], dout[4];
  for (int i = 0; i < n; i += 4) {
    const int left = n - i;
    v4 v;
    if (left >= 4) v = ld(act + i);
    else { for (int j = 0; j < 4; j++) in[j] = j < left ? act[i + j] : 0; v = ld(in); }
    const v4 v2 = wasm_f32x4_mul(v, v);
    v4 z = wasm_f32x4_mul(k0, madd(v, k1, wasm_f32x4_mul(v2, v)));
    z = wasm_f32x4_min(wasm_f32x4_max(z, lo), hi);
    const v4 e = exp4(wasm_f32x4_mul(two, z));
    const v4 t = wasm_f32x4_div(wasm_f32x4_sub(e, one), wasm_f32x4_add(e, one));
    const v4 h1 = wasm_f32x4_mul(half, wasm_f32x4_add(one, t));
    const v4 y = wasm_f32x4_mul(v, h1);
    const v4 slope = madd(h1, wasm_f32x4_mul(wasm_f32x4_mul(half, v), wasm_f32x4_sub(one, wasm_f32x4_mul(t, t))),
                          wasm_f32x4_mul(k0, madd(one, k3, v2)));
    if (left >= 4) { wasm_v128_store(act + i, y); wasm_v128_store(dact + i, slope); }
    else { wasm_v128_store(out, y); wasm_v128_store(dout, slope); for (int j = 0; j < left; j++) { act[i + j] = out[j]; dact[i + j] = dout[j]; } }
  }
}

/* Softmax over each row of scores, already divided by the number of rows (ready to be the gradient). */
EXPORT(softmax_rows) void softmax_rows(const float *lg, float *dl, float *tmp, int N, int V, float inv) {
  for (int n = 0; n < N; n++) {
    const float *row = lg + n * V;
    float *out = dl + n * V;
    float mx = -1e30f;
    for (int v = 0; v < V; v++) { tmp[v] = row[v]; if (row[v] > mx) mx = row[v]; }
    const float k = inv / exp_rows(tmp, V, mx);
    for (int v = 0; v < V; v++) out[v] = tmp[v] * k;
  }
}
