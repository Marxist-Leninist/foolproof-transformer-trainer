/* The learning chart: how wrong its guesses are, over time.
   Two lines on one axis. Built as plain SVG, text set with textContent only. */
"use strict";
(function () {
  const NS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
    for (const k of kids) if (k != null) el.append(k);
    return el;
  }
  function t(tag, attrs, text) { const el = s(tag, attrs); el.textContent = text; return el; }

  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
  }
  function thin(arr, max) {
    if (arr.length <= max) return arr;
    const out = [], step = arr.length / max;
    for (let i = 0; i < max; i++) out.push(arr[Math.floor(i * step)]);
    out.push(arr[arr.length - 1]);
    return out;
  }
  function timeUnit(tmax) {
    if (tmax < 150) return { div: 1, name: "seconds", short: "s" };
    if (tmax < 3 * 3600) return { div: 60, name: "minutes", short: "min" };
    return { div: 3600, name: "hours", short: "h" };
  }
  function fmtWhen(sec) {
    if (sec < 90) return Math.round(sec) + " seconds in";
    if (sec < 5400) return (sec / 60).toFixed(sec < 600 ? 1 : 0) + " minutes in";
    return (sec / 3600).toFixed(1) + " hours in";
  }

  /* host: the .viz element.  m: [{t, loss}]  e: [{t, train, val}] */
  function draw(host, m, e, names) {
    const W = 560, H = 250, L = 40, R = 52, T = 12, B = 30;
    const train = thin(m.filter(p => p.loss != null), 400).map(p => ({ t: p.t, v: p.loss }));
    const val = e.filter(p => p.val != null).map(p => ({ t: p.t, v: p.val }));
    const old = host.querySelector("svg");
    const oldTip = host.querySelector(".tip");
    if (oldTip) oldTip.remove();
    if (train.length < 2) {
      const ph = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The chart will appear after the first few seconds of training." });
      ph.append(t("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "axis-wait", fill: "#5b6680", "font-size": 14 }, "The chart appears after a few seconds of training."));
      if (old) old.replaceWith(ph); else host.append(ph);
      return;
    }
    const tmax = Math.max(train[train.length - 1].t, val.length ? val[val.length - 1].t : 0, 1);
    const unit = timeUnit(tmax);
    // Keep the interesting part readable: the first seconds drop off a cliff, so cap the top
    // of the scale at where things stood 3% of the way in (everything above is clipped).
    const tCut = tmax > 90 ? tmax * 0.03 : 0;
    const visible = train.filter(p => p.t >= tCut).concat(val.filter(p => p.t >= tCut));
    let lo = Math.min(...visible.map(p => p.v)), hi = Math.max(...visible.map(p => p.v));
    if (hi - lo < 0.05) { hi += 0.05; lo -= 0.05; }
    const step = niceStep(hi - lo, 4);
    const pad = (hi - lo) * 0.06;
    lo -= pad; hi += pad;   // the scale hugs the data; gridlines sit on round numbers inside it
    const x = v => L + (v / tmax) * (W - L - R);
    const y = v => T + (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (H - T - B);
    const dec = step < 0.1 ? 2 : step < 1 ? 1 : 0;

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0",
      "aria-label": `Line chart of how wrong its guesses are over time. ${names[0]}: now ${train[train.length - 1].v.toFixed(2)}.` +
        (val.length ? ` ${names[1]}: now ${val[val.length - 1].v.toFixed(2)}.` : "") + " Lower is better." });
    const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      grid.append(s("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }));
      axis.append(t("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(dec)));
    }
    const xs = niceStep(tmax / unit.div, 5);
    for (let v = 0; v <= tmax / unit.div + 1e-9; v += xs) {
      axis.append(t("text", { x: x(v * unit.div), y: H - 9, "text-anchor": "middle" },
        (xs < 1 ? v.toFixed(1) : String(Math.round(v))) + (v === 0 ? "" : " " + unit.short)));
    }
    svg.append(grid, axis);
    const clip = "c" + Math.random().toString(36).slice(2, 8);
    svg.append(s("clipPath", { id: clip }, s("rect", { x: L, y: T - 2, width: W - L - R + 2, height: H - T - B + 4 })));
    const path = pts => pts.map((p, i) => (i ? "L" : "M") + x(p.t).toFixed(1) + " " + y(p.v).toFixed(1)).join("");
    const lines = s("g", { "clip-path": `url(#${clip})` });
    lines.append(s("path", { class: "line s1", d: path(train) }));
    if (val.length > 1) lines.append(s("path", { class: "line s2", d: path(val) }));
    svg.append(lines);
    // end dots + the current value at each line's end
    const ends = [{ p: train[train.length - 1], c: "s1", fill: "var(--series-1)" }];
    if (val.length) ends.push({ p: val[val.length - 1], c: "s2", fill: "var(--series-2)" });
    let ly = ends.map(en => y(en.p.v) + 4);
    if (ends.length === 2 && Math.abs(ly[0] - ly[1]) < 14) {
      const mid = (ly[0] + ly[1]) / 2, up = ly[0] <= ly[1] ? 0 : 1;
      ly[up] = mid - 7; ly[1 - up] = mid + 7;
    }
    ends.forEach((en, i) => {
      svg.append(s("circle", { class: "end", cx: x(en.p.t), cy: y(en.p.v), r: 4.5, fill: en.fill }));
      svg.append(t("text", { class: "endlabel", x: x(en.p.t) + 9, y: ly[i] }, en.p.v.toFixed(2)));
    });

    // hover / keyboard: a crosshair that snaps to the nearest moment, one tooltip for both lines
    const cross = s("line", { class: "cross", y1: T, y2: H - B, visibility: "hidden" });
    svg.append(cross);
    const tip = document.createElement("div");
    tip.className = "tip hidden";
    const stops = (val.length ? val : train).map(p => p.t);
    const nearest = (arr, tt) => arr.reduce((b, p) => (Math.abs(p.t - tt) < Math.abs(b.t - tt) ? p : b), arr[0]);
    let cur = -1;
    function show(i) {
      cur = Math.max(0, Math.min(stops.length - 1, i));
      const tt = stops[cur];
      cross.setAttribute("x1", x(tt)); cross.setAttribute("x2", x(tt)); cross.setAttribute("visibility", "visible");
      tip.replaceChildren();
      const when = document.createElement("div"); when.className = "when"; when.textContent = fmtWhen(tt);
      tip.append(when);
      const rows = [[names[0], nearest(train, tt), "var(--series-1)"]];
      if (val.length) rows.push([names[1], nearest(val, tt), "var(--series-2)"]);
      for (const [name, p, color] of rows) {
        const row = document.createElement("div");
        const key = document.createElement("i"); key.style.borderColor = color;
        const b = document.createElement("b"); b.textContent = p.v.toFixed(2);
        row.append(key, b, document.createTextNode(" " + name));
        tip.append(row);
      }
      tip.classList.remove("hidden");
      const box = svg.getBoundingClientRect(), hb = host.getBoundingClientRect();
      const px = box.left - hb.left + (x(tt) / W) * box.width;
      const left = px > hb.width * 0.55 ? px - tip.offsetWidth - 12 : px + 12;
      tip.style.left = Math.max(4, left) + "px";
      tip.style.top = (box.top - hb.top + 8) + "px";
    }
    function hide() { cur = -1; cross.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); }
    svg.addEventListener("pointermove", ev => {
      const box = svg.getBoundingClientRect();
      const tt = ((ev.clientX - box.left) / box.width * W - L) / (W - L - R) * tmax;
      let best = 0;
      stops.forEach((st, i) => { if (Math.abs(st - tt) < Math.abs(stops[best] - tt)) best = i; });
      show(best);
    });
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", ev => {
      if (ev.key === "ArrowLeft") { show(cur < 0 ? stops.length - 1 : cur - 1); ev.preventDefault(); }
      else if (ev.key === "ArrowRight") { show(cur < 0 ? 0 : cur + 1); ev.preventDefault(); }
      else if (ev.key === "Escape") hide();
    });
    if (old) old.replaceWith(svg); else host.append(svg);
    host.append(tip);
  }

  window.FPChart = { draw, fmtWhen };
})();
