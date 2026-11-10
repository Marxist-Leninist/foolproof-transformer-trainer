/* Device self-check (check.html).

   Runs the same builder and the same in-page network as the main page on whatever device opens it,
   and writes what happened on the page. Text from the network goes on the page with textContent only.

   Nothing is sent anywhere, with one exception that exists for automated testing: when the address ends
   in ?report=1 the result is also posted to "__report" on the server the page came from. The test
   scripts in tests/web/ run a local server that listens for that. The public website has no such
   address, so there the post simply fails and is ignored.

   ?auto=1 is accepted for the test scripts as well; the checks start by themselves either way. */
"use strict";
(function () {
  const D = window.FPDesign, B = window.FPBuilder, $ = id => document.getElementById(id);
  const query = new URLSearchParams(location.search);
  const REPORT = query.get("report") === "1";
  const VERSION = 1;
  const PROMPT = "To be, or ";
  const START = { version: 1, width: 64, heads: 4, context: 64, blocks: [{ type: "transformer" }, { type: "attention" }, { type: "feed_forward" }] };

  const CHECKS = [
    ["scripts", "The page's own scripts loaded"],
    ["builder", "The block builder draws its blocks"],
    ["tap", "Tapping a block in the tray adds it, and its remove button takes it off"],
    ["drag", "Dragging re-orders the stack (pointer events made by this page, not by a hand)"],
    ["worker", "The network starts in the background"],
    ["learns", "It learns: its mistakes on text it never saw go down"],
    ["writes", "Its writing changes as it learns"],
    ["prompt", "It carries on from words it is given"],
    ["pause", "Pause stops it and Carry on starts it again"],
  ];
  const state = {};            // id -> { state, detail }
  const facts = {};
  const errors = [];           // script errors, refused promises and anything the page's security policy blocked
  let builder = null, changes = [], worker = null, running = false, finger = "not tried";
  let touchDown = false, ghostSeen = false, orderAtTouch = "";

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const kinds = () => Array.from(document.querySelectorAll("#builder .bld-block")).map(e => e.dataset.kind);
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  const tidy = t => String(t || "").replace(/[\u0000-\u0008\u000B-\u001F\u007F]/g, "�");

  // ------------------------------------------------------------------ the list on the page
  const WORD = { wait: "Waiting", run: "Checking", ok: "Worked", fail: "Failed", skip: "Skipped" };
  function drawList() {
    const list = $("checks");
    list.replaceChildren();
    for (const [id, label] of CHECKS) {
      const s = state[id] || { state: "wait", detail: "" };
      const li = el("li");
      li.append(el("span", "mark " + s.state, WORD[s.state]), el("span", "what", label));
      if (s.detail) li.append(el("span", "detail", s.detail));
      list.append(li);
    }
  }
  function set(id, st, detail) { state[id] = { state: st, detail: detail || "" }; drawList(); }
  function verdict(text, cls) { const v = $("verdict"); v.textContent = text; v.className = "chk-verdict" + (cls ? " " + cls : ""); }

  function drawFacts() {
    const dl = $("facts");
    dl.replaceChildren();
    for (const k of Object.keys(facts)) dl.append(el("dt", null, k), el("dd", null, String(facts[k])));
  }
  function gatherFacts() {
    const n = navigator, coarse = window.matchMedia && matchMedia("(pointer: coarse)").matches;
    facts["Browser says it is"] = n.userAgent;
    facts["Screen"] = `${screen.width} x ${screen.height}, page ${window.innerWidth} x ${window.innerHeight}, ${window.devicePixelRatio || 1}x`;
    facts["Touch"] = `${n.maxTouchPoints || 0} touch point${n.maxTouchPoints === 1 ? "" : "s"}; main pointer is ${coarse ? "a finger" : "a mouse or pen"}`;
    facts["Processor threads"] = n.hardwareConcurrency || "not said";
    facts["WebAssembly"] = typeof WebAssembly === "object" ? "available" : "not available";
    drawFacts();
  }

  // ------------------------------------------------------------------ builder checks
  function mountBuilder() {
    changes = [];
    if (builder) builder.destroy();
    builder = B.mount($("builder"), { design: START, limits: D.WEB, vocab: 65, onChange: (d, reason) => { changes.push(reason); onFingerChange(); } });
  }
  function fire(target, type, x, y, id) {
    target.dispatchEvent(new PointerEvent(type, { bubbles: true, cancelable: true, composed: true, pointerId: id, pointerType: "mouse",
      isPrimary: true, button: type === "pointermove" ? -1 : 0, buttons: type === "pointerup" ? 0 : 1, clientX: x, clientY: y }));
  }
  async function syntheticDrag() {
    // carry the first block to just below the last one, in small steps, the way a hand would
    const blocks = document.querySelectorAll("#builder .bld-block");
    const from = blocks[0], last = blocks[blocks.length - 1];
    from.scrollIntoView({ block: "center" });
    await sleep(120);
    const a = from.getBoundingClientRect(), b = last.getBoundingClientRect();
    const grip = from.querySelector(".bld-grip") || from, g = grip.getBoundingClientRect();
    const x0 = g.left + g.width / 2, y0 = g.top + g.height / 2, x1 = x0, y1 = b.bottom - 3;
    fire(grip, "pointerdown", x0, y0, 9041);
    for (let i = 1; i <= 10; i++) { fire(from, "pointermove", x0 + (x1 - x0) * i / 10, y0 + (y1 - y0) * i / 10, 9041); await sleep(16); }
    const lifted = !!document.querySelector(".bld-ghost");
    fire(from, "pointerup", x1, y1, 9041);
    await sleep(80);
    return { lifted, height: a.height };
  }

  // ------------------------------------------------------------------ the finger check (needs a person)
  function fingerStatus(text, good) { const s = $("fingerStatus"); s.textContent = text; s.className = "chk-status" + (good ? " good" : ""); }
  function watchFinger() {
    const host = $("builder");
    host.addEventListener("pointerdown", ev => {
      if (ev.isTrusted && ev.pointerType === "touch") { touchDown = true; ghostSeen = false; orderAtTouch = kinds().join(","); }
    }, true);
    host.addEventListener("pointermove", ev => {
      if (ev.isTrusted && ev.pointerType === "touch" && touchDown && document.querySelector(".bld-ghost")) ghostSeen = true;
    }, true);
    window.addEventListener("pointerup", ev => { if (ev.isTrusted && ev.pointerType === "touch") setTimeout(() => { touchDown = false; }, 400); }, true);
  }
  function onFingerChange() {
    if (!touchDown || !ghostSeen) return;
    const now = kinds().join(",");
    if (now === orderAtTouch) return;
    finger = "worked";
    fingerStatus("A finger drag worked on this device: the stack went from " + orderAtTouch.replace(/,/g, ", ") + " to " + now.replace(/,/g, ", ") + ".", true);
    summarise();
    if (REPORT) send();
  }

  // ------------------------------------------------------------------ the network
  function startWorker(text) {
    return new Promise((resolve, reject) => {
      const box = { ready: null, ticks: [], evals: [], samples: [], wrote: {}, error: null };
      let w;
      try { w = new Worker("nn-worker.js"); } catch (e) { reject(new Error("the browser would not start a background worker")); return; }
      const timer = setTimeout(() => reject(new Error("no answer from the network after 40 seconds")), 40000);
      w.onerror = ev => { box.error = (ev && ev.message) || "the worker stopped"; clearTimeout(timer); reject(new Error(box.error)); };
      w.onmessage = ev => {
        const m = ev.data || {};
        if (m.type === "ready") { box.ready = m; clearTimeout(timer); resolve(box); }
        else if (m.type === "tick") box.ticks.push(m);
        else if (m.type === "eval") box.evals.push(m);
        else if (m.type === "sample") box.samples.push(m);
        else if (m.type === "wrote") { const j = box.wrote[m.id] || (box.wrote[m.id] = { text: "", done: false }); j.text += m.text; j.done = m.done; }
        else if (m.type === "error") { box.error = m.message; clearTimeout(timer); reject(new Error(m.message)); }
      };
      box.post = msg => w.postMessage(msg);
      box.stop = () => w.terminate();
      worker = box;
      w.postMessage({ type: "init", design: START, text, batch: 8, lr: 4e-3, seed: 1 });
    });
  }
  const loss = e => (e.val != null ? e.val : e.train);
  async function until(test, ms, every) {
    const end = performance.now() + ms;
    while (performance.now() < end) { if (test()) return true; await sleep(every || 150); }
    return !!test();
  }

  // ------------------------------------------------------------------ the run
  async function run() {
    if (running) return;
    running = true;
    $("again").disabled = true;
    for (const [id] of CHECKS) state[id] = { state: "wait", detail: "" };
    drawList();
    verdict("Checking this device…");
    if (worker) { worker.stop(); worker = null; }

    // 1 scripts
    set("scripts", "run");
    const haveScripts = !!(D && B && typeof D.classic === "function" && typeof B.mount === "function");
    set("scripts", haveScripts ? "ok" : "fail", haveScripts ? "" : "design.js or builder.js did not load");

    // 2-4 builder
    if (haveScripts) {
      try {
        set("builder", "run");
        mountBuilder();
        await sleep(60);
        const first = kinds();
        set("builder", first.join(",") === "transformer,attention,feed_forward" ? "ok" : "fail", "blocks on the page: " + (first.join(", ") || "none"));

        set("tap", "run");
        const piece = document.querySelector("#builder .bld-piece.feed_forward");
        if (piece) piece.click();
        await sleep(80);
        const added = kinds();
        const blocks = document.querySelectorAll("#builder .bld-block");
        const remove = blocks.length ? blocks[blocks.length - 1].querySelector('[data-act="remove"]') : null;
        if (remove) remove.click();
        await sleep(80);
        const removed = kinds();
        const tapOk = added.length === 4 && added[3] === "feed_forward" && removed.length === 3 && changes.length >= 2;
        set("tap", tapOk ? "ok" : "fail", `after the tap: ${added.length} blocks; after remove: ${removed.length} blocks`);

        set("drag", "run");
        mountBuilder();
        await sleep(60);
        if (typeof PointerEvent !== "function") set("drag", "fail", "this browser has no pointer events");
        else {
          const d = await syntheticDrag();
          const after = kinds();
          const dragOk = d.lifted && after.join(",") === "attention,feed_forward,transformer";
          set("drag", dragOk ? "ok" : "fail", `block lifted: ${d.lifted ? "yes" : "no"}; order afterwards: ${after.join(", ")}`);
        }
        mountBuilder();
      } catch (e) {
        for (const id of ["builder", "tap", "drag"]) if (state[id].state !== "ok" && state[id].state !== "fail") set(id, "fail", String(e && e.message || e));
      }
    } else for (const id of ["builder", "tap", "drag"]) set(id, "skip", "needs the scripts");

    // 5-9 network
    let box = null;
    try {
      set("worker", "run");
      const r = await fetch("data/shakespeare.txt");
      if (!r.ok) throw new Error("could not fetch the text to learn from");
      const text = await r.text();
      box = await startWorker(text);
      facts["Loops doing the sums"] = box.ready.backend === "wasm" ? "WebAssembly (the fast path)" : "plain JavaScript (the slow path)";
      facts["Network size"] = Math.round(box.ready.params).toLocaleString("en") + " numbers to learn";
      drawFacts();
      set("worker", "ok", `started on ${box.ready.backend === "wasm" ? "WebAssembly" : "plain JavaScript"}, ${Math.round(box.ready.params).toLocaleString("en")} numbers to learn`);
    } catch (e) {
      set("worker", "fail", String(e && e.message || e));
    }
    if (box) {
      try {
        set("learns", "run");
        box.post({ type: "run" });
        const fell = await until(() => box.error || (box.evals.length >= 2 && loss(box.evals[0]) - loss(box.evals[box.evals.length - 1]) >= 0.6), 60000, 200);
        if (box.error) throw new Error(box.error);
        const e0 = box.evals[0], e1 = box.evals[box.evals.length - 1], t = box.ticks[box.ticks.length - 1];
        const speed = t && t.seconds > 0 ? t.step / t.seconds : 0;
        facts["Speed"] = speed ? speed.toFixed(1) + " learning steps a second" : "not measured";
        drawFacts();
        set("learns", fell ? "ok" : "fail", e0 && e1 ? `from ${loss(e0).toFixed(2)} to ${loss(e1).toFixed(2)} in ${e1.step} steps (${e1.seconds.toFixed(1)} seconds of sums)` : "no measurements arrived");

        set("writes", "run");
        // a page written after the loss fell, not one from the first few steps
        const got = await until(() => box.samples.length >= 2 && box.samples[box.samples.length - 1].step >= e1.step, 30000, 200);
        const s0 = box.samples[0], s1 = box.samples[box.samples.length - 1];
        const writesOk = got && s0 && s1 && s0.text !== s1.text && s1.text.indexOf(" ") >= 0;
        set("writes", writesOk ? "ok" : "fail", s1 ? "latest: " + JSON.stringify(tidy(s1.text).slice(0, 60)) : "no writing arrived");

        set("prompt", "run");
        box.post({ type: "write", id: "chk", prompt: PROMPT, n: 80, temperature: 0.8 });
        await until(() => box.wrote.chk && box.wrote.chk.done, 25000, 150);
        const w = box.wrote.chk;
        set("prompt", w && w.done && w.text.length >= 40 ? "ok" : "fail", w ? JSON.stringify(tidy(PROMPT + w.text).slice(0, 70)) : "nothing came back");

        set("pause", "run");
        box.post({ type: "pause" });
        await sleep(900);
        const stoppedAt = box.ticks.length ? box.ticks[box.ticks.length - 1].step : 0;
        await sleep(1500);
        const stillAt = box.ticks.length ? box.ticks[box.ticks.length - 1].step : 0;
        box.post({ type: "run" });
        const moved = await until(() => box.ticks.length && box.ticks[box.ticks.length - 1].step > stillAt, 10000, 150);
        box.post({ type: "pause" });
        set("pause", stoppedAt === stillAt && moved ? "ok" : "fail", `held at step ${stillAt} while paused, then went on to step ${box.ticks[box.ticks.length - 1].step}`);
      } catch (e) {
        for (const id of ["learns", "writes", "prompt", "pause"]) if (state[id].state === "run" || state[id].state === "wait") set(id, state[id].state === "run" ? "fail" : "skip", String(e && e.message || e));
      }
      box.stop();
      worker = null;
    } else for (const id of ["learns", "writes", "prompt", "pause"]) set(id, "skip", "needs the network to start");

    const bad = CHECKS.filter(([id]) => state[id].state !== "ok");
    if (!bad.length && !errors.length) verdict(`All ${CHECKS.length} automatic checks worked on this device.`, "good");
    else if (!bad.length) verdict(`The ${CHECKS.length} automatic checks worked, but the browser reported ${errors.length === 1 ? "an error" : errors.length + " errors"} along the way. See the result below.`, "bad");
    else verdict(`${bad.length} of ${CHECKS.length} automatic checks did not work on this device. The list below says which.`, "bad");
    running = false;
    $("again").disabled = false;
    summarise();
    window.scrollTo(0, 0);       // the drag check scrolled down to the builder; the verdict is at the top
    if (REPORT) send();
  }

  // ------------------------------------------------------------------ the result, as text and as data
  function result() {
    const checks = CHECKS.map(([id, label]) => ({ id, label, state: state[id] ? state[id].state : "wait", detail: state[id] ? state[id].detail : "" }));
    const ok = checks.filter(c => c.state === "ok").length;
    return { version: VERSION, when: new Date().toISOString(), passed: ok === checks.length && !errors.length, ok, total: checks.length, finger,
             errors: errors.slice(0, 10), checks, facts: Object.assign({}, facts), page: location.pathname };
  }
  function summarise() {
    const r = result();
    const lines = [`Foolproof website check v${VERSION}: ${r.passed ? "ALL WORKED" : "PROBLEMS"} (${r.ok} of ${r.total}); finger drag: ${r.finger}`, r.when];
    for (const c of r.checks) if (c.state !== "ok") lines.push(`- ${WORD[c.state]}: ${c.label}${c.detail ? " [" + c.detail + "]" : ""}`);
    for (const e of r.errors) lines.push("- Browser error: " + e);
    for (const k of Object.keys(r.facts)) lines.push(`${k}: ${r.facts[k]}`);
    $("summary").value = lines.join("\n");
  }
  function send() {
    try {
      fetch("__report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(result()) }).catch(() => {});
    } catch (e) { /* no test server listening: nothing to do */ }
  }

  $("again").addEventListener("click", run);
  $("copy").addEventListener("click", () => {
    const box = $("summary"), told = $("copied");
    const done = ok => { told.textContent = ok ? "Copied." : "Select the text and copy it by hand."; };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(box.value).then(() => done(true), () => { box.select(); done(false); });
    else { box.select(); done(false); }
  });
  const note = text => { if (errors.length < 10) errors.push(String(text).slice(0, 200)); };
  window.addEventListener("error", ev => note("script error: " + ((ev && ev.message) || "unknown")));
  window.addEventListener("unhandledrejection", ev => note("refused promise: " + ((ev && ev.reason && ev.reason.message) || (ev && ev.reason) || "unknown")));
  document.addEventListener("securitypolicyviolation", ev => note("blocked by the page's security policy: " + ev.violatedDirective + " " + (ev.blockedURI || "")));
  gatherFacts();
  drawList();
  if (D && B) watchFinger();
  run();
})();
