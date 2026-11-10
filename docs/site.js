/* The website's own script: the builder on the page, the network in a worker beside it,
   and everything it reports drawn here. Text from the network goes on the page with
   textContent only. Nothing is sent anywhere. */
"use strict";
(function () {
  const D = window.FPDesign, $ = id => document.getElementById(id);
  const STAGES = [
    ["Random noise", "It is guessing at random. Everyone starts here."],
    ["Letter soup", "It has noticed which letters are common, and where the spaces go."],
    ["Word salad", "Real words are turning up, in an order that makes no sense yet."],
    ["Sentences", "Mostly real words in sentence-like order. Meaning comes and goes."],
    ["Fluent", "At a glance it reads like its text. This is as good as small networks get."],
  ];
  const SERIES = ["Text it studies", "Text it has never seen"];
  const REST_AFTER = 600;        // seconds of learning before it pauses itself (phones get warm)
  const OWN_MIN = 2000, OWN_MAX = 2000000;
  const EMPTY = "Nothing yet. Switch it on and its writing appears here: rubbish at first, then, bit by bit, something that sounds like ";

  const S = {
    design: null, problem: "", builder: null,
    shakespeare: null, own: null,           // the two possible texts
    worker: null, run: 0, state: "idle",    // idle | starting | running | paused | stale | failed
    info: null, tick: null, m: [], e: [], samples: [], pick: -1, stage: 0,
    asked: null, restAt: REST_AFTER, lastDraw: 0, seed: 1, slowTold: false,
  };

  // ------------------------------------------------------------------ small helpers
  const tidy = t => String(t || "").replace(/[\u0000-\u0008\u000B-\u001F\u007F]/g, "�");
  const int = n => Math.round(n).toLocaleString("en");
  function big(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + " million";
    if (n >= 1e4) return Math.round(n / 1e3) + " thousand";
    return int(n);
  }
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  const source = () => (S.own ? "your text" : "Shakespeare");
  const steps = n => int(n) + (n === 1 ? " step" : " steps");

  // ------------------------------------------------------------------ the design
  function startDesign() {
    const wanted = D.parse(new URLSearchParams(location.search).get("d"));
    return D.fix(wanted || { width: 64, heads: 4, context: 64, blocks: [{ type: "transformer" }] }, D.WEB);
  }
  function onDesign(design, reason, problem) {
    S.design = design; S.problem = problem || "";
    $("code").textContent = D.code(design);
    try { history.replaceState(null, "", "?d=" + D.code(design) + location.hash); } catch (e) { /* file: pages cannot do this; harmless */ }
    if (S.state === "running" || S.state === "paused" || S.state === "starting") {
      if (S.worker) S.worker.postMessage({ type: "pause" });
      S.state = "stale";
    }
    drawControls();
  }

  // ------------------------------------------------------------------ the text it learns from
  async function textToLearn() {
    if (S.own) return S.own;
    if (!S.shakespeare) {
      const r = await fetch("data/shakespeare.txt");
      if (!r.ok) throw new Error("Could not fetch the Shakespeare text. Check your connection and try again.");
      S.shakespeare = await r.text();
    }
    return S.shakespeare;
  }
  function lettersIn(text) {       // how many different bytes: that is the network's alphabet
    const seen = new Uint8Array(256);
    let n = 0;
    for (const b of new TextEncoder().encode(text.length > 300000 ? text.slice(0, 300000) : text)) if (!seen[b]) { seen[b] = 1; n++; }
    return n;
  }

  // ------------------------------------------------------------------ starting and stopping
  function stopWorker() { if (S.worker) { S.worker.terminate(); S.worker = null; } }
  async function start() {
    stopWorker();
    const run = ++S.run;
    Object.assign(S, { state: "starting", info: null, tick: null, m: [], e: [], samples: [], pick: -1, stage: 0, asked: null, restAt: REST_AFTER, slowTold: false });
    $("hint").classList.add("hidden");
    drawAll();
    let text;
    try { text = await textToLearn(); } catch (e) { return failed(e.message); }
    if (run !== S.run) return;
    let worker;
    try { worker = new Worker("nn-worker.js"); } catch (e) { return failed("This browser would not start the network. Try a current version of Chrome, Edge, Firefox or Safari."); }
    S.worker = worker;
    worker.onmessage = ev => { if (run === S.run) onMessage(ev.data || {}); };
    worker.onerror = () => { if (run === S.run) failed("The network stopped unexpectedly. Press Start again from nothing."); };
    const lr = 4e-3 * Math.sqrt(64 / Math.max(64, S.design.width));
    worker.postMessage({ type: "init", design: S.design, text, batch: 8, lr, seed: S.seed++ });
    const lab = $("lab").getBoundingClientRect();
    if (lab.top > window.innerHeight * 0.6) $("controls").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }
  function failed(message) {
    stopWorker();
    S.state = "failed";
    drawControls();
    const st = $("status"); st.textContent = message; st.classList.add("bad");
  }
  function pause(why) {
    if (S.state !== "running") return;
    S.worker.postMessage({ type: "pause" });
    S.state = "paused";
    drawControls(why);
  }
  function resume() {
    if (S.state !== "paused") return;
    S.worker.postMessage({ type: "run" });
    S.state = "running";
    drawControls();
  }

  function onMessage(m) {
    if (m.type === "ready") {
      S.info = m;
      S.builder.setVocab(m.vocab);
      S.state = "running";
      S.worker.postMessage({ type: "run" });
      drawControls(); drawFacts();
    } else if (m.type === "tick") {
      S.tick = m;
      if (m.loss != null) S.m.push({ t: m.seconds, loss: m.loss });
      if (S.state === "running" && m.seconds >= S.restAt) { S.restAt += REST_AFTER; pause("rest"); }
      if (!S.slowTold && S.state === "running" && m.seconds > 6 && m.speed > 0 && m.speed < 3) {
        S.slowTold = true;
        const hint = $("hint");
        hint.textContent = "This design is heavy for this device, so it learns slowly. Fewer blocks or a smaller width will be quicker. On your own computer, the app can use the graphics card.";
        hint.classList.remove("hidden");
      }
      drawControls(); drawFacts(); drawChart(false);
    } else if (m.type === "eval") {
      S.e.push({ t: m.seconds, train: m.train, val: m.val });
      S.stage = stageFor(m.val != null ? m.val : m.train);
      drawStage(); drawNumbers(); drawChart(true);
    } else if (m.type === "sample") {
      S.samples.push(m);
      drawSheet();
    } else if (m.type === "wrote") {
      if (S.asked && S.asked.id === m.id) { S.asked.text += m.text; S.asked.done = m.done; drawSheet(); if (m.done) drawTry(); }
    } else if (m.type === "error") failed(m.message);
  }

  /* Where its score sits between three yardsticks measured on the same text (see nn-worker.js):
     the same ladder the app uses. */
  function stageFor(loss) {
    const b = S.info && S.info.base;
    if (!b || loss == null) return 0;
    const h0 = b.uniform, h1 = Math.min(b.unigram, h0), h2 = Math.min(b.bigram, h1);
    if (loss > h1 + 0.35 * (h0 - h1)) return 1;
    if (loss > h2 + 0.40 * (h1 - h2)) return 2;
    if (loss > 0.80 * h2) return 3;
    if (loss > 0.58 * h2) return 4;
    return 5;
  }

  // ------------------------------------------------------------------ drawing
  function drawControls(why) {
    const go = $("go"), again = $("again"), st = $("status"), t = S.tick;
    st.classList.remove("bad");
    const label = { idle: "Switch it on", starting: "Setting up…", running: "Pause", paused: "Carry on", stale: "Start this new design", failed: "Try again" }[S.state];
    go.textContent = label;
    go.disabled = S.state === "starting" || (!!S.problem && S.state !== "running" && S.state !== "paused");
    go.classList.toggle("primary", S.state !== "running");
    again.classList.toggle("hidden", !(S.state === "running" || S.state === "paused"));
    if (S.problem && S.state !== "running" && S.state !== "paused") st.textContent = S.problem;
    else if (S.state === "idle") st.textContent = "Not switched on yet. It knows nothing.";
    else if (S.state === "starting") st.textContent = S.own || S.shakespeare ? "Building your network…" : "Fetching Shakespeare…";
    else if (S.state === "stale") st.textContent = "You changed the design. A new design starts from nothing, so switch it on again.";
    else if (S.state === "paused") st.textContent = why === "rest" ? "Paused by itself after " + Math.round((S.restAt - REST_AFTER) / 60) + " minutes, to be kind to your battery. Press Carry on for more." : "Paused after " + steps(t ? t.step : 0) + ". Nothing is lost.";
    else if (S.state === "running") {
      st.textContent = t && t.step ? `Learning from ${source()}: ${steps(t.step)} so far, about ${t.speed >= 10 ? Math.round(t.speed) : t.speed.toFixed(1)} a second.` : "Learning from " + source() + "…";
    }
    $("tryBtn").disabled = !S.info || S.state === "starting" || S.state === "failed" || (S.asked && !S.asked.done);
    document.title = S.state === "running" ? "Learning… Foolproof Transformer Trainer" : "Foolproof Transformer Trainer: build a transformer and watch it learn";
  }
  function drawSheet() {
    const body = $("writing"), slider = $("pages"), note = $("pagesNote"), n = S.samples.length;
    if (S.asked) {
      $("sheetTitle").textContent = "Carrying on from your words";
      body.replaceChildren(el("span", "given", S.asked.prompt), el("span", S.asked.done ? "said" : "said caret", tidy(S.asked.text)));
      $("when").textContent = steps(S.asked.step);
      body.scrollTop = body.scrollHeight;
      return;
    }
    $("sheetTitle").textContent = "Written by your network";
    slider.max = String(Math.max(0, n - 1));
    if (!n) {
      body.replaceChildren(el("span", "empty", EMPTY + source() + "."));
      $("when").textContent = "0 steps"; slider.value = "0";
      note.textContent = "Every few seconds it writes a fresh page. Slide back to see how it wrote before.";
      return;
    }
    const i = S.pick < 0 || S.pick >= n ? n - 1 : S.pick, smp = S.samples[i];
    slider.value = String(i);
    body.replaceChildren(el("span", "said", tidy(smp.text)));
    body.scrollTop = 0;
    $("when").textContent = steps(smp.step);
    note.textContent = i === n - 1 ? (n > 1 ? `Page ${n} of ${n}, the newest. Slide left to see how it wrote before.` : "Its first page, before any learning: pure guesswork.")
                                   : `Page ${i + 1} of ${n}, from earlier. Slide right for the newest.`;
  }
  function drawStage() {
    const st = S.stage;
    [...$("stageBoxes").children].forEach((b, i) => b.classList.toggle("on", i < st));
    [...$("stageNames").children].forEach((b, i) => b.classList.toggle("on", i === st - 1));
    $("stageTitle").textContent = `Stage ${Math.max(1, st)} of 5: ${STAGES[Math.max(1, st) - 1][0]}`;
    $("stageDesc").textContent = st ? STAGES[st - 1][1] : "Before it is switched on it can only guess at random. Everyone starts here.";
  }
  function drawChart(force) {
    const now = Date.now(), viz = $("viz");
    if (!force && now - S.lastDraw < 1200) return;
    if (viz.matches(":hover") && S.m.length > 2 && !force) return;    // do not pull the chart from under a pointer
    S.lastDraw = now;
    window.FPChart.draw(viz, S.m, S.e, SERIES);
  }
  function drawNumbers() {
    $("nums").replaceChildren(...S.e.slice(-200).reverse().map(p => {
      const tr = document.createElement("tr");
      tr.append(el("td", "", window.FPChart.fmtWhen(p.t)), el("td", "", p.train != null ? p.train.toFixed(3) : ""), el("td", "", p.val != null ? p.val.toFixed(3) : ""));
      return tr;
    }));
  }
  function drawFacts() {
    const t = S.tick, facts = [];
    if (S.info) facts.push([big(S.info.params), "numbers in your network"]);
    if (t) facts.push([int(t.step), "learning steps"], [big(t.seen), "letters read"]);
    if (t && S.state === "running" && t.speed > 0) facts.push([t.speed >= 10 ? String(Math.round(t.speed)) : t.speed.toFixed(1), "steps a second"]);
    $("facts").replaceChildren(...facts.map(([v, k]) => { const d = document.createElement("div"); d.append(el("b", "", v), el("span", "", k)); return d; }));
  }
  function drawTry() {
    $("tryBack").classList.toggle("hidden", !S.asked);
    $("tryBtn").disabled = !S.info || S.state === "starting" || S.state === "failed" || (S.asked && !S.asked.done);
  }
  function drawAll() { drawControls(); drawSheet(); drawStage(); drawChart(true); drawNumbers(); drawFacts(); drawTry(); }

  // ------------------------------------------------------------------ wiring up the page
  function boot() {
    if (!window.Worker || !window.FPBuilder) {
      $("builder").replaceChildren(el("p", "note bad", "This browser is too old to run the network. A current version of Chrome, Edge, Firefox or Safari will work."));
      return;
    }
    S.design = startDesign();
    S.builder = window.FPBuilder.mount($("builder"), { design: S.design, limits: D.WEB, vocab: 65, onChange: onDesign });
    onDesign(S.builder.get(), "start", S.builder.problem());

    $("go").addEventListener("click", () => {
      if (S.state === "running") pause();
      else if (S.state === "paused") resume();
      else start();
    });
    $("again").addEventListener("click", start);
    $("pages").addEventListener("input", e => { const i = Number(e.target.value); S.pick = i >= S.samples.length - 1 ? -1 : i; drawSheet(); });

    $("tryForm").addEventListener("submit", e => {
      e.preventDefault();
      if (!S.worker || !S.info) return;
      const prompt = $("tryIn").value;
      S.asked = { id: "ask-" + Date.now(), prompt, text: "", done: false, step: S.tick ? S.tick.step : 0 };
      S.worker.postMessage({ type: "write", id: S.asked.id, prompt, n: 280, temperature: 0.8 });
      drawSheet(); drawTry();
    });
    $("tryBack").addEventListener("click", () => { if (S.worker) S.worker.postMessage({ type: "stop_write" }); S.asked = null; drawSheet(); drawTry(); });

    const own = $("ownText");
    const ownState = () => {
      const n = own.value.trim().length;
      $("ownUse").disabled = n < OWN_MIN;
      $("ownCount").textContent = n < OWN_MIN ? `${int(n)} letters so far. It needs at least ${int(OWN_MIN)}, about one page.` : `${int(Math.min(n, OWN_MAX))} letters. ${n > OWN_MAX ? "Only the first two million are used. " : ""}Ready.`;
    };
    own.addEventListener("input", ownState);
    $("ownUse").addEventListener("click", () => {
      S.own = own.value.trim().slice(0, OWN_MAX);
      S.builder.setVocab(lettersIn(S.own));
      $("ownBack").classList.remove("hidden");
      start();
    });
    $("ownBack").addEventListener("click", () => { S.own = null; S.builder.setVocab(65); $("ownBack").classList.add("hidden"); start(); });

    for (const btn of document.querySelectorAll("[data-copy]")) {
      btn.addEventListener("click", async () => {
        const text = $(btn.getAttribute("data-copy")).textContent;
        try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
        catch (e) { const r = document.createRange(); r.selectNodeContents($(btn.getAttribute("data-copy"))); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); btn.textContent = "Selected: now copy"; }
        setTimeout(() => { btn.textContent = "Copy"; }, 2200);
      });
    }
    window.addEventListener("resize", () => drawChart(true));
    drawAll();
    // fetch the text quietly, so "Switch it on" is instant
    textToLearn().then(t => { if (!S.own && S.state === "idle") S.builder.setVocab(lettersIn(t)); }).catch(() => {});
  }
  boot();
})();
