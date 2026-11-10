/* Foolproof Trainer - the window.
   No framework, no build step. Everything that comes from the internet or from a
   model is put on the page with textContent, never as HTML. */
"use strict";
(function () {
  const TOKEN = document.querySelector('meta[name="fp-token"]').content;
  const root = document.getElementById("app");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ tiny helpers
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "value") el.value = v;
      else if (k === "checked" || k === "disabled" || k === "open" || k === "selected") el[k] = !!v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    put(el, kids);
    return el;
  }
  function put(el, kids) {
    for (const k of kids.flat(Infinity)) {
      if (k == null || k === false) continue;
      el.append(k.nodeType ? k : document.createTextNode(String(k)));
    }
    return el;
  }
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  // early in training a model emits junk bytes: show them as a harmless mark instead of control codes
  const tidy = t => String(t || "").replace(/[\u0000-\u0008\u000B-\u001F\u007F]/g, "\uFFFD");
  const int = n => Math.round(n).toLocaleString("en");
  function big(n) {
    if (n == null) return "?";
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " billion";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + " million";
    if (n >= 1e4) return Math.round(n / 1e3) + " thousand";
    return int(n);
  }
  function dur(sec) {
    sec = Math.max(0, Math.round(sec));
    if (sec < 60) return sec + " seconds";
    const m = Math.round(sec / 60);
    if (m < 60) return m + (m === 1 ? " minute" : " minutes");
    const hh = Math.floor(m / 60), mm = m % 60;
    return hh + (hh === 1 ? " hour" : " hours") + (mm ? " " + mm + " min" : "");
  }
  function mb(bytes) {
    if (!bytes) return "";
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + " GB";
    if (bytes >= 1e6) return Math.round(bytes / 1e6) + " MB";
    return Math.max(1, Math.round(bytes / 1e3)) + " KB";
  }
  const STAGES = [
    ["Random noise", "It is guessing at random. Everyone starts here."],
    ["Letter soup", "It has noticed which letters and word-pieces are common, and where the spaces go."],
    ["Word salad", "Real words are turning up, in an order that makes no sense yet."],
    ["Sentences", "Mostly real words in sentence-like order. Meaning comes and goes."],
    ["Fluent", "At a glance it reads like its text. This is as good as small AIs get."],
  ];

  async function call(method, path, body, params) {
    let url = path;
    if (params) url += "?" + new URLSearchParams(params);
    const opts = { method, headers: { "X-FP-Token": TOKEN } };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    let r;
    try { r = await fetch(url, opts); } catch (e) { throw new Error("The app is not answering. Is its window still open?"); }
    let j = null;
    try { j = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) throw new Error((j && j.error) || `The app answered with an error (${r.status}).`);
    return j;
  }
  const get = (path, params) => call("GET", path, undefined, params);
  const post = (path, body) => call("POST", path, body || {});

  // ------------------------------------------------------------------ state
  const S = {
    app: null, view: "home", arg: null, tab: "starter",
    draft: null, prep: null, prepError: null, rec: null, recFor: null,
    live: null, model: null,
    chat: { slug: null, msgs: [], outs: [], busy: false, id: null, temp: 0.8, len: 0 },
    hf: { q: "", results: null, busy: false, error: null, info: null, infoBusy: false },
    files: { list: [], text: "", folder: "", busy: false, sent: 0 },
    glossary: false, lost: false, toastTimer: null,
  };
  const R = {}; // live DOM references for the training screen
  function newDraft() {
    return { kind: null, starter: null, lang: "fr", hf: null, amount: null, tokenizer: "auto", dataName: "",
             dataset: null, preset: null, budget: null, aiName: "", adv: {}, autostop: true,
             build: false, architecture: null, problem: "", fit: null };
  }
  function newLive(run) {
    return { run, start: null, m: [], e: [], samples: [], notices: [], metrics: null, ev: null, paused: false,
             status: "starting", end: null, budget: 0, pick: -1, tryText: "", tryBusy: false, loaded: false };
  }
  const job = () => (S.app && S.app.job) || null;
  const modelBySlug = slug => (S.app.models || []).find(m => m.slug === slug) || null;
  const datasetBySlug = slug => (S.app.datasets || []).find(d => d.slug === slug) || null;

  function toast(msg, bad) {
    document.querySelectorAll(".toast").forEach(n => n.remove());
    const el = h("div", { class: "toast" + (bad ? " bad" : ""), role: "status", text: msg });
    document.body.append(el);
    clearTimeout(S.toastTimer);
    S.toastTimer = setTimeout(() => el.remove(), bad ? 9000 : 4500);
  }
  async function act(fn) {  // run a button action; show any problem in plain words
    try { return await fn(); } catch (e) { toast(e.message, true); return null; }
  }
  async function refresh() { S.app = await get("/api/state"); }

  // ------------------------------------------------------------------ navigation
  function go(view, arg) {
    const target = "#" + view + (arg ? "/" + encodeURIComponent(arg) : "");
    if (location.hash === target) route(); else location.hash = target;
  }
  async function route() {
    const [v, a] = location.hash.slice(1).split("/");
    S.view = VIEWS[v] ? v : "home";
    S.arg = a ? decodeURIComponent(a) : null;
    if (S.view === "data" && !S.draft) S.draft = newDraft();
    if (S.view === "brain") {
      if (!S.draft || !S.draft.dataset || !datasetBySlug(S.draft.dataset)) { go("data"); return; }
      await loadRec();
    }
    if (S.view === "train") {
      const slug = S.arg || (job() && job().run) || (S.live && S.live.run);
      if (!slug) { go("home"); return; }
      S.arg = slug;
      if (!S.live || S.live.run !== slug || !S.live.loaded) await loadLive(slug);
    }
    if (S.view === "ai") {
      if (!S.arg || !modelBySlug(S.arg)) { go("home"); return; }
      if (S.chat.slug !== S.arg) S.chat = { slug: S.arg, msgs: [], outs: [], busy: false, id: null, temp: 0.8, len: 0 };
      S.model = modelBySlug(S.arg);
    }
    render();
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  async function loadRec() {
    const slug = S.draft.dataset;
    if (S.recFor === slug && S.rec) return;
    S.rec = null; S.recFor = slug;
    try {
      S.rec = await get("/api/recommend", { dataset: slug });
      if (!S.draft.preset) S.draft.preset = S.rec.recommended;
    } catch (e) { toast(e.message, true); }
  }
  async function loadLive(slug) {
    const live = newLive(slug);
    try {
      const m = await get("/api/model", { slug });
      const hist = m.history || {};
      live.m = (hist.metrics || []).map(o => ({ t: o.t, loss: o.loss }));
      live.e = (hist.evals || []).map(o => ({ t: o.t, train: o.train, val: o.val, stage: o.stage }));
      live.samples = hist.samples || [];
      live.cfg = m;
      const st = m.stats || {};
      if (st.stage) live.ev = { stage: st.stage, bpb: st.bpb, val: st.last_val, train: st.last_train, best_val: st.best_val };
      const snap = S.app.live || {};
      const running = job() && job().kind === "train" && job().run === slug;
      if (snap.run === slug) {
        live.start = snap.start || null; live.metrics = snap.metrics || null;
        if (snap.eval) live.ev = snap.eval;
        live.notices = snap.notices || []; live.paused = !!snap.paused; live.end = snap.end || null;
        live.budget = snap.budget_sec || (snap.metrics && snap.metrics.budget_sec) || (snap.start && snap.start.budget_sec) || 0;
      }
      live.status = running ? "training" : "ended";
      if (!running && !live.end) live.end = { reason: m.status === "finished" ? "budget" : "stopped", step: st.step, elapsed: st.elapsed, stage: st.stage };
    } catch (e) { toast(e.message, true); }
    live.loaded = true;
    S.live = live;
  }

  // ------------------------------------------------------------------ events from the app
  async function poll() {
    let since = S.app.seq, fails = 0;
    for (;;) {
      try {
        const r = await get("/api/events", { since, wait: 20 });
        fails = 0;
        if (S.lost) { S.lost = false; await refresh(); render(); }
        if (r.boot !== S.app.boot || r.reset) { await refresh(); since = S.app.seq; await route(); continue; }
        for (const ev of r.events) await onEvent(ev);
        since = r.seq;
      } catch (e) {
        fails++;
        if (fails >= 3 && !S.lost && !S.quit) { S.lost = true; render(); }
        await sleep(1500);
      }
    }
  }

  async function onEvent(ev) {
    const t = ev.type;
    if (t === "hw") { S.app.hw = ev.hw; if (S.view === "home" || S.view === "brain") { S.rec = null; S.recFor = null; await route(); } return; }
    if (t === "job_start") {
      await refresh();
      if (ev.kind === "train") await loadLive(ev.run);
      if (ev.kind === "prepare") { S.prep = null; S.prepError = null; }
      render();
      return;
    }
    if (t === "prep") { S.prep = ev; patchPrep(); return; }
    if (t === "error") {
      if (job() && job().kind === "prepare") S.prepError = ev.msg;
      else if (ev.job && String(ev.job).startsWith("serve")) chatFailed(ev.msg);
      else { if (S.live) S.live.error = ev.msg; toast(ev.msg, true); }
      return;
    }
    if (t === "job_end") {
      const was = ev.kind;
      await refresh();
      if (was === "prepare") {
        if (ev.reason === "done" && ev.dataset) {
          if (!S.draft) S.draft = newDraft();
          S.draft.dataset = ev.dataset; S.draft.preset = null; S.rec = null; S.recFor = null;
        }
        S.prep = null;
        render();
      } else if (was === "train") {
        if (S.live && S.live.run === ev.run) { S.live.status = "ended"; if (!S.live.end) S.live.end = { reason: ev.reason }; }
        if (S.view === "ai") S.model = modelBySlug(S.arg);
        render();
      }
      return;
    }
    if (t === "gen" || t === "gen_end") { onGen(ev); return; }
    const L = S.live;
    if (!L) return;
    if (t === "train_start") { L.start = ev; L.status = "training"; L.budget = ev.budget_sec || 0; if (S.view === "train") render(); return; }
    if (t === "metrics") {
      L.metrics = ev; L.budget = ev.budget_sec || L.budget;
      L.m.push({ t: ev.t, loss: ev.loss });
      if (L.m.length > 3000) L.m = L.m.filter((_, i) => i % 2 === 0);
      patchTrain("metrics");
      return;
    }
    if (t === "eval") { L.ev = ev; L.e.push({ t: ev.t, train: ev.train, val: ev.val, stage: ev.stage }); patchTrain("eval"); return; }
    if (t === "sample") { L.samples.push(ev); patchTrain("sample"); return; }
    if (t === "notice") { L.notices.push(ev); patchTrain("notices"); return; }
    if (t === "paused") { L.paused = !!ev.paused; if (S.view === "train") render(); return; }
    if (t === "budget") { L.budget = ev.budget_sec; patchTrain("metrics"); return; }
    if (t === "train_end") { L.end = ev; L.status = "ended"; return; }
  }

  // ------------------------------------------------------------------ frame
  function render() {
    for (const k of Object.keys(R)) delete R[k];
    if (S.view !== "train") document.title = "Foolproof Trainer";
    const kids = [topbar(), jobbar(), VIEWS[S.view]()];
    if (S.glossary) kids.push(glossary());
    if (S.lost) kids.push(h("div", { class: "lost", role: "alertdialog" },
      h("div", { class: "stack" }, h("h2", { text: "The app has stopped" }),
        h("p", { class: "lead", text: "This page lost contact with Foolproof Trainer. Start it again (double-click START-WINDOWS, or run start.sh), then press the button." }),
        h("button", { class: "btn primary", onclick: () => location.reload(), text: "Reload this page" }))));
    root.replaceChildren(...kids.flat(Infinity).filter(k => k != null && k !== false));
    if (S.view === "train") patchTrain("all");
  }

  function topbar() {
    return h("header", { class: "top" },
      h("a", { class: "brand", href: "#home" },
        h("img", { class: "brand-mark", src: "icon.svg", alt: "" }), "Foolproof Trainer"),
      h("span", { class: "top-spacer" }),
      h("button", { class: "linkbtn", onclick: () => { S.glossary = true; render(); }, text: "Words explained" }),
      h("button", { class: "linkbtn", onclick: () => act(() => post("/api/open", { what: "home" })), text: "Open my folder" }),
      h("button", { class: "linkbtn", onclick: quit, text: "Quit" }));
  }
  async function quit() {
    const j = job();
    const msg = j && j.kind === "train" ? "Training will stop and be saved. Quit Foolproof Trainer?" : "Quit Foolproof Trainer?";
    if (!window.confirm(msg)) return;
    S.quit = true;
    try { await post("/api/quit"); } catch (e) { /* closing anyway */ }
    root.replaceChildren(h("main", { class: "narrow stack" }, h("h1", { text: "Foolproof Trainer has closed" }),
      h("p", { class: "lead", text: "Everything is saved. You can close this tab. To use it again, start it the same way you did before." })));
  }

  function jobbar() {
    const j = job();
    if (!j) return null;
    if (j.kind === "prepare" && S.view !== "data")
      return h("div", { class: "jobbar" }, h("span", { class: "dot live" }), "Getting text ready: " + (j.name || ""),
        h("span", { class: "top-spacer" }), h("button", { onclick: () => go("data"), text: "Show" }));
    if (j.kind === "train" && !(S.view === "train" && S.arg === j.run)) {
      const m = modelBySlug(j.run);
      return h("div", { class: "jobbar" }, h("span", { class: "dot live" }), "Training now: " + ((m && m.name) || j.run),
        h("span", { class: "top-spacer" }), h("button", { onclick: () => go("train", j.run), text: "Watch" }));
    }
    return null;
  }

  function steps(now) {
    const names = ["Text to read", "Brain and time", "Learning", "Chat"];
    return h("ol", { class: "steps", "aria-label": "Steps" }, names.map((n, i) =>
      h("li", { class: i + 1 === now ? "now" : i + 1 < now ? "done" : "", "aria-current": i + 1 === now ? "step" : null },
        h("span", { class: "n", text: String(i + 1) }), n)));
  }

  // ------------------------------------------------------------------ home
  // Real writing from one run: the smallest brain, on a 2-core computer with no graphics card.
  const DIARY = [
    ["After 4 seconds", "\uFFFD\uFFFDeQ\uFFFDs r\uFFFDiRA\uFFFDJQed ."],
    ["After 35 seconds", "DHAn mor ant iy is yo tin yeh nt nanr I dount ive han than min wos ty le shown tou"],
    ["After 2 minutes", "But redt not will to evees you by this I salanke have."],
    ["After 25 minutes", "RICHARD:\nBut I you would to peace, in the conscience\nTo be I well, should be a more one a from Kate,\nAnd but that we not warrant love and power"],
  ];
  function diarySheet() {
    return h("figure", { class: "paper diary", style: "margin:0" },
      h("div", { class: "paper-head" }, h("span", {}, "Written by ", h("b", { text: "a brand-new AI" })), h("span", {}, "reading ", h("b", { text: "Shakespeare" }))),
      h("div", { class: "paper-body" }, DIARY.map(([when, text]) => [h("span", { class: "when", text: when + "\n" }), text + "\n"])),
      h("figcaption", { class: "paper-foot small muted", text: "A real run of this app: the smallest brain size, on a laptop-class processor with no graphics card." }));
  }

  function viewHome() {
    const hw = S.app.hw;
    const models = S.app.models || [];
    const list = models.length ? h("ul", { class: "ailist" }, models.map(aiRow)) :
      h("p", { class: "note", text: "Nothing here yet. Your first AI takes about five minutes to make." });
    return h("main", { class: "stack-lg" },
      h("div", { class: "hero" },
        h("div", { class: "stack" },
          h("h1", { text: "Teach a computer to write" }),
          h("p", { class: "lead", text: "Pick some text for it to read, press Start, and watch it learn, right here on your own computer. No coding, no accounts, nothing to pay." }),
          h("div", { class: "btnrow" }, h("button", { class: "btn primary", onclick: () => { S.draft = newDraft(); S.tab = "starter"; go("data"); }, text: "Make a new AI" })),
          h("p", { class: "note" }, h("b", { text: "Your computer: " }), hw ? hw.summary + "." : "having a look…",
            hw && hw.device === "cpu" && !hw.problem ? " That is fine for small AIs; bigger ones want a graphics card." : ""),
          hw && hw.problem ? h("p", { class: "note warn", text: hw.problem }) : null),
        diarySheet()),
      h("div", { class: "stack" }, h("h2", { text: "Your AIs" }), list));
  }
  function aiSentence(m) {
    const preset = m.preset ? m.preset[0].toUpperCase() + m.preset.slice(1) : "Custom";
    const st = m.stats || {};
    let s = `${preset} brain (${big(m.params)} connections), taught with ${m.dataset_name || "its text"}`;
    if (st.elapsed) s += ` for ${dur(st.elapsed)}`;
    return s + ".";
  }
  function aiRow(m) {
    const j = job();
    const live = j && j.kind === "train" && j.run === m.slug;
    const st = m.stats || {};
    const stage = st.stage ? `Reached stage ${st.stage} of 5: ${STAGES[st.stage - 1][0]}.` : (live ? "Just getting started." : "Not trained yet.");
    return h("li", {},
      h("div", {},
        h("div", { class: "name" }, m.name, live ? h("span", { class: "pill live", text: "Training now" }) : null),
        h("div", { class: "sub", text: aiSentence(m) }),
        h("div", { class: "sub muted", text: stage })),
      h("div", { class: "btnrow" },
        live ? h("button", { class: "btn", onclick: () => go("train", m.slug), text: "Watch it learn" })
             : h("button", { class: "btn", disabled: !m.has_model, onclick: () => go("ai", m.slug), text: m.chat ? "Chat with it" : "Try it" })));
  }

  // ------------------------------------------------------------------ step 1: text to read
  const AMOUNTS = [
    { tokens: 3e6, name: "A taste", note: "about 12 MB" },
    { tokens: 3e7, name: "A good chunk", note: "about 120 MB" },
    { tokens: 1.5e8, name: "A lot", note: "about 600 MB" },
    { tokens: 6e8, name: "A huge amount", note: "about 2.4 GB" },
  ];
  function amountChooser(approxTokens) {
    const d = S.draft;
    const known = approxTokens > 0;
    if (known && approxTokens <= 2e7) { d.amount = 0; return null; }  // small: just take all of it
    const opts = AMOUNTS.filter(a => !known || a.tokens < approxTokens * 0.9);
    if (!known || approxTokens <= 2e9) opts.push({ tokens: 0, name: "All of it", note: known ? "about " + mb(approxTokens * 4.2) : "however big it is" });
    if (d.amount == null || !opts.some(o => o.tokens === d.amount)) {
      const cap = S.app.default_tokens || 1e7;
      const fit = opts.filter(o => o.tokens && o.tokens <= cap);
      d.amount = fit.length ? fit[fit.length - 1].tokens : opts[0].tokens;
    }
    return h("div", { class: "stack" },
      h("h3", { text: "How much of it?" }),
      h("div", { class: "seg", role: "group", "aria-label": "How much of it" }, opts.map(o =>
        h("button", { type: "button", "aria-pressed": String(d.amount === o.tokens), onclick: () => { d.amount = o.tokens; render(); } },
          o.name, h("small", { text: o.note })))),
      h("p", { class: "small muted", text: "More text makes a better AI, but takes longer to download and to learn from. Starting small is fine: you can always come back for more." }));
  }

  function viewData() {
    const d = S.draft, j = job();
    const head = [steps(1), h("h1", { text: "What should it read?" }),
      h("p", { class: "lead", text: "An AI learns to write by reading. Whatever you give it here is what it will sound like." })];
    if (j && j.kind === "prepare") {
      const p = S.prep || j.prep || { frac: 0, msg: "Starting…" };
      R.prepBar = h("i", { style: `width:${Math.round(p.frac * 100)}%` });
      R.prepMsg = h("p", { class: "muted", text: p.msg });
      R.prepStage = h("h2", { text: prepTitle(p.stage) });
      return h("main", { class: "narrow stack" }, head,
        h("div", { class: "sheet stack" }, R.prepStage,
          h("div", { class: "bar", role: "progressbar", "aria-label": "Progress" }, R.prepBar), R.prepMsg,
          h("div", { class: "btnrow" }, h("button", { class: "btn quiet", onclick: () => act(() => post("/api/job/stop")), text: "Cancel" }))));
    }
    if (d.dataset && datasetBySlug(d.dataset)) return h("main", { class: "narrow stack" }, head, dataReady(datasetBySlug(d.dataset)));
    const tabs = [["starter", "Starter library"], ["hf", "Search Hugging Face"], ["own", "My own text"]];
    if ((S.app.datasets || []).length) tabs.push(["ready", `Already prepared (${S.app.datasets.length})`]);
    if (!tabs.some(t => t[0] === S.tab)) S.tab = "starter";
    const panel = { starter: tabStarter, hf: tabHF, own: tabOwn, ready: tabReady }[S.tab]();
    return h("main", { class: "narrow stack" }, head,
      S.prepError ? h("p", { class: "note bad", role: "alert", text: S.prepError }) : null,
      h("div", {},
        h("div", { class: "tabs", role: "tablist" }, tabs.map(([id, name]) =>
          h("button", { class: "tab", role: "tab", "aria-selected": String(S.tab === id), onclick: () => { S.tab = id; render(); }, text: name }))),
        h("div", { class: "tabpanel", role: "tabpanel" }, panel)));
  }
  function prepTitle(stage) {
    return { reading: "Reading the text", vocab: "Learning its vocabulary", tokenizing: "Chopping it into pieces", done: "Ready" }[stage] || "Getting your text ready";
  }
  function patchPrep() {
    if (!R.prepBar || !S.prep) return;
    R.prepBar.style.width = Math.round(S.prep.frac * 100) + "%";
    R.prepMsg.textContent = S.prep.msg;
    R.prepStage.textContent = prepTitle(S.prep.stage);
  }

  function advancedData() {
    const d = S.draft;
    return h("details", { class: "adv" }, h("summary", { text: "Advanced choices (you can ignore these)" }),
      h("div", { class: "stack" },
        h("label", { class: "field" }, h("span", { text: "How to chop the text into pieces (the “tokenizer”)" }),
          h("select", { onchange: e => { d.tokenizer = e.target.value; } },
            [["auto", "Decide for me (best)"], ["bytes", "Letter by letter"], ["bpe", "Learn word-pieces from my text"],
             ["tiktoken:gpt2", "Standard GPT-2 pieces (tiktoken)"], ["tiktoken:cl100k_base", "Standard GPT-4 pieces (tiktoken cl100k)"],
             ["tiktoken:o200k_base", "Standard GPT-4o pieces (tiktoken o200k)"]]
              .map(([v, n]) => h("option", { value: v, selected: d.tokenizer === v, text: n })))),
        h("p", { class: "small muted", text: "Left on “Decide for me”, small amounts of text are read letter by letter, and larger amounts get a vocabulary of word-pieces learned from the text itself." })));
  }
  async function startPrepare(source, name) {
    const d = S.draft;
    S.prepError = null;
    const ok = await act(() => post("/api/prepare", { name, source, tokenizer: d.tokenizer, max_tokens: d.amount || 0 }));
    if (ok) { await refresh(); render(); }
  }

  function tabStarter() {
    const d = S.draft;
    const out = [];
    // Without a graphics card the built-in Shakespeare is the quickest win; with one, TinyStories shines.
    const best = (S.app.hw || {}).device && S.app.hw.device !== "cpu" ? "tinystories" : "shakespeare";
    if (!d.kind && !d.touched && !(S.app.models || []).length) { d.kind = "starter"; d.starter = best; }
    d.touched = true;
    for (const g of S.app.groups) {
      const items = S.app.starter.filter(s => s.group === g.id);
      out.push(h("div", { class: "group-title", text: g.title }));
      out.push(h("ul", { class: "choices" }, items.map(s => {
        const on = d.kind === "starter" && d.starter === s.key;
        return h("li", { class: "choice" + (on ? " on" : "") },
          h("label", {},
            h("input", { type: "radio", name: "starter", checked: on, onchange: () => { d.kind = "starter"; d.starter = s.key; d.amount = null; render(); } }),
            h("div", {}, h("div", { class: "t" }, h("span", { text: s.title }), s.key === best ? h("span", { class: "tag star", text: "Best first choice" }) : null),
              h("div", { class: "d", text: s.blurb })),
            h("div", { class: "meta" }, s.size, h("br"), s.licence)),
          on && s.languages ? h("div", { class: "choice-body" }, h("span"), h("label", { class: "field" }, h("span", { text: "Which language?" }),
            h("select", { onchange: e => { d.lang = e.target.value; } }, S.app.wiki_langs.map(([code, name]) =>
              h("option", { value: code, selected: d.lang === code, text: name }))))) : null);
      })));
    }
    const sel = d.kind === "starter" ? S.app.starter.find(s => s.key === d.starter) : null;
    out.push(h("div", { class: "stack", style: "margin-top:22px" },
      sel ? amountChooser(sel.approx_tokens) : null,
      advancedData(),
      h("div", { class: "actionbar" }, h("span", { class: "picked", text: sel ? "Chosen: " + sel.title : "Pick one from the list." }),
        h("button", { class: "btn primary", disabled: !sel, text: "Get this text ready", onclick: () => {
        const src = JSON.parse(JSON.stringify(sel.source));
        let name = sel.title;
        if (sel.languages) { src.config = src.config.replace("{lang}", d.lang); name = "Wikipedia (" + (S.app.wiki_langs.find(l => l[0] === d.lang) || ["", d.lang])[1] + ")"; }
        startPrepare(src, name);
      } }))));
    return out;
  }

  function tabHF() {
    const d = S.draft, H = S.hf;
    const search = async () => {
      if (!H.q.trim()) return;
      H.busy = true; H.error = null; H.info = null; H.all = false; render();
      // a pasted link or an exact "owner/name": go straight to that collection
      const direct = H.q.trim().match(/^(?:https?:\/\/huggingface\.co\/datasets\/)?([\w.-]+\/[\w.-]+)\/?(?:[?#].*)?$/);
      try {
        H.results = direct ? [] : (await get("/api/hf/search", { q: H.q })).results;
        if (direct) H.results = [{ id: direct[1], downloads: null, description: "The collection you named.", license: null, locked: false }];
      } catch (e) { H.error = e.message; H.results = null; }
      H.busy = false; render();
      if (direct) inspect(direct[1]);
    };
    const inspect = async (id, config, split) => {
      H.infoBusy = id; H.error = null; render();
      try { H.info = await get("/api/hf/inspect", { id, config: config || "", split: split || "" }); d.kind = "hf"; d.amount = null; }
      catch (e) { H.error = e.message; H.info = null; }
      H.infoBusy = false; render();
      const el = document.getElementById("hfinfo"); if (el) el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
    };
    const out = [
      h("p", { style: "margin-top:16px", text: "Hugging Face is the internet's biggest free library of text collections for teaching AIs (they call them “datasets”). Search it from here. There is nothing to sign up for." }),
      h("form", { class: "btnrow", style: "margin-top:12px", onsubmit: e => { e.preventDefault(); search(); } },
        h("input", { class: "input", style: "flex:1;min-width:200px", type: "search", placeholder: "For example: fairy tales, recipes, poems, python code", value: H.q,
          "aria-label": "Search Hugging Face", oninput: e => { H.q = e.target.value; } }),
        h("button", { class: "btn", type: "submit", text: H.busy ? "Searching…" : "Search" })),
      h("div", { class: "btnrow small", style: "margin-top:10px" }, h("span", { class: "muted", text: "Ideas:" }),
        ["stories", "fairy tales", "poems", "recipes", "jokes", "dialogue", "python code"].map(q =>
          h("button", { class: "linkbtn", type: "button", text: q, onclick: () => { H.q = q; search(); } }))),
      H.error ? h("p", { class: "note bad", role: "alert", style: "margin-top:14px", text: H.error }) : null,
    ];
    const info = H.info;
    const detail = () => {
      const cfg = (info.configs || []).find(c => c.name === info.config);
      const approx = info.bytes ? info.bytes / 4.2 : 0;
      return h("div", { id: "hfinfo", class: "choice-body detail", style: "cursor:default" }, h("span"), h("div", { class: "stack extra" },
        h("p", { text: info.schema.describe }),
        info.preview && info.preview.length ? h("div", { class: "stack" }, h("p", { class: "small muted", text: "A sample of what it will read:" }),
          h("div", { class: "preview", text: info.preview.slice(0, 2).join("\n\n———\n\n") })) : null,
        (info.configs || []).length > 1 ? h("label", { class: "field" }, h("span", { text: "This collection comes in several versions. Which one?" }),
          h("select", { onchange: e => inspect(info.id, e.target.value) }, info.configs.map(c => h("option", { value: c.name, selected: c.name === info.config, text: c.name })))) : null,
        cfg && cfg.splits.length > 1 ? h("label", { class: "field" }, h("span", { text: "Which part of it? (“train” is the main part)" }),
          h("select", { onchange: e => inspect(info.id, info.config, e.target.value) }, cfg.splits.map(sp => h("option", { value: sp, selected: sp === info.split, text: sp })))) : null,
        h("dl", { class: "kv" },
          h("dt", { text: "Size" }), h("dd", { text: info.bytes ? mb(info.bytes) + " of text" + (info.rows ? ", in " + big(info.rows) + " pieces" : "") : "not stated" }),
          h("dt", { text: "Licence" }), h("dd", { text: info.license || "not stated: check its page before sharing what you make" }),
          h("dt", { text: "Its page" }), h("dd", {}, h("a", { href: info.url, target: "_blank", rel: "noopener noreferrer", text: info.url }))),
        amountChooser(approx),
        advancedData(),
        h("div", { class: "btnrow end" }, h("button", { class: "btn primary", text: "Get this text ready", onclick: () =>
          startPrepare({ type: "hf", id: info.id, config: info.config, split: info.split, schema: info.schema.mode === "files" ? null : info.schema,
                         revision: info.sha, licence: info.license }, info.id.split("/").pop()) }))));
    };
    if (H.results) {
      const shown = H.all ? H.results : H.results.slice(0, 10);
      out.push(H.results.length ? h("ul", { class: "choices", style: "margin-top:16px" }, shown.map(r => {
        const on = info && info.id === r.id;
        return h("li", { class: "choice" + (on ? " on" : "") + (r.locked ? " dim" : "") },
          h("label", {},
            h("input", { type: "radio", name: "hf", checked: on, disabled: r.locked, onchange: () => inspect(r.id) }),
            h("div", {}, h("div", { class: "t" }, h("span", { text: r.id }), r.locked ? h("span", { class: "tag no", text: "Locked" }) : null,
                H.infoBusy === r.id ? h("span", { class: "tag", text: "Looking inside…" }) : null),
              h("div", { class: "d", text: r.description || "No description." })),
            r.downloads == null ? h("div", { class: "meta" }) :
              h("div", { class: "meta" }, big(r.downloads) + " downloads", h("br"), r.license ? "Licence: " + r.license : "No licence stated")),
          on ? detail() : null);
      })) : h("p", { class: "note", style: "margin-top:14px", text: "Nothing found for that. Try a simpler word." }));
      if (!H.all && H.results.length > shown.length)
        out.push(h("div", { class: "btnrow", style: "margin-top:12px" }, h("button", { class: "btn quiet", text: `Show ${H.results.length - shown.length} more`, onclick: () => { H.all = true; render(); } })));
    }
    return out;
  }

  function tabOwn() {
    const F = S.files, d = S.draft;
    const addFiles = fl => { for (const f of fl) if (f && f.size > 0 && !f.name.startsWith(".")) F.list.push(f); render(); };
    const walk = async entry => {  // a dropped folder: collect every file inside it
      if (entry.isFile) return [await new Promise((res, rej) => entry.file(res, rej))];
      const reader = entry.createReader(), all = [];
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) all.push(...await walk(e));
      }
      return all;
    };
    const total = F.list.reduce((a, f) => a + f.size, 0);
    const fileIn = h("input", { type: "file", multiple: true, class: "sr", tabindex: "-1", onchange: e => { addFiles(e.target.files); e.target.value = ""; } });
    const dirIn = h("input", { type: "file", webkitdirectory: true, class: "sr", tabindex: "-1", onchange: e => { addFiles(e.target.files); e.target.value = ""; } });
    const drop = h("div", { class: "drop",
      ondragover: e => { e.preventDefault(); drop.classList.add("over"); },
      ondragleave: () => drop.classList.remove("over"),
      ondrop: async e => {
        e.preventDefault(); drop.classList.remove("over");
        const items = [...(e.dataTransfer.items || [])].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
        if (items.length) { const got = []; for (const it of items) { try { got.push(...await walk(it)); } catch (err) { /* skip unreadable */ } } addFiles(got); }
        else addFiles(e.dataTransfer.files);
      } },
      h("p", {}, h("b", { text: "Drop files or a whole folder here" })),
      h("p", { class: "small muted", text: "Stories, notes, chat logs, ebooks, code… Text files, Word documents, PDFs, ebooks (EPUB), web pages, spreadsheets (CSV), JSON." }),
      h("div", { class: "btnrow", style: "justify-content:center;margin-top:12px" },
        h("button", { class: "btn", type: "button", onclick: () => fileIn.click(), text: "Choose files" }),
        h("button", { class: "btn", type: "button", onclick: () => dirIn.click(), text: "Choose a folder" })), fileIn, dirIn);
    const ready = F.list.length > 0 || F.text.trim().length >= 200 || F.folder.trim();
    const goBtn = h("button", { class: "btn primary", disabled: !ready || F.busy, text: F.busy ? `Copying files in… ${F.sent} of ${F.list.length}` : "Get this text ready", onclick: async () => {
      d.amount = 0; d.kind = "own";
      if (F.list.length) {
        F.busy = true; F.sent = 0; render();
        const session = "u" + Date.now().toString(36);
        try {
          for (const f of F.list) {
            const r = await fetch("/api/upload?" + new URLSearchParams({ session, name: f.name }), { method: "POST", headers: { "X-FP-Token": TOKEN }, body: f });
            if (!r.ok) { let j = {}; try { j = await r.json(); } catch (e) { /* ignore */ } throw new Error(j.error || "A file could not be copied in."); }
            F.sent++;
            const b = document.querySelector("#ownbtn"); if (b) b.textContent = `Copying files in… ${F.sent} of ${F.list.length}`;
          }
        } catch (e) { F.busy = false; toast(e.message, true); render(); return; }
        F.busy = false;
        const name = F.list.length === 1 ? F.list[0].name.replace(/\.[^.]+$/, "") : "My files";
        await startPrepare({ type: "files", session }, name);
        F.list = [];
      } else if (F.folder.trim()) {
        await startPrepare({ type: "folder", path: F.folder.trim() }, "My folder");
      } else {
        await startPrepare({ type: "text", text: F.text }, "My text");
      }
    } });
    goBtn.id = "ownbtn";
    return [
      h("div", { class: "stack", style: "margin-top:18px" }, drop,
        F.list.length ? h("div", {}, h("p", { class: "small" }, h("b", { text: `${F.list.length} file${F.list.length === 1 ? "" : "s"} chosen, ${mb(total)} in all. ` }),
            h("button", { class: "linkbtn", text: "Clear the list", onclick: () => { F.list = []; render(); } })),
          h("ul", { class: "filelist" }, F.list.slice(0, 200).map(f => h("li", {}, h("span", { text: f.webkitRelativePath || f.name }), h("span", { class: "muted", text: mb(f.size) }))))) : null,
        F.list.length ? null : h("label", { class: "field", style: "max-width:none" }, h("span", { text: "Or paste text here" }),
          h("textarea", { placeholder: "Paste at least a few pages of writing…", value: F.text, oninput: e => { F.text = e.target.value; goBtn.disabled = !(F.text.trim().length >= 200 || F.folder.trim()); } })),
        h("details", { class: "adv" }, h("summary", { text: "A very big folder? Point to it instead of copying it in" }),
          h("label", { class: "field", style: "max-width:none" }, h("span", { text: "The folder's location on this computer" }),
            h("input", { type: "text", placeholder: "For example  C:\\Users\\me\\Documents\\my-writing", value: F.folder, oninput: e => { F.folder = e.target.value; goBtn.disabled = !(F.list.length || F.text.trim().length >= 200 || F.folder.trim()); } }))),
        h("p", { class: "note", text: "Your files stay on this computer. Nothing is uploaded anywhere." }),
        advancedData(),
        h("div", { class: "btnrow end" }, goBtn)),
    ];
  }

  function tabReady() {
    const d = S.draft;
    return [h("p", { style: "margin-top:16px", text: "Text you got ready earlier. Pick one to teach another AI with it." }),
      h("ul", { class: "choices", style: "margin-top:10px" }, S.app.datasets.map(ds =>
        h("li", { class: "choice" },
          h("div", { class: "choice-body", style: "cursor:default" }, h("span"),
            h("div", {}, h("div", { class: "t", text: ds.name }),
              h("div", { class: "d", text: `${big(ds.n_train)} pieces of text. ${ds.source_label}.` })),
            h("div", { class: "btnrow" },
              h("button", { class: "btn", text: "Use this", onclick: () => { d.dataset = ds.slug; d.preset = null; go("brain"); } }),
              h("button", { class: "linkbtn danger", text: "Delete", onclick: async () => {
                if (!window.confirm(`Delete the prepared text “${ds.name}”? AIs already taught with it keep working, but cannot be trained further.`)) return;
                await act(() => post("/api/dataset/delete", { slug: ds.slug })); await refresh(); render(); } })))))) ];
  }

  function dataReady(ds) {
    const tok = ds.tokenizer || {};
    const skipped = ds.skipped || [];
    return h("div", { class: "stack" },
      h("p", { class: "note good" }, h("b", { text: "Ready. " }),
        `${big(ds.n_train)} pieces of text to learn from` + (ds.n_val ? `, plus ${big(ds.n_val)} held back to test it on text it has never seen.` : ". (Too little to hold any back for testing.)")),
      h("div", { class: "sheet stack" },
        h("h3", { text: ds.name }),
        h("dl", { class: "kv" },
          h("dt", { text: "From" }), h("dd", { text: ds.source_label }),
          h("dt", { text: "Amount" }), h("dd", { text: `${int(ds.n_chars)} letters in ${int(ds.n_docs)} piece${ds.n_docs === 1 ? "" : "s"}` + (ds.truncated ? " (the first part of the collection, as you asked)" : "") }),
          h("dt", { text: "How it reads" }), h("dd", { text: tok.why || "" }),
          h("dt", { text: "Kind" }), h("dd", { text: ds.chat ? "Conversations: this AI will learn to chat." : "Plain writing: this AI will learn to continue text." }),
          ds.licence ? [h("dt", { text: "Licence" }), h("dd", { text: ds.licence })] : null),
        ds.preview && ds.preview.length ? h("div", { class: "preview", text: ds.preview[0] }) : null,
        skipped.length ? h("p", { class: "note warn", text: `I could not read ${skipped.length} file${skipped.length === 1 ? "" : "s"}: ` + skipped.slice(0, 6).map(s => `${s[0]} (${s[1]})`).join(", ") + (skipped.length > 6 ? ", and more." : ".") }) : null,
        ds.n_train < 200000 ? h("p", { class: "note warn", text: "This is a small amount of text. It will work, but the AI will mostly memorise it. More text gives a much better result." }) : null),
      h("div", { class: "btnrow between" },
        h("button", { class: "btn quiet", text: "Choose different text", onclick: () => { S.draft.dataset = null; render(); } }),
        h("button", { class: "btn primary", text: "Next: choose a brain", onclick: () => go("brain") })));
  }

  // ------------------------------------------------------------------ step 2: brain and time
  /* Step 2, the hands-on way: the same cut-and-stick builder as the website, at full size.
     What is stacked here is sent as "architecture" and is exactly what gets trained. */
  let fitTimer = null;
  function brainBuilder(d, ds, preset, fromPreset) {
    const FD = window.FPDesign;
    const host = h("div", {}), note = h("p", { class: "note", role: "status", text: "Checking it against your computer…" });
    const code = h("b", { text: "" });
    const showFit = () => {
      const f = d.fit, bad = d.problem || (f && f.level === "no");
      note.className = "note" + (d.problem ? " warn" : !f ? "" : f.level === "no" ? " bad" : f.level === "slow" ? " warn" : " good");
      note.textContent = d.problem ? d.problem : !f ? "Checking it against your computer…"
        : f.level === "ok" ? `This will train on your computer. It has ${big(f.params)} connections, closest to the ${f.preset_name} size.`
        : f.level === "slow" ? `It will train, with a catch. ${f.note}` : `Too big for this computer. ${f.note}`;
      if (R.startBtn) R.startBtn.disabled = !!bad;
    };
    const check = () => {
      code.textContent = FD.code(d.architecture);
      d.fit = null; showFit();
      clearTimeout(fitTimer);
      if (d.problem) return;
      const asked = FD.code(d.architecture);
      fitTimer = setTimeout(async () => {
        try { const f = await post("/api/fit", { dataset: d.dataset, architecture: d.architecture }); if (S.draft === d && d.build && FD.code(d.architecture) === asked) { d.fit = f; showFit(); } }
        catch (e) { if (S.draft === d && FD.code(d.architecture) === asked) { d.problem = e.message; showFit(); } }
      }, 250);
    };
    const builder = window.FPBuilder.mount(host, { design: d.architecture, limits: FD.APP, vocab: ds.vocab_size,
      words: { unit: "piece of text", units: "pieces of text", count: "connections" },
      onChange: (design, reason, problem) => { d.architecture = design; d.problem = problem; check(); } });
    d.architecture = builder.get(); d.problem = builder.problem();
    check();
    const paste = h("input", { class: "input", type: "text", maxlength: "120", placeholder: "Built one on the website? Paste its code here", "aria-label": "Design code from the website" });
    const useCode = () => {
      const got = FD.parse(paste.value);
      if (!got) return toast("That is not a design code. They look like this: fp1-w64-h4-c64-tt", true);
      builder.set(FD.fix(got, FD.APP)); paste.value = ""; toast("Design loaded.");
    };
    return h("div", { class: "stack" }, host, note,
      h("form", { class: "btnrow", onsubmit: e => { e.preventDefault(); useCode(); } }, paste, h("button", { class: "btn quiet", type: "submit", text: "Use this code" })),
      h("div", { class: "btnrow between" },
        h("span", { class: "small muted" }, "This design’s code: ", code),
        h("button", { class: "linkbtn", type: "button", text: `Start again from the ${preset.name} size`, onclick: () => builder.set(fromPreset()) })));
  }

  function viewBrain() {
    const d = S.draft, ds = datasetBySlug(d.dataset), rec = S.rec;
    if (!rec) return h("main", { class: "narrow stack" }, steps(2), h("h1", { text: "How big a brain?" }), h("p", { class: "lead", text: "Checking what your computer can handle…" }));
    if (d.budget == null) d.budget = ds.n_train < 5e6 || (S.app.hw || {}).device === "cpu" ? 300 : 1800;
    const autoName = () => (d.build ? `${ds.name} My design` : `${ds.name} ${d.preset[0].toUpperCase() + d.preset.slice(1)}`).slice(0, 60);
    if (!d.aiName) d.aiName = autoName();
    const preset = rec.cards.find(c => c.id === d.preset) || rec.cards[0];
    const FD = window.FPDesign;
    // "Build it yourself" starts from the ready-made size that is ticked, as far as the builder goes
    const fromPreset = () => FD.fix({ width: Math.min(FD.HARD.width[1], preset.n_embd), heads: preset.n_head, context: Math.min(FD.HARD.context[1], preset.block_size),
      blocks: Array.from({ length: Math.min(FD.HARD.maxBlocks, preset.n_layer) }, () => ({ type: "transformer" })) }, FD.APP);
    const setMode = build => { const auto = d.aiName === autoName(); d.build = build; if (build && !d.architecture) d.architecture = fromPreset(); if (auto) d.aiName = ""; render(); };
    const advField = (key, label, ph, hint) => h("label", { class: "field" }, h("span", { text: label }),
      h("input", { type: "number", disabled: !!d.build && ["n_layer", "n_embd", "n_head", "block_size"].includes(key), step: "any", min: "0", placeholder: String(ph), value: d.adv[key] || "", oninput: e => { d.adv[key] = e.target.value; } }),
      hint ? h("span", { class: "small muted", style: "font-weight:400", text: hint }) : null);
    const hw = S.app.hw || {};
    return h("main", { class: "narrow stack-lg" },
      h("div", { class: "stack" }, steps(2), h("h1", { text: "How big a brain, and for how long?" }),
        h("p", { class: "lead" }, "It will learn from ", h("b", { text: ds.name }), ` (${big(ds.n_train)} pieces of text) on your ${hw.device === "cpu" ? "computer's main processor" : hw.gpu || "graphics card"}.`)),
      h("div", {}, h("h2", { text: "Brain size" }),
        h("p", { class: "muted", style: "margin:6px 0 12px", text: "Bigger brains can learn more, but need a stronger computer, more text and more time." }),
        h("div", { class: "seg", role: "group", "aria-label": "How to choose the brain", style: "margin-bottom:14px" },
          h("button", { type: "button", "aria-pressed": String(!d.build), onclick: () => setMode(false) }, "Pick a ready-made size", h("small", { text: "The easy way" })),
          h("button", { type: "button", "aria-pressed": String(!!d.build), onclick: () => setMode(true) }, "Build it yourself", h("small", { text: "Stack the blocks by hand" }))),
        d.build ? brainBuilder(d, ds, preset, fromPreset) : h("ul", { class: "choices" }, rec.cards.map(c => {
          const fit = rec.fits[c.id], on = d.preset === c.id, no = fit.level === "no";
          return h("li", { class: "choice" + (on ? " on" : "") + (no ? " dim" : "") },
            h("label", {},
              h("input", { type: "radio", name: "preset", checked: on, disabled: no, onchange: () => {
                const auto = d.aiName === `${ds.name} ${d.preset[0].toUpperCase() + d.preset.slice(1)}`.slice(0, 60);
                d.preset = c.id; if (auto) d.aiName = ""; render(); } }),
              h("div", {}, h("div", { class: "t" }, h("span", { text: c.name }),
                  c.id === rec.recommended ? h("span", { class: "tag star", text: "Recommended" }) : null,
                  no ? h("span", { class: "tag no", text: "Too big for this computer" }) : fit.level === "slow" && c.id !== rec.recommended ? h("span", { class: "tag slow", text: "Not ideal" }) : null),
                h("div", { class: "d", text: c.blurb + (fit.note ? " " + fit.note : "") })),
              h("div", { class: "meta", text: big(c.params) + " connections" })));
        }))),
      h("div", { class: "stack" }, h("h2", { text: "How long should it study?" }),
        h("div", { class: "seg", role: "group", "aria-label": "How long" }, S.app.budgets.map(b =>
          h("button", { type: "button", "aria-pressed": String(d.budget === b.seconds), onclick: () => { d.budget = b.seconds; render(); } }, b.name, h("small", { text: b.blurb })))),
        h("p", { class: "small muted", text: "You can stop early at any moment and nothing is lost, and you can always give it more time later. It also stops by itself if it starts memorising instead of learning." })),
      h("label", { class: "field" }, h("span", { text: "Give your AI a name" }),
        h("input", { type: "text", maxlength: "60", value: d.aiName, oninput: e => { d.aiName = e.target.value; } })),
      h("p", { class: "note" }, h("b", { text: "What to expect. " }),
        "A small AI picks up the style of what it reads: its words, its rhythm, its way of putting sentences together. It will not know facts or hold a real conversation. The chatbots you have used are around 100,000 times bigger and read a large part of the internet."),
      h("details", { class: "adv" }, h("summary", { text: "Advanced choices (you can ignore these)" }),
        h("div", { class: "stack" },
          h("p", { class: "small muted", text: d.build ? "Layers, width, heads and memory come from the blocks you stacked. The other choices below still apply." : "Leave a box empty to keep the setting that comes with the brain size you picked." }),
          h("div", { class: "grid2" },
            advField("n_layer", "Layers", preset.n_layer), advField("n_embd", "Width", preset.n_embd, "Must divide by the number of heads."),
            advField("n_head", "Attention heads", preset.n_head), advField("block_size", "How far back it looks (context)", preset.block_size, "In pieces of text."),
            advField("batch", "Batch size", preset.batch), advField("lr", "Learning speed (learning rate)", preset.lr),
            advField("dropout", "Dropout", rec.dropout, "0 to 0.5. Helps with small amounts of text.")),
          h("label", {}, h("input", { type: "checkbox", checked: d.autostop, onchange: e => { d.autostop = e.target.checked; } }), " Stop by itself when it starts memorising (recommended)"))),
      h("div", { class: "btnrow between" },
        h("button", { class: "btn quiet", text: "Back", onclick: () => { d.dataset = null; go("data"); } }),
        R.startBtn = h("button", { class: "btn primary", text: "Start training", disabled: !!d.build && !!(d.problem || (d.fit && d.fit.level === "no")), onclick: async () => {
          const overrides = {};
          for (const [k, v] of Object.entries(d.adv)) if (v !== "" && v != null && !isNaN(Number(v))) overrides[k] = Number(v);
          // a hand-built design borrows its learning speed and batch size from the nearest ready-made size
          const body = { dataset: d.dataset, preset: d.build && d.fit ? d.fit.preset : d.preset, name: d.aiName.trim(), budget_sec: d.budget, overrides, autostop: d.autostop };
          if (d.build) body.architecture = d.architecture;
          const r = await act(() => post("/api/train", body));
          if (r) { await refresh(); S.draft = null; go("train", r.run); }
        } })));
  }

  // ------------------------------------------------------------------ step 3: learning
  const SERIES = ["Text it studies", "Text it has never seen"];
  function viewTrain() {
    const L = S.live, cfg = (L && L.cfg) || modelBySlug(S.arg) || {};
    const running = L.status !== "ended";
    const name = cfg.name || S.arg;
    R.left = h("span", {});
    R.bar = h("i", {});
    R.barWrap = h("div", { class: "bar", role: "progressbar", "aria-label": "Training progress" }, R.bar);
    R.who = h("b", { text: name });
    R.when = h("b", { text: "" });
    R.body = h("div", { class: "paper-body", "aria-live": "off" });
    R.slider = h("input", { type: "range", min: "0", max: "0", value: "0", "aria-label": "Look back at earlier writing",
      oninput: e => { const i = Number(e.target.value); L.pick = i >= L.samples.length - 1 ? -1 : i; showSample(false); } });
    R.sliderNote = h("span", { class: "small muted" });
    R.stageBoxes = h("div", { class: "stage-boxes" }, STAGES.map(() => h("i")));
    R.stageNames = h("div", { class: "stage-names" }, STAGES.map(s => h("span", { text: s[0] })));
    R.stageTitle = h("h3", { text: "" });
    R.stageDesc = h("p", { class: "muted", text: "" });
    R.viz = h("div", { class: "viz" },
      h("div", { class: "viz-title", text: "How wrong its guesses are" }),
      h("div", { class: "viz-sub", text: "Lower is better. This number is called the “loss”." }),
      h("div", { class: "legend" }, h("span", {}, h("i", { style: "border-color:var(--series-1)" }), SERIES[0]),
        h("span", {}, h("i", { style: "border-color:var(--series-2)" }), SERIES[1])));
    R.table = h("tbody", {});
    R.facts = h("div", { class: "facts" });
    R.notices = h("ul", { class: "notices" });
    R.tryOut = h("div", { class: "preview hidden" });
    const tryIn = h("input", { class: "input", type: "text", maxlength: "300", placeholder: cfg.chat ? "Ask it something…" : "Start a sentence for it to finish…", "aria-label": "Test it yourself" });
    const tryForm = h("form", { class: "btnrow", onsubmit: async e => {
      e.preventDefault();
      if (L.tryBusy) return;
      const prompt = tryIn.value;
      L.tryBusy = true; L.tryText = ""; L.tryId = "try-" + Date.now();
      R.tryOut.classList.remove("hidden"); R.tryOut.replaceChildren(h("span", { class: "given", text: cfg.chat ? "" : prompt }), h("span", { class: "caret" }));
      const ok = await act(() => post("/api/generate", { slug: S.arg, prompt, id: L.tryId, max_new_tokens: (cfg.tokenizer || {}).kind === "bytes" ? 300 : 110 }));
      if (!ok) { L.tryBusy = false; R.tryOut.classList.add("hidden"); }
    } }, tryIn, h("button", { class: "btn", type: "submit", text: "Test it" }));

    const controls = running ? h("div", { class: "btnrow" },
      L.paused ? h("button", { class: "btn", text: "Carry on", onclick: () => act(() => post("/api/train/resume")) })
               : h("button", { class: "btn", text: "Pause", onclick: () => act(() => post("/api/train/pause")) }),
      h("button", { class: "btn", text: "Stop and save", onclick: () => act(async () => { await post("/api/job/stop"); toast("Stopping and saving…"); }) }),
      L.budget > 0 ? h("button", { class: "btn quiet", text: "Add 15 minutes", onclick: () => act(() => post("/api/train/add_time", { seconds: 900 })) }) : null) : null;

    let banner = null;
    if (!running) {
      const end = L.end || {};
      const st = (cfg.stats || {}).stage || end.stage;
      const why = { budget: "The time is up.", autostop: "It stopped by itself because it had started memorising. The best version was kept.", stopped: "You stopped it. Everything is saved.", crashed: "It stopped unexpectedly, but its last saved progress is kept.", error: "It ran into a problem." }[end.reason] || "Training has ended.";
      banner = h("div", { class: "note " + (end.reason === "error" || end.reason === "crashed" ? "bad" : "good"), role: "status" },
        h("p", {}, h("b", { text: "Finished. " }), why, st ? ` It reached stage ${st} of 5: ${STAGES[st - 1][0]}.` : ""),
        L.error ? h("p", { text: L.error }) : null,
        h("div", { class: "btnrow", style: "margin-top:12px" },
          h("button", { class: "btn primary", disabled: !(modelBySlug(S.arg) || {}).has_model, text: cfg.chat ? "Chat with it" : "Try it out", onclick: () => go("ai", S.arg) }),
          h("button", { class: "btn quiet", text: "Back to my AIs", onclick: () => go("home") })));
    }
    return h("main", { class: "stack" }, steps(3),
      h("div", { class: "btnrow between" }, h("h1", { text: running ? (L.paused ? "Paused: " : "Learning: ") + name : name }), R.left),
      running ? R.barWrap : null, banner,
      h("div", { class: "train-grid" },
        h("div", { class: "stack" },
          h("div", { class: "paper" },
            h("div", { class: "paper-head" }, h("span", {}, "Written by ", R.who), h("span", {}, "after ", R.when, " of study")),
            R.body,
            h("div", { class: "paper-foot" }, R.slider, R.sliderNote)),
          controls,
          h("div", { class: "stack" }, h("h3", { text: "Test it yourself" }),
            running ? tryForm : h("p", { class: "muted", text: "Training has ended. Use the button above to talk to the finished AI." }), R.tryOut)),
        h("div", { class: "stack" },
          h("div", { class: "sheet stage" }, R.stageTitle, R.stageBoxes, R.stageNames, R.stageDesc),
          R.viz,
          h("details", { class: "adv", style: "border:0;padding:0" }, h("summary", { class: "small", text: "Show the numbers" }),
            h("div", { style: "max-height:220px;overflow:auto" }, h("table", { class: "nums" },
              h("thead", {}, h("tr", {}, h("th", { text: "When" }), h("th", { text: SERIES[0] }), h("th", { text: SERIES[1] }))), R.table))),
          R.facts, R.notices)));
  }

  let typer = null;
  function showSample(animate) {
    const L = S.live;
    if (!R.body || !L) return;
    clearInterval(typer);
    const n = L.samples.length;
    if (!n) {
      R.body.replaceChildren(h("span", { class: "empty", text: L.status === "ended" ? "No writing was saved for this AI." : "Its first piece of writing will appear here in a few seconds. Expect gibberish at first: that is how every AI starts." }));
      R.when.textContent = "0 seconds"; R.slider.max = "0"; R.sliderNote.textContent = "";
      return;
    }
    const idx = L.pick < 0 ? n - 1 : Math.min(L.pick, n - 1);
    const sm = L.samples[idx];
    R.slider.max = String(n - 1); R.slider.value = String(idx);
    R.when.textContent = dur(sm.t);
    R.sliderNote.textContent = n > 1 ? (L.pick < 0 ? "This is its newest writing. Slide left to see how it wrote earlier." : `Earlier writing (${idx + 1} of ${n}). Slide right to come back to now.`) : "More writing will appear as it learns.";
    const given = sm.chat ? h("span", { class: "q", text: sm.prompt }) : h("span", { class: "given", text: sm.prompt });
    const out = h("span", {});
    R.body.replaceChildren(given, out);
    const text = tidy(sm.text);
    if (animate && !reduceMotion && L.pick < 0 && text.length) {
      let i = 0;
      const stepN = Math.max(2, Math.ceil(text.length / 45));
      typer = setInterval(() => { i += stepN; out.textContent = text.slice(0, i); if (i >= text.length) clearInterval(typer); }, 20);
    } else out.textContent = text;
  }

  function patchTrain(what) {
    const L = S.live;
    if (S.view !== "train" || !L || !R.body) return;
    const all = what === "all";
    if (all || what === "sample") showSample(what === "sample");
    if (all || what === "metrics") {
      const m = L.metrics;
      if (L.status !== "ended") {
        if (L.budget > 0 && m) {
          const frac = Math.min(1, m.session_elapsed / L.budget);
          R.bar.style.width = (frac * 100).toFixed(1) + "%";
          R.left.textContent = L.paused ? "Paused" : dur(L.budget - m.session_elapsed) + " left";
        } else if (m) { R.bar.style.width = "100%"; R.left.textContent = "Running until you stop it (" + dur(m.session_elapsed) + " so far)"; }
        else R.left.textContent = "Getting started…";
        document.title = (L.paused ? "Paused" : (L.budget > 0 && m ? dur(L.budget - m.session_elapsed) + " left" : "Training")) + " - " + ((L.cfg || {}).name || "Foolproof Trainer");
      } else { R.left.textContent = ""; document.title = "Finished - " + ((L.cfg || {}).name || "Foolproof Trainer"); }
      const facts = [];
      const st = (L.cfg || {}).stats || {};
      const stepN = m ? m.step : st.step, seen = m ? m.tokens_seen : st.tokens_seen;
      if (m && L.status !== "ended") facts.push([big(m.tps), "pieces read per second"]);
      if (stepN != null) facts.push([int(stepN), "learning steps"]);
      if (seen != null && L.start) facts.push([(seen / Math.max(1, L.start.n_train)).toFixed(seen / L.start.n_train < 10 ? 1 : 0) + " times", "through the whole text"]);
      else if (seen != null) facts.push([big(seen), "pieces read"]);
      if (L.cfg && L.cfg.params) facts.push([big(L.cfg.params), "connections in its brain"]);
      if (m && m.mem_gb) facts.push([m.mem_gb.toFixed(1) + " GB", "graphics memory used"]);
      R.facts.replaceChildren(...facts.map(([v, k]) => h("div", {}, h("b", { text: v }), h("span", { text: k }))));
      const now = Date.now();
      if ((all || now - (R.lastDraw || 0) > 2500) && !R.viz.matches(":hover")) { R.lastDraw = now; window.FPChart.draw(R.viz, L.m, L.e, SERIES); }
    }
    if (all || what === "eval") {
      const ev = L.ev;
      const stage = ev ? ev.stage : 0;
      [...R.stageBoxes.children].forEach((b, i) => b.classList.toggle("on", i < stage));
      [...R.stageNames.children].forEach((b, i) => b.classList.toggle("on", i === stage - 1));
      R.stageTitle.textContent = stage ? `Stage ${stage} of 5: ${STAGES[stage - 1][0]}` : "Stage 1 of 5: warming up";
      R.stageDesc.textContent = stage ? STAGES[stage - 1][1] : "The first check-up happens a few seconds in.";
      const rows = L.e.slice(-200).reverse().map(p => h("tr", {}, h("td", { text: window.FPChart.fmtWhen(p.t) }),
        h("td", { text: p.train != null ? p.train.toFixed(3) : "" }), h("td", { text: p.val != null ? p.val.toFixed(3) : "" })));
      R.table.replaceChildren(...rows);
      if (!all && !R.viz.matches(":hover")) { R.lastDraw = Date.now(); window.FPChart.draw(R.viz, L.m, L.e, SERIES); }
    }
    if (all || what === "notices") {
      R.notices.replaceChildren(...L.notices.slice(-4).map(n => h("li", { class: "note " + (n.level === "warn" ? "warn" : ""), text: n.msg })));
    }
  }

  function onGen(ev) {
    const L = S.live, C = S.chat;
    if (L && ev.id && ev.id === L.tryId && R.tryOut) {
      if (ev.type === "gen") {
        L.tryText += ev.text;
        const kids = R.tryOut.children;
        if (kids.length >= 2) kids[kids.length - 1].before(document.createTextNode(tidy(ev.text)));
      } else {
        L.tryBusy = false;
        const c = R.tryOut.querySelector(".caret"); if (c) c.remove();
        if (!L.tryText.trim()) R.tryOut.append(h("span", { class: "muted", text: String(ev.reason || "").startsWith("error") ? ev.reason.slice(7) : "(It wrote nothing this time. Try again.)" }));
      }
      return;
    }
    if (ev.id && ev.id === C.id) {
      const node = document.getElementById("gen-live");
      if (ev.type === "gen") {
        C.pending.text += tidy(ev.text);
        if (node) node.textContent = C.pending.text;
        const box = document.getElementById("msgs"); if (box) box.scrollTop = box.scrollHeight;
      } else {
        C.busy = false; C.id = null;
        if (!C.pending.text.trim()) C.pending.text = String(ev.reason || "").startsWith("error") ? ev.reason.slice(7) : "(It wrote nothing this time. Try again.)";
        C.pending.done = true;
        if (S.view === "ai") render();
      }
    }
  }
  function chatFailed(msg) {
    const C = S.chat;
    if (!C.busy) return;
    C.busy = false; C.id = null;
    if (C.pending) { C.pending.text = msg; C.pending.done = true; }
    if (S.view === "ai") render();
  }

  // ------------------------------------------------------------------ step 4: chat, and looking after an AI
  function viewAI() {
    const m = S.model, C = S.chat, st = m.stats || {};
    const j = job();
    const live = j && j.kind === "train" && j.run === m.slug;
    const bytes = (m.tokenizer || {}).kind === "bytes";
    if (!C.len) C.len = bytes ? 400 : 150;
    const words = Math.round(C.len * (m.bytes_per_token || (bytes ? 1 : 4)) / 5.5);
    const send = async text => {
      if (C.busy || (!text.trim() && m.chat)) return;
      const history = m.chat ? C.msgs.filter(x => x.done).map(x => [x.role === "ai" ? "assistant" : "user", x.text]).slice(-12) : [];
      C.id = "chat-" + Date.now(); C.busy = true;
      C.pending = { role: "ai", text: "", done: false, given: m.chat ? "" : text };
      if (m.chat) C.msgs.push({ role: "user", text, done: true });
      C.msgs.push(C.pending);
      render();
      const ok = await act(() => post("/api/generate", { slug: m.slug, id: C.id, prompt: text, history, mode: m.chat ? "chat" : "text", max_new_tokens: C.len, temperature: C.temp }));
      if (!ok) { C.busy = false; C.msgs.pop(); if (m.chat) C.msgs.pop(); render(); }
    };
    const box = h("textarea", { placeholder: m.chat ? "Say something to it…" : "Start a piece of writing and it will carry on…", "aria-label": m.chat ? "Your message" : "Your opening words",
      onkeydown: e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const v = box.value; box.value = ""; send(v); } } });
    const msgs = h("div", { class: "msgs", id: "msgs", "aria-live": "polite" },
      C.msgs.length ? C.msgs.map(x => {
        if (x.role === "user") return h("div", { class: "msg user", text: x.text });
        const el = h("div", { class: "msg ai" });
        if (x.given) el.append(h("span", { class: "muted", text: x.given }));
        const sp = h("span", { text: x.text }); if (!x.done) { sp.id = "gen-live"; sp.className = "caret"; }
        el.append(sp);
        return el;
      }) : h("p", { class: "muted", text: m.chat ? "Say hello. Keep it simple: it is a very small AI." : "Type the first few words of something, and it will carry on in the style it learned." }));
    const keep = { sec: 1800 };
    const share = { repo: m.slug, token: "", private: false, busy: false };
    const shareOut = h("p", { class: "small" });
    const main = h("main", { class: "stack" }, steps(4),
      h("div", { class: "stack" }, h("h1", { text: m.name }),
        h("p", { class: "lead", text: aiSentence(m) + (st.stage ? ` Stage ${st.stage} of 5: ${STAGES[st.stage - 1][0]}.` : "") })),
      live ? h("p", { class: "note warn" }, "This AI is training right now. ", h("button", { class: "linkbtn", text: "Watch it learn", onclick: () => go("train", m.slug) })) : null,
      h("div", { class: "two" },
        h("div", { class: "chat" }, msgs,
          h("div", { class: "composer" }, box,
            C.busy ? h("button", { class: "btn", text: "Stop", onclick: () => act(() => post("/api/generate/stop")) })
                   : h("button", { class: "btn primary", text: m.chat ? "Send" : "Carry on", onclick: () => { const v = box.value; box.value = ""; send(v); } })),
          h("div", { class: "sliders" },
            h("label", { class: "slider" }, h("span", {}, "Creativity", h("b", { text: C.temp.toFixed(1) })),
              h("input", { type: "range", min: "0.1", max: "1.5", step: "0.1", value: String(C.temp), oninput: e => { C.temp = Number(e.target.value); e.target.previousSibling.lastChild.textContent = C.temp.toFixed(1); } }),
              h("small", {}, h("span", { text: "Careful" }), h("span", { text: "Wild" }))),
            h("label", { class: "slider" }, h("span", {}, "Length of reply", h("b", { text: "about " + words + " words" })),
              h("input", { type: "range", min: bytes ? "100" : "30", max: bytes ? "2000" : "600", step: "10", value: String(C.len), oninput: e => { C.len = Number(e.target.value); e.target.previousSibling.lastChild.textContent = "about " + Math.round(C.len * (m.bytes_per_token || (bytes ? 1 : 4)) / 5.5) + " words"; } }),
              h("small", {}, h("span", { text: "Short" }), h("span", { text: "Long" })))),
          C.msgs.length ? h("div", { class: "btnrow" }, h("button", { class: "linkbtn", text: "Start again with a clean page", onclick: () => { if (!C.busy) { C.msgs = []; render(); } } })) : null,
          h("p", { class: "note small", text: "It makes things up. Treat what it writes as style practice, never as facts or advice." })),
        h("div", { class: "sidecard" },
          h("div", { class: "sheet stack" }, h("h3", { text: "Teach it some more" }),
            h("p", { class: "small muted", text: m.can_resume ? "It carries on from where it stopped." : "Its prepared text has been deleted, so it cannot be trained further." }),
            h("select", { class: "input", "aria-label": "How much longer", onchange: e => { keep.sec = Number(e.target.value); } },
              S.app.budgets.map(b => h("option", { value: String(b.seconds), selected: b.seconds === 1800, text: b.name }))),
            h("button", { class: "btn", disabled: !m.can_resume || !!live, text: "Keep training", onclick: async () => {
              const r = await act(() => post("/api/train/continue", { slug: m.slug, budget_sec: keep.sec }));
              if (r) { await refresh(); go("train", m.slug); } } }),
            h("button", { class: "linkbtn", style: "justify-self:start", text: "See how it learned (chart and old writing)", onclick: () => { S.live = null; go("train", m.slug); } })),
          h("div", { class: "sheet stack" }, h("h3", { text: "Keep it or share it" }),
            h("p", { class: "small muted", text: "Makes a folder with the AI and a small script, so it runs without this app." }),
            h("div", { class: "btnrow" },
              h("button", { class: "btn", text: "Save a copy", onclick: async () => { toast("Packing it up…"); const r = await act(() => post("/api/model/export", { slug: m.slug })); if (r) toast("Saved in: " + r.path); } }),
              h("a", { class: "btn quiet", href: "/api/download?" + new URLSearchParams({ slug: m.slug, t: TOKEN }), text: "Download as a zip" })),
            h("details", { class: "adv" }, h("summary", { text: "Put it on Hugging Face" }),
              h("div", { class: "stack" },
                h("p", { class: "small" }, "Hugging Face gives every AI a free web page where others can download it. You need a free account and a “token” (a password for apps) with write permission: ",
                  h("a", { href: "https://huggingface.co/settings/tokens", target: "_blank", rel: "noopener noreferrer", text: "make one here" }), "."),
                h("label", { class: "field" }, h("span", { text: "Your token" }), h("input", { type: "password", autocomplete: "off", placeholder: S.app.settings.has_hf_token ? "(saved: leave empty to use it)" : "hf_…", oninput: e => { share.token = e.target.value; } })),
                h("label", { class: "field" }, h("span", { text: "Name for its page" }), h("input", { type: "text", value: share.repo, oninput: e => { share.repo = e.target.value; } })),
                h("label", {}, h("input", { type: "checkbox", onchange: e => { share.private = e.target.checked; } }), " Keep the page private"),
                h("button", { class: "btn", text: "Upload it", onclick: async e => {
                  if (share.busy) return;
                  share.busy = true; e.target.textContent = "Uploading…"; shareOut.textContent = "";
                  const r = await act(() => post("/api/model/upload", { slug: m.slug, repo: share.repo, token: share.token, private: share.private, remember: true }));
                  share.busy = false; e.target.textContent = "Upload it";
                  if (r) shareOut.replaceChildren("Done. Its page: ", h("a", { href: r.url, target: "_blank", rel: "noopener noreferrer", text: r.url }));
                } }), shareOut))),
          h("div", { class: "sheet stack" }, h("h3", { text: "About this AI" }),
            h("dl", { class: "kv" },
              h("dt", { text: "Brain" }), h("dd", { text: `${m.model.n_layer} layers, width ${m.model.n_embd}, looks back ${m.model.block_size} pieces` }),
              h("dt", { text: "Read" }), h("dd", { text: m.dataset_label || m.dataset_name || "" }),
              st.tokens_seen ? [h("dt", { text: "Practice" }), h("dd", { text: `${int(st.step || 0)} steps, ${big(st.tokens_seen)} pieces read` })] : null,
              st.best_val ? [h("dt", { text: "Best score" }), h("dd", { text: `${st.best_val} on unseen text (lower is better)` })] : null),
            h("div", { class: "btnrow" },
              h("button", { class: "linkbtn", text: "Open its folder", onclick: () => act(() => post("/api/open", { what: "model", slug: m.slug })) }),
              h("button", { class: "linkbtn", text: "Rename", onclick: async () => {
                const name = window.prompt("New name for this AI:", m.name);
                if (!name || !name.trim()) return;
                if (await act(() => post("/api/model/rename", { slug: m.slug, name: name.trim() }))) { await refresh(); S.model = modelBySlug(m.slug); render(); } } }),
              h("button", { class: "linkbtn danger", text: "Delete", onclick: async () => {
                if (!window.confirm(`Delete “${m.name}” for good? This cannot be undone.`)) return;
                if (await act(() => post("/api/model/delete", { slug: m.slug }))) { await refresh(); go("home"); } } }))))));
    setTimeout(() => { const b = document.getElementById("msgs"); if (b) b.scrollTop = b.scrollHeight; }, 0);
    return main;
  }

  // ------------------------------------------------------------------ words explained
  const WORDS = [
    ["AI, model, brain", "The thing you are making: a program that has learned, from reading, to guess what comes next in a piece of text.", "The real word is “language model”."],
    ["Connections", "The millions of adjustable numbers inside the brain. Learning means nudging them until its guesses get better.", "The real word is “parameters” or “weights”."],
    ["Text to read", "The writing the AI learns from. It will sound like whatever you give it.", "The real word is “dataset”."],
    ["Hugging Face", "A free website where people share text collections and finished AIs. This app can search it and fetch text from it for you.", "huggingface.co"],
    ["Pieces", "Before reading, text is chopped into small pieces: single letters for small amounts of text, parts of words for bigger amounts.", "The real word is “tokens”; the chopper is a “tokenizer”."],
    ["Training", "Showing the AI text again and again while it guesses the next piece each time and gets corrected.", "One correction is a “step”."],
    ["How wrong its guesses are", "A score for how surprised it was by the real next piece. Lower is better. Around 5 or more is random guessing.", "The real word is “loss”."],
    ["Text it has never seen", "A slice of your text is hidden during training and used only for testing, to check it is really learning and not just memorising.", "The real word is “validation”."],
    ["Memorising", "When the brain is big and the text is small, it learns the text by heart instead of learning how the language works. The app notices and stops.", "The real word is “overfitting”."],
    ["Graphics card", "The part of a computer made for games is also very good at training AIs: often 20 to 100 times faster than the main processor.", "Also called a “GPU”. NVIDIA cards and Apple Silicon Macs work here."],
    ["How far back it looks", "How many pieces of text it can keep in view while guessing the next one.", "The real word is “context length”."],
    ["Creativity", "How adventurous it is when choosing each next piece. Low gives safe, repetitive writing; high gives surprising, messier writing.", "The real word is “temperature”."],
    ["Transformer", "The design of the brain: the same design behind ChatGPT and friends, only far smaller here.", "The “T” in GPT."],
    ["Blocks", "The brain is a stack of blocks. Text goes in at the top, passes through each block in turn, and a guess at the next piece comes out at the bottom. With “Build it yourself” you choose the blocks and their order.", "The real word for the whole stack is the “architecture”."],
    ["Attention", "A block that looks back over everything read so far and picks out the pieces that matter for the next guess. Without one, the brain can only see the piece it is on.", "The real word is “self-attention”. “Heads” are how many things it looks for at once."],
    ["Feed-forward", "A block that works something out from what attention found. Most of what the brain learns is kept in these.", "Also called an “MLP”."],
    ["Transformer layer", "One Attention block followed by one Feed-forward block: the pair that every modern language AI repeats many times over.", "Also called a “transformer block”."],
    ["Width", "How many numbers the brain uses to describe each piece of text on its way down the stack. Wider can hold more, and costs more time and memory.", "The real words are “embedding size” or “hidden size”."],
  ];
  function glossary() {
    const close = () => { S.glossary = false; render(); };
    return [h("div", { class: "scrim", onclick: close }),
      h("aside", { class: "drawer", role: "dialog", "aria-label": "Words explained" },
        h("div", { class: "btnrow between" }, h("h2", { text: "Words explained" }), h("button", { class: "btn quiet", onclick: close, text: "Close" })),
        h("dl", { class: "gloss" }, WORDS.map(([term, plain, real]) => [h("dt", { text: term }), h("dd", {}, plain, h("em", { text: real }))])))];
  }
  document.addEventListener("keydown", e => { if (e.key === "Escape" && S.glossary) { S.glossary = false; render(); } });

  const VIEWS = { home: viewHome, data: viewData, brain: viewBrain, train: viewTrain, ai: viewAI };

  // ------------------------------------------------------------------ go
  (async function boot() {
    try { await refresh(); } catch (e) {
      root.replaceChildren(h("main", { class: "narrow stack" }, h("h1", { text: "Could not start" }), h("p", { class: "lead", text: e.message })));
      return;
    }
    await route();
    poll();
  })();
})();
