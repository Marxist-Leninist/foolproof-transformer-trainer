/* Foolproof Trainer - the block builder.
   A cut-and-stick worksheet: pick up a block, drop it on the sheet, and the stack you make
   IS the network that gets trained (design.js says what a design is; the app's trainer and
   the website's in-page engine both build exactly that stack).

     const b = FPBuilder.mount(host, { design, limits, vocab, words, onChange });
     b.get()  ->  the design      b.set(design)      b.problem()  ->  "" when ready      b.destroy()

   Mouse: drag a block anywhere. Touch: drag by the dotted handle (so the page still scrolls).
   Everything can also be done without dragging: tap a cut-out to add it, and each block has
   up, down and remove buttons. No framework; text goes on the page with textContent only. */
"use strict";
(function () {
  const D = window.FPDesign;
  const SVG = "http://www.w3.org/2000/svg";

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "value") el.value = v;
      else if (k === "disabled" || k === "selected") el[k] = !!v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) el.append(k.nodeType ? k : document.createTextNode(String(k)));
    return el;
  }
  function svg(paths, box) {
    const el = document.createElementNS(SVG, "svg");
    el.setAttribute("viewBox", box || "0 0 32 32"); el.setAttribute("aria-hidden", "true"); el.setAttribute("focusable", "false");
    for (const [tag, attrs] of paths) {
      const p = document.createElementNS(SVG, tag);
      for (const [k, v] of Object.entries(attrs)) p.setAttribute(k, v);
      el.append(p);
    }
    return el;
  }
  const ICON = {
    // the newest letter looking back at two earlier ones
    attention: () => svg([
      ["path", { d: "M26 19C26 6 6 6 6 17", fill: "none" }], ["path", { d: "M2.6 13.4 6 18l3.6-4.4", fill: "none" }],
      ["path", { d: "M26 19c0-7-10-7-10-2", fill: "none" }], ["path", { d: "M12.9 13.9 16 18l3.3-3.9", fill: "none" }],
      ["circle", { cx: 6, cy: 24, r: 3 }], ["circle", { cx: 16, cy: 24, r: 3 }], ["circle", { cx: 26, cy: 24, r: 3, class: "solid" }]]),
    // narrow, wide, narrow: it spreads the numbers out to think, then packs them back
    feed_forward: () => svg([
      ["path", { d: "M5 12 16 4l11 8v8l-11 8-11-8z", fill: "none" }], ["path", { d: "M16 4v24M5 12l22 8M27 12 5 20", fill: "none", class: "thin" }]]),
    grip: () => svg([["circle", { cx: 5, cy: 4, r: 1.7 }], ["circle", { cx: 11, cy: 4, r: 1.7 }], ["circle", { cx: 5, cy: 10, r: 1.7 }],
      ["circle", { cx: 11, cy: 10, r: 1.7 }], ["circle", { cx: 5, cy: 16, r: 1.7 }], ["circle", { cx: 11, cy: 16, r: 1.7 }]], "0 0 16 20"),
    up: () => svg([["path", { d: "M5 14l7-7 7 7", fill: "none" }]], "0 0 24 24"),
    down: () => svg([["path", { d: "M5 10l7 7 7-7", fill: "none" }]], "0 0 24 24"),
    remove: () => svg([["path", { d: "M6 6l12 12M18 6 6 18", fill: "none" }]], "0 0 24 24"),
  };
  function big(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " billion";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + " million";
    return Math.round(n).toLocaleString("en");
  }
  const reduceMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function mount(host, opts) {
    opts = opts || {};
    const limits = opts.limits || D.WEB;
    const words = Object.assign({ unit: "letter", units: "letters" }, opts.words);
    let vocab = opts.vocab || 65;
    let design = D.fix(opts.design || D.classic(1, D.defaultHeads(limits.defaultWidth, limits), limits.defaultWidth, limits.defaultContext), limits);
    const twoIcons = () => h("span", { class: "bld-ico two" }, ICON.attention(), ICON.feed_forward());
    let drag = null, swallowClick = false, raf = 0;

    const live = h("div", { class: "sr", "aria-live": "polite" });
    const count = h("span", { class: "bld-count" });
    const say = h("p", { class: "bld-say" });
    const wrong = h("p", { class: "bld-wrong hidden", role: "status" });
    const stack = h("ol", { class: "bld-stack", "aria-label": "Your blocks, in the order the numbers pass through them" });
    const slot = h("div", { class: "bld-slot", "aria-hidden": "true" });
    const full = h("p", { class: "bld-full small hidden" });
    const bin = h("div", { class: "bld-bin", "aria-hidden": "true" }, ICON.remove(), h("span", { text: "Drop here to take it off" }));

    const piece = (kind, title, plain) => {
      const el = h("button", { type: "button", class: "bld-piece " + kind, draggable: "false", "data-kind": kind },
        kind === "transformer" ? twoIcons() : h("span", { class: "bld-ico" }, ICON[kind]()),
        h("span", { class: "bld-words" }, h("b", { text: title }), h("small", { text: plain })));
      el.addEventListener("click", () => { if (swallowClick) { swallowClick = false; return; } add(kind, design.blocks.length, true); });
      el.addEventListener("pointerdown", ev => begin(ev, el, { from: "tray", kind }));
      return el;
    };
    const tray = h("div", { class: "bld-tray" },
      h("p", { class: "bld-tray-title", text: "Blocks" }),
      h("p", { class: "bld-tray-how small", text: "Drag one onto the sheet, or tap it." }),
      h("div", { class: "bld-pieces" },
        D.TYPES.map(kind => piece(kind, D.KINDS[kind].name, D.KINDS[kind].plain))),
      full, bin);

    const pick = (list, value, label, onPick) => h("select", { class: "bld-pick", "aria-label": label, onchange: e => onPick(Number(e.target.value)) },
      list.map(v => h("option", { value: String(v), selected: v === value, text: String(v) })));
    const reader = h("div", { class: "bld-end bld-reader" });
    const writer = h("div", { class: "bld-end bld-writer" },
      h("b", { text: "Writer" }), h("span", { text: " turns the numbers back into a guess at the next " + words.unit + "." }));

    const sheet = h("div", { class: "paper bld-sheet" },
      h("div", { class: "paper-head" }, h("span", { text: "Worksheet: build a transformer" }), count),
      h("div", { class: "bld-body" }, reader, h("div", { class: "bld-rail" }, stack, slot), writer),
      h("div", { class: "paper-foot" }, say, wrong));
    const rootEl = h("div", { class: "bld" }, tray, sheet, live);
    host.replaceChildren(rootEl);

    // ---------------------------------------------------------------- drawing
    function blockEl(b, i) {
      const kind = D.KINDS[b.type], n = design.blocks.length;
      const li = h("li", { class: "bld-block " + b.type, tabindex: "0", "data-i": String(i), "data-kind": b.type,
        "aria-label": `${kind.name}, block ${i + 1} of ${n}` },
        h("span", { class: "bld-grip", title: "Drag to move" }, ICON.grip()),
        b.type === "transformer" ? twoIcons() : h("span", { class: "bld-ico" }, ICON[b.type]()),
        h("span", { class: "bld-words" }, h("b", { text: kind.name }), h("small", { text: kind.plain })),
        h("span", { class: "bld-btns" },
          h("button", { type: "button", class: "bld-btn", "aria-label": "Move up", "data-act": "up", disabled: i === 0, onclick: () => move(i, i - 1, "up") }, ICON.up()),
          h("button", { type: "button", class: "bld-btn", "aria-label": "Move down", "data-act": "down", disabled: i === n - 1, onclick: () => move(i, i + 1, "down") }, ICON.down()),
          h("button", { type: "button", class: "bld-btn", "aria-label": "Remove this block", "data-act": "remove", onclick: () => remove(i) }, ICON.remove())));
      li.addEventListener("pointerdown", ev => begin(ev, li, { from: "stack", index: i }));
      li.addEventListener("keydown", ev => {
        if (ev.target !== li) return;
        if ((ev.altKey || ev.ctrlKey) && ev.key === "ArrowUp") { ev.preventDefault(); if (i > 0) move(i, i - 1, "block"); }
        else if ((ev.altKey || ev.ctrlKey) && ev.key === "ArrowDown") { ev.preventDefault(); if (i < n - 1) move(i, i + 1, "block"); }
        else if (ev.key === "ArrowUp") { ev.preventDefault(); focusBlock(i - 1, "block"); }
        else if (ev.key === "ArrowDown") { ev.preventDefault(); focusBlock(i + 1, "block"); }
        else if (ev.key === "Delete" || ev.key === "Backspace") { ev.preventDefault(); remove(i); }
      });
      return li;
    }
    function draw() {
      reader.replaceChildren(
        h("b", { text: "Reader" }), h("span", { text: " turns each " + words.unit + " into " }),
        pick(limits.width, design.width, "Width: how many numbers describe each " + words.unit, v => { design.width = v; changed("width"); }),
        h("span", { text: " numbers and remembers the last " }),
        pick(limits.context, design.context, "Memory: how many " + words.units + " back it can see", v => { design.context = v; changed("context"); }),
        h("span", { text: " " + words.units + ". Attention looks for " }),
        pick(D.headsFor(design.width, limits), design.heads, "Heads: how many things attention looks for at once", v => { design.heads = v; changed("heads"); }),
        h("span", { text: design.heads === 1 ? " thing at a time." : " things at once." }));
      stack.replaceChildren(...(design.blocks.length ? design.blocks.map(blockEl)
        : [h("li", { class: "bld-empty", text: "Nothing here yet. Drag a block onto this spot, or tap one." })]));
      count.replaceChildren(h("b", { text: big(D.countParams(design, vocab)) }), " numbers to learn");
      say.textContent = D.describe(design);
      const atMax = design.blocks.length >= limits.maxBlocks, trouble = problem();
      for (const p of tray.querySelectorAll(".bld-piece")) p.disabled = atMax;
      full.textContent = atMax ? `${limits.maxBlocks} blocks is the most that fits here.` : "";
      full.classList.toggle("hidden", !atMax);
      wrong.textContent = trouble;
      wrong.classList.toggle("hidden", !trouble);
    }
    function focusBlock(i, what) {
      const li = stack.querySelector(`.bld-block[data-i="${Math.max(0, Math.min(design.blocks.length - 1, i))}"]`);
      if (!li) { const first = tray.querySelector(".bld-piece"); if (first) first.focus(); return; }
      const btn = what && what !== "block" ? li.querySelector(`[data-act="${what}"]`) : null;
      (btn && !btn.disabled ? btn : li).focus({ preventScroll: false });
    }
    function changed(reason) {
      design = D.fix(design, limits);
      draw();
      if (opts.onChange) opts.onChange(get(), reason, problem());
    }
    const get = () => JSON.parse(JSON.stringify(design));
    const problem = () => D.problem(design, limits, vocab);
    const name = b => D.KINDS[b.type].name;

    // ---------------------------------------------------------------- the four things you can do
    function add(kind, at, moveFocus) {
      if (design.blocks.length >= limits.maxBlocks) { live.textContent = `No room: ${limits.maxBlocks} blocks is the most that fits here.`; return; }
      design.blocks.splice(at, 0, { type: kind });
      changed("add");
      live.textContent = `${D.KINDS[kind].name} added at position ${at + 1} of ${design.blocks.length}.`;
      const li = stack.querySelector(`.bld-block[data-i="${at}"]`);
      if (li) { if (!reduceMotion()) li.classList.add("bld-landed"); if (moveFocus) li.scrollIntoView({ block: "nearest", behavior: reduceMotion() ? "auto" : "smooth" }); }
    }
    function move(from, to, focusWhat) {
      if (to < 0 || to >= design.blocks.length || to === from) return;
      const [b] = design.blocks.splice(from, 1);
      design.blocks.splice(to, 0, b);
      changed("move");
      live.textContent = `${name(b)} moved to position ${to + 1} of ${design.blocks.length}.`;
      if (focusWhat) focusBlock(to, focusWhat);
    }
    function remove(i) {
      const [b] = design.blocks.splice(i, 1);
      changed("remove");
      live.textContent = `${name(b)} removed. ${design.blocks.length} block${design.blocks.length === 1 ? "" : "s"} left.`;
      focusBlock(Math.min(i, design.blocks.length - 1), "block");
    }

    // ---------------------------------------------------------------- dragging
    function begin(ev, el, source) {
      if (drag || (ev.button != null && ev.button !== 0) || el.disabled) return;
      if (ev.target.closest("select, .bld-btn")) return;                            // the buttons on a block are not handles
      if (source.from === "stack" && ev.pointerType === "touch" && !ev.target.closest(".bld-grip")) return;   // let fingers scroll the page
      const box = el.getBoundingClientRect();
      const w = Math.min(box.width, 330);     // what you carry is card-sized, so you can see where it is going
      drag = { el, source, id: ev.pointerId, x0: ev.clientX, y0: ev.clientY, x: ev.clientX, y: ev.clientY,
               dx: Math.min(ev.clientX - box.left, w - 28), dy: ev.clientY - box.top, w, active: false, ghost: null, at: null, bin: false };
      try { el.setPointerCapture(ev.pointerId); } catch (e) { /* older browsers: events still arrive while over the element */ }
      el.addEventListener("pointermove", onMove);
      el.addEventListener("pointerup", onUp);
      el.addEventListener("pointercancel", cancel);
      document.addEventListener("keydown", onKey, true);
    }
    function onMove(ev) {
      if (!drag || ev.pointerId !== drag.id) return;
      drag.x = ev.clientX; drag.y = ev.clientY;
      if (!drag.active) {
        if (Math.hypot(drag.x - drag.x0, drag.y - drag.y0) < 6) return;
        drag.active = true;
        const g = drag.el.cloneNode(true);
        g.classList.add("bld-ghost"); g.removeAttribute("tabindex"); g.style.width = drag.w + "px";
        document.body.append(g); drag.ghost = g;
        drag.el.classList.add("bld-lifted"); rootEl.classList.add("bld-dragging");
        if (drag.source.from === "stack") rootEl.classList.add("bld-can-bin");
        document.documentElement.classList.add("bld-noselect");
        raf = requestAnimationFrame(edgeScroll);
      }
      ev.preventDefault();
      place();
    }
    function place() {
      const d = drag;
      d.ghost.style.transform = `translate(${Math.round(d.x - d.dx)}px, ${Math.round(d.y - d.dy)}px) rotate(-1.5deg)`;
      const area = sheet.getBoundingClientRect(), rail = stack.getBoundingClientRect();
      const overBin = d.source.from === "stack" && inside(tray.getBoundingClientRect(), d.x, d.y, 0);
      d.bin = overBin; bin.classList.toggle("over", overBin);
      let at = null;
      if (!overBin && inside(area, d.x, d.y, 36)) {
        const blocks = Array.from(stack.querySelectorAll(".bld-block"));
        at = 0;
        for (const b of blocks) { const r = b.getBoundingClientRect(); if (d.y > r.top + r.height / 2) at++; }
        // show the gap: between the two blocks it would land between
        let y;
        if (!blocks.length) y = rail.height / 2;
        else if (at === 0) y = blocks[0].getBoundingClientRect().top - rail.top - 6;
        else if (at === blocks.length) y = blocks[at - 1].getBoundingClientRect().bottom - rail.top + 6;
        else y = (blocks[at - 1].getBoundingClientRect().bottom + blocks[at].getBoundingClientRect().top) / 2 - rail.top;
        slot.style.top = Math.round(y) + "px";
      }
      d.at = at;
      const same = at != null && d.source.from === "stack" && (at === d.source.index || at === d.source.index + 1);
      slot.classList.toggle("on", at != null && !same);
    }
    const inside = (r, x, y, pad) => x >= r.left - pad && x <= r.right + pad && y >= r.top - pad && y <= r.bottom + pad;
    function edgeScroll() {
      if (!drag || !drag.active) return;
      // near the top or bottom of the screen the page creeps along, faster the closer to the edge
      const edge = 64, up = edge - drag.y, down = drag.y - (window.innerHeight - edge);
      if (up > 0) window.scrollBy(0, -Math.round(2 + 6 * Math.min(1, up / edge)));
      else if (down > 0) window.scrollBy(0, Math.round(2 + 6 * Math.min(1, down / edge)));
      place();
      raf = requestAnimationFrame(edgeScroll);
    }
    function onUp(ev) {
      if (!drag || ev.pointerId !== drag.id) return;
      const d = drag, was = d.active;
      end();
      if (!was) return;                        // it was a tap, not a drag: the click does the work
      swallowClick = true; setTimeout(() => { swallowClick = false; }, 0);
      if (d.source.from === "tray") { if (d.at != null) add(d.source.kind, d.at, false); }
      else if (d.bin) remove(d.source.index);
      else if (d.at != null) {
        const to = d.at > d.source.index ? d.at - 1 : d.at;
        if (to !== d.source.index) move(d.source.index, to, "block"); else focusBlock(to, "block");
      }
    }
    function onKey(ev) { if (ev.key === "Escape" && drag) { ev.preventDefault(); cancel(); } }
    function cancel() { if (drag) end(); }
    function end() {
      const d = drag; drag = null;
      cancelAnimationFrame(raf);
      d.el.removeEventListener("pointermove", onMove); d.el.removeEventListener("pointerup", onUp); d.el.removeEventListener("pointercancel", cancel);
      document.removeEventListener("keydown", onKey, true);
      try { d.el.releasePointerCapture(d.id); } catch (e) { /* already released */ }
      if (d.ghost) d.ghost.remove();
      d.el.classList.remove("bld-lifted");
      rootEl.classList.remove("bld-dragging", "bld-can-bin"); bin.classList.remove("over"); slot.classList.remove("on");
      document.documentElement.classList.remove("bld-noselect");
    }

    draw();
    return {
      get,
      problem,
      set(next, quiet) { design = D.fix(next, limits); draw(); if (!quiet && opts.onChange) opts.onChange(get(), "set", problem()); },
      setVocab(v) { vocab = v; draw(); },
      destroy() { if (drag) end(); host.replaceChildren(); },
    };
  }

  window.FPBuilder = { mount };
})();
