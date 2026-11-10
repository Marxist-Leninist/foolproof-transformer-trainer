/* Foolproof Trainer - what a "design" is.
   A design is the stack of blocks a person built. It is exactly what the app's trainer
   accepts as "architecture" (see foolproof/architecture.py):

     { version: 1, width: 128, heads: 4, context: 128,
       blocks: [ { type: "transformer" }, { type: "attention" }, { type: "feed_forward" } ] }

   Every block reads the running "stream" of numbers, works something out, and adds its
   answer back on (normalise -> block -> add). A "transformer" block is simply attention
   followed by feed-forward. Because of that, any order is a network that runs.

   No page, no network: just rules. Used by the builder, by the in-page engine and by tests. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.FPDesign = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  const KINDS = {
    transformer: { name: "Transformer layer", plain: "looks back, then thinks: the classic pair", letter: "t" },
    attention: { name: "Attention", plain: "looks back over what it has read", letter: "a" },
    feed_forward: { name: "Feed-forward", plain: "thinks about what it found", letter: "f" },
  };
  const TYPES = Object.keys(KINDS);

  /* What may be chosen. The page keeps to small numbers so a phone can train it;
     the app trains with PyTorch and goes as far as its own checker allows. */
  const WEB = {
    width: [32, 48, 64, 96, 128], heads: [1, 2, 4, 8], context: [32, 64, 128],
    minBlocks: 0, maxBlocks: 12, needAttention: false, defaultWidth: 64, defaultContext: 64,
  };
  const APP = {
    width: [32, 48, 64, 96, 128, 192, 256, 320, 384, 512, 640, 768], heads: [1, 2, 4, 6, 8, 12, 16],
    context: [32, 64, 128, 256, 512, 1024],
    minBlocks: 1, maxBlocks: 16, needAttention: true, defaultWidth: 128, defaultContext: 128,
  };
  /* The outer fence, the same numbers as architecture.py. */
  const HARD = { width: [32, 768], heads: [1, 16], context: [32, 1024], maxBlocks: 16, maxParams: 350e6 };

  const isInt = n => typeof n === "number" && isFinite(n) && Math.floor(n) === n;
  const looksBack = d => d.blocks.some(b => b.type !== "feed_forward");

  /* Heads that work for a width: they must divide it, and each head needs an even
     share (positions are marked by turning the numbers in pairs). */
  function headsFor(width, limits) {
    return ((limits && limits.heads) || APP.heads).filter(hd => width % hd === 0 && (width / hd) % 2 === 0);
  }
  function nearest(list, want) {
    let best = list[0];
    for (const v of list) if (Math.abs(v - want) < Math.abs(best - want)) best = v;
    return best;
  }
  function defaultHeads(width, limits) {
    const ok = headsFor(width, limits);
    return ok.length ? nearest(ok, Math.max(1, Math.round(width / 16))) : 1;
  }

  /* Strict: returns a clean design or throws a sentence a person can read. */
  function normalise(d) {
    if (!d || typeof d !== "object") throw new Error("That is not a design.");
    const width = d.width, heads = d.heads, context = d.context;
    for (const [name, v, lo, hi] of [["Width", width, HARD.width[0], HARD.width[1]], ["Heads", heads, HARD.heads[0], HARD.heads[1]], ["Memory (context)", context, HARD.context[0], HARD.context[1]]]) {
      if (!isInt(v) || v < lo || v > hi) throw new Error(`${name} must be a whole number from ${lo} to ${hi}.`);
    }
    if (width % heads || (width / heads) % 2) throw new Error(`Width ${width} cannot be shared out evenly between ${heads} heads. Pick another number of heads.`);
    if (!Array.isArray(d.blocks)) throw new Error("A design needs a list of blocks.");
    if (d.blocks.length > HARD.maxBlocks) throw new Error(`That is more than ${HARD.maxBlocks} blocks.`);
    const blocks = d.blocks.map(b => {
      if (!b || typeof b !== "object" || !TYPES.includes(b.type)) throw new Error("One of the blocks is not a kind this app knows.");
      return { type: b.type };
    });
    return { version: 1, width, heads, context, blocks };
  }

  /* Forgiving, for the builder: never throws, snaps anything odd to the nearest thing that works. */
  function fix(d, limits) {
    limits = limits || APP;
    d = d && typeof d === "object" ? d : {};
    const width = nearest(limits.width, Number(d.width) || limits.defaultWidth);
    const context = nearest(limits.context, Number(d.context) || limits.defaultContext);
    const ok = headsFor(width, limits);
    const heads = ok.includes(Number(d.heads)) ? Number(d.heads) : Number(d.heads) > 0 ? nearest(ok, Number(d.heads)) : defaultHeads(width, limits);
    const blocks = (Array.isArray(d.blocks) ? d.blocks : []).filter(b => b && TYPES.includes(b.type)).slice(0, limits.maxBlocks).map(b => ({ type: b.type }));
    return { version: 1, width, heads, context, blocks };
  }

  /* What still has to change before this design can be trained here ("" when it is ready). */
  function problem(d, limits, vocab) {
    limits = limits || APP;
    if (d.blocks.length < limits.minBlocks) return "The sheet is empty. Add a block to give it a brain.";
    if (limits.needAttention && !looksBack(d)) return "Add an Attention block or a Transformer layer. Without one it cannot look back at what it has read.";
    if (vocab && countParams(d, vocab) > HARD.maxParams) return "That is over 350 million numbers, more than this app will train. Use a smaller width or fewer blocks.";
    return "";
  }

  function classic(layers, heads, width, context) {
    return { version: 1, width, heads, context, blocks: Array.from({ length: layers }, () => ({ type: "transformer" })) };
  }

  /* How many numbers the network has to learn. Same sum as architecture.py: no bias terms,
     and the table that reads tokens in is the same one that writes them out. */
  function countParams(d, vocab) {
    const w = d.width;
    let n = vocab * w + w;
    for (const b of d.blocks) n += b.type === "transformer" ? 12 * w * w + 2 * w : b.type === "attention" ? 4 * w * w + w : 8 * w * w + w;
    return n;
  }

  /* The design as the engine runs it: one entry per (normalise -> op -> add) step, with the
     same table names PyTorch gives them, so weights can be compared name for name. */
  function expand(d) {
    const steps = [];
    d.blocks.forEach((b, i) => {
      const p = "blocks." + i + ".";
      if (b.type === "transformer") {
        steps.push({ type: "attention", heads: d.heads, norm: p + "norm1.weight", a: p + "attn.qkv.weight", b: p + "attn.proj.weight" });
        steps.push({ type: "feedforward", mult: 4, norm: p + "norm2.weight", a: p + "mlp.fc.weight", b: p + "mlp.proj.weight" });
      } else if (b.type === "attention") {
        steps.push({ type: "attention", heads: d.heads, norm: p + "norm.weight", a: p + "op.qkv.weight", b: p + "op.proj.weight" });
      } else {
        steps.push({ type: "feedforward", mult: 4, norm: p + "norm.weight", a: p + "op.fc.weight", b: p + "op.proj.weight" });
      }
    });
    return { width: d.width, context: d.context, layers: d.blocks.length, blocks: steps };
  }

  /* A short code a person can copy from the website into the app, e.g. fp1-w64-h4-c64-tta */
  function code(d) {
    return "fp1-w" + d.width + "-h" + d.heads + "-c" + d.context + "-" + (d.blocks.map(b => KINDS[b.type].letter).join("") || "0");
  }
  function parse(text) {
    const m = /^\s*fp1-w(\d{1,4})-h(\d{1,2})-c(\d{1,5})-([taf]{1,64}|0)\s*$/i.exec(String(text || ""));
    if (!m) return null;
    const byLetter = { t: "transformer", a: "attention", f: "feed_forward" };
    const blocks = m[4] === "0" ? [] : m[4].toLowerCase().split("").map(ch => ({ type: byLetter[ch] }));
    try { return normalise({ width: Number(m[1]), heads: Number(m[2]), context: Number(m[3]), blocks }); } catch (e) { return null; }
  }

  /* One honest sentence about what was built. */
  function describe(d) {
    const n = d.blocks.length;
    if (!n) return "Nothing in the middle yet. Like this it can only guess the next letter from the one it is on.";
    if (!looksBack(d)) return "No Attention, so it cannot look back. It guesses each letter from the one before it and nothing else.";
    if (d.blocks.every(b => b.type === "transformer")) {
      return n === 1 ? "One transformer layer: it looks back, then thinks. The smallest real transformer."
        : n + " transformer layers, one after another. The big chatbots are built this same way, only far wider and taller.";
    }
    if (d.blocks.every(b => b.type === "attention")) return "Attention only: it can look back, but has nowhere to think about what it finds. Add a Feed-forward block and compare.";
    return "Your own order of " + n + " blocks. Not the usual recipe, and still a real network that learns.";
  }

  return { KINDS, TYPES, WEB, APP, HARD, normalise, fix, problem, headsFor, defaultHeads, classic, countParams, expand, code, parse, describe, looksBack };
});
