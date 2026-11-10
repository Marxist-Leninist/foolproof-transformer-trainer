"use strict";
// Exercise the actual report sender without running the browser/network training suite.
const fs = require("fs"), vm = require("vm"), assert = require("assert"), path = require("path");
const source = fs.readFileSync(path.join(__dirname, "../../docs/check.js"), "utf8");
const start = source.indexOf("  function send() {");
const end = source.indexOf('\n  $("again")', start);
assert(start >= 0 && end > start, "report sender is present");
const posts = [];
const ctx = {
  running: true,
  result: () => ({passed: false, finger: "worked", checks: [{state: "run"}]}),
  fetch: (url, opts) => { posts.push(JSON.parse(opts.body)); return Promise.resolve(); }
};
vm.createContext(ctx);
vm.runInContext(source.slice(start, end), ctx);
vm.runInContext("send()", ctx);
assert.equal(posts.length, 0, "a finger drag during automatic checks must not submit");
ctx.running = false;
ctx.result = () => ({passed: true, finger: "worked", checks: [{state: "ok"}]});
vm.runInContext("send()", ctx);
assert.equal(posts.length, 1, "the finished run submits");
assert.equal(posts[0].finger, "worked", "earlier finger evidence is retained");
assert.equal(posts[0].passed, true);
vm.runInContext("send()", ctx);
assert.equal(posts.length, 2, "a later finger update can submit");
ctx.result = () => ({passed: false, finger: "worked", checks: [{state: "fail"}]});
vm.runInContext("send()", ctx);
assert.equal(posts.length, 3, "finished failures must also submit");
assert.equal(posts[2].passed, false);
console.log("PASS: no early report; completed success/failure and later updates submit.");
