"""End-to-end check that everything works on this computer.

    python -m foolproof.selftest            (about a minute, no internet needed)
    python -m foolproof.selftest --online   (also streams a little data from Hugging Face)

Used by the launchers' --selftest flag and by the automatic tests on GitHub.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def _check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        raise SystemExit(f"self-test failed: {what}")


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def _server_roundtrip(tmp: str, architecture=None) -> None:
    """Start the real app and use it the way the window does: over HTTP, with worker processes."""
    import socket
    import subprocess
    import urllib.error
    import urllib.request

    home = Path(tmp) / "server-home"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, FOOLPROOF_HOME=str(home))
    proc = subprocess.Popen([sys.executable, "-m", "foolproof", "--no-browser", "--port", str(port)],
                            cwd=str(Path(__file__).resolve().parent.parent), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        lock = home / ".server.json"
        deadline = time.time() + 90
        while time.time() < deadline and not lock.exists():
            time.sleep(0.3)
        _check(lock.exists(), "the app's window server starts")
        info = json.loads(lock.read_text(encoding="utf-8"))
        base, token = f"http://127.0.0.1:{info['port']}", info["token"]

        def api(method, path, body=None, tok=token, timeout=60):
            req = urllib.request.Request(base + path, method=method,
                                         data=None if body is None else json.dumps(body).encode("utf-8"),
                                         headers={"X-FP-Token": tok, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))

        try:
            api("GET", "/api/state", tok="wrong")
            locked = False
        except urllib.error.HTTPError as e:
            locked = e.code == 403
        _check(locked, "the app refuses requests that do not carry its secret token")
        with urllib.request.urlopen(base + "/", timeout=30) as r:
            page = r.read().decode("utf-8")
        _check('name="fp-token"' in page and "app.js" in page, "the page loads")

        state = api("GET", "/api/state")
        deadline = time.time() + 240
        while state.get("hw") is None and time.time() < deadline:
            time.sleep(1.0)
            state = api("GET", "/api/state")
        _check(state.get("hw") is not None, "the app works out what this computer can do")
        seq = state["seq"]

        def wait_for(pred, seconds):
            nonlocal seq
            deadline = time.time() + seconds
            seen = []
            while time.time() < deadline:
                r = api("GET", f"/api/events?since={seq}&wait=10", timeout=40)
                seq = r["seq"]
                for ev in r["events"]:
                    seen.append(ev)
                    if ev.get("type") == "error":
                        print("   app reported:", ev.get("msg"), (ev.get("detail") or "")[-800:])
                    if pred(ev):
                        return ev, seen
            return None, seen

        api("POST", "/api/prepare", {"name": "Shakespeare", "source": {"type": "builtin", "id": "shakespeare"}})
        ev, _ = wait_for(lambda e: e.get("type") == "job_end" and e.get("kind") == "prepare", 180)
        _check(ev is not None and ev.get("reason") == "done", "prepares text through the app")
        if architecture:
            fit = api("POST", "/api/fit", {"dataset": ev["dataset"], "architecture": architecture})
            _check(fit.get("level") in ("ok", "slow") and fit.get("params") == 64 * 260 + 64 + (4 + 8 + 12) * 64 * 64 + 4 * 64,
                   f"checks a hand-built design against this computer: {fit.get('level')}, {fit.get('params')} connections")
            try:
                api("POST", "/api/fit", {"dataset": ev["dataset"], "architecture": dict(architecture, blocks=[{"type": "feed_forward"}])})
                _check(False, "refuses a design that cannot look back")
            except urllib.error.HTTPError as e:
                _check("ttention" in e.read().decode("utf-8", "replace"), "refuses a design that cannot look back, in plain words")
        r = api("POST", "/api/train", {"dataset": ev["dataset"], "preset": "pocket", "name": "Server test",
                                       "budget_sec": 6, "architecture": architecture,
                                       "overrides": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64, "batch": 16,
                                                     "min_steps": 60}})
        run = r["run"]
        ev, seen = wait_for(lambda e: e.get("type") == "job_end" and e.get("kind") == "train", 600)
        _check(ev is not None and ev.get("reason") == "budget", "trains in a separate process and finishes on time")
        _check(any(e.get("type") == "sample" for e in seen) and any(e.get("type") == "eval" for e in seen),
               "reports progress and writing samples to the window")
        api("POST", "/api/generate", {"slug": run, "id": "t1", "prompt": "ROMEO:", "max_new_tokens": 40})
        ev, seen = wait_for(lambda e: e.get("type") == "gen_end" and e.get("id") == "t1", 240)
        text = "".join(e.get("text", "") for e in seen if e.get("type") == "gen" and e.get("id") == "t1")
        _check(ev is not None and len(text) > 5, f"the finished AI answers in the window: {text[:30]!r}")
        models = api("GET", "/api/state")["models"]
        _check(len(models) == 1 and models[0]["status"] == "finished" and models[0]["has_model"], "lists the finished AI")
        if architecture:
            _check(models[0]["model"]["block_types"] == [b["type"] for b in architecture["blocks"]],
                   "the actual trained model preserves the visual builder block order")
        api("POST", "/api/quit", {})
        proc.wait(timeout=60)
        _check(proc.returncode == 0, "quits cleanly")
    finally:
        if proc.poll() is None:
            proc.kill()
            out = proc.stdout.read() if proc.stdout else ""
            print("   app output:", out[-1500:])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    online = "--online" in argv
    seconds = 12.0
    for a in argv:
        if a.startswith("--seconds="):
            seconds = float(a.split("=", 1)[1])
    tmp = tempfile.mkdtemp(prefix="foolproof-selftest-")
    os.environ["FOOLPROOF_HOME"] = tmp
    from . import data, export, hardware, paths, presets, runs, textio, worker
    from .tokenizers import BPETokenizer, ByteTokenizer, StreamDecoder, Tokenizer

    t0 = time.time()
    print("Foolproof Trainer self-test", flush=True)
    hw = hardware.detect()
    print(f"  computer: {hw['summary']} | PyTorch {hw['torch']}", flush=True)
    _check(hw["torch"] is not None, "PyTorch starts")

    # ---- reading files of every kind -----------------------------------------------------
    d = Path(tmp) / "files"
    d.mkdir()
    (d / "a.txt").write_text("Hello there.\nThis is a plain text file.\n" * 600, encoding="utf-8")
    (d / "b.jsonl").write_text("\n".join(json.dumps({"instruction": f"Say {i}", "output": f"{i}!"})
                                         for i in range(30)), encoding="utf-8")
    (d / "c.csv").write_text("title,text\n" + "\n".join(f"T{i},\"Body number {i}, with a comma.\"" for i in range(30)),
                             encoding="utf-8")
    (d / "d.html").write_text("<html><head><style>x{}</style></head><body><p>Para one.</p><p>Para two.</p></body></html>",
                              encoding="utf-8")
    with zipfile.ZipFile(d / "e.docx", "w") as z:
        z.writestr("word/document.xml", "<w:document><w:body><w:p><w:r><w:t>Docx first.</w:t></w:r></w:p>"
                                        "<w:p><w:r><w:t>Docx second &amp; last.</w:t></w:r></w:p></w:body></w:document>")
    (d / "f.bin").write_bytes(bytes(range(256)) * 20)
    skipped = []
    got = {p.name: list(textio.iter_docs_from_path(p, skipped)) for p in sorted(d.iterdir())}
    _check(len(got["a.txt"]) == 1 and "plain text" in got["a.txt"][0], "reads .txt")
    _check(len(got["b.jsonl"]) == 30 and got["b.jsonl"][0][0] == ("user", "Say 0"), "reads question/answer .jsonl as chat")
    _check(len(got["c.csv"]) == 30 and got["c.csv"][3].startswith("T3\n\nBody number 3, with"), "reads .csv text column")
    _check(got["d.html"] == ["Para one.\n\nPara two."], "reads .html without the tags")
    _check(got["e.docx"] == ["Docx first.\nDocx second & last."], "reads .docx")
    _check(got["f.bin"] == [] and skipped and skipped[0][0] == "f.bin", "skips non-text files politely")

    # ---- tokenizers ----------------------------------------------------------------------
    text = (paths.ASSETS_DIR / "tiny_shakespeare.txt").read_text(encoding="utf-8")
    sample = "Héllo wörld — 你好! 🙂\n"
    bt = ByteTokenizer()
    _check(bt.decode(bt.encode(sample)) == sample, "letter-by-letter tokenizer round-trips any text")
    bpe = BPETokenizer.train(iter([text[:300_000]]), 600)
    bpe.save(Path(tmp) / "tok.json")
    bpe2 = Tokenizer.load(Path(tmp) / "tok.json")
    ids = bpe2.encode(sample + text[:200])
    _check(bpe2.decode(ids) == sample + text[:200] and bpe2.vocab_size == 604, "learned vocabulary round-trips and reloads")
    dec = StreamDecoder(bt)
    streamed = "".join(dec.push(t) for t in bt.encode(sample)) + dec.flush()
    _check(streamed == sample, "streaming never splits a character")

    # ---- data preparation ----------------------------------------------------------------
    events = []
    meta = data.prepare({"name": "Shakespeare", "source": {"type": "builtin", "id": "shakespeare"},
                         "tokenizer": "auto"}, events.append, lambda: False)
    _check(meta["tokenizer"]["kind"] == "bytes" and meta["n_train"] > 900_000 and meta["n_val"] > 50_000,
           f"prepares the built-in text ({meta['n_train']:,} train / {meta['n_val']:,} held back)")
    meta2 = data.prepare({"name": "My files", "source": {"type": "files", "paths": [str(d)]},
                          "tokenizer": "bpe:300"}, events.append, lambda: False)
    _check(meta2["n_docs"] >= 60 and meta2["tokenizer"]["kind"] == "bpe", "prepares a folder of mixed files")
    _check(len(data.list_datasets()) == 2, "lists prepared data")

    # ---- training ------------------------------------------------------------------------
    rec = presets.recommend(hw, meta["n_train"], meta["vocab_size"])
    _check(rec["recommended"] in presets.PRESET_BY_ID, f"recommends a brain size ({rec['recommended']})")
    job = runs.create("Self test", meta, "pocket", seconds, hw,   # a miniature brain: quick on any machine
                      overrides={"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64, "batch": 16,
                                 "min_steps": 200})
    cq, eq = queue.Queue(), queue.Queue()
    worker.main(job, cq, eq)
    ev = _drain(eq)
    errors = [e for e in ev if e["type"] == "error"]
    if errors:
        print(errors[0].get("detail") or errors[0].get("msg"))
    _check(not errors, "training runs without errors")
    ms = [e for e in ev if e["type"] == "metrics"]
    evals = [e for e in ev if e["type"] == "eval"]
    samples = [e for e in ev if e["type"] == "sample"]
    end = [e for e in ev if e["type"] == "train_end"]
    _check(end and end[0]["reason"] == "budget", f"stops by itself when the time is up ({end[0]['step'] if end else 0} steps)")
    start = [e for e in ev if e["type"] == "train_start"][0]["random_loss"]
    _check(ms and ms[-1]["loss"] < start - 1.5, f"it learned: loss {start:.2f} -> {ms[-1]['loss']:.2f}")
    _check(evals and evals[-1]["val"] is not None and evals[-1]["val"] < start - 1.5, "it also improved on text it never saw")
    _check(samples and len(samples[-1]["text"]) > 20, "it writes sample text while training")
    run_dir = Path(job["run_dir"])
    _check((run_dir / "ckpt_best.pt").exists() and (run_dir / "ckpt_last.pt").exists(), "saves its progress")

    # ---- resume ----------------------------------------------------------------------------
    cfg = runs.get(job["run"])
    job2 = runs.job_for(cfg, hw, resume=True, budget_sec=4)
    worker.main(job2, cq, eq)
    ev2 = _drain(eq)
    st2 = [e for e in ev2 if e["type"] == "train_start"]
    _check(st2 and st2[0]["resumed"] and st2[0]["step"] == end[0]["step"] and not [e for e in ev2 if e["type"] == "error"],
           "continues training from where it stopped")

    # ---- chatting with the finished model -------------------------------------------------
    import threading

    th = threading.Thread(target=worker.main, daemon=True,
                          args=({"kind": "serve", "id": "s", "run_dir": str(run_dir), "hw": hw}, cq, eq))
    th.start()
    cq.put({"cmd": "generate", "id": "g1", "prompt": "ROMEO:", "max_new_tokens": 60, "temperature": 0.8})
    ev3, deadline = [], time.time() + 60
    while time.time() < deadline and not any(e["type"] in ("gen_end", "error") for e in ev3):
        try:
            ev3.append(eq.get(timeout=0.5))
        except queue.Empty:
            pass
    cq.put({"cmd": "stop"})
    th.join(timeout=30)
    ev3 += _drain(eq)
    out = "".join(e["text"] for e in ev3 if e["type"] == "gen")
    _check(len(out) > 10 and any(e["type"] == "gen_end" for e in ev3), f"the finished AI writes text: {out[:40]!r}")

    # ---- export ----------------------------------------------------------------------------
    folder = export.export(run_dir)
    names = {p.name for p in folder.iterdir()}
    _check({"model.safetensors", "config.json", "tokenizer.json", "chat.py", "README.md"} <= names, "exports a shareable folder")

    # The app's own window is driven with a hand-built design (attention, feed-forward, then a
    # transformer layer), so every launcher run on every platform also proves the block builder.
    # The ready-made sizes were trained directly just above.
    architecture = dict(version=1, width=64, heads=2, context=64,
                        blocks=[{"type": "attention"}, {"type": "feed_forward"}, {"type": "transformer"}])
    _server_roundtrip(tmp, architecture)

    if online:
        from . import hub

        res = hub.search("tinystories", 5)
        _check(any(r["id"] == "roneneldan/TinyStories" for r in res), "searches Hugging Face")
        meta3 = data.prepare({"name": "TinyStories taste",
                              "source": {"type": "hf", "id": "roneneldan/TinyStories", "config": "default", "split": "train"},
                              "tokenizer": "auto", "max_tokens": 150_000}, events.append, lambda: False)
        _check(meta3["n_train"] > 80_000 and meta3["truncated"], f"streams part of a big dataset ({meta3['n_chars']:,} letters)")

    print(f"ALL GOOD in {time.time() - t0:.0f}s. Everything works on this computer.", flush=True)
    if "--keep" not in argv:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
