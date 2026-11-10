"""The block builder, used the way a person uses it, in a real browser.

Two parts:

  site  the website (docs/): drag blocks with a mouse and with a finger, use the buttons and
        the keyboard, switch the network on, and check that it really learns in the page.
  app   the app's "How big a brain?" step: build a design, see the fit check, paste a design
        code from the website, and train the hand-built design with PyTorch.

Run from the repository root (needs:  pip install playwright  and a Chromium):

    python tests/web/browser_check.py            both parts
    python tests/web/browser_check.py site       one part
    FP_CHROME=/usr/bin/google-chrome python tests/web/browser_check.py      use an installed browser
    FP_SHOTS=/tmp/shots python tests/web/browser_check.py                   also save screenshots
    FP_BROWSER=firefox python tests/web/browser_check.py site               or webkit (Safari's engine)

Finger gestures are sent through Chromium's own touch input, so the phone part runs in Chromium only.

The page's Content-Security-Policy stays switched on, so a script or style the policy would
block fails here too.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SHOTS = os.environ.get("FP_SHOTS")
ENGINE = os.environ.get("FP_BROWSER", "chromium")
failed = 0


def check(ok: bool, what: str) -> None:
    global failed
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        failed += 1


def shot(page, name: str, full: bool = False) -> None:
    # No pictures in WebKit: to take one there Playwright adds a <style> element to the page,
    # which the page's Content-Security-Policy refuses (as it should) and reports as an error.
    # caret="initial" stops the other engines needing one.
    if SHOTS and ENGINE != "webkit":
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(SHOTS) / f"{ENGINE}-{name}.png"), full_page=full, caret="initial")


def kinds(page) -> list[str]:
    return page.eval_on_selector_all(".bld-block", "els => els.map(e => e.dataset.kind)")


def centre(page, selector: str) -> tuple[float, float]:
    box = page.locator(selector).bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def mouse_drag(page, source: str, target: str, dy: float = 0, drop: bool = True) -> None:
    x0, y0 = centre(page, source)
    x1, y1 = centre(page, target)
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1 + dy, steps=12)
    if drop:
        page.mouse.up()
        time.sleep(0.35)


def wait_until(test, seconds: float, every: float = 0.2) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if test():
            return True
        time.sleep(every)
    return False


def watch(page) -> list[str]:
    problems: list[str] = []
    def note(m):
        if m.type in ("error", "warning"):
            where = m.location or {}
            problems.append(f"{m.type}: {m.text} (at {where.get('url', '?')}:{where.get('lineNumber', '?')})")

    page.on("console", note)
    page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
    return problems


# ---------------------------------------------------------------------------- the website
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102
        pass


def site(browser) -> None:
    print(f"website (docs/) in {ENGINE}")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "docs")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        # ---- a desk computer with a mouse
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        problems = watch(page)
        page.goto(url)
        page.wait_for_selector(".bld-block")
        check(kinds(page) == ["transformer"], "opens with one transformer layer on the worksheet")
        mouse_drag(page, ".bld-piece.attention", ".bld-block >> nth=0", dy=-30)
        check(kinds(page) == ["attention", "transformer"], "mouse: a block dragged from the tray lands where it was dropped")
        mouse_drag(page, ".bld-piece.feed_forward", ".bld-writer", dy=-30)
        check(kinds(page) == ["attention", "transformer", "feed_forward"], "mouse: dropped at the end, it goes last")
        mouse_drag(page, ".bld-block >> nth=0", ".bld-block >> nth=2", dy=30)
        check(kinds(page) == ["transformer", "feed_forward", "attention"], "mouse: dragging a block re-orders the stack")
        mouse_drag(page, ".bld-block >> nth=2", ".bld-tray", drop=False)
        check(page.locator(".bld-bin").is_visible(), "while a block is in the air the tray becomes the bin")
        shot(page, "site-dragging")
        page.mouse.up()
        time.sleep(0.35)
        check(kinds(page) == ["transformer", "feed_forward"], "mouse: dropped on the tray, the block is taken off")
        page.mouse.move(*centre(page, ".bld-block >> nth=0"))
        page.mouse.down()
        page.mouse.move(640, 500, steps=6)
        page.keyboard.press("Escape")
        page.mouse.up()
        check(kinds(page) == ["transformer", "feed_forward"] and page.locator(".bld-ghost").count() == 0, "Escape puts a block back where it was")
        page.click(".bld-piece.transformer")
        page.click(".bld-block >> nth=2 >> [data-act=up]")
        page.focus(".bld-block >> nth=0")
        page.keyboard.press("Alt+ArrowDown")
        check(kinds(page) == ["transformer", "transformer", "feed_forward"], "no dragging needed: tap to add, arrow buttons, Alt+arrow keys")
        check("moved to position" in page.inner_text(".bld .sr"), "screen readers are told what moved")
        check(page.inner_text("#code") == "fp1-w64-h4-c64-ttf" and page.url.endswith("?d=fp1-w64-h4-c64-ttf"), "the design code and the page address follow the design")
        page.select_option(".bld-reader select >> nth=0", "48")
        offered = page.eval_on_selector(".bld-reader select >> nth=2", "s => [...s.options].map(o => o.value)")
        check(offered == ["1", "2", "4", "8"] and "w48" in page.inner_text("#code"), "width 48 offers only head counts that work with it")
        page.select_option(".bld-reader select >> nth=0", "64")
        shot(page, "site-built")

        page.click("#go")
        started = wait_until(lambda: page.locator("#facts > div").count() >= 3, 40)
        check(started, "Switch it on: the network starts learning in the page")
        check(page.locator("html").get_attribute("data-engine") == "wasm", "the fast WebAssembly loops are in use in this browser")
        first = page.inner_text("#writing")
        time.sleep(14)
        stage = page.inner_text("#stageTitle")
        check("Stage 2" in stage or "Stage 3" in stage or "Stage 4" in stage, f"after 14 seconds it has left random noise ({stage})")
        rows = page.locator("#nums tr").count()
        if rows == 0:
            page.click("details.plain summary")
        losses = [float(t) for t in page.eval_on_selector_all("#nums tr td:nth-child(3)", "tds => tds.map(td => td.textContent)") if t]
        check(len(losses) >= 3 and losses[0] < losses[-1] - 0.8, f"its mistakes on text it never saw fell from {losses[-1]:.2f} to {losses[0]:.2f}" if losses else "it reports its mistakes")
        check(page.locator("#viz svg path.line").count() == 2, "the chart draws both lines")
        now = page.inner_text("#writing")
        check(now != first and " " in now, "its writing changed and has spaces in it: " + json.dumps(now[:48]))
        check(page.locator("#pages").get_attribute("max") not in (None, "0"), "earlier pages are kept to slide back through")
        shot(page, "site-learning", full=True)
        page.fill("#tryIn", "To be, or ")
        page.click("#tryBtn")
        carried = wait_until(lambda: len(page.inner_text("#writing")) > 60, 20)
        check(carried and page.inner_text("#writing").startswith("To be, or "), "it carries on from words you type")
        page.click("#tryBack")
        page.click("#go")
        check(wait_until(lambda: page.inner_text("#go") == "Carry on", 5) and "Paused" in page.inner_text("#status"), "Pause pauses")
        page.click("#go")
        check(wait_until(lambda: page.inner_text("#go") == "Pause", 5), "Carry on carries on")
        page.click(".bld-block >> nth=0 >> [data-act=remove]")
        check(wait_until(lambda: page.inner_text("#go") == "Start this new design", 5), "changing the design stops the old network and offers to start the new one")
        page.click("#go")
        check(wait_until(lambda: "steps so far" in page.inner_text("#status"), 30), "the new design trains too")
        compared = wait_until(lambda: page.locator("#versus").is_visible() and page.locator("#viz svg path.line").count() == 3, 25)
        check(compared and "your last design scored" in page.inner_text("#versus"), "the chart compares it with the design before: " + json.dumps(page.inner_text("#versus")[:90]))
        # your own words
        page.click("#own summary")
        own = ("the cat sat on the mat. the dog sat on the log. a cat and a dog sat on a mat and a log. " * 40)
        page.fill("#ownText", own)
        check(not page.locator("#ownUse").is_disabled(), "a page of your own text can be used")
        page.click("#ownUse")
        learnt = wait_until(lambda: "your text" in page.inner_text("#status") and "steps so far" in page.inner_text("#status"), 30)
        time.sleep(6)
        text = page.inner_text("#writing")
        check(learnt and ("cat" in text or "dog" in text or "sat" in text), "it learns from your own text: " + json.dumps(text[:40]))
        check(not problems, "no errors or warnings in the browser console" + ("" if not problems else ": " + " | ".join(problems[:3])))
        page.close()

        # ---- a browser that cannot have the fast path: the plain JavaScript loops must carry it
        page = browser.new_page(viewport={"width": 1000, "height": 800})
        page.route("**/kernels.wasm", lambda route: route.abort())
        page.goto(url)
        page.wait_for_selector(".bld-block")
        page.click("#go")
        slow_path = wait_until(lambda: page.locator("html").get_attribute("data-engine") == "js" and "steps so far" in page.inner_text("#status"), 60)
        time.sleep(4)
        check(slow_path and "Stage" in page.inner_text("#stageTitle") and page.locator("#viz svg path.line").count() >= 1,
              "with kernels.wasm unavailable it still learns, on the plain JavaScript loops: " + page.inner_text("#status")[-34:])
        page.close()

        if ENGINE != "chromium":
            print("  (the finger checks need Chromium's touch input; run without FP_BROWSER for those)")
            return
        # ---- a phone, with a finger
        ctx = browser.new_context(viewport={"width": 390, "height": 800}, device_scale_factor=2, is_mobile=True, has_touch=True)
        page = ctx.new_page()
        problems = watch(page)
        page.goto(url + "?d=fp1-w48-h4-c32-t")
        page.wait_for_selector(".bld-block")
        check(page.inner_text("#code") == "fp1-w48-h4-c32-t", "a design code in the page address opens that design")
        cdp = ctx.new_cdp_session(page)

        def finger(x0, y0, x1, y1, steps=14):
            cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x0, "y": y0}]})
            time.sleep(0.05)
            for i in range(1, steps + 1):
                cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x0 + (x1 - x0) * i / steps, "y": y0 + (y1 - y0) * i / steps}]})
                time.sleep(0.02)

        def lift():
            cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            time.sleep(0.4)

        page.tap(".bld-piece.attention")
        page.tap(".bld-piece.feed_forward")
        check(kinds(page) == ["transformer", "attention", "feed_forward"], "finger: tapping a block adds it")
        page.locator(".bld-block >> nth=2").scroll_into_view_if_needed()
        time.sleep(0.3)
        x0, y0 = centre(page, ".bld-block >> nth=2 >> .bld-grip")
        x1, y1 = centre(page, ".bld-block >> nth=0")
        finger(x0, y0, x1, y1 - 30)
        shot(page, "phone-dragging")
        lift()
        check(kinds(page) == ["feed_forward", "transformer", "attention"], "finger: dragging by the dotted handle re-orders the stack")
        before = page.evaluate("window.scrollY")
        x0, y0 = centre(page, ".bld-block >> nth=1 >> .bld-words")
        finger(x0, y0, x0, y0 - 220)
        lift()
        check(page.evaluate("window.scrollY") - before > 100 and kinds(page) == ["feed_forward", "transformer", "attention"], "finger: swiping across a block scrolls the page and moves nothing")
        page.evaluate("window.scrollTo(0, 330)")
        time.sleep(0.4)
        x0, y0 = centre(page, ".bld-piece.transformer")
        box = page.locator(".bld-block >> nth=0").bounding_box()
        finger(x0, y0, box["x"] + box["width"] / 2, box["y"] + box["height"] + 6, steps=16)
        lift()
        check(kinds(page) == ["feed_forward", "transformer", "transformer", "attention"], "finger: a block dragged from the tray lands between two others")
        wide = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        check(wide[0] <= wide[1], "nothing sticks out sideways on a phone")
        page.evaluate("window.scrollTo(0, 0)")
        page.tap("#go")
        check(wait_until(lambda: "steps so far" in page.inner_text("#status"), 40), "it trains on a phone-sized screen too")
        shot(page, "phone-learning", full=True)
        check(not problems, "no errors or warnings in the browser console" + ("" if not problems else ": " + " | ".join(problems[:3])))
        ctx.close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------- the app
def app(browser) -> None:
    print("the app's Brain step")
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc = subprocess.Popen([sys.executable, "-m", "foolproof", "--no-browser", "--port", str(port)], cwd=str(ROOT),
                                env=dict(os.environ, FOOLPROOF_HOME=str(home)), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            lock = home / ".server.json"
            check(wait_until(lock.exists, 90, 0.3), "the app starts")
            info = json.loads(lock.read_text(encoding="utf-8"))
            base, token = f"http://127.0.0.1:{info['port']}", info["token"]

            def api(method, path, body=None):
                req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                             headers={"X-FP-Token": token, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode())

            api("POST", "/api/prepare", {"name": "Shakespeare", "source": {"type": "builtin", "id": "shakespeare"}})
            check(wait_until(lambda: len(api("GET", "/api/state").get("datasets", [])) == 1 and not api("GET", "/api/state").get("job"), 180, 1.0), "the built-in text is ready")

            page = browser.new_page(viewport={"width": 1280, "height": 900})
            problems = watch(page)
            page.goto(base + "/")
            page.wait_for_selector("text=Make a new AI", timeout=30000)
            page.click("text=Make a new AI")
            page.click("button[role=tab]:has-text('Already prepared')")
            page.click("button:has-text('Use this')")
            page.wait_for_selector("text=How big a brain", timeout=30000)
            page.wait_for_selector("button:has-text('Build it yourself')", timeout=30000)
            page.click("button:has-text('Build it yourself')")
            page.wait_for_selector(".bld-block")
            start = kinds(page)
            check(len(start) >= 1 and set(start) == {"transformer"}, f"Build it yourself starts from the ticked ready-made size ({len(start)} transformer layers)")
            note = ".note[role=status]"
            check(wait_until(lambda: "connections" in page.inner_text(note), 15), "it is checked against this computer: " + page.inner_text(note)[:70])
            mouse_drag(page, ".bld-piece.attention", ".bld-block >> nth=0", dy=-30)
            check(kinds(page)[0] == "attention" and len(kinds(page)) == len(start) + 1, "mouse dragging works in the app")
            page.select_option(".bld-reader select >> nth=0", "768")
            page.select_option(".bld-reader select >> nth=1", "1024")
            while not page.locator(".bld-piece.transformer").is_disabled():
                page.click(".bld-piece.transformer")
            check(len(kinds(page)) == 16, "16 blocks is the limit, and the tray says so")
            too_big = wait_until(lambda: "Too big" in page.inner_text(note), 15)
            check(too_big and page.locator("button:has-text('Start training')").is_disabled(), "a design too big for this computer is caught before training")
            page.click("button:has-text('Start again from')")
            while page.locator(".bld-block").count():
                page.click(".bld-block >> nth=0 >> [data-act=remove]")
            page.click(".bld-piece.feed_forward")
            check(wait_until(lambda: "look back" in page.inner_text(note), 5) and page.locator("button:has-text('Start training')").is_disabled(),
                  "a design with no attention is refused in plain words")
            page.fill("input[aria-label='Design code from the website']", "not a code")
            page.click("button:has-text('Use this code')")
            check(wait_until(lambda: page.locator(".toast").count() and "not a design code" in page.inner_text(".toast"), 5), "a wrong design code is explained, not swallowed")
            page.fill("input[aria-label='Design code from the website']", "fp1-w48-h4-c32-atf")
            page.click("button:has-text('Use this code')")
            check(wait_until(lambda: kinds(page) == ["attention", "transformer", "feed_forward"] and "This will train" in page.inner_text(note), 15),
                  "a design code from the website loads, phone-sized width included")
            shot(page, "app-builder", full=True)
            page.fill("input[maxlength='60']", "Built by hand")
            page.click("button:has-text('5 minutes')")
            page.click("button:has-text('Start training')")
            page.wait_for_selector("text=Test it yourself", timeout=90000)
            check(wait_until(lambda: "learning steps" in page.inner_text(".facts"), 90), "the hand-built design trains with PyTorch")
            time.sleep(5)
            shot(page, "app-training", full=True)
            page.click("button:has-text('Stop and save')")
            page.wait_for_selector("text=Finished.", timeout=180000)
            model = [m for m in api("GET", "/api/state")["models"] if m.get("name") == "Built by hand"]
            check(len(model) == 1 and model[0]["model"].get("block_types") == ["attention", "transformer", "feed_forward"]
                  and model[0]["model"]["n_embd"] == 48 and model[0]["model"]["block_size"] == 32,
                  "the saved model is the design that was stacked: " + json.dumps(model[0]["model"] if model else None))
            check(not problems, "no errors or warnings in the browser console" + ("" if not problems else ": " + " | ".join(problems[:3])))
            page.close()
            api("POST", "/api/quit", {})
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()


def main() -> int:
    parts = [a for a in sys.argv[1:] if a in ("site", "app")] or ["site", "app"]
    with sync_playwright() as p:
        chrome = os.environ.get("FP_CHROME")
        if ENGINE == "chromium":
            browser = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"]) if chrome else p.chromium.launch()
        else:
            browser = getattr(p, ENGINE).launch()
        try:
            if "site" in parts:
                site(browser)
            if "app" in parts:
                app(browser)
        finally:
            browser.close()
    print(f"\n{failed} check(s) FAILED" if failed else "\nall browser checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
