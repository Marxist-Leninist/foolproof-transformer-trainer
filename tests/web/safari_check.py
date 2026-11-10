"""The website in real Safari, on a Mac.

tests/web/browser_check.py covers WebKit, the engine inside Safari, as Playwright builds it for Linux.
That is close to Safari but it is not Safari. This script drives the Safari that ships with macOS through
Apple's own safaridriver, and repeats the checks that matter most there: the builder with a mouse and
without one, the design code, and the network learning inside the page on the WebAssembly loops.

Run on a Mac from the repository root:

    sudo safaridriver --enable          once, to allow automation
    pip install selenium
    python tests/web/safari_check.py
    FP_SHOTS=/tmp/shots python tests/web/safari_check.py      also save screenshots

Finger gestures need a touch screen, so they are not covered here; see docs/check.html for those.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select

ROOT = Path(__file__).resolve().parents[2]
SHOTS = os.environ.get("FP_SHOTS")
failed = 0


def check(ok: bool, what: str) -> None:
    global failed
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        failed += 1


def wait_until(test, seconds: float, every: float = 0.25) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            if test():
                return True
        except Exception:  # noqa: BLE001  (the page may be redrawing)
            pass
        time.sleep(every)
    return False


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102
        pass


class Local(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # Skip HTTPServer's reverse name lookup: on macOS it puts up a "find devices on local networks" prompt.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


LOOK = """
const pick = s => { const e = document.querySelector(s); if (!e) return null; const r = e.getBoundingClientRect(), c = getComputedStyle(e);
  return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height), 'opacity ' + c.opacity, c.visibility, c.display, c.color]; };
const mid = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
return {page: document.visibilityState, focused: document.hasFocus(), window: [innerWidth, innerHeight], scrolled: Math.round(scrollY),
        height: document.documentElement.scrollHeight, fonts: document.fonts ? document.fonts.status : '?',
        heading: pick('h1'), builder: pick('#builder'), first_block: pick('.bld-block'), go: pick('#go'), verdict: pick('#verdict'),
        in_the_middle: mid ? mid.nodeName.toLowerCase() + '.' + (mid.className && mid.className.baseVal !== undefined ? mid.className.baseVal : mid.className) : null};
"""


def main() -> int:
    server = Local(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "docs")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    driver = webdriver.Safari()
    try:
        caps = driver.capabilities
        print(f"website (docs/) in Safari {caps.get('browserVersion', '?')} on {caps.get('platformName', '?')}")
        driver.set_window_size(1280, 900)
        js = driver.execute_script
        one = lambda css: driver.find_element(By.CSS_SELECTOR, css)  # noqa: E731
        count = lambda css: len(driver.find_elements(By.CSS_SELECTOR, css))  # noqa: E731
        text = lambda css: js("return document.querySelector(arguments[0]).innerText", css)  # noqa: E731
        kinds = lambda: js("return Array.from(document.querySelectorAll('.bld-block')).map(e => e.dataset.kind)")  # noqa: E731

        def shot(name: str) -> None:
            # Two pictures: safaridriver's own, and one of the whole screen. On a machine with nobody at it
            # safaridriver's picture can come out blank even though the page is drawn, so the screen is kept too.
            if SHOTS:
                Path(SHOTS).mkdir(parents=True, exist_ok=True)
                driver.save_screenshot(str(Path(SHOTS) / f"safari-{name}.png"))
                if sys.platform == "darwin":
                    subprocess.run(["screencapture", "-x", str(Path(SHOTS) / f"safari-screen-{name}.png")], check=False, timeout=20)
            print(f"       (what Safari says is on screen at '{name}': {json.dumps(js(LOOK))})", flush=True)

        def block(i: int, inner: str = ""):
            return driver.find_elements(By.CSS_SELECTOR, ".bld-block")[i].find_element(By.CSS_SELECTOR, inner) if inner \
                else driver.find_elements(By.CSS_SELECTOR, ".bld-block")[i]

        def drag(source, target, dy: int) -> None:
            """Press on source, carry it in small steps to target's middle plus dy, let go."""
            js("arguments[0].scrollIntoView({block: 'center'})", source)
            time.sleep(0.2)
            box = js("const r = arguments[0].getBoundingClientRect(), q = arguments[1].getBoundingClientRect();"
                     "return [r.left + r.width / 2, r.top + r.height / 2, q.left + q.width / 2, q.top + q.height / 2];", source, target)
            dx, dist = box[2] - box[0], box[3] + dy - box[1]
            chain = ActionChains(driver, duration=40).move_to_element(source).click_and_hold()
            for _ in range(10):
                chain = chain.move_by_offset(round(dx / 10), round(dist / 10))
            chain.pause(0.15).release().perform()
            time.sleep(0.4)

        # A control for the pictures: a page that is nothing but a red block. If this comes out blank in the
        # pictures too, the machine cannot show Safari's pages at all and the pictures say nothing about the site.
        driver.get("data:text/html,<title>paint control</title><body style='margin:0;background:rgb(220,0,0)'><h1 style='font:700 80px sans-serif;color:white;padding:60px'>PAINT CONTROL</h1>")
        time.sleep(1.0)
        if SHOTS:
            Path(SHOTS).mkdir(parents=True, exist_ok=True)
            driver.save_screenshot(str(Path(SHOTS) / "safari-paint-control.png"))
            if sys.platform == "darwin":
                subprocess.run(["screencapture", "-x", str(Path(SHOTS) / "safari-screen-paint-control.png")], check=False, timeout=20)

        driver.get(url)
        check(wait_until(lambda: count(".bld-block") >= 1, 20), "the page opens and the builder draws")
        js("window.__fpErrors = [];"
           "window.addEventListener('error', e => window.__fpErrors.push('error: ' + (e.message || '?')));"
           "window.addEventListener('unhandledrejection', e => window.__fpErrors.push('promise: ' + ((e.reason && e.reason.message) || e.reason)));"
           "document.addEventListener('securitypolicyviolation', e => window.__fpErrors.push('policy: ' + e.violatedDirective + ' ' + e.blockedURI"
           " + ' at ' + (e.sourceFile || '?').split('/').pop() + ':' + e.lineNumber + ' on <' + ((e.target && e.target.nodeName) || '?').toLowerCase()"
           " + ' class=\"' + ((e.target && e.target.className && e.target.className.baseVal !== undefined ? e.target.className.baseVal : (e.target && e.target.className)) || '') + '\">'));")

        seen = 0

        def reports(after: str) -> None:
            """Say when the browser reported something, so a report can be tied to the step that caused it."""
            nonlocal seen
            now = js("return window.__fpErrors")
            for line in now[seen:]:
                print(f"       (browser report after {after}: {line})", flush=True)
            seen = len(now)

        check(kinds() == ["transformer"], "opens with one transformer layer on the worksheet")

        # ---- without dragging: tap to add, arrow buttons, the keyboard
        one(".bld-piece.attention").click()
        one(".bld-piece.feed_forward").click()
        check(kinds() == ["transformer", "attention", "feed_forward"], "tapping a block in the tray adds it at the end")
        block(2, "[data-act=up]").click()
        check(kinds() == ["transformer", "feed_forward", "attention"], "the arrow button moves a block up")
        check("moved to position" in text(".bld .sr"), "screen readers are told what moved")
        reports("the buttons")
        # The builder moves a block on Ctrl or Alt with an arrow key. Typed through safaridriver, a Ctrl chord arrives
        # as a real keyboard would send it. An Option chord does not (the arrow arrives with no key name, which no real
        # keyboard does), so the Alt path is exercised with a scripted key event and labelled as such.
        js("window.__fpKeys = []; document.addEventListener('keydown', e => window.__fpKeys.push(e.key + (e.altKey ? '+alt' : '')"
           " + (e.ctrlKey ? '+ctrl' : '') + (e.isTrusted ? '' : ' (scripted)')), true);")
        before = kinds()
        js("arguments[0].focus()", block(0))
        block(0).send_keys(Keys.CONTROL, Keys.ARROW_DOWN, Keys.NULL)              # the null key lets go of every modifier
        ActionChains(driver).key_up(Keys.CONTROL).key_up(Keys.ALT).perform()      # and again, the other way: never leave one held
        time.sleep(0.4)
        check(kinds() == [before[1], before[0], before[2]],
              f"keyboard: Ctrl and the down arrow, typed, move the block that has the keyboard (keys the page saw: {', '.join(js('return window.__fpKeys')) or 'none'})")
        js("arguments[0].dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowUp', altKey: true, bubbles: true, cancelable: true}))", block(1))
        time.sleep(0.4)
        check(kinds() == before, "keyboard: Alt and the up arrow, as a scripted key event, move it back")
        reports("the keyboard")

        # ---- with a mouse
        before = kinds()
        drag(block(0), block(2), 30)
        check(kinds() == before[1:] + before[:1], f"mouse: dragging a block re-orders the stack ({', '.join(kinds())})")
        reports("the first drag")
        before = kinds()
        drag(one(".bld-piece.attention"), block(0), -30)
        check(kinds() == ["attention"] + before, "mouse: a block dragged from the tray lands where it was dropped")
        reports("the drag from the tray")
        drag(block(0), one(".bld-tray"), 0)
        check(kinds() == before, "mouse: dropped on the tray, the block is taken off")
        reports("the drag to the bin")
        code, want = text("#code"), "fp1-w64-h4-c64-" + "".join(k[0] for k in kinds())
        check(code == want and driver.current_url.endswith("?d=" + want), f"the design code and the page address follow the design ({code})")
        selects = driver.find_elements(By.CSS_SELECTOR, ".bld-reader select")
        Select(selects[0]).select_by_value("48")
        time.sleep(0.3)
        selects = driver.find_elements(By.CSS_SELECTOR, ".bld-reader select")
        offered = [o.get_attribute("value") for o in Select(selects[2]).options]
        check(offered == ["1", "2", "4", "8"] and "w48" in text("#code"), "width 48 offers only head counts that work with it")
        Select(driver.find_elements(By.CSS_SELECTOR, ".bld-reader select")[0]).select_by_value("64")
        time.sleep(0.3)
        reports("the drop-downs")
        shot("built")
        reports("the first screenshot")

        # ---- the network, learning in the page
        one("#go").click()
        check(wait_until(lambda: count("#facts > div") >= 3, 40), "Switch it on: the network starts learning in the page")
        engine = js("return document.documentElement.getAttribute('data-engine')")
        check(engine == "wasm", f"the fast WebAssembly loops are in use in Safari (engine: {engine})")
        first = text("#writing")
        time.sleep(14)
        stage = text("#stageTitle")
        check(any(s in stage for s in ("Stage 2", "Stage 3", "Stage 4", "Stage 5")), f"after 14 seconds it has left random noise ({stage})")
        if count("#nums tr") == 0:
            one("details.plain summary").click()
            time.sleep(0.3)
        losses = [float(t) for t in js("return Array.from(document.querySelectorAll('#nums tr td:nth-child(3)')).map(td => td.textContent)") if t]
        check(len(losses) >= 3 and losses[0] < losses[-1] - 0.8,
              f"its mistakes on text it never saw fell from {losses[-1]:.2f} to {losses[0]:.2f}" if losses else "it reports its mistakes")
        check(count("#viz svg path.line") == 2, "the chart draws both lines")
        now = text("#writing")
        check(now != first and " " in now, "its writing changed and has spaces in it: " + json.dumps(now[:48]))
        reports("learning")
        shot("learning")
        reports("the second screenshot")
        box = one("#tryIn")
        box.send_keys(Keys.NULL)
        box.send_keys("To be, or ")
        typed = js("return document.getElementById('tryIn').value")
        if typed != "To be, or ":
            print(f"       (safaridriver typed {typed!r} into the box, so the words are put in by script instead)", flush=True)
            js("const b = document.getElementById('tryIn'); b.value = arguments[0]; b.dispatchEvent(new Event('input', {bubbles: true}));", "To be, or ")
        one("#tryBtn").click()
        carried = wait_until(lambda: len(text("#writing")) > 60, 20)
        check(carried and text("#writing").startswith("To be, or "), "it carries on from words you type")
        one("#tryBack").click()
        one("#go").click()
        check(wait_until(lambda: text("#go") == "Carry on", 5) and "Paused" in text("#status"), "Pause pauses")
        one("#go").click()
        check(wait_until(lambda: text("#go") == "Pause", 5), "Carry on carries on")
        block(0, "[data-act=remove]").click()
        check(wait_until(lambda: text("#go") == "Start this new design", 5), "changing the design stops the old network and offers to start the new one")
        one("#go").click()
        pressed = time.time()

        def new_design_trains() -> bool:
            if "steps so far" in text("#status"):
                return True
            # On a busy shared Mac the press sometimes does not take. If the button still offers to start
            # the design five seconds later, press it again rather than wait for something that was never asked.
            nonlocal pressed
            if time.time() - pressed > 5 and text("#go") == "Start this new design":
                one("#go").click()
                pressed = time.time()
            return False

        trained = wait_until(new_design_trains, 90, every=0.5)
        check(trained, "the new design trains too" if trained else
              f"the new design trains too (after 90 seconds the button says {text('#go')!r} and the status says {text('#status')[:80]!r})")
        reports("the new design")
        problems = js("return window.__fpErrors")
        # Safari on macOS reports an inline-style violation for every <select> a page creates, before the page has
        # touched it: the report comes from Safari's own drawing of the drop-down. Prove that here with a bare
        # element the site never sees, and only then leave those reports out. Anything else still fails.
        js("window.__fpProbe = window.__fpErrors.length; document.body.append(document.createElement('select')); document.body.lastChild.remove();")
        time.sleep(0.5)
        quirk = js("return window.__fpErrors.length - window.__fpProbe") > 0
        own = [p for p in problems if not (quirk and p.startswith("policy: style-src-attr inline") and "<select" in p)]
        if quirk:
            print(f"       (Safari reports an inline-style violation for a bare document.createElement('select'); "
                  f"{len(problems) - len(own)} such reports on the builder's drop-downs are Safari's own and are not counted)", flush=True)
        check(not own, "no script errors, and nothing of the site's own blocked by the page's security policy" + ("" if not own else ": " + " | ".join(own[:3])))
        shot("end")

        # ---- the self-check page (docs/check.html) in the same Safari
        driver.get(url + "check.html")
        done = wait_until(lambda: "automatic checks" in text("#verdict"), 120)
        verdict = text("#verdict") if done else "no verdict after two minutes"
        check(done and verdict.startswith("All 9 automatic checks worked"), "the self-check page passes its own checks in Safari: " + verdict)
        if not (done and verdict.startswith("All 9")):
            print("       " + js("return document.getElementById('summary').value").replace("\n", "\n       "), flush=True)
        shot("selfcheck")
    finally:
        driver.quit()
        server.shutdown()
    print("ALL GOOD" if not failed else f"{failed} check(s) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
