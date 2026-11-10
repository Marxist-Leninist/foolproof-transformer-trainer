"""The device self-check page (docs/check.html), run by a browser that this script does not control.

docs/check.html tests the builder and the in-page network on whatever opens it. Opened with ?report=1 it
also posts its result back to the server it came from. This script is that server: it serves docs/,
waits for the result, prints it and exits 0 only if every automatic check passed.

That makes it usable for browsers no test library can drive, such as Safari in an iPhone simulator or
Chrome on a phone that is plugged in:

    python tests/web/device_check.py serve --port 8377 --out report.json
    ... then open  http://127.0.0.1:8377/check.html?auto=1&report=1  in the browser under test

For a quick check of the page itself in an ordinary engine (needs: pip install playwright):

    python tests/web/device_check.py playwright chromium      or firefox, or webkit

A finger drag is the one thing the page cannot check by itself, so it is reported but never required here.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
MAX_REPORT = 200_000


class Handler(http.server.SimpleHTTPRequestHandler):
    reports: list = []
    arrived = threading.Event()

    def log_message(self, *args):  # noqa: D102
        pass

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0].rstrip("/").endswith("__report"):
            length = int(self.headers.get("Content-Length") or 0)
            if 0 < length <= MAX_REPORT:
                try:
                    data = json.loads(self.rfile.read(length).decode("utf-8"))
                    if isinstance(data, dict) and isinstance(data.get("checks"), list):
                        Handler.reports.append(data)
                        Handler.arrived.set()
                        self.send_response(204)
                        self.end_headers()
                        return
                except (ValueError, UnicodeDecodeError):
                    pass
            self.send_response(400)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()


class Local(http.server.ThreadingHTTPServer):
    def server_bind(self):
        # Skip HTTPServer's reverse name lookup: on macOS it puts up a "find devices on local networks" prompt.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def start(host: str, port: int) -> http.server.ThreadingHTTPServer:
    Handler.reports = []
    Handler.arrived = threading.Event()
    server = Local((host, port), functools.partial(Handler, directory=str(DOCS)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def wait_for_report(timeout: float) -> dict | None:
    """The report the page posts when its automatic checks finish, or None if none arrives in time."""
    if not Handler.arrived.wait(timeout):
        return None
    time.sleep(1.0)  # an update (a finger drag being noticed) may follow straight away; take the latest
    return Handler.reports[-1]


def show(report: dict) -> bool:
    facts = report.get("facts") or {}
    print(f"  browser: {facts.get('Browser says it is', '?')}")
    for key in ("Screen", "Touch", "Loops doing the sums", "Speed"):
        if key in facts:
            print(f"  {key.lower()}: {facts[key]}")
    for c in report.get("checks", []):
        ok = c.get("state") == "ok"
        detail = f" [{c.get('detail')}]" if c.get("detail") else ""
        print(("  ok   " if ok else "  FAIL ") + str(c.get("label")) + detail, flush=True)
    for e in report.get("errors") or []:
        print("  FAIL browser reported: " + str(e))
    print(f"  finger drag: {report.get('finger', 'not tried')} (needs a person; not required here)")
    return bool(report.get("passed"))


def serve(args) -> int:
    server = start(args.host, args.port)
    port = server.server_address[1]
    print(f"device self-check: waiting up to {args.timeout:.0f}s for a browser to open")
    print(f"  http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{port}/check.html?auto=1&report=1", flush=True)
    report = wait_for_report(args.timeout)
    server.shutdown()
    if report is None:
        print("  FAIL no browser reported back in time")
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    ok = show(report)
    if ok and args.expect_wasm and "WebAssembly" not in str((report.get("facts") or {}).get("Loops doing the sums", "")):
        print("  FAIL the fast WebAssembly loops were expected in this browser, but the slow path was used")
        ok = False
    print("ALL GOOD" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def playwright(args) -> int:
    from playwright.sync_api import sync_playwright

    server = start("127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}/check.html?auto=1&report=1"
    print(f"device self-check page in {args.engine}")
    with sync_playwright() as p:
        browser = getattr(p, args.engine).launch()
        page = browser.new_page(viewport={"width": 412, "height": 860})
        page.goto(url)
        report = wait_for_report(args.timeout)
        verdict = page.inner_text("#verdict") if report else ""
        browser.close()
    server.shutdown()
    if report is None:
        print("  FAIL the page did not report back in time")
        return 2
    ok = show(report)
    if ok and "All 9 automatic checks worked" not in verdict:
        print("  FAIL the page's own verdict line disagrees: " + verdict)
        ok = False
    print("ALL GOOD" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("serve", help="serve docs/ and wait for any browser to report")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8377)
    s.add_argument("--timeout", type=float, default=300)
    s.add_argument("--out", help="write the report here as JSON")
    s.add_argument("--expect-wasm", action="store_true", help="fail if the browser fell back to plain JavaScript")
    s.set_defaults(func=serve)
    w = sub.add_parser("playwright", help="open the page with Playwright and wait")
    w.add_argument("engine", choices=["chromium", "firefox", "webkit"])
    w.add_argument("--timeout", type=float, default=180)
    w.set_defaults(func=playwright)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
