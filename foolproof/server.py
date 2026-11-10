"""The local web server behind the window.

Plain standard-library HTTP: serves the page, a small JSON API, and a long-poll
event stream.  It only listens on this computer (127.0.0.1) unless started with
--lan, checks the Host header, and every API call needs the per-launch token.
"""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import APP_NAME, REPO_URL, __version__, data, hub, paths, presets, runs
from .jobs import Manager

MAX_JSON = 8 << 20


class ApiError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


class App:
    def __init__(self, port: int, lan: bool = False):
        self.port = port
        self.lan = lan
        self.token = secrets.token_urlsafe(18)
        self.mgr = Manager()
        self.mgr.on_job_end = self._job_ended
        self.httpd: ThreadingHTTPServer | None = None
        self.started = time.time()
        self.hw_ready = threading.Event()
        self.settings = paths.read_json(paths.home() / "settings.json", {}) or {}
        data.clean_partials()
        runs.mark_interrupted()
        threading.Thread(target=self._detect_hw, daemon=True).start()

    # ------------------------------------------------------------------ hardware
    def _detect_hw(self):
        hw = None
        try:
            out = subprocess.run([sys.executable, "-m", "foolproof.hardware"], capture_output=True,
                                 text=True, timeout=180, cwd=str(paths.PACKAGE_DIR.parent),
                                 **_no_window())
            line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
            hw = json.loads(line[-1]) if line else None
        except Exception:  # noqa: BLE001
            hw = None
        if not hw:
            hw = {"os": sys.platform, "cpu": "", "cores": os.cpu_count() or 1, "ram_gb": 0, "device": "cpu",
                  "gpu": None, "vram_gb": 0, "torch": None, "bf16": False,
                  "problem": "I could not check this computer's hardware, so I will play safe and use the main processor.",
                  "summary": "Main processor (CPU)"}
        self.mgr.hw = hw
        self.hw_ready.set()
        self.mgr.push({"type": "hw", "hw": hw})

    def hw(self) -> dict:
        self.hw_ready.wait(200)
        return self.mgr.hw or {}

    # ------------------------------------------------------------------ helpers
    def _job_ended(self, info: dict, ev: dict):
        spec = info.get("spec") or {}
        session = (spec.get("source") or {}).get("session")
        if info["kind"] == "prepare" and session:
            shutil.rmtree(paths.uploads_dir() / paths.slugify(session), ignore_errors=True)

    def save_settings(self):
        paths.write_json_atomic(paths.home() / "settings.json", self.settings)

    def state(self) -> dict:
        snap = self.mgr.snapshot()
        active = (snap["job"] or {}).get("run")
        models = runs.list_all()
        for m in models:
            if m["slug"] != active and m.get("status") in ("training", "paused"):
                m["status"] = "stopped"
        return {
            "app": APP_NAME, "version": __version__, "repo": REPO_URL,
            "home": str(paths.home()), "hw": self.mgr.hw,
            "budgets": presets.BUDGETS, "starter": presets.STARTER, "groups": presets.GROUPS,
            "wiki_langs": presets.WIKI_LANGS,
            "default_tokens": presets.default_token_budget(self.mgr.hw or {}),
            "datasets": data.list_datasets(), "models": models,
            "settings": {"has_hf_token": bool(self.settings.get("hf_token"))},
            **snap,
        }

    def dataset(self, slug: str) -> dict:
        for d in data.list_datasets():
            if d["slug"] == slug:
                return d
        raise ApiError("That data is no longer there. Prepare it again.", 404)

    def model(self, slug: str) -> dict:
        m = runs.get(paths.slugify(slug))
        if not m:
            raise ApiError("That AI is no longer there.", 404)
        return m

    # ------------------------------------------------------------------ API: GET
    def api_get(self, path: str, q: dict):
        if path == "/api/state":
            return self.state()
        if path == "/api/events":
            return self.mgr.wait_events(int(q.get("since", 0) or 0), float(q.get("wait", 20) or 20))
        if path == "/api/recommend":
            ds = self.dataset(q.get("dataset", ""))
            hw = self.hw()
            rec = presets.recommend(hw, int(ds["n_train"]), int(ds["vocab_size"]))
            return {"cards": presets.preset_cards(int(ds["vocab_size"])), **rec,
                    "dropout": presets.dropout_for(int(ds["n_train"]))}
        if path == "/api/hf/search":
            term = (q.get("q") or "").strip()
            if not term:
                return {"results": []}
            return {"results": self._hub(lambda: hub.search(term, 24, self.settings.get("hf_token")))}
        if path == "/api/hf/inspect":
            return self._hub(lambda: hub.inspect(q.get("id", ""), q.get("config") or None,
                                                 q.get("split") or None, self.settings.get("hf_token")))
        if path == "/api/model":
            m = self.model(q.get("slug", ""))
            m["history"] = runs.history(m["slug"])
            return m
        raise ApiError("Not found", 404)

    @staticmethod
    def _hub(fn):
        try:
            return fn()
        except hub.HubError as e:
            raise ApiError(str(e), 502) from e

    # ------------------------------------------------------------------ API: POST
    def api_post(self, path: str, body: dict, q: dict):
        mgr = self.mgr
        if path == "/api/prepare":
            src = dict(body.get("source") or {})
            if src.get("type") == "files" and src.get("session"):
                folder = paths.uploads_dir() / paths.slugify(src["session"])
                if not folder.is_dir() or not any(folder.iterdir()):
                    raise ApiError("No files arrived. Drop them in again.")
                src["paths"] = [str(folder)]
            if src.get("type") == "folder":
                p = Path(str(src.get("path", "")).strip().strip('"')).expanduser()
                if not p.exists():
                    raise ApiError("I cannot find that folder. Check the path and try again.")
                src = {"type": "files", "paths": [str(p)]}
            if src.get("type") == "text" and len((src.get("text") or "").strip()) < 200:
                raise ApiError("That is too little text to learn from. Paste at least a few pages.")
            if src.get("type") not in ("builtin", "hf", "files", "text"):
                raise ApiError("Pick something to learn from first.")
            spec = {"name": (body.get("name") or "My data").strip()[:80], "source": src,
                    "tokenizer": body.get("tokenizer") or "auto",
                    "max_tokens": int(body.get("max_tokens") or 0),
                    "hf_token": self.settings.get("hf_token")}
            job = {"kind": "prepare", "id": f"prep-{int(time.time() * 1000)}", "spec": spec,
                   "name": spec["name"]}
            self._start(job)
            return {"ok": True, "job": job["id"]}
        if path == "/api/job/stop":
            return {"ok": mgr.stop()}
        if path == "/api/train":
            ds = self.dataset(body.get("dataset", ""))
            hw = self.hw()
            try:
                job = runs.create((body.get("name") or "").strip()[:60], ds, body.get("preset") or "small",
                                  float(body.get("budget_sec") or 0), hw, body.get("overrides") or {},
                                  bool(body.get("autostop", True)), architecture=body.get("architecture"))
            except ValueError as e:
                raise ApiError(str(e)) from e
            try:
                self._start(job)
            except ApiError:
                runs.delete(job["run"])
                raise
            return {"ok": True, "run": job["run"]}
        if path == "/api/train/continue":
            m = self.model(body.get("slug", ""))
            if not m.get("can_resume"):
                raise ApiError("This AI cannot be continued (its saved progress or its data is missing).")
            job = runs.job_for(m, self.hw(), resume=True, budget_sec=float(body.get("budget_sec") or 0))
            self._start(job)
            return {"ok": True, "run": job["run"]}
        if path in ("/api/train/pause", "/api/train/resume"):
            return {"ok": mgr.command({"cmd": path.rsplit("/", 1)[1]})}
        if path == "/api/train/add_time":
            return {"ok": mgr.command({"cmd": "add_time", "seconds": float(body.get("seconds") or 0)})}
        if path == "/api/train/prompt":
            return {"ok": mgr.command({"cmd": "set_prompt", "prompt": str(body.get("prompt") or "")[:500]})}
        if path == "/api/generate":
            m = self.model(body.get("slug", ""))
            if not m.get("has_model") and not (mgr.snapshot()["job"] or {}).get("run") == m["slug"]:
                raise ApiError("This AI has not learned anything yet. Train it first.")
            req = {"slug": m["slug"], "id": str(body.get("id") or f"g{int(time.time() * 1000)}"),
                   "prompt": str(body.get("prompt") or "")[:20000],
                   "history": [(str(r), str(t)[:8000]) for r, t in (body.get("history") or [])][-40:],
                   "mode": body.get("mode") if body.get("mode") in ("chat", "text") else None,
                   "max_new_tokens": int(body.get("max_new_tokens") or 200),
                   "temperature": float(body.get("temperature", 0.8)),
                   "top_k": int(body.get("top_k") or 50)}
            try:
                mgr.generate(req)
            except RuntimeError as e:
                raise ApiError(str(e), 409) from e
            return {"ok": True, "id": req["id"]}
        if path == "/api/generate/stop":
            mgr.stop_generate()
            return {"ok": True}
        if path == "/api/model/delete":
            m = self.model(body.get("slug", ""))
            snap = mgr.snapshot()
            if (snap["job"] or {}).get("run") == m["slug"]:
                raise ApiError("This AI is training right now. Stop it first.")
            if snap.get("chat_run") == m["slug"]:
                mgr.stop_chat()
            runs.delete(m["slug"])
            return {"ok": True}
        if path == "/api/model/rename":
            m = self.model(body.get("slug", ""))
            name = (body.get("name") or "").strip()[:60]
            if not name:
                raise ApiError("Give it a name.")
            if (mgr.snapshot()["job"] or {}).get("run") == m["slug"]:
                raise ApiError("Rename it after training has finished.")
            cfg = paths.read_json(runs.folder(m["slug"]) / "config.json")
            cfg["name"] = name
            paths.write_json_atomic(runs.folder(m["slug"]) / "config.json", cfg)
            return {"ok": True}
        if path == "/api/dataset/delete":
            ds = self.dataset(body.get("slug", ""))
            runs.delete_dataset(ds["slug"])
            return {"ok": True}
        if path == "/api/model/export":
            m = self.model(body.get("slug", ""))
            folder = self._export(m["slug"], "folder")
            _open_folder(folder)
            return {"ok": True, "path": str(folder)}
        if path == "/api/model/upload":
            m = self.model(body.get("slug", ""))
            token = (body.get("token") or self.settings.get("hf_token") or "").strip()
            if not token:
                raise ApiError("Paste your Hugging Face token first.")
            repo = paths.slugify(body.get("repo") or m["slug"], "my-ai")
            folder = self._export(m["slug"], "folder")
            url = self._hub(lambda: hub.upload_folder(folder, repo, token, bool(body.get("private"))))
            if body.get("remember"):
                self.settings["hf_token"] = token
                self.save_settings()
            return {"ok": True, "url": url}
        if path == "/api/open":
            what = body.get("what")
            target = paths.home()
            if what == "model":
                target = runs.folder(self.model(body.get("slug", ""))["slug"])
            elif what == "models":
                target = paths.models_dir()
            _open_folder(target)
            return {"ok": True, "path": str(target)}
        if path == "/api/settings":
            if "hf_token" in body:
                tok = (body.get("hf_token") or "").strip()
                if tok:
                    self.settings["hf_token"] = tok
                else:
                    self.settings.pop("hf_token", None)
                self.save_settings()
            return {"ok": True, "has_hf_token": bool(self.settings.get("hf_token"))}
        if path == "/api/quit":
            threading.Thread(target=self.shutdown, daemon=True).start()
            return {"ok": True}
        raise ApiError("Not found", 404)

    def _start(self, job: dict):
        self.hw()  # make sure the hardware check has finished
        job["hw"] = self.mgr.hw or {}
        try:
            self.mgr.start(job)
        except RuntimeError as e:
            raise ApiError(str(e), 409) from e

    def _export(self, slug: str, mode: str) -> Path:
        out = subprocess.run([sys.executable, "-m", "foolproof.export", str(runs.folder(slug)), mode],
                             capture_output=True, text=True, timeout=1800,
                             cwd=str(paths.PACKAGE_DIR.parent), **_no_window())
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
        res = json.loads(line[-1]) if line else {"ok": False, "error": (out.stderr or "")[-300:]}
        if not res.get("ok"):
            raise ApiError("Could not package this AI: " + str(res.get("error")), 500)
        return Path(res["path"])

    # ------------------------------------------------------------------ lifecycle
    def shutdown(self):
        time.sleep(0.3)
        try:
            self.mgr.shutdown()
        finally:
            try:
                (paths.home() / ".server.json").unlink()
            except OSError:
                pass
            if self.httpd is not None:
                self.httpd.shutdown()


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows "reuse address" would let two copies share one port: refuse instead, and move on to the next.
    allow_reuse_address = sys.platform != "win32"


def _no_window() -> dict:
    """On Windows, keep helper processes from flashing a console window."""
    if sys.platform == "win32":
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


def _open_folder(path: Path) -> None:
    path = Path(path)
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001 - no file manager (e.g. a server): the path is shown in the app anyway
        pass


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "Foolproof"

        def log_message(self, *args):  # keep the console quiet
            pass

        # ---- plumbing ----------------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            if app.lan:
                return True  # reachable by IP on the home network; the token is the lock
            return host in (f"127.0.0.1:{app.port}", f"localhost:{app.port}")

        def _token_ok(self, q: dict) -> bool:
            tok = self.headers.get("X-FP-Token") or q.get("t") or ""
            return secrets.compare_digest(str(tok), app.token)

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            host = (self.headers.get("Host") or "").lower()
            return urllib.parse.urlparse(origin).netloc.lower() == host

        def _route(self):
            parsed = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            return parsed.path, q

        # ---- GET ---------------------------------------------------------------------
        def do_GET(self):
            try:
                path, q = self._route()
                if not self._host_ok():
                    return self._send(403, b"Forbidden", "text/plain")
                if path.startswith("/api/"):
                    if not self._token_ok(q):
                        return self._json({"error": "This window is out of date. Close it and start the app again."}, 403)
                    if path == "/api/download":
                        return self._download(q)
                    return self._json(app.api_get(path, q))
                if path in ("/", "/index.html"):
                    if app.lan and not self._token_ok(q):
                        return self._send(403, b"Open the full link shown in the app window (it ends with ?t=...).",
                                          "text/plain")
                    html = (paths.WEB_DIR / "index.html").read_text(encoding="utf-8")
                    html = html.replace("__FP_TOKEN__", app.token).replace("__FP_VERSION__", __version__)
                    return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                                      {"Content-Security-Policy":
                                       "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                                       "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})
                rel = path.lstrip("/")
                target = (paths.WEB_DIR / rel).resolve()
                if paths.WEB_DIR.resolve() not in target.parents or not target.is_file():
                    return self._send(404, b"Not found", "text/plain")
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if target.suffix == ".js":
                    ctype = "text/javascript; charset=utf-8"
                elif target.suffix == ".css":
                    ctype = "text/css; charset=utf-8"
                return self._send(200, target.read_bytes(), ctype)
            except ApiError as e:
                self._json({"error": str(e)}, e.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                try:
                    self._json({"error": f"Something went wrong inside the app ({type(e).__name__}: {e})"}, 500)
                except Exception:  # noqa: BLE001
                    pass

        do_HEAD = do_GET

        def _download(self, q: dict):
            m = app.model(q.get("slug", ""))
            archive = app._export(m["slug"], "zip")
            size = archive.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{archive.name}"')
            self.end_headers()
            with open(archive, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        # ---- POST --------------------------------------------------------------------
        def do_POST(self):
            try:
                path, q = self._route()
                if not self._host_ok() or not self._origin_ok():
                    return self._send(403, b"Forbidden", "text/plain")
                if not path.startswith("/api/") or not self._token_ok(q):
                    return self._json({"error": "This window is out of date. Close it and start the app again."}, 403)
                length = int(self.headers.get("Content-Length") or 0)
                if path == "/api/upload":
                    return self._upload(q, length)
                if length > MAX_JSON:
                    return self._json({"error": "That is too much text to paste. Save it as a file and drop the file in instead."}, 413)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(body, dict):
                    raise ApiError("Bad request")
                self._json(app.api_post(path, body, q))
            except ApiError as e:
                self._json({"error": str(e)}, e.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                try:
                    self._json({"error": f"Something went wrong inside the app ({type(e).__name__}: {e})"}, 500)
                except Exception:  # noqa: BLE001
                    pass

        def _upload(self, q: dict, length: int):
            session = paths.slugify(q.get("session") or "", "")
            name = Path(urllib.parse.unquote(q.get("name") or "file")).name
            name = "".join(c for c in name if c.isalnum() or c in " ._-()[]+,'&")[:120].strip() or "file"
            if not session:
                raise ApiError("Bad upload")
            folder = paths.uploads_dir() / session
            folder.mkdir(parents=True, exist_ok=True)
            if paths.free_disk_bytes(folder) < length + (200 << 20):
                self.rfile.read(0)
                raise ApiError("Your disk is too full to take that file.", 507)
            dest, n = folder / name, 2
            while dest.exists():  # two dropped files with the same name: keep both
                dest = folder / f"{Path(name).stem}-{n}{Path(name).suffix}"
                n += 1
            left = length
            with open(dest, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left > 0:
                dest.unlink(missing_ok=True)
                raise ApiError("The file did not arrive completely. Try again.")
            self._json({"ok": True, "name": dest.name, "size": length})

    return Handler


def serve(port: int = 8765, lan: bool = False, open_browser: bool = True) -> int:
    """Start the app and block until it is closed. Returns a process exit code."""
    import socket
    import urllib.request
    import webbrowser

    lock = paths.home() / ".server.json"
    old = paths.read_json(lock)
    if old:  # already running? then just bring its window back
        try:
            url = f"http://127.0.0.1:{old['port']}/api/state?t={old['token']}"
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    link = f"http://127.0.0.1:{old['port']}/"
                    print(f"{APP_NAME} is already running. Opening its window: {link}", flush=True)
                    if open_browser:
                        webbrowser.open(link)
                    return 0
        except Exception:  # noqa: BLE001
            pass

    host = "0.0.0.0" if lan else "127.0.0.1"
    httpd = None
    for p in range(port, port + 40):
        try:
            httpd = LocalServer((host, p), BaseHTTPRequestHandler)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        print("Could not find a free port to start on. Restart your computer and try again.", flush=True)
        return 1
    app = App(port, lan)
    app.httpd = httpd
    httpd.RequestHandlerClass = make_handler(app)
    paths.write_json_atomic(lock, {"port": port, "token": app.token, "pid": os.getpid()})

    link = f"http://127.0.0.1:{port}/"
    print("", flush=True)
    print(f"  {APP_NAME} {__version__} is running.", flush=True)
    print(f"  Your browser should open by itself. If it does not, open:  {link}", flush=True)
    if lan:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
        except OSError:
            ip = "<this computer's address>"
        print(f"  From a phone on the same Wi-Fi:  http://{ip}:{port}/?t={app.token}", flush=True)
    print("  Keep this window open while you use the app. To quit, close it (or press Ctrl+C).", flush=True)
    print(f"  Your AIs and data are kept in:  {paths.home()}", flush=True)
    print("", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(link)).start()
    try:
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n  Closing… saving anything that is training.", flush=True)
        app.mgr.shutdown()
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
        httpd.server_close()
    return 0
