"""Runs worker processes for the app and turns what they report into one event stream.

There is at most one "job" (preparing data or training) at a time, plus an
optional "chat" worker that keeps a finished model loaded for the Chat screen.
"""
from __future__ import annotations

import collections
import multiprocessing as mp
import queue
import threading
import time
import uuid

from . import paths, worker


class Manager:
    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.lock = threading.RLock()
        self.cond = threading.Condition()
        self.events: collections.deque = collections.deque(maxlen=6000)
        self.seq = 0
        self.boot = uuid.uuid4().hex[:12]
        self.job: dict | None = None      # the running prepare/train job
        self.last_job: dict | None = None  # how the previous one ended (for the UI after a reload)
        self.chat: dict | None = None
        self.live: dict = {}               # latest numbers of the current/last training run
        self.hw: dict | None = None
        self.on_job_end = None             # callback(job_info, event)

    # ------------------------------------------------------------------ events
    def push(self, ev: dict) -> None:
        with self.cond:
            self.seq += 1
            ev["seq"] = self.seq
            self.events.append(ev)
            self.cond.notify_all()

    def wait_events(self, since: int, timeout: float = 20.0):
        deadline = time.time() + timeout
        with self.cond:
            if since > self.seq:
                return {"reset": True, "seq": self.seq, "boot": self.boot, "events": []}
            while self.seq <= since:
                left = deadline - time.time()
                if left <= 0:
                    break
                self.cond.wait(left)
            evs = [e for e in self.events if e["seq"] > since]
            oldest = self.events[0]["seq"] if self.events else self.seq + 1
            return {"events": evs, "seq": self.seq, "boot": self.boot,
                    "reset": bool(evs) and since + 1 < oldest}

    # ------------------------------------------------------------------ processes
    def _spawn(self, job: dict) -> dict:
        cmd_q, evt_q = self.ctx.Queue(), self.ctx.Queue()
        proc = self.ctx.Process(target=worker.main, args=(job, cmd_q, evt_q), daemon=True)
        proc.start()
        info = {"id": job["id"], "kind": job["kind"], "proc": proc, "cmd_q": cmd_q, "evt_q": evt_q,
                "started": time.time(), "run": job.get("run"), "ended": False,
                "name": job.get("name"), "spec": job.get("spec")}
        t = threading.Thread(target=self._reader, args=(info,), daemon=True)
        t.start()
        return info

    def _reader(self, info: dict) -> None:
        proc, evt_q = info["proc"], info["evt_q"]
        while True:
            try:
                ev = evt_q.get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    try:  # one last look: events can still be in the pipe
                        ev = evt_q.get(timeout=0.5)
                    except queue.Empty:
                        break
                else:
                    continue
            except (EOFError, OSError):
                break
            self._on_event(info, ev)
        if not info["ended"]:
            code = proc.exitcode
            if info["kind"] != "serve" or not info.get("stopping"):
                self._on_event(info, {"type": "error", "job": info["id"],
                                      "msg": "The worker stopped unexpectedly"
                                             + (" (your computer may have run out of memory)."
                                                if code not in (0, None) else "."),
                                      "detail": f"exit code {code}"})
            self._on_event(info, {"type": "job_end", "job": info["id"], "kind": info["kind"],
                                  "reason": "crashed"})

    def _on_event(self, info: dict, ev: dict) -> None:
        t = ev.get("type")
        kind = info["kind"]
        with self.lock:
            if kind == "train":
                live = self.live
                if t == "train_start":
                    live.clear()
                    live.update({"run": info["run"], "start": ev, "notices": [], "paused": False,
                                 "status": "training"})
                elif t == "metrics":
                    live["metrics"] = ev
                elif t == "eval":
                    live["eval"] = ev
                elif t == "sample":
                    live["sample"] = ev
                elif t == "notice":
                    live.setdefault("notices", []).append(ev)
                    del live["notices"][:-20]
                elif t == "paused":
                    live["paused"] = bool(ev.get("paused"))
                elif t == "budget":
                    live["budget_sec"] = ev.get("budget_sec")
                elif t == "train_end":
                    live["end"] = ev
                    live["status"] = "ended"
            if kind == "prepare" and t == "prep":
                info["prep"] = ev
            if t == "error":
                info["error"] = ev
            if t == "serve_ready":
                info["ready"] = True
            if t == "job_end":
                info["ended"] = True
                if kind == "serve":
                    if self.chat is info:
                        self.chat = None
                else:
                    if self.job is info:
                        self.job = None
                    self.last_job = {"id": info["id"], "kind": kind, "run": info.get("run"),
                                     "reason": ev.get("reason"), "dataset": ev.get("dataset"),
                                     "error": (info.get("error") or {}).get("msg"), "ended": time.time()}
        self.push(ev)
        if t == "job_end" and kind != "serve" and self.on_job_end:
            try:
                self.on_job_end(info, ev)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ public API
    def busy(self) -> bool:
        with self.lock:
            return self.job is not None

    def start(self, job: dict) -> dict:
        with self.lock:
            if self.job is not None:
                raise RuntimeError("Something is already running. Stop it first, or wait for it to finish.")
            job.setdefault("home", str(paths.home()))
            if job["kind"] == "train":
                self.stop_chat()  # free the graphics card for training
                self.live.clear()
                self.live.update({"run": job.get("run"), "status": "starting", "notices": []})
            self.last_job = None
            self.job = self._spawn(job)
            self.push({"type": "job_start", "job": job["id"], "kind": job["kind"], "run": job.get("run")})
            return self.job

    def command(self, cmd: dict) -> bool:
        with self.lock:
            if self.job is None:
                return False
            self.job["cmd_q"].put(cmd)
            return True

    def stop(self, grace: float = 90.0) -> bool:
        with self.lock:
            info = self.job
            if info is None:
                return False
            if info.get("stopping"):
                return True
            info["stopping"] = True
            info["cmd_q"].put({"cmd": "stop"})

        def reap():
            info["proc"].join(grace)
            if info["proc"].is_alive():
                info["proc"].terminate()

        threading.Thread(target=reap, daemon=True).start()
        return True

    def stop_chat(self) -> None:
        with self.lock:
            info, self.chat = self.chat, None
        if info is None:
            return
        info["stopping"] = True
        try:
            info["cmd_q"].put({"cmd": "stop"})
        except Exception:  # noqa: BLE001
            pass
        info["proc"].join(8)
        if info["proc"].is_alive():
            info["proc"].terminate()

    def generate(self, req: dict) -> None:
        """Route a "write something" request to whichever process holds that model."""
        slug = req["slug"]
        with self.lock:
            if self.job is not None and self.job["kind"] == "train" and self.job.get("run") == slug:
                self.job["cmd_q"].put({"cmd": "generate", **req})
                return
            if self.job is not None and self.job["kind"] == "train" and (self.hw or {}).get("device") != "cpu":
                raise RuntimeError("Another AI is training right now and needs the graphics card. "
                                   "You can chat with this one when it finishes.")
            chat = self.chat
            if chat is None or chat.get("run") != slug or not chat["proc"].is_alive():
                if chat is not None:
                    self.stop_chat()
                job = {"kind": "serve", "id": f"serve-{slug}-{int(time.time())}", "run": slug,
                       "run_dir": str(paths.models_dir() / slug), "hw": self.hw or {},
                       "home": str(paths.home())}
                chat = self.chat = self._spawn(job)
            chat["cmd_q"].put({"cmd": "generate", **req})

    def stop_generate(self) -> None:
        with self.lock:
            for info in (self.job, self.chat):
                if info is not None and info["kind"] in ("train", "serve"):
                    info["cmd_q"].put({"cmd": "gen_stop"})

    def snapshot(self) -> dict:
        with self.lock:
            job = None
            if self.job is not None:
                j = self.job
                job = {"id": j["id"], "kind": j["kind"], "run": j.get("run"), "started": j["started"],
                       "stopping": bool(j.get("stopping")), "prep": j.get("prep"), "name": j.get("name")}
            return {"job": job, "last_job": self.last_job, "live": dict(self.live), "seq": self.seq,
                    "boot": self.boot, "chat_run": self.chat.get("run") if self.chat else None}

    def shutdown(self) -> None:
        self.stop_chat()
        with self.lock:
            info = self.job
        if info is not None:
            try:
                info["cmd_q"].put({"cmd": "stop"})
            except Exception:  # noqa: BLE001
                pass
            info["proc"].join(30)
            if info["proc"].is_alive():
                info["proc"].terminate()
