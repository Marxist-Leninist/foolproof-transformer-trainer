"""The worker process: prepares data, trains, and answers "write something" requests.

It runs separately from the window so that the window never freezes, Stop always
works, and a crash here (say, the graphics card running out of memory) cannot
take the app down.  It talks to the app through two queues: commands in,
events out.
"""
from __future__ import annotations

import gc
import json
import math
import os
import queue
import time
import traceback
from pathlib import Path

from . import paths
from .data import stage_for

LN2 = math.log(2)

def friendly_error(e: BaseException) -> str:
    text = f"{type(e).__name__}: {e}"
    low = text.lower()
    if "out of memory" in low or isinstance(e, MemoryError):
        return ("This brain is too big for your computer's memory. "
                "Pick a smaller brain size and try again.")
    if isinstance(e, FileNotFoundError):
        return "A file this AI needs has gone missing (was its folder moved or deleted?)."
    if "no space left" in low or "disk full" in low or "errno 28" in low:
        return "Your disk is full. Free up some space and try again."
    if "cuda" in low and ("driver" in low or "initialization" in low):
        return ("Your graphics card could not be started. Updating the NVIDIA driver usually "
                "fixes this. You can also train on the main processor instead.")
    return "Something went wrong: " + text[:300]


def is_oom(e: BaseException) -> bool:
    if isinstance(e, MemoryError):
        return True
    low = str(e).lower()
    return "out of memory" in low or "can't allocate memory" in low or "not enough memory" in low


def main(job: dict, cmd_q, evt_q) -> None:
    """Entry point (runs in the child process)."""
    if job.get("home"):
        os.environ["FOOLPROOF_HOME"] = job["home"]
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Macs: borrow the CPU for the odd missing operation

    def emit(ev: dict) -> None:
        ev.setdefault("ts", time.time())
        ev.setdefault("job", job.get("id"))
        evt_q.put(ev)

    _watch_parent()
    kind = job.get("kind")
    try:
        if kind == "prepare":
            from . import data

            def should_stop() -> bool:
                try:
                    while True:
                        if cmd_q.get_nowait().get("cmd") == "stop":
                            should_stop.flag = True
                except queue.Empty:
                    pass
                return should_stop.flag

            should_stop.flag = False
            try:
                meta = data.prepare(job["spec"], emit, should_stop)
                emit({"type": "job_end", "kind": kind, "reason": "done", "dataset": meta["slug"]})
            except data.Cancelled:
                emit({"type": "job_end", "kind": kind, "reason": "stopped"})
            except data.DataError as e:
                emit({"type": "error", "msg": str(e), "detail": ""})
                emit({"type": "job_end", "kind": kind, "reason": "error"})
        elif kind == "train":
            Trainer(job, cmd_q, emit).run()
        elif kind == "serve":
            Chatter(job, cmd_q, emit).run()
        else:
            raise ValueError(f"unknown job kind {kind!r}")
    except KeyboardInterrupt:
        emit({"type": "job_end", "kind": kind, "reason": "stopped"})
    except BaseException as e:  # noqa: BLE001 - last line of defence, report everything
        emit({"type": "error", "msg": friendly_error(e), "detail": traceback.format_exc()[-3000:]})
        emit({"type": "job_end", "kind": kind, "reason": "error"})


def _watch_parent() -> None:
    """If the app is killed, do not keep training in the background holding the graphics card."""
    import multiprocessing as mp
    import threading

    parent = mp.parent_process()
    if parent is None:  # running in-process (self-test)
        return

    def watch():
        while True:
            time.sleep(2.0)
            if not parent.is_alive():
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


# =============================================================================== shared

class ModelHost:
    """Things both training and chatting need: device, tokenizer, text generation."""

    def __init__(self, job: dict, cmd_q, emit):
        self.job, self.cmd_q, self.emit = job, cmd_q, emit
        self.pending: list[dict] = []
        self.model = None
        self.tok = None
        self.device = "cpu"
        self.chat = False

    def setup_device(self):
        import torch

        hw = self.job.get("hw") or {}
        dev = hw.get("device", "cpu")
        if dev == "cuda" and torch.cuda.is_available():
            self.device = f"cuda:{int(hw.get('gpu_index') or 0)}"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif dev == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        self.device_type = self.device.split(":")[0]
        self.amp_dtype = None
        if self.device_type == "cuda":
            self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(hw.get("amp"))

    def autocast(self):
        import contextlib

        import torch

        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def next_command(self, block: float = 0.0):
        if self.pending:
            return self.pending.pop(0)
        try:
            return self.cmd_q.get(timeout=block) if block > 0 else self.cmd_q.get_nowait()
        except queue.Empty:
            return None

    def build_prompt(self, req: dict):
        """Request -> list of token ids to start from, plus the ids that should end the reply."""
        tok = self.tok
        if req.get("mode") == "chat" or (req.get("mode") is None and self.chat):
            ids: list[int] = []
            for role, text in req.get("history") or []:
                if role in ("user", "assistant", "system") and text:
                    ids.append(tok.role_id(role))
                    ids.extend(tok.encode(text))
            ids.append(tok.user_id)
            ids.extend(tok.encode(req.get("prompt") or ""))
            ids.append(tok.assistant_id)
            return ids, (tok.eot_id, tok.user_id, tok.system_id)
        ids = tok.encode(req.get("prompt") or "")
        if not ids:
            ids = tok.encode("\n")
        return ids, (tok.eot_id,)

    def generate(self, req: dict) -> None:
        """Stream a reply as 'gen' events. Other commands that arrive meanwhile are kept for later."""
        import torch

        from .tokenizers import StreamDecoder

        gid = req.get("id")
        t0 = time.time()
        n = 0
        reason = "length"
        try:
            ids, stop_ids = self.build_prompt(req)
            block = self.model.cfg.block_size
            ids = ids[-(block - 1):]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            dec = StreamDecoder(self.tok)
            max_new = int(max(1, min(4000, req.get("max_new_tokens") or 200)))
            limit_sec = float(req.get("limit_sec") or 0)
            buf, last_flush = "", time.time()
            with self.autocast():
                for t in self.model.generate(x, max_new, temperature=float(req.get("temperature", 0.8)),
                                             top_k=req.get("top_k", 50), top_p=req.get("top_p"),
                                             stop_ids=stop_ids):
                    n += 1
                    if t in stop_ids:
                        reason = "end"
                        break
                    if t >= self.tok.base_vocab:
                        continue  # a stray special token: do not print it
                    buf += dec.push(t)
                    now = time.time()
                    if buf and now - last_flush > 0.05:
                        self.emit({"type": "gen", "id": gid, "text": buf})
                        buf, last_flush = "", now
                    stop = False
                    while True:  # peek at incoming commands without losing them
                        try:
                            c = self.cmd_q.get_nowait()
                        except queue.Empty:
                            break
                        if c.get("cmd") == "gen_stop":
                            stop = True
                        else:
                            self.pending.append(c)
                            if c.get("cmd") == "stop":
                                stop = True
                    if stop:
                        reason = "stopped"
                        break
                    if limit_sec and now - t0 > limit_sec:
                        reason = "time"
                        break
            buf += dec.flush()
            if buf:
                self.emit({"type": "gen", "id": gid, "text": buf})
        except Exception as e:  # noqa: BLE001
            self.emit({"type": "gen", "id": gid, "text": ""})
            reason = "error: " + friendly_error(e)
        dt = max(1e-6, time.time() - t0)
        self.emit({"type": "gen_end", "id": gid, "reason": reason, "tokens": n,
                   "tok_per_sec": round(n / dt, 1)})


def load_tokenizer(run_dir: Path, ds_dir: Path | None):
    from .tokenizers import Tokenizer

    p = run_dir / "tokenizer.json"
    if not p.exists() and ds_dir is not None:
        p = ds_dir / "tokenizer.json"
    return Tokenizer.load(p)


def save_atomic(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: someone is reading the old file this instant
            time.sleep(0.5)
    os.replace(tmp, path)


# =============================================================================== chatting

class Chatter(ModelHost):
    """Loads a finished model and answers generate requests until told to stop (or idle)."""

    def run(self):
        import torch

        from .model import GPT, ModelConfig

        run_dir = Path(self.job["run_dir"])
        self.setup_device()
        cfg = paths.read_json(run_dir / "config.json") or {}
        self.chat = bool(cfg.get("chat"))
        self.tok = load_tokenizer(run_dir, None)
        ck_path = run_dir / "ckpt_best.pt"
        if not ck_path.exists():
            ck_path = run_dir / "ckpt_last.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        self.model = GPT(ModelConfig.from_dict(ck["model_config"]))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        del ck
        self.emit({"type": "serve_ready", "run": run_dir.name})
        idle_since = time.time()
        while True:
            c = self.next_command(block=1.0)
            if c is None:
                if time.time() - idle_since > 900:
                    break
                continue
            idle_since = time.time()
            if c.get("cmd") == "stop":
                break
            if c.get("cmd") == "generate":
                self.generate(c)
        self.emit({"type": "job_end", "kind": "serve", "reason": "done"})


# =============================================================================== training

class Trainer(ModelHost):
    def run(self):
        import numpy as np
        import torch

        from .model import GPT, ModelConfig

        job = self.job
        self.run_dir = run_dir = Path(job["run_dir"])
        ds_dir = Path(job["dataset_dir"])
        self.cfg_path = run_dir / "config.json"
        self.rc = paths.read_json(self.cfg_path) or {}
        meta = paths.read_json(ds_dir / "meta.json")
        if not meta:
            raise FileNotFoundError("dataset meta.json")
        self.meta = meta
        self.chat = bool(meta.get("chat"))
        self.tok = load_tokenizer(run_dir, ds_dir)
        if not (run_dir / "tokenizer.json").exists():
            self.tok.save(run_dir / "tokenizer.json")
        self.bpt = float(meta.get("bytes_per_token") or 1.0) or 1.0
        self.base = meta.get("baseline") or None
        self.setup_device()

        tr = job["train"]
        mcfg = ModelConfig.from_dict({**job["model"], "vocab_size": self.tok.vocab_size})
        dtype = np.dtype(meta.get("dtype", "uint16"))
        self.train_data = np.memmap(ds_dir / "train.bin", dtype=dtype, mode="r")
        n_val = int(meta.get("n_val") or 0)
        self.val_data = np.memmap(ds_dir / "val.bin", dtype=dtype, mode="r") if n_val > 0 else None
        if len(self.train_data) < mcfg.block_size + 2:
            mcfg.block_size = max(8, len(self.train_data) - 2)
        if self.val_data is not None and len(self.val_data) < mcfg.block_size + 2:
            self.val_data = None
        self.block = mcfg.block_size
        self.np_rng = np.random.default_rng(1337 + int(self.rc.get("sessions") or 0))

        torch.manual_seed(1337 + int(self.rc.get("sessions") or 0))
        self.model = GPT(mcfg)
        n_params = self.model.num_params()
        self.base_lr = float(tr.get("lr") or 1e-3)
        self.opt = None
        self.scaler = None
        self.step = 0
        self.elapsed_before = 0.0
        self.tokens_seen = 0
        self.best_val = float("inf")
        self.ema = None
        self.ema_acc, self.ema_n = 0.0, 0
        resumed = False
        last = run_dir / "ckpt_last.pt"
        if job.get("resume") and last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ck["model"])
            st = ck.get("state") or {}
            self.step = int(st.get("step", 0))
            self.elapsed_before = float(st.get("elapsed", 0.0))
            self.tokens_seen = int(st.get("tokens_seen", 0))
            self.best_val = float(st.get("best_val", float("inf")))
            self.ema = st.get("ema")
            if self.ema is not None:
                self.ema_acc, self.ema_n = float(self.ema), 500
            resumed = True
        self.model.to(self.device)
        self.opt = self.model.optimizer(self.base_lr, float(tr.get("weight_decay", 0.1)),
                                        fused=self.device_type == "cuda")
        if resumed and ck.get("opt"):
            try:
                self.opt.load_state_dict(ck["opt"])
            except Exception:
                pass
        if self.amp_dtype is not None and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda")
            if resumed and ck.get("scaler"):
                try:
                    self.scaler.load_state_dict(ck["scaler"])
                except Exception:
                    pass
        if resumed:
            del ck
        self.peak_lr = self.base_lr * (0.5 if resumed else 1.0)

        # batch: the "effective" batch stays fixed; the micro-batch shrinks to fit memory
        self.eff_batch = int(tr.get("batch") or 32)
        self.micro = self.pick_micro_batch(mcfg)
        self.accum = max(1, math.ceil(self.eff_batch / self.micro))
        self.budget = float(tr.get("budget_sec") or 0)
        self.autostop = bool(tr.get("autostop", True))
        self.min_steps = int(tr.get("min_steps") or 0)  # self-test only: do not stop on time before this
        self.warmup_steps = int(tr.get("warmup_steps") or (30 if resumed else 100))
        self.warmup_sec = max(5.0, min(60.0, 0.03 * self.budget)) if self.budget > 0 else 60.0
        self.test_prompt = tr.get("test_prompt") or None

        self.update_config(status="training", params=n_params, device=self.device_label(),
                           sessions=int(self.rc.get("sessions") or 0) + 1,
                           model=mcfg.to_dict())
        self.emit({"type": "train_start", "run": run_dir.name, "params": n_params,
                   "device": self.device_label(), "micro_batch": self.micro, "accum": self.accum,
                   "block_size": self.block, "resumed": resumed, "budget_sec": self.budget,
                   "n_train": int(len(self.train_data)), "has_val": self.val_data is not None,
                   "random_loss": math.log(self.tok.vocab_size), "bytes_per_token": self.bpt,
                   "step": self.step, "elapsed": self.elapsed_before})
        self.loop()

    # ------------------------------------------------------------------ helpers
    def device_label(self) -> str:
        hw = self.job.get("hw") or {}
        if self.device_type == "cuda":
            return hw.get("gpu") or "graphics card"
        if self.device_type == "mps":
            return "Apple Silicon GPU"
        return "main processor (CPU)"

    def pick_micro_batch(self, mcfg) -> int:
        from . import presets

        want = self.eff_batch
        if self.device_type != "cuda":
            return want
        import torch

        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return want
        budget_gb = free / 2**30 * 0.80
        p = {"n_layer": mcfg.n_layer, "n_embd": mcfg.n_embd, "block_size": mcfg.block_size}
        micro = want
        amp = self.amp_dtype is not None
        while micro > 1 and presets.train_memory_gb(p, mcfg.vocab_size, micro, amp) > budget_gb:
            micro = max(1, micro // 2)
        return micro

    def get_batch(self, split: str, n: int, rng=None):
        import numpy as np
        import torch

        data = self.train_data if split == "train" else self.val_data
        rng = rng or self.np_rng
        ix = rng.integers(0, len(data) - self.block - 1, size=n)
        idx = ix[:, None] + np.arange(self.block + 1)[None, :]
        chunk = torch.from_numpy(data[idx].astype(np.int64))
        x, y = chunk[:, :-1], chunk[:, 1:]
        if self.device_type == "cuda":
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)

    def update_config(self, **kw):
        stats = kw.pop("stats", None)
        self.rc.update(kw)
        if stats:
            self.rc.setdefault("stats", {}).update(stats)
        try:
            paths.write_json_atomic(self.cfg_path, self.rc)
        except OSError:
            pass

    def log_line(self, name: str, obj: dict):
        try:
            with open(self.run_dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def lr_now(self, session_elapsed: float, session_step: int) -> float:
        # warm up by steps on fast machines, by the clock on slow ones (whichever comes first)
        warm = min(1.0, max((session_step + 1) / max(1, self.warmup_steps),
                            session_elapsed / self.warmup_sec))
        decay = 1.0
        if self.budget > 0:
            p = min(1.0, session_elapsed / self.budget)
            decay = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))
        return self.peak_lr * warm * decay

    def shrink(self, why: str):
        import torch

        if self.micro <= 1:
            raise MemoryError("out of memory even with the smallest batch")
        self.micro = max(1, self.micro // 2)
        self.accum = max(1, math.ceil(self.eff_batch / self.micro))
        self.opt.zero_grad(set_to_none=True)
        gc.collect()
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.emit({"type": "notice", "level": "info", "code": "shrink",
                   "msg": f"{why} I made each step smaller (now {self.micro} at a time) and carried on. "
                          "Nothing was lost."})

    def train_step(self) -> float:
        import torch

        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(self.accum):
            x, y = self.get_batch("train", self.micro)
            with self.autocast():
                _, loss = self.model(x, y)
            scaled = loss / self.accum
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total += float(loss.item())
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        return total / self.accum

    def estimate(self, split: str, batches: int) -> float:
        import numpy as np
        import torch

        rng = np.random.default_rng(4242)  # the same test passages every time: comparable numbers
        self.model.eval()
        losses = []
        try:
            with torch.no_grad():
                for _ in range(batches):
                    x, y = self.get_batch(split, self.micro, rng)
                    with self.autocast():
                        _, loss = self.model(x, y)
                    losses.append(float(loss.item()))
        finally:
            self.model.train()
        return sum(losses) / max(1, len(losses))

    def sample_prompt_ids(self):
        """A short, real opening for the model to continue (from text it never trained on)."""
        if self.test_prompt:
            return None
        src = self.val_data if self.val_data is not None else self.train_data
        want = 16 if self.tok.kind == "bytes" else 6
        ids = [int(t) for t in src[: want * 3]]
        ids = [t for t in ids if t < self.tok.base_vocab][:want]
        return ids or None

    def make_sample(self):
        if self.chat or self.test_prompt:
            prompt = self.test_prompt or self.meta.get("sample_prompt") or "Hello!"
            req = {"prompt": prompt, "mode": "chat" if self.chat else "text"}
            shown = prompt
        else:
            ids = self.sample_prompt_ids()
            shown = self.tok.decode(ids) if ids else ""
            req = {"prompt": shown, "mode": "text"}
        n_new = 220 if self.tok.kind == "bytes" else 70
        gid = f"sample-{self.step}"
        req.update({"id": gid, "max_new_tokens": n_new, "temperature": 0.8, "top_k": 50,
                    "limit_sec": 4.0})
        chunks: list[str] = []
        real_emit = self.emit

        def capture(ev):
            if ev.get("type") == "gen" and ev.get("id") == gid:
                chunks.append(ev.get("text", ""))
            elif ev.get("type") != "gen_end":
                real_emit(ev)

        self.emit = capture
        try:
            self.generate(req)
        finally:
            self.emit = real_emit
        text = "".join(chunks)
        rec = {"step": self.step, "t": round(self.total_elapsed(), 1), "prompt": shown, "text": text,
               "chat": bool(self.chat)}
        self.log_line("samples.jsonl", rec)
        self.emit({"type": "sample", **rec})

    def total_elapsed(self) -> float:
        return self.elapsed_before + self.session_elapsed

    def state_dict(self) -> dict:
        return {"step": self.step, "elapsed": self.total_elapsed(), "tokens_seen": self.tokens_seen,
                "best_val": self.best_val, "ema": self.ema}

    def save_last(self):
        save_atomic({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                     "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                     "model_config": self.model.cfg.to_dict(), "state": self.state_dict()},
                    self.run_dir / "ckpt_last.pt")

    def save_best(self, val: float):
        save_atomic({"model": self.model.state_dict(), "model_config": self.model.cfg.to_dict(),
                     "state": {"step": self.step, "val_loss": val}},
                    self.run_dir / "ckpt_best.pt")

    # ------------------------------------------------------------------ main loop
    def handle_commands(self, paused: bool):
        """Process everything waiting in the command queue. Returns (paused, stop_requested)."""
        stop = False
        while True:
            c = self.next_command(block=0.2 if paused else 0.0)
            if c is None:
                break
            cmd = c.get("cmd")
            t0 = time.time()
            if cmd == "stop":
                stop = True
                break
            if cmd == "pause" and not paused:
                paused = True
                self.update_config(status="paused")
                self.emit({"type": "paused", "paused": True})
            elif cmd == "resume" and paused:
                paused = False
                self.update_config(status="training")
                self.emit({"type": "paused", "paused": False})
            elif cmd == "generate":
                self.generate(c)
                self.skip_time += time.time() - t0
            elif cmd == "add_time" and self.budget > 0:
                self.budget += float(c.get("seconds") or 0)
                self.emit({"type": "budget", "budget_sec": self.budget})
            elif cmd == "set_prompt":
                self.test_prompt = (c.get("prompt") or "").strip() or None
                try:
                    self.make_sample()
                except Exception:  # noqa: BLE001
                    pass
                self.skip_time += time.time() - t0
        return paused, stop

    def loop(self):
        import torch

        self.model.train()
        self.session_elapsed = 0.0
        self.skip_time = 0.0  # time spent answering "try it" requests: not counted as training
        session_step = 0
        paused = False
        reason = None
        last_tick = last_metrics = last_ckpt = last_eval = time.time()
        base_eval_every = 12.0 if 0 < self.budget <= 600 else 60.0 if (self.budget == 0 or self.budget > 3600) else 30.0
        eval_every = base_eval_every
        eval_batches = 0  # decided at the first evaluation from how fast this machine is
        first_eval_done = False
        tokens_window, window_t = 0, time.time()
        bad_steps = 0
        worse_evals = 0
        told_overfit = False
        last_best_save = 0.0
        best_save_gap = 0.0 if self.model.num_params() < 30_000_000 else 120.0
        self.saved_best = float((self.rc.get("stats") or {}).get("saved_best") or float("inf")) \
            if self.job.get("resume") else float("inf")
        checked_mem = False
        tps = 0.0
        ckpt_every = 60.0 if 0 < self.budget <= 900 else 300.0
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        while reason is None:
            was_paused = paused
            paused, stop = self.handle_commands(paused)
            if stop:
                reason = "stopped"
                break
            if paused:
                last_tick = time.time()
                continue
            if was_paused:
                last_tick = time.time()
                window_t, tokens_window = time.time(), 0
            if self.skip_time:
                last_tick += self.skip_time
                window_t += self.skip_time
                self.skip_time = 0.0

            # ---- one training step --------------------------------------------------------
            lr = self.lr_now(self.session_elapsed, session_step)
            for g in self.opt.param_groups:
                g["lr"] = lr
            try:
                loss = self.train_step()
            except Exception as e:  # noqa: BLE001
                if is_oom(e):
                    del e
                    self.shrink("Your computer ran out of memory for a moment, so")
                    continue
                raise
            if not math.isfinite(loss):
                bad_steps += 1
                self.opt.zero_grad(set_to_none=True)
                if bad_steps >= 5:
                    raise RuntimeError("training became unstable (the numbers blew up)")
                self.peak_lr *= 0.5
                self.emit({"type": "notice", "level": "warn", "code": "unstable",
                           "msg": "Learning got unstable for a moment, so I slowed the learning speed down."})
                continue
            bad_steps = 0
            self.step += 1
            session_step += 1
            n_tok = self.micro * self.accum * self.block
            self.tokens_seen += n_tok
            tokens_window += n_tok
            self.ema_n += 1
            self.ema_acc = 0.9 * self.ema_acc + 0.1 * loss
            self.ema = self.ema_acc / (1.0 - 0.9 ** min(self.ema_n, 500))
            now = time.time()
            self.session_elapsed += max(0.0, now - last_tick)
            last_tick = now

            # Windows quietly spills into slow shared memory instead of failing: stay clear of the edge
            if not checked_mem and session_step >= 3 and self.device_type == "cuda":
                checked_mem = True
                try:
                    total = torch.cuda.get_device_properties(self.device).total_memory
                    if torch.cuda.max_memory_reserved(self.device) > 0.93 * total and self.micro > 1:
                        self.shrink("Your graphics card's memory was nearly full, so")
                        checked_mem = False
                except Exception:  # noqa: BLE001
                    pass

            # ---- reporting ----------------------------------------------------------------
            if now - last_metrics >= 1.0 or session_step == 1:
                dt = max(1e-6, now - window_t)
                if session_step > 1:
                    tps = tokens_window / dt
                    if dt > 5:
                        window_t, tokens_window = now, 0
                last_metrics = now
                ev = {"type": "metrics", "step": self.step, "t": round(self.total_elapsed(), 1),
                      "loss": round(float(self.ema), 4), "lr": lr, "tps": round(tps),
                      "tokens_seen": self.tokens_seen,
                      "passes": round(self.tokens_seen / max(1, len(self.train_data)), 3),
                      "session_elapsed": round(self.session_elapsed, 1), "budget_sec": self.budget}
                if self.device_type == "cuda":
                    try:
                        ev["mem_gb"] = round(torch.cuda.max_memory_reserved(self.device) / 2**30, 2)
                    except Exception:  # noqa: BLE001
                        pass
                self.emit(ev)
                self.log_line("metrics.jsonl", {"k": "m", "step": ev["step"], "t": ev["t"],
                                                "loss": ev["loss"], "tps": ev["tps"]})

            due = now - last_eval >= eval_every or (not first_eval_done and self.session_elapsed > 4)
            out_of_time = self.budget > 0 and self.session_elapsed >= self.budget and \
                (session_step >= self.min_steps or self.session_elapsed >= self.budget * 30)
            if due or out_of_time:
                first_eval_done = True
                t_eval = time.time()
                tr_loss = float(self.ema)
                val = None
                if self.val_data is not None:
                    if eval_batches == 0:  # time one batch, then fix the count for comparable numbers
                        t1 = time.time()
                        self.estimate("val", 1)
                        per = max(1e-3, time.time() - t1)
                        eval_batches = int(max(2, min(12, 3.0 / per)))
                    val = self.estimate("val", eval_batches)
                ref = val if val is not None else tr_loss
                bpb = ref / LN2 / self.bpt
                stage = stage_for(ref, self.base, self.bpt)
                if val is not None:
                    if val < self.best_val:
                        self.best_val = val
                        worse_evals = 0
                    elif val > self.best_val * 1.02:
                        worse_evals += 1
                    if val < self.saved_best and (time.time() - last_best_save >= best_save_gap or out_of_time):
                        self.save_best(val)
                        self.saved_best = val
                        last_best_save = time.time()
                ev = {"type": "eval", "step": self.step, "t": round(self.total_elapsed(), 1),
                      "train": round(tr_loss, 4), "val": None if val is None else round(val, 4),
                      "best_val": None if val is None else round(self.best_val, 4),
                      "bpb": round(bpb, 3), "stage": stage}
                self.emit(ev)
                self.log_line("metrics.jsonl", {"k": "e", "step": self.step, "t": ev["t"],
                                                "train": ev["train"], "val": ev["val"], "stage": stage})
                self.update_config(stats={"step": self.step, "elapsed": round(self.total_elapsed(), 1),
                                          "tokens_seen": self.tokens_seen, "last_train": ev["train"],
                                          "last_val": ev["val"], "bpb": ev["bpb"], "stage": stage,
                                          "best_val": ev["best_val"],
                                          "saved_best": None if not math.isfinite(self.saved_best)
                                          else self.saved_best})
                try:
                    self.make_sample()
                except Exception:  # noqa: BLE001 - a failed sample must never stop training
                    pass
                if worse_evals >= 3 and not told_overfit:
                    told_overfit = True
                    self.emit({"type": "notice", "level": "warn", "code": "overfit",
                               "msg": "It has started memorising your text instead of learning from it "
                                      "(it is getting worse on text it has not seen). The best version "
                                      "is saved. More text, or a smaller brain, would let it go further."})
                if worse_evals >= 6 and self.autostop:
                    reason = "autostop"
                # never let checking take more than about a tenth of the time
                eval_every = max(base_eval_every, 9.0 * (time.time() - t_eval))
                last_eval = time.time()
            if time.time() - last_ckpt >= ckpt_every:
                self.save_last()
                last_ckpt = time.time()
                self.emit({"type": "saved", "step": self.step})
            if out_of_time and reason is None:
                reason = "budget"

        # ---- wrap up ----------------------------------------------------------------------
        if self.step > 0:
            if self.val_data is not None:
                try:
                    final_val = self.estimate("val", eval_batches or 4)
                    self.best_val = min(self.best_val, final_val)
                    if final_val < self.saved_best:
                        self.save_best(final_val)
                        self.saved_best = final_val
                except Exception:  # noqa: BLE001
                    pass
            self.save_last()
            if not (self.run_dir / "ckpt_best.pt").exists():
                self.save_best(float(self.ema or 0.0))
        status = "stopped" if reason == "stopped" else "finished"
        ref = self.best_val if math.isfinite(self.best_val) else float(self.ema or 0.0)
        bpb = ref / LN2 / self.bpt if ref else None
        self.update_config(status=status, stats={
            "step": self.step, "elapsed": round(self.total_elapsed(), 1), "tokens_seen": self.tokens_seen,
            "best_val": None if not math.isfinite(self.best_val) else round(self.best_val, 4),
            "saved_best": None if not math.isfinite(self.saved_best) else self.saved_best,
            "last_train": None if self.ema is None else round(float(self.ema), 4),
            "bpb": None if bpb is None else round(bpb, 3),
            "stage": None if bpb is None else stage_for(ref, self.base, self.bpt)})
        self.emit({"type": "train_end", "run": self.run_dir.name, "reason": reason, "step": self.step,
                   "best_val": None if not math.isfinite(self.best_val) else round(self.best_val, 4),
                   "stage": None if bpb is None else stage_for(ref, self.base, self.bpt),
                   "elapsed": round(self.total_elapsed(), 1)})
        self.emit({"type": "job_end", "kind": "train", "reason": reason, "run": self.run_dir.name})
