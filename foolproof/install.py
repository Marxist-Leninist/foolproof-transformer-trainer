"""First-start setup: install the PyTorch build that suits this computer, then the other parts.

Run by the launchers with the app's private Python, before PyTorch exists:

    python install.py <path-to-uv> <app-folder>

Choosing the build is the fiddly part of installing PyTorch, so it is done here once, carefully:

* no NVIDIA card (or a Mac)   -> the ordinary build
* a current NVIDIA card       -> the newest CUDA build for its driver
* an older NVIDIA card        -> GTX 900/1000-series and other pre-2018 cards were dropped from
                                 PyTorch 2.15; they get the last build that still supports them
* whatever was installed is then tested by doing real maths on the graphics card. If that fails,
  the next option is tried, and in the end the app simply uses the main processor.

Standard library only.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

OLD_CARD = re.compile(
    r"GTX\s*(9\d\d|10\d\d)|GTX\s*TITAN|TITAN\s*(X|Xp|V)\b|GT\s*10[13]0|MX\s?[1-3][1-5]0|"
    r"Quadro\s*(K|M|P|GP)\d|Tesla\s*(K|M|P|V)\d", re.I)

GPU_TEST = r"""
import json
r = {"cuda": False, "works": False, "error": "", "name": "", "cap": None}
try:
    import torch
    r["torch"] = torch.__version__
    r["cuda"] = bool(torch.cuda.is_available())
    if r["cuda"]:
        r["name"] = torch.cuda.get_device_name(0)
        r["cap"] = list(torch.cuda.get_device_capability(0))
        x = torch.randn(1, 2, 64, 32, device="cuda", requires_grad=True)
        y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True).sum()
        y.backward()
        float(y.item())
        torch.cuda.synchronize()
        r["works"] = True
except Exception as e:
    r["error"] = (type(e).__name__ + ": " + str(e))[:400]
print("FPGPU " + json.dumps(r))
"""


def say(msg: str) -> None:
    print("  " + msg, flush=True)


def no_window() -> dict:
    return {"creationflags": 0x08000000} if sys.platform == "win32" else {}


def nvidia_cards() -> list[dict]:
    """[{name, cap (float or None), driver}] from nvidia-smi; [] if there is no NVIDIA card."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    for fields in ("name,compute_cap,driver_version", "name,driver_version"):
        try:
            out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30, **no_window())
        except Exception:  # noqa: BLE001
            return []
        if out.returncode != 0 or not out.stdout.strip():
            continue
        cards = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            cap = None
            if fields.startswith("name,compute_cap") and len(parts) >= 2:
                try:
                    cap = float(parts[1])
                except ValueError:
                    cap = None
            cards.append({"name": parts[0], "cap": cap, "driver": parts[-1]})
        return cards
    return []


def make_plan(cards: list[dict]) -> list[dict]:
    """Ordered things to try. Each: {backend, spec, why}. backend None = plain PyPI build."""
    forced = os.environ.get("FOOLPROOF_TORCH_BACKEND", "").strip()
    if forced:
        return [{"backend": forced, "spec": "torch", "why": f"Installing the “{forced}” build, as you asked."}]
    if sys.platform == "darwin":
        return [{"backend": None, "spec": "torch", "why": "Installing the Mac build."}]
    if not cards:
        return [{"backend": "cpu", "spec": "torch",
                 "why": "No NVIDIA graphics card found, so installing the standard build (works on any computer)."}]
    best = max(cards, key=lambda c: c["cap"] or 0.0)
    name = best["name"]
    old = (best["cap"] is not None and best["cap"] < 7.5) or (best["cap"] is None and OLD_CARD.search(name))
    legacy = {"backend": "cu126", "spec": "torch<2.15",
              "why": f"Found: {name}. It is an older card, so installing the last PyTorch build that still supports it."}
    modern = {"backend": "auto", "spec": "torch",
              "why": f"Found: {name}. Installing the build that uses it."}
    return [legacy, modern] if old else [modern, legacy]


def uv_install(uv: str, args: list[str]) -> bool:
    cmd = [uv, "pip", "install", "--python", sys.executable, *args]
    try:
        return subprocess.run(cmd).returncode == 0
    except OSError as e:
        say(f"Could not run the installer helper ({e}).")
        return False


def test_gpu() -> dict:
    try:
        out = subprocess.run([sys.executable, "-c", GPU_TEST], capture_output=True, text=True,
                             timeout=600, **no_window())
        for line in reversed(out.stdout.splitlines()):
            if line.startswith("FPGPU "):
                return json.loads(line[6:])
        return {"cuda": False, "works": False, "error": (out.stderr or "no answer")[-300:]}
    except Exception as e:  # noqa: BLE001
        return {"cuda": False, "works": False, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python install.py <path-to-uv> <app-folder>")
        return 2
    uv, app = sys.argv[1], Path(sys.argv[2])
    extra = ["numpy<2"] if (sys.platform == "darwin" and platform.machine() == "x86_64") else []
    cards = nvidia_cards()
    plan = make_plan(cards)
    installed = False
    gpu_ok = False
    for i, step in enumerate(plan):
        say(step["why"])
        args = [step["spec"], *extra]
        if step["backend"]:
            args.append(f"--torch-backend={step['backend']}")
        if installed:
            args += ["--reinstall-package", "torch"]
        if not uv_install(uv, args):
            say("That build would not install.")
            continue
        installed = True
        if not cards or step["backend"] in ("cpu", None):
            break
        say("Checking that the graphics card really works (this can take a minute)...")
        res = test_gpu()
        if res.get("works"):
            say(f"The graphics card works: {res.get('name')}.")
            gpu_ok = True
            break
        reason = res.get("error") or ("PyTorch cannot see the card (the NVIDIA driver may be too old)"
                                      if not res.get("cuda") else "unknown reason")
        say("The graphics card could not be used with this build.")
        say("  (" + reason.splitlines()[0][:200] + ")")
        if i + 1 < len(plan):
            say("Trying another build...")
    if not installed:
        say("Trying the standard build instead...")
        if not uv_install(uv, ["torch", *extra, "--torch-backend=cpu"]):
            say("PROBLEM: PyTorch could not be installed. Check your internet connection and that you")
            say("have 5 GB of free disk space, then start the app again.")
            return 1
    if cards and not gpu_ok and installed and not os.environ.get("FOOLPROOF_TORCH_BACKEND"):
        say("Training will use the main processor for now, which is slower but works.")
        say("Updating your NVIDIA driver and starting the app again may let it use the graphics card.")
    say("Installing the remaining parts...")
    if not uv_install(uv, ["-r", str(app / "requirements.txt"), *extra]):
        say("PROBLEM: some parts could not be installed. Check your internet connection and start the app again.")
        return 1
    venv = Path(sys.prefix)
    shutil.copyfile(app / "requirements.txt", venv / "fp-installed.txt")
    say("Setup finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
