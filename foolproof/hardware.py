"""Find out what this computer can do (GPU, memory) without the user knowing any of it.

Run as ``python -m foolproof.hardware`` it prints one JSON object. The app runs
it in a short-lived child process so the window itself never loads PyTorch.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys


def ram_bytes() -> int:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMSTAT()
            st.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip())
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5)
            if out.stdout.strip():
                return out.stdout.strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            name = os.environ.get("PROCESSOR_IDENTIFIER", "")
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                    name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
            except Exception:
                pass
            return name
    except Exception:
        pass
    return platform.processor() or platform.machine()


def detect() -> dict:
    info = {
        "os": {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux"),
        "cpu": cpu_name(),
        "cores": os.cpu_count() or 1,
        "ram_gb": round(ram_bytes() / 2**30, 1),
        "device": "cpu",
        "gpu": None,
        "vram_gb": 0.0,
        "torch": None,
        "bf16": False,
        "amp": None,
        "problem": None,
    }
    forced = os.environ.get("FOOLPROOF_DEVICE", "").strip().lower()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch

        info["torch"] = torch.__version__
        if forced != "cpu" and torch.cuda.is_available():
            best, best_mem = 0, 0
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory
                if mem > best_mem:
                    best, best_mem = i, mem
            name = torch.cuda.get_device_name(best)
            cap = torch.cuda.get_device_capability(best)
            try:  # do some real maths on it: "available" is not the same as "works"
                dev = f"cuda:{best}"
                x = torch.randn(1, 2, 64, 32, device=dev, requires_grad=True)
                y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True).sum()
                y.backward()
                float(y.item())
                works = True
                err = ""
            except Exception as e:  # noqa: BLE001
                works, err = False, f"{type(e).__name__}: {e}"
            if works:
                info.update({"device": "cuda", "gpu_index": best, "gpu": name,
                             "vram_gb": round(best_mem / 2**30, 1), "compute_cap": f"{cap[0]}.{cap[1]}"})
                try:
                    try:
                        info["bf16"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                    except TypeError:
                        info["bf16"] = bool(torch.cuda.is_bf16_supported())
                except Exception:  # noqa: BLE001
                    info["bf16"] = False
                # Mixed precision only where it is known to be both fast and safe:
                # bf16 on RTX 30-series and newer; fp16 on RTX 20-series / T4 class cards.
                # GTX cards compute half precision slowly, and the GTX 16-series produces
                # garbage (NaN) with it, so they run in full precision.
                if info["bf16"] and cap[0] >= 8:
                    info["amp"] = "bf16"
                elif cap >= (7, 0) and "GTX" not in name.upper():
                    info["amp"] = "fp16"
                else:
                    info["amp"] = None
            else:
                old = "no kernel image" in err or "not compatible" in err or cap < (7, 5)
                info["problem"] = (
                    f"Your graphics card ({name}) is too old for the PyTorch that is installed, so "
                    "training will use the slower main processor." if old else
                    f"Your graphics card ({name}) did not work when tested, so training will use the "
                    "slower main processor. Updating your NVIDIA driver usually fixes this.")
                info["gpu_error"] = err[:300]
        elif forced != "cpu" and getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            try:  # some Macs (and virtual machines) report a GPU that cannot actually compute
                x = torch.randn(1, 2, 64, 32, device="mps", requires_grad=True)
                y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True).sum()
                y.backward()
                float(y.item())
                info.update({"device": "mps", "gpu": "Apple Silicon GPU",
                             "vram_gb": round(info["ram_gb"] * 0.6, 1)})
            except Exception as e:  # noqa: BLE001
                info["problem"] = ("This Mac's graphics chip did not work when tested, so training will use "
                                   "the slower main processor.")
                info["gpu_error"] = f"{type(e).__name__}: {e}"[:300]
        elif forced != "cpu":
            # No usable GPU. If an NVIDIA card exists but PyTorch cannot use it, say so.
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=8)
                if out.returncode == 0 and out.stdout.strip():
                    info["problem"] = (f"You have a graphics card ({out.stdout.strip().splitlines()[0]}) "
                                       "but it could not be switched on, so training will use the "
                                       "slower main processor. Updating your NVIDIA driver usually fixes this.")
            except Exception:
                pass
    except Exception as e:  # PyTorch missing or broken
        info["problem"] = f"PyTorch could not start ({type(e).__name__}: {e})"
    info["summary"] = summary(info)
    return info


def summary(info: dict) -> str:
    if info["device"] == "cuda":
        return f"{info['gpu']} graphics card with {info['vram_gb']:.0f} GB of memory"
    if info["device"] == "mps":
        return f"Apple Silicon Mac with {info['ram_gb']:.0f} GB of memory"
    return f"No graphics card found, so it will use the main processor ({info['cores']} cores)"


if __name__ == "__main__":
    print(json.dumps(detect()))
