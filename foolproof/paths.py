"""Where things live on disk.

Everything the user makes (prepared data, trained models) goes in one visible
folder in their home directory so it survives re-downloading the app:

    ~/FoolproofTrainer/
        datasets/<name>/   prepared (tokenised) data
        models/<name>/     trained models, checkpoints, charts
        uploads/           files dropped into the app
        cache/             downloaded tokenizer files etc.

Set the FOOLPROOF_HOME environment variable to put it somewhere else.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"
ASSETS_DIR = PACKAGE_DIR / "assets"


def home() -> Path:
    env = os.environ.get("FOOLPROOF_HOME")
    p = Path(env).expanduser() if env else Path.home() / "FoolproofTrainer"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def datasets_dir() -> Path:
    return _sub("datasets")


def models_dir() -> Path:
    return _sub("models")


def uploads_dir() -> Path:
    return _sub("uploads")


def cache_dir() -> Path:
    return _sub("cache")


def slugify(name: str, fallback: str = "untitled") -> str:
    """Turn any user-typed name into a safe folder name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._")
    s = s[:60] or fallback
    if s.upper() in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}:  # Windows reserved names
        s = "_" + s
    return s


def unique_dir(parent: Path, slug: str) -> Path:
    """parent/slug, or parent/slug-2, slug-3 ... if it already exists."""
    p = parent / slug
    n = 2
    while p.exists():
        p = parent / f"{slug}-{n}"
        n += 1
    return p


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def free_disk_bytes(path: Path | None = None) -> int:
    import shutil

    try:
        return shutil.disk_usage(str(path or home())).free
    except OSError:
        return 0
