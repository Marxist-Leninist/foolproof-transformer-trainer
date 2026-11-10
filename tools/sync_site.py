#!/usr/bin/env python3
"""Keep the website's copies of the app's shared files in step.

The website (docs/, served by GitHub Pages) uses the very same builder, design rules, chart,
styles and fonts as the app (foolproof/web/). Pages can only serve files that are inside
docs/, so they are copied to docs/app/. This script does the copying; with --check it only
reports whether the copies are up to date (the tests run it that way).

    python tools/sync_site.py            copy
    python tools/sync_site.py --check    exit 1 if anything is out of date
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "foolproof" / "web"
SITE = ROOT / "docs"
SHARED = ["style.css", "builder.css", "design.js", "builder.js", "chart.js", "icon.svg"]
TEXT_BYTES = 400_000  # how much Shakespeare the page gets (the app ships all of it)


def wanted() -> dict[Path, bytes]:
    out: dict[Path, bytes] = {}
    for name in SHARED:
        out[SITE / "app" / name] = (WEB / name).read_bytes()
    for f in sorted((WEB / "fonts").iterdir()):
        out[SITE / "app" / "fonts" / f.name] = f.read_bytes()
    text = (ROOT / "foolproof" / "assets" / "tiny_shakespeare.txt").read_bytes()
    cut = text.rfind(b"\n\n", 0, TEXT_BYTES)  # stop at the end of a speech
    out[SITE / "data" / "shakespeare.txt"] = text[: cut + 1]
    return out


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    files = wanted()
    for path, data in files.items():
        if path.exists() and path.read_bytes() == data:
            continue
        stale.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    extra = [p for p in (SITE / "app").rglob("*") if p.is_file() and p not in files] if (SITE / "app").exists() else []
    for p in extra:
        stale.append(p)
        if not check:
            p.unlink()
    if check and stale:
        print("The website's copies are out of date. Run:  python tools/sync_site.py")
        for p in stale:
            print("   ", p.relative_to(ROOT))
        return 1
    print(f"website copies {'are up to date' if not stale else 'updated'} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
