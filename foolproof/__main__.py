"""Start the app:  python -m foolproof   (the START launchers do this for you)."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(prog="foolproof", description="Foolproof Transformer Trainer")
    ap.add_argument("--port", type=int, default=8765, help="port for the local window (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    ap.add_argument("--lan", action="store_true",
                    help="also allow other devices on your home network (e.g. your phone) to open the app")
    ap.add_argument("--home", help="folder to keep AIs and data in (default: ~/FoolproofTrainer)")
    ap.add_argument("--selftest", action="store_true", help="check that everything works, then exit")
    ap.add_argument("--online", action="store_true", help="with --selftest: also test Hugging Face access")
    args = ap.parse_args()
    if args.home:
        os.environ["FOOLPROOF_HOME"] = args.home
    if args.selftest:
        from . import selftest

        return selftest.main(["--online"] if args.online else [])
    from . import server

    return server.serve(port=args.port, lan=args.lan, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
