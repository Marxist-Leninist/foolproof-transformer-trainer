#!/usr/bin/env bash
# Foolproof Transformer Trainer - launcher for macOS and Linux.
#
# Run it by double-clicking start-mac.command (Mac), or in a terminal:   ./start.sh
# Or, without downloading anything first, paste this one line into a terminal:
#   curl -fsSL https://raw.githubusercontent.com/Marxist-Leninist/foolproof-transformer-trainer/main/start.sh | bash
#
# The first run sets up a private copy of Python and PyTorch (a few minutes, needs internet).
# Nothing else on your computer is changed. After that it starts in seconds, offline too.
set -u

APP_NAME="Foolproof Trainer"
REPO="${FOOLPROOF_REPO:-Marxist-Leninist/foolproof-transformer-trainer}"
BRANCH="${FOOLPROOF_BRANCH:-main}"

say()  { printf '  %s\n' "$*"; }
fail() {
  printf '\n  PROBLEM: %s\n' "$*" >&2
  printf '  If you are stuck, ask for help here: https://github.com/%s/issues\n\n' "$REPO" >&2
  if [ -t 0 ] && [ -z "${CI:-}" ]; then read -r -p "  Press Enter to close." _ || true; fi
  exit 1
}
fetch() {  # fetch URL DEST
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then wget -q --tries=3 "$1" -O "$2"
  else return 127; fi
}

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin) DATA="$HOME/Library/Application Support/FoolproofTrainer" ;;
  Linux)  DATA="${XDG_DATA_HOME:-$HOME/.local/share}/foolproof-trainer" ;;
  *) fail "This launcher is for macOS and Linux. On Windows, double-click START-WINDOWS.bat instead." ;;
esac
mkdir -p "$DATA/bin" || fail "Could not create the folder $DATA"

printf '\n  %s\n  %s\n\n' "$APP_NAME" "-----------------"

# ---- 1. Find the app itself. If this script was run on its own, download the rest. ----------
SELF_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
DOWNLOADED=0
if [ -n "$SELF_DIR" ] && [ -f "$SELF_DIR/foolproof/__main__.py" ]; then
  if [ -e "$SELF_DIR/.git" ] || [ "$SELF_DIR" = "$DATA/app" ]; then
    APP_DIR="$SELF_DIR"   # a developer's git clone (or the installed copy) runs where it is
  else
    # A downloaded copy installs itself into a fixed place, so the shortcut keeps working
    # after the download is tidied away.
    say "Installing $APP_NAME..."
    rm -rf "$DATA/app.new" && mkdir -p "$DATA/app.new" && cp -R "$SELF_DIR/." "$DATA/app.new/" \
      && rm -rf "$DATA/app" && mv "$DATA/app.new" "$DATA/app" \
      || fail "Could not copy the app into place. Is your disk full?"
    chmod +x "$DATA/app/start.sh" "$DATA/app/start-mac.command" 2>/dev/null || true
    APP_DIR="$DATA/app"
    DOWNLOADED=1
  fi
else
  APP_DIR="$DATA/app"
  if [ ! -f "$APP_DIR/foolproof/__main__.py" ] || [ "${FOOLPROOF_UPDATE:-0}" = "1" ]; then
    say "Downloading $APP_NAME..."
    TMP="$(mktemp -d)" || fail "Could not create a temporary folder."
    fetch "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" "$TMP/app.tar.gz" \
      || fail "Could not download the app. Check your internet connection and try again."
    rm -rf "$APP_DIR.new" && mkdir -p "$APP_DIR.new"
    tar -xzf "$TMP/app.tar.gz" -C "$APP_DIR.new" --strip-components=1 \
      || fail "The download arrived damaged. Please try again."
    rm -rf "$APP_DIR" && mv "$APP_DIR.new" "$APP_DIR"
    rm -rf "$TMP"
    chmod +x "$APP_DIR/start.sh" "$APP_DIR/start-mac.command" 2>/dev/null || true
  fi
  DOWNLOADED=1
fi

# ---- 2. The installer helper (uv): one small program that fetches Python and packages. -------
UV="$DATA/bin/uv"
if ! "$UV" --version >/dev/null 2>&1; then
  say "Step 1 of 3: fetching the installer helper..."
  TMP="$(mktemp -d)" || fail "Could not create a temporary folder."
  fetch "https://astral.sh/uv/install.sh" "$TMP/uv-install.sh" \
    || fail "Could not reach the internet. The first start needs it; later starts do not."
  env UV_UNMANAGED_INSTALL="$DATA/bin" INSTALLER_NO_MODIFY_PATH=1 sh "$TMP/uv-install.sh" >"$TMP/uv.log" 2>&1 \
    || { tail -n 5 "$TMP/uv.log" >&2; fail "The installer helper could not be set up."; }
  rm -rf "$TMP"
  "$UV" --version >/dev/null 2>&1 || fail "The installer helper did not start on this computer."
fi

# ---- 3. A private Python with PyTorch, kept out of the way in: $DATA -------------------------
export UV_PYTHON_INSTALL_DIR="$DATA/python"
export UV_PYTHON_PREFERENCE="only-managed"
export PYTHONUTF8=1
VENV="$DATA/venv"
PY="$VENV/bin/python"
if [ ! -x "$PY" ] || ! cmp -s "$APP_DIR/requirements.txt" "$VENV/fp-installed.txt"; then
  if [ ! -x "$PY" ]; then
    say "Step 2 of 3: setting up a private copy of Python..."
    "$UV" venv "$VENV" --python 3.12 --quiet || fail "Could not set up Python."
  fi
  say "Step 3 of 3: installing PyTorch, the AI engine."
  say "This is a big download (up to 3 GB with an NVIDIA graphics card). It happens only once."
  "$PY" "$APP_DIR/foolproof/install.py" "$UV" "$APP_DIR" \
    || fail "Setup did not finish. Check your internet connection and free disk space (5 GB), then start it again: it carries on from where it stopped."
fi

# ---- 4. A shortcut for next time (only when we downloaded the app ourselves). ----------------
if [ "$DOWNLOADED" = "1" ] && [ -z "${CI:-}" ]; then
  if [ "$OS" = "Darwin" ] && [ -d "$HOME/Desktop" ] && [ ! -e "$HOME/Desktop/$APP_NAME.command" ]; then
    printf '#!/bin/bash\nexec "%s/start.sh"\n' "$APP_DIR" > "$HOME/Desktop/$APP_NAME.command" \
      && chmod +x "$HOME/Desktop/$APP_NAME.command" \
      && say "Added “$APP_NAME” to your Desktop: double-click it next time."
  elif [ "$OS" = "Linux" ]; then
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    if [ ! -e "$APPS/foolproof-trainer.desktop" ] && mkdir -p "$APPS" 2>/dev/null; then
      printf '[Desktop Entry]\nType=Application\nName=%s\nComment=Train your own small AI\nExec="%s/start.sh"\nIcon=%s/foolproof/web/icon.svg\nTerminal=true\nCategories=Education;Science;\n' \
        "$APP_NAME" "$APP_DIR" "$APP_DIR" > "$APPS/foolproof-trainer.desktop" \
        && say "Added “$APP_NAME” to your applications menu for next time."
    fi
  fi
fi

cd "$APP_DIR" || fail "Could not open the app folder."
exec "$PY" -m foolproof "$@"
