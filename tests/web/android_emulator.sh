#!/bin/bash
# The device self-check page (docs/check.html) in Chrome on an Android emulator.
# Run inside a booted emulator session (CI: reactivecircus/android-emulator-runner). The emulator reaches
# this machine's own 127.0.0.1 as 10.0.2.2, so the page is served from here and reports back to here.
set -uo pipefail
OUT="${RUNNER_TEMP:-/tmp}"
mkdir -p "$OUT/shots"
python tests/web/device_check.py serve --port 8377 --timeout 480 --expect-wasm --out "$OUT/shots/android-report.json" &
SERVER=$!
sleep 2
URL="http://10.0.2.2:8377/check.html?auto=1&report=1"
in_front() { adb shell dumpsys window 2>/dev/null | grep -m1 mCurrentFocus | tr -d '\r'; }
# "Boot completed" comes before the home screen has finished arriving, and a home screen that arrives late
# lands on top of the browser. Wait for it first.
for _ in $(seq 1 30); do
  case "$(in_front)" in *auncher*) break ;; esac
  sleep 2
done
echo "before opening the page the screen shows: $(in_front)"
if adb shell pm list packages | grep -q "package:com.android.chrome$"; then
  echo "browser on the emulator: Chrome $(adb shell dumpsys package com.android.chrome | grep -m1 versionName | tr -d ' \r')"
  # Chrome's welcome screens would sit in front of the page; these switches tell it to skip them.
  adb shell "echo 'chrome --disable-fre --no-default-browser-check --no-first-run --disable-search-engine-choice-screen' > /data/local/tmp/chrome-command-line"
  adb shell am set-debug-app --persistent com.android.chrome
  open_page() { adb shell "am start -n com.android.chrome/com.google.android.apps.chrome.Main -a android.intent.action.VIEW -d '$URL'"; }
  BROWSER=chrome
else
  echo "browser on the emulator: no Chrome in this system image, using whatever opens web links"
  adb shell pm list packages | grep -i -E "chrom|webview|browser" || true
  open_page() { adb shell "am start -a android.intent.action.VIEW -d '$URL'"; }
  BROWSER=
fi
open_page
# The page needs about half a minute. Look every 20 seconds: if the test server is still waiting and the
# browser is not what is on screen, the page was never shown, so open it again.
for look in 1 2 3 4 5 6; do
  sleep 20
  kill -0 $SERVER 2>/dev/null || break
  NOW="$(in_front)"
  echo "$((look * 20)) seconds in, still waiting for the page's report; the screen shows: $NOW"
  if [ -n "$BROWSER" ]; then
    case "$NOW" in *"$BROWSER"*) ;; *) echo "the browser is not in front, opening the page again"; open_page ;; esac
  fi
done
wait $SERVER
RC=$?
adb exec-out screencap -p > "$OUT/shots/android-emulator.png" || true
exit $RC
