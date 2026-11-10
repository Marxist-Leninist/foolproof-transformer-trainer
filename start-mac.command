#!/bin/bash
# Double-click me on a Mac. (If macOS says it cannot verify this file, see the README:
# the one-line Terminal command there avoids that warning.)
cd "$(dirname "$0")" || exit 1
exec ./start.sh "$@"
