#!/usr/bin/env bash
# Rebuilds docs/kernels.wasm from docs/kernels.c. Needs clang with the wasm32 target and wasm-ld
# (Ubuntu: apt install clang lld). The built file is committed, so nobody else has to run this.
set -euo pipefail
cd "$(dirname "$0")/.."
clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding -fno-builtin \
  -Wl,--no-entry -Wl,--import-memory -Wl,--export=__heap_base -Wl,-z,stack-size=65536 \
  -Wl,--strip-all -o docs/kernels.wasm docs/kernels.c
ls -l docs/kernels.wasm
