# Verification — 10 November 2026

## Published initial trainer and Pages
- GitHub launcher run 42880997835: all seven jobs passed, covering Windows,
  Ubuntu, macOS and the best-effort Intel macOS job, from folders and standalone launchers.
- Independent Linux CPU self-test passed in 65 seconds using PyTorch 2.10.0+cpu.
  It exercised real training, held-out loss, checkpoint/resume, generation,
  export and actual HTTP worker requests. This is functionality evidence,
  not a useful-language-quality benchmark.
- GitHub Pages guide: HTTP 200, successful build at cac61d4. Pages serves a
  static guide/download site; training runs locally after installation.

## Visual builder
- Five Python unit tests: strict rejection, invalid-before-folder-creation,
  actual module order and parameter count, forward/backward, checkpoint
  equality, cached generation, persistence/resume config and legacy presets.
- Actual HTTP worker test trained a custom attention → feed-forward →
  Transformer model, generated text, preserved its block order and quit cleanly.
- CPU only for the custom architecture. CUDA/MPS are not independently verified.
- `node --check` passed for app.js and builder.js; `git diff --check` passed.
- Mock-DOM interaction tests passed for Add, Move up/down, Duplicate, Remove,
  Clear, preset copy, palette drop, drag reorder, the 16-block cap, invalid
  dimensions/attention requirements, and exact architecture serialization.
  The normal buttons provide the same operations for keyboard and touch.
- These interaction tests are not browser rendering or assistive-technology
  certification. Browser visual/layout and real touch tests remain unverified.
- Limits deliberately exclude arbitrary graph wiring and per-block widths.
  No claim that a larger or rearranged model is automatically more capable.
