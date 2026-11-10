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

## Block builder (app) and website, as of version 1.1.0

What was run, and where. "Locally" means a Linux machine with 2 processor cores and no
graphics card; the same checks run on GitHub's machines on every push
(`.github/workflows/site.yml`, `.github/workflows/launchers.yml`).

### The model the builder makes (PyTorch)
- Five Python unit tests: strict rejection of bad designs, nothing written to disk for a
  rejected design, module order and parameter count, forward and backward pass, checkpoint
  equality, cached generation, persistence and resume settings, and unchanged ready-made sizes.
- The app's self-test now always drives the app's own window (real HTTP requests, real worker
  process) with a hand-built design, attention then feed-forward then a transformer layer:
  the fit check answers, a design that cannot look back is refused in words, it trains,
  generates text, and the saved model keeps the block order. Because the launchers run the
  self-test, this is exercised on Windows, macOS and Linux in CI. 35 checks, all passing locally.
- A fit check for hand-built designs (`presets.fit_custom`, `POST /api/fit`) uses the same
  rules as the ready-made sizes. It is an estimate, as theirs is.
- CPU only. Hand-built designs on CUDA or Apple MPS are not independently verified.

### The network on the website (JavaScript and WebAssembly)
- `node tests/web/test_nn.js`: every hand-written gradient against finite differences in
  double precision for seven designs (worst relative error about 1e-6); the design rules;
  the WebAssembly loops against the plain JavaScript ones over several training steps; and
  a learning check on Shakespeare (one transformer layer, 400 steps: loss 4.18 to about 2.2;
  the same size without attention does measurably worse).
- `python tests/web/parity.py`: five designs are built as the app's PyTorch model with random
  weights, and the same weights are loaded into the page's network under PyTorch's own
  parameter names. Scores for every position, the loss and every gradient table agree to
  about one part in a million, on both the JavaScript and the WebAssembly path
  (measured with PyTorch 2.15.0 CPU).
- `kernels.wasm` imports nothing except its memory. It is built from `docs/kernels.c` by
  `tools/build_kernels.sh` (clang 18). CI does not rebuild it; CI tests its behaviour.

### The builder in a real browser
- `python tests/web/browser_check.py`, 46 checks, all passing locally in headless Chrome with
  the page's Content-Security-Policy enforced:
  website with a mouse (drag from the tray to a position, re-order, drag off, Escape to
  cancel), without dragging (tap to add, arrow buttons, Alt+arrow keys, live announcements),
  training in the page (leaves random noise within 14 seconds, held-out loss falls, chart,
  earlier pages, carrying on from typed words, pause, a changed design, your own text);
  website on a phone-sized touch screen (tap, drag by the handle, swipe scrolls instead of
  dragging, drag from the tray between two blocks, no sideways overflow, training);
  and the app (starts from the ticked size, fit check, 16-block limit, too-big and
  no-attention designs refused, design code from the website, real PyTorch training, saved
  model equals the stack).
- The finger gestures are sent through Chromium's touch input on an emulated phone screen.
  That is real pointer and touch event handling, but it is not a physical phone or tablet.
- CI also runs the website checks in Firefox and in WebKit (the engine Safari uses).
  WebKit on Linux is close to Safari but is not Safari itself.
- Screen readers: the builder announces every change through a live region and every control
  is a labelled button or drop-down. No screen reader was actually run.

### Limits, stated plainly
- The page trains a small network one letter at a time (at most a few hundred thousand
  parameters). It shows that a design learns; it is not a preview of what the same design
  would do at full size on real data in the app.
- A larger or rearranged design is not automatically a better one.
- The builder makes a stack of blocks along one stream. It does not make arbitrary wiring,
  and every block shares the same width, head count and memory.
