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
- These checks ran on the main processor. A graphics card (CUDA) was checked afterwards: see "Real devices"
  below. Apple MPS is still not verified.

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
- `python tests/web/browser_check.py`, about 50 checks, all passing locally in headless Chrome with
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
- CI also runs the website checks in Firefox and in WebKit (the engine Safari uses), and
  they pass in both, including training in the page on the WebAssembly loops. A separate
  check blocks `kernels.wasm` and confirms that the page still learns on the plain
  JavaScript loops. WebKit on Linux is close to Safari but is not Safari itself;
  Safari itself was checked afterwards ("Real devices" below).
- First fully green run of both workflows with the shared builder: commit 236143a
  (launchers on Windows, macOS Apple Silicon, macOS Intel and Ubuntu; engine, PyTorch
  parity, Chromium, Firefox and WebKit).
- Screen readers: the builder announces every change through a live region and every control
  is a labelled button or drop-down. No screen reader was actually run.

### Limits, stated plainly
- The page trains a small network one letter at a time (at most a few hundred thousand
  parameters). It shows that a design learns; it is not a preview of what the same design
  would do at full size on real data in the app.
- A larger or rearranged design is not automatically a better one.
- The builder makes a stack of blocks along one stream. It does not make arbitrary wiring,
  and every block shares the same width, head count and memory.

## Real devices, added 10 November 2026

The notes above left three things unverified: a graphics card, Safari itself, and a real phone. This is
what has been done about each, and what is still open.

### A graphics card (CUDA)
- Machine: NVIDIA GeForce GTX 1660 SUPER (6 GB), AMD Ryzen 5 5600, Linux, Python 3.11.14,
  PyTorch 2.9.1+cu128. Commit 4edd2da.
- `python -m foolproof.selftest` passed every check ("ALL GOOD in 32s"). The app picked the card by
  itself. Its own run records name the card as the device for the ready-made size trained directly
  (1,648 steps in 12 seconds, loss 5.56 to 2.21) and for the hand-built design (attention, feed-forward,
  transformer layer) trained in a separate process through the app's window. Carrying on from a saved
  run, writing text and exporting ran there too. The five unit tests pass on that machine.
- This is the full-precision path, the one the app uses on GTX 16-series cards. The mixed-precision
  paths (bf16 on RTX 30-series and newer, fp16 on RTX 20-series and T4 cards) and Apple MPS have still
  not been run by anyone.

### Safari itself, on a Mac
- `python tests/web/safari_check.py` drives the Safari that ships with macOS through Apple's
  safaridriver. It runs on every change to the website (`.github/workflows/devices.yml`).
  First green run: Safari 27.2 on macOS 26.7.2, 25 checks.
- What passes there: tap to add, the arrow buttons, Ctrl with an arrow key, dragging with a mouse
  (from the tray to a position, re-ordering, dropping on the bin), the design code and page address,
  the drop-downs, the network learning in the page on the WebAssembly loops (held-out loss 4.18 to
  2.26 in 14 seconds), the chart, carrying on from typed words, pause and carry on, a changed design,
  and the self-check page described below.
- Two limits of the tool, not of the site. safaridriver sends Option with an arrow key without a key
  name, which no real keyboard does, so the Alt chord is checked with a scripted key event and the
  Ctrl chord with typed keys. And no picture could be taken: on GitHub's Macs Safari's window shows
  the page background and nothing else, even for a control page that is one large white heading on
  red. So Safari on a Mac is verified by what it does and by where things are laid out, not by eye.
- One thing Safari does that the other browsers do not: it files an inline-style report against the
  page's security policy for every drop-down any page creates, before the page has touched it. The
  check proves this with a bare element the site never sees. The drop-downs work, the policy is
  unchanged, and those reports are counted apart from the site's own (of which there are none).

### Safari on an iPhone, in Apple's simulator
- `docs/check.html` is a page that tests the site on whatever opens it: the builder draws, tap to add
  and remove, the drag logic, the network starts, it learns, it writes, it carries on from a prompt,
  pause and carry on. It keeps the site's security policy and sends nothing anywhere.
- The same workflow opens it in Safari on an iPhone simulator. First green run: iPhone Air, iOS 26.5,
  9 of 9 checks, WebAssembly loops, about 60 learning steps a second. The simulator's screen is kept
  as a picture with each run, and it shows the page drawn correctly.
- It also runs in Chromium, Firefox and WebKit on Linux, so the page itself is known to be sound.
- A simulator is real Safari for iPhone, but it is not a phone in the hand, and its "drag" is the
  page's own pointer events rather than a finger.

### A phone or tablet in the hand
- Open <https://marxist-leninist.github.io/foolproof-transformer-trainer/check.html> on the device.
  The automatic checks take about half a minute. One more check waits for a real finger: drag a block
  by its handle and the page says whether that worked.
- Results from real devices belong in the list below, with the text the page offers to copy.
  As of this writing nobody has reported one, so physical phones and tablets, Android included,
  are still unverified.
