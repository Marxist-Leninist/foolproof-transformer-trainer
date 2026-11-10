# Things bundled with this app

| What | Where | Licence |
|---|---|---|
| Lexend typeface (The Lexend Project Authors) | `foolproof/web/fonts/lexend-*.woff2` | SIL Open Font License 1.1 (`OFL-Lexend.txt`) |
| Andika typeface (SIL International) | `foolproof/web/fonts/andika-*.woff2` | SIL Open Font License 1.1 (`OFL-Andika.txt`) |
| Tiny Shakespeare (collected by Andrej Karpathy from Shakespeare's plays) | `foolproof/assets/tiny_shakespeare.txt` | The plays are in the public domain |
| The same typefaces and the first part of the same Shakespeare text, copied for the website | `docs/app/fonts/`, `docs/data/shakespeare.txt` | As above |
| Polynomial for e to the power x in the website's WebAssembly loops, after the Cephes Math Library (Stephen L. Moshier) | `docs/kernels.c` | Cephes is distributed free of charge by its author; only the published coefficients are used |

Downloaded at first start, not bundled: [uv](https://github.com/astral-sh/uv) (Apache-2.0 / MIT),
Python, [PyTorch](https://pytorch.org) (BSD-3-Clause), NumPy, tiktoken, tokenizers, PyArrow,
huggingface_hub, safetensors, pypdf.

Datasets in the starter library are fetched from Hugging Face when you choose them; each one's
licence is shown next to it in the app.
