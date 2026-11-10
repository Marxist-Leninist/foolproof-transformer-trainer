"""Is the network on the website the same network the app trains?

Builds the app's PyTorch model for several builder designs, fills it with random weights,
and hands the weights to the in-page engine (docs/nn.js, both its JavaScript and WebAssembly
paths) under PyTorch's own table names. The scores for every letter, the loss and every
gradient must then agree.

Run from the repository root:   python tests/web/parity.py      (needs torch and node)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from foolproof.model import GPT, ModelConfig  # noqa: E402

LETTER = {"t": "transformer", "a": "attention", "f": "feed_forward"}
CASES = [  # name, width, heads, context, blocks, vocab, batch, sequence length
    ("two transformer layers", 64, 4, 32, "tt", 65, 3, 32),
    ("attention, feed-forward, transformer", 64, 2, 32, "aft", 97, 2, 32),
    ("own order, 8 heads", 48, 8, 64, "fatta", 131, 2, 48),
    ("one head, text shorter than memory", 32, 1, 64, "ta", 40, 4, 20),
    ("widest the page offers", 128, 8, 32, "tf", 70, 2, 32),
]


def main() -> int:
    node = shutil.which("node")
    if not node:
        print("node is not installed: cannot compare with the in-page engine")
        return 2
    torch.manual_seed(20261110)
    out = []
    for name, width, heads, context, blocks, vocab, B, T in CASES:
        kinds = [LETTER[ch] for ch in blocks]
        model = GPT(ModelConfig(vocab_size=vocab, n_layer=len(kinds), n_head=heads, n_embd=width,
                                block_size=context, block_types=kinds))
        with torch.no_grad():  # far from the tidy starting values, so every part of the sums matters
            for pname, p in model.named_parameters():
                if p.dim() >= 2:
                    p.normal_(0.0, 0.25 if "wte" in pname else 0.12)
                else:
                    p.uniform_(0.6, 1.6)
        model.train()
        idx = torch.randint(0, vocab, (B, T))
        targets = torch.randint(0, vocab, (B, T))
        logits, loss = model(idx, targets)
        loss.backward()
        state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
        grads = {k: p.grad.flatten().tolist() for k, p in model.named_parameters()}
        out.append({
            "name": name, "vocab": vocab, "B": B, "T": T,
            "design": {"version": 1, "width": width, "heads": heads, "context": context,
                       "blocks": [{"type": k} for k in kinds]},
            "idx": idx.flatten().tolist(), "targets": targets.flatten().tolist(),
            "state": state, "grads": grads, "loss": loss.item(), "logits": logits.detach().flatten().tolist(),
        })
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "cases.json"
        f.write_text(json.dumps(out))
        print(f"PyTorch {torch.__version__} vs the in-page engine, {len(out)} designs")
        return subprocess.run([node, str(ROOT / "tests" / "web" / "parity.js"), str(f)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
