"""Package a trained model into a folder anyone can use without this app.

    python -m foolproof.export <run_dir>      -> prints the export folder as JSON

The folder holds the weights (safetensors), the tokenizer, a model card and a
small ``chat.py`` so the model runs with nothing but PyTorch installed.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from . import REPO_URL, paths

CHAT_PY = '''"""Run this model:   python chat.py "your prompt"     (needs: pip install torch safetensors{extra})"""
import json, sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from fp_model import GPT, ModelConfig
from fp_tokenizers import StreamDecoder, Tokenizer

here = Path(__file__).resolve().parent
cfg = json.loads((here / "config.json").read_text(encoding="utf-8"))
tok = Tokenizer.load(here / "tokenizer.json")
model = GPT(ModelConfig.from_dict(cfg["model"]))
state = load_file(str(here / "model.safetensors"))
state["lm_head.weight"] = state["wte.weight"]  # the two share one table
model.load_state_dict(state)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device).eval()

prompt = " ".join(sys.argv[1:]) or input("You: ")
if cfg.get("chat"):
    ids = [tok.user_id] + tok.encode(prompt) + [tok.assistant_id]
    stop = (tok.eot_id, tok.user_id)
else:
    ids = tok.encode(prompt) or tok.encode("\\n")
    stop = (tok.eot_id,)
    print(prompt, end="")
dec = StreamDecoder(tok)
x = torch.tensor([ids[-(model.cfg.block_size - 1):]], device=device)
for t in model.generate(x, 300, temperature=0.8, top_k=50, stop_ids=stop):
    if t in stop:
        break
    if t < tok.base_vocab:
        print(dec.push(t), end="", flush=True)
print(dec.flush())
'''


def export(run_dir: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    run_dir = Path(run_dir)
    cfg = paths.read_json(run_dir / "config.json") or {}
    ck_path = run_dir / "ckpt_best.pt"
    if not ck_path.exists():
        ck_path = run_dir / "ckpt_last.pt"
    if not ck_path.exists():
        raise FileNotFoundError("This AI has not been trained yet, so there is nothing to export.")
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    out = run_dir / "export"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    state = {k: v.contiguous() for k, v in ck["model"].items() if k != "lm_head.weight"}
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt", "made_with": "foolproof-trainer"})
    stats = cfg.get("stats") or {}
    public = {
        "architecture": "foolproof-gpt (RoPE, RMSNorm, tied embeddings)",
        "model": ck["model_config"], "chat": bool(cfg.get("chat")),
        "params": cfg.get("params"), "trained_on": cfg.get("dataset_label"),
        "data_licence": cfg.get("licence"), "steps": stats.get("step"),
        "tokens_seen": stats.get("tokens_seen"), "best_val_loss": stats.get("best_val"),
        "tokenizer": cfg.get("tokenizer"),
    }
    (out / "config.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    shutil.copyfile(run_dir / "tokenizer.json", out / "tokenizer.json")
    shutil.copyfile(paths.PACKAGE_DIR / "model.py", out / "fp_model.py")
    shutil.copyfile(paths.PACKAGE_DIR / "tokenizers.py", out / "fp_tokenizers.py")
    kind = (cfg.get("tokenizer") or {}).get("kind")
    extra = {"bpe": " tokenizers", "tiktoken": " tiktoken"}.get(kind, "")
    (out / "chat.py").write_text(CHAT_PY.replace("{extra}", extra), encoding="utf-8")
    (out / "README.md").write_text(model_card(cfg, public, extra), encoding="utf-8")
    return out


def model_card(cfg: dict, public: dict, extra: str) -> str:
    m = public["model"]
    params = public.get("params") or 0
    when = time.strftime("%Y-%m-%d", time.localtime(cfg.get("created", time.time())))
    kind = "chat" if public["chat"] else "text-continuation"
    lines = [
        "---", "tags:", "- text-generation", "- foolproof-trainer", "- from-scratch", "---", "",
        f"# {cfg.get('name', 'My AI')}", "",
        f"A small {kind} language model ({params / 1e6:.1f} million parameters) trained from scratch "
        f"with [Foolproof Transformer Trainer]({REPO_URL}).",
        "", "| | |", "|---|---|",
        f"| Trained on | {public.get('trained_on') or 'custom text'} |",
        f"| Data licence | {public.get('data_licence') or 'not stated'} |",
        f"| Size | {m['n_layer']} layers, width {m['n_embd']}, {m['n_head']} heads, context {m['block_size']} tokens |",
        f"| Vocabulary | {m['vocab_size']:,} tokens ({(public.get('tokenizer') or {}).get('kind', '?')}) |",
        f"| Training steps | {public.get('steps') or '?'} |",
        f"| Best validation loss | {public.get('best_val_loss') or '?'} |",
        f"| Made | {when} |", "",
        "## Run it", "", "```", f"pip install torch safetensors{extra}", 'python chat.py "Once upon a time"', "```", "",
        "It is a small model: expect text in the *style* of its training data, not facts you can rely on.", "",
    ]
    return "\n".join(lines)


def make_zip(run_dir: Path) -> Path:
    out = export(run_dir)
    base = run_dir / f"{run_dir.name}-export"
    archive = shutil.make_archive(str(base), "zip", root_dir=str(out))
    return Path(archive)


if __name__ == "__main__":
    try:
        mode = sys.argv[2] if len(sys.argv) > 2 else "folder"
        p = make_zip(Path(sys.argv[1])) if mode == "zip" else export(Path(sys.argv[1]))
        print(json.dumps({"ok": True, "path": str(p)}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{e}"}))
        sys.exit(1)
