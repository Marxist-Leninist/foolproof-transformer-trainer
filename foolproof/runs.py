"""Bookkeeping for trained models ("runs"): create, list, read history, delete."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import paths, presets


def create(name: str, dataset: dict, preset_id: str, budget_sec: float, hw: dict,
           overrides: dict | None = None, autostop: bool = True, architecture: dict | None = None) -> dict:
    """Make the folder + config for a new model and return the job for the worker."""
    custom = None
    if architecture is not None:
        from .architecture import validate
        custom, custom_params = validate(architecture, int(dataset["vocab_size"]))
    overrides = dict(overrides or {})
    if custom:
        for key in ("n_layer", "n_head", "n_embd", "block_size"):
            overrides.pop(key, None)
    p = dict(presets.PRESET_BY_ID.get(preset_id) or presets.PRESET_BY_ID["small"])
    for k in ("n_layer", "n_head", "n_embd", "block_size", "batch", "lr"):
        if overrides.get(k):
            p[k] = type(p[k])(overrides[k])
    if custom:
        p.update(custom)
    if p["n_embd"] % p["n_head"] != 0 or (p["n_embd"] // p["n_head"]) % 2 != 0:
        raise ValueError("The width must divide evenly by the number of heads (and give an even number).")
    n_tokens = int(dataset.get("n_train") or 0)
    dropout = overrides.get("dropout")
    if dropout is None or dropout == "":
        dropout = presets.dropout_for(n_tokens)
    slug = paths.slugify(name or f"{dataset['slug']}-{preset_id}", "my-ai")
    run_dir = paths.unique_dir(paths.models_dir(), slug)
    run_dir.mkdir(parents=True)
    model = {"n_layer": p["n_layer"], "n_head": p["n_head"], "n_embd": p["n_embd"],
             "block_size": p["block_size"], "dropout": float(dropout),
             "vocab_size": int(dataset["vocab_size"])}
    if custom:
        model["block_types"] = custom["block_types"]
    train = {"lr": float(p["lr"]), "batch": int(p["batch"]), "budget_sec": float(budget_sec or 0),
             "weight_decay": 0.1, "autostop": bool(autostop)}
    if overrides.get("min_steps"):
        train["min_steps"] = int(overrides["min_steps"])
    cfg = {
        "name": name or run_dir.name, "slug": run_dir.name, "created": time.time(),
        "dataset": dataset["slug"], "dataset_name": dataset.get("name"),
        "dataset_label": dataset.get("source_label"), "licence": dataset.get("licence"),
        "chat": bool(dataset.get("chat")), "preset": preset_id, "model": model, "train": train,
        "params": custom_params if custom else presets.count_params(p["n_layer"], p["n_embd"], model["vocab_size"]),
        "status": "new", "stats": {}, "sessions": 0,
        "tokenizer": dataset.get("tokenizer"), "bytes_per_token": dataset.get("bytes_per_token"),
        "sample_prompt": dataset.get("sample_prompt"),
    }
    paths.write_json_atomic(run_dir / "config.json", cfg)
    shutil.copyfile(paths.datasets_dir() / dataset["slug"] / "tokenizer.json", run_dir / "tokenizer.json")
    return job_for(cfg, hw, resume=False)


def job_for(cfg: dict, hw: dict, resume: bool, budget_sec: float | None = None) -> dict:
    train = dict(cfg["train"])
    if budget_sec is not None:
        train["budget_sec"] = float(budget_sec)
    return {
        "kind": "train", "id": f"train-{cfg['slug']}-{int(time.time())}",
        "run": cfg["slug"], "run_dir": str(paths.models_dir() / cfg["slug"]),
        "dataset_dir": str(paths.datasets_dir() / cfg["dataset"]),
        "model": cfg["model"], "train": train, "hw": hw, "resume": resume,
        "home": str(paths.home()),
    }


def get(slug: str) -> dict | None:
    d = paths.models_dir() / slug
    cfg = paths.read_json(d / "config.json")
    if not cfg:
        return None
    cfg["slug"] = d.name
    cfg["has_model"] = (d / "ckpt_best.pt").exists() or (d / "ckpt_last.pt").exists()
    cfg["can_resume"] = (d / "ckpt_last.pt").exists() and \
        (paths.datasets_dir() / str(cfg.get("dataset")) / "train.bin").exists()
    return cfg


def list_all() -> list[dict]:
    out = []
    for d in paths.models_dir().iterdir():
        if d.is_dir():
            cfg = get(d.name)
            if cfg:
                out.append(cfg)
    out.sort(key=lambda c: -c.get("created", 0))
    return out


def mark_interrupted(active_slug: str | None = None) -> None:
    """After a crash/restart nothing is training any more: fix stale 'training' labels."""
    for d in paths.models_dir().iterdir():
        if not d.is_dir() or d.name == active_slug:
            continue
        cfg = paths.read_json(d / "config.json")
        if cfg and cfg.get("status") in ("training", "paused", "new"):
            has = (d / "ckpt_last.pt").exists()
            if cfg["status"] == "new" and not has:
                shutil.rmtree(d, ignore_errors=True)
                continue
            cfg["status"] = "stopped"
            paths.write_json_atomic(d / "config.json", cfg)


def history(slug: str, max_points: int = 800) -> dict:
    """Loss curve, check-ups and writing samples saved so far (thinned for the chart)."""
    d = paths.models_dir() / slug
    metrics, evals, samples = [], [], []
    try:
        with open(d / "metrics.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                (metrics if o.get("k") == "m" else evals).append(o)
    except OSError:
        pass
    if len(metrics) > max_points:
        stride = len(metrics) / max_points
        metrics = [metrics[int(i * stride)] for i in range(max_points)]
    try:
        with open(d / "samples.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    if len(samples) > 300:
        stride = len(samples) / 300
        samples = [samples[int(i * stride)] for i in range(300)]
    return {"metrics": metrics, "evals": evals[-max_points:], "samples": samples}


def delete(slug: str) -> None:
    d = (paths.models_dir() / slug).resolve()
    if d.parent == paths.models_dir().resolve() and d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def delete_dataset(slug: str) -> None:
    d = (paths.datasets_dir() / slug).resolve()
    if d.parent == paths.datasets_dir().resolve() and d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def folder(slug: str) -> Path:
    return paths.models_dir() / slug
