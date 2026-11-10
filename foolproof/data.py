"""Turning any source of text into training data.

    source (built-in / Hugging Face / your files / pasted text)
      -> docs spooled to disk
      -> tokenizer chosen (or trained) to suit the amount of text
      -> train.bin + val.bin (arrays of token ids) + tokenizer.json + meta.json

Runs inside the worker process; reports progress through ``emit``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import hub, paths, textio
from .tokenizers import BPETokenizer, ByteTokenizer, TiktokenTokenizer, Tokenizer, auto_choice

CHARS_PER_TOKEN_GUESS = 4.0
MIN_TRAIN_TOKENS = 2000


class Cancelled(Exception):
    pass


class DataError(Exception):
    """Message is shown to the user as-is: keep it plain and actionable."""


def _clean(s: str) -> str:
    if "\r" in s:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in s:
        s = s.replace("\x00", "")
    return s


def _doc_chars(doc) -> int:
    if isinstance(doc, str):
        return len(doc)
    return sum(len(t) for _, t in doc)


# --------------------------------------------------------------------------- sources

def _iter_local(paths_in: list[str], skipped: list, should_stop):
    files: list[Path] = []
    for p in paths_in:
        p = Path(p)
        if p.is_dir():
            for root, dirs, names in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in
                                 {"node_modules", "__pycache__", "venv", ".venv"})
                for n in sorted(names):
                    if not n.startswith("."):
                        files.append(Path(root) / n)
        elif p.exists():
            files.append(p)
    total = sum(f.stat().st_size for f in files if f.exists()) or 1
    done = 0
    for f in files:
        if should_stop():
            raise Cancelled()
        size = f.stat().st_size if f.exists() else 0
        ext = f.suffix.lower()
        if ext not in textio.SUPPORTED_EXT and size > 50_000_000:
            skipped.append((f.name, "not a text file"))
        else:
            for doc in textio.iter_docs_from_path(f, skipped):
                yield doc, (done + size * 0.5) / total
        done += size
        yield None, done / total


def _iter_hf(src: dict, token, skipped: list, should_stop, state: dict):
    ds, config, split = src["id"], src.get("config"), src.get("split")
    schema = src.get("schema")
    if not schema or schema.get("mode") == "files" or not config:
        info = hub.inspect(ds, config, split, token)
        state["revision"], state["licence"] = info.get("sha"), info.get("license")
        if info["mode"] == "files":
            yield from _iter_hf_raw(ds, token, skipped, should_stop, state)
            return
        schema, config, split = info["schema"], info["config"], info["split"]
    files = hub.list_parquet(ds, config, split, token)
    if not files:
        raise DataError("Hugging Face has not prepared this dataset for reading yet. Try another one.")
    state["schema"] = schema
    frac = [0.0]

    def progress(done, total):
        frac[0] = done / max(1, total)

    cols = hub.schema_columns(schema)
    for row in hub.iter_parquet_rows(files, cols, token, progress, should_stop):
        doc = textio.row_to_doc(row, schema)
        if doc:
            yield doc, frac[0]


def _iter_hf_raw(ds: str, token, skipped: list, should_stop, state: dict):
    files = hub.list_raw_files(ds, token)
    if not files:
        raise DataError("I could not find readable text files in this dataset.")
    total = sum(f["size"] for f in files) or 1
    done = 0
    cache = paths.cache_dir() / "hf_raw" / paths.slugify(ds)
    budget = state.get("max_chars") or 0
    for f in files:
        if should_stop():
            raise Cancelled()
        dest = cache / paths.slugify(f["path"].replace("/", "__"))
        dest = dest.with_suffix(Path(f["path"]).suffix.lower())
        try:
            hub.download(f["url"], dest, token, None, should_stop,
                         max_bytes=(budget * 4 if budget and dest.suffix in textio.TEXT_EXT else None))
        except InterruptedError as e:
            raise Cancelled() from e
        for doc in textio.iter_docs_from_path(dest, skipped):
            yield doc, (done + f["size"] * 0.5) / total
        done += f["size"]
        try:
            dest.unlink()
        except OSError:
            pass


def _source_label(src: dict) -> str:
    t = src.get("type")
    if t == "builtin":
        return "Built in: Shakespeare"
    if t == "hf":
        extra = "/".join(x for x in (src.get("config"), src.get("split")) if x and x != "default")
        return f"Hugging Face: {src['id']}" + (f" ({extra})" if extra else "")
    if t == "text":
        return "Text you pasted"
    return "Your own files"


# --------------------------------------------------------------------------- main job

def prepare(spec: dict, emit, should_stop) -> dict:
    src = spec["source"]
    token = spec.get("hf_token") or None
    max_tokens = int(spec.get("max_tokens") or 0)
    max_chars = int(max_tokens * CHARS_PER_TOKEN_GUESS) if max_tokens else 0
    slug = paths.slugify(spec.get("name") or "my-data", "my-data")
    out_dir = paths.unique_dir(paths.datasets_dir(), slug)
    out_dir.mkdir(parents=True)
    (out_dir / ".partial").write_text("being prepared", encoding="utf-8")
    try:
        meta = _prepare_into(out_dir, spec, src, token, max_tokens, max_chars, emit, should_stop)
        (out_dir / ".partial").unlink()
        return meta
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _prepare_into(out_dir, spec, src, token, max_tokens, max_chars, emit, should_stop) -> dict:
    t0 = time.time()
    last_emit = [0.0]

    def report(stage, frac, msg, force=False, **kw):
        now = time.time()
        if force or now - last_emit[0] > 0.25:
            last_emit[0] = now
            emit({"type": "prep", "stage": stage, "frac": float(max(0.0, min(1.0, frac))),
                  "msg": msg, **kw})

    # ---- 1. read everything into a spool file --------------------------------------------
    skipped: list = []
    state: dict = {"max_chars": max_chars}
    stype = src.get("type")
    if stype == "builtin":
        it = _iter_local([str(paths.ASSETS_DIR / "tiny_shakespeare.txt")], skipped, should_stop)
        state["licence"] = "Public domain"
    elif stype == "hf":
        it = _iter_hf(src, token, skipped, should_stop, state)
    elif stype == "files":
        it = _iter_local(list(src.get("paths") or []), skipped, should_stop)
    elif stype == "text":
        it = iter([(src.get("text") or "", 1.0)])
    else:
        raise DataError("Unknown kind of data source.")

    need = (max_chars or 200_000_000) * 3
    if paths.free_disk_bytes(out_dir) < min(need, 2_000_000_000) and stype == "hf":
        raise DataError("Your disk is almost full. Free up a few gigabytes and try again.")

    spool = out_dir / "raw.jsonl"
    n_docs = n_chars = n_chat = n_dupes = chat_chars = 0
    seen: set[bytes] = set()
    previews: list[str] = []
    sample_prompt = None
    non_latin = sampled = 0
    truncated = False
    report("reading", 0.0, "Starting to read…", force=True)
    try:
        with open(spool, "w", encoding="utf-8") as f:
            for doc, frac in it:
                if should_stop():
                    raise Cancelled()
                if doc is None:
                    report("reading", frac, f"Read {n_chars:,} letters so far…")
                    continue
                if isinstance(doc, str):
                    doc = _clean(doc)
                    if not doc.strip():
                        continue
                    if sampled < 300_000:
                        s = doc[:5000]
                        non_latin += sum(1 for ch in s if ord(ch) > 0x24F)
                        sampled += len(s)
                else:
                    doc = [(r, _clean(t)) for r, t in doc]
                    n_chat += 1
                    if sample_prompt is None:
                        sample_prompt = next((t for r, t in doc if r == "user"), None)
                c = _doc_chars(doc)
                if c >= 200 and len(seen) < 3_000_000:  # the same long document twice teaches nothing new
                    key = hashlib.blake2b(json.dumps(doc, ensure_ascii=False).encode("utf-8", "replace"),
                                          digest_size=8).digest()
                    if key in seen:
                        n_dupes += 1
                        continue
                    seen.add(key)
                if max_chars and n_chars + c > max_chars and isinstance(doc, str):
                    doc = doc[: max(0, max_chars - n_chars)]
                    c = len(doc)
                    truncated = True
                if c == 0:
                    break
                f.write(json.dumps(doc, ensure_ascii=False))
                f.write("\n")
                n_docs += 1
                n_chars += c
                if not isinstance(doc, str):
                    chat_chars += c
                if len(previews) < 3:
                    previews.append(textio.doc_preview(doc, 600))
                shown = max(frac, n_chars / max_chars) if max_chars else frac
                report("reading", shown, f"Read {n_chars:,} letters from {n_docs:,} pieces of text…",
                       docs=n_docs, chars=n_chars)
                if max_chars and n_chars >= max_chars:
                    truncated = True
                    break
    except InterruptedError as e:
        raise Cancelled() from e
    except hub.HubError as e:
        raise DataError(str(e)) from e

    if n_chars < 200:
        why = ""
        if skipped:
            why = " Files I could not read: " + ", ".join(f"{n} ({r})" for n, r in skipped[:5]) + "."
        raise DataError("I found almost no text to learn from." + why +
                        " Give it more: at least a few pages of writing.")

    seen.clear()
    chat = chat_chars > n_chars / 2  # mostly conversations (by amount of text, not by count)
    # ---- 2. choose / train the tokenizer ------------------------------------------------
    want = (spec.get("tokenizer") or "auto").lower()
    non_latin_frac = non_latin / max(1, sampled)
    choice = auto_choice(n_chars, non_latin_frac) if want == "auto" else _parse_choice(want, n_chars)
    report("vocab", 0.0, "Working out how to chop the text into pieces…", force=True)
    tok = _build_tokenizer(choice, spool, should_stop)
    tok.save(out_dir / "tokenizer.json")

    # ---- 3. tokenise into train.bin / val.bin -------------------------------------------
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    val_frac = 0.10 if n_chars < 5_000_000 else 0.05 if n_chars < 100_000_000 else \
        0.01 if n_chars < 2_000_000_000 else 0.005
    stride = max(2, round(1 / val_frac))
    seg_chars = int(max(2000, min(200_000, n_chars // 400)))
    counts = {"train": 0, "val": 0, "bytes": 0, "base_tokens": 0}
    seg_index = [0]

    with open(out_dir / "train.bin", "wb") as ftrain, open(out_dir / "val.bin", "wb") as fval:
        batch_texts: list[str] = []
        batch_plan: list[tuple] = []  # ("t", split_is_val, n_texts=1, add_eot) / ("c", is_val, roles)
        batch_chars = [0]

        def flush():
            if not batch_texts:
                return
            encoded = tok.encode_batch(batch_texts)
            pos = 0
            for kind, is_val, extra in batch_plan:
                if kind == "t":
                    ids = encoded[pos]
                    pos += 1
                    counts["base_tokens"] += len(ids)
                    if extra:
                        ids = ids + [tok.eot_id]
                else:
                    ids = []
                    for role in extra:
                        ids.append(tok.role_id(role))
                        ids.extend(encoded[pos])
                        counts["base_tokens"] += len(encoded[pos])
                        pos += 1
                    ids.append(tok.eot_id)
                arr = np.asarray(ids, dtype=dtype)
                arr.tofile(fval if is_val else ftrain)
                counts["val" if is_val else "train"] += len(ids)
            batch_texts.clear()
            batch_plan.clear()
            batch_chars[0] = 0

        done_chars = 0
        with open(spool, "r", encoding="utf-8") as f:
            for line in f:
                if should_stop():
                    raise Cancelled()
                doc = json.loads(line)
                if isinstance(doc, str):
                    segs = list(textio.split_segments(doc, seg_chars))
                    for i, seg in enumerate(segs):
                        is_val = seg_index[0] % stride == stride - 1
                        seg_index[0] += 1
                        batch_texts.append(seg)
                        batch_plan.append(("t", is_val, i == len(segs) - 1))
                        batch_chars[0] += len(seg)
                        counts["bytes"] += len(seg.encode("utf-8"))
                else:
                    is_val = seg_index[0] % stride == stride - 1
                    seg_index[0] += 1
                    roles = []
                    for role, text in doc:
                        roles.append(role)
                        batch_texts.append(text)
                        batch_chars[0] += len(text)
                        counts["bytes"] += len(text.encode("utf-8"))
                    batch_plan.append(("c", is_val, roles))
                done_chars += _doc_chars(doc)
                if batch_chars[0] >= 4_000_000:
                    flush()
                    report("tokenizing", done_chars / max(1, n_chars),
                           f"Chopped {counts['train'] + counts['val']:,} pieces so far…")
            flush()

    if counts["val"] < 200 or counts["train"] < MIN_TRAIN_TOKENS:
        # too little text for an honest hold-out: keep everything for training
        if counts["train"] + counts["val"] < MIN_TRAIN_TOKENS:
            raise DataError("That is too little text to learn from. Add more: at least a few pages.")
        with open(out_dir / "train.bin", "ab") as ftrain, open(out_dir / "val.bin", "rb") as fval:
            shutil.copyfileobj(fval, ftrain)
        open(out_dir / "val.bin", "wb").close()
        counts["train"] += counts["val"]
        counts["val"] = 0
    try:
        spool.unlink()
    except OSError:
        pass

    bpt = counts["bytes"] / max(1, counts["base_tokens"])
    report("tokenizing", 1.0, "Measuring how predictable the text is…", force=True)
    base = baselines(out_dir, tok.vocab_size, dtype, counts["train"], counts["val"])
    meta = {
        "name": spec.get("name") or out_dir.name,
        "slug": out_dir.name,
        "created": time.time(),
        "source": {k: v for k, v in src.items() if k not in ("text", "schema")},
        "source_label": _source_label(src),
        "revision": state.get("revision") or src.get("revision"),
        "licence": state.get("licence") or src.get("licence"),
        "tokenizer": {"kind": tok.kind, "vocab_size": tok.vocab_size, "why": choice.get("why", ""),
                      "name": choice.get("name")},
        "vocab_size": tok.vocab_size,
        "chat": chat,
        "n_docs": n_docs,
        "n_chars": n_chars,
        "n_train": counts["train"],
        "n_val": counts["val"],
        "dtype": np.dtype(dtype).name,
        "bytes_per_token": round(bpt, 3),
        "baseline": base,
        "preview": previews,
        "sample_prompt": (sample_prompt or "")[:300] or None,
        "skipped": skipped[:50],
        "duplicates_removed": n_dupes,
        "truncated": truncated,
        "seconds": round(time.time() - t0, 1),
    }
    paths.write_json_atomic(out_dir / "meta.json", meta)
    report("done", 1.0, "Your data is ready.", force=True)
    return meta


def _parse_choice(want: str, n_chars: int) -> dict:
    if want == "bytes":
        return {"kind": "bytes", "why": "Letter by letter (you chose this)."}
    if want.startswith("tiktoken"):
        name = want.split(":", 1)[1] if ":" in want else "gpt2"
        return {"kind": "tiktoken", "name": name, "why": f"OpenAI's standard “{name}” vocabulary (you chose this)."}
    if want.startswith("bpe"):
        if ":" in want:
            vocab = int(want.split(":", 1)[1])
        else:
            vocab = auto_choice(max(n_chars, 2_000_000)).get("vocab", 8192)
        return {"kind": "bpe", "vocab": vocab,
                "why": f"Its own vocabulary of {vocab:,} word-pieces learned from your text (you chose this)."}
    raise DataError("Unknown tokenizer choice.")


def _spool_texts(spool: Path, limit_chars: int, should_stop):
    seen = 0
    with open(spool, "r", encoding="utf-8") as f:
        for line in f:
            if should_stop():
                raise Cancelled()
            doc = json.loads(line)
            texts = [doc] if isinstance(doc, str) else [t for _, t in doc]
            for t in texts:
                for seg in textio.split_segments(t, 100_000):
                    yield seg
                    seen += len(seg)
            if seen >= limit_chars:
                return


def _build_tokenizer(choice: dict, spool: Path, should_stop) -> Tokenizer:
    kind = choice["kind"]
    if kind == "bytes":
        return ByteTokenizer()
    if kind == "tiktoken":
        try:
            return TiktokenTokenizer(choice.get("name") or "gpt2")
        except Exception as e:
            raise DataError("Could not download the standard vocabulary (needs internet the first "
                            "time). Check your connection and try again.") from e
    if kind == "bpe":
        return BPETokenizer.train(_spool_texts(spool, 300_000_000, should_stop), int(choice["vocab"]))
    raise DataError("Unknown tokenizer kind.")


# --------------------------------------------------------------------------- yardsticks

def baselines(out_dir: Path, vocab_size: int, dtype, n_train: int, n_val: int) -> dict:
    """How well would two very simple "copycats" predict this text?  (in nats per token)

    * unigram: always guesses by how common each piece is overall;
    * bigram:  guesses from the one piece just before (smoothed, scored on held-out text).

    The trainer places the model on a five-stage ladder relative to these, so the ladder
    means the same thing for letter-by-letter Shakespeare as for word-piece Wikipedia.
    """
    import math

    out = {"uniform": math.log(vocab_size)}
    try:
        train = np.memmap(out_dir / "train.bin", dtype=dtype, mode="r")
        a = np.asarray(train[: min(n_train, 6_000_000)], dtype=np.int64)
        if n_val >= 2000:
            val = np.memmap(out_dir / "val.bin", dtype=dtype, mode="r")
            b = np.asarray(val[: min(n_val, 400_000)], dtype=np.int64)
        else:  # nothing held back: score on the last part of the sample instead
            cut = int(len(a) * 0.9)
            a, b = a[:cut], a[cut:]
        if len(a) < 1000 or len(b) < 200:
            return out
        V = int(vocab_size)
        c1 = np.bincount(a, minlength=V).astype(np.float64)
        p1 = (c1 + 0.5) / (c1.sum() + 0.5 * V)
        out["unigram"] = float(-np.log(p1[b]).mean())
        keys, counts = np.unique(a[:-1] * V + a[1:], return_counts=True)
        ctx = np.bincount(a[:-1], minlength=V).astype(np.float64)        # times each piece is followed
        kinds = np.bincount(keys // V, minlength=V).astype(np.float64)   # distinct followers
        prev, nxt = b[:-1], b[1:]
        want = prev * V + nxt
        pos = np.searchsorted(keys, want)
        pos[pos >= len(keys)] = 0
        c2 = np.where(keys[pos] == want, counts[pos], 0).astype(np.float64)
        lam = ctx[prev] / np.maximum(1.0, ctx[prev] + kinds[prev])       # Witten-Bell smoothing
        p = lam * c2 / np.maximum(1.0, ctx[prev]) + (1.0 - lam) * p1[nxt]
        out["bigram"] = float(min(out["unigram"], -np.log(np.maximum(p, 1e-12)).mean()))
    except Exception:  # noqa: BLE001 - yardsticks are a nicety; never fail the job over them
        pass
    return out


def stage_for(loss: float, base: dict | None, bytes_per_token: float = 1.0) -> int:
    """Place a loss on the ladder: 1 random noise, 2 letter soup, 3 word salad, 4 sentences, 5 fluent."""
    import math

    if base and base.get("unigram") and base.get("bigram"):
        h0, h1, h2 = base["uniform"], base["unigram"], base["bigram"]
        if loss > h1 + 0.35 * (h0 - h1):
            return 1
        if loss > h2 + 0.40 * (h1 - h2):
            return 2
        if loss > 0.80 * h2:
            return 3
        if loss > 0.58 * h2:
            return 4
        return 5
    bpb = loss / math.log(2) / max(0.5, bytes_per_token)  # fallback: rough cut-offs in bits per byte
    return 1 if bpb > 4.2 else 2 if bpb > 3.3 else 3 if bpb > 2.5 else 4 if bpb > 1.6 else 5


# --------------------------------------------------------------------------- listing

def list_datasets() -> list[dict]:
    out = []
    for d in sorted(paths.datasets_dir().iterdir()):
        if not d.is_dir() or (d / ".partial").exists():
            continue
        meta = paths.read_json(d / "meta.json")
        if meta:
            meta["slug"] = d.name
            out.append(meta)
    out.sort(key=lambda m: -m.get("created", 0))
    return out


def clean_partials() -> None:
    """Remove half-prepared datasets left behind by a crash."""
    for d in paths.datasets_dir().iterdir():
        if d.is_dir() and (d / ".partial").exists():
            shutil.rmtree(d, ignore_errors=True)
