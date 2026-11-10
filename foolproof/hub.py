"""Talking to Hugging Face (the big public library of datasets) for the user.

Uses only the plain web API plus pyarrow: no ``datasets`` library, no dataset
loading scripts.  Rows are streamed straight out of the hub's parquet files
with HTTP range requests, so "use just the first 10 million tokens of a
300 GB dataset" downloads megabytes, not gigabytes.
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__, textio

HUB = "https://huggingface.co"
DSS = "https://datasets-server.huggingface.co"
UA = f"foolproof-trainer/{__version__}"


RAW_DATA_EXT = {".txt", ".text", ".md", ".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet"}


class HubError(Exception):
    """An error with a message that is safe and useful to show to a beginner."""


def _request(url: str, token: str | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    if token and urllib.parse.urlparse(url).netloc.endswith("huggingface.co"):
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def http_json(url: str, token: str | None = None, timeout: float = 30, tries: int = 3):
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(_request(url, token), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read(2000)
            except Exception:
                pass
            msg = ""
            try:
                msg = json.loads(body.decode("utf-8", "replace")).get("error", "")
            except Exception:
                pass
            if e.code in (401, 403):
                raise HubError("This dataset is locked (it needs a Hugging Face account and permission). "
                               "Pick a different one, or add your Hugging Face token in Settings.") from e
            if e.code == 404:
                raise HubError("Hugging Face has no dataset with that name.") from e
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                time.sleep(1.5 * (attempt + 1))
                last = e
                continue
            raise HubError(f"Hugging Face answered with an error ({e.code}). {msg}".strip()) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt + 1 < tries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise HubError("Could not reach Hugging Face. Check your internet connection and try again.") from e
    raise HubError(f"Could not reach Hugging Face ({last}).")


def _q(s: str) -> str:
    return urllib.parse.quote(s, safe="")


# --------------------------------------------------------------------------- search / inspect

def _tag(tags, prefix):
    return [t[len(prefix):] for t in (tags or []) if isinstance(t, str) and t.startswith(prefix)]


def search(query: str, limit: int = 24, token: str | None = None) -> list[dict]:
    url = (f"{HUB}/api/datasets?search={_q(query)}&sort=downloads&direction=-1"
           f"&limit={int(limit) * 2}&full=false")
    out = []
    for d in http_json(url, token):
        tags = d.get("tags") or []
        mods = _tag(tags, "modality:")
        if mods and "text" not in mods:
            continue  # pictures / audio only: nothing for a text model to read
        desc = " ".join((d.get("description") or "").split())
        desc = re.sub(r"^Dataset Card for .{0,120}?(Dataset Summary|Dataset Description|Summary|Overview)\s+", "", desc)
        desc = re.sub(r"^Dataset Card for \S+\s*", "", desc)
        if desc.lower().startswith("more information needed"):
            desc = ""
        out.append({
            "id": d.get("id"),
            "downloads": d.get("downloads") or 0,
            "likes": d.get("likes") or 0,
            "locked": bool(d.get("gated")) or bool(d.get("disabled")),
            "license": (_tag(tags, "license:") or [None])[0],
            "languages": _tag(tags, "language:")[:4],
            "rows": (_tag(tags, "size_categories:") or [None])[0],
            "description": desc[:220],
        })
        if len(out) >= limit:
            break
    return out


def inspect(ds_id: str, config: str | None = None, split: str | None = None,
            token: str | None = None) -> dict:
    """Everything the wizard needs to show about one dataset."""
    ds_id = ds_id.strip().strip("/")
    if ds_id.startswith("http"):
        m = urllib.parse.urlparse(ds_id).path.strip("/").split("/")
        if m and m[0] == "datasets":
            m = m[1:]
        ds_id = "/".join(m[:2])
    info = http_json(f"{HUB}/api/datasets/{urllib.parse.quote(ds_id)}", token)
    tags = info.get("tags") or []
    card = info.get("cardData") or {}
    lic = card.get("license") or (_tag(tags, "license:") or [None])[0]
    if isinstance(lic, list):
        lic = ", ".join(str(x) for x in lic)
    out = {
        "id": info.get("id", ds_id), "sha": info.get("sha"), "license": lic,
        "downloads": info.get("downloads") or 0,
        "locked": bool(info.get("gated")) and not token,
        "url": f"{HUB}/datasets/{info.get('id', ds_id)}",
    }
    if info.get("disabled"):
        raise HubError("This dataset has been switched off by Hugging Face.")
    if out["locked"]:
        raise HubError("This dataset is locked: its owner asks you to sign in and accept terms on "
                       "the Hugging Face website first. Pick a different one, or accept the terms "
                       "there and add your Hugging Face token in Settings.")

    try:
        sp = http_json(f"{DSS}/splits?dataset={_q(out['id'])}", token)
    except HubError:
        sp = {"splits": []}
    splits = sp.get("splits") or []
    if not splits:
        files = list_raw_files(out["id"], token)
        if not files:
            raise HubError("I could not find readable text in this dataset. It may be pictures or "
                           "audio, or stored in a format I do not understand. Try another one.")
        total = sum(f["size"] for f in files)
        out.update({"mode": "files", "configs": [], "splits": [], "config": None, "split": None,
                    "bytes": total, "rows": None, "raw_files": len(files),
                    "schema": {"mode": "files", "chat": False,
                               "describe": f"{len(files)} text file(s) stored in this dataset."},
                    "preview": []})
        return out

    configs: dict[str, list[str]] = {}
    for s in splits:
        configs.setdefault(s["config"], []).append(s["split"])
    if config not in configs:
        config = "default" if "default" in configs else next(iter(configs))
    if split not in configs[config]:
        split = "train" if "train" in configs[config] else configs[config][0]

    rows = nbytes = None
    try:
        sz = http_json(f"{DSS}/size?dataset={_q(out['id'])}&config={_q(config)}", token)
        for s in (sz.get("size") or {}).get("splits") or []:
            if s.get("split") == split:
                rows = s.get("num_rows") or s.get("estimated_num_rows")
                nbytes = s.get("num_bytes_memory") or s.get("num_bytes_parquet_files")
    except HubError:
        pass

    preview, schema, columns = [], None, []
    try:
        fr = http_json(f"{DSS}/first-rows?dataset={_q(out['id'])}&config={_q(config)}&split={_q(split)}",
                       token)
        columns = [f.get("name") for f in fr.get("features") or []]
        sample = [r.get("row") for r in fr.get("rows") or [] if isinstance(r.get("row"), dict)]
        schema = textio.detect_schema(sample)
        if schema:
            for r in sample:
                d = textio.row_to_doc(r, schema)
                if d:
                    preview.append(textio.doc_preview(d, 600))
                if len(preview) >= 3:
                    break
    except HubError:
        pass
    if schema is None:
        raise HubError("I could not find a column of text in this dataset (it may hold numbers, "
                       "labels or pictures instead). Try another one.")
    out.update({
        "mode": "parquet",
        "configs": [{"name": c, "splits": v} for c, v in configs.items()][:400],
        "config": config, "split": split, "rows": rows, "bytes": nbytes,
        "columns": columns, "schema": schema, "preview": preview,
    })
    return out


def list_parquet(ds_id: str, config: str, split: str, token: str | None = None) -> list[dict]:
    p = http_json(f"{DSS}/parquet?dataset={_q(ds_id)}&config={_q(config)}", token)
    files = [f for f in p.get("parquet_files") or [] if f.get("config") == config and f.get("split") == split]
    files.sort(key=lambda f: f.get("filename", ""))
    return [{"url": f["url"], "size": int(f.get("size") or 0), "name": f.get("filename")} for f in files]


def list_raw_files(ds_id: str, token: str | None = None) -> list[dict]:
    """Text-like files stored directly in a dataset repo (fallback when there is no parquet)."""
    try:
        tree = http_json(f"{HUB}/api/datasets/{urllib.parse.quote(ds_id)}/tree/main?recursive=true", token)
    except HubError:
        return []
    out = []
    for f in tree if isinstance(tree, list) else []:
        path = f.get("path", "")
        ext = Path(path).suffix.lower()
        # data files only: a repo's loading script or web page is not the text to learn from
        if f.get("type") != "file" or ext not in RAW_DATA_EXT or path.startswith("."):
            continue
        if Path(path).name.lower() in {"readme.md", "license.md", "license.txt", "dataset_infos.json",
                                       "dataset_info.json", ".gitattributes"}:
            continue
        out.append({"path": path, "size": int(f.get("size") or 0),
                    "url": f"{HUB}/datasets/{ds_id}/resolve/main/{urllib.parse.quote(path)}"})
    out.sort(key=lambda f: f["path"])
    return out


# --------------------------------------------------------------------------- streaming parquet

class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable 'file' backed by HTTP range requests (with a small block cache)."""

    def __init__(self, url: str, size: int, token: str | None = None, block: int = 8 << 20,
                 on_bytes=None, should_stop=None):
        super().__init__()
        self.url, self.size, self.token, self.block = url, int(size), token, block
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.order: list[int] = []
        self.on_bytes, self.should_stop = on_bytes, should_stop

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        self.pos = max(0, min(self.size, self.pos))
        return self.pos

    def _fetch(self, bi: int) -> bytes:
        if bi in self.cache:
            return self.cache[bi]
        start = bi * self.block
        end = min(self.size, start + self.block) - 1
        last = None
        for attempt in range(5):
            if self.should_stop and self.should_stop():
                raise InterruptedError("stopped")
            try:
                req = _request(self.url, self.token, {"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status == 200 and start > 0:
                        raise HubError("range requests are not supported by this server")
                    data = r.read()
                if len(data) < end - start + 1:
                    raise OSError("short read")
                data = data[: end - start + 1]
                break
            except HubError:
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise HubError("The download kept failing. Check your internet connection and try again."
                           ) from last
        if self.on_bytes:
            self.on_bytes(len(data))
        self.cache[bi] = data
        self.order.append(bi)
        while len(self.order) > 4:
            self.cache.pop(self.order.pop(0), None)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            bi = self.pos // self.block
            data = self._fetch(bi)
            off = self.pos - bi * self.block
            chunk = data[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def iter_parquet_rows(files: list[dict], columns: list[str] | None = None, token: str | None = None,
                      progress=None, should_stop=None):
    """Yield row dicts from remote parquet files, fetching only what is read.

    progress(done_bytes, total_bytes) is called as data arrives.
    """
    import pyarrow.parquet as pq

    total = sum(f["size"] for f in files) or 1
    done_before = 0
    for f in files:
        got = [0]

        def on_bytes(n, got=got, base=done_before, size=f["size"]):
            got[0] = min(size, got[0] + n)
            if progress:
                progress(base + got[0], total)

        src = HttpRangeFile(f["url"], f["size"], token, on_bytes=on_bytes, should_stop=should_stop)
        pf = pq.ParquetFile(src)
        cols = None
        if columns:
            names = set(pf.schema_arrow.names)
            cols = [c for c in columns if c in names] or None
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            if should_stop and should_stop():
                raise InterruptedError("stopped")
            yield from batch.to_pylist()
        done_before += f["size"]
        if progress:
            progress(done_before, total)


def download(url: str, dest: Path, token: str | None = None, progress=None, should_stop=None,
             max_bytes: int | None = None) -> Path:
    """Plain download to a file (used for raw text files in a dataset repo)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(_request(url, token), timeout=60) as r, open(tmp, "wb") as f:
            n = 0
            while True:
                if should_stop and should_stop():
                    raise InterruptedError("stopped")
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                n += len(chunk)
                if progress:
                    progress(len(chunk))
                if max_bytes and n >= max_bytes:
                    break
    except urllib.error.HTTPError as e:
        raise HubError(f"Download failed ({e.code}).") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise HubError("Download failed. Check your internet connection and try again.") from e
    tmp.replace(dest)
    return dest


def schema_columns(schema: dict) -> list[str] | None:
    """Only download the columns we will actually read."""
    mode = schema.get("mode")
    if mode == "text":
        return [c for c in (schema.get("column"), schema.get("title")) if c]
    if mode == "chat_pair":
        return [c for c in (schema.get("user"), schema.get("context"), schema.get("assistant"),
                            schema.get("system")) if c]
    if mode in ("chat_messages", "chat_turns"):
        return [schema["column"]]
    return None


# --------------------------------------------------------------------------- sharing a model

def upload_folder(folder: Path, repo_name: str, token: str, private: bool = False) -> str:
    """Create (or update) a model page on Hugging Face from an exported folder. Returns its URL."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise HubError("The Hugging Face upload helper is not installed.") from e
    try:
        api = HfApi(token=token)
        user = api.whoami()["name"]
        repo_id = f"{user}/{repo_name}"
        api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                          commit_message="Upload model trained with Foolproof Trainer")
        return f"{HUB}/{repo_id}"
    except HubError:
        raise
    except Exception as e:
        text = str(e)
        if "401" in text or "Invalid" in text or "token" in text.lower():
            raise HubError("Hugging Face did not accept that token. Make a new one with WRITE "
                           "permission at huggingface.co/settings/tokens and paste it again.") from e
        raise HubError(f"The upload failed: {text[:300]}") from e
