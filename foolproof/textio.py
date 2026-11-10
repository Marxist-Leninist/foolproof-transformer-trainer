"""Reading text out of whatever the user throws at us.

A "doc" is either a plain string, or a chat: a list of (role, text) pairs with
role in {"system", "user", "assistant"}.

``detect_schema`` looks at a few table rows (from a Hugging Face dataset, a
CSV, a JSON file ...) and works out which columns hold the text, so the user
never has to know what a "column" is.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

TEXT_EXT = {
    ".txt", ".text", ".md", ".markdown", ".rst", ".log", ".srt", ".vtt", ".tex", ".org",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs",
    ".rb", ".php", ".swift", ".kt", ".lua", ".sh", ".bat", ".ps1", ".sql", ".r", ".m", ".scala",
    ".css", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml",
}
ROW_EXT = {".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet"}
DOC_EXT = {".pdf", ".docx", ".epub", ".html", ".htm"}
SUPPORTED_EXT = TEXT_EXT | ROW_EXT | DOC_EXT | {".zip"}

TEXT_NAMES = ["text", "content", "story", "article", "document", "body", "passage", "code",
              "sentence", "paragraph", "abstract", "review", "poem", "quote", "summary",
              "description", "func_code_string", "whole_func_string", "markdown", "transcript"]
USER_NAMES = ["instruction", "prompt", "question", "query", "human", "user", "input_text"]
ASSISTANT_NAMES = ["output", "response", "answer", "completion", "reply", "assistant", "gpt",
                   "target", "output_text", "chosen"]
CONTEXT_NAMES = ["input", "context"]
MESSAGE_LIST_NAMES = ["messages", "conversations", "conversation", "dialog", "dialogue", "turns",
                      "utterances", "chat"]
ROLE_KEYS = ["role", "from", "speaker", "author", "sender"]
CONTENT_KEYS = ["content", "value", "text", "message", "utterance"]
ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user", "customer": "user", "question": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant", "model": "assistant",
    "ai": "assistant", "chatbot": "assistant", "answer": "assistant", "bard": "assistant",
    "system": "system",
}


# --------------------------------------------------------------------------- schema

def _as_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            return None
    return None


def _find_key(d: dict, names: list[str]):
    low = {str(k).lower(): k for k in d.keys()}
    for n in names:
        if n in low:
            return low[n]
    return None


def detect_schema(rows: list[dict]) -> dict | None:
    """Work out how to turn table rows into docs. Returns None if no text is found."""
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return None
    first = rows[0]

    # 1) a column holding a list of {role, content} messages
    for col in list(first.keys()):
        lst = None
        for r in rows[:20]:
            lst = _as_list(r.get(col))
            if lst:
                break
        if not lst:
            continue
        if isinstance(lst[0], dict):
            rk, ck = _find_key(lst[0], ROLE_KEYS), _find_key(lst[0], CONTENT_KEYS)
            if ck is not None:
                return {"mode": "chat_messages", "column": col, "role_key": rk, "content_key": ck,
                        "chat": True,
                        "describe": f"Conversations (from the column “{col}”). Your AI will learn to chat."}
        elif isinstance(lst[0], str) and str(col).lower() in MESSAGE_LIST_NAMES and len(lst) >= 2:
            return {"mode": "chat_turns", "column": col, "chat": True,
                    "describe": f"Back-and-forth dialogues (from the column “{col}”). Your AI will learn to chat."}

    # 2) question -> answer columns
    u, a = _find_key(first, USER_NAMES), _find_key(first, ASSISTANT_NAMES)
    if u is not None and a is not None and u != a:
        ctx = _find_key(first, CONTEXT_NAMES)
        if ctx in (u, a):
            ctx = None
        sysk = _find_key(first, ["system", "system_prompt"])
        extra = f" (+ “{ctx}”)" if ctx else ""
        return {"mode": "chat_pair", "user": u, "context": ctx, "assistant": a, "system": sysk,
                "chat": True,
                "describe": f"Questions from “{u}”{extra} and answers from “{a}”. Your AI will learn to answer questions."}

    # 3) a plain text column
    col = _find_key(first, TEXT_NAMES)
    if col is None or not isinstance(first.get(col), str):
        best, best_len = None, 0.0
        for c in first.keys():
            vals = [r.get(c) for r in rows[:50]]
            strs = [v for v in vals if isinstance(v, str)]
            if len(strs) < max(1, len(vals) // 2):
                continue
            avg = sum(len(s) for s in strs) / len(strs)
            if avg > best_len:
                best, best_len = c, avg
        col = best
    if col is None:
        return None
    title = _find_key(first, ["title"])
    if title == col or not isinstance(first.get(title), str):
        title = None
    return {"mode": "text", "column": col, "title": title, "chat": False,
            "describe": f"Plain text (from the column “{col}”). Your AI will learn to write text like it."}


def row_to_doc(row: dict, schema: dict):
    """Apply a schema to one row -> doc (str or list of (role, text)) or None to skip."""
    mode = schema["mode"]
    if mode == "text":
        t = row.get(schema["column"])
        if not isinstance(t, str) or not t.strip():
            return None
        title = row.get(schema.get("title")) if schema.get("title") else None
        if isinstance(title, str) and title.strip() and len(title) < 200 and not t.startswith(title):
            return title.strip() + "\n\n" + t
        return t
    if mode == "chat_pair":
        u, a = row.get(schema["user"]), row.get(schema["assistant"])
        if not isinstance(u, str) or not isinstance(a, str) or not u.strip() or not a.strip():
            return None
        ctx = row.get(schema["context"]) if schema.get("context") else None
        if isinstance(ctx, str) and ctx.strip():
            u = u.strip() + "\n\n" + ctx.strip()
        turns = []
        sysv = row.get(schema["system"]) if schema.get("system") else None
        if isinstance(sysv, str) and sysv.strip():
            turns.append(("system", sysv.strip()))
        turns += [("user", u.strip()), ("assistant", a.strip())]
        return turns
    if mode == "chat_messages":
        lst = _as_list(row.get(schema["column"]))
        if not lst:
            return None
        turns, expect = [], "user"
        for m in lst:
            if not isinstance(m, dict):
                continue
            text = m.get(schema["content_key"])
            if not isinstance(text, str) or not text.strip():
                continue
            raw = str(m.get(schema["role_key"], "")).lower() if schema.get("role_key") else ""
            role = ROLE_MAP.get(raw)
            if role is None:
                role = expect
            turns.append((role, text.strip()))
            if role != "system":
                expect = "assistant" if role == "user" else "user"
        return turns if any(r == "assistant" for r, _ in turns) else None
    if mode == "chat_turns":
        lst = _as_list(row.get(schema["column"]))
        if not lst or len(lst) < 2:
            return None
        turns = []
        for i, t in enumerate(lst):
            if isinstance(t, str) and t.strip():
                turns.append(("user" if i % 2 == 0 else "assistant", t.strip()))
        return turns if len(turns) >= 2 else None
    return None


def doc_preview(doc, limit: int = 700) -> str:
    if isinstance(doc, str):
        return doc[:limit]
    parts = []
    for role, text in doc:
        label = {"user": "Person", "assistant": "AI", "system": "Instructions"}[role]
        parts.append(f"{label}: {text}")
    return "\n".join(parts)[:limit]


# --------------------------------------------------------------------------- decoding

def decode_bytes(raw: bytes) -> str | None:
    """Bytes -> str, guessing the encoding. None if it looks like a binary file."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
             "article", "blockquote", "pre", "title"}
    SKIP = {"script", "style", "head", "nav", "svg", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    try:
        p.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    import html as _html

    return re.sub(r"\n{3,}", "\n\n", _html.unescape(text)).strip()


def _epub_text(data: bytes) -> str:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for n in sorted(names):
            out.append(html_to_text(z.read(n).decode("utf-8", errors="replace")))
    return "\n\n".join(t for t in out if t)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(p.strip() for p in pages if p and p.strip())


# --------------------------------------------------------------------------- rows

def _json_rows(obj):
    """Find the list of records inside an arbitrary JSON document."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        best = None
        for v in obj.values():
            if isinstance(v, list) and v and (best is None or len(v) > len(best)):
                best = v
        if best is not None:
            return best
        return [obj]
    return []


def _string_leaves(obj, out: list[str]):
    if isinstance(obj, str):
        if len(obj) >= 20:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _string_leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _string_leaves(v, out)


def iter_rows_from_file(path: Path):
    """Yield dict rows (or bare strings) from a .jsonl/.json/.csv/.tsv/.parquet file."""
    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    elif ext == ".json":
        text = decode_bytes(path.read_bytes())
        if text is None:
            return
        try:
            obj = json.loads(text)
        except ValueError:
            # maybe it is really JSON-lines
            for line in text.splitlines():
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
            return
        yield from _json_rows(obj)
    elif ext in (".csv", ".tsv"):
        text = decode_bytes(path.read_bytes())
        if text is None:
            return
        csv.field_size_limit(1 << 30)
        delim = "\t" if ext == ".tsv" else None
        if delim is None:
            try:
                delim = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
            except csv.Error:
                delim = ","
        yield from csv.DictReader(io.StringIO(text), delimiter=delim)
    elif ext == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=512):
            yield from batch.to_pylist()


def docs_from_rows(rows_iter, schema: dict | None = None):
    """Rows -> docs, detecting the schema from the first rows if not given."""
    buf, it = [], iter(rows_iter)
    for r in it:
        buf.append(r)
        if len(buf) >= 50:
            break
    if not buf:
        return
    if all(isinstance(r, str) for r in buf):
        for r in buf:
            if r.strip():
                yield r
        for r in it:
            if isinstance(r, str) and r.strip():
                yield r
        return
    if schema is None:
        schema = detect_schema(buf)
    if schema is None:
        leaves: list[str] = []
        _string_leaves(buf, leaves)
        for r in it:
            _string_leaves(r, leaves)
        yield from leaves
        return
    for r in buf:
        if isinstance(r, dict):
            d = row_to_doc(r, schema)
            if d:
                yield d
    for r in it:
        if isinstance(r, dict):
            d = row_to_doc(r, schema)
            if d:
                yield d


# --------------------------------------------------------------------------- files

def iter_docs_from_path(path: Path, skipped: list | None = None):
    """Yield docs from one file. Unknown/binary files are recorded in ``skipped``."""
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext == ".zip":
            yield from _iter_zip(path, skipped)
        elif ext in ROW_EXT:
            yield from docs_from_rows(iter_rows_from_file(path))
        elif ext == ".pdf":
            t = _pdf_text(path)
            if t:
                yield t
            elif skipped is not None:
                skipped.append((path.name, "no readable text (a scanned PDF?)"))
        elif ext == ".docx":
            t = _docx_text(path.read_bytes())
            if t:
                yield t
        elif ext == ".epub":
            t = _epub_text(path.read_bytes())
            if t:
                yield t
        elif ext in (".html", ".htm"):
            raw = decode_bytes(path.read_bytes())
            if raw:
                t = html_to_text(raw)
                if t:
                    yield t
        else:
            # known text types, and anything else that turns out to be text
            raw = path.read_bytes()
            t = decode_bytes(raw)
            if t is None:
                if skipped is not None:
                    skipped.append((path.name, "not a text file"))
            elif t.strip():
                yield t
    except Exception as e:  # one bad file must never sink the whole job
        if skipped is not None:
            skipped.append((path.name, f"could not read ({type(e).__name__})"))


def _iter_zip(path: Path, skipped):
    import tempfile

    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as td:
        for info in z.infolist():
            if info.is_dir() or info.file_size > 2_000_000_000:
                continue
            name = Path(info.filename).name
            if not name or name.startswith("."):
                continue
            target = Path(td) / name
            with z.open(info) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            if target.suffix.lower() == ".zip":
                continue
            yield from iter_docs_from_path(target, skipped)
            try:
                target.unlink()
            except OSError:
                pass


def split_segments(text: str, seg_chars: int):
    """Cut a long text into pieces of about seg_chars, preferring to cut at line breaks."""
    n = len(text)
    if n <= seg_chars * 1.5:
        yield text
        return
    start = 0
    while start < n:
        end = min(n, start + seg_chars)
        if end < n:
            cut = text.rfind("\n", start + seg_chars // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + seg_chars // 2, end)
            if cut != -1:
                end = cut + 1
        yield text[start:end]
        start = end
