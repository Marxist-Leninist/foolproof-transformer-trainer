"""Tokenizers: how text is chopped into the pieces ("tokens") the model reads.

Three kinds behind one interface, picked automatically from the data size:

* ``bytes``    - every letter/byte is a token. Nothing to learn, works for any
                 language, best for very small data.
* ``bpe``      - a word-piece vocabulary learned from *your* data (byte-level
                 BPE via the ``tokenizers`` library). Best for small models.
* ``tiktoken`` - OpenAI's standard vocabularies (gpt2 / cl100k_base /
                 o200k_base). Best for large English or code corpora.

Every tokenizer gets the same four special tokens appended after its own
vocabulary, so chat formatting works identically everywhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]
TIKTOKEN_NAMES = ["gpt2", "cl100k_base", "o200k_base"]


class Tokenizer:
    kind = "base"

    def __init__(self, base_vocab: int):
        self.base_vocab = int(base_vocab)
        self.vocab_size = self.base_vocab + len(SPECIALS)
        self.eot_id = self.base_vocab
        self.user_id = self.base_vocab + 1
        self.assistant_id = self.base_vocab + 2
        self.system_id = self.base_vocab + 3

    # --- to implement -----------------------------------------------------
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        raise NotImplementedError

    def _decode_base(self, ids: list[int]) -> str:
        raise NotImplementedError

    def spec(self) -> dict:
        raise NotImplementedError

    # --- shared -----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.encode_batch([text])[0]

    def decode(self, ids, show_specials: bool = False) -> str:
        out, run = [], []
        for i in ids:
            i = int(i)
            if i < self.base_vocab:
                run.append(i)
                continue
            if run:
                out.append(self._decode_base(run))
                run = []
            if show_specials and i < self.vocab_size:
                out.append(SPECIALS[i - self.base_vocab])
        if run:
            out.append(self._decode_base(run))
        return "".join(out)

    def role_id(self, role: str) -> int:
        return {"user": self.user_id, "assistant": self.assistant_id, "system": self.system_id}[role]

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "foolproof-tokenizer-1", "specials": SPECIALS, **self.spec()}, f)

    @staticmethod
    def load(path: Path) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return from_spec(d)


class ByteTokenizer(Tokenizer):
    kind = "bytes"

    def __init__(self):
        super().__init__(256)

    def encode_batch(self, texts):
        return [list(t.encode("utf-8")) for t in texts]

    def _decode_base(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def spec(self):
        return {"kind": "bytes"}


class BPETokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, tok):
        self.tok = tok
        super().__init__(tok.get_vocab_size())

    @staticmethod
    def train(text_iter, vocab_size: int) -> "BPETokenizer":
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers

        tok = HFTokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=False,
        )
        tok.train_from_iterator(text_iter, trainer=trainer)
        return BPETokenizer(tok)

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def _decode_base(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def spec(self):
        return {"kind": "bpe", "tokenizer": json.loads(self.tok.to_str())}


class TiktokenTokenizer(Tokenizer):
    kind = "tiktoken"

    def __init__(self, name: str):
        ensure_tiktoken_cache()
        import tiktoken

        self.name = name
        self.enc = tiktoken.get_encoding(name)
        self.threads = max(1, min(8, (os.cpu_count() or 2) - 1))
        super().__init__(self.enc.n_vocab)

    def encode_batch(self, texts):
        return self.enc.encode_ordinary_batch(texts, num_threads=self.threads)

    def _decode_base(self, ids):
        return self.enc.decode_bytes(ids).decode("utf-8", errors="replace")

    def spec(self):
        return {"kind": "tiktoken", "name": self.name}


def ensure_tiktoken_cache() -> None:
    """Keep tiktoken's vocabulary files in our own folder (the default is a temp dir)."""
    if os.environ.get("TIKTOKEN_CACHE_DIR"):
        return
    try:
        from . import paths
    except ImportError:  # this file is being used on its own, outside the app
        return
    d = paths.cache_dir() / "tiktoken"
    d.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(d)


def from_spec(d: dict) -> Tokenizer:
    kind = d.get("kind")
    if kind == "bytes":
        return ByteTokenizer()
    if kind == "bpe":
        from tokenizers import Tokenizer as HFTokenizer

        return BPETokenizer(HFTokenizer.from_str(json.dumps(d["tokenizer"])))
    if kind == "tiktoken":
        return TiktokenTokenizer(d["name"])
    raise ValueError(f"unknown tokenizer kind: {kind!r}")


def auto_choice(total_chars: int, non_latin_fraction: float = 0.0) -> dict:
    """Pick the tokenizer that suits the amount (and script) of text.

    Returns {"kind": ..., "vocab": ... or "name": ..., "why": plain-English reason}.
    """
    if total_chars < 2_000_000:
        return {"kind": "bytes",
                "why": "Small amount of text, so the AI will learn letter by letter. "
                       "That is the most reliable way to learn from little data."}
    if total_chars < 30_000_000:
        vocab = 4096
    elif total_chars < 200_000_000:
        vocab = 8192
    else:
        vocab = 16384
    if total_chars >= 400_000_000 and non_latin_fraction < 0.05:
        return {"kind": "tiktoken", "name": "gpt2",
                "why": "Large amount of English-style text, so it uses the standard GPT-2 "
                       "word-piece vocabulary (about 50,000 pieces)."}
    return {"kind": "bpe", "vocab": vocab,
            "why": f"It learned its own vocabulary of {vocab:,} word-pieces from your text, "
                   "which lets a small AI learn much faster than going letter by letter."}


class StreamDecoder:
    """Turns a stream of token ids into text chunks without splitting a character in half."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.pending: list[int] = []

    def push(self, token_id: int) -> str:
        if token_id >= self.tok.base_vocab:
            return self.flush()
        self.pending.append(int(token_id))
        s = self.tok._decode_base(self.pending)
        if s.endswith("�") and len(self.pending) < 8:
            return ""  # probably the first half of a multi-byte character: wait for the rest
        self.pending = []
        return s

    def flush(self) -> str:
        if not self.pending:
            return ""
        s = self.tok._decode_base(self.pending)
        self.pending = []
        return s
