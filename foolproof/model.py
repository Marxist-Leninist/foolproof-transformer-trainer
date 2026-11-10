"""A small, modern GPT: the "brain" that gets trained.

Decoder-only transformer with rotary position embeddings (RoPE), RMSNorm
pre-normalisation, flash attention, tied input/output embeddings and a
key/value cache so chatting with it is fast.  Deliberately one readable file.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256  # how many tokens the model can look back over
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        keys = ModelConfig.__dataclass_fields__.keys()
        return ModelConfig(**{k: d[k] for k in keys if k in d})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "head size must be even for rotary embeddings"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: dict | None = None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, n_head, T, head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        mask = None
        causal = True
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
                causal = False
                if T > 1:  # several new tokens on top of a cache: build the mask by hand
                    total = k.size(2)
                    mask = torch.ones(T, total, dtype=torch.bool, device=x.device).tril(total - T)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal and mask is None,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, cache))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied: same table reads tokens in and out
        cos, sin = rope_tables(cfg.block_size, cfg.n_embd // cfg.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):  # residual outputs start small: stable deep nets
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos_offset: int = 0):
        B, T = idx.shape
        if pos_offset + T > self.cfg.block_size:
            raise ValueError(f"sequence too long: {pos_offset + T} > {self.cfg.block_size}")
        cos = self.rope_cos[pos_offset:pos_offset + T]
        sin = self.rope_sin[pos_offset:pos_offset + T]
        x = self.drop(self.wte(idx))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, None if caches is None else caches[i])
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return self.lm_head(x[:, -1:, :]), None  # at inference only the last position matters

    def optimizer(self, lr: float, weight_decay: float, fused: bool):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
        if fused:
            try:
                return torch.optim.AdamW(groups, fused=True, **kwargs)
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(groups, **kwargs)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float | None = None, stop_ids=()):
        """Yield new token ids one at a time (ints). idx: (1, T) prompt."""
        was_training = self.training
        self.eval()
        try:
            block = self.cfg.block_size
            seq = idx[:, -block:]
            if seq.size(1) == 0:
                raise ValueError("empty prompt")
            caches = [dict() for _ in self.blocks]
            logits, _ = self(seq, caches=caches, pos_offset=0)
            pos = seq.size(1)
            stop = set(int(s) for s in stop_ids)
            for _ in range(max_new_tokens):
                nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p)
                tok = int(nxt.item())
                yield tok
                if tok in stop:
                    break
                seq = torch.cat((seq, nxt), dim=1)
                if pos >= block:
                    # context is full: keep the most recent half and rebuild the cache
                    seq = seq[:, -(block // 2):]
                    caches = [dict() for _ in self.blocks]
                    logits, _ = self(seq, caches=caches, pos_offset=0)
                    pos = seq.size(1)
                else:
                    logits, _ = self(nxt, caches=caches, pos_offset=pos)
                    pos += 1
        finally:
            if was_training:
                self.train()


def sample_next(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None):
    """logits: (1, vocab) -> (1, 1) chosen token id."""
    logits = logits.float()
    if temperature is None or temperature <= 1e-5:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
        cum = sorted_p.cumsum(dim=-1)
        drop = (cum - sorted_p) > top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_p, 1)
        return sorted_i.gather(-1, choice)
    return torch.multinomial(probs, 1)

