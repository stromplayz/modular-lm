"""Skill-Modular Transformer (SMT) - a from-scratch tiny language model.

Architecture:
    tokens ──► shared trunk (the "base brain": language understanding)
           ──► SKILL ROUTER (reads pooled trunk state, picks ONE skill)
           ──► the chosen Skill Expert block runs  (load -> use -> detach)
           ──► shared LM head

Not mixture-of-experts: there are no anonymous per-token expert shards.
There are dedicated SKILL modules - each one block trained to own a task
(story / qa / math / count) - and a router that loads exactly the right
one per input. Only ~1/(1+n_skills) of the skill capacity is ever active
for a given request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SkillModularConfig:
    vocab_size: int = 1024
    dim: int = 128
    n_heads: int = 4
    block_size: int = 256
    n_base_layers: int = 2
    n_skills: int = 4
    expert_hidden: int = 192
    dropout: float = 0.1
    rope_base: float = 10000.0
    skill_names: tuple = field(default=("story", "qa", "math", "count"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["skill_names"] = list(self.skill_names)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SkillModularConfig":
        d = dict(d)
        d["skill_names"] = tuple(d.get("skill_names", ("story", "qa", "math", "count")))
        return cls(**d)


# ---------------------------------------------------------------------- #
# building blocks
# ---------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype)).to(dtype)


def _rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                      # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (T, hd)  half-split convention
    return emb.cos().to(dtype)[None, :, None, :], emb.sin().to(dtype)[None, :, None, :]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, hd); cos/sin: (1, T, 1, hd)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SkillModularConfig) -> None:
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q = q.transpose(1, 2)  # (B, H, T, hd)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP, both with residuals."""

    def __init__(self, cfg: SkillModularConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim)
        self.mlp = MLP(cfg.dim, cfg.expert_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------- #
# the model
# ---------------------------------------------------------------------- #
class SkillModularLM(nn.Module):
    def __init__(self, cfg: SkillModularConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.base_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_base_layers))
        self.skill_blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_skills))
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied embeddings
        self.router = nn.Linear(cfg.dim, cfg.n_skills, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        cos, sin = _rope_cache(cfg.block_size, cfg.dim // cfg.n_heads,
                               cfg.rope_base, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # depth-scaled residual init
        n_res = cfg.n_base_layers + cfg.n_skills
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_res))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    def _trunk(self, idx: torch.Tensor) -> torch.Tensor:
        cos = self.rope_cos[:, : idx.size(1)]
        sin = self.rope_sin[:, : idx.size(1)]
        x = self.drop(self.wte(idx))
        for block in self.base_blocks:
            x = block(x, cos, sin)
        return x

    def _dispatch_skills(self, h: torch.Tensor, assigned: torch.Tensor) -> torch.Tensor:
        """Load -> use -> detach: run ONLY the skill block each example needs.

        `assigned` is (B,) int64 skill ids. Examples are routed per-example,
        so a single batch can exercise several skills at once while each
        skill block touches exactly its own examples and nothing else.
        """
        out = h
        for s, block in enumerate(self.skill_blocks):
            rows = (assigned == s).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue  # skill not needed by anyone in this batch: stays detached
            cos = self.rope_cos[:, : h.size(1)]
            sin = self.rope_sin[:, : h.size(1)]
            out = out.index_copy(0, rows, block(h[rows], cos, sin))
        return out

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        skill_label: torch.Tensor | None = None,
        force_route: bool = False,
    ):
        """Returns dict(logits, lm_loss, router_loss, assigned, skill_logits)."""
        h = self._trunk(idx)
        pooled = h.mean(dim=1)                 # (B, C)
        skill_logits = self.router(pooled)     # (B, S)

        if force_route and skill_label is not None:
            assigned = skill_label
            router_loss = F.cross_entropy(skill_logits, skill_label)
        else:
            assigned = skill_logits.argmax(dim=-1)
            router_loss = torch.tensor(0.0, device=idx.device)
            if skill_label is not None:
                router_loss = F.cross_entropy(skill_logits, skill_label)

        h = self._dispatch_skills(h, assigned)
        logits = self.lm_head(self.norm_f(h))

        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return {
            "logits": logits,
            "lm_loss": lm_loss,
            "router_loss": router_loss,
            "assigned": assigned,
            "skill_logits": skill_logits,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        skill: int | str = "auto",
        greedy: bool = False,
    ):
        """Auto mode: the router reads the prompt once and loads one skill;
        that skill stays attached for the whole generation."""
        if isinstance(skill, str):
            assert skill == "auto", f"unknown skill {skill!r}"
            h = self._trunk(idx)
            skill = int(self.router(h.mean(dim=1)).argmax(dim=-1)[0].item())
        assigned = torch.full((idx.size(0),), int(skill), dtype=torch.long, device=idx.device)

        for _ in range(max_new_tokens):
            idx_c = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            h = self._trunk(idx_c)
            h = self._dispatch_skills(h, assigned[: idx_c.size(0)] if assigned.size(0) != idx_c.size(0) else assigned)
            logits = self.lm_head(self.norm_f(h))[:, -1, :]
            if greedy:
                idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
                continue
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx, int(skill)

    # ------------------------------------------------------------------ #
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_params(self) -> int:
        """Params touched for a single request: embeddings (tied head) +
        shared trunk + ONE skill block + router."""
        n = sum(p.numel() for p in self.base_blocks.parameters())
        n += sum(p.numel() for p in self.skill_blocks[0].parameters())
        n += self.wte.weight.numel() + sum(p.numel() for p in self.norm_f.parameters())
        n += sum(p.numel() for p in self.router.parameters())
        return n
