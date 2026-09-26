"""FrozenCore v7 - "Aurora": architecture upgrades mined from the public
state of the art (DeepSeek-V3/V2, Qwen3, MiniCPM, MobileLLM, BitNet, YaRN)
and adapted to our Skill-Expert (NOT mixture-of-experts) design.

What we took from where (all public papers/open weights, no proprietary or
leaked code):

  DeepSeek-V3  sigmoid expert affinity + TOP-K gating with an auxiliary-
               loss-free balance BIAS updated by usage -> v2 routing below
               (bias lives in the runtime, NOT in pack files: old packs work)
  DeepSeek-V3  Multi-Token Prediction as a cheap aux loss during training
               -> implemented in train_pack (--mtp, zero new parameters)
  DeepSeek     SwiGLU FFN inside experts -> SwiGLUExpertBlock (pack v2 preview)
  YaRN         context extension via interpolated RoPE -> precompute_rope_yarn
  MobileLLM   "immediate blockwise weight reuse" philosophy -> we already
               reuse the trunk lm_head for every pack (tied, zero copies)
  MiniCPM      tiny core + on-demand capability modules (RAG-style) -> the
               existing Hub fetch/deload loop IS this pattern
  BitNet       aggressive low-bit weights -> int4 groupwise pack export
               (research preview in this module)

v2 routing contract (100% backward compatible):
  * every pack keeps its one-vs-rest head
  * affinity = sigmoid(head_logit + balance_bias)  (was: softmax/argmax)
  * per input, the TOP-2 packs run and their outputs blend by normalized
    gates; the trunk acts as the always-on "shared expert"
  * usage rebalancing moves ONLY the runtime bias, never any weight
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen import SkillMixer, TrunkModel, TrunkConfig
from .model import Block, CausalSelfAttention, RMSNorm


# ---------------------------------------------------------------------- #
# v2 runtime: sigmoid affinity + top-2 soft gating + balance bias
# ---------------------------------------------------------------------- #
class SkillMixerV2(SkillMixer):
    """Drop-in v7 runtime on top of any v6 SkillMixer state.

    Old packs load unchanged; the gating policy upgrades in place.
    """

    VERSION = 2

    def __init__(self, trunk: TrunkModel, topk: int = 2) -> None:
        super().__init__(trunk)
        self.topk = topk
        self.biases: dict[str, float] = {}

    # -- pack management (bias bookkeeping) ----------------------------- #
    def add_pack(self, pack) -> "SkillMixerV2":
        super().add_pack(pack)
        self.biases.setdefault(pack.name, 0.0)
        return self

    def drop_pack(self, name: str) -> "SkillMixerV2":
        super().drop_pack(name)
        self.biases.pop(name, None)
        return self

    # -- v2 gating -------------------------------------------------------- #
    def affinity(self, pooled: torch.Tensor) -> torch.Tensor:
        """(B, K) sigmoid(own-logit + balance bias) - DeepSeek-V3 style."""
        names = self.pack_names
        logits = torch.stack(
            [p.route_logit(pooled) + self.biases.get(nm, 0.0)
             for nm, p in zip(names, self.packs.values())], dim=1)
        return torch.sigmoid(logits)

    def route_v2(self, h: torch.Tensor):
        """Returns (per-example [(name, gate), ...] of length topk, affinity)."""
        aff = self.affinity(h.mean(dim=1))
        k = min(self.topk, aff.size(1))
        vals, idx = torch.topk(aff, k, dim=1)
        gates = vals / vals.sum(dim=1, keepdim=True).clamp_min(1e-6)
        names = self.pack_names
        sel = [[(names[int(j)], float(g.detach())) for j, g in zip(row_i, row_g)]
               for row_i, row_g in zip(idx, gates)]
        return sel, aff

    # -- forward ---------------------------------------------------------- #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                force_pack: str | None = None):
        with torch.no_grad():
            h = self.trunk.hidden(idx)
        h = h.detach()
        cos = self._cos[:, : idx.size(1)]
        sin = self._sin[:, : idx.size(1)]

        if force_pack is not None:
            sel = [[(force_pack, 1.0)] for _ in range(idx.size(0))]
            aff = None
        else:
            sel, aff = self.route_v2(h)

        # group examples by expert-combination to batch expert calls
        out = h.clone()
        combos: dict[tuple, list[int]] = {}
        for i, pairs in enumerate(sel):
            combos.setdefault(tuple(nm for nm, _ in pairs), []).append(i)
        for combo, rows in combos.items():
            rows_t = torch.tensor(rows, dtype=torch.long)
            acc = torch.zeros(len(rows), h.size(1), h.size(2))
            gsum = 0.0
            for nm in combo:
                g = next(g for n, g in sel[rows[0]] if n == nm)
                acc = acc + g * self.packs[nm].run_expert(h[rows_t], cos, sin)
                gsum += g
            out[rows_t] = acc / gsum

        logits = self.trunk.logits_from_h(out)
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "lm_loss": lm_loss, "assigned": sel,
                "affinity": aff}

    # -- auxiliary-loss-free load balancing (DeepSeek-V3) ------------------ #
    def rebalance(self, usage: dict[str, float], gamma: float = 1e-3) -> dict[str, float]:
        """bias_i += gamma * (mean_usage - usage_i): under-used packs get a
        nudge up, over-used packs down. NO weights move - this is the
        aux-loss-free trick adapted from DeepSeek-V3."""
        if not usage:
            return self.biases
        vals = [usage.get(nm, 0.0) for nm in self.packs]
        mean = sum(vals) / max(1, len(vals))
        for nm in self.packs:
            self.biases[nm] = self.biases.get(nm, 0.0) + gamma * (mean - usage.get(nm, 0.0))
        return self.biases


# ---------------------------------------------------------------------- #
# YaRN-style RoPE extension (for packs that must read longer contexts)
# ---------------------------------------------------------------------- #
def _yarn_corr_dim(num_rotations: float, head_dim: int, base: float, max_pos: float) -> float:
    return (head_dim * math.log(max_pos * num_rotations / (2 * math.pi))
            / (2 * math.log(base)))


def precompute_rope_yarn(seq_len: int, head_dim: int, base: float = 10000.0,
                         factor: float = 4.0, beta_fast: float = 32.0,
                         beta_slow: float = 1.0):
    """YaRN (ntk-by-parts + linear blend) frequency interpolation.

    Low-frequency dims are stretched by `factor` (context xN), high-frequency
    dims are kept, with a smooth ramp between - exactly the public YaRN
    recipe. Returns cos/sin shaped like the trunk rope caches.
    """
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half).float() / head_dim))
    low = max(math.floor(_yarn_corr_dim(beta_fast, head_dim, base, seq_len * factor)), 0.0)
    high = min(math.ceil(_yarn_corr_dim(beta_slow, head_dim, base, seq_len * factor)), half - 1)
    ramp = torch.clamp((torch.arange(half).float() - low) / max(1.0, high - low), 0.0, 1.0)
    t = torch.arange(seq_len).float()
    interp = t[:, None] * inv[None, :] / factor
    extra = t[:, None] * inv[None, :]
    freqs = interp * (1 - ramp[None, :]) + extra * ramp[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def run_expert_long(pack, h: torch.Tensor, factor: float = 4.0):
    """Run a pack expert over a longer window than the trunk block size,
    using a YaRN-extended rope cache (documented edge-use preview)."""
    T = h.size(1)
    cos, sin = precompute_rope_yarn(T, pack.cfg.dim // pack.cfg.n_heads,
                                    pack.cfg.rope_base, factor)
    return pack.expert(h, cos, sin)


# ---------------------------------------------------------------------- #
# SwiGLU expert (pack format v2 preview - DeepSeek/Qwen-style FFN)
# ---------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """3-matrix SwiGLU FFN; hidden sized at 2/3 to keep parameter parity."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


class SwiGLUExpertBlock(nn.Module):
    """Drop-in expert for FUTURE pack files (format v2): same attention,
    SwiGLU FFN. v1 packs remain fully supported."""

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        smt = cfg.to_smt()
        self.attn = CausalSelfAttention(smt)
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        swiglu_hidden = int(cfg.expert_hidden * 2 / 3 // 8 * 8) or 8
        self.mlp = SwiGLU(cfg.dim, swiglu_hidden, cfg.dropout)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), cos, sin)
        h = h + self.mlp(self.norm2(h))
        return h


# ---------------------------------------------------------------------- #
# int4 groupwise pack export (BitNet-style low-bit research preview)
# ---------------------------------------------------------------------- #
def export_int4(pack_path: str, out_path: str, group: int = 32) -> str:
    """Groupwise symmetric int4 export of a pack file.

    Every >=2-D weight is split along the output dim into groups of `group`;
    each group stores int4 codes (packed 2/byte) + one fp16 scale. 1-D params
    stay fp16. Load via load_int4_pack. Roughly halves size vs int8.
    """
    from .frozen import SkillPack  # local import avoids cycle at module load

    blob = torch.load(pack_path, map_location="cpu", weights_only=False)
    pack = SkillPack.load(pack_path)
    q_state = {}

    def quant(t: torch.Tensor):
        flat = t.detach().float()
        od = flat.shape[0]
        pad = (-flat.shape[1]) % group
        if pad:
            flat = F.pad(flat, (0, pad))
        g = flat.view(od, -1, group)
        scale = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
        codes = (g / scale).round().clamp(-8, 7).to(torch.int8)
        lo = codes & 0x0F
        packed = (lo[:, :, 0::2] | (lo[:, :, 1::2] << 4)).to(torch.uint8)
        return packed, scale.squeeze(-1).half().view(od, -1)

    for k, v in pack.state_dict().items():
        if v.dim() >= 2:
            packed, scale = quant(v)
            q_state[k] = {"packed": packed, "scale": scale, "shape": tuple(v.shape)}
        else:
            q_state[k] = v.half()

    torch.save({"version": blob.get("version", 1), "name": blob["name"],
                "description": blob.get("description", ""),
                "config": blob["config"], "meta": blob.get("meta", {}),
                "quant": {"bits": 4, "group": group, "state": q_state}}, out_path)
    return out_path


def load_int4_pack(path: str):
    """Materialize an int4 pack back into a normal SkillPack (dequant-on-load)."""
    from .frozen import SkillPack

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "quant" in blob, "not an int4 pack"
    group = blob["quant"]["group"]
    pack = SkillPack(blob["name"], TrunkConfig.from_dict(blob["config"]),
                     description=blob.get("description", ""))
    sd = {}
    for k, v in blob["quant"]["state"].items():
        if isinstance(v, dict):
            od, cols = v["shape"]
            padded = cols + (-cols) % group
            n_groups = padded // group
            packed = v["packed"].long()
            lo = packed & 0x0F
            hi = packed >> 4
            codes = torch.stack([lo, hi], dim=-1).reshape(od, n_groups, group)
            codes = torch.where(codes > 7, codes - 16, codes).float()
            deq = codes * v["scale"].float().unsqueeze(-1)
            sd[k] = deq.reshape(od, padded)[:, :cols].contiguous()
        else:
            sd[k] = v.float()
    pack.load_state_dict(sd)
    pack.meta = blob.get("meta", {})
    pack.meta["quantized"] = "int4-groupwise"
    return pack
