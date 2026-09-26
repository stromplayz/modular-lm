"""FrozenCore Skill-Modular LM (v6) - packs, not one big model.

Architecture contract:

    ┌───────────────────────────────────────────────────────────┐
    │  TRUNK  (FROZEN FOREVER - the "main model")               │
    │  wte + base blocks + norm_f + tied lm_head                │
    │  trained ONCE (genesis); never receives gradients again   │
    └───────────────┬───────────────────────────────────────────┘
                    │ pooled hidden state h (detached)
        ┌───────────┴─────────────┐
        │  SKILL PACKS (trainable)│   each pack = 1 file on disk:
        │  pack: expert block     │   packs/<name>.pack
        │  pack: one-vs-rest head │   -> logit "is this MY input?"
        └───────────┬─────────────┘
                    │ softmax over LOADED packs only  (co-loading)
                    ▼
        chosen pack's expert runs -> trunk lm_head -> tokens

Why this gives "train only submodels":
  * trunk params are leaf-frozen and run under no_grad -> gradients
    physically cannot reach the main model
  * pack files are independent: training pack A rewrites only packs/A.pack
  * co-loading: load ANY subset of packs; routing happens among loaded
    packs only, so new packs compose with old ones with zero retraining

Zero-forgetting: because old packs' files are never opened for writing
and the trunk never changes, benchmark scores of old skills are bit-identical
before and after training a new pack (verified by bench_lua).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (Block, RMSNorm, SkillModularConfig, _rope_cache)


# ---------------------------------------------------------------------- #
# config
# ---------------------------------------------------------------------- #
@dataclass
class TrunkConfig:
    vocab_size: int = 1024
    dim: int = 128
    n_heads: int = 4
    block_size: int = 256
    n_base_layers: int = 2
    expert_hidden: int = 192
    dropout: float = 0.1
    rope_base: float = 10000.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrunkConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_smt(self) -> SkillModularConfig:
        """View as the v5 monolithic config (for Block construction)."""
        return SkillModularConfig(
            vocab_size=self.vocab_size, dim=self.dim, n_heads=self.n_heads,
            block_size=self.block_size, n_base_layers=self.n_base_layers,
            n_skills=1, expert_hidden=self.expert_hidden,
            dropout=self.dropout, rope_base=self.rope_base,
            skill_names=("unused",),
        )


# ---------------------------------------------------------------------- #
# frozen trunk
# ---------------------------------------------------------------------- #
class TrunkModel(nn.Module):
    """The main model - shared language core, frozen after genesis."""

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        smt_cfg = cfg.to_smt()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.base_blocks = nn.ModuleList(Block(smt_cfg) for _ in range(cfg.n_base_layers))
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied embeddings
        self.drop = nn.Dropout(cfg.dropout)

        cos, sin = _rope_cache(cfg.block_size, cfg.dim // cfg.n_heads,
                               cfg.rope_base, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # -- the only entry points downstream code may use ------------------ #
    def hidden(self, idx: torch.Tensor) -> torch.Tensor:
        cos = self.rope_cos[:, : idx.size(1)]
        sin = self.rope_sin[:, : idx.size(1)]
        x = self.drop(self.wte(idx))
        for block in self.base_blocks:
            x = block(x, cos, sin)
        return x

    def logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm_f(h))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        h = self.hidden(idx)
        logits = self.logits_from_h(h)
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return {"logits": logits, "lm_loss": lm_loss, "h": h}

    def freeze(self) -> None:
        """Leaf-freeze every parameter + force eval/no_grad discipline."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str, meta: dict | None = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"trunk": self.state_dict(), "config": self.cfg.to_dict(),
                    "meta": meta or {}}, path)
        return path

    @classmethod
    def load(cls, path: str, frozen: bool = True) -> "TrunkModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(TrunkConfig.from_dict(blob["config"]))
        m.load_state_dict(blob["trunk"])
        return m.freeze() if frozen else m


# ---------------------------------------------------------------------- #
# skill pack
# ---------------------------------------------------------------------- #
class SkillPack(nn.Module):
    """A self-contained, independently-trained skill submodel.

    expert      : one transformer Block operating on trunk hidden states
    router_head : one-vs-rest logit - "does this input belong to MY skill?"

    Because the router is one-vs-rest (not one-of-N), any SET of packs can
    be loaded together and routing is a softmax across whoever is loaded.
    """

    FILE_VERSION = 1

    def __init__(self, name: str, cfg: TrunkConfig, description: str = "") -> None:
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.description = description
        self.expert = Block(cfg.to_smt())
        self.router_head = nn.Linear(cfg.dim, 1, bias=False)
        # depth-scaled residual init (block sits on top of trunk residuals)
        nn.init.normal_(self.expert.attn.proj.weight, std=0.02 / math.sqrt(4))
        nn.init.normal_(self.expert.mlp.proj.weight, std=0.02 / math.sqrt(4))
        nn.init.normal_(self.router_head.weight, std=0.01)

    # -- routing -------------------------------------------------------- #
    def route_logit(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: (B, dim) -> (B,) one-vs-rest logits."""
        return self.router_head(pooled).squeeze(-1)

    # -- forward (use SkillMixer, or run_expert with rope caches) -------- #
    def run_expert(self, h: torch.Tensor, cos, sin) -> torch.Tensor:
        return self.expert(h, cos, sin)

    # -- persistence ----------------------------------------------------- #
    def save(self, path: str, meta: dict | None = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = {
            "version": self.FILE_VERSION,
            "name": self.name,
            "description": self.description,
            "config": self.cfg.to_dict(),
            "state": self.state_dict(),
            "meta": meta or {},
        }
        torch.save(blob, path)
        return path

    @classmethod
    def load(cls, path: str) -> "SkillPack":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        pack = cls(blob["name"], TrunkConfig.from_dict(blob["config"]),
                   description=blob.get("description", ""))
        pack.load_state_dict(blob["state"])
        pack.meta = blob.get("meta", {})
        return pack

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------- #
# co-loading runtime
# ---------------------------------------------------------------------- #
class SkillMixer(nn.Module):
    """Load trunk + ANY subset of packs. Route among loaded packs only.

    packs = {"qa": SkillPack, ...} - insertion order breaks ties (first
    loaded wins) when two one-vs-rest heads fire equally.
    """

    def __init__(self, trunk: TrunkModel) -> None:
        super().__init__()
        self.trunk = trunk.freeze()
        self.packs: dict[str, SkillPack] = {}
        self._cos = trunk.rope_cos
        self._sin = trunk.rope_sin

    # -- pack management ------------------------------------------------- #
    def add_pack(self, pack: SkillPack) -> "SkillMixer":
        assert pack.cfg == self.trunk.cfg or pack.cfg.to_dict() == self.trunk.cfg.to_dict(), \
            f"pack {pack.name} config mismatch with trunk"
        pack = pack.to("cpu")
        pack.eval()  # inference-safe: dropout off for every loaded pack
        self.packs[pack.name] = pack
        return self

    def add_pack_file(self, path: str) -> "SkillMixer":
        return self.add_pack(SkillPack.load(path))

    def drop_pack(self, name: str) -> "SkillMixer":
        self.packs.pop(name, None)
        return self

    @property
    def pack_names(self) -> list[str]:
        return list(self.packs.keys())

    # -- routing --------------------------------------------------------- #
    def route(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (assigned pack index per example, score matrix (B, K))."""
        pooled = h.mean(dim=1)
        logits = torch.stack([p.route_logit(pooled) for p in self.packs.values()], dim=1)
        return logits.argmax(dim=-1), logits

    def name_of(self, idx: int) -> str:
        return self.pack_names[idx]

    # -- forward ---------------------------------------------------------- #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                force_pack: str | int | None = None):
        with torch.no_grad():          # <- gradients can NEVER reach the trunk
            h = self.trunk.hidden(idx)
        h = h.detach()

        if force_pack is not None:
            assigned_names = ([force_pack] * idx.size(0) if isinstance(force_pack, str)
                              else [self.pack_names[force_pack]] * idx.size(0))
            scores = None
        else:
            aidx, scores = self.route(h)
            assigned_names = [self.pack_names[int(i)] for i in aidx]

        out = h
        cos = self._cos[:, : idx.size(1)]
        sin = self._sin[:, : idx.size(1)]
        for nm in set(assigned_names):
            rows = [i for i, x in enumerate(assigned_names) if x == nm]
            rows_t = torch.tensor(rows, dtype=torch.long)
            out = out.index_copy(0, rows_t, self.packs[nm].run_expert(h[rows_t], cos, sin))

        logits = self.trunk.logits_from_h(out)
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return {"logits": logits, "lm_loss": lm_loss, "assigned": assigned_names,
                "scores": scores}

    # -- generation -------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 48,
                 temperature: float = 0.8, top_k: int = 40,
                 pack: str | None = None, greedy: bool = False):
        """pack=None -> auto route on the prompt; else pin one pack."""
        if pack is None and self.packs:
            h = self.trunk.hidden(idx)
            aidx, _ = self.route(h.detach())
            pack = self.pack_names[int(aidx[0])]
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.trunk.cfg.block_size:]
            with torch.no_grad():
                h = self.trunk.hidden(idx_c).detach()
            if pack is not None:
                cos = self._cos[:, : idx_c.size(1)]
                sin = self._sin[:, : idx_c.size(1)]
                h = self.packs[pack].run_expert(h, cos, sin)
            logits = self.trunk.logits_from_h(h)[:, -1, :]
            if greedy:
                idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
                continue
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx, pack

    # -- stats --------------------------------------------------------------- #
    def n_params(self, loaded_only: bool = True) -> int:
        n = self.trunk.n_params()
        if loaded_only:
            n += sum(p.n_params() for p in self.packs.values())
        return n


# ---------------------------------------------------------------------- #
# genesis: split the v5 monolithic checkpoint into trunk + packs
# ---------------------------------------------------------------------- #
def genesis_from_v5(ckpt_path: str, out_dir: str,
                    skill_names: tuple[str, ...] | None = None) -> dict:
    """One-time surgery: v5 monolithic ckpt -> frozen trunk + per-skill packs.

    The v5 model was trained end-to-end; its weights ARE the genesis trunk.
    From this point on the trunk file is immutable - all future learning
    happens in pack files only.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg5 = SkillModularConfig.from_dict(blob["config"])
    names = skill_names or tuple(cfg5.skill_names)
    assert len(names) == cfg5.n_skills, "skill name/count mismatch with ckpt"

    tcfg = TrunkConfig(vocab_size=cfg5.vocab_size, dim=cfg5.dim, n_heads=cfg5.n_heads,
                       block_size=cfg5.block_size, n_base_layers=cfg5.n_base_layers,
                       expert_hidden=cfg5.expert_hidden, dropout=cfg5.dropout,
                       rope_base=cfg5.rope_base)

    # --- trunk ---
    trunk = TrunkModel(tcfg)
    sd5 = blob["model"]
    trunk_sd = {}
    for k in ["wte.weight", "norm_f.weight", "lm_head.weight"]:
        trunk_sd[k] = sd5[k]
    for i in range(cfg5.n_base_layers):
        for k, v in sd5.items():
            if k.startswith(f"base_blocks.{i}."):
                trunk_sd[k] = v
    trunk.load_state_dict(trunk_sd)
    trunk.freeze()
    trunk_path = os.path.join(out_dir, "trunk.pt")
    trunk.save(trunk_path, meta={
        "genesis": f"extracted from v5 ckpt {os.path.basename(ckpt_path)} @ step {blob.get('step')}",
        "policy": "FROZEN - never trained again; all learning happens in packs",
    })

    # --- packs (expert blocks only; router heads start fresh) ---
    os.makedirs(os.path.join(out_dir, "packs"), exist_ok=True)
    made = []
    for s, name in enumerate(names):
        pack = SkillPack(name, tcfg, description=f"genesis pack split from v5 expert #{s}")
        expert_sd = {}
        for k, v in sd5.items():
            if k.startswith(f"skill_blocks.{s}."):
                expert_sd[k.replace(f"skill_blocks.{s}.", "expert.")] = v
        pack.load_state_dict(expert_sd, strict=False)
        path = os.path.join(out_dir, "packs", f"{name}.pack")
        pack.save(path, meta={"origin": "v5-genesis-split"})
        made.append(path)
    return {"trunk": trunk_path, "packs": made, "trunk_params": trunk.n_params(),
            "pack_params": SkillPack(names[0], tcfg).n_params()}
