"""Genesis: create the FrozenCore system from the v5 monolithic checkpoint.

Step 1  split   : v5 ckpt -> ckpt/trunk.pt (FROZEN) + packs/<skill>.pack
Step 2  routers : train each pack's one-vs-rest router head on pooled trunk
                  features (positives = own skill corpus, negatives = every
                  other skill's corpus). Trunk + expert blocks stay frozen.

After genesis, the main model NEVER trains again. New skills = new pack
files (see train_pack.py); existing files are never modified.

Usage:
    python -m skill_lm.genesis --ckpt ckpt/model.pt --out ckpt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from . import data as D
from .frozen import TrunkConfig, TrunkModel, SkillPack, SkillMixer, genesis_from_v5
from .tokenizer import BPETokenizer, GPT2_SPLIT_PATTERN

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(msg, flush=True)


def fast_encode(tok: BPETokenizer, text: str, cache: dict) -> torch.Tensor:
    out: list[int] = []
    get = cache.get
    for w in GPT2_SPLIT_PATTERN.findall(text):
        ids = get(w)
        if ids is None:
            ids = tok._encode_chunk(w.encode("utf-8"))
            cache[w] = ids
        out.extend(ids)
    return torch.tensor(out, dtype=torch.int64)


def get_batch(t: torch.Tensor, block: int, bs: int) -> torch.Tensor:
    ix = torch.randint(len(t) - block - 1, (bs,))
    return torch.stack([t[i: i + block] for i in ix]).long()


# ---------------------------------------------------------------------- #
def build_corpora(args) -> dict[str, torch.Tensor]:
    """Small per-skill corpora for router-head training (trunk stays frozen)."""
    story_path = os.path.join(args.data_dir, "tinystories.txt")
    if not os.path.exists(story_path):
        try:
            log(f"[data] downloading TinyStories slice ({args.story_mb} MB)...")
            D.download_tinystories(story_path, max_mb=args.story_mb)
        except Exception as exc:  # noqa: BLE001
            log(f"[data] WARNING: story download failed ({exc}); continuing without story")
    texts = D.build_skill_texts(
        story_path=story_path if os.path.exists(story_path) else None,
        facts_path=os.path.join(REPO, "assets", "facts.txt"),
        optometry_path=os.path.join(REPO, "assets", "optometry.txt"),
        story_mb=args.story_mb,
        n_math=args.n_math, n_count=args.n_count,
        n_qa=args.n_qa, n_optometry=args.n_optometry, seed=args.seed,
    )
    tok = BPETokenizer.load(os.path.join(args.ckpt_dir, "tokenizer.json"))
    cache: dict = {}
    tensors: dict[str, torch.Tensor] = {}
    for name, txt in texts.items():
        t = fast_encode(tok, txt, cache)
        if len(t) > args.block + 2:
            tensors[name] = t
            log(f"[data] {name:<9} {len(t):>9,} tokens")
    return tensors


@torch.no_grad()
def pool_features(trunk: TrunkModel, tensors: dict[str, torch.Tensor],
                  block: int, per_skill: int, bs: int,
                  short_frac: float = 0.5, short_min: int = 6, short_max: int = 64,
                  seed: int = 0) -> dict[str, torch.Tensor]:
    """Pooled trunk features per skill - computed ONCE (trunk is frozen).

    Domain-matched augmentation: at inference the router sees SHORT prompts,
    but a 256-token window pooled over dozens of examples looks nothing like
    that. So half the features come from variable-length windows (6..64
    tokens) - this closes the long-window/short-prompt gap and is the single
    biggest routing-accuracy win over v5.
    """
    g = torch.Generator().manual_seed(seed)
    feats: dict[str, list] = {k: [] for k in tensors}
    trunk.eval()
    for name, t in tensors.items():
        need = per_skill
        while need > 0:
            b = min(bs, need)
            rows = []
            for _ in range(b):
                if torch.rand(1, generator=g).item() < short_frac:
                    ln = int(torch.randint(short_min, short_max + 1, (1,), generator=g))
                    ln = min(ln, len(t) - 2)
                    s = int(torch.randint(0, len(t) - ln - 1, (1,), generator=g))
                else:
                    ln = block
                    s = int(torch.randint(0, len(t) - ln - 1, (1,), generator=g))
                rows.append(t[s: s + ln])
            L = max(r.numel() for r in rows)
            x = torch.stack([torch.cat([r, r[:1].repeat(L - r.numel())]) for r in rows]).long()
            h = trunk.hidden(x)
            # mean-pool over TRUE lengths (causal attn keeps prefix clean; mask pads)
            lens = torch.tensor([r.numel() for r in rows], dtype=torch.float32)
            mask = (torch.arange(L)[None, :] < lens[:, None]).float().unsqueeze(-1)
            pooled = (h * mask).sum(1) / lens.unsqueeze(1).clamp(min=1)
            feats[name].append(pooled.cpu())
            need -= b
    return {k: torch.cat(v) for k, v in feats.items()}


def train_router_head(pack: SkillPack, feats: dict[str, torch.Tensor],
                      own: str, epochs: int = 60, lr: float = 5e-3,
                      seed: int = 0) -> float:
    """One-vs-rest BCE: own skill -> 1, every other loaded-world skill -> 0."""
    rng = random.Random(seed)
    others = [v for k, v in feats.items() if k != own]
    neg_all = torch.cat(others)
    pos_all = feats[own]
    n = min(len(pos_all), len(neg_all))
    pos_all, neg_all = pos_all[:n], neg_all[:n]

    opt = torch.optim.AdamW(pack.router_head.parameters(), lr=lr, weight_decay=0.0)
    best_acc = 0.0
    for ep in range(epochs):
        cur_lr = lr * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * ep / max(1, epochs))))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        perm = torch.randperm(n)
        correct = total = 0
        for i in range(0, n, 256):
            idx = perm[i: i + 256]
            pos, neg = pos_all[idx], neg_all[idx]
            x = torch.cat([pos, neg])
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
            logit = pack.route_logit(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += (((logit > 0) == (y > 0.5)).sum().item())
            total += len(y)
        best_acc = max(best_acc, correct / total)
    return best_acc


def calibrate_router_heads(packs: list[SkillPack], feats: dict[str, torch.Tensor],
                           epochs: int = 200, lr: float = 5e-3) -> float:
    """Joint softmax calibration over the co-loaded world (the REAL objective).

    Independent BCE heads can all fire on the same input; co-loading routes by
    softmax across loaded heads, so after the BCE warm-start we jointly
    calibrate all heads with cross-entropy over the loaded-pack softmax.
    Only router heads (128 params each) move - experts + trunk stay frozen.
    """
    heads = [p.router_head for p in packs]
    names = [p.name for p in packs]
    name_to_i = {n: i for i, n in enumerate(names)}
    samples, labels = [], []
    for name, f in feats.items():
        if name not in name_to_i:
            continue
        samples.append(f)
        labels.append(torch.full((len(f),), name_to_i[name], dtype=torch.long))
    X, Y = torch.cat(samples), torch.cat(labels)
    n = len(X)
    opt = torch.optim.AdamW([{"params": h.parameters()} for h in heads],
                            lr=lr, weight_decay=0.0)
    best = 0.0
    for ep in range(epochs):
        cur = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ep / max(1, epochs))))
        for g in opt.param_groups:
            g["lr"] = cur
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, 512):
            idx = perm[i: i + 512]
            logits = torch.cat([h(X[idx]) for h in heads], dim=1)  # (b, K)
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == Y[idx]).sum().item()
        best = max(best, correct / n)
    return best


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(REPO, "ckpt", "model.pt"),
                   help="v5 monolithic checkpoint (genesis source)")
    p.add_argument("--ckpt-dir", default=os.path.join(REPO, "ckpt"),
                   help="dir holding tokenizer.json; trunk.pt written here")
    p.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    p.add_argument("--story-mb", type=float, default=4.0)
    p.add_argument("--n-math", type=int, default=12_000)
    p.add_argument("--n-count", type=int, default=8_000)
    p.add_argument("--n-qa", type=int, default=6_000)
    p.add_argument("--n-optometry", type=int, default=6_000)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--per-skill", type=int, default=4000,
                   help="pooled feature samples per skill for router heads")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    t0 = time.time()

    # ---- step 1: split ------------------------------------------------- #
    log("[1/4] genesis split: v5 ckpt -> frozen trunk + expert packs")
    res = genesis_from_v5(args.ckpt, args.ckpt_dir)
    log(f"      trunk  -> {res['trunk']}  ({res['trunk_params']:,} params, FROZEN)")
    for path in res["packs"]:
        log(f"      pack   -> {path}  ({res['pack_params']:,} params)")

    # ---- step 2: pooled features ---------------------------------------- #
    log("[2/4] pooled trunk features for router training (trunk frozen, no_grad)")
    tensors = build_corpora(args)
    trunk = TrunkModel.load(res["trunk"], frozen=True)
    feats = pool_features(trunk, tensors, args.block, args.per_skill, bs=64)
    for k, v in feats.items():
        log(f"      feats[{k:<9}] {tuple(v.shape)}")

    # ---- step 3: one-vs-rest router heads -------------------------------- #
    log("[3/4] training one-vs-rest router heads per pack (BCE warm-start)")
    tcfg = trunk.cfg
    packs = []
    for path in res["packs"]:
        name = os.path.splitext(os.path.basename(path))[0]
        pack = SkillPack.load(path)
        acc = train_router_head(pack, feats, own=name, epochs=args.epochs, seed=args.seed)
        log(f"      {name:<9} one-vs-rest router acc {acc * 100:.1f}%")
        packs.append((path, pack))

    # ---- step 4: joint calibration for co-loading ------------------------- #
    log("[4/4] joint router calibration (softmax over co-loaded packs)")
    acc = calibrate_router_heads([p for _, p in packs], feats, epochs=250)
    log(f"      co-loaded routing acc {acc * 100:.1f}%")
    for path, pack in packs:
        pack.save(path, meta={"origin": "v5-genesis-split",
                              "router_acc_1vr": round(acc, 4),
                              "calibrated": "joint-softmax"})
        log(f"      saved {path}")

    # ---- sanity: co-load everything and route ---------------------------- #
    mixer = SkillMixer(trunk)
    for path in res["packs"]:
        mixer.add_pack_file(path)
    tok = BPETokenizer.load(os.path.join(args.ckpt_dir, "tokenizer.json"))
    probes = {
        "story": "One day, a little girl named",
        "qa": "Q: What is the capital of France?\n",
        "math": "What is 7 + 8?\n",
        "count": "How many letters are in the word 'apple'?\n",
        "optometry": "Eye Q: What does OD mean on a prescription?\n",
    }
    log("co-load routing sanity check (all packs loaded together):")
    ok = tot = 0
    for want, prompt in probes.items():
        if want not in mixer.packs:
            continue
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            h = mixer.trunk.hidden(ids)
            aidx, _ = mixer.route(h.detach())
        picked = mixer.name_of(int(aidx[0]))
        ok += picked == want
        tot += 1
        log(f"      {prompt.strip()!r:<55} -> {picked}")
    log(f"      routing {ok}/{tot}")
    log(f"[done] genesis complete in {time.time() - t0:.0f}s - trunk is now FROZEN")


if __name__ == "__main__":
    main()
