"""Train the Skill-Modular Language Model.

Round-robin skill batches: every step trains ONE skill's language modeling
(its expert block force-loaded) plus the Skill Router (cross-entropy on the
skill label read from the shared trunk). Story data comes from TinyStories,
QA from the bundled facts bank, math/count are generated.

Usage:
    python -m skill_lm.train --steps 3000 --out ckpt
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch

from . import data as D
from .model import SkillModularConfig, SkillModularLM
from .tokenizer import BPETokenizer, GPT2_SPLIT_PATTERN

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------- #
def fast_encode(tok: BPETokenizer, text: str, cache: dict) -> torch.Tensor:
    """Memoized encode - words repeat heavily, this is 10-50x faster."""
    out: list[int] = []
    get = cache.get
    for w in GPT2_SPLIT_PATTERN.findall(text):
        ids = get(w)
        if ids is None:
            ids = tok._encode_chunk(w.encode("utf-8"))
            cache[w] = ids
        out.extend(ids)
    t = torch.tensor(out, dtype=torch.int64)
    return t


def build_all(args, log=log):
    story_path = os.path.join(args.data_dir, "tinystories.txt")
    if not os.path.exists(story_path):
        log(f"[data] downloading TinyStories slice ({args.story_mb} MB)...")
        try:
            D.download_tinystories(story_path, max_mb=args.story_mb)
        except Exception as exc:  # noqa: BLE001
            log(f"[data] WARNING: download failed ({exc}); continuing without story data")
    facts_path = os.path.join(REPO, "assets", "facts.txt")
    optometry_path = os.path.join(REPO, "assets", "optometry.txt")

    log("[data] building skill corpora...")
    texts = D.build_skill_texts(
        story_path=story_path if os.path.exists(story_path) else None,
        facts_path=facts_path,
        optometry_path=optometry_path,
        story_mb=args.story_mb,
        n_math=args.n_math, n_count=args.n_count, n_qa=args.n_qa,
        n_optometry=args.n_optometry, seed=args.seed,
    )
    missing = [s for s in D.SKILL_NAMES if s not in texts]
    if missing:
        raise RuntimeError(f"missing skill corpora: {missing}")

    tok_path = os.path.join(args.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
        log(f"[tok ] loaded tokenizer ({tok.vocab_size} ids) from {tok_path}")
    else:
        sample = texts["story"][:1_500_000] + texts["qa"][:500_000] + \
                 texts["optometry"][:500_000] + \
                 texts["math"][:600_000] + texts["count"][:600_000]
        tok = BPETokenizer.train(sample, args.vocab)
        os.makedirs(args.out, exist_ok=True)
        tok.save(tok_path)
        log(f"[tok ] trained BPE: vocab {tok.vocab_size} -> {tok_path}")

    os.makedirs(args.out, exist_ok=True)
    cache: dict = {}
    tensors = {}
    for name in D.SKILL_NAMES:
        t = fast_encode(tok, texts[name], cache)
        n_val = max(int(0.02 * len(t)), 64)
        tensors[name] = {"train": t[:-n_val], "val": t[-n_val:]}
        log(f"[data] {name:<6} {len(t):>9,} tokens (train {len(t) - n_val:,} / val {n_val:,})")
    return tok, tensors


# ---------------------------------------------------------------------- #
def get_batch(t: torch.Tensor, block: int, bs: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(t) - block - 1, (bs,))
    x = torch.stack([t[i: i + block] for i in ix]).long()
    y = torch.stack([t[i + 1: i + 1 + block] for i in ix]).long()
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tensors, args, device) -> tuple[dict, float]:
    model.eval()
    losses, labels = {}, []
    correct = total = 0
    for sid, name in enumerate(D.SKILL_NAMES):
        t = tensors[name]["val"]
        tot = 0.0
        for _ in range(args.eval_batches):
            x, y = get_batch(t, args.block, min(args.batch, 16), device)
            out = model(x, targets=y, force_route=False)
            tot += out["lm_loss"].item()
            pred = out["skill_logits"].argmax(-1)
            correct += (pred == sid).sum().item()
            total += x.size(0)
        losses[name] = tot / args.eval_batches
        labels.append(sid)
    model.train()
    return losses, correct / max(total, 1)


PROMPTS = {
    "math": "Compute: 23 + 45\n",
    "qa": "Q: What is the capital of France?\n",
    "count": "How many letters are in the word 'apple'?\n",
    "story": "One day, a little girl named",
    "optometry": "Eye Q: What is blurry distance vision called?\n",
}


@torch.no_grad()
def write_samples(model, tok, tensors, device, path, block) -> None:
    model.eval()
    lines = [f"--- samples @ {time.strftime('%H:%M:%S')} ---"]
    for name in D.SKILL_NAMES:
        ids = torch.tensor([tok.encode(PROMPTS[name])], dtype=torch.long, device=device)
        out, skill = model.generate(ids, max_new_tokens=48, temperature=0.7, top_k=20)
        text = tok.decode(out[0].tolist())
        lines.append(f"[router picked: {D.SKILL_NAMES[skill]} | prompt skill: {name}]")
        lines.append(text)
        lines.append("")
    model.train()
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    p.add_argument("--out", default=os.path.join(REPO, "ckpt"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--story-mb", type=float, default=30.0)
    p.add_argument("--n-math", type=int, default=40_000)
    p.add_argument("--n-count", type=int, default=30_000)
    p.add_argument("--n-qa", type=int, default=20_000)
    p.add_argument("--n-optometry", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=6)
    p.add_argument("--max-minutes", type=float, default=55.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cpu"
    t0 = time.time()

    tok, tensors = build_all(args, log)

    cfg = SkillModularConfig(vocab_size=tok.vocab_size, dim=args.dim if hasattr(args, "dim") else 128,
                             block_size=args.block)
    model = SkillModularLM(cfg)
    log(f"[mdl ] total params: {model.total_params():,} | active/request: {model.active_params():,}")
    log(f"[mdl ] checkpoint size ~{model.total_params() * 4 / 1e6:.1f} MB (fp32)")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "skill", "lm_loss", "router_loss", "router_acc", "tok_per_s"])

    best_val = float("inf")
    deadline = t0 + args.max_minutes * 60
    model.train()
    cache: dict = {}

    log(f"[trn ] training {args.steps} steps (budget {args.max_minutes:.0f} min)...")
    for step in range(args.steps):
        if time.time() > deadline:
            log(f"[trn ] time budget reached at step {step}; stopping gracefully")
            break
        sid = step % len(D.SKILL_NAMES)
        name = D.SKILL_NAMES[sid]
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(tensors[name]["train"], args.block, args.batch, device)
        label = torch.full((x.size(0),), sid, dtype=torch.long)
        out = model(x, targets=y, skill_label=label, force_route=True)
        loss = out["lm_loss"] + out["router_loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == args.steps - 1:
            tps = args.batch * args.block / max(1e-9, time.time() - t0) * (step + 1)
            log(f"step {step:>5} | skill {name:<6} | lm {out['lm_loss'].item():.3f} "
                f"| router_ce {out['router_loss'].item():.3f} | lr {lr:.2e} | ~{tps:,.0f} tok/s")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, name, round(out['lm_loss'].item(), 4),
                                        round(out['router_loss'].item(), 4), "", int(tps)])

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            losses, racc = evaluate(model, tensors, args, device)
            vmean = sum(losses.values()) / len(losses)
            log(f"EVAL  {step:>5} | val loss " +
                " ".join(f"{k}={v:.3f}" for k, v in losses.items()) +
                f" | mean {vmean:.3f} | router acc {racc * 100:.1f}%")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, "EVAL", round(vmean, 4), "", round(racc, 4), ""])
            if vmean < best_val:
                best_val = vmean
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                            "step": step, "val_mean": vmean},
                           os.path.join(args.out, "model.pt"))
            write_samples(model, tok, tensors, device, os.path.join(args.out, "samples.txt"), args.block)

    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "step": args.steps, "val_mean": best_val},
               os.path.join(args.out, "model.pt"))
    log(f"[done] best val mean {best_val:.3f} | checkpoint -> {args.out}/model.pt "
        f"({os.path.getsize(os.path.join(args.out, 'model.pt')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
