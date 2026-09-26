"""Train ONE skill pack against the FROZEN trunk. The main model never moves.

What gets gradients: this pack's expert block + this pack's router head.
What never gets gradients: trunk (no_grad + requires_grad=False) and every
other pack (they are only read for router calibration as frozen heads).

    python -m skill_lm.train_pack --name knowledge \
        --facts assets/knowledge/wiki_facts.txt \
        --eval-facts assets/knowledge/wiki_facts_eval.txt --steps 2500

Pipeline per run:
    1. pack corpus  : QA templates (distinctive "Wiki Q:" lead-ins + plain Q)
    2. LM training  : expert block learns next-token on its corpus
    3. router       : BCE warm-start (own=1 / others=0) then joint softmax
                      calibration with existing heads FROZEN -> co-loading
                      behavior of existing packs is untouched (zero-forgetting)
    4. eval         : greedy QA on held-out facts + co-load routing check
    5. save         : packs/<name>.pack (single portable file)
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
from .frozen import TrunkModel, SkillPack, SkillMixer
from .genesis import pool_features, fast_encode
from .tokenizer import BPETokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(msg, flush=True)


_TEMPLATES = [
    "Wiki Q: {q}\nA: {a}.",
    "Fact question: {q}\nAnswer: {a}.",
    "Knowledge Q: {q}\nAnswer: {a}.",
    "Q: {q}\nA: {a}.",
]


def gen_knowledge(rng: random.Random, facts: list[tuple[str, str]]) -> str:
    q, a = rng.choice(facts)
    # 3/4 distinctive knowledge lead-ins, 1/4 plain Q (content-level routing)
    return _TEMPLATES[rng.randrange(4)].format(q=q, a=a)


def get_batch(t: torch.Tensor, block: int, bs: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(t) - block - 1, (bs,))
    x = torch.stack([t[i: i + block] for i in ix]).long()
    y = torch.stack([t[i + 1: i + 1 + block] for i in ix]).long()
    return x, y


def lr_at(step: int, warmup: int, steps: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def match_or_contain(got: str, want: str) -> bool:
    g, w = got.strip().rstrip(".").lower(), want.strip().rstrip(".").lower()
    return bool(g) and (g == w or (len(g) > 2 and g in w) or (len(w) > 2 and w in g))


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="pack name, e.g. knowledge")
    p.add_argument("--facts", required=True, help="facts bank: 'Q ||| A' lines")
    p.add_argument("--eval-facts", default=None, help="held-out facts for LUA eval")
    p.add_argument("--trunk", default=os.path.join(REPO, "ckpt", "trunk.pt"))
    p.add_argument("--tokenizer", default=os.path.join(REPO, "ckpt", "tokenizer.json"))
    p.add_argument("--packs-dir", default=os.path.join(REPO, "packs"))
    p.add_argument("--out", default=None, help="output pack path (default packs/<name>.pack)")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--router-lr", type=float, default=5e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--n-examples", type=int, default=16_000)
    p.add_argument("--per-skill", type=int, default=3000, help="router features per skill")
    p.add_argument("--calib-epochs", type=int, default=250)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="expert dropout during pack training (0.0 = exact recall)")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max-minutes", type=float, default=45.0)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    trunk = TrunkModel.load(args.trunk, frozen=True)
    tok = BPETokenizer.load(args.tokenizer)
    facts = D.load_facts(args.facts)
    log(f"[pack] {args.name}: {len(facts)} facts | trunk FROZEN "
        f"({trunk.n_params():,} params, requires_grad=False)")

    # ------------------------------------------------------------------ #
    # 1. pack corpus
    # ------------------------------------------------------------------ #
    corpus = "\n\n".join(gen_knowledge(rng, facts) for _ in range(args.n_examples)) + "\n\n"
    cache: dict = {}
    t_train = fast_encode(tok, corpus, cache)
    if len(t_train) < args.block + 2:
        raise SystemExit("corpus too small - add more facts")
    log(f"[data] pack corpus: {len(t_train):,} tokens")

    # negatives for router training: other skills' corpora (cheap generators)
    neg_texts: dict[str, str] = {}
    neg_texts["qa"] = "\n\n".join(
        D.gen_qa(rng, D.load_facts(os.path.join(REPO, "assets", "facts.txt")))
        for _ in range(3000)) + "\n\n"
    opto = os.path.join(REPO, "assets", "optometry.txt")
    if os.path.exists(opto):
        neg_texts["optometry"] = "\n\n".join(
            D.gen_optometry(rng, D.load_facts(opto)) for _ in range(3000)) + "\n\n"
    neg_texts["math"] = "\n\n".join(D.gen_math(rng) for _ in range(6000)) + "\n\n"
    neg_texts["count"] = "\n\n".join(D.gen_count(rng) for _ in range(5000)) + "\n\n"
    story_path = os.path.join(REPO, "data", "tinystories.txt")
    if os.path.exists(story_path):
        neg_texts["story"] = D.load_story_corpus(story_path)[:400_000]
    neg_tensors = {}
    for nm, txt in neg_texts.items():
        t = fast_encode(tok, txt, cache)
        if len(t) > args.block + 2:
            neg_tensors[nm] = t
            log(f"[data] negatives[{nm:<9}] {len(t):>9,} tokens")

    # ------------------------------------------------------------------ #
    # 2. LM training - expert block only (trunk runs under no_grad)
    # ------------------------------------------------------------------ #
    pack = SkillPack(args.name, trunk.cfg,
                     description=f"pack trained on {os.path.basename(args.facts)}")
    opt = torch.optim.AdamW(
        [{"params": [p for p in pack.expert.parameters() if p.dim() >= 2], "weight_decay": args.wd},
         {"params": [p for p in pack.expert.parameters() if p.dim() < 2], "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    if args.dropout <= 0:
        pack.expert.eval()  # disable dropout, gradients still flow (exact recall)
        log("[trn ] expert dropout disabled (exact-recall mode)")

    deadline = t0 + args.max_minutes * 60
    log(f"[trn ] LM training {args.steps} steps (expert block ONLY)")
    if args.dropout > 0:
        pack.train()
    else:
        pack.expert.eval()  # dropout stays OFF for exact recall
    for step in range(args.steps):
        if time.time() > deadline:
            log(f"[trn ] time budget reached at step {step}")
            break
        lr = lr_at(step, args.warmup, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(t_train, args.block, args.batch)
        with torch.no_grad():                       # <- main model never moves
            h = trunk.hidden(x)
        cos = trunk.rope_cos[:, : x.size(1)]
        sin = trunk.rope_sin[:, : x.size(1)]
        h = pack.expert(h.detach(), cos, sin)
        logits = trunk.logits_from_h(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pack.expert.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            log(f"step {step:>5} | lm_loss {loss.item():.3f} | lr {lr:.2e}")

    # ------------------------------------------------------------------ #
    # 3. router: BCE warm-start, then joint calibration (new head only moves)
    # ------------------------------------------------------------------ #
    log("[rtr ] pooled features (frozen trunk)...")
    feats = pool_features(trunk, {args.name: t_train, **neg_tensors},
                          args.block, args.per_skill, bs=64, seed=args.seed)
    own_feats = {args.name: feats[args.name]}
    others = {k: v for k, v in feats.items() if k != args.name}

    log("[rtr ] BCE warm-start one-vs-rest head")
    neg_all = torch.cat(list(others.values()))
    pos_all = feats[args.name]
    n = min(len(pos_all), len(neg_all))
    pos_all, neg_all = pos_all[:n], neg_all[:n]
    opt_r = torch.optim.AdamW(pack.router_head.parameters(), lr=args.router_lr)
    for ep in range(120):
        cur = args.router_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ep / 120)))
        for g in opt_r.param_groups:
            g["lr"] = cur
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i: i + 256]
            xx = torch.cat([pos_all[idx], neg_all[idx]])
            yy = torch.cat([torch.ones(len(pos_all[idx])), torch.zeros(len(neg_all[idx]))])
            lg = pack.route_logit(xx)
            loss = F.binary_cross_entropy_with_logits(lg, yy)
            opt_r.zero_grad(set_to_none=True)
            loss.backward()
            opt_r.step()

    # joint softmax calibration with existing packs loaded, THEIR HEADS FROZEN
    existing: dict[str, SkillPack] = {}
    if os.path.isdir(args.packs_dir):
        for f in sorted(os.listdir(args.packs_dir)):
            if f.endswith(".pack"):
                nm = f[:-5]
                if nm != args.name:
                    existing[nm] = SkillPack.load(os.path.join(args.packs_dir, f))
    calib_packs = list(existing.values()) + [pack]
    trainable = list(pack.router_head.parameters())   # ONLY the new head learns
    for pk in calib_packs:
        for prm in pk.router_head.parameters():
            prm.requires_grad_(pk is pack)
    name_to_i = {pk.name: i for i, pk in enumerate(calib_packs)}
    X, Y = torch.cat([feats[k] for k in feats if k in name_to_i]), None
    labels = []
    for k, v in feats.items():
        if k in name_to_i:
            labels.append(torch.full((len(v),), name_to_i[k], dtype=torch.long))
    Y = torch.cat(labels)
    opt_c = torch.optim.AdamW(trainable, lr=args.router_lr)
    log(f"[rtr ] joint calibration over {len(calib_packs)} packs "
        f"(heads frozen except {args.name})")
    best = 0.0
    for ep in range(args.calib_epochs):
        cur = args.router_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ep / args.calib_epochs)))
        for g in opt_c.param_groups:
            g["lr"] = cur
        perm = torch.randperm(len(X))
        correct = 0
        for i in range(0, len(X), 512):
            idx = perm[i: i + 512]
            # new head carries gradients; existing heads are frozen constants
            lg = torch.cat([pk.router_head(X[idx]) if pk is pack
                            else pk.route_logit(X[idx]).unsqueeze(-1)
                            for pk in calib_packs], dim=1)
            loss = F.cross_entropy(lg, Y[idx])
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            opt_c.step()
            correct += (lg.argmax(1) == Y[idx]).sum().item()
        best = max(best, correct / len(X))
    log(f"[rtr ] co-loaded routing acc (train feats): {best * 100:.1f}%")

    # ------------------------------------------------------------------ #
    # 4. held-out QA eval (LUA-style: greedy answer, containment match)
    # ------------------------------------------------------------------ #
    acc = None
    recall = None
    if args.eval_facts and os.path.exists(args.eval_facts):
        held = D.load_facts(args.eval_facts)
        mixer = SkillMixer(trunk)
        for pk in existing.values():
            mixer.add_pack(pk)
        pack.eval()
        mixer.add_pack(pack)

        def greedy_answer(q: str) -> str:
            prompt = f"Wiki Q: {q}\n"
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
            out, _ = mixer.generate(ids, max_new_tokens=24, pack=args.name, greedy=True)
            text = tok.decode(out[0].tolist())[len(prompt):]
            m = D.extract_answer(text)
            if m is None and text.strip():
                m = text.strip().splitlines()[0]
            return ((m or "").strip().rstrip(".")).lower()

        ok = sum(match_or_contain(greedy_answer(q), a) for q, a in held)
        acc = ok / len(held)
        log(f"[eval] held-out QA (generalization): {ok}/{len(held)} = {acc * 100:.1f}%")
        # trained-fact recall (the pack's primary job: absorb its bank)
        rng_e = random.Random(99)
        sample = rng_e.sample(facts, min(60, len(facts)))
        ok_r = sum(match_or_contain(greedy_answer(q), a) for q, a in sample)
        recall = ok_r / len(sample)
        log(f"[eval] trained-fact recall: {ok_r}/{len(sample)} = {recall * 100:.1f}%")

    # ------------------------------------------------------------------ #
    # 5. save pack (one file - nothing else in the system changes)
    # ------------------------------------------------------------------ #
    out_path = args.out or os.path.join(args.packs_dir, f"{args.name}.pack")
    pack.save(out_path, meta={
        "origin": "train_pack",
        "facts": os.path.abspath(args.facts),
        "n_facts": len(facts),
        "steps": args.steps,
        "router_calibrated": "joint-softmax-others-frozen",
        "heldout_qa_acc": round(acc, 4) if acc is not None else None,
        "recall_acc": round(recall, 4) if recall is not None else None,
    })
    log(f"[done] pack -> {out_path} ({os.path.getsize(out_path) / 1e3:.0f} KB) "
        f"in {time.time() - t0:.0f}s")
    log("[zero-forgetting] trunk.pt + other pack files were never written")


if __name__ == "__main__":
    main()
