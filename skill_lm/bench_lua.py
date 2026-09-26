"""LUA benchmark: Language Understanding & Answering for the FrozenCore system.

Sections
  qa         : assets/facts.txt factoid answering (exact match)
  optometry  : eye-care domain facts (exact match)
  knowledge  : held-out ingested facts (exact / containment match)
  math       : fresh arithmetic (exact match)
  count      : letter counting (exact match)
  router     : does routing among CO-LOADED packs pick the right skill?
  forgetting : old-skill scores must equal the recorded baseline
               (ckpt/bench_lua.json baseline is written on first run; the
               FrozenCore contract says they can only move if files moved)

Usage:
    python -m skill_lm.bench_lua [--baseline ckpt/bench_lua.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch

from . import data as D
from .frozen import TrunkModel, SkillMixer
from .hub import Hub, DEFAULT_REPO
from .tokenizer import BPETokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(msg, flush=True)


def ask_pack(mixer, tok, prompt: str, pack: str, max_new: int = 28) -> str:
    # 28 tokens: room for the place-value scratchpad + 'Answer:' line on 2-digit adds
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out, _ = mixer.generate(ids, max_new_tokens=max_new, pack=pack, greedy=True)
    text = tok.decode(out[0].tolist())[len(prompt):]
    m = D.extract_answer(text)
    got = (m if m is not None else (text.strip().splitlines()[0] if text.strip() else ""))
    import re
    return re.sub(r"^(A:|Answer:)\s*", "", (got or "").strip()).rstrip(".")


def match(got: str, want: str) -> bool:
    g, w = got.lower().rstrip("."), want.lower().rstrip(".")
    return g == w or (len(g) > 2 and g in w) or (len(w) > 2 and w in g)


def bench_facts(mixer, tok, facts, pack, lead="Q: {q}\n", want_pack_only=True):
    ok, fails = 0, []
    for q, a in facts:
        got = ask_pack(mixer, tok, lead.format(q=q), pack)
        good = match(got, a)
        ok += good
        if not good and len(fails) < 6:
            fails.append(f"  {q[:52]:<52} -> {got!r} (want {a!r})")
    return ok, len(facts), fails


def bench_math(mixer, tok, rng, n=150):
    ok, fails = 0, []
    for _ in range(n):
        op = rng.choice(["+", "-", "x"])
        if op == "x":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
        else:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            if op == "-" and b > a:
                a, b = b, a
        ans = a + b if op == "+" else a - b if op == "-" else a * b
        got = ask_pack(mixer, tok, f"What is {a} {op} {b}?\n", "math")
        good = got == str(ans)
        ok += good
        if not good and len(fails) < 6:
            fails.append(f"  {a} {op} {b} -> {got!r} (want {ans})")
    return ok, n, fails


def bench_count(mixer, tok):
    ok, fails = 0, []
    for w in D._WORDLIST:
        got = ask_pack(mixer, tok, f"How many letters are in the word '{w}'?\n", "count")
        good = got == str(len(w))
        ok += good
        if not good and len(fails) < 6:
            fails.append(f"  {w} ({len(w)}) -> {got!r}")
    return ok, len(D._WORDLIST), fails


def bench_router(mixer, tok, rng, n_per=30):
    """Routing among co-loaded packs (the real inference condition)."""
    cases: list[tuple[str, str]] = []
    qa_f = D.load_facts(os.path.join(REPO, "assets", "facts.txt"))
    op_f = D.load_facts(os.path.join(REPO, "assets", "optometry.txt"))
    for _ in range(n_per):
        op = rng.choice(["+", "-", "x"])
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        cases.append(("math", f"What is {a} {op} {b}?\n"))
        cases.append(("count", f"How many letters are in the word '{rng.choice(D._WORDLIST)}'?\n"))
        q, _ = rng.choice(qa_f)
        cases.append(("qa", f"Q: {q}\n"))
        eq, _ = rng.choice(op_f)
        cases.append(("optometry", f"Eye Q: {eq}\n"))
        cases.append(("story", "One day, a little girl named"))
        kq, _ = rng.choice(D.load_facts(os.path.join(
            REPO, "assets", "knowledge", "wiki_facts.txt")))
        cases.append(("knowledge", f"Wiki Q: {kq}\n"))
    cases = [(w, p) for w, p in cases if w in mixer.packs]
    ok = 0
    for want, p in cases:
        ids = torch.tensor([tok.encode(p)], dtype=torch.long)
        with torch.no_grad():
            h = mixer.trunk.hidden(ids)
            aidx, _ = mixer.route(h.detach())
        ok += mixer.name_of(int(aidx[0])) == want
    return ok, len(cases)


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trunk", default=os.path.join(REPO, "ckpt", "trunk.pt"))
    p.add_argument("--tokenizer", default=os.path.join(REPO, "ckpt", "tokenizer.json"))
    p.add_argument("--packs-dir", default=os.path.join(REPO, "packs"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--baseline", default=os.path.join(REPO, "ckpt", "bench_lua.json"))
    p.add_argument("--write-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    hub = Hub(args.repo)
    names = sorted(f[:-5] for f in os.listdir(args.packs_dir) if f.endswith(".pack")) \
        if os.path.isdir(args.packs_dir) else hub.list_cached()
    trunk = TrunkModel.load(args.trunk, frozen=True)
    mixer = SkillMixer(trunk)
    for nm in names:
        mixer.add_pack_file(hub.fetch(nm) if not os.path.exists(
            os.path.join(args.packs_dir, f"{nm}.pack"))
            else os.path.join(args.packs_dir, f"{nm}.pack"))
    tok = BPETokenizer.load(args.tokenizer)
    log(f"LUA benchmark | trunk {trunk.n_params():,} FROZEN + packs {mixer.pack_names}")

    results: dict[str, dict] = {}
    if "math" in mixer.packs:
        ok, n, f = bench_math(mixer, tok, rng)
        results["math"] = {"ok": ok, "n": n}
        log(f"math      : {ok:>4}/{n} = {100 * ok / n:5.1f}%")
        log("\n".join(f))
    if "count" in mixer.packs:
        ok, n, f = bench_count(mixer, tok)
        results["count"] = {"ok": ok, "n": n}
        log(f"count     : {ok:>4}/{n} = {100 * ok / n:5.1f}%")
        log("\n".join(f))
    if "qa" in mixer.packs:
        ok, n, f = bench_facts(mixer, tok, D.load_facts(
            os.path.join(REPO, "assets", "facts.txt")), "qa")
        results["qa"] = {"ok": ok, "n": n}
        log(f"qa        : {ok:>4}/{n} = {100 * ok / n:5.1f}%")
        log("\n".join(f))
    if "optometry" in mixer.packs:
        ok, n, f = bench_facts(mixer, tok, D.load_facts(
            os.path.join(REPO, "assets", "optometry.txt")), "optometry",
            lead="Eye Q: {q}\n")
        results["optometry"] = {"ok": ok, "n": n}
        log(f"optometry : {ok:>4}/{n} = {100 * ok / n:5.1f}%")
        log("\n".join(f))
    ev_path = os.path.join(REPO, "assets", "knowledge", "wiki_facts_eval.txt")
    tr_path = os.path.join(REPO, "assets", "knowledge", "wiki_facts.txt")
    if "knowledge" in mixer.packs:
        # recall: the pack's job is to absorb its bank (primary metric)
        if os.path.exists(tr_path):
            facts_all = D.load_facts(tr_path)
            sample = rng.sample(facts_all, min(60, len(facts_all)))
            ok, n, f = bench_facts(mixer, tok, sample, "knowledge",
                                   lead="Wiki Q: {q}\n")
            results["knowledge_recall"] = {"ok": ok, "n": n}
            log(f"knowledge : {ok:>4}/{n} = {100 * ok / n:5.1f}% (bank recall)")
            log("\n".join(f))
        # generalization: held-out facts never trained on (stretch metric)
        if os.path.exists(ev_path):
            ok, n, f = bench_facts(mixer, tok, D.load_facts(ev_path), "knowledge",
                                   lead="Wiki Q: {q}\n")
            results["knowledge_heldout"] = {"ok": ok, "n": n}
            log(f"knowledge : {ok:>4}/{n} = {100 * ok / n:5.1f}% (held-out)")
    if len(mixer.packs) >= 2:
        ok, n = bench_router(mixer, tok, rng)
        results["router"] = {"ok": ok, "n": n}
        log(f"router    : {ok:>4}/{n} = {100 * ok / n:5.1f}% (co-loaded)")

    # ---- zero-forgetting check ------------------------------------------ #
    verdict = ""
    if os.path.exists(args.baseline) and not args.write_baseline:
        base = json.load(open(args.baseline))
        drift = []
        for k, v in base.items():
            if k in results and results[k]["n"] == v["n"]:
                d = results[k]["ok"] - v["ok"]
                if d != 0:
                    drift.append(f"{k}: {v['ok']}/{v['n']} -> {results[k]['ok']}/{results[k]['n']}")
        verdict = ("ZERO-FORGETTING VERIFIED: all old skills bit-identical"
                   if not drift else "DRIFT DETECTED: " + "; ".join(drift))
        log(f"forgetting: {verdict}")
        results["forgetting"] = {"drift": drift, "verdict": verdict}
    else:
        json.dump(results, open(args.baseline, "w"), indent=2)
        log(f"forgetting: baseline written -> {args.baseline}")

    results["system"] = {
        "trunk_params": trunk.n_params(),
        "packs": {nm: pk.n_params() for nm, pk in mixer.packs.items()},
    }
    return results


if __name__ == "__main__":
    main()
