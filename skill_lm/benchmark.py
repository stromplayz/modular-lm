"""Benchmark the trained SMT per skill: exact-match accuracy on fresh examples.

Usage: python -m skill_lm.benchmark --ckpt ckpt_v1
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys

import torch

from . import data as D
from .generate import load_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANS = re.compile(r"Answer:\s*([^\n]+)")


def ask(model, tok, prompt: str, skill_id: int, max_new: int = 12) -> str:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        out, _ = model.generate(ids, max_new_tokens=max_new, skill=skill_id, greedy=True)
    text = tok.decode(out[0].tolist())[len(prompt):]
    m = ANS.search(text)
    got = (m.group(1).strip().rstrip(".") if m else
           text.strip().splitlines()[0] if text.strip() else "")
    # models may answer with the "A:" / "Answer:" label still attached
    got = re.sub(r"^(A:|Answer:)\s*", "", got).rstrip(".")
    return got


def bench_math(model, tok, rng, n=150) -> tuple[int, int, list]:
    ok, cases, fails = 0, [], []
    for _ in range(n):
        op = rng.choice(["+", "-", "x"])
        if op == "x":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
        else:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            if op == "-" and b > a:
                a, b = b, a
        ans = a + b if op == "+" else a - b if op == "-" else a * b
        prompt = f"What is {a} {op} {b}?\n"
        got = ask(model, tok, prompt, D.SKILL_IDS["math"])
        good = got == str(ans)
        ok += good
        if not good and len(fails) < 8:
            fails.append(f"  {prompt.strip()} -> {got!r} (want {ans})")
        cases.append(good)
    return ok, n, fails


def bench_qa(model, tok) -> tuple[int, int, list]:
    facts = D.load_facts(os.path.join(REPO, "assets", "facts.txt"))
    ok, fails = 0, []
    for q, a in facts:
        prompt = f"Q: {q}\n"
        got = ask(model, tok, prompt, D.SKILL_IDS["qa"])
        good = got.lower() == a.lower()
        ok += good
        if not good and len(fails) < 8:
            fails.append(f"  {q} -> {got!r} (want {a!r})")
    return ok, len(facts), fails


def bench_count(model, tok) -> tuple[int, int, list]:
    ok, fails = 0, []
    words = D._WORDLIST
    for w in words:
        prompt = f"How many letters are in the word '{w}'?\n"
        got = ask(model, tok, prompt, D.SKILL_IDS["count"])
        good = got == str(len(w))
        ok += good
        if not good and len(fails) < 8:
            fails.append(f"  {w} ({len(w)}) -> {got!r}")
    return ok, len(words), fails


def bench_router(model, tok, rng, n=80) -> tuple[int, int]:
    """Can the router tell which skill a prompt needs?"""
    ok = 0
    prompts = []
    for _ in range(n // 4):
        op = rng.choice(["+", "-", "x"])
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        prompts.append(("math", f"What is {a} {op} {b}?\n"))
        prompts.append(("count", f"How many letters are in the word '{rng.choice(D._WORDLIST)}'?\n"))
        q, _ = rng.choice(D.load_facts(os.path.join(REPO, "assets", "facts.txt")))
        prompts.append(("qa", f"Q: {q}\n"))
        prompts.append(("story", "One day, a little girl named"))
    for want, p in prompts:
        ids = torch.tensor([tok.encode(p)], dtype=torch.long)
        with torch.no_grad():
            h = model._trunk(ids)
            pick = int(model.router(h.mean(dim=1)).argmax(-1)[0])
        ok += pick == D.SKILL_IDS[want]
    return ok, len(prompts)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(REPO, "ckpt_v1"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    model, tok, blob = load_model(os.path.join(args.ckpt, "model.pt"))
    rng = random.Random(args.seed)
    step = blob.get("step", "?")
    print(f"SMT benchmark | ckpt step {step} | {model.total_params():,} params\n")

    m_ok, m_n, m_fails = bench_math(model, tok, rng)
    print(f"math  : {m_ok:>4}/{m_n} = {100 * m_ok / m_n:5.1f}% exact-match")
    print("\n".join(m_fails))
    q_ok, q_n, q_fails = bench_qa(model, tok)
    print(f"qa    : {q_ok:>4}/{q_n} = {100 * q_ok / q_n:5.1f}% exact-match")
    print("\n".join(q_fails))
    c_ok, c_n, c_fails = bench_count(model, tok)
    print(f"count : {c_ok:>4}/{c_n} = {100 * c_ok / c_n:5.1f}% exact-match")
    print("\n".join(c_fails))
    r_ok, r_n = bench_router(model, tok, rng)
    print(f"router: {r_ok:>4}/{r_n} = {100 * r_ok / r_n:5.1f}% skill picked correctly")


if __name__ == "__main__":
    main()
