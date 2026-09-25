"""Chat with the Skill-Modular Language Model.

    python -m skill_lm.generate --demo
    python -m skill_lm.generate --prompt "Compute: 12 + 34"
    python -m skill_lm.generate --prompt "Q: What is the capital of Japan?" --skill qa
"""
from __future__ import annotations

import argparse
import os

import torch

from . import data as D
from .model import SkillModularConfig, SkillModularLM
from .tokenizer import BPETokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMO_PROMPTS = [
    ("math", "Compute: 23 + 45\n"),
    ("qa", "Q: What is the capital of France?\n"),
    ("count", "How many letters are in the word 'apple'?\n"),
    ("story", "One day, a little girl named"),
]


def load_model(ckpt_path: str):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SkillModularConfig.from_dict(blob["config"])
    model = SkillModularLM(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(os.path.dirname(ckpt_path), "tokenizer.json"))
    return model, tok, blob


def run_one(model, tok, prompt, skill, tokens, temperature, top_k, greedy):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    sk = D.SKILL_IDS.get(skill, "auto")
    with torch.no_grad():
        h = model._trunk(ids)
        probs = torch.softmax(model.router(h.mean(dim=1)), dim=-1)[0]
        auto_pick = int(probs.argmax())          # what the ROUTER would load
        out, chosen = model.generate(
            ids, max_new_tokens=tokens, temperature=temperature, top_k=top_k,
            skill=sk if isinstance(sk, int) else "auto", greedy=greedy,
        )
    text = tok.decode(out[0].tolist())
    probs_s = ", ".join(f"{D.SKILL_NAMES[i]}={probs[i]:.2f}" for i in probs.argsort(descending=True))
    return text, D.SKILL_NAMES[auto_pick], probs_s


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Skill-Modular LM generator")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--skill", type=str, default="auto",
                   choices=["auto"] + D.SKILL_NAMES)
    p.add_argument("--ckpt", type=str, default=os.path.join(REPO, "ckpt", "model.pt"))
    p.add_argument("--tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--demo", action="store_true", help="run one prompt per skill")
    args = p.parse_args(argv)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no checkpoint at {args.ckpt} - train first: python -m skill_lm.train")
    model, tok, blob = load_model(args.ckpt)
    step = blob.get("step", "?")
    print(f"Skill-Modular LM | {model.total_params():,} params "
          f"({model.active_params():,} active/request) | ckpt step {step}")

    prompts = DEMO_PROMPTS if args.demo else [(args.skill, args.prompt or "Hello")]
    for intended, prompt in prompts:
        text, chosen, probs = run_one(
            model, tok, prompt, intended, args.tokens, args.temperature, args.top_k, args.greedy
        )
        marker = "" if intended == "auto" else (
            " [router correct]" if intended == chosen else f" [router MISSED - wanted {intended}]"
        )
        print("\n" + "=" * 62)
        print(f"PROMPT : {prompt.strip()!r}")
        print(f"ROUTER : {chosen}{marker}   (probs: {probs})")
        print(f"OUTPUT :\n{text}")


if __name__ == "__main__":
    main()
