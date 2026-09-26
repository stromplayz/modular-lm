"""FrozenCore runtime: tiny main model + packs fetched on demand + deload.

    python -m skill_lm.serve --ask "What is the largest ocean?"
    python -m skill_lm.serve --demo
    python -m skill_lm.serve --packs qa knowledge --ask "..."
    python -m skill_lm.serve --repl

Resolution order per pack: repo packs/ dir -> hub cache -> GitHub hub fetch.
Deloading is explicit (SkillMixer.drop_pack) - packs only occupy RAM while
loaded; the hub cache keeps bytes on disk.
"""
from __future__ import annotations

import argparse
import os

import torch

from .hub import Hub, DEFAULT_REPO
from .tokenizer import BPETokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUNK = os.path.join(REPO, "ckpt", "trunk.pt")
TOK = os.path.join(REPO, "ckpt", "tokenizer.json")
PACKS_DIR = os.path.join(REPO, "packs")

DEMO = [
    ("story", "One day, a little girl named"),
    ("qa", "Q: What is the capital of France?\n"),
    ("math", "What is 23 + 45?\n"),
    ("count", "How many letters are in the word 'apple'?\n"),
    ("optometry", "Eye Q: What does OD mean on a prescription?\n"),
    ("knowledge", "Wiki Q: What is the largest ocean on Earth?\n"),
]


def build_runtime(load_packs: list[str] | None, repo: str, autoload: bool,
                  trunk_path: str = TRUNK, tok_path: str = TOK):
    """Trunk + requested packs. autoload=True co-loads everything available."""
    hub = Hub(repo)
    if load_packs is None:
        load_packs = (hub.list_cached() if autoload else [])
        if os.path.isdir(PACKS_DIR):
            local = sorted(f[:-5] for f in os.listdir(PACKS_DIR) if f.endswith(".pack"))
            for nm in local:
                if nm not in load_packs:
                    load_packs.insert(0, nm)
    mixer = hub.load_mixer(trunk_path, load_packs, local_dirs=[PACKS_DIR])
    tok = BPETokenizer.load(tok_path)
    return mixer, tok, hub


def ask(mixer, tok, question: str, tokens: int = 14, show_scores: bool = True):
    """Auto-route among loaded packs; return (answer, pack_used, scores)."""
    prompt = question if question.endswith("\n") else question + "\n"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        h = mixer.trunk.hidden(ids)
        aidx, scores = mixer.route(h.detach())
    pick = mixer.name_of(int(aidx[0]))
    out, used = mixer.generate(ids, max_new_tokens=tokens, pack=pick, greedy=True)
    text = tok.decode(out[0].tolist())[len(prompt):]
    from . import data as D
    ans = D.extract_answer(text)
    if ans is None and text.strip():
        ans = text.strip().splitlines()[0]
    ans = (ans or "").strip().rstrip(".")
    probs = ""
    if show_scores and scores is not None:
        s = torch.softmax(scores[0], dim=0)
        probs = ", ".join(f"{nm}={s[i]:.2f}" for i, nm in enumerate(mixer.pack_names)
                          if s[i] > 0.01)
    return ans, used, probs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="FrozenCore runtime")
    p.add_argument("--packs", nargs="*", default=None,
                   help="pack names to co-load (default: everything available)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="hub repo for on-demand fetch")
    p.add_argument("--ask", type=str, default=None)
    p.add_argument("--tokens", type=int, default=14)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--repl", action="store_true")
    p.add_argument("--trunk", default=TRUNK)
    p.add_argument("--tokenizer", default=TOK)
    p.add_argument("--no-autoload", action="store_true",
                   help="only load --packs list, do not scan repo/cache")
    args = p.parse_args(argv)

    mixer, tok, hub = build_runtime(args.packs, args.repo, not args.no_autoload,
                                    args.trunk, args.tokenizer)
    tpar = mixer.trunk.n_params()
    ppar = sum(pk.n_params() for pk in mixer.packs.values())
    print(f"FrozenCore runtime | trunk {tpar:,} (FROZEN) + {len(mixer.packs)} packs "
          f"{par_list(mixer)} = {tpar + ppar:,} params loaded")
    print(f"packs loaded: {', '.join(mixer.pack_names) or '-'} | hub: {hub.repo}")

    if args.demo:
        for _want, prompt in DEMO:
            if _want not in mixer.packs:
                continue
            ans, used, probs = ask(mixer, tok, prompt, tokens=args.tokens)
            print("\n" + "=" * 60)
            print(f"Q      : {prompt.strip()!r}")
            print(f"pack   : {used}   ({probs})")
            print(f"answer : {ans}")
        return

    if args.ask:
        ans, used, probs = ask(mixer, tok, args.ask, tokens=args.tokens)
        print(f"[{used}] {ans}" + (f"   ({probs})" if probs else ""))
        return

    if args.repl:
        print("repl: type a question; 'packs' lists loaded; ':deload name' drops; ':load name' fetches; 'quit' exits")
        while True:
            try:
                line = input("\nask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line == "quit":
                break
            if line == "packs":
                print(f"loaded: {', '.join(mixer.pack_names) or '-'} | "
                      f"remote: {', '.join(hub.list_remote())}")
                continue
            if line.startswith(":deload "):
                nm = line.split()[1]
                mixer.drop_pack(nm)
                print(f"deloaded {nm} -> loaded: {', '.join(mixer.pack_names) or '-'}")
                continue
            if line.startswith(":load "):
                nm = line.split()[1]
                path = hub.fetch(nm)
                mixer.add_pack_file(path)
                print(f"loaded {nm} from {path}")
                continue
            ans, used, probs = ask(mixer, tok, line, tokens=args.tokens)
            print(f"[{used}] {ans}" + (f"   ({probs})" if probs else ""))


def par_list(mixer) -> str:
    return "(" + " ".join(f"{nm}:{pk.n_params() // 1000}k" for nm, pk in mixer.packs.items()) + ")"


if __name__ == "__main__":
    main()
