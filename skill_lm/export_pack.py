"""Export packs (and optionally the trunk) to int8 - minimal storage size.

fp32 -> int8 dynamic weight quantization on every Linear in the expert.
A ~460 KB fp32 pack typically lands near ~150 KB; the whole 6-pack system
+ trunk shrinks from ~4.3 MB to ~1.5 MB. int8 packs load like any other
pack (torch.deserialization of quantized modules).

    python -m skill_lm.export_pack --all
    python -m skill_lm.export_pack --pack knowledge
"""
from __future__ import annotations

import argparse
import os

import torch

from .frozen import SkillPack, TrunkModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def quantize_pack(path: str, out_dir: str | None = None) -> str:
    pack = SkillPack.load(path)
    pack.eval()
    qpack = torch.ao.quantization.quantize_dynamic(
        pack, {torch.nn.Linear}, dtype=torch.qint8)
    out = (out_dir or os.path.dirname(path) or ".")
    out_path = os.path.join(out, os.path.basename(path).replace(".pack", ".pack.q8"))
    torch.save(qpack, out_path)
    return out_path


def size_report(paths: list[str]) -> None:
    print(f"{'file':<26} {'size':>10}")
    tot = 0
    for p in paths:
        s = os.path.getsize(p)
        tot += s
        print(f"{os.path.basename(p):<26} {s / 1e3:>8.0f} KB")
    print(f"{'TOTAL':<26} {tot / 1e6:>8.2f} MB")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packs-dir", default=os.path.join(REPO, "packs"))
    p.add_argument("--pack", default=None, help="single pack name")
    p.add_argument("--all", action="store_true")
    p.add_argument("--trunk", action="store_true",
                   help="also export ckpt/trunk.q8.pt")
    args = p.parse_args(argv)

    if args.pack:
        out = quantize_pack(os.path.join(args.packs_dir, f"{args.pack}.pack"))
        size_report([os.path.join(args.packs_dir, f"{args.pack}.pack"), out])
        return
    if args.all:
        outs, srcs = [], []
        for f in sorted(os.listdir(args.packs_dir)):
            if f.endswith(".pack"):
                srcs.append(os.path.join(args.packs_dir, f))
                outs.append(quantize_pack(srcs[-1]))
        size_report(srcs + outs)
    if args.trunk:
        trunk = TrunkModel.load(os.path.join(REPO, "ckpt", "trunk.pt"), frozen=True)
        qtrunk = torch.ao.quantization.quantize_dynamic(
            trunk, {torch.nn.Linear}, dtype=torch.qint8)
        out = os.path.join(REPO, "ckpt", "trunk.q8.pt")
        torch.save(qtrunk, out)
        size_report([os.path.join(REPO, "ckpt", "trunk.pt"), out])


if __name__ == "__main__":
    main()
