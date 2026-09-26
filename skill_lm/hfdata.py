"""Sample real corpora from HUGE Hugging Face datasets without downloading them.

The datasets-server Rows API exposes paginated rows of parquet-backed
datasets. c4_200m is 18.28M grammar-correction pairs (~2GB+); we fetch only
N pages of 100 rows - kilobytes of traffic for a dataset measured in GB.

    python -m skill_lm.hfdata --dataset liweili/c4_200m --split train \
        --n 3500 --in-field input --out-field output \
        --out assets/grammar/c4_pairs.txt

Pair mode  (--in-field/--out-field): writes "bad ||| good" lines.
Text mode  (--text-field):            writes raw text blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "modular-lm-hfdata/0.4 (edge LM pack training)"}
PAGE = 100


def rows(dataset: str, config: str, split: str, offset: int,
         length: int = PAGE, timeout: int = 30) -> dict:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                "split": split, "offset": offset, "length": length})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def total_rows(dataset: str, config: str, split: str) -> int:
    return rows(dataset, config, split, 0, 1).get("num_rows_total", 0)


def extract(row: dict, field: str) -> str:
    v = row.get(field, "")
    return v.strip() if isinstance(v, str) else ""


def sample_pairs(dataset: str, config: str, split: str, n: int,
                 in_field: str, out_field: str, seed: int,
                 lo: int = 24, hi: int = 160) -> list[tuple[str, str]]:
    """n (bad, good) pairs sampled at random offsets - only n/100 pages move."""
    rng = random.Random(seed)
    total = total_rows(dataset, config, split)
    if total <= 0:
        raise SystemExit(f"dataset {dataset}/{config}/{split} unavailable")
    print(f"[hfdata] {dataset} {split}: {total:,} rows total; sampling {n}")
    pairs, seen = [], set()
    tries = 0
    while len(pairs) < n and tries < n * 12:
        tries += 1
        off = rng.randrange(0, max(1, total - PAGE))
        try:
            data = rows(dataset, config, split, off)
        except Exception as exc:  # noqa: BLE001 - transient server hiccups
            print(f"[hfdata] page @{off} failed: {exc}")
            continue
        for row in data.get("rows", []):
            r = extract(row["row"], in_field)
            g = extract(row["row"], out_field)
            if not r or not g or r == g:
                continue
            if not (lo <= len(r) <= hi and len(g) <= hi):
                continue
            if "|||" in r or "|||" in g:
                continue
            key = r[:60]
            if key in seen:
                continue
            seen.add(key)
            pairs.append((r, g))
            if len(pairs) >= n:
                break
        if tries % 10 == 0:
            print(f"[hfdata] {len(pairs)}/{n} pairs ({tries} pages fetched)")
    return pairs


def sample_text(dataset: str, config: str, split: str, n: int,
                text_field: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    total = total_rows(dataset, config, split)
    out, tries = [], 0
    while len(out) < n and tries < n * 12:
        tries += 1
        off = rng.randrange(0, max(1, total - PAGE))
        try:
            data = rows(dataset, config, split, off)
        except Exception:  # noqa: BLE001
            continue
        for row in data.get("rows", []):
            t = extract(row["row"], text_field)
            if 60 <= len(t) <= 600:
                out.append(t)
                if len(out) >= n:
                    break
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="default")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--in-field", default=None)
    p.add_argument("--out-field", default=None)
    p.add_argument("--text-field", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args(argv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.in_field and args.out_field:
        pairs = sample_pairs(args.dataset, args.config, args.split, args.n,
                             args.in_field, args.out_field, args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            for bad, good in pairs:
                f.write(f"{bad} ||| {good}\n")
        print(f"[done] {len(pairs)} pairs -> {args.out}")
    elif args.text_field:
        texts = sample_text(args.dataset, args.config, args.split, args.n,
                            args.text_field, args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts) + "\n")
        print(f"[done] {len(texts)} text blocks -> {args.out}")
    else:
        raise SystemExit("pass --in-field/--out-field (pairs) or --text-field (text)")


if __name__ == "__main__":
    main()
