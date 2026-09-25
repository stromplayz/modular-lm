"""Minimal byte-level BPE tokenizer, written from scratch. No dependencies.

Byte alphabet (256 ids) + learned merge ids on top. GPT-2 style regex
pre-tokenization keeps merges inside words/numbers.
"""
from __future__ import annotations

import json
import re
from collections import Counter

GPT2_SPLIT_PATTERN = re.compile(
    r"""'[sS]|'[tT]|'[rR]|'[vV]|'[mM]|'[lL]|'[dD]| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: dict | None = None) -> None:
        self.merges: dict[tuple[int, int], int] = dict(merges or {})
        self._build_vocab()

    def _build_vocab(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ------------------------------------------------------------------ #
    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """Learn `vocab_size - 256` merges from `text`."""
        assert vocab_size >= 256, "vocab_size must cover the 256 byte alphabet"
        tok = cls()
        word_freqs = Counter(GPT2_SPLIT_PATTERN.findall(text))
        seqs = [(list(w.encode("utf-8")), f) for w, f in word_freqs.items()]
        merges: dict[tuple[int, int], int] = {}
        for i in range(vocab_size - 256):
            stats: Counter = Counter()
            for ids, f in seqs:
                for pair in zip(ids, ids[1:]):
                    stats[pair] += f
            if not stats:
                break
            pair = max(stats, key=lambda p: stats[p])
            idx = 256 + i
            merges[pair] = idx
            seqs = [(_merge(ids, pair, idx), f) for ids, f in seqs]
        tok.merges = merges
        tok._build_vocab()
        return tok

    # ------------------------------------------------------------------ #
    def _encode_chunk(self, chunk: bytes) -> list[int]:
        ids = list(chunk)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, 1 << 30))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in GPT2_SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(w.encode("utf-8")))
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": "bpe-v1",
                 "merges": [[a, b, i] for (a, b), i in self.merges.items()]},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(merges={(a, b): i for a, b, i in d["merges"]})
