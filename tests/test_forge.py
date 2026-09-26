"""Tests for the dataset forge (50-agent expansion) and HF sampler plumbing."""
import random

from skill_lm.forge import (KB, grammar_pairs, vocab_items, expand_fact,
                            load_vocab, _dedupe, _mcq)
from skill_lm.ingest import norm_q


def test_kb_structure():
    assert len(KB) >= 380, "KB should hold ~400 curated optometry facts"
    for dom, q, a, key, stmt in KB[:50]:
        assert dom and q.endswith("?") and a
        assert key and stmt.endswith(".")


def test_expand_fact_forms():
    rng = random.Random(3)
    fact = KB[0]
    key_pool = sorted({f[3] for f in KB if f[3] and len(f[3]) > 2})
    qa, lm = expand_fact(fact, 0, rng, key_pool)
    forms = {f for _q, _a, f, _i in qa}
    assert {"canon", "para", "mcq", "tf"} <= forms
    assert all(item[3] == 0 for item in qa)        # fact idx travels with items
    assert len(lm) == 1 and lm[0].endswith(".")
    # mcq answer must be a letter among A-D
    mcq = [a for q, a, f, _ in qa if f == "mcq"]
    assert all(m in ("A", "B", "C", "D") for m in mcq)


def test_mcq_contains_four_options():
    rng = random.Random(5)
    key_pool = sorted({f[3] for f in KB if f[3] and len(f[3]) > 2})[:40]
    q, a = _mcq(rng, "What is X?", "540", key_pool)
    assert a in ("A", "B", "C", "D")
    assert q.count("(") == 4 and q.count(")") == 4


def test_grammar_pairs_valid():
    rng = random.Random(7)
    pairs = grammar_pairs(rng)
    assert len(pairs) > 100
    for wrong, right in pairs:
        assert wrong != right
        assert wrong[-1] in ".?" and right[-1] in ".?"


def test_vocab_items(tmp_path):
    words = [("abate", "verb", "to lessen in intensity", "subside"),
             ("banal", "adjective", "lacking originality", "trite"),
             ("cogent", "adjective", "clear and convincing", "compelling"),
             ("dearth", "noun", "a scarcity or lack", "shortage"),
             ("ebullient", "adjective", "cheerful and full of energy", "exuberant")]
    rng = random.Random(11)
    qa, lm = vocab_items(rng, words)
    assert len(qa) == 20 and len(lm) == 5
    forms = {f for _q, _a, f in qa}
    assert forms == {"def", "syn", "rev", "mcq"}


def test_dedupe_and_reserved():
    qa = [("What is X?", "y", "canon"), ("What is X?", "y", "canon"),
          ("What is Z?", "w", "canon")]
    out = _dedupe(qa, reserved={norm_q("What is Z?")})
    assert len(out) == 1 and out[0][0] == "What is X?"


def test_vocab_file_loads():
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "grammar", "vocab_words.txt")
    words = load_vocab(p)
    assert len(words) >= 200
    for w, pos, gloss, syn in words:
        assert pos in ("noun", "verb", "adjective") and gloss and syn and w.isalpha()
