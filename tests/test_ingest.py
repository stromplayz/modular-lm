"""Tests for ingestion QA mining + the hub + the stream slicer."""
import os

from skill_lm.ingest import mine_facts, norm_q, existing_questions, write_banks


TEXT = """
The blue whale is a marine mammal. It can reach lengths of 30 meters.
The blue whale was born in the ocean during the Miocene era.
African elephant is a species of mammal. African elephants have large ears.
Paris is the capital of France. The city has a population of about 2,100,000.
Isaac Newton was born in 1643. Isaac Newton died in 1727.
The telescope was invented by Hans Lippershey. It changed astronomy.
Mount Everest is located in Nepal.
"""

QA = """Q1 ||| A1
Q2 ||| A2
Q3 ||| A3
Q4 ||| A4
Q5 ||| A5
Q6 ||| A6
Q7 ||| A7
"""


def test_mine_facts_patterns():
    facts = mine_facts(TEXT)
    qs = {norm_q(q) for q, a, s in facts}
    assert any("what is the blue whale" in q for q in qs)
    assert any("what is the capital of france" in q for q in qs)
    assert any("when was isaac newton born" in q for q in qs)
    assert any("when did isaac newton die" in q for q in qs)
    assert any("who invented the telescope" in q for q in qs)
    assert any("where is mount everest located" in q for q in qs)
    # answers look sane
    amap = {norm_q(q): a for q, a, s in facts}
    assert amap["what is the capital of france"] == "Paris"
    assert amap["where is mount everest located"] == "Nepal"
    assert amap["when was isaac newton born"] == "1643"
    assert amap["who invented the telescope"] == "Hans Lippershey"


def test_population_pattern():
    facts = mine_facts(TEXT)
    found = [a for q, a, s in facts if "population" in q.lower()]
    assert found and "2,100,000" in found[0]


def test_write_banks_split_and_dedupe(tmp_path):
    facts = [("What is X?", "x", "s"), ("What is X?", "x", "s"),
             ("What is Y?", "y", "s"), ("What is Z?", "z", "s")]
    tr, ev, n = write_banks(facts, str(tmp_path), set())
    assert n == 3  # deduped
    train_lines = open(tr).read().strip().splitlines()
    eval_lines = open(ev).read().strip().splitlines()
    assert len(train_lines) + len(eval_lines) == 3
    # reserved questions are excluded
    tr2, ev2, n2 = write_banks(facts, str(tmp_path), {norm_q("What is X?")})
    assert n2 == 2


def test_existing_questions(tmp_path):
    fp = os.path.join(tmp_path, "f.txt")
    open(fp, "w").write(QA)
    seen = existing_questions([fp])
    assert norm_q("q1") in seen


def test_stream_helpers_offline():
    from skill_lm.stream import _decode
    import gzip
    raw = gzip.compress("hello world".encode())
    assert _decode(raw) == "hello world"
    assert _decode("plain".encode()) == "plain"
