import pytest

from skill_lm.tokenizer import BPETokenizer

SAMPLE = (
    "Compute: 23 + 45\nAnswer: 68\n\nQ: What is the capital of France?\nA: Paris. "
    "The little girl named Lily ran quickly! She had 12 apples, 3 pears & 1 dog. "
    "How many letters are in the word 'apple'? Emoji: \U0001f600 ok? yes."
)


def test_roundtrip():
    tok = BPETokenizer.train(SAMPLE * 3, vocab_size=512)
    ids = tok.encode(SAMPLE)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == SAMPLE


def test_bpe_compresses():
    tok = BPETokenizer.train(SAMPLE * 3, vocab_size=512)
    naive = len(SAMPLE.encode("utf-8"))
    ids = tok.encode(SAMPLE)
    assert len(ids) < naive * 0.8, "merges should compress text"


def test_save_load_roundtrip(tmp_path):
    tok = BPETokenizer.train(SAMPLE * 3, vocab_size=512)
    path = tmp_path / "tok.json"
    tok.save(str(path))
    tok2 = BPETokenizer.load(str(path))
    assert tok2.merges == tok.merges
    assert tok2.encode(SAMPLE) == tok.encode(SAMPLE)


def test_empty_and_unknown():
    tok = BPETokenizer.train(SAMPLE * 3, vocab_size=512)
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    weird = "café naïve 日本語 \U0001f600"
    assert tok.decode(tok.encode(weird)) == weird


def test_vocab_size_cap():
    tok = BPETokenizer.train("aaaa bbbb aaaa bbbb", vocab_size=400)
    assert tok.vocab_size <= 400
