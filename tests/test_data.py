import re

import pytest

from skill_lm import data as D


def test_math_examples_parse():
    import random
    rng = random.Random(0)
    for _ in range(200):
        ex = D.gen_math(rng)
        ans = D.extract_answer(ex)
        assert ans is not None and ans.lstrip("-").isdigit()
        m = re.search(r"(-?\d+) ([+\-x]) (-?\d+)", ex)
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        expected = a + b if op == "+" else a - b if op == "-" else a * b
        assert int(ans) == expected, ex


def test_math_subtraction_non_negative():
    import random
    rng = random.Random(1)
    for _ in range(200):
        ex = D.gen_math(rng)
        if "-" in ex.split("\n")[0]:
            m = re.search(r"(-?\d+) - (\d+)", ex)
            assert int(m.group(1)) >= int(m.group(2))


def test_math_operand_curriculum():
    """Small operands must dominate: curriculum gives them many more exposures."""
    import random
    rng = random.Random(3)
    small = mid = big = 0
    for _ in range(2000):
        r = random.Random(rng.random())
        a = D._operand(r)
        assert 0 <= a <= 99
        if a <= 12:
            small += 1
        elif a <= 29:
            mid += 1
        else:
            big += 1
    assert small > mid > big, (small, mid, big)
    assert small > 2000 * 0.4  # ~50% expected


def test_count_examples_parse():
    import random
    rng = random.Random(2)
    for _ in range(100):
        ex = D.gen_count(rng)
        ans = D.extract_answer(ex)
        assert ans
        if "letters" in ex:
            word = re.search(r"'(\w+)'", ex).group(1)
            assert int(ans) == len(word)
        else:
            word = re.search(r"'(\w+)'", ex).group(1)
            assert ans == word[0]


def test_facts_load(tmp_path):
    f = tmp_path / "facts.txt"
    f.write_text("# comment\nQ one?|||Answer one\nQ two?|||Answer two\n")
    facts = D.load_facts(str(f))
    assert len(facts) == 2
    assert facts[0] == ("Q one?", "Answer one")


def test_build_skill_texts_has_all_skills(tmp_path):
    facts = tmp_path / "facts.txt"
    facts.write_text("Q: test?|||yes\n")
    eye = tmp_path / "optometry.txt"
    eye.write_text("Eye test question?|||yes\n")
    texts = D.build_skill_texts(
        story_path=None, facts_path=str(facts), optometry_path=str(eye),
        n_math=50, n_count=50, n_qa=20, n_optometry=20, seed=0,
    )
    assert set(D.SKILL_NAMES) - {"story"} <= set(texts)
    assert texts["math"].count("\n\n") >= 49
    assert "Eye test question?" in texts["optometry"] or "Eye Q" in texts["optometry"]


def test_skill_ids():
    assert D.SKILL_IDS["story"] == 0
    assert D.SKILL_IDS["qa"] == 1
    assert D.SKILL_IDS["math"] == 2
    assert D.SKILL_IDS["count"] == 3
    assert D.SKILL_IDS["optometry"] == 4


def test_optometry_examples_parse():
    import random
    rng = random.Random(4)
    facts = D.load_facts(
        __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))),
            "assets", "optometry.txt",
        )
    )
    assert len(facts) > 100
    plain = 0
    for _ in range(300):
        ex = D.gen_optometry(rng, facts)
        ans = D.extract_answer(ex)
        assert ans, ex
        if ex.startswith("Q:"):
            plain += 1
    assert 0 < plain < 150  # plain-Q slice exists but is a minority
