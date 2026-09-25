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
    texts = D.build_skill_texts(
        story_path=None, facts_path=str(facts),
        n_math=50, n_count=50, n_qa=20, seed=0,
    )
    assert set(D.SKILL_NAMES) - {"story"} <= set(texts)
    assert texts["math"].count("\n\n") >= 49


def test_skill_ids():
    assert D.SKILL_IDS["story"] == 0
    assert D.SKILL_IDS["qa"] == 1
    assert D.SKILL_IDS["math"] == 2
    assert D.SKILL_IDS["count"] == 3
