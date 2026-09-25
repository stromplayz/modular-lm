"""Skill datasets for the Skill-Modular Language Model.

Four skills, each with its own training corpus:

  story : fluent simple English   -> TinyStories slice (downloaded, ranged)
  qa    : factual answering       -> bundled facts bank (assets/facts.txt)
  math  : exact arithmetic        -> generated on the fly (infinite)
  count : counting letters        -> generated from a word list

Each example is a small self-contained text; corpora are built by joining
examples with blank lines. The skill label for a training window is simply
which corpus the window was cropped from.
"""
from __future__ import annotations

import os
import random
import re
import urllib.request

SKILL_NAMES = ["story", "qa", "math", "count"]
SKILL_IDS = {name: i for i, name in enumerate(SKILL_NAMES)}

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
)

_WORDLIST = """apple dog cat sun moon star tree house water happy smile bird fish book
cloud rain snow green blue yellow table chair window garden river mountain forest music
friend school letter orange banana purple flower butterfly elephant monkey dinner breakfast
summer winter spring morning evening night dream story game ball kite milk bread cheese
doctor farmer teacher student computer phone paper pencil rocket planet ocean island castle
dragon wizard knight princess robot shadow whisper journey village market chicken rabbit
turtle spider penguin dolphin tiger lion zebra camel koala panda""".split()

# ---------------------------------------------------------------------- #
# story
# ---------------------------------------------------------------------- #
def download_tinystories(path: str, max_mb: float = 30.0) -> str:
    """Range-download the first `max_mb` megabytes of TinyStories train split."""
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    end = int(max_mb * 1_000_000) - 1
    req = urllib.request.Request(TINYSTORIES_URL, headers={"Range": f"bytes=0-{end}"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


def load_story_corpus(path: str, min_words: int = 35, max_words: int = 350,
                      opener_every: int = 4) -> str:
    """Load stories AND short opener fragments.

    The opener fragments (first ~10 words, emitted for every `opener_every`-th
    story) teach the Skill Router that a SHORT fluent-English prompt is a
    story request - without them the router maps any short input to the
    short-form skills (math/qa/count).
    """
    text = open(path, encoding="utf-8", errors="ignore").read()
    parts = re.split(r"<\|end_of_text\|>|<|endoftext\|>|\n\s*\n", text)
    keep = []
    n_openers = 0
    for p in parts:
        p = " ".join(p.split())
        n = len(p.split())
        if min_words <= n <= max_words:
            keep.append(p)
            if n_openers % opener_every == 0:
                keep.append(" ".join(p.split()[:10]))
            n_openers += 1
    return "\n\n".join(keep) + "\n\n"

# ---------------------------------------------------------------------- #
# qa (facts bank)
# ---------------------------------------------------------------------- #
def load_facts(path: str) -> list[tuple[str, str]]:
    facts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|||" in line:
                q, a = line.split("|||", 1)
                facts.append((q.strip(), a.strip()))
    return facts


_QA_TEMPLATES = [
    "Q: {q}\nA: {a}.",
    "Question: {q}\nAnswer: {a}.",
    "{q}\nAnswer: {a}.",
]


def gen_qa(rng: random.Random, facts: list[tuple[str, str]]) -> str:
    q, a = rng.choice(facts)
    return rng.choice(_QA_TEMPLATES).format(q=q, a=a)

# ---------------------------------------------------------------------- #
# math
# ---------------------------------------------------------------------- #
_MATH_TEMPLATES = [
    "Compute: {a} {op} {b}\nAnswer: {ans}",
    "What is {a} {op} {b}?\nAnswer: {ans}",
    "{a} {op} {b} = ?\nAnswer: {ans}",
]


def _operand(rng: random.Random) -> int:
    """Curriculum: small operands dominate so (a, b) pairs get many exposures;
    a long tail keeps coverage of the full 0..99 range."""
    r = rng.random()
    if r < 0.50:
        return rng.randint(0, 12)
    if r < 0.80:
        return rng.randint(0, 29)
    return rng.randint(0, 99)


def _add_scratchpad(a: int, b: int, ans: int) -> str:
    """Place-value decomposition: teaches the ALGORITHM, not memorization.

    23 + 45 -> '20 + 40 = 60. 3 + 5 = 8. 60 + 8 = 68.'
    Each intermediate lives in a tiny learned space (tens <= 180, ones <= 18).
    """
    t, u = a - a % 10, a % 10
    t2, u2 = b - b % 10, b % 10
    tens_sum, ones_sum = t + t2, u + u2
    return f"{t} + {t2} = {tens_sum}. {u} + {u2} = {ones_sum}. " \
           f"{tens_sum} + {ones_sum} = {ans}."


def gen_math(rng: random.Random) -> str:
    op = rng.choice(["+", "-", "x"])
    if op == "x":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
    else:
        a, b = _operand(rng), _operand(rng)
        if op == "-" and b > a:
            a, b = b, a
    ans = a + b if op == "+" else a - b if op == "-" else a * b
    head = rng.choice(_MATH_TEMPLATES).split("\n")[0].format(a=a, op=op, b=b, ans=ans)
    if op == "+" and rng.random() < 0.6:
        return f"{head}\n{_add_scratchpad(a, b, ans)}\nAnswer: {ans}"
    return rng.choice(_MATH_TEMPLATES).format(a=a, op=op, b=b, ans=ans)

# ---------------------------------------------------------------------- #
# count
# ---------------------------------------------------------------------- #
def gen_count(rng: random.Random) -> str:
    w = rng.choice(_WORDLIST)
    n = len(w)
    kind = rng.random()
    if kind < 0.5:
        return f"How many letters are in the word '{w}'?\nAnswer: {n}"
    if kind < 0.8:
        return f"Count the letters in '{w}'.\nAnswer: {n}"
    first = w[0]
    return f"What is the first letter of '{w}'?\nAnswer: {first}"

# ---------------------------------------------------------------------- #
# corpus assembly
# ---------------------------------------------------------------------- #
def build_skill_texts(
    story_path: str | None = None,
    facts_path: str | None = None,
    story_mb: float = 30.0,
    n_math: int = 40_000,
    n_count: int = 30_000,
    n_qa: int = 20_000,
    seed: int = 1337,
) -> dict[str, str]:
    """Return {skill_name: corpus_text}. Heavy skills get many examples so
    round-robin batches see fresh data every step."""
    rng = random.Random(seed)
    texts: dict[str, str] = {}

    if story_path and os.path.exists(story_path):
        texts["story"] = load_story_corpus(story_path)

    if facts_path and os.path.exists(facts_path):
        facts = load_facts(facts_path)
        texts["qa"] = "\n\n".join(gen_qa(rng, facts) for _ in range(n_qa)) + "\n\n"

    texts["math"] = "\n\n".join(gen_math(rng) for _ in range(n_math)) + "\n\n"
    texts["count"] = "\n\n".join(gen_count(rng) for _ in range(n_count)) + "\n\n"
    return texts


# ---------------------------------------------------------------------- #
_ANS_RE = re.compile(r"Answer:\s*(.+)")


def extract_answer(text: str) -> str | None:
    m = _ANS_RE.search(text)
    return m.group(1).strip().rstrip(".") if m else None
