"""Corpus ingestion: crawl Wikipedia -> extract QA facts -> pack training data.

The knowledge pipeline (runs anywhere with clean network egress, e.g.
GitHub Actions runners):

    1. crawl      : en.wikipedia.org api.php, plaintext extracts per topic
    2. sentences  : split, clean, length-filter
    3. QA mining  : precise regex patterns ("X is a Y", "X was born in Y",
                    "X is the capital of Y", "population of ...", ...)
    4. dedupe     : skip questions already covered by existing fact banks
    5. split      : 85% train (assets/knowledge/wiki_facts.txt)
                    15% held-out eval (assets/knowledge/wiki_facts_eval.txt)

Offline mode (`--offline`) skips the network and uses the bundled seed bank
(assets/knowledge/seed_facts.txt) so the knowledge pack can always be trained,
even with zero connectivity.

Usage:
    python -m skill_lm.ingest --topics 40 --out assets/knowledge
    python -m skill_lm.ingest --offline
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------- #
# topic list: broad coverage across science / geography / history / tech
# ---------------------------------------------------------------------- #
DEFAULT_TOPICS = [
    # animals & nature
    "Blue whale", "African elephant", "Peregrine falcon", "Great white shark",
    "Honey bee", "Bamboo", "Redwood tree", "Octopus", "Emperor penguin",
    "Chameleon", " photosynthesis", "Volcano", "Coral reef", "Lightning",
    # space
    "Mars", "Jupiter", "Saturn", "Moon", "Sun", "Black hole", "Milky Way",
    "International Space Station", "Comet", "Neptune",
    # science & tech
    "Electricity", "Gravity", "DNA", "Vaccine", "Antibiotic", "Telescope",
    "Battery (electricity)", "Laser", "Internet", "Transistor", "Magnet",
    "Periodic table", "Atom", "X-ray", "Penicillin",
    # geography (capitals handled by pattern; countries/landmarks)
    "Amazon rainforest", "Sahara", "Mount Everest", "Nile", "Great Barrier Reef",
    "Antarctica", "Pacific Ocean", "Alps", "Grand Canyon", "Dead Sea",
    # history & human world
    "Great Wall of China", "Pyramid of Giza", "Renaissance", "Industrial Revolution",
    "Printing press", "Steam engine", "Wright Flyer", "Paper", "Compass",
    "Olympic Games", "Leonardo da Vinci", "Isaac Newton", "Marie Curie",
    "Albert Einstein", "Charles Darwin", "Ada Lovelace",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "modular-lm-ingest/0.3 (from-scratch research LM; GitHub Actions)"}


def fetch_page(title: str, timeout: int = 30) -> str | None:
    """Plain-text extract of a Wikipedia article (explaintext)."""
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": title,
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------- #
# sentence splitting + QA mining patterns
# ---------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_PATTERNS: list[tuple[re.Pattern, callable]] = []


def _pat(rx: str, fn: callable) -> None:
    _PATTERNS.append((re.compile(rx), fn))


# NOTE: most-specific patterns first - the generic "X is a/an/the Y" rule
# must come LAST or it shadows "X is the capital of Y" etc.

# "X is the capital of Y."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,40}?)\s+is\s+the\s+capital\s+of\s+(?P<y>[A-Z][\w'’\- ]{2,40}?)[.]$",
     lambda m: (f"What is the capital of {m['y'].strip()}?", m["x"].strip()))
# "The capital of X is Y."
_pat(r"^The\s+capital\s+of\s+(?P<x>[A-Z][\w'’\- ]{2,40}?)\s+is\s+(?P<y>[A-Z][\w'’\- ]{2,40}?)[.]$",
     lambda m: (f"What is the capital of {m['x'].strip()}?", m["y"].strip()))
# "X was born on/in DATE ..."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+was\s+born\s+(?:on|in)\s+(?P<y>[A-Z0-9][^.]{3,60}?)[.]$",
     lambda m: (f"When was {m['x'].strip()} born?", m["y"].strip()))
# "X died on/in DATE ..."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+died\s+(?:on|in)\s+(?P<y>[A-Z0-9][^.]{3,60}?)[.]$",
     lambda m: (f"When did {m['x'].strip()} die?", m["y"].strip()))
# "X is located/situated in Y ..."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+is\s+(?:located|situated)\s+in\s+(?P<y>[A-Z][\w'’\- ]{2,60}?)(?:\s*\([^)]*\))?\s*[.]$",
     lambda m: (f"Where is {m['x'].strip()} located?", m["y"].strip()))
# "X has a population of about/around N (...)."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+has\s+a\s+population\s+of\s+(?:about\s+|around\s+|over\s+)?(?P<y>[0-9][0-9,.]{2,20}?)(?:\s*\([^)]*\))?\s*[.]$",
     lambda m: (f"What is the population of {m['x'].strip()}?", m["y"].strip()))
# "X was discovered/invented/written/created by Y ..."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+was\s+(?P<verb>discovered|invented|written|created|developed|designed)\s+by\s+(?P<y>[A-Z][^.]{2,60}?)[.]$",
     lambda m: (f"Who { {'discovered':'discovered','invented':'invented','written':'wrote','created':'created','developed':'developed','designed':'designed'}[m['verb']] } {m['x'].strip()}?", m["y"].strip()))
# "X is a species of Y."
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+is\s+a\s+species\s+of\s+(?P<y>[A-Za-z][\w'’\- ]{3,60}?)[.]$",
     lambda m: (f"What is {m['x'].strip()}?", m["y"].strip()))
# "X is a/an/the Y ( ... )."   <- generic, keep LAST
_pat(r"^(?P<x>[A-Z][\w'’\- ]{2,60}?)\s+is\s+(?:a|an|the)\s+(?P<y>[A-Za-z][\w'’\- ]{3,80}?)(?:\s*\([^)]*\))?\s*[.]$",
     lambda m: (f"What is {m['x'].strip()}?", m["y"].strip()))

_BAD_Q = re.compile(r"\b(this|these|those|it|he|she|they|also|however|there)\b", re.I)


def mine_facts(text: str) -> list[tuple[str, str, str]]:
    """Return (question, answer, source_sentence) triples from article text."""
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para or para.startswith(("==", "|", "!", "*")):
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = " ".join(sent.split())
            if not (22 <= len(sent) <= 220):
                continue
            for rx, fn in _PATTERNS:
                m = rx.match(sent)
                if not m:
                    continue
                try:
                    q, a = fn(m)
                except Exception:  # noqa: BLE001
                    continue
                if not q or not a or len(a) < 2 or len(a) > 90:
                    continue
                if _BAD_Q.search(q.split("?", 1)[0].split()[-1] if q else ""):
                    continue
                out.append((q.strip(), a.strip().rstrip("."), sent))
                break
    return out


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def existing_questions(fact_files: list[str]) -> set[str]:
    """Question strings already owned by other packs (avoid routing clashes)."""
    seen = set()
    for fp in fact_files:
        if not os.path.exists(fp):
            continue
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "|||" in line:
                seen.add(norm_q(line.split("|||", 1)[0]))
    return seen


# ---------------------------------------------------------------------- #
def crawl(topics: list[str], max_pages: int, delay: float = 0.5) -> list[tuple[str, str, str]]:
    facts, pages_done = [], 0
    for t in topics:
        if pages_done >= max_pages:
            break
        title = t.strip()
        if not title:
            continue
        text = fetch_page(title)
        pages_done += 1
        if text:
            got = mine_facts(text)
            print(f"[crawl] {title:<32} {len(got):>3} facts")
            facts.extend(got)
        else:
            print(f"[crawl] {title:<32} MISS")
        time.sleep(delay)
    return facts


def write_banks(facts: list[tuple[str, str, str]], out_dir: str,
                reserved_qs: set[str]) -> tuple[str, str, int]:
    os.makedirs(out_dir, exist_ok=True)
    seen, unique = set(), []
    for q, a, _src in facts:
        k = norm_q(q)
        if not k or k in seen or k in reserved_qs:
            continue
        seen.add(k)
        unique.append((q, a))
    # deterministic held-out split: every 7th fact is eval
    train, held = [], []
    for i, (q, a) in enumerate(unique):
        (held if i % 7 == 0 else train).append((q, a))
    tr_path = os.path.join(out_dir, "wiki_facts.txt")
    ev_path = os.path.join(out_dir, "wiki_facts_eval.txt")
    with open(tr_path, "w", encoding="utf-8") as f:
        for q, a in train:
            f.write(f"{q} ||| {a}\n")
    with open(ev_path, "w", encoding="utf-8") as f:
        for q, a in held:
            f.write(f"{q} ||| {a}\n")
    return tr_path, ev_path, len(unique)


# ---------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", type=int, default=40, help="max pages to crawl")
    p.add_argument("--out", default=os.path.join(REPO, "assets", "knowledge"))
    p.add_argument("--offline", action="store_true",
                   help="use bundled seed bank only (no network)")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = p.parse_args(argv)

    reserved = existing_questions([
        os.path.join(REPO, "assets", "facts.txt"),
        os.path.join(REPO, "assets", "optometry.txt"),
    ])

    if args.offline:
        seed = os.path.join(REPO, "assets", "knowledge", "seed_facts.txt")
        facts = []
        for line in open(seed, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "|||" in line:
                q, a = line.split("|||", 1)
                facts.append((q.strip(), a.strip(), "seed"))
        print(f"[offline] seed bank: {len(facts)} facts")
    else:
        topics = DEFAULT_TOPICS[: args.topics]
        print(f"[crawl] {len(topics)} topics, reserved questions: {len(reserved)}")
        facts = crawl(topics, args.topics, args.delay)
        seed = os.path.join(REPO, "assets", "knowledge", "seed_facts.txt")
        if os.path.exists(seed):
            for line in open(seed, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "|||" in line:
                    q, a = line.split("|||", 1)
                    facts.append((q.strip(), a.strip(), "seed"))
            print(f"[seed ] merged seed bank -> {len(facts)} total candidates")

    tr, ev, n = write_banks(facts, args.out, reserved)
    print(f"[done] {n} unique facts | train -> {tr} | held-out eval -> {ev}")

if __name__ == "__main__":
    main()
