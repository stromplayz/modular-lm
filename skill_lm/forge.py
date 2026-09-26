"""Dataset forge: 50 parallel agents expand a curated seed KB into a huge
training bank, then grammar/vocabulary corpora are synthesized on top.

Optometry expansion per fact (question, answer, key, statement):
    3 QA surfaces   canonical + 2 paraphrase lead-ins
    2 MCQs          4 options, distractors drawn from the same domain
    2 True/False    statement + same-domain answer-swap corruption
    1 cloze         "Fill in the blank: ___" -> key
    1 statement     -> LM corpus (grammar-quality declarative text)

Agents: the fact list is split into N shards; N worker processes expand
their shard independently (own RNG seed) and report per-agent counts.

    python -m skill_lm.forge --agents 50
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import multiprocessing as mp
import os
import random
import re

from .kb_optometry import KB1
from .kb_optometry2 import KB2
from .kb_optometry3 import KB3
from .ingest import norm_q, existing_questions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB: list[tuple[str, str, str, str, str]] = KB1 + KB2 + KB3


# ---------------------------------------------------------------------- #
# optometry expansion (runs inside each agent)
# ---------------------------------------------------------------------- #
_LEADS = ["Quick question: {q}", "Exam prep: {q}"]


def _mcq(rng: random.Random, q: str, ans: str, pool: list[str]) -> tuple[str, str]:
    distractors = [d for d in pool if d != ans]
    rng.shuffle(distractors)
    opts = distractors[:3] + [ans]
    rng.shuffle(opts)
    letters = ["A", "B", "C", "D"]
    body = " ".join(f"({letters[i]}) {o}" for i, o in enumerate(opts))
    qi = f"{q} {body}"
    return qi, letters[opts.index(ans)]


def expand_fact(fact: tuple[str, str, str, str, str], fact_idx: int,
                rng: random.Random, key_pool: list[str]) -> tuple[list, list]:
    """fact -> (qa_items, lm_lines).
    qa item = (question, answer, form, fact_idx) - index travels with the item
    so the held-out split is exact (whole FACTS held out, never forms)."""
    dom, q, a, key, stmt = fact
    qa, lm = [], []
    qa.append((q, a, "canon", fact_idx))
    ql = q[0].lower() + q[1:] if q[:2] != "I " else q
    for lead in _LEADS:
        qa.append((lead.format(q=ql).rstrip("?.") + ("?" if q.endswith("?") else "."), a, "para", fact_idx))
    for _ in range(2):
        qi, ai = _mcq(rng, q, a, key_pool)
        qa.append((qi, ai, "mcq", fact_idx))
    qa.append((f"True or false: {stmt}", "true", "tf", fact_idx))
    wrongs = [k for k in key_pool if k != key and len(k) > 2]
    if key and key.lower() in stmt.lower() and wrongs:
        wrong = rng.choice(wrongs)
        fstmt = re.sub(re.escape(key), wrong, stmt, count=1, flags=re.I)
        if fstmt != stmt:
            qa.append((f"True or false: {fstmt}", "false", "tf", fact_idx))
        cloze = re.sub(re.escape(key), "____", stmt, count=1, flags=re.I)
        if "____" in cloze:
            qa.append((f"Fill in the blank: {cloze}", key, "cloze", fact_idx))
    lm.append(stmt)
    return qa, lm


def agent_worker(payload: tuple) -> dict:
    """One 'agent': expand its shard of facts with its own RNG seed."""
    agent_id, shard, seed, key_pool = payload
    rng = random.Random(seed)
    qa, lm = [], []
    for fact_idx, fact in shard:
        fqa, flm = expand_fact(fact, fact_idx, rng, key_pool)
        qa.extend(fqa)
        lm.extend(flm)
    return {"agent": agent_id, "qa": qa, "lm": lm, "n": len(qa)}


# ---------------------------------------------------------------------- #
# grammar synthesis (rule-based wrong/right pairs)
# ---------------------------------------------------------------------- #
_SV = [("He", "go", "goes"), ("She", "study", "studies"), ("It", "work", "works"),
       ("The doctor", "write", "writes"), ("The engineer", "read", "reads"),
       ("The student", "watch", "watches"), ("My friend", "finish", "finishes"),
       ("The manager", "enjoy", "enjoys"), ("The professor", "teach", "teaches"),
       ("The pilot", "run", "runs")]
_SV_OBJ = ["the report carefully", "a novel at night", "the results with interest",
           "the lecture notes", "the contract closely", "the data every morning"]
_IRR = [("go", "went"), ("see", "saw"), ("take", "took"), ("buy", "bought"),
        ("bring", "brought"), ("think", "thought"), ("teach", "taught"),
        ("catch", "caught"), ("write", "wrote"), ("give", "gave"),
        ("find", "found"), ("make", "made"), ("come", "came"), ("know", "knew"),
        ("speak", "spoke"), ("drive", "drove"), ("eat", "ate"), ("drink", "drank"),
        ("swim", "swam"), ("win", "won"), ("fall", "fell"), ("break", "broke")]
_COMP = [("more better", "better"), ("more faster", "faster"), ("more easier", "easier"),
         ("most easiest", "easiest"), ("more larger", "larger"), ("most simple", "simplest")]
_MUCH = [("How much {n} did you review?", "How many {n} did you review?",
          ["books", "patients", "lenses", "results", "reports", "students"]),
         ("There were much {n} in the study.", "There was much {n} in the study.",
          ["water", "time", "information", "evidence", "research", "progress"])]
_YOURE = [("Your welcome to join us.", "You're welcome to join us."),
          ("You're car is ready.", "Your car is ready."),
          ("You're analysis was precise.", "Your analysis was precise."),
          ("Your going to succeed.", "You're going to succeed."),
          ("The team appreciated you're effort.", "The team appreciated your effort.")]
_THEYRE = [("Their going home early.", "They're going home early."),
           ("There books were returned.", "Their books were returned."),
           ("There waiting outside the clinic.", "They're waiting outside the clinic."),
           ("The students left they're notes.", "The students left their notes."),
           ("They're names were on the list.", "Their names were on the list.")]
_ITS = [("Its raining again.", "It's raining again."),
        ("The committee gave it's approval.", "The committee gave its approval."),
        ("The machine lost it's calibration.", "The machine lost its calibration."),
        ("Its a clear diagnosis.", "It's a clear diagnosis.")]
_THENTHAN = [("The new lens is sharper then the old one.", "The new lens is sharper than the old one."),
             ("She finished the exam, than she reviewed it.", "She finished the exam, then she reviewed it."),
             ("Results were more consistent then expected.", "Results were more consistent than expected.")]
_FEWER = [("There were less errors this quarter.", "There were fewer errors this quarter."),
          ("We saw less patients today.", "We saw fewer patients today."),
          ("He had fewer time for the exam.", "He had less time for the exam."),
          ("Fewer information was provided.", "Less information was provided.")]
_BETW = [("Between you and I, the plan needs work.", "Between you and I".replace("I", "me") + ", the plan needs work."),
         ("This stays between you and I.", "This stays between you and me.")]
_PREP = [("The meeting is in Monday.", "The meeting is on Monday."),
         ("The trial began on 2021.", "The trial began in 2021."),
         ("We test patients at the morning.", "We test patients in the morning."),
         ("The clinic opens in Friday afternoon.", "The clinic opens on Friday afternoon."),
         ("Results improve on the evening.", "Results improve in the evening.")]
_DOESNT = [("He doesn't likes tea.", "He doesn't like tea."),
           ("She don't agree with the report.", "She doesn't agree with the report."),
           ("It don't affect the outcome.", "It doesn't affect the outcome.")]
_ADVERB = [("He runs quick before practice.", "He runs quickly before practice."),
           ("She sings beautiful at recitals.", "She sings beautifully at recitals."),
           ("The exam was extreme difficult.", "The exam was extremely difficult."),
           ("He answered honest during the review.", "He answered honestly during the review.")]
_MEI = [("Me and my colleague presented.", "My colleague and I presented."),
        ("Me and him went to the lab.", "He and I went to the lab.")]
_SINCEFOR = [("I have practiced since two years.", "I have practiced for two years."),
             ("She worked there for 2019.", "She worked there since 2019.")]
_THEREIS = [("There is many options available.", "There are many options available."),
            ("There were a problem with the fit.", "There was a problem with the fit.")]


def grammar_pairs(rng: random.Random) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for subj, base, third in _SV:
        for obj in _SV_OBJ:
            pairs.append((f"{subj} {base} {obj}.", f"{subj} {third} {obj}."))
    for base, past in _IRR:
        place = rng.choice(["to the conference", "the samples", "the exam", "the ward",
                            "the calibration", "the lecture", "across the city"])
        pairs.append((f"Yesterday he {base}ed {place}." if not base.endswith("e")
                      else f"Yesterday he {base}d {place}.",
                      f"Yesterday he {past} {place}."))
    for wrong, right in _COMP:
        pairs.append((f"This solution is {wrong} than the standard.", f"This solution is {right} than the standard."))
    for tmpl_w, tmpl_r, nouns in _MUCH:
        for n in nouns:
            pairs.append((tmpl_w.format(n=n), tmpl_r.format(n=n)))
    for w, r in _YOURE + _THEYRE + _ITS + _THENTHAN + _FEWER + _BETW + _PREP + _DOESNT + _ADVERB + _MEI + _SINCEFOR + _THEREIS:
        pairs.append((w, r))
    return pairs


# ---------------------------------------------------------------------- #
# vocabulary expansion
# ---------------------------------------------------------------------- #
def load_vocab(path: str) -> list[tuple[str, str, str, str]]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        w, pos, gloss, syn = line.split("|")
        out.append((w, pos, gloss, syn))
    return out


def vocab_items(rng: random.Random, words: list[tuple[str, str, str, str]]) -> tuple[list, list]:
    qa, lm = [], []
    pool = [w for w, _, _, _ in words]
    for w, pos, gloss, syn in words:
        qa.append((f"What does '{w}' mean?", gloss, "def"))
        qa.append((f"What is a synonym for '{w}'?", syn, "syn"))
        qa.append((f"Which word means '{gloss}'?", w, "rev"))
        distractors = rng.sample([p for p in pool if p != w], 3)
        opts = distractors + [w]
        rng.shuffle(opts)
        letters = ["A", "B", "C", "D"]
        qi = (f"Which word means '{gloss}'? "
              + " ".join(f"({letters[i]}) {o}" for i, o in enumerate(opts)))
        qa.append((qi, letters[opts.index(w)], "mcq"))
        lm.append(f"{w.capitalize()} ({pos}): {gloss}.")
    return qa, lm


# ---------------------------------------------------------------------- #
def _dedupe(qa: list[tuple[str, str, str]], reserved: set[str]) -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for q, a, form in qa:
        k = norm_q(q) + "||" + norm_q(a)[:40]
        if not k.strip("|") or k in seen:
            continue
        if form in ("canon", "para") and norm_q(q) in reserved:
            continue
        seen.add(k)
        out.append((q, a, form))
    return out


def write_bank(path: str, qa: list[tuple[str, str, str]], plain: bool = False) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for q, a, _f in qa:
            a_txt = str(a).replace("\n", " ")
            f.write(f"{q.strip()} ||| {a_txt}\n")
    return len(qa)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=50)
    p.add_argument("--out", default=os.path.join(REPO, "assets", "optometry_v2"))
    p.add_argument("--grammar-out", default=os.path.join(REPO, "assets", "grammar"))
    p.add_argument("--hf-pairs", default=os.path.join(REPO, "assets", "grammar", "c4_pairs.txt"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--shard", type=int, default=0,
                   help="1-based shard id: expand ONLY this shard (CI matrix mode)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-out", default=None,
                   help="write this shard's expansion as JSON and exit")
    p.add_argument("--merge-glob", default=None,
                   help="glob of shard JSONs to merge instead of local Pool")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.grammar_out, exist_ok=True)
    rng = random.Random(args.seed)

    reserved = existing_questions([
        os.path.join(REPO, "assets", "facts.txt"),
        os.path.join(REPO, "assets", "knowledge", "wiki_facts.txt"),
    ])

    # ---- 1. optometry: agents expand shards -------------------------- #
    key_pool = sorted({f[3] for f in KB if f[3] and len(f[3]) > 2})

    if args.shard_out:
        # CI matrix mode: one agent, one shard, one JSON artifact
        assert 1 <= args.shard <= args.num_shards, "bad shard id"
        per = (len(KB) + args.num_shards - 1) // args.num_shards
        lo, hi = (args.shard - 1) * per, min(args.shard * per, len(KB))
        res = agent_worker((args.shard, list(enumerate(KB[lo:hi], start=lo)),
                            args.seed + (args.shard - 1) * 101, key_pool))
        with open(args.shard_out, "w") as f:
            json.dump(res, f)
        print(f"[shard {args.shard}/{args.num_shards}] {res['n']} items -> {args.shard_out}")
        return

    if args.merge_glob:
        files = sorted(_glob.glob(args.merge_glob))
        assert files, f"no shard files match {args.merge_glob}"
        results = [json.load(open(f)) for f in files]
        results.sort(key=lambda r: r["agent"])
        print(f"[forge] merged {len(results)} shard artifacts "
              f"({sum(r['n'] for r in results)} items)")
    else:
        shards = []
        per = (len(KB) + args.agents - 1) // args.agents
        for i in range(args.agents):
            lo, hi = i * per, min((i + 1) * per, len(KB))
            if lo < hi:
                shards.append((i + 1, list(enumerate(KB[lo:hi], start=lo)),
                               args.seed + i * 101, key_pool))
        print(f"[forge] {len(KB)} seed facts -> {len(shards)} agents x ~{per} facts")
        with mp.Pool(processes=args.agents) as pool:
            results = pool.map(agent_worker, shards)
        results.sort(key=lambda r: r["agent"])
    for r in results:
        print(f"[agent {r['agent']:02d}] expanded {r['n']:>3} items")

    # merge agents: every 9th FACT (all its forms, exact via carried index) -> eval
    train_qa, eval_qa, train_lm, eval_lm = [], [], [], []
    for r in results:
        for q, a, form, idx in r["qa"]:
            (eval_qa if idx % 9 == 0 else train_qa).append((q, a, form))
        train_lm.extend(r["lm"])

    # merge the OLD optometry bank so nothing regresses (train side only)
    old = os.path.join(REPO, "assets", "optometry.txt")
    if os.path.exists(old):
        for line in open(old, encoding="utf-8"):
            line = line.strip()
            if line and "|||" in line:
                q, a = line.split("|||", 1)
                train_qa.append((q.strip(), a.strip(), "canon-legacy"))

    train_qa = _dedupe(train_qa, reserved)
    eval_qa = _dedupe(eval_qa, reserved)
    tr = os.path.join(args.out, "optometry_v2_facts.txt")
    ev = os.path.join(args.out, "optometry_v2_eval.txt")
    lm = os.path.join(args.out, "optometry_v2_lm.txt")
    n1 = write_bank(tr, train_qa)
    n2 = write_bank(ev, eval_qa)
    with open(lm, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_lm) + "\n")
    print(f"[optometry] train {n1} QA | held-out {n2} QA | LM {len(train_lm)} statements")

    # ---- 2. grammar: rule synthesis + real HF pairs ------------------- #
    gpairs = grammar_pairs(rng)
    core_qa = []
    for w, r in gpairs:
        core_qa.append((f"Fix this sentence: {w}", r, "fix"))
        flip = rng.random() < 0.5
        A, B = (w, r) if not flip else (r, w)
        core_qa.append((f"Which sentence is correct? (A) {A} (B) {B}", "B" if not flip else "A", "mcq"))
    # real corpus pairs (c4_200m via datasets-server): these train as LM TEXT
    # ONLY - 3500 unique long sentences are not exactly recallable at pack
    # scale, and using them as QA targets poisons the loss floor
    n_hf = 0
    if os.path.exists(args.hf_pairs):
        n_hf = sum(1 for l in open(args.hf_pairs, encoding="utf-8") if " ||| " in l)
    words = load_vocab(os.path.join(REPO, "assets", "grammar", "vocab_words.txt"))
    vqa, vlm = vocab_items(rng, words)
    core_qa.extend(vqa)
    gv_eval, gv_train = [], []
    for i, item in enumerate(core_qa):
        (gv_eval if i % 11 == 0 else gv_train).append(item)
    gv_train = _dedupe(gv_train, set())
    gv_eval = _dedupe(gv_eval, set())
    gl = os.path.join(args.grammar_out, "grammar_facts.txt")
    ge = os.path.join(args.grammar_out, "grammar_eval.txt")
    gc = os.path.join(args.grammar_out, "grammar_core.txt")
    glm = os.path.join(args.grammar_out, "grammar_lm.txt")
    n3 = write_bank(gl, gv_train)
    n4 = write_bank(ge, gv_eval)
    # core = vocab + rule items WITHOUT the raw HF pairs (bench recall suite;
    # HF items train LM richness + generalization, not exact recall)
    n_core = write_bank(gc, [g for g in gv_train if g[2] != "hf"])
    hf_lm: list[str] = []
    if os.path.exists(args.hf_pairs):
        for line in open(args.hf_pairs, encoding="utf-8"):
            if " ||| " in line:
                hf_lm.append(line.rstrip("\n").split(" ||| ", 1)[1].strip())
    with open(glm, "w", encoding="utf-8") as f:
        f.write("\n\n".join([r for _w, r in gpairs] + hf_lm + vlm) + "\n")
    print(f"[grammar] train {n3} QA | held-out {n4} QA | core recall bank {n_core} | HF pairs used as LM text: {n_hf}")

    stats = {"seed_facts": len(KB), "agents": args.agents,
             "optometry_train": n1, "optometry_eval": n2, "optometry_lm": len(train_lm),
             "grammar_train": n3, "grammar_eval": n4, "grammar_hf_pairs": n_hf}
    with open(os.path.join(args.out, "forge_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] {json.dumps(stats)}")


if __name__ == "__main__":
    main()
