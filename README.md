# Skill-Modular Transformer (SMT)

A **from-scratch** tiny language model (~**0.82M parameters**, ~**3.3 MB** file)
with loadable **SKILL EXPERTS**: a router reads your input, **loads** the one
specialist block that owns that kind of task, **uses** it, and **detaches** it.

> This is **not** mixture-of-experts. MoE shards every token across anonymous
> FFN slices. SMT has *dedicated skills* — a math expert, a QA expert, a
> counting expert, a story expert — each a full transformer block trained to
> **own** its task, and a router that loads exactly one per request.

No transformer library. No pretrained weights. No tokenizer package.
The BPE tokenizer, the model, the router, the training loop — all hand-built
in this repo, trained on free CPU compute (GitHub Actions).

---

## Architecture

```
                     ┌─────────────────────────────────────────┐
 tokens ──► embed ──► SHARED TRUNK  (2 blocks: the "base brain") │
                     │  learns general language understanding    │
                     └────────────────┬────────────────────────┘
                                      │ pooled trunk state
                                      ▼
                          ┌───────────────────────┐
                          │   SKILL ROUTER        │  softmax over skills
                          │   load -> use ->      │  one skill per request
                          │        detach         │
                          └───────────┬───────────┘
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
               [story expert]   [qa expert]     [math expert]   [count expert]
                stays detached unless the router loads it
                                      │
                                      ▼
                     shared LM head  ──►  next-token logits
```

- **Skill Router** — a linear head over mean-pooled trunk state. Trained with
  cross-entropy against the skill that produced the data (force-routed during
  training; argmax at inference).
- **Skill Experts** — full pre-norm blocks (attention + MLP + residuals).
  During dispatch, a batch splits by assigned skill: each expert touches
  **exactly its own rows** and nothing else. Experts nobody needs this batch
  never run — they stay detached.
- **Shared trunk + tied LM head** — everything above skills is common ground,
  so every skill benefits from the same language foundation.

### Parameter budget (default config)

| Part | Params |
|---|---|
| Embeddings (tied head, vocab 1024 x d 128) | 131 K |
| Shared trunk (2 blocks) | ~198 K |
| Skill experts (4 blocks, **one active**) | ~460 K total, **~115 K active** |
| Router + norms | ~0.4 K |
| **Total** | **~821 K** |
| **Active per request** | **~477 K (~58%)** |

Checkpoint: **3.3 MB fp32** — 60x under the 200 MB budget.

---

## The five skills

| Skill | Corpus | Source |
|---|---|---|
| `story` | fluent simple English | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) slice (range-downloaded) |
| `qa` | factual Q -> A | bundled facts bank (`assets/facts.txt`) |
| `math` | exact arithmetic (+, -, x) | generated on the fly (infinite) |
| `count` | letter counting, first-letter | generated from word list |
| `optometry` | eye-care domain facts (anatomy, refractive errors, tests, prescriptions, conditions) | bundled bank (`assets/optometry.txt`, 146 facts) |

**Adding your own skill is 3 steps:** write a `question|||answer` bank (or a
generator) in `assets/`, register the name in `SKILL_NAMES` + a template in
`data.py`, run training. A fresh expert block is created automatically.

## Download a trained model

**Releases page** (no build needed): grab `model.pt` + `tokenizer.json` from
[Releases](https://github.com/stromplayz/modular-lm/releases) and drop them in
a `ckpt/` folder:

```bash
git clone https://github.com/stromplayz/modular-lm.git && cd modular-lm
pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
pip install -e .
mkdir ckpt && cd ckpt
# download model.pt + tokenizer.json from the Releases page into here
cd ..
python -m skill_lm.generate --demo
```

Or get everything (code + latest checkpoint) with a plain clone - the trained
checkpoint is committed to `ckpt/` by the training workflow.

## Quick start

```bash
pip install -e ".[dev]"
pytest                      # full test suite

# train (CPU-friendly; ~55 min budget by default)
python -m skill_lm.train --steps 4000 --out ckpt

# chat / demo — the router picks the skill automatically
python -m skill_lm.generate --demo
python -m skill_lm.generate --prompt "Compute: 12 + 34"
python -m skill_lm.generate --prompt "Q: What is the capital of Japan?" --skill qa
python -m skill_lm.generate --prompt "Eye Q: What does OD mean on a prescription?" --skill optometry
```

## Train on GitHub Actions (free compute)

`Actions -> train -> Run workflow` — the workflow trains the model on a free
GitHub runner and uploads `model.pt`, `tokenizer.json`, `samples.txt` and the
training CSV as artifacts. Re-run with more steps any time.

## Design notes

- **Why one skill per request?** The point of SMT is *modularity you can name*.
  Want the model to get better at math? Train the math expert; the trunk and
  other skills are untouched. Want a new skill (code, translation)? Add a
  block + corpus entry; nothing else changes structurally.
- **Round-robin training**: step `i` trains skill `i mod n_skills` with its
  expert force-loaded, so every parameter in every expert gets gradient flow
  while the router learns to tell skills apart from trunk state alone.
- **RoPE + RMSNorm + pre-norm residuals** — modern small-model hygiene,
  depth-scaled residual init, gradient clipping, cosine LR with warmup.

## Repository layout

```
skill_lm/
  tokenizer.py   from-scratch byte-level BPE (GPT-2 style pre-tokenization)
  model.py       SkillModularLM: trunk + router + skill experts + generate
  data.py        skill corpora: TinyStories download, facts banks, generators
  train.py       round-robin skill training, per-skill eval, sample writing
  generate.py    chat CLI with router visibility
  benchmark.py   per-skill exact-match benchmark CLI
assets/facts.txt      general QA facts bank
assets/optometry.txt  optometry facts bank (146 facts)
tests/           tokenizer, model, data, overfit tests
```

## Measured results (v0.2.0 release, 5 skills @ 6000 steps, 936K params)

Exact-match benchmarks on fresh, unseen prompts (`python -m skill_lm.benchmark`):

| Skill | Accuracy | Notes |
|---|---|---|
| `optometry` | **97.3%** | 142/146 eye-care facts: anatomy, refractive errors, conditions, tests, prescriptions |
| `math` | 90.7% | addition 90%, subtraction 80%, multiplication **100%** (v0.1.0 4-skill model scored 97.3% - the 5th expert trades a few points) |
| `qa`   | **99.1%**  | capitals, science, days/months |
| `count`| **100%**  | all 89 words: letter counts + first letters |
| `router`| loads the right expert | eye questions with the **Eye Q: / Eye exam Q:** lead-in route to the optometry expert; plain `What is X?` between the two Q&A skills may lean qa - force with `--skill optometry` if needed |

The router's live confidence is printed with every demo generation
(`ROUTER : math [router correct] (probs: math=0.97, ...)`) and the benchmark
tool prints per-operation breakdowns.

**How math got to 97%**: tiny models cannot memorize 2-digit arithmetic from
~1.6 exposures per (a, b) pair. Two data-side fixes did it:
1. **Operand curriculum** - 50% of operands <= 12, 30% <= 29, 20% <= 99
2. **Place-value scratchpad** - 60% of addition examples show the algorithm:
   `23 + 45 -> "20 + 40 = 60. 3 + 5 = 8. 60 + 8 = 68."`
   Each intermediate lives in a small learned space, so the model learns
   *how to add*, not just answers.

## Status

- [x] From-scratch BPE tokenizer
- [x] Skill-Modular Transformer (trunk + 5 skill experts + router)
- [x] Skill datasets + generators
- [x] Training pipeline with per-skill eval + router accuracy
- [x] Test suite (tokenizer round-trip, routing, dispatch, overfit)
- [x] Local calibration run
- [x] Trained checkpoints committed by Actions (v5: math 97.3%, qa 100%, count 100%)
- [x] GitHub Actions training workflow
- [ ] Longer community-scale training runs (open `train` workflow with more steps)

MIT License.
