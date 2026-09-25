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

## The four skills

| Skill | Corpus | Source |
|---|---|---|
| `story` | fluent simple English | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) slice (range-downloaded) |
| `qa` | factual Q -> A | bundled facts bank (`assets/facts.txt`) |
| `math` | exact arithmetic (+, -, x) | generated on the fly (infinite) |
| `count` | letter counting, first-letter | generated from word list |

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
- **Round-robin training**: step `i` trains skill `i mod 4` with its expert
  force-loaded, so every parameter in every expert gets gradient flow while
  the router learns to tell skills apart from trunk state alone.
- **RoPE + RMSNorm + pre-norm residuals** — modern small-model hygiene,
  depth-scaled residual init, gradient clipping, cosine LR with warmup.

## Repository layout

```
skill_lm/
  tokenizer.py   from-scratch byte-level BPE (GPT-2 style pre-tokenization)
  model.py       SkillModularLM: trunk + router + skill experts + generate
  data.py        skill corpora: TinyStories download, facts bank, generators
  train.py       round-robin skill training, per-skill eval, sample writing
  generate.py    chat CLI with router visibility
assets/facts.txt QA facts bank
tests/           tokenizer, model, data, overfit tests
```

## Status

- [x] From-scratch BPE tokenizer
- [x] Skill-Modular Transformer (trunk + 4 skill experts + router)
- [x] Skill datasets + generators
- [x] Training pipeline with per-skill eval + router accuracy
- [x] Test suite (tokenizer round-trip, routing, dispatch, overfit)
- [x] Local calibration run
- [x] GitHub Actions training workflow
- [ ] Longer community-scale training runs (open `train` workflow with more steps)

MIT License.
