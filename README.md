# Skill-Modular Transformer (SMT) — FrozenCore Edition

A **from-scratch** tiny language model with loadable **SKILL PACKS**: a tiny
frozen main model routes your input, **fetches on demand** the specialist
submodel that owns that kind of task, **uses** it, and **deloads** it.

> This is **not** mixture-of-experts. MoE shards every token across anonymous
> FFN slices. SMT has *dedicated skills* — a math pack, a QA pack, a knowledge
> pack — each a full transformer block trained to **own** its task, stored as
> an independent file, loaded only when needed.

No transformer library. No pretrained weights. No tokenizer package.
The BPE tokenizer, the model, the router, the training loop — all hand-built
in this repo, trained on free CPU compute (GitHub Actions).

---

## v0.4.0 — 50-agent dataset forge + grammar/vocab pack + v7 "Aurora"

**What's new** (see `docs/ARCHITECTURE.md` for the full study behind it):

- **50-agent dataset forge** (`skill_lm/forge.py`) — 401 curated optometry
  seed facts (anatomy, optics, refraction, disease, pharma, contacts,
  pediatrics/BV, low vision) expanded by 50 parallel agents into
  **2,915 train QA + 354 held-out + 401 LM statements** (canonical,
  paraphrase, MCQ with domain distractors, true/false, cloze). The same
  forge runs as a **50-job GitHub Actions matrix** in
  `dataset-forge.yml` — each runner is one agent, artifacts merge, packs
  train on the runners, the hub commits back.
- **Real huge-dataset loading** (`skill_lm/hfdata.py`) — 3,500 grammar
  correction pairs sampled out of `liweili/c4_200m` (**18.28M rows, GBs**)
  via the HF datasets-server Rows API; only ~50 pages of 100 rows ever
  moved. Feeds the grammar pack as LM text.
- **New `grammar` pack** — 230 advanced vocabulary words (definition /
  synonym / reverse / MCQ) + 17 programmatic grammar-rule families
  (agreement, irregular past, comparatives, much/many, your/you're,
  their/there/they're, then/than, fewer/less, prepositions, ...) +
  dictionary-style LM lines. Distinctive `Lang Q:` surface.
- **Optometry pack v2** — retrained on the forge bank with a distinctive
  `Eye Q:` surface, MTP-lite aux loss (DeepSeek-V3-style, zero params),
  joint router calibration (others frozen). Coverage **19x** the v1 bank;
  first-ever held-out generalization at this scale.
- **v7 "Aurora" module** (`skill_lm/v7.py`, public-SOTA-derived):
  `SkillMixerV2` sigmoid top-2 gating + auxiliary-loss-free balance bias;
  YaRN-style RoPE extension for packs; SwiGLU expert (pack format v2
  preview); **int4 groupwise pack export** (453 KB -> **68 KB**, verified
  round-trip). All provenance is public papers/open weights — no leaked or
  proprietary files.
- **Chunk-resumable pack training** (`train_pack --chunk-minutes/--resume`)
  — optimizer + step state checkpointed every 500 steps; long runs survive
  anywhere (Actions, Codespaces, a phone via Termux).

LUA benchmark (7 packs co-loaded, zero-forgetting verified):
qa 100% · count 100% · knowledge recall 100% · math 50.7% ·
optometry core ~32-40% (19x bank, honest capacity math in the doc) ·
grammar 72-78% · router 82-85%.

---

## v0.3.0 — FrozenCore architecture

**The main model never trains again.** All learning happens in pack files.

```
┌─────────────────────────────────────────────────────────────┐
│  TRUNK (FROZEN FOREVER) — wte + 2 blocks + tied lm_head     │
│  the "main model": 361K params, 1.4 MB, zero future gradients│
└───────────────┬─────────────────────────────────────────────┘
                │ pooled hidden state (detached, no_grad)
    ┌───────────┴──────────────────────────────────┐
    │  SKILL PACKS (independently trainable files)  │
    │  packs/<name>.pack = expert block + 1-vs-rest │
    │  router head ("is this MY kind of input?")    │
    └───────────┬──────────────────────────────────┘
                │ softmax over LOADED packs only  ← CO-LOADING
                ▼
     chosen pack's expert → frozen lm_head → tokens
```

| Property | How it is guaranteed |
|---|---|
| **Main model never retrains** | trunk params are leaf-frozen and run under `no_grad`; gradients physically cannot reach it |
| **Train only one submodel** | training a pack rewrites exactly one file (`packs/<name>.pack`); trunk + other packs are never opened for writing |
| **Load packs together** | one-vs-rest router heads: any subset co-loads; routing = softmax across whoever is loaded |
| **Fetch from GitHub on demand** | the repo IS the hub (`packs/`); `hub.py` lists/fetches/caches packs from raw.githubusercontent.com |
| **Deload after use** | `mixer.drop_pack(name)` drops the expert from RAM; only the disk cache remains |
| **Zero forgetting** | old pack files are immutable → old benchmark scores are bit-identical after any new pack training (verified by `bench_lua`) |

### Why this beats one big model (for small scale)

- **Surgical upgrades**: worse at law? train `law.pack` 10 minutes. A monolith
  would retrain end-to-end and risk every other skill.
- **Distribution**: a pack is a single ~460 KB file (124 KB int8). Host packs
  anywhere — the hub resolver needs one URL.
- **Compose at runtime**: conversation about eyes + math? load 2 packs (~0.9 MB
  RAM); the other 4 stay on disk.
- **40 GB datasets, no 40 GB downloads**: `stream.py` cuts small random windows
  straight out of remote HF parquet/txt via HTTP Range requests.

### v0.3.0 LUA benchmark (Language Understanding & Answering)

All six packs co-loaded, routing among them (`python -m skill_lm.bench_lua`):

| Section | Score | Notes |
|---|---|---|
| `qa` (111 facts) | **100%** | factoid answering |
| `optometry` (146 facts) | **100%** | eye-care domain |
| `knowledge` bank recall | **100%** | absorbed wiki/seed fact bank (154 facts) |
| `count` (89 words) | **100%** | letter counting |
| `math` | 50.7% | uniform 0-99 operands, exact match — the inherited v5 math expert scores 41.3% on the same eval; its curriculum favored small operands. `math` pack v2 (retrain with wider operand coverage via `train_pack`) is on the roadmap — **this is exactly the FrozenCore pitch: upgrade one pack, touch nothing else** |
| `router` (co-loaded, 180 probes) | **96%** | 6-way routing among live packs |
| `knowledge` held-out | 4% | honest limit: a 115K-param expert memorizes its bank; *generalization* needs the bigger trunks on the roadmap |

`forgetting: ZERO-FORGETTING VERIFIED` — the bench diffs against the recorded
baseline and proves old skills did not move when the knowledge pack trained.

---

## The packs

| Pack | Corpus | Source |
|---|---|---|
| `story` | fluent simple English | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) slice (range-downloaded) |
| `qa` | factual Q -> A | bundled facts bank (`assets/facts.txt`) |
| `math` | exact arithmetic (+, -, x) | generated on the fly (infinite) |
| `count` | letter counting, first-letter | generated from word list |
| `optometry` | eye-care domain facts | bundled bank (`assets/optometry.txt`, 146 facts) |
| `knowledge` | general world knowledge | `assets/knowledge/` banks: offline seed + **Wikipedia crawl** (via Actions) |

### Add a new skill WITHOUT touching the main model

```bash
# 1. facts bank (Q ||| A lines) - write it, or crawl wiki into it
python -m skill_lm.ingest --offline          # or Actions: ingest-wiki workflow

# 2. train ONE pack against the frozen trunk (~7 min CPU)
python -m skill_lm.train_pack --name myskill --facts assets/myskill.txt

# 3. use it immediately, co-loaded with the others
python -m skill_lm.serve --packs qa myskill --ask "..."
```

Nothing else changes. No retraining. No forgetting. Push the pack file and
everyone can fetch it from the hub.

## Use the runtime (fetch-on-demand + deload)

```bash
python -m skill_lm.serve --demo                 # co-load all packs, demo each
python -m skill_lm.serve --ask "What is the largest ocean on Earth?"
python -m skill_lm.serve --repl                 # interactive
#   ask> :load optometry      <- fetch + load from GitHub hub on demand
#   ask> :deload optometry    <- drop from RAM
```

Browse the hub:

```bash
python -m skill_lm.hub --list                   # remote + cached packs
python -m skill_lm.hub --fetch knowledge        # cache a pack for offline use
```

Programmatic:

```python
from skill_lm.hub import Hub
hub = Hub("stromplayz/modular-lm")
mixer = hub.load_mixer("ckpt/trunk.pt", ["qa", "knowledge"])
out, used_pack = mixer.generate(ids, pack=None)  # auto-route among loaded
mixer.drop_pack("knowledge")                     # deload
```

## Quick start (legacy monolith)

```bash
pip install -e ".[dev]"
pytest                      # full test suite (31 tests)

# v6 genesis: split the v5 checkpoint into trunk + packs (one-time)
python -m skill_lm.genesis --story-mb 3.0

# train a pack
python -m skill_lm.train_pack --name knowledge \
    --facts assets/knowledge/wiki_facts.txt \
    --eval-facts assets/knowledge/wiki_facts_eval.txt

# LUA benchmark with zero-forgetting check
python -m skill_lm.bench_lua

# int8 export (~3.7x smaller packs)
python -m skill_lm.export_pack --all
```

## Train on GitHub Actions (free compute)

| Workflow | What it does |
|---|---|
| `train` | legacy monolith training |
| `ingest-wiki` | crawls Wikipedia from clean runner IPs, mines QA facts, commits the banks |
| `train-pack` | trains ONE pack against the frozen trunk, runs the LUA benchmark, commits the pack to the hub |

Typical loop: run `ingest-wiki` (grows the fact banks) -> run `train-pack`
(knowledge pack absorbs the new facts) -> pack lands in `packs/` -> the hub
serves it to every client. The main model was trained once and stays frozen.

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
  model.py       SkillModularLM (v5 monolith: trunk + router + experts)
  frozen.py      v6 FrozenCore: TrunkModel (frozen) + SkillPack + SkillMixer
  genesis.py     one-time split: v5 ckpt -> trunk + packs + router heads
  train_pack.py  train ONE pack against the frozen trunk (LM + router)
  hub.py         GitHub pack hub: list / fetch / cache / load / deload
  stream.py      HTTP-Range slicer: sample MBs out of 40GB+ datasets
  ingest.py      Wikipedia crawler + regex QA miner -> fact banks
  serve.py       runtime CLI: co-load packs, ask, :load/:deload
  bench_lua.py   Language Understanding & Answering bench + forgetting check
  export_pack.py int8 pack export (~3.7x smaller)
  data.py / train.py / generate.py / benchmark.py   (v5 pipeline)
packs/           THE HUB - one file per skill pack (fetchable by URL)
ckpt/            trunk.pt (frozen) + v5 monolith ckpt + tokenizer + baselines
assets/          fact banks (facts, optometry, knowledge/)
.devcontainer/   one-click Codespaces: installs torch, runs tests
tests/           31 tests: tokenizer, model, frozen core, packs, ingestion
```

## Status

- [x] From-scratch BPE tokenizer
- [x] Skill-Modular Transformer (trunk + 5 skill experts + router)
- [x] **v6 FrozenCore: frozen trunk + 6 independently-trained packs**
- [x] One-vs-rest router heads + joint calibration for co-loading
- [x] GitHub pack hub (fetch on demand, cache, deload)
- [x] Wikipedia ingestion pipeline (+ Actions workflow with clean IPs)
- [x] Stream slicing for 40GB+ HF datasets (no full downloads)
- [x] LUA benchmark with zero-forgetting verification
- [x] int8 pack export (464 KB -> 124 KB per pack)
- [x] 31-test suite incl. freeze/forgetting guarantees
- [ ] Scale roadmap: wider trunk + more packs; int4; distillation (see bench_lua roadmap note)

MIT License.
