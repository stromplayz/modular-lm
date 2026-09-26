# FrozenCore v7 "Aurora" - Architecture Study & Upgrade Design

> How the public state of the art (DeepSeek-V3, Qwen3, MiniCPM, MobileLLM,
> BitNet, YaRN) maps onto our Skill-Expert system, and what we adopted in
> code for v0.4.0.
>
> **Provenance note:** every idea below comes from public papers, open
> weights, or official open-source repos. No leaked or reverse-engineered
> proprietary source files were used or needed - the open designs are the
> strongest ones available anyway.

---

## 1. What we studied (public sources only)

| System | Key public ideas | Edge relevance to us |
|---|---|---|
| **DeepSeek-V3** (open weights + tech report) | Multi-head Latent Attention (MLA); auxiliary-loss-free MoE load balancing via per-expert bias updates; Multi-Token Prediction (MTP) objective; shared-expert + routed-expert MoE | Routing/gating policy, cheap aux losses, capacity economics |
| **DeepSeek-R1** (open weights) | RL-trained reasoning; distillation into small models | Pack distillation roadmap |
| **Qwen3** (open weights) | Thinking / non-thinking modes in one model | "Deep mode" packs (a reasoning pack behind the same router) |
| **MiniCPM / MiniCPM-V** (open weights) | Tiny core + on-demand capability modules (RAG-style retrieval of ability) | Exactly the Hub fetch/load/deload loop we already ship |
| **MobileLLM** (Meta, open) | Immediate blockwise weight reuse; embedding-layer tricks; narrow-deep beats wide-shallow at tiny scale | Our tied lm_head reuse across trunk + all packs is the same economics |
| **BitNet b1.58 / b1.58-2B4T** (Microsoft, open) | Ternary weights (BitLinear), native low-bit training | Pack quantization: int8 today, int4 research preview shipped, ternary QAT roadmap |
| **YaRN** (public paper) | NTK-by-parts + interpolation RoPE extension | Longer context per pack without touching the frozen trunk |
| **Phi-3-mini** (open) | Curriculum + data quality > parameter count | The 50-agent dataset forge is our version of this |

Two explicit non-goals: (1) we do **not** convert to token-level
Mixture-of-Experts - our experts are *Skill Experts* with semantic identity
(a router that knows "this is optometry"), which is what makes packs
independently trainable and hot-swappable; (2) we do not use any
non-public model files.

## 2. What FrozenCore already had right (v0.3.0)

- **Frozen trunk (361,088 params)** - leaf-frozen + `no_grad` contract;
  gradients physically cannot reach the main model. Zero-forgetting is
  structural, not aspirational (bench-verified bit-identical scores).
- **Skill packs as files** (`packs/<name>.pack`) - expert block + one-vs-rest
  router head; any subset co-loads; softmax routing among loaded packs only.
- **GitHub as the pack hub** - raw-URL fetch, sha fingerprint, disk cache,
  deload after use (MiniCPM-style ability-on-demand).
- **Huge-dataset streaming** - HTTP-Range window slicing (`stream.py`) and
  datasets-server page sampling (`hfdata.py`): a 2 GB/18 M-row dataset
  costs kilobytes of traffic for one pack's data.

## 3. v7 "Aurora" upgrades - shipped in v0.4.0 code

### 3.1 v2 routing: sigmoid affinity + top-2 soft gating + balance bias
*Adapted from DeepSeek-V3's auxiliary-loss-free routing.*

- Every pack keeps its one-vs-rest head (no file changes; old packs load).
- `affinity_i = sigmoid(head_i(h) + bias_i)`; the **top-2** packs run per
  input and their outputs blend by normalized gates; the trunk acts as the
  always-on shared expert.
- `SkillMixerV2.rebalance(usage, gamma)` nudges `bias_i` toward the mean
  usage - under-used packs get more likely, over-used less - with **zero
  gradient pressure and zero weight changes**, so balance is a runtime
  policy, not a training compromise.
- File: `skill_lm/v7.py` (`SkillMixerV2`), tests in `tests/test_v7.py`.

### 3.2 MTP-lite: multi-token prediction as a free aux loss
*Adapted from DeepSeek-V3's MTP objective.*

- During pack training, the same trunk lm_head also predicts token **t+2**
  from position t: `loss += λ * CE(logits[:, :-2], y[:, 2:])`, λ=0.15.
- Zero new parameters, zero inference cost, better expert representations.
- Flag: `train_pack --mtp 0.15`.

### 3.3 Distinctive per-pack QA surfaces (routing at the surface level)
The v5 genesis skills each own a lead-in (`Q:`, `Eye Q:`, `Wiki Q:` ...).
v0.4.0 makes that a first-class training knob again:

- `train_pack --lead 'Eye Q: {q}\nA: {a}.'` - optometry owns `Eye Q:`,
  grammar owns `Lang Q:`; surface collision with qa/knowledge is designed
  away instead of calibrated away.
- Empirically: single dominant surface beats 4-way surface splits on both
  recall and routing (see §5 benchmarks).

### 3.4 YaRN-style context extension for packs
`v7.precompute_rope_yarn(seq_len, head_dim, base, factor)` implements the
public NTK-by-parts blend (low-frequency dims stretched by `factor`,
high-frequency kept, smooth ramp). `v7.run_expert_long(pack, h, factor)`
runs any existing pack over windows longer than the trunk block size -
the frozen trunk never changes.

### 3.5 SwiGLU expert (pack format v2 preview)
`v7.SwiGLUExpertBlock` - DeepSeek/Qwen-style 3-matrix SwiGLU FFN sized for
parameter parity with the v1 GELU MLP. v1 packs stay fully supported;
future genesis can mint v2 packs.

### 3.6 Low-bit pack distribution
- int8 per-pack export (`export_pack.py`): 453 KB -> 124 KB.
- **int4 groupwise research preview** (`v7.export_int4` /
  `v7.load_int4_pack`): 453 KB -> **68 KB** per pack, dequant-on-load,
  round-trip verified to the parameter. BitNet-style capacity/byte economics
  without native QAT (ternary QAT = roadmap).

## 4. The architecture at a glance (v0.4.0)

```
                       ┌────────────────────────────────┐
   GitHub = pack hub   │  packs/*.pack (453 KB fp32 /   │
   raw URLs, no auth   │  124 KB int8 / 68 KB int4)     │
                       └───────────┬────────────────────┘
                     fetch on demand │  cache -> load -> use -> DELOAD
                       ┌─────────────▼──────────────┐
   edge runtime        │ SkillMixer (v1) / V2       │
                       │  trunk 361,088 FROZEN      │  <- never trains again
                       │  + N co-loaded packs       │
                       │  route: softmax (v1) or    │
                       │  sigmoid top-2 + bias (v2) │
                       └────────────────────────────┘
   data side           forge.py (50 agents) · hfdata.py (HF datasets-server
                       pages out of 18M-row corpora) · stream.py (Range
                       windows into 40GB+ files) · ingest.py (Wikipedia)
```

## 5. Honest benchmark position (LUA, v0.4.0)

| Suite | v0.3.0 | v0.4.0 | Note |
|---|---|---|---|
| qa | 100% | **100%** | unchanged (zero-forgetting) |
| count | 100% | **100%** | unchanged |
| math | 50.7% | 50.7% | unchanged (math pack v2 = roadmap) |
| knowledge recall | 100% | **100%** | unchanged |
| optometry (v1 bank, 146 facts) | 100% | superseded | pack now covers **19x** more |
| optometry CORE exam recall (555 facts) | - | **~32-40%** | 115K-param expert, ~12 surface exposures/fact |
| optometry held-out (never-seen facts) | 0% (at 115K scale) | **14-28%** | first generalization signal at this scale |
| grammar/vocab (new pack) | - | **72-78%** bank recall | 230 vocab words + 17 rule families + 3,500 real C4_200m pairs as LM text |
| router (7 packs co-loaded) | 96.1% (6 packs) | **82-85%** | 7th pack + overlapping vocab/knowledge surfaces; v2 bias rebalancing is the lever |
| zero-forgetting | verified | **verified** | old suites bit-identical |

**The capacity law we measured** (worth stating honestly): exact recall per
fact needs roughly ~100 surface exposures at the 115K-param expert scale
(154 facts x 104 exposures -> 100%; 555 core facts x ~12 -> ~35%). The
architecture answer is not "train harder" - it is the FrozenCore dial:
**expert width per pack** (dim 256 expert ≈ 450 KB fp32 / 120 KB int8)
gets core recall past 80% while the system stays edge-sized. That is the
v0.5 headline change.

## 6. Roadmap (ranked by expected LUA gain)

1. **Wider experts for new packs** (dim 256/384, SwiGLU v2 format) -
   direct capacity fix, packs still <0.5 MB.
2. **v2 gating + bias rebalancing as default runtime** (`serve.py --v2`) -
   recover router headroom with zero retraining.
3. **Math pack v2** - wider expert + curriculum; the FrozenCore pitch
   ("retrain one pack, main model untouched") proven on the hardest suite.
4. **Ternary QAT packs** (BitNet-style) - 34 KB packs.
5. **Distillation from a bigger teacher** (DeepSeek-R1-distill class) into
   a reasoning pack - Qwen3 "thinking mode" behind our router.
