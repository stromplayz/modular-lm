"""Overfit sanity: the math skill expert must nail its task after a short run."""
import torch

from skill_lm import data as D
from skill_lm.model import SkillModularConfig, SkillModularLM
from skill_lm.tokenizer import BPETokenizer


def test_math_skill_overfits():
    torch.manual_seed(0)
    import random
    rng = random.Random(7)

    vocab_text = " ".join(D.gen_math(rng) for _ in range(400)) + " " + \
        " ".join(D.gen_count(rng) for _ in range(200))
    tok = BPETokenizer.train(vocab_text, vocab_size=300)

    cfg = SkillModularConfig(
        vocab_size=tok.vocab_size, dim=48, n_heads=4, block_size=48,
        n_base_layers=1, n_skills=2, expert_hidden=64, dropout=0.0,
        skill_names=("qa", "math"),
    )
    model = SkillModularLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    examples = [D.gen_math(rng) + "\n\n" for _ in range(60)]
    ids = torch.tensor([i for ex in examples for i in tok.encode(ex)], dtype=torch.long)
    sid = D.SKILL_IDS["math"] % 2  # 2-skill config -> math maps to skill 1
    block = 48

    first = None
    for step in range(90):
        i = torch.randint(len(ids) - block - 1, (8,))
        x = torch.stack([ids[j: j + block] for j in i])
        y = torch.stack([ids[j + 1: j + 1 + block] for j in i])
        label = torch.full((8,), sid, dtype=torch.long)
        out = model(x, targets=y, skill_label=label, force_route=True)
        loss = out["lm_loss"] + out["router_loss"]
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < first * 0.45, (
        f"math skill should overfit: first {first:.3f} -> last {loss.item():.3f}"
    )
