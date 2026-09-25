import torch
import pytest

from skill_lm.model import SkillModularConfig, SkillModularLM

TINY = SkillModularConfig(
    vocab_size=300, dim=32, n_heads=4, block_size=32,
    n_base_layers=1, n_skills=3, expert_hidden=48, dropout=0.0,
)


def test_forward_shapes_and_losses():
    torch.manual_seed(0)
    model = SkillModularLM(TINY)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    y = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    label = torch.tensor([0, 1, 2, 0])
    out = model(x, targets=y, skill_label=label, force_route=True)
    assert out["logits"].shape == (4, TINY.block_size, TINY.vocab_size)
    assert torch.isfinite(out["lm_loss"])
    assert torch.isfinite(out["router_loss"])
    assert (out["assigned"] == label).all()


def test_skill_blocks_load_selectively():
    """THE core behavior: each example loads ONLY its own skill block."""
    torch.manual_seed(0)
    model = SkillModularLM(TINY)
    model.eval()
    calls = [0, 0, 0]

    hooks = []
    for i, blk in enumerate(model.skill_blocks):
        def make_hook(idx):
            def hook(module, args, output):
                calls[idx] += 1
            return hook
        hooks.append(blk.register_forward_hook(make_hook(i)))

    x = torch.randint(0, TINY.vocab_size, (3, 16))
    # force: example 0 -> skill 0, example 1 -> skill 1, example 2 -> skill 2
    label = torch.tensor([0, 1, 2])
    with torch.no_grad():
        model(x, skill_label=label, force_route=True)
    assert calls == [1, 1, 1], f"each skill block should run exactly once, got {calls}"

    # auto routing: one chosen skill for the batch prompt
    calls[:] = [0, 0, 0]
    with torch.no_grad():
        model(x[:1], force_route=False)
    assert sum(calls) == 1 and calls.count(1) == 1, f"exactly one skill should load, got {calls}"
    for h in hooks:
        h.remove()


def test_auto_routing_matches_logits():
    torch.manual_seed(0)
    model = SkillModularLM(TINY)
    model.eval()
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(x)
    expected = out["skill_logits"].argmax(-1)
    assert (out["assigned"] == expected).all()


def test_backward_flows_to_skill_and_router():
    torch.manual_seed(0)
    model = SkillModularLM(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    y = torch.randint(0, TINY.vocab_size, (2, 16))
    label = torch.tensor([1, 1])
    out = model(x, targets=y, skill_label=label, force_route=True)
    (out["lm_loss"] + out["router_loss"]).backward()
    assert model.router.weight.grad is not None
    assert model.skill_blocks[1].attn.qkv.weight.grad is not None
    assert torch.isfinite(model.skill_blocks[1].attn.qkv.weight.grad).all()


def test_param_counts():
    model = SkillModularLM(TINY)
    total = model.total_params()
    active = model.active_params()
    assert total > active > 0
    # 3 skill blocks but only one is active per request
    skills_total = sum(p.numel() for p in model.skill_blocks.parameters())
    assert active + 2 * sum(p.numel() for p in model.skill_blocks[0].parameters()) \
        == total  # total = active + the 2 idle skills


def test_generation_runs_and_detects_skill():
    torch.manual_seed(0)
    model = SkillModularLM(TINY)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    out, skill = model.generate(ids, max_new_tokens=8, temperature=1.0, top_k=10)
    assert out.shape[1] == ids.shape[1] + 8
    assert 0 <= skill < TINY.n_skills
    out_g, _ = model.generate(ids, max_new_tokens=5, greedy=True)
    assert out_g.shape[1] == ids.shape[1] + 5
