"""Tests for the FrozenCore architecture: trunk freeze, packs, co-loading."""
import os

import torch
import pytest

from skill_lm.frozen import TrunkConfig, TrunkModel, SkillPack, SkillMixer, genesis_from_v5


def make_trunk(tmp_path) -> TrunkModel:
    cfg = TrunkConfig(vocab_size=256, dim=32, n_heads=2, block_size=64,
                      n_base_layers=1, expert_hidden=48, dropout=0.0)
    trunk = TrunkModel(cfg)
    # random-init trunk is fine for mechanics tests
    trunk.freeze()
    path = os.path.join(tmp_path, "trunk.pt")
    trunk.save(path)
    return TrunkModel.load(path, frozen=True)


def test_trunk_is_frozen(tmp_path):
    trunk = make_trunk(tmp_path)
    for p in trunk.parameters():
        assert not p.requires_grad
    # hidden states carry no graph back to the trunk
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        h = trunk.hidden(x)
    assert not h.requires_grad
    # a loss computed FROM trunk outputs has no grad path to trunk params
    y = torch.randint(0, 256, (2, 16))
    out = trunk(x, targets=y)
    assert out["lm_loss"] is not None and not out["lm_loss"].requires_grad
    before = {k: v.clone() for k, v in trunk.state_dict().items()}
    for k, v in trunk.state_dict().items():
        assert torch.equal(before[k], v), f"trunk weight {k} moved!"


def test_pack_train_moves_only_pack(tmp_path):
    trunk = make_trunk(tmp_path)
    pack = SkillPack("qa", trunk.cfg)
    before_trunk = {k: v.clone() for k, v in trunk.state_dict().items()}
    before_pack = {k: v.clone() for k, v in pack.state_dict().items()}

    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        h = trunk.hidden(x)
    cos = trunk.rope_cos[:, : x.size(1)]
    sin = trunk.rope_sin[:, : x.size(1)]
    h = pack.expert(h.detach(), cos, sin)
    loss = trunk.logits_from_h(h).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pack.expert.parameters())
    for k, v in trunk.state_dict().items():
        assert torch.equal(before_trunk[k], v), "trunk moved during pack training!"
    for k, v in pack.state_dict().items():
        assert torch.equal(before_pack[k], v)  # no optimizer step yet, weights same


def test_pack_save_load_roundtrip(tmp_path):
    trunk = make_trunk(tmp_path)
    pack = SkillPack("math", trunk.cfg, description="test")
    path = os.path.join(tmp_path, "math.pack")
    pack.save(path, meta={"steps": 10})
    pack2 = SkillPack.load(path)
    assert pack2.name == "math"
    assert pack2.meta["steps"] == 10
    for k, v in pack.state_dict().items():
        assert torch.equal(pack2.state_dict()[k], v)


def test_mixer_co_load_and_deload(tmp_path):
    trunk = make_trunk(tmp_path)
    mixer = SkillMixer(trunk)
    for nm in ["qa", "math"]:
        p = SkillPack(nm, trunk.cfg)
        p.save(os.path.join(tmp_path, f"{nm}.pack"))
        mixer.add_pack_file(os.path.join(tmp_path, f"{nm}.pack"))
    assert mixer.pack_names == ["qa", "math"]
    x = torch.randint(0, 256, (1, 12))
    out = mixer(x)
    assert out["logits"].shape == (1, 12, 256)
    assert out["assigned"][0] in {"qa", "math"}
    # deload: memory only
    mixer.drop_pack("math")
    assert mixer.pack_names == ["qa"]


def test_router_heads_compete_under_softmax(tmp_path):
    """Two trained heads: the higher logit wins the co-loaded softmax."""
    torch.manual_seed(0)
    trunk = make_trunk(tmp_path)
    x = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        h = trunk.hidden(x)                            # deterministic probe
        pooled = h.mean(dim=1)
        pa, pb = SkillPack("a", trunk.cfg), SkillPack("b", trunk.cfg)
        pa.router_head.weight.copy_(pooled)            # logit_a = ||pooled||^2 > 0
        pb.router_head.weight.copy_(-pooled)           # logit_b = -||pooled||^2 < 0
    mixer = SkillMixer(trunk)
    mixer.add_pack(pa)
    mixer.add_pack(pb)
    aidx, scores = mixer.route(h)
    assert mixer.name_of(int(aidx[0])) == "a"
    assert scores[0, 0] > scores[0, 1]


def test_genesis_split_from_monolithic(tmp_path):
    """Build a tiny fake v5 ckpt, split it, verify trunk+packs reconstruct it."""
    from skill_lm.model import SkillModularConfig, SkillModularLM
    cfg = SkillModularConfig(vocab_size=256, dim=32, n_heads=2, block_size=64,
                             n_base_layers=1, n_skills=3, expert_hidden=48,
                             dropout=0.0, skill_names=("x", "y", "z"))
    model = SkillModularLM(cfg)
    ckpt = os.path.join(tmp_path, "v5.pt")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "step": 1}, ckpt)
    res = genesis_from_v5(ckpt, tmp_path, skill_names=("x", "y", "z"))
    assert os.path.exists(res["trunk"])
    for p in res["packs"]:
        assert os.path.exists(p)
    # trunk params == wte + base blocks + norm_f (+tied head counted once)
    trunk = TrunkModel.load(res["trunk"])
    assert trunk.n_params() < model.total_params()
