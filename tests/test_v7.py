"""Tests for FrozenCore v7 (Aurora): v2 gating, YaRN, SwiGLU, int4."""
import math

import pytest
import torch

from skill_lm.frozen import TrunkConfig, TrunkModel, SkillPack, SkillMixer
from skill_lm import v7


@pytest.fixture()
def trunk(tmp_path):
    cfg = TrunkConfig(vocab_size=256, dim=32, n_heads=4, block_size=64,
                      n_base_layers=1, expert_hidden=48, dropout=0.0)
    m = TrunkModel(cfg)
    path = tmp_path / "trunk.pt"
    m.save(str(path))
    return TrunkModel.load(str(path), frozen=True)


@pytest.fixture()
def packs(tmp_path, trunk):
    out = []
    for nm in ("alpha", "beta", "gamma"):
        p = SkillPack(nm, trunk.cfg)
        path = tmp_path / f"{nm}.pack"
        p.save(str(path))
        out.append(SkillPack.load(str(path)))
    return out


def test_v2_top2_gating(trunk, packs):
    m = v7.SkillMixerV2(trunk)
    for p in packs:
        m.add_pack(p)
    x = torch.randint(0, 256, (3, 10))
    out = m.forward(x)
    assert len(out["assigned"]) == 3
    for sel in out["assigned"]:
        assert len(sel) == 2                       # top-2 among 3 packs
        assert abs(sum(g for _, g in sel) - 1.0) < 1e-4   # normalized gates
        assert sel[0][1] >= sel[1][1]              # sorted by affinity


def test_v2_force_pack_matches_single_expert(trunk, packs):
    m = v7.SkillMixerV2(trunk)
    for p in packs:
        m.add_pack(p)
    x = torch.randint(0, 256, (2, 8))
    out = m.forward(x, force_pack="beta")
    assert all(sel == [("beta", 1.0)] for sel in out["assigned"])


def test_v2_rebalance_direction(trunk, packs):
    m = v7.SkillMixerV2(trunk)
    for p in packs:
        m.add_pack(p)
    before = dict(m.biases)
    m.rebalance({"alpha": 0.9, "beta": 0.05, "gamma": 0.05}, gamma=0.1)
    assert m.biases["alpha"] < before["alpha"]     # over-used -> bias down
    assert m.biases["beta"] > before["beta"]       # under-used -> bias up


def test_v1_mixer_still_works(trunk, packs):
    m = SkillMixer(trunk)
    for p in packs:
        m.add_pack(p)
    x = torch.randint(0, 256, (2, 8))
    out = m.forward(x)
    assert out["logits"].shape == (2, 8, 256)      # backward compatible


def test_yarn_shapes_and_finiteness():
    cos, sin = v7.precompute_rope_yarn(256, 16, 10000.0, factor=4.0)
    assert cos.shape == (1, 256, 1, 16)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
    # interpolation stretches positions: cos at pos t for factor>1 differs from base
    from skill_lm.model import _rope_cache
    c0, _ = _rope_cache(256, 16, 10000.0, "cpu", torch.float32)
    assert not torch.allclose(cos, c0)


def test_swiglu_expert(trunk):
    blk = v7.SwiGLUExpertBlock(trunk.cfg)
    x = torch.randn(2, 16, trunk.cfg.dim)
    y = blk(x, trunk.rope_cos[:, :16], trunk.rope_sin[:, :16])
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_int4_roundtrip(tmp_path, trunk):
    p = SkillPack("alpha", trunk.cfg)
    fp32 = tmp_path / "a.pack"
    p.save(str(fp32))
    q = tmp_path / "a.int4.pack"
    v7.export_int4(str(fp32), str(q), group=16)
    pk = v7.load_int4_pack(str(q))
    assert pk.name == "alpha"
    assert pk.n_params() == p.n_params()
    for k, v in p.state_dict().items():
        assert pk.state_dict()[k].shape == v.shape
    import os
    assert os.path.getsize(q) < os.path.getsize(fp32) / 2   # int4 < half of fp32
