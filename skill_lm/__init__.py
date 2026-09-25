"""Skill-Modular Language Model (SMT) - from scratch.

One tiny shared brain + a chest of Skill Experts. A Skill Router reads the
input, LOADS the matching skill block, uses it, DETACHES it.
"""
__version__ = "0.1.0"

from .model import SkillModularLM, SkillModularConfig
from .tokenizer import BPETokenizer
from .data import SKILL_NAMES

__all__ = ["SkillModularLM", "SkillModularConfig", "BPETokenizer", "SKILL_NAMES", "__version__"]
