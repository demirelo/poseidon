"""
poseidon_attack -- independent, verifier-exact reimplementation of the
Poseidon1 / KoalaBear primitive and the four 2026-bounty predicates.

Built from the written spec and validated against the published golden test
vectors -- not by executing the reference repo. This is the shared substrate
used by the paper's attack analyses.
"""

from .poseidon1 import Poseidon1, grain_round_constants, cauchy_mds, circulant_mds
from . import constants, field, verifiers

__all__ = [
    "Poseidon1",
    "grain_round_constants",
    "cauchy_mds",
    "circulant_mds",
    "constants",
    "field",
    "verifiers",
]
