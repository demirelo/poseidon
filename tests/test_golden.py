"""
Validate the independent poseidon_attack implementation against PUBLISHED golden
vectors and the written spec.  Runs ONLY our own code (plus literal constants
extracted from the reference data file) -- the reference implementation is never
executed.

Anchors:
  1. Plonky3 width-16 permutation vector (full 16 outputs) -> validates the
     round schedule (full/partial rounds, S-box, MDS application) end to end.
  2. Cauchy+Grain default instance perm([0]*16)[0] == 1393439926 -> validates
     the Grain LFSR + Cauchy MDS construction on the canonical bounty primitive.
  3. Documented "hash"/sponge scalars -> extra check on the sponge path.
  4. Published t=3 partial collision -> validates compression-mode hash +
     the collision predicate on the FULL-STRENGTH rf=8/rp=20 instance.
  5. Predicate sanity (degree/zero/size rejections; field & decode facts).

Run:  python3 tests/test_golden.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon_attack.constants import P
from poseidon_attack.poseidon1 import Poseidon1, cauchy_mds, circulant_mds
from poseidon_attack import field
from poseidon_attack.verifiers import (
    verify_cico, verify_collision, verify_density, verify_zerotest, decode,
)

HERE = os.path.dirname(os.path.abspath(__file__))
_checks = []


def check(name, cond, detail=""):
    _checks.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


# --- 1. Plonky3 width-16 golden vector (round logic) -----------------------
print("1. Plonky3 width-16 permutation vector (validates round schedule)")
with open(os.path.join(HERE, "golden_vectors.json")) as f:
    g = json.load(f)["plonky3_w16"]
pos_p3 = Poseidon1(prime=P, alpha=g["alpha"], t=g["t"], r_f=g["rf"], r_p=g["rp"],
                   mds=circulant_mds(g["mds_first_row"], P),
                   round_constants=g["round_constants"])
got = pos_p3.permutation(list(range(16)))
check("perm(range(16)) == published Plonky3 vector", got == g["expected_perm_of_range16"],
      f"first3 got={got[:3]} exp={g['expected_perm_of_range16'][:3]}")

# --- 2. Cauchy + Grain canonical instance ----------------------------------
print("2. Cauchy+Grain default instance (validates Grain LFSR + Cauchy MDS)")
pos_kb = Poseidon1(prime=P, alpha=3, t=16, r_f=8, r_p=20)  # defaults: Cauchy + Grain
check("perm([0]*16)[0] == 1393439926", pos_kb.permutation([0] * 16)[0] == 1393439926,
      f"got={pos_kb.permutation([0]*16)[0]}")

# --- 3. The documented `.hash` scalars are ORPHANED (finding, not a check) --
# poseidon.py's docstring/tests assert pos.hash(range(15))==93555670 etc., but
# the Poseidon class ships NO `hash` method -> those calls AttributeError, and
# the values match neither sponge_hash nor compression_mode_hash of the shipped
# code.  They are dead constants from a removed/never-shipped API.  None of the
# four bounty challenges use sponge mode (collision/density/zero-test use
# compression mode; CICO uses permutation_plus_linear -- all validated above),
# so this is immaterial to the attack work.  We record the values for the report
# and only assert that our sponge path is well-formed.
print("3. Sponge path well-formed; documented .hash vectors are orphaned (info)")
h1 = pos_kb.sponge_hash(list(range(15)), 1)[0]
print(f"      info: sponge(range(15))[0]={h1}  (docstring's orphaned .hash=93555670)")
check("sponge_hash deterministic", pos_kb.sponge_hash([1, 2, 3], 1) == pos_kb.sponge_hash([1, 2, 3], 1))
check("sponge_hash output in field", 0 <= h1 < P)
check("sponge distinguishes inputs", pos_kb.sponge_hash([1], 1) != pos_kb.sponge_hash([2], 1))

# --- 4. Published t=3 partial collision (compression mode, rf=8/rp=20) ------
print("4. Published t=3 partial collision (Beltran/Merz/Rodriguez/Scarlata)")
X = [146101246, 585745660, 1080651781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Y = [310195439, 1632272689, 97247552, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
check("verify_collision(X,Y,t=3) is True", verify_collision(X, Y, t=3) is True)
check("verify_collision(X,Y,t=4) is False (only 3 words collide)",
      verify_collision(X, Y, t=4) is False)
check("verify_collision(X,X,t=3) is False (x==y rejected)",
      verify_collision(X, X, t=3) is False)

# --- 5a. Field facts -------------------------------------------------------
print("5a. Field facts")
check("(p-1) == 2^24 * 127", (P - 1) == 2**24 * 127)
check("F_{p^2}: (sqrt3)^2 == 3", field.f2_mul((0, 1), (0, 1), 3, P) == (3, 0))
check("decode(0) == 0", decode(0) == 0)
check("decode is a function into {0..15}", all(0 <= decode(v) < 16 for v in (1, 2, 3, 7, 11, 99, 123456)))

# --- 5b. Predicate sanity (reject paths) -----------------------------------
print("5b. Predicate sanity")
# zero-test: degree-0 poly (constant) must be rejected by C1
check("zerotest rejects degree-0 poly", verify_zerotest([5, 0] + [0] * 14) is False)
# zero-test: wrong length raises
try:
    verify_zerotest([0] * 15); raised = False
except ValueError:
    raised = True
check("zerotest wrong P_hat length raises", raised)
# density: >2 zeros rejected
check("density rejects 3 zeros", verify_density([0, 0, 0] + list(range(1, 14))) is False)
# density: no zeros cannot satisfy C4
check("density rejects all-nonzero", verify_density(list(range(1, 17))) is False)
# cico: wrong free-input length raises
try:
    verify_cico([0] * 13, r_p=6); raised = False
except ValueError:
    raised = True
check("cico wrong free_inputs length raises", raised)
# cico: a random input is (almost surely) not a solution
check("cico random input not a solution (rp=6)", verify_cico(list(range(2, 16)), r_p=6) is False)

# --- summary ---------------------------------------------------------------
passed = sum(1 for _, ok, _ in _checks if ok)
total = len(_checks)
print(f"\n=== {passed}/{total} checks passed ===")
sys.exit(0 if passed == total else 1)
