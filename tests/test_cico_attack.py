"""
Validate the algebraic CICO attack end-to-end.

  1. The symbolic permutation (over F_p[Y]) agrees with the numeric one.
  2. CICO-1 (root-finding) produces a verifier-accepted solution (reduced rounds).
  3. CICO-2 (resultant route) produces a verifier-accepted solution (reduced rounds).
  4. Degree scaling: measured forward degree == 3^(R_F+R_P); extrapolation to the
     real bounty instances (why the native core is required for RP>=6).

Run:  python3 tests/test_cico_attack.py
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon_attack.constants import P, CICO
from poseidon_attack.poseidon1 import Poseidon1
from poseidon_attack import cico_attack as ca
from poseidon_attack.cico_attack import perm_plus_linear_poly, _build_state, forward_degree
from poseidon_attack import poly as pl

_ok = []


def check(name, cond, detail=""):
    _ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# --- 1. symbolic permutation == numeric permutation ------------------------
print("1. Symbolic permutation over F_p[Y] agrees with numeric permutation_plus_linear")
rng = random.Random(1)
pos = Poseidon1(prime=P, alpha=3, t=16, r_f=6, r_p=2)   # real RF, partial rounds; keep deg modest
k, var = 2, 5
fixed = {i: rng.randrange(P) for i in range(k, 16) if i != var}
assign = {i: ([0, 1] if i == var else [fixed[i]]) for i in range(k, 16)}
sym = perm_plus_linear_poly(pos, _build_state(pos, CICO["constants"], k, assign))
agree = True
for _ in range(5):
    yv = rng.randrange(P)
    numeric_in = [CICO["constants"][0], CICO["constants"][1]] + \
                 [yv if i == var else fixed[i] for i in range(k, 16)]
    numeric_out = pos.permutation_plus_linear(numeric_in)
    sym_out = [pl.peval(c, yv, P) for c in sym]
    if sym_out != numeric_out:
        agree = False
        break
check("symbolic(Y=y) == numeric for all 16 coords, 5 random y", agree,
      f"deg(out0)={pl.deg(sym[0])} (= 3^(R_F+R_P) = 3^{pos.r_f + pos.r_p} = {3**(pos.r_f + pos.r_p)})")

# --- 2. CICO-1 end-to-end (reduced rounds) ---------------------------------
print("2. CICO-1 solve (1 var, 1 constraint) -> verifier-accepted")
pos1 = Poseidon1(prime=P, alpha=3, t=16, r_f=6, r_p=0)   # reduced: forward deg 3^6=729
t0 = time.time()
sol, info = ca.solve_cico1(pos1, tries=12, seed=11)
dt = time.time() - t0
check("CICO-1 RF=6/RP=0 solved & verified", sol is not None,
      f"info={info}, {dt:.1f}s")
if sol is not None:
    # independent re-verification with a fresh verifier call
    from poseidon_attack.verifiers import verify_cico
    rc = [c for row in pos1.rc for c in row]
    ok = verify_cico(sol, pos1.r_p, constants=(CICO["constants"][0], 0), k=1, t=16,
                     r_f=pos1.r_f, mds=pos1.mds, round_constants=rc)
    check("re-verify CICO-1 solution independently", ok)

# --- 3. CICO-2 end-to-end (reduced rounds, resultant route) ----------------
print("3. CICO-2 solve (2 vars, 2 constraints, Res_Y route) -> verifier-accepted")
pos2 = Poseidon1(prime=P, alpha=3, t=16, r_f=2, r_p=0)   # reduced: forward deg 3^2=9
t0 = time.time()
sol2, info2 = ca.solve_cico2(pos2, tries=8, seed=5)
dt = time.time() - t0
check("CICO-2 RF=2/RP=0 solved & verified", sol2 is not None,
      f"info={info2}, {dt:.1f}s")
if sol2 is not None:
    from poseidon_attack.verifiers import verify_cico
    rc = [c for row in pos2.rc for c in row]
    ok = verify_cico(sol2, pos2.r_p, constants=CICO["constants"], k=2, t=16,
                     r_f=pos2.r_f, mds=pos2.mds, round_constants=rc)
    check("re-verify CICO-2 solution independently", ok)

# --- 4. degree scaling -----------------------------------------------------
print("4. Degree scaling (measured forward deg == 3^(R_F+R_P))")
okdeg = True
for (rf, rp) in [(2, 0), (2, 1), (4, 0), (2, 3)]:
    pp = Poseidon1(prime=P, alpha=3, t=16, r_f=rf, r_p=rp)
    d = forward_degree(pp, k=2)
    exp = 3 ** (rf + rp)
    print(f"      RF={rf} RP={rp}: forward deg(out0) = {d}   (3^{rf+rp} = {exp})  {'ok' if d==exp else 'MISMATCH'}")
    okdeg = okdeg and (d == exp)
check("measured forward degree == 3^(R_F+R_P)", okdeg)

print("\n  Extrapolation to the real bounty CICO (RF=6):")
print("  RP | forward 3^(6+RP) | ideal D_I=3^(2*5+RP)  (skip-first-rounds, 2026/150)")
for rp in (6, 8, 10):
    print(f"  {rp:2d} | {3**(6+rp):>16,} | {3**(10+rp):>22,}")
print("  -> RP>=6 final-solve degree is 3^16..3^20 (~2^25..2^32); ~2^53..2^66 ops.")
print("     Pure Python validates the pipeline; the scale solve needs the native core.")

passed = sum(_ok)
print(f"\n=== {passed}/{len(_ok)} checks passed ===")
sys.exit(0 if passed == len(_ok) else 1)
