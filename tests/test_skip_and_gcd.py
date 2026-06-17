"""
Tests for the CICO-2 GCD route and the resultant degree law.

  1. GCD route recovers a planted solution (line through a known solution) and the
     result is verifier-accepted.
  2. GCD route does not false-positive on a generic random line.
  3. Degree law: MEASURED bivariate resultant deg(R) == 3^(2*R_F+R_P) (NOT the
     naive Bezout 3^(2*(R_F+R_P))); we print the ideal D_I = 3^(2*(R_F-1)+R_P)
     (skip-first-rounds) and the constant-9x gap.

Run:  python3 tests/test_skip_and_gcd.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon_attack.constants import P, CICO
from poseidon_attack.poseidon1 import Poseidon1
from poseidon_attack import cico_attack as ca
from poseidon_attack.verifiers import verify_cico

_ok = []


def check(name, cond, detail=""):
    _ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# --- 1. GCD route recovers a planted solution ------------------------------
print("1. GCD route recovers a planted solution (verifier-accepted)")
pos = Poseidon1(prime=P, alpha=3, t=16, r_f=2, r_p=0)        # reduced rounds
sol, info = ca.solve_cico2(pos, tries=8, seed=5)
assert sol is not None, "precondition: need a known CICO-2 solution to plant"
nfree = 14
rng = random.Random(99)
direction = [rng.randrange(1, P) for _ in range(nfree)]
x0 = rng.randrange(P)
base = [(sol[j] - x0 * direction[j]) % P for j in range(nfree)]   # line hits sol at X=x0
got, ginfo = ca.solve_cico2_gcd(pos, base=base, direction=direction)
rc = [c for row in pos.rc for c in row]
ok = got is not None and verify_cico(got, pos.r_p, constants=CICO["constants"], k=2, t=16,
                                     r_f=pos.r_f, mds=pos.mds, round_constants=rc)
check("GCD route finds a verifier-accepted solution on the planted line", ok,
      f"info={ginfo}")
check("recovered solution equals the planted one", got == sol)

# --- 2. No false positive on a random line ---------------------------------
print("2. GCD route does not false-positive on a generic random line")
none_got, _ = ca.solve_cico2_gcd(pos, lines=1, seed=123456)   # single random line
check("random line yields no (spurious) solution", none_got is None)

# --- 3. Resultant degree law (MEASURED) ------------------------------------
# Measured: deg(R) = 3^(2*R_F + R_P).  Ideal (skip-first-rounds, 2026/150)
# D_I = 3^(2*(R_F-1)+R_P): a CONSTANT factor 3^2 = 9 smaller, independent of R_P.
print("3. Measured resultant degree law:  deg(R) == 3^(2*R_F + R_P)")
okdeg = True
for (rf, rp) in [(2, 0), (2, 1), (2, 2)]:
    pp = Poseidon1(prime=P, alpha=3, t=16, r_f=rf, r_p=rp)
    d = ca.measure_resultant_degree(pp)
    exp = 3 ** (2 * rf + rp)
    ideal = 3 ** (2 * (rf - 1) + rp)
    print(f"      RF={rf} RP={rp}: measured deg(R)={d}  law 3^{2*rf+rp}={exp}  "
          f"ideal D_I 3^{2*(rf-1)+rp}={ideal}  (skip saves x{exp//ideal})  "
          f"{'ok' if d == exp else 'MISMATCH'}")
    okdeg = okdeg and (d == exp)
check("measured deg(R) == 3^(2*R_F + R_P)  (3 data points)", okdeg)
check("skip-first-rounds gap is the constant 3^2 = 9", (3 ** (2 * 2 + 1)) // (3 ** (2 * 1 + 1)) == 9)

print("\n  Real CICO (RF=6) root-find degree -- the native-core target:")
print("  RP | our resultant deg 3^(12+RP) |  ideal D_I 3^(10+RP)  | bits log2(deg)")
import math
for rp in (6, 8, 10):
    dnaive = 3 ** (12 + rp)
    print(f"  {rp:2d} | {dnaive:>27,} | {3**(10+rp):>20,} | ~2^{math.log2(dnaive):.1f}")
print("  Note: at high degree the explicit interpolated resultant is memory/cost-heavy,")
print("  while the line-GCD route is linear-memory but has ~1/p hit rate. Both are")
print("  calibration routes; the economic path requires amortized resultants.")

passed = sum(_ok)
print(f"\n=== {passed}/{len(_ok)} checks passed ===")
sys.exit(0 if passed == len(_ok) else 1)
