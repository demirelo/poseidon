"""
Empirical density-decode profiler.

Resolves a spec-vs-implementation discrepancy that changes tactics:

  bounty2026.tex claims  mu(0) = 1/p,  mu(i!=0) = (p-1)/(p*r)
  -> would make index 0 "rare" and advise never zeroing position 0.

  The IMPLEMENTED Decode(x) = log_omega(x^((p-1)/16)) maps F_p^* uniformly onto
  the 16 sixteenth-roots of unity, so every index incl. 0 has prob ~1/16.

We measure both (a) Decode over random field elements and (b) Decode of real
RP=6 compression-hash outputs, to confirm the implemented distribution is
uniform and that index 0 is a perfectly good zero position.

Run:  python3 tests/profile_density_decode.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon_attack.constants import P
from poseidon_attack.poseidon1 import Poseidon1
from poseidon_attack.verifiers import decode

rng = random.Random(2026)


def dist(counts, n):
    return [round(counts.get(i, 0) / n, 4) for i in range(16)]


# (a) Decode over random nonzero field elements -----------------------------
N = 200_000
ca = {}
for _ in range(N):
    idx = decode(rng.randrange(1, P))
    ca[idx] = ca.get(idx, 0) + 1
print(f"(a) Decode of {N} random nonzero F_p elements:")
print(f"    per-index freq: {dist(ca, N)}")
print(f"    index 0 freq = {ca.get(0,0)/N:.4f}   (spec says 1/p~={1/P:.2e}; uniform says {1/16:.4f})")

# (b) Decode of real RP=6 compression-hash outputs --------------------------
M = 4000
pos = Poseidon1(prime=P, alpha=3, t=16, r_f=6, r_p=6)  # density instance
cb = {}
for _ in range(M):
    S = [rng.randrange(0, P) for _ in range(16)]
    out = pos.compression_mode_hash(S, out_length=16)
    for a in out:
        idx = decode(a)
        cb[idx] = cb.get(idx, 0) + 1
nb = M * 16
print(f"\n(b) Decode of {nb} real RP=6 hash output words:")
print(f"    per-index freq: {dist(cb, nb)}")
exp = nb / 16
chisq = sum((cb.get(i, 0) - exp) ** 2 / exp for i in range(16))
print(f"    chi-square = {chisq:.1f} on 15 dof (5% critical ~25.0)  ->",
      "UNIFORM (no exploitable 1st-order bias)" if chisq < 25 else "NON-UNIFORM (investigate!)")

print("\nConclusion: implemented Decode is ~uniform over {0..15}; index 0 is NOT")
print("special (spec's mu(0)=1/p is inconsistent with the shipped verifier).")
print("=> 'never zero position 0' advice is WRONG for the real instance.")
