"""
Frozen constants for the Poseidon Cryptanalysis Initiative 2026 challenge suite.

Every value here is transcribed from PRIMARY sources and annotated with the
source location, so the instance definitions are auditable:

  [tex]  reference/poseidon-tools/bounties/docs/bounty2026.tex
  [ref]  reference/poseidon-tools/  (khovratovich/poseidon-tools @ 60075da7, 2026-05-14)

These are the *load-bearing* numbers: a solver that targets any other matrix /
round-constant family / round count produces solutions the default verifier
rejects.
"""

# --- Base field: KoalaBear -------------------------------------------------
P = 2130706433              # 2^31 - 2^24 + 1                         [tex §3]
ALPHA = 3                   # S-box exponent x -> x^3                 [tex §3]
assert P == 2**31 - 2**24 + 1
# p - 1 = 2^24 * 127  =>  2-adic valuation 24 (FFT/NTT-friendly), smooth.
assert (P - 1) == (2**24) * 127

# --- Permutation width -----------------------------------------------------
T = 16                      # state width, all four challenges        [tex §3]

# --- Per-challenge instance parameters -------------------------------------
# Partial collision (the big-money ladder q=t=3..7).  FULL STRENGTH.   [tex §3.1]
#   NOTE: RF=8, RP=20 -- NOT RF=6.  RF=6 applies only to CICO/density/zero-test.
COLLISION = dict(rf=8, rp=20, ell=16, t_perm=16, seed=0xC09DE4)

# CICO-2: two constrained input + two constrained output words.        [tex §3.4]
#   Uses permutation_plus_linear (an extra MDS layer before round 1).
#   Funded partial-round set is {6, 8, 10}; RP=10 is the open target.
CICO = dict(rf=6, rp_set=(6, 8, 10), rp_open=10, k=2, t=16,
            constants=(0xC09DE4, 0xEE6282, 0, 0))   # C1, C2, C3=C4=0

# Density: <=2 zeros in S (len 16); all 16 decoded outputs hit a zero pos. [tex §3.2]
#   Decode(x) = log_omega( x^((p-1)/r) ), r=16, omega a 16th root of unity.
#   Best-attack rate (d/r)^16 = (2/16)^16 = 2^-48.
DENSITY = dict(rf=6, rp=6, k=1, d=2, r=16, ell=16, omega=148625052)

# Zero-test: degree-<=7 poly over F_{p^2}=F_p[x]/(x^2-3); P(a_0)=0.     [tex §3.3]
#   a_0 = (out[0], out[1]) is the first hash output as one F_{p^2} element.
#   Best-attack rate d/p^2 = 7/p^2 = 2^-59.2.
ZEROTEST = dict(rf=6, rp=6, r=2, d=7, ell=8, s=1, ext_beta=3)

# 148625052 is a primitive 16th root of unity in F_p; 3 is a QNR (zero-test ext).
assert pow(DENSITY["omega"], 16, P) == 1
assert pow(DENSITY["omega"], 8, P) == P - 1
assert pow(ZEROTEST["ext_beta"], (P - 1) // 2, P) == P - 1
