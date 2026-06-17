"""
Density challenge verifier for the Poseidon Initiative 2026 bounty.

Reference: bounty2026.tex §2.2

The challenge asks for a vector S ∈ F_p^r such that:

  (C1)  #{i : S[i] = 0}  ≤  d                    (sparsity bound)
  (C2)  (a_0, …, a_{ℓ-1}) := H(S)                (hash S with Poseidon1)
  (C3)  (i_0, …, i_{kℓ-1}) := Decode(a_0) ‖ … ‖ Decode(a_{ℓ-1})
  (C4)  S[i_j] = 0  for all j ∈ [t]              (decoded indices hit zeros)

Decode is defined as:
    Decode(x) = log_ω( x^((p-1)//r) )  mod r
where ω is a primitive r-th root of unity in F_p.
The element 0 ∈ F_p maps to index 0 (special case).

Bounty instance parameters (bounty2026.tex §3.2):
    p   = 2^31 - 2^24 + 1  (KoalaBear prime)
    k   = 1
    d   = 2
    r   = 16
    ell = 16
    t   = k * ell = 16
    R_F = 6,  R_P = 6  (Poseidon1, R_P must be > 5)
    ω   = 148625052     (primitive 16th root of unity in F_p)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from poseidon.mds_matrix import verify_mds_matrix, generate_mds_matrix

# ---------------------------------------------------------------------------
# Bounty instance constants
# ---------------------------------------------------------------------------

DENSITY_P   = 2130706433  # 2^31 - 2^24 + 1  (KoalaBear)
DENSITY_ALPHA = 3           # S-box exponent for KoalaBear
DENSITY_K   = 1
DENSITY_D   = 2           # max zeros allowed in S
DENSITY_R   = 16          # len(S)
DENSITY_ELL = 16          # hash output words
DENSITY_T   = DENSITY_K * DENSITY_ELL  # decoded indices that must hit zeros
DENSITY_RF  = 6
DENSITY_RP  = 6           # must be > 5 per spec
DENSITY_TPERM = 16        # Poseidon1 compression mode 16->16

# ω = 148625052 is the primitive 16th root of unity in F_{KoalaBear}.
# Verification: pow(148625052, 16, 2130706433) == 1
#               pow(148625052,  8, 2130706433) == 2130706432  (≠ 1)
_OMEGA = 148625052
_OMEGA_TABLE: dict[int, int] = {
    pow(_OMEGA, e, DENSITY_P): e for e in range(DENSITY_R)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode(x: int, prime: int = DENSITY_P, r: int = DENSITY_R) -> int:
    """
    Apply the Decode function to a single field element.

    Decode(x) = log_ω( x^((p-1)//r) )  mod r

    For the bounty instance (r=16, ω=148625052) a precomputed O(1) lookup
    table is used.  For other parameters a KeyError will be raised if x does
    not map to a power of the hardcoded ω; supply a different omega_table if
    needed.

    Args:
        x:     Field element in F_p.
        prime: Field modulus (default: KoalaBear prime).
        r:     Length of S / modulus for the output index (default: 16).

    Returns:
        Integer index in {0, 1, …, r-1}.
    """
    x = x % prime
    if x == 0:
        return 0
    order = r
    y = pow(x, (prime - 1) // order, prime)
    return _OMEGA_TABLE[y]


def verify_density_solution(
    S: list,
    prime: int = DENSITY_P,
    d: int = DENSITY_D,
    r: int = DENSITY_R,
    t: int = DENSITY_T,
    k: int = DENSITY_K,
    ell: int = DENSITY_ELL,
    r_f: int = DENSITY_RF,
    r_p: int = DENSITY_RP,
    t_perm: int = DENSITY_TPERM,
    alpha: int = DENSITY_ALPHA,
    mds=None,
    round_constants=None,
) -> bool:
    """
    Verify an alleged solution to the Poseidon density challenge.

    Args:
        S:               Candidate solution vector, length r, elements in F_p.
        prime:           Field modulus p.
        d:               Maximum allowed number of zeros in S (sparsity bound).
        r:               Length of S (also the hash input length).
        t:               Number of decoded indices that must be zero positions.
        k:               Number of Decode outputs concatenated per hash word.
        ell:             Number of hash output words (should be smaller than t_perm).
        r_f:             Poseidon full rounds.
        r_p:             Poseidon partial rounds.
        t_perm:          Poseidon permutation state width
        alpha:           S-box exponent
        mds:             Optional custom MDS matrix.
        round_constants: Optional flat list of pre-computed round constants.

    Returns:
        True iff all four conditions (C1–C4) are satisfied.

    Raises:
        ValueError: if S has the wrong length, or t > k*ell, or ell>t_perm.
    """
    if len(S) != r:
        raise ValueError(f"S must have length r={r}, got {len(S)}")
    if t > k * ell:
        raise ValueError(f"t={t} exceeds k*ell={k * ell}")
    if ell > t_perm:
        raise ValueError(f"ell={ell} exceeds t_perm={t_perm}")

    p = prime

    # ------------------------------------------------------------------
    # C1: sparsity — at most d zeros in S
    # ------------------------------------------------------------------
    zero_positions = {i for i, v in enumerate(S) if v % p == 0}
    if len(zero_positions) > d:
        return False

    # ------------------------------------------------------------------
    # C2: hash S with Poseidon1 to obtain ell output words
    # Poseidon1 compression mode: state width = t_perm
    # ------------------------------------------------------------------
    _mds = mds if mds is not None else generate_mds_matrix(t_perm, p)
    if not verify_mds_matrix(_mds, p):
        return False

    pos = Poseidon(
        prime=p,
        alpha=alpha,
        t=t_perm,
        r_f=r_f,
        r_p=r_p,
        mds=_mds,
        round_constants=round_constants,
    )

    # Compression mode: pad S to exactly t_perm elements and hash in one call.
    padded_S = ([v % p for v in S] + [0] * t_perm)[:t_perm]
    hash_output = pos.compression_mode_hash(padded_S, out_length=ell)

    # ------------------------------------------------------------------
    # C3: Decode each output word into k indices in Z_r
    # ω = 148625052 is the primitive 16th root of unity in F_{KoalaBear};
    # use the precomputed O(1) lookup table.
    # ------------------------------------------------------------------
    decoded_indices = []
    for a in hash_output:
        for _ in range(k):
            decoded_indices.append(decode(a, prime=p, r=r))

    # ------------------------------------------------------------------
    # C4: the first t decoded indices must all be zero positions in S
    # ------------------------------------------------------------------
    for j in range(t):
        if decoded_indices[j] not in zero_positions:
            return False

    return True
