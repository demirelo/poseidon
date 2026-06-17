"""
CICO (Constrained Input Constrained Output) challenge verifier for the
Poseidon Initiative 2026 bounty.

Reference: bounty2026.tex §2.4

The challenge asks for a free-input portion (x_{k+1}, …, x_t) ∈ F_p^{t−k}
such that applying the Poseidon permutation P to the full state

    s = (C_1, C_2, …, C_k, x_{k+1}, …, x_t)

produces an output whose first k words match the prescribed output constants:

    P(s)[i] = C_{k+1+i}   for i = 0, 1, …, k−1

where all remaining output positions are unconstrained ("*").

Conditions verified:
    (C1)  free_inputs has length t − k                      (correct size)
    (C2)  s := [C_1, …, C_k, x_{k+1}, …, x_t]             (build full state)
    (C3)  y := Poseidon::permutation_plus_linear(s)         (apply permutation)
    (C4)  y[i] == C_{k+i+1} % p  for i = 0, …, k−1        (output matches)

Bounty instance parameters (bounty2026.tex §3.4):
    p    = 2^31 − 2^24 + 1  (KoalaBear prime)
    k    = 2    (number of constrained input / output words)
    t    = 16   (Poseidon permutation state width)
    R_F  = 6,   R_P ∈ {6, 8, 10}

Fixed constants:
    C_1 = 0xc09de4,  C_2 = 0xee6282,  C_3 = C_4 = 0

Best-attack success probability: 1 / p^k ≈ 1 / (2^31)^2 ≈ 2^{−62}  (§3.4)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from poseidon.mds_matrix import verify_mds_matrix, generate_mds_matrix

# ---------------------------------------------------------------------------
# Bounty instance constants
# ---------------------------------------------------------------------------

CICO_P      = 2130706433   # 2^31 − 2^24 + 1  (KoalaBear)
CICO_ALPHA  = 3            # S-box exponent
CICO_K      = 2            # constrained input / output words
CICO_T      = 16           # Poseidon permutation state width
CICO_RF     = 6
CICO_RP     = 8            # bounty offers R_P ∈ {6, 8, 10}; default = 8 (hardest)

# Fixed constants C_1, C_2, …, C_{2k}  (indexed 0-based: constants[0] = C_1)
# C_1 = 0xc09de4,  C_2 = 0xee6282,  C_3 = C_4 = 0
CICO_CONSTANTS: list[int] = [0xC09DE4, 0xEE6282, 0, 0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_cico_solution(
    free_inputs: list[int],
    prime: int = CICO_P,
    alpha: int = CICO_ALPHA,
    k: int = CICO_K,
    t: int = CICO_T,
    r_f: int = CICO_RF,
    r_p: int = CICO_RP,
    constants: list[int] | None = None,
    mds: list | None = None,
    round_constants: list | None = None,
) -> bool:
    """
    Verify an alleged solution to the Poseidon CICO challenge.

    The solution is provided as the free-input portion
    (x_{k+1}, …, x_t) ∈ F_p^{t−k}.  The verifier prepends the fixed input
    constants C_1, …, C_k to form the full permutation state, applies
    Poseidon::permutation_plus_linear(), and checks that the first k output
    words equal
    the prescribed output constants C_{k+1}, …, C_{2k}.

    Conditions checked:
        (C1)  len(free_inputs) == t − k
        (C2)  s := [C_1, …, C_k] + free_inputs   (mod p)
        (C3)  y := Poseidon::permutation_plus_linear(s)
        (C4)  y[i] == constants[k + i] % p   for i = 0, …, k−1

    Args:
        free_inputs:     Free portion of the permutation input,
                         (x_{k+1}, …, x_t), as a list of t − k integers.
        prime:           Base-field modulus p.
        alpha:           Poseidon S-box exponent.
        k:               Number of constrained input and output words.
        t:               Poseidon permutation state width.
        r_f:             Number of full rounds (must be even).
        r_p:             Number of partial rounds.
        constants:       List of 2k field elements [C_1, …, C_k, C_{k+1}, …,
                         C_{2k}].  Defaults to CICO_CONSTANTS.
        mds:             Optional custom t×t MDS matrix.
        round_constants: Optional flat list of (r_f + r_p)·t pre-computed
                         round constants.

    Returns:
        True iff all conditions (C1–C4) are satisfied.

    Raises:
        ValueError: if parameters are inconsistent or free_inputs has the
                    wrong length.
    """
    if constants is None:
        constants = CICO_CONSTANTS

    if k <= 0 or k > t:
        raise ValueError(f"k must satisfy 1 ≤ k ≤ t={t}, got k={k}")
    if len(constants) < 2 * k:
        raise ValueError(
            f"constants must have at least 2*k={2 * k} elements, "
            f"got {len(constants)}"
        )

    expected_len = t - k
    if len(free_inputs) != expected_len:
        raise ValueError(
            f"free_inputs must have length t−k={expected_len}, "
            f"got {len(free_inputs)}"
        )

    p = prime

    # ------------------------------------------------------------------
    # C2: build the full permutation input state
    # ------------------------------------------------------------------
    state_in: list[int] = (
        [int(c) % p for c in constants[:k]]
        + [int(x) % p for x in free_inputs]
    )

    # ------------------------------------------------------------------
    # C3: apply Poseidon permutation  y = P(s)
    # ------------------------------------------------------------------
    _mds = mds if mds is not None else generate_mds_matrix(t, p)
    if not verify_mds_matrix(_mds, p):
        return False

    pos = Poseidon(
        prime=p,
        alpha=alpha,
        t=t,
        r_f=r_f,
        r_p=r_p,
        mds=_mds,
        round_constants=round_constants,
    )
    state_out = pos.permutation_plus_linear(state_in)

    # ------------------------------------------------------------------
    # C4: first k output words must match C_{k+1}, …, C_{2k}
    # ------------------------------------------------------------------
    for i in range(k):
        if state_out[i] != int(constants[k + i]) % p:
            return False
    return True


def verify_cico_solution_relaxed(
    free_inputs: list[int],
    m: int,
    prime: int = CICO_P,
    alpha: int = CICO_ALPHA,
    k: int = CICO_K,
    t: int = CICO_T,
    r_f: int = CICO_RF,
    r_p: int = CICO_RP,
    constants: list[int] | None = None,
    mds: list | None = None,
    round_constants: list | None = None,
) -> bool:
    """
    Relaxed verifier for the Poseidon CICO challenge.

    Identical to verify_cico_solution except that condition (C4) is relaxed:
    instead of requiring exact equality, it only requires that the lowest m
    bits of (y[i] − C_{k+i+1}) are zero for each constrained output word i:

        (y[i] - constants[k + i]) & mask == 0,   mask = (1 << m) − 1

    i.e. the difference between each constrained output and the target
    constant is divisible by 2^m.

    Note:
        m = 0  is trivially satisfied by any input.
        m ≥ 1  becomes progressively harder; full equality is achieved when
               m is large enough that the only solution to the congruence
               condition is the exact match (for p < 2^31, m = 31 suffices).

    Args:
        free_inputs:     Free portion of the permutation input
                         (x_{k+1}, …, x_t) as a list of t − k integers.
        m:               Number of least-significant bits that must be zero in
                         each (y[i] − C_{k+i+1}) difference.  Must be ≥ 0.
        prime:           Base-field modulus p.
        alpha:           Poseidon S-box exponent.
        k:               Number of constrained input and output words.
        t:               Poseidon permutation state width.
        r_f:             Number of full rounds (must be even).
        r_p:             Number of partial rounds.
        constants:       List of 2k field elements [C_1, …, C_k, C_{k+1}, …,
                         C_{2k}].  Defaults to CICO_CONSTANTS.
        mds:             Optional custom t×t MDS matrix.
        round_constants: Optional flat list of (r_f + r_p)·t pre-computed
                         round constants.

    Returns:
        True iff conditions (C1–C3) and the relaxed (C4) are all satisfied.

    Raises:
        ValueError: if m < 0, or parameters are inconsistent, or free_inputs
                    has the wrong length.
    """
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")

    if constants is None:
        constants = CICO_CONSTANTS

    if k <= 0 or k > t:
        raise ValueError(f"k must satisfy 1 ≤ k ≤ t={t}, got k={k}")
    if len(constants) < 2 * k:
        raise ValueError(
            f"constants must have at least 2*k={2 * k} elements, "
            f"got {len(constants)}"
        )

    expected_len = t - k
    if len(free_inputs) != expected_len:
        raise ValueError(
            f"free_inputs must have length t−k={expected_len}, "
            f"got {len(free_inputs)}"
        )

    p = prime

    # ------------------------------------------------------------------
    # Build full input state and apply permutation
    # ------------------------------------------------------------------
    state_in: list[int] = (
        [int(c) % p for c in constants[:k]]
        + [int(x) % p for x in free_inputs]
    )

    _mds = mds if mds is not None else generate_mds_matrix(t, p)
    if not verify_mds_matrix(_mds, p):
        return False

    pos = Poseidon(
        prime=p,
        alpha=alpha,
        t=t,
        r_f=r_f,
        r_p=r_p,
        mds=_mds,
        round_constants=round_constants,
    )
    state_out = pos.permutation_plus_linear(state_in)

    # ------------------------------------------------------------------
    # Relaxed C4: lowest m bits of each (y[i] − target) must be zero
    # ------------------------------------------------------------------
    mask = (1 << m) - 1
    for i in range(k):
        target = int(constants[k + i]) % p
        diff = (state_out[i] - target) % p
        if (diff & mask) != 0:
            return False
    return True
