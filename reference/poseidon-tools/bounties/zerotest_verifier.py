"""
Zero-test challenge verifier for the Poseidon Initiative 2026 bounty.

Reference: bounty2026.tex §2.3

The challenge asks for a polynomial P : (F_{p^r})^s → F_{p^r} defined over
the extension field F_{p^r} such that:

  (C1)  1 ≤ degree(P) ≤ d                    (non-trivial degree bound)
  (C2)  (a_0, …, a_{ℓ-1}) := H(P̂)           (hash the coefficient vector)
  (C3)  P(a_0, …, a_{s-1}) = 0 ∈ F_{p^r}    (P vanishes at the first s hash outputs)

where P̂ is the flat list of base-field coefficients of P.
The length of P̂ is (d+1)·r base-field elements for s=1.

Extension field:
    F_{p^r} = F_p[x] / (x^2 − 3)

    3 is a quadratic non-residue mod KoalaBear:
        pow(3, (p-1)//2, p) == p-1  ←→  Legendre symbol (3|p) = −1.
    Proof via QR: p ≡ 2 (mod 3), and by quadratic reciprocity
        (3/p)(p/3) = (−1)^{(p−1)/2} = 1  (since (p−1)/2 is even),
    so (3/p) = (p/3) = (2/3) = −1.

    An element is stored as a pair (a, b) ∈ F_p² representing a + b·√3.

Bounty instance parameters (bounty2026.tex §3.3):
    p   = 2^31 − 2^24 + 1  (KoalaBear prime)
    r   = 2    (quadratic extension)
    d   = 7    (max polynomial degree)
    ℓ   = 8    (hash output words in ext-field elements; ℓ·r = 16 base-field elements)
    s   = 1    (univariate: P : F_{p^r} → F_{p^r})
    R_F = 6,   R_P > 5  (Poseidon1 compression mode, state width = ℓ·r = 16)

Best-attack success probability: d / p^r ≈ 7 / (2^31)^2 ≈ 2^{−59.2}  (§3.3)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from poseidon.mds_matrix import verify_mds_matrix, generate_mds_matrix

# ---------------------------------------------------------------------------
# Bounty instance constants
# ---------------------------------------------------------------------------

ZEROTEST_P        = 2130706433   # 2^31 − 2^24 + 1  (KoalaBear)
ZEROTEST_ALPHA    = 3            # S-box exponent
ZEROTEST_R        = 2            # extension field degree
ZEROTEST_D        = 7            # maximum polynomial degree
ZEROTEST_ELL      = 8            # hash output words (ext-field elements)
ZEROTEST_S        = 1            # number of polynomial input variables (univariate)
ZEROTEST_RF       = 6
ZEROTEST_RP       = 6            # must be > 5 per spec
ZEROTEST_TPERM    = ZEROTEST_ELL * ZEROTEST_R  # = 16  (state width, base-field elements)

# Length of P̂ for s=1: (d+1)·r = 8·2 = 16 base-field elements.
ZEROTEST_COEFF_LEN = (ZEROTEST_D + 1) * ZEROTEST_R  # = 16

# ---------------------------------------------------------------------------
# Extension field  F_{p^2} = F_p[x] / (x^2 − EXT_BETA)
#
# EXT_BETA = 3 is a quadratic non-residue mod KoalaBear (see module docstring).
# Verification: pow(3, (2130706433 - 1) // 2, 2130706433) == 2130706432 == p-1.
# ---------------------------------------------------------------------------

_EXT_BETA = 3   # irreducible polynomial for the extension: x^2 − 3


def _ext_add(
    a: tuple[int, int],
    b: tuple[int, int],
    p: int,
) -> tuple[int, int]:
    """Component-wise addition in F_{p^2}."""
    return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)


def _ext_mul(
    a: tuple[int, int],
    b: tuple[int, int],
    p: int,
    beta: int = _EXT_BETA,
) -> tuple[int, int]:
    """
    Multiplication in F_{p^2} = F_p[x] / (x^2 − beta).

    (a0 + a1·x)(b0 + b1·x) = (a0·b0 + beta·a1·b1)  +  (a0·b1 + a1·b0)·x
    """
    c0 = (a[0] * b[0] + beta * a[1] * b[1]) % p
    c1 = (a[0] * b[1] + a[1] * b[0]) % p
    return (c0, c1)


def _ext_eval_poly(
    coeffs: list[tuple[int, int]],
    x: tuple[int, int],
    p: int,
    beta: int = _EXT_BETA,
) -> tuple[int, int]:
    """
    Evaluate a univariate polynomial over F_{p^2} at x using Horner's method.

    Args:
        coeffs: Coefficient list [c_0, c_1, …, c_deg] where index = degree.
                Each c_i is an (a, b) pair representing a + b·√beta ∈ F_{p^2}.
        x:      Evaluation point in F_{p^2}.
        p:      Base-field modulus.
        beta:   Non-residue β defining the extension F_p[x]/(x^2 − β).

    Returns:
        P(x) as a pair (a, b) ∈ F_{p^2}.
    """
    if not coeffs:
        return (0, 0)
    # Horner: P(x) = c_0 + x·(c_1 + x·(c_2 + … + x·c_deg))
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = _ext_mul(result, x, p, beta)
        result = _ext_add(result, c, p)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_zerotest_solution(
    P_hat: list[int],
    prime: int = ZEROTEST_P,
    r: int = ZEROTEST_R,
    d: int = ZEROTEST_D,
    ell: int = ZEROTEST_ELL,
    s: int = ZEROTEST_S,
    r_f: int = ZEROTEST_RF,
    r_p: int = ZEROTEST_RP,
    t_perm: int = ZEROTEST_TPERM,
    alpha: int = ZEROTEST_ALPHA,
    ext_beta: int = _EXT_BETA,
    mds: list | None = None,
    round_constants: list | None = None,
) -> bool:
    """
    Verify an alleged solution to the Poseidon zero-test challenge.

    The solution is provided as the flat coefficient vector P̂ of a polynomial
    P : F_{p^r} → F_{p^r} (for s=1) that must satisfy:

        (C1)  1 ≤ degree(P) ≤ d
        (C2)  (a_0, …, a_{ℓ-1}) := H(P̂)     via Poseidon1 compression
        (C3)  P(a_0) = 0 ∈ F_{p^r}           (s=1: evaluate at first output word)

    Coefficient encoding (s=1):
        P_hat has length (d+1)·r base-field elements.
        The coefficient of degree j occupies P_hat[j·r : (j+1)·r], representing
        the extension-field element  P_hat[j·r] + P_hat[j·r+1]·√ext_beta.

    Hash construction (Poseidon1 compression mode):
        The full P_hat of length t_perm = ell·r is absorbed into the all-zero
        Poseidon state in a single permutation call.  The first ell·r base-field
        elements of the output state are then grouped into ell extension-field
        words a_0, …, a_{ℓ-1}.

    Args:
        P_hat:           Flat list of base-field integers — the coefficient
                         vector of P.  Expected length: (d+1)·r for s=1.
        prime:           Base-field modulus p.
        r:               Extension field degree.
        d:               Maximum allowed polynomial degree.
        ell:             Number of hash output words (extension-field elements).
        s:               Number of polynomial input variables (only s=1 supported).
        r_f:             Number of Poseidon full rounds.
        r_p:             Number of Poseidon partial rounds (must be > 5 for bounty).
        t_perm:          Poseidon state width in base-field elements; must be at least ell·r and at least d*r.
        alpha:           Poseidon S-box exponent.
        ext_beta:        Non-residue β defining F_{p^r} = F_p[x]/(x^r − β).
        mds:             Optional custom MDS matrix (t_perm × t_perm).
        round_constants: Optional flat list of (r_f + r_p)·t_perm pre-computed
                         round constants.

    Returns:
        True iff all three conditions (C1–C3) are satisfied.

    Raises:
        ValueError:        if parameters are inconsistent or P_hat has wrong length.
        NotImplementedError: if s ≠ 1 or r ≠ 2.
    """
    if s != 1:
        raise NotImplementedError(
            "Only univariate polynomials (s=1) are currently supported."
        )
    if r != 2:
        raise NotImplementedError(
            "Only quadratic extensions (r=2) are currently supported."
        )
    if t_perm < ell * r or t_perm < d * r:
        raise ValueError(
            f"t_perm={t_perm} must be at least ell*r={ell * r} and at least d*r={d * r}"
        )

    expected_len = (d + 1) * r
    if len(P_hat) != expected_len:
        raise ValueError(
            f"P_hat must have length (d+1)*r = {expected_len} for s=1, "
            f"got {len(P_hat)}"
        )

    p = prime

    # ------------------------------------------------------------------
    # Parse coefficients from the flat base-field vector.
    # Coefficient of degree j: (P_hat[j*r], P_hat[j*r+1])
    # ------------------------------------------------------------------
    coeffs: list[tuple[int, int]] = [
        (int(P_hat[j * r]) % p, int(P_hat[j * r + 1]) % p)
        for j in range(d + 1)
    ]

    # ------------------------------------------------------------------
    # C1: degree bound — 1 ≤ degree(P) ≤ d
    # The degree is the largest j with coeffs[j] ≠ (0, 0).
    # ------------------------------------------------------------------
    poly_degree = -1
    for j in range(d, -1, -1):
        if coeffs[j] != (0, 0):
            poly_degree = j
            break

    if not (1 <= poly_degree <= d):
        return False

    # ------------------------------------------------------------------
    # C2: hash P̂ with Poseidon1 to obtain ell extension-field output words.
    #
    # Poseidon1 compression mode: state width = t_perm = ell*r.
    # P_hat has exactly t_perm elements → absorbed in a single permutation call.
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

    hash_output = pos.compression_mode_hash(
        [int(v) % p for v in P_hat], out_length=ell * r
    )

    # Group the first ell*r base-field outputs into ell extension-field elements.
    # a[i] = (hash_output[i*r], hash_output[i*r+1])  ∈ F_{p^2}
    a: list[tuple[int, int]] = [
        (hash_output[i * r] % p, hash_output[i * r + 1] % p)
        for i in range(ell)
    ]

    # ------------------------------------------------------------------
    # C3: P(a_0) = 0 in F_{p^r}   (s=1: evaluate at the first output word)
    # ------------------------------------------------------------------
    result = _ext_eval_poly(coeffs, a[0], p, beta=ext_beta)
    return result == (0, 0)


def verify_zerotest_solution_relaxed(
    P_hat: list[int],
    k: int,
    prime: int = ZEROTEST_P,
    r: int = ZEROTEST_R,
    d: int = ZEROTEST_D,
    ell: int = ZEROTEST_ELL,
    s: int = ZEROTEST_S,
    r_f: int = ZEROTEST_RF,
    r_p: int = ZEROTEST_RP,
    t_perm: int = ZEROTEST_TPERM,
    alpha: int = ZEROTEST_ALPHA,
    ext_beta: int = _EXT_BETA,
    mds: list | None = None,
    round_constants: list | None = None,
) -> bool:
    """
    Relaxed verifier for the Poseidon zero-test challenge.

    Identical to verify_zerotest_solution except that condition (C3) is relaxed:
    instead of requiring P(a_0) = 0 ∈ F_{p^r} exactly, it only requires that
    the lowest k bits of every base-field component of P(a_0) are zero.

    Formally, writing P(a_0) = (c0, c1) ∈ F_{p^2}, the relaxed condition is:

        c0 & mask == 0  AND  c1 & mask == 0,    mask = (1 << k) − 1

    i.e. both base-field components are divisible by 2^k.

    Note:
        k = 0  is trivially satisfied by any polynomial (all integers have 0
                low bits set to zero when mask = 0).
        k ≥ 1  becomes progressively harder; full equality corresponds to k
                being large enough that the only solution is the all-zero residue,
                which for p < 2^31 is achieved at k = 31 (since p < 2^31 means
                any non-zero element has at least one of its low 31 bits set).

    Args:
        P_hat:           Flat list of base-field integers — the coefficient
                         vector of P.  Expected length: (d+1)·r for s=1.
        k:               Number of least-significant bits that must be zero in
                         each base-field component of P(a_0).  Must be ≥ 0.
        prime:           Base-field modulus p.
        r:               Extension field degree.
        d:               Maximum allowed polynomial degree.
        ell:             Number of hash output words (extension-field elements).
        s:               Number of polynomial input variables (only s=1 supported).
        r_f:             Number of Poseidon full rounds.
        r_p:             Number of Poseidon partial rounds (must be > 5 for bounty).
        t_perm:          Poseidon state width in base-field elements.
        alpha:           Poseidon S-box exponent.
        ext_beta:        Non-residue β defining F_{p^r} = F_p[x]/(x^r − β).
        mds:             Optional custom MDS matrix (t_perm × t_perm).
        round_constants: Optional flat list of (r_f + r_p)·t_perm pre-computed
                         round constants.

    Returns:
        True iff conditions (C1), (C2), and the relaxed (C3) are all satisfied.

    Raises:
        ValueError:        if k < 0, or parameters are inconsistent, or P_hat
                           has wrong length.
        NotImplementedError: if s ≠ 1 or r ≠ 2.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    if s != 1:
        raise NotImplementedError(
            "Only univariate polynomials (s=1) are currently supported."
        )
    if r != 2:
        raise NotImplementedError(
            "Only quadratic extensions (r=2) are currently supported."
        )
    if t_perm < ell * r or t_perm < d * r:
        raise ValueError(
            f"t_perm={t_perm} must be at least ell*r={ell * r} and at least d*r={d * r}"
        )

    expected_len = (d + 1) * r
    if len(P_hat) != expected_len:
        raise ValueError(
            f"P_hat must have length (d+1)*r = {expected_len} for s=1, "
            f"got {len(P_hat)}"
        )

    p = prime

    # ------------------------------------------------------------------
    # Parse coefficients from the flat base-field vector.
    # ------------------------------------------------------------------
    coeffs: list[tuple[int, int]] = [
        (int(P_hat[j * r]) % p, int(P_hat[j * r + 1]) % p)
        for j in range(d + 1)
    ]

    # ------------------------------------------------------------------
    # C1: 1 ≤ degree(P) ≤ d
    # ------------------------------------------------------------------
    poly_degree = -1
    for j in range(d, -1, -1):
        if coeffs[j] != (0, 0):
            poly_degree = j
            break

    if not (1 <= poly_degree <= d):
        return False

    # ------------------------------------------------------------------
    # C2: hash P̂ with Poseidon1
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

    hash_output = pos.compression_mode_hash(
        [int(v) % p for v in P_hat], out_length=ell * r
    )

    a: list[tuple[int, int]] = [
        (hash_output[i * r] % p, hash_output[i * r + 1] % p)
        for i in range(ell)
    ]

    # ------------------------------------------------------------------
    # Relaxed C3: lowest k bits of each component of P(a_0) are zero
    # ------------------------------------------------------------------
    c0, c1 = _ext_eval_poly(coeffs, a[0], p, beta=ext_beta)
    mask = (1 << k) - 1
    return (c0 & mask) == 0 and (c1 & mask) == 0
