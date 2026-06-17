"""
Sample solver for a relaxed zero-test challenge instance.

Reference: bounty2026.tex §2.3

This is a relaxed instance designed to demonstrate that solutions are easy to
find when only the lowest k bits of P(a_0) are required to be zero:

    p       = 2^31 - 2^24 + 1  (KoalaBear prime)
    alpha   = 3
    r       = 2    extension field degree  (F_{p^2} = F_p[x]/(x^2 - 3))
    d       = 7    max polynomial degree
    ell     = 8    hash output words (ext-field elements)
    s       = 1    univariate polynomial  P : F_{p^2} -> F_{p^2}
    t_perm  = 16   Poseidon1 state width  (= ell * r, full-state absorption)
    k       = 4    number of low bits that must be zero in each component
                   (solver parameter, passed to solve() and --k on the CLI)
    R_F     = 6,   R_P = 6

Challenge (relaxed, default k=4):
    Find P : F_{p^2} -> F_{p^2} with 1 <= deg(P) <= 7 and
    (a_0, ..., a_7) := H(P_hat)  (Poseidon1, state=16)
    such that the lowest k bits of each base-field component of P(a_0) are 0:
        P(a_0).c0 & ((1<<k)-1) == 0  AND  P(a_0).c1 & ((1<<k)-1) == 0

    where P_hat is the flat list of 16 base-field coefficients of P and
    P(a_0) = (c0, c1) in F_{p^2}.

Polynomial encoding:
    coeffs[j] = (P_hat[j*2], P_hat[j*2+1])  for j = 0, ..., 7.
    coeffs[j] represents the degree-j coefficient as a + b*sqrt(3).

Extension field:
    F_{p^2} = F_p[x] / (x^2 - 3).
    3 is a quadratic non-residue mod KoalaBear:
        pow(3, (p-1)//2, p) == p - 1.

Expected success probability per attempt: (1/2^k)^2 = 1/2^(2k)
Expected attempts needed: ~2^(2k)  (e.g. ~256 for k=4)
"""

import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from bounties.zerotest_verifier import (
    _EXT_BETA,
    _ext_eval_poly,
    verify_zerotest_solution_relaxed,
    ZEROTEST_P,
    ZEROTEST_ALPHA,
    ZEROTEST_R,
    ZEROTEST_D,
    ZEROTEST_ELL,
    ZEROTEST_S,
    ZEROTEST_RF,
    ZEROTEST_RP,
    ZEROTEST_TPERM,
    ZEROTEST_COEFF_LEN,
)

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

P      = ZEROTEST_P        # 2^31 - 2^24 + 1  (KoalaBear)
ALPHA  = ZEROTEST_ALPHA    # 3
R      = ZEROTEST_R        # 2  (quadratic extension)
D      = ZEROTEST_D        # 7  (max degree)
ELL    = ZEROTEST_ELL      # 8  (hash output words in ext-field elements)
S      = ZEROTEST_S        # 1  (univariate)
T_PERM = ZEROTEST_TPERM    # 16 (= ELL * R, base-field state width)
RF     = ZEROTEST_RF       # 6
RP     = ZEROTEST_RP       # 6
DEFAULT_K = 4              # default: lowest 4 bits of each component must be zero


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(
    k: int = DEFAULT_K,
    max_attempts: int = 100_000,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[int], int] | None:
    """
    Search for a relaxed solution to the zero-test challenge.

    Strategy:
        Repeat up to max_attempts times:
          1. Draw a random polynomial P of random degree 1..d over F_{p^2}.
             Coefficients are random pairs (a, b) in F_p^2; the leading
             coefficient is chosen to be non-zero so the degree is exact.
          2. Flatten the D+1 coefficients into P_hat (16 base-field integers).
          3. Hash P_hat with Poseidon1 (state=T_PERM=16, full-state absorption).
          4. Evaluate P(a_0) in F_{p^2} using Horner's method.
          5. Accept if the lowest k bits of both base-field components are 0.
          6. Double-check via verify_zerotest_solution_relaxed() before returning.

    Args:
        k:            Number of least-significant bits that must be zero in each
                      base-field component of P(a_0).  Must be >= 0.
        max_attempts: Maximum number of random trials before giving up.
        seed:         Optional RNG seed for reproducibility.
        verbose:      Print progress and result.

    Returns:
        (P_hat, attempt_number) if a solution is found, else None.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    rng = random.Random(seed)
    mask = (1 << k) - 1

    # Build the Poseidon instance once (round constants derived from Grain LFSR).
    pos = Poseidon(
        prime=P,
        alpha=ALPHA,
        t=T_PERM,
        r_f=RF,
        r_p=RP,
    )

    if verbose:
        print(f"Zero-test sample solver (relaxed, k={k})")
        print(f"  p=KoalaBear, r={R}, d={D}, ell={ELL}, t_perm={T_PERM}, "
              f"RF={RF}, RP={RP}")
        print(f"  Expected success prob per attempt: (1/2^{k})^2 = "
              f"1/{2**(2*k)}  (~{2**(2*k)} attempts needed)")
        print()

    t0 = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        # ----------------------------------------------------------------
        # Step 1: random polynomial of random degree in [1, D] over F_{p^2}
        # ----------------------------------------------------------------
        degree = rng.randint(1, D)

        # Non-zero leading coefficient ensures the polynomial has exactly
        # the chosen degree.
        def _rand_ext_nonzero() -> tuple[int, int]:
            while True:
                a = rng.randint(0, P - 1)
                b = rng.randint(0, P - 1)
                if a != 0 or b != 0:
                    return (a, b)

        coeffs: list[tuple[int, int]] = (
            [(rng.randint(0, P - 1), rng.randint(0, P - 1)) for _ in range(degree)]
            + [_rand_ext_nonzero()]                    # c_degree != (0,0)
            + [(0, 0)] * (D - degree)                 # zero-pad to D+1 entries
        )
        # coeffs[j] is the degree-j ext-field coefficient, len = D+1 = 8.

        # ----------------------------------------------------------------
        # Step 2: flatten to base-field vector P_hat
        # ----------------------------------------------------------------
        P_hat = [x for c in coeffs for x in c]  # length = (D+1)*R = 16

        # ----------------------------------------------------------------
        # Step 3: hash P_hat with Poseidon1 (compression mode)
        # ----------------------------------------------------------------
        hash_output = pos.compression_mode_hash(P_hat, out_length=ELL * R)

        # Group the first ELL*R base-field outputs into ELL ext-field words.
        a: list[tuple[int, int]] = [
            (hash_output[i * R] % P, hash_output[i * R + 1] % P)
            for i in range(ELL)
        ]

        # ----------------------------------------------------------------
        # Step 4 & 5: evaluate P(a_0) and check lowest K bits
        # ----------------------------------------------------------------
        c0, c1 = _ext_eval_poly(coeffs, a[0], P, beta=_EXT_BETA)

        if (c0 & mask) == 0 and (c1 & mask) == 0:
            # ----------------------------------------------------------------
            # Step 6: confirm with the full relaxed verifier
            # ----------------------------------------------------------------
            assert verify_zerotest_solution_relaxed(
                P_hat, k=k,
                prime=P, r=R, d=D, ell=ELL, s=S,
                r_f=RF, r_p=RP, t_perm=T_PERM, alpha=ALPHA,
            ), "verify_zerotest_solution_relaxed rejected a candidate that passed the inline check"

            elapsed = time.perf_counter() - t0
            if verbose:
                # Display coefficients as ext-field elements
                nonzero_coeffs = [(j, coeffs[j]) for j in range(D + 1) if coeffs[j] != (0, 0)]
                print(f"Solution found after {attempt} attempt(s) "
                      f"({elapsed * 1000:.2f} ms)")
                print(f"  degree          = {degree}")
                print(f"  non-zero coeffs = {nonzero_coeffs}  "
                      f"(format: [(degree, (a, b)), ...])")
                print(f"  H(P_hat)[0]     = {a[0]}  (a_0 in F_{{p^2}})")
                print(f"  P(a_0)          = ({c0}, {c1})")
                print(f"  low {k} bits: c0 & 0x{mask:X} = {c0 & mask}  "
                      f"c1 & 0x{mask:X} = {c1 & mask}  (both 0)  OK")
                print(f"  verify_zerotest_solution_relaxed(k={k}) -> True  OK")
            return P_hat, attempt

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"No solution found after {max_attempts} attempts "
              f"({elapsed * 1000:.2f} ms).")
    return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-test challenge sample solver (relaxed instance)."
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Number of low bits that must be zero in each component of P(a_0) "
             f"(default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=100_000,
        help="Maximum number of random trials (default: 100000).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    result = solve(
        k=args.k,
        max_attempts=args.max_attempts,
        seed=args.seed,
        verbose=not args.quiet,
    )
    sys.exit(0 if result is not None else 1)
