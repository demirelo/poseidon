"""
Sample solver for a relaxed CICO challenge instance.

Reference: bounty2026.tex §2.4

This is a relaxed instance designed to demonstrate that solutions are easy to
find when only the lowest m bits of each constrained output difference are
required to be zero:

    p       = 2^31 - 2^24 + 1  (KoalaBear prime)
    alpha   = 3
    k       = 2    constrained input / output words
    t       = 16   Poseidon permutation state width
    R_F     = 6,   R_P = 8

Fixed constants:
    C_1 = 0xC09DE4,  C_2 = 0xEE6282,  C_3 = C_4 = 0

Challenge (relaxed, default m=5):
    Find free inputs (x_3, …, x_16) ∈ F_p^{t-k} = F_p^14 such that, writing

        s = (C_1, C_2, x_3, …, x_16)
        y = Poseidon::permutation_plus_linear(s)

    the lowest m bits of each constrained output residue are zero:

        (y[i] - C_{k+1+i}) & ((1 << m) - 1) == 0   for i = 0, …, k-1

Strategy:
    Repeat until success:
      1. Draw t-k random field elements as free_inputs.
      2. Build the full state s = [C_1, …, C_k] + free_inputs.
    3. Apply Poseidon::permutation_plus_linear(s) to get output y.
      4. Accept if (y[i] - target[i]) & mask == 0 for all i in [k].
      5. Double-check via verify_cico_solution_relaxed() before returning.

Expected success probability per attempt: (1/2^m)^k
Expected attempts needed: ~2^(m*k)
    m=5, k=2 → ~1 024 attempts
    m=6, k=2 → ~4 096 attempts
"""

import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from bounties.cico_verifier import (
    verify_cico_solution_relaxed,
    CICO_P,
    CICO_ALPHA,
    CICO_K,
    CICO_T,
    CICO_RF,
    CICO_RP,
    CICO_CONSTANTS,
)

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

P          = CICO_P         # 2^31 - 2^24 + 1  (KoalaBear)
ALPHA      = CICO_ALPHA     # 3
K          = CICO_K         # 2  (constrained words)
T          = CICO_T         # 16 (state width)
RF         = CICO_RF        # 6
RP         = CICO_RP        # 8
CONSTANTS  = CICO_CONSTANTS # [C_1, C_2, C_3, C_4]
DEFAULT_M  = 5              # default: lowest 5 bits of each output diff must be zero


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(
    m: int = DEFAULT_M,
    k: int = K,
    t: int = T,
    r_f: int = RF,
    r_p: int = RP,
    constants: list[int] | None = None,
    max_attempts: int = 10_000_000,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[int], int] | None:
    """
    Search for a relaxed solution to the CICO challenge.

    Strategy:
        Repeat up to max_attempts times:
          1. Draw t-k random field elements as the free portion of the input.
          2. Build the full state s = [C_1, …, C_k] + free_inputs.
          3. Apply Poseidon::permutation_plus_linear(s) → output y.
          4. Accept if (y[i] - target[i]) & mask == 0  for all i in [k].
          5. Double-check via verify_cico_solution_relaxed() before returning.

    Args:
        m:            Number of least-significant bits that must be zero in each
                      constrained output difference.  Must be >= 0.
        k:            Number of constrained input / output words.
        t:            Poseidon permutation state width.
        r_f:          Number of full rounds (must be even).
        r_p:          Number of partial rounds.
        constants:    List of 2k field elements [C_1, …, C_{2k}].
                      Defaults to CICO_CONSTANTS.
        max_attempts: Maximum number of random trials before giving up.
        seed:         Optional RNG seed for reproducibility.
        verbose:      Print progress and result.

    Returns:
        (free_inputs, attempt_number) if a solution is found, else None.
    """
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if constants is None:
        constants = CONSTANTS

    free_len = t - k
    if free_len <= 0:
        raise ValueError(f"t-k must be positive, got t={t}, k={k}")
    if len(constants) < 2 * k:
        raise ValueError(
            f"constants must have at least 2*k={2 * k} elements, "
            f"got {len(constants)}"
        )

    rng = random.Random(seed)
    mask = (1 << m) - 1
    targets = [int(constants[k + i]) % P for i in range(k)]
    input_prefix = [int(c) % P for c in constants[:k]]

    # Build the Poseidon instance once (round constants derived from Grain LFSR).
    pos = Poseidon(
        prime=P,
        alpha=ALPHA,
        t=t,
        r_f=r_f,
        r_p=r_p,
    )

    expected_attempts = 2 ** (m * k)

    if verbose:
        print(f"CICO sample solver (relaxed, m={m})")
        print(f"  p=KoalaBear, k={k}, t={t}, RF={r_f}, RP={r_p}")
        print(f"  free inputs: t-k = {free_len} field elements")
        print(f"  input  constants: {[hex(c) for c in constants[:k]]}")
        print(f"  output targets:   {[hex(tgt) for tgt in targets]}")
        print(f"  Expected success prob per attempt: "
              f"(1/2^{m})^{k} = 1/{expected_attempts}  "
              f"(~{expected_attempts} attempts needed)")
        print()

    t0 = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        # ----------------------------------------------------------------
        # Step 1: random free inputs
        # ----------------------------------------------------------------
        free_inputs = [rng.randint(0, P - 1) for _ in range(free_len)]

        # ----------------------------------------------------------------
        # Step 2 & 3: build full state and apply permutation
        # ----------------------------------------------------------------
        state_in = input_prefix + free_inputs
        y = pos.permutation_plus_linear(state_in)

        # ----------------------------------------------------------------
        # Step 4: check lowest m bits of each constrained output difference
        # ----------------------------------------------------------------
        if all((y[i] - targets[i]) & mask == 0 for i in range(k)):
            # ----------------------------------------------------------------
            # Step 5: confirm with the full relaxed verifier
            # ----------------------------------------------------------------
            assert verify_cico_solution_relaxed(
                free_inputs, m=m,
                prime=P, alpha=ALPHA, k=k, t=t,
                r_f=r_f, r_p=r_p, constants=constants,
            ), "verify_cico_solution_relaxed rejected a candidate that passed the inline check"

            elapsed = time.perf_counter() - t0
            if verbose:
                diffs = [(y[i] - targets[i]) % P for i in range(k)]
                print(f"Solution found after {attempt} attempt(s) "
                      f"({elapsed * 1000:.2f} ms)")
                print(f"  free_inputs (first 4) = {free_inputs[:4]}  …")
                print(f"  y[0:{k}]       = {y[:k]}")
                print(f"  targets        = {targets}")
                print(f"  diffs mod p    = {diffs}")
                print(f"  low {m} bits:  "
                      + "  ".join(
                          f"diff[{i}] & 0x{mask:X} = {diffs[i] & mask}"
                          for i in range(k)
                      )
                      + "  (all 0)  OK")
                print(f"  verify_cico_solution_relaxed(m={m}) -> True  OK")
            return free_inputs, attempt

        if verbose and attempt % 50_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  … {attempt} attempts so far ({elapsed:.1f}s)")

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
        description="CICO challenge sample solver (relaxed instance)."
    )
    parser.add_argument(
        "--m", type=int, default=DEFAULT_M,
        help=f"Number of low bits that must be zero in each constrained "
             f"output difference (default: {DEFAULT_M}).",
    )
    parser.add_argument(
        "--k", type=int, default=K,
        help=f"Number of constrained input/output words (default: {K}).",
    )
    parser.add_argument(
        "--rp", type=int, default=RP,
        help=f"Number of partial rounds R_P (default: {RP}).",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=10_000_000,
        help="Maximum number of random trials (default: 10000000).",
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

    # For non-default k, generate random constants of the required length 2k.
    constants = CONSTANTS
    if args.k != K:
        rng = random.Random(0)
        constants = [rng.randint(0, P - 1) for _ in range(2 * args.k)]
        print(f"  (using random constants for k={args.k}: {[hex(c) for c in constants]})")

    result = solve(
        m=args.m,
        k=args.k,
        r_p=args.rp,
        constants=constants,
        max_attempts=args.max_attempts,
        seed=args.seed,
        verbose=not args.quiet,
    )
    sys.exit(0 if result is not None else 1)
