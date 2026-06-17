"""
Sample solver for a simplified density challenge instance.

Reference: bounty2026.tex §2.2

This is a reduced instance designed to demonstrate that solutions are easy
to find when the number of required collisions (t) is small:

    p       = 2^31 - 2^24 + 1  (KoalaBear prime)
    alpha   = 3
    d       = 2    max zeros in S          (sparsity bound)
    r       = 16   len(S)
    k       = 1    Decode outputs per hash word
    ell     = 2    hash output words
    t_perm  = 16   Poseidon1 state width   (rate = 2, capacity = 14)
    t       = 2    decoded indices that must hit zero positions
    R_F     = 6,   R_P = 6

Challenge:
    Find S ∈ F_p^16 with #{i : S[i]=0} ≤ 2 and
    Decode(H(S)[0]) ∈ {zero positions of S} AND
    Decode(H(S)[1]) ∈ {zero positions of S}.

Decode (from density_verifier):
    Decode(x) = log_ω( x^((p-1)//r) )  mod r
    where ω = 148625052 is the primitive r-th root of unity in F_p.
    Gives a near-uniform distribution over {0, …, r-1} for random x,
    so the probability that both decoded indices hit zero positions
    is approximately (d/r)^2 = (2/16)^2 = 1/64.

Expected success probability per attempt: (d/r)^2 = 1/64
Expected attempts needed: ~64
"""

import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from bounties.density_verifier import decode, verify_density_solution

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

P       = 2130706433  # 2^31 - 2^24 + 1 (KoalaBear)
ALPHA   = 3
D       = 2           # max zeros in S
R       = 16          # len(S)
K       = 1           # Decode outputs per hash word
ELL     = 2           # hash output words  (rate = ELL = 2)
T_PERM  = 16          # Poseidon1 state width
T       = K * ELL     # decoded indices that must hit zero positions
RF      = 6
RP      = 6


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(
    max_attempts: int = 10_000,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[int], int] | None:
    """
    Search for a solution to the simplified density challenge.

    Strategy:
        Repeat up to max_attempts times:
          1. Pick d=2 distinct positions uniformly at random; set those to 0.
          2. Fill remaining R-d positions with random non-zero field elements.
          3. Hash S with Poseidon1 (t_perm=T_PERM, rate=ELL=2, RF=RF, RP=RP).
          4. Decode the ELL=2 output elements via density_verifier.decode().
          5. Accept if all decoded indices are zero positions.
          6. Double-check via verify_density_solution() before returning.

    Args:
        max_attempts: Maximum number of random trials before giving up.
        seed:         Optional RNG seed for reproducibility.
        verbose:      Print progress and result.

    Returns:
        (S, attempt_number) if a solution is found, else None.
    """
    rng = random.Random(seed)

    # Build the Poseidon instance once (round constants derived from Grain LFSR)
    pos = Poseidon(
        prime=P,
        alpha=ALPHA,
        t=T_PERM,
        r_f=RF,
        r_p=RP,
    )

    if verbose:
        print(f"Density sample solver — instance: p=KoalaBear, d={D}, r={R}, "
              f"ell={ELL}, t={T}, t_perm={T_PERM}, RF={RF}, RP={RP}")
        print(f"Expected success prob per attempt: ({D}/{R})^{ELL} ≈ {(D/R)**ELL:.4f}  "
              f"(~{(R//D)**ELL} attempts needed)")
        print()

    t0 = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        # Step 1 & 2: build a candidate S with exactly d zeros
        zero_positions = set(rng.sample(range(R), D))
        S = [
            0 if i in zero_positions
            else rng.randint(1, P - 1)
            for i in range(R)
        ]

        # Step 3: hash S with Poseidon1 in compression mode
        padded_S = (S + [0] * T_PERM)[:T_PERM]
        hash_output = pos.compression_mode_hash(padded_S, out_length=ELL)

        # Step 4 & 5: decode each output word and check all hit zero positions
        decoded = [decode(a) for a in hash_output]
        if all(idx in zero_positions for idx in decoded):
            # Step 6: confirm with the full verifier
            assert verify_density_solution(
                S,
                prime=P, d=D, r=R, t=T, k=K, ell=ELL,
                r_f=RF, r_p=RP, t_perm=T_PERM, alpha=ALPHA,
            ), "verify_density_solution rejected a candidate that passed the inline check"

            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"Solution found after {attempt} attempt(s) "
                      f"({elapsed*1000:.2f} ms)")
                print(f"  S              = {S}")
                print(f"  zero positions = {sorted(zero_positions)}")
                print(f"  H(S)           = {hash_output}")
                for j, (a, idx) in enumerate(zip(hash_output, decoded)):
                    print(f"  Decode(H(S)[{j}]) = {idx}  ✓ (S[{idx}] = {S[idx]})")
                print(f"  verify_density_solution → True  ✓")
            return S, attempt

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"No solution found after {max_attempts} attempts "
              f"({elapsed*1000:.2f} ms).")
    return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Density challenge sample solver (simplified instance)."
    )
    parser.add_argument(
        "--max-attempts", type=int, default=10_000,
        help="Maximum number of random trials (default: 10000).",
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
        max_attempts=args.max_attempts,
        seed=args.seed,
        verbose=not args.quiet,
    )
    sys.exit(0 if result is not None else 1)
