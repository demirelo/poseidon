"""
Sample solver for the partial collision challenge using Floyd's rho method.

Reference: bounty2026.tex §2.1

Instance parameters:
    p       = 2^31 - 2^24 + 1  (KoalaBear prime)
    alpha   = 3
    ell     = 16   hash output words
    t_perm  = 16   Poseidon1 state width  (rate = ell; full-state absorption)
    t       = 1    number of leading output words that must collide
    R_F     = 8,   R_P = 20

Challenge (t=1):
    Find x != y in F_p such that H([x])[0] = H([y])[0].

Rho method (Floyd's cycle detection — memoryless birthday attack):
    Define  f : F_p -> F_p  as  f(v) = H([v])[0].
    Walk the sequence  v_0, f(v_0), f(f(v_0)), ...

    By the birthday paradox a repeated value appears after ~sqrt(p) ≈ 2^15.5
    steps.  Floyd's algorithm detects the repeat in O(sqrt(p)) steps and O(1)
    memory (only two integers stored — tortoise and hare):

      Phase 1 — cycle detection:
        tortoise advances one step, hare advances two steps, until they meet.
        Meeting guarantees we are somewhere inside the cycle.

      Phase 2 — find cycle entry mu:
        Reset tortoise to v_0.  Advance both one step at a time until they
        meet.  The meeting point is v_mu (the first repeated value).

      Phase 3 — extract the collision:
        v_{mu-1} is the tail predecessor of v_mu  (mu > 0 guaranteed).
        Walk from v_mu around the cycle until we return to v_mu; the element
        just before v_mu in the cycle is the cycle predecessor.
        Both tail_pred and cycle_pred map to v_mu under f and are distinct
        (one lives in the tail, the other in the cycle).

Expected hash calls: ~3 * sqrt(p) ≈ 140 000.
"""

import random
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poseidon.poseidon import Poseidon
from bounties.partial_collision_verifier import (
    _hash,
    verify_collision_solution,
    COLLISION_P,
    COLLISION_ALPHA,
    COLLISION_ELL,
    COLLISION_RF,
    COLLISION_RP,
    COLLISION_TPERM,
)

# ---------------------------------------------------------------------------
# Instance parameters
# ---------------------------------------------------------------------------

P      = COLLISION_P        # 2^31 - 2^24 + 1  (KoalaBear)
ALPHA  = COLLISION_ALPHA    # 3
ELL    = COLLISION_ELL      # 16  (hash output words = rate)
T_PERM = COLLISION_TPERM    # 16  (state width)
RF     = COLLISION_RF       # 8
RP     = COLLISION_RP       # 20
T      = 1                  # number of leading output words that must collide
SEED = 0xc09de4

# ---------------------------------------------------------------------------
# Rho walk
# ---------------------------------------------------------------------------

def _f(v: int, pos: Poseidon) -> int:
    """
    One step of the rho walk.

    f(v) = H([SEED, v, 0, ..., 0])[0]  — hash a 15-element input (= t_perm-1)
    prefixed with the domain-separation constant SEED, then return the first
    output word.  The 15 elements are zero-padded to t_perm=16 by _hash before
    being passed to compression_mode_hash.

    Args:
        v:   Current field element.
        pos: Pre-built Poseidon instance.

    Returns:
        Next field element in the rho sequence.
    """
    return _hash([SEED, v] + [0] * (T_PERM - 3), pos, P, ELL, T_PERM)[0]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[int], list[int], int] | None:
    """
    Search for a t=1 partial collision using Floyd's rho method.

    The method is deterministic (given a starting point) and uses O(1) memory.
    A fresh random starting point is drawn for each restart; in practice
    a solution is always found on the first attempt.

    Algorithm:
        1. Pick a random v_0 in F_p.
        2. Phase 1 — Floyd cycle detection: advance tortoise / hare until
           they meet inside the cycle.
        3. Phase 2 — find cycle entry mu: reset tortoise to v_0, advance
           both one step at a time until they meet at v_mu.
        4. Phase 3 — extract collision:
             tail_pred  = v_{mu-1}  (walk mu-1 steps from v_0)
             cycle_pred = v_{mu+lambda-1}  (walk the cycle until just before
                          returning to v_mu)
           Both satisfy f(tail_pred) = v_mu = f(cycle_pred), with
           tail_pred != cycle_pred.
        5. Verify with verify_collision_solution() before returning.

    Args:
        seed:    Optional RNG seed for reproducibility.
        verbose: Print progress and result.

    Returns:
        (x, y, hash_calls) where x=[tail_pred], y=[cycle_pred], if successful.
        None if all restarts fail (should not occur in practice).
    """
    rng = random.Random(seed)

    pos = Poseidon(
        prime=P,
        alpha=ALPHA,
        t=T_PERM,
        r_f=RF,
        r_p=RP,
    )

    if verbose:
        import math
        expected = int(3 * math.isqrt(P))
        print(f"Partial collision solver — rho method, t={T}")
        print(f"  p=KoalaBear, ell={ELL}, t_perm={T_PERM}, RF={RF}, RP={RP}")
        print(f"  f(v) = H([v])[0],  expected hash calls: ~3*sqrt(p) ≈ {expected:,}")
        print()

    t0 = time.perf_counter()
    hash_calls = 0

    for restart in range(1, 1001):
        v0 = rng.randint(0, P - 1)

        # ------------------------------------------------------------------
        # Phase 1: Floyd's cycle detection
        # Invariant: tortoise = v_i, hare = v_{2i} after i iterations.
        # ------------------------------------------------------------------
        tortoise = _f(v0, pos);  hash_calls += 1
        hare     = _f(tortoise, pos);  hash_calls += 1  # hare = v_2

        while tortoise != hare:
            tortoise = _f(tortoise, pos);            hash_calls += 1
            hare     = _f(_f(hare, pos), pos);       hash_calls += 2

        # tortoise == hare == v_{mu + k*lambda} for some k >= 1

        # ------------------------------------------------------------------
        # Phase 2: find cycle entry mu
        # ------------------------------------------------------------------
        tortoise2 = v0
        mu = 0
        while tortoise2 != hare:
            tortoise2 = _f(tortoise2, pos);  hash_calls += 1
            hare      = _f(hare, pos);       hash_calls += 1
            mu += 1

        # tortoise2 == hare == v_mu  (first element of the cycle)

        if mu == 0:
            # v_0 is already inside the cycle — no tail predecessor exists.
            # This is astronomically rare for a large-domain function; restart.
            if verbose:
                print(f"  restart {restart}: mu=0 (start inside cycle), retrying...")
            continue

        x_mu = tortoise2   # = v_mu

        # ------------------------------------------------------------------
        # Phase 3a: tail predecessor  v_{mu-1}
        # Walk mu-1 steps from v_0.
        # ------------------------------------------------------------------
        tail_pred = v0
        for _ in range(mu - 1):
            tail_pred = _f(tail_pred, pos);  hash_calls += 1
        # f(tail_pred) = v_mu

        # ------------------------------------------------------------------
        # Phase 3b: cycle predecessor  v_{mu + lambda - 1}
        # Walk from v_mu until the next step returns to v_mu.
        # ------------------------------------------------------------------
        cycle_pred = x_mu
        while True:
            nxt = _f(cycle_pred, pos);  hash_calls += 1
            if nxt == x_mu:
                break
            cycle_pred = nxt
        # f(cycle_pred) = v_mu

        # Sanity: the two preimages must be distinct.
        if tail_pred == cycle_pred:
            # Pathological coincidence (value collision between tail and cycle).
            if verbose:
                print(f"  restart {restart}: tail_pred == cycle_pred (degenerate), retrying...")
            continue

        # ------------------------------------------------------------------
        # Verify with the official verifier
        # ------------------------------------------------------------------
        # Full preimage vectors: [SEED, pred, 0, ..., 0] with t_perm-1=15 elements
        x = [SEED, tail_pred]  + [0] * (T_PERM - 3)
        y = [SEED, cycle_pred] + [0] * (T_PERM - 3)

        assert verify_collision_solution(
            x, y, t=T,
            prime=P, ell=ELL, r_f=RF, r_p=RP, t_perm=T_PERM, alpha=ALPHA,
        ), "verify_collision_solution rejected the rho collision"

        elapsed = time.perf_counter() - t0

        if verbose:
            hash_x = _hash(x, pos, P, ELL, T_PERM)
            hash_y = _hash(y, pos, P, ELL, T_PERM)
            print(f"Collision found after {hash_calls:,} hash calls "
                  f"({elapsed * 1000:.2f} ms, {restart} restart(s))")
            print(f"  mu              = {mu}   (cycle entry)")
            print(f"  tail_pred       = {tail_pred}")
            print(f"  cycle_pred      = {cycle_pred}")
            print(f"  x = [SEED, tail_pred,  0*13] (len={len(x)})")
            print(f"  y = [SEED, cycle_pred, 0*13] (len={len(y)})")
            print(f"  H(x)[0]         = {hash_x[0]}")
            print(f"  H(y)[0]         = {hash_y[0]}  (match)")
            print(f"  verify_collision_solution(t={T}) -> True  OK")

        return x, y, hash_calls

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"No collision found after {hash_calls:,} hash calls "
              f"({elapsed * 1000:.2f} ms).")
    return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Partial collision sample solver (t=1, Floyd rho method)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for the starting point (default: random).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    result = solve(
        seed=args.seed,
        verbose=not args.quiet,
    )
    sys.exit(0 if result is not None else 1)
