"""
Finite-field arithmetic for the Poseidon challenge suite.

Base field F_p (KoalaBear) is just Python ints mod p -- we keep thin helpers
so the attack code reads in field terms and so a later Rust/C++ port has a
1:1 reference.  The quadratic extension F_{p^2} = F_p[x]/(x^2 - beta) is needed
by the zero-test challenge.

Correctness-first (pure Python, exact). Speed-critical paths live in the native
Rust core.
"""

from .constants import P


# --- F_p -------------------------------------------------------------------

def fp(x: int, p: int = P) -> int:
    return x % p


def fp_inv(x: int, p: int = P) -> int:
    return pow(x % p, -1, p)


def fp_pow(x: int, e: int, p: int = P) -> int:
    return pow(x % p, e, p)


# --- F_{p^2} = F_p[x] / (x^2 - beta),  elements are (a, b) = a + b*sqrt(beta) ---
# This matches reference/poseidon-tools/bounties/zerotest_verifier.py exactly
# (beta = 3, which is a quadratic non-residue mod KoalaBear).

def f2_add(a, b, p: int = P):
    return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)


def f2_sub(a, b, p: int = P):
    return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)


def f2_mul(a, b, beta: int, p: int = P):
    # (a0 + a1 x)(b0 + b1 x) = (a0 b0 + beta a1 b1) + (a0 b1 + a1 b0) x
    c0 = (a[0] * b[0] + beta * a[1] * b[1]) % p
    c1 = (a[0] * b[1] + a[1] * b[0]) % p
    return (c0, c1)


def f2_eval_poly(coeffs, x, beta: int, p: int = P):
    """Horner evaluation of a univariate poly over F_{p^2}.

    coeffs[j] is the degree-j coefficient as an (a, b) pair.
    """
    if not coeffs:
        return (0, 0)
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = f2_mul(result, x, beta, p)
        result = f2_add(result, c, p)
    return result
