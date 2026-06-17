"""
Validate poseidon_attack.poly primitives against sympy over GF(p).

sympy is an independent, trusted oracle (not the blocked reference repo).
Covers: mul, divmod, gcd, resultant, root-finding, interpolation.

Run:  python3 tests/test_poly.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy
from sympy import Poly, symbols, GF, resultant

from poseidon_attack.constants import P
from poseidon_attack import poly as pl

x = symbols("x")
rng = random.Random(7)
_checks = []


def check(name, cond, detail=""):
    _checks.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def to_sympy(f, p=P):
    return Poly(list(reversed(f)) or [0], x, domain=GF(p, symmetric=False))


def from_sympy(poly, p=P):
    d = poly.all_coeffs()
    return pl._trim([int(c) % p for c in reversed(d)])


def rpoly(dmax, p=P):
    d = rng.randint(0, dmax)
    return pl._trim([rng.randrange(p) for _ in range(d + 1)]) or [rng.randrange(1, p)]


# --- arithmetic vs sympy ---------------------------------------------------
print("1. Arithmetic vs sympy (mul, divmod, gcd) over GF(p), small p and KoalaBear p")
for p in (1009, P):
    okmul = okdiv = okgcd = True
    for _ in range(60):
        a, b = rpoly(8, p), rpoly(6, p)
        # mul
        if from_sympy(to_sympy(a, p) * to_sympy(b, p), p) != pl.pmul(a, b, p):
            okmul = False
        # divmod (b nonzero)
        q, r = pl.pdivmod(a, b, p)
        sq, sr = divmod(to_sympy(a, p), to_sympy(b, p))
        if from_sympy(sq, p) != q or from_sympy(sr, p) != r:
            okdiv = False
        # gcd (monic)
        g = pl.pgcd(a, b, p)
        sg = from_sympy(sympy.gcd(to_sympy(a, p), to_sympy(b, p)), p)
        if g != sg:
            okgcd = False
    check(f"pmul  matches sympy (p={p})", okmul)
    check(f"pdivmod matches sympy (p={p})", okdiv)
    check(f"pgcd  matches sympy (p={p})", okgcd)

# --- resultant vs Sylvester determinant (the textbook definition) ----------
# NOTE: sympy's resultant(f,g) uses the opposite Sylvester block order, so it
# differs from the definition by the known sign (-1)^{deg f * deg g}.  Only
# Res==0 matters for the attack, so we validate against the Sylvester
# determinant directly and additionally assert mine in {+sympy, -sympy}.
print("2. Resultant vs Sylvester determinant (definition) + sympy up to sign")


def sylvester_res(A, B, p):
    A, B = pl._trim(A), pl._trim(B)
    if not A or not B:
        return 0
    m, n = len(A) - 1, len(B) - 1
    N = m + n
    if N == 0:
        return 1
    Ah, Bh = list(reversed(A)), list(reversed(B))      # high -> low
    M = [[0] * N for _ in range(N)]
    for i in range(n):
        for j, c in enumerate(Ah):
            M[i][i + j] = c % p
    for i in range(m):
        for j, c in enumerate(Bh):
            M[n + i][i + j] = c % p
    det = 1
    for col in range(N):
        piv = next((r for r in range(col, N) if M[r][col] % p), None)
        if piv is None:
            return 0
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            det = (-det) % p
        det = (det * M[col][col]) % p
        inv = pow(M[col][col], -1, p)
        for r in range(col + 1, N):
            f = (M[r][col] * inv) % p
            if f:
                M[r] = [(M[r][k] - f * M[col][k]) % p for k in range(N)]
    return det % p


for p in (1009, P):
    okdef = oksign = True
    for _ in range(60):
        a, b = rpoly(7, p), rpoly(5, p)
        mine = pl.presultant(a, b, p)
        if mine != sylvester_res(a, b, p):
            okdef = False
            break
        sym = int(resultant(to_sympy(a, p), to_sympy(b, p))) % p
        if mine != sym and mine != (-sym) % p:
            oksign = False
            break
    check(f"presultant == Sylvester determinant (p={p})", okdef)
    check(f"presultant == +/- sympy (p={p})", oksign)
# resultant is 0 iff common root
g = pl.pmul([3, 1], [5, 1], P)          # (x+3)(x+5)
h = pl.pmul([3, 1], [9, 1], P)          # (x+3)(x+9)  -> shares root -3
check("presultant == 0 for polys with common factor", pl.presultant(g, h, P) == 0)

# --- root finding ----------------------------------------------------------
print("3. F_p root-finding")
okroots = True
for _ in range(20):
    roots = sorted({rng.randrange(P) for _ in range(rng.randint(1, 6))})
    f = [1]
    for r in roots:
        f = pl.pmul(f, [(-r) % P, 1], P)        # product (x - r)
    # add an irreducible-ish quadratic factor sometimes (no extra F_p roots, generically)
    found = pl.proots(f, P)
    if not set(roots).issubset(set(found)):
        okroots = False
        print(f"      missing roots: wanted {roots} got {found}")
        break
check("proots finds all planted roots", okroots)
check("proots([]) == [] and constant has no roots", pl.proots([], P) == [] and pl.proots([7], P) == [])

# --- interpolation ---------------------------------------------------------
print("4. Interpolation round-trips")
okint = True
for _ in range(40):
    f = rpoly(10, P)
    xs = random.sample(range(P), len(f) + 2)
    ys = [pl.peval(f, xi, P) for xi in xs]
    if pl.interpolate(xs, ys, P) != f:
        okint = False
        break
check("interpolate recovers the polynomial", okint)

passed = sum(_checks)
print(f"\n=== {passed}/{len(_checks)} checks passed ===")
sys.exit(0 if passed == len(_checks) else 1)
