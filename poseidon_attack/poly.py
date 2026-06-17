"""
Univariate polynomial arithmetic over F_p (KoalaBear), plus resultant and
F_p root-finding -- the solver primitives for the CICO-2 algebraic attack.

Polynomials are Python lists of coefficients, low degree first; the zero
polynomial is [].  Pure Python and correctness-first; validated against sympy
in tests/test_poly.py.  A later native port (Rust/C++ + NTT) replaces this hot
path; this module is the executable reference.
"""

import random
from .constants import P


def _trim(f):
    f = list(f)
    while f and f[-1] == 0:
        f.pop()
    return f


def deg(f):
    f = _trim(f)
    return len(f) - 1


def padd(a, b, p=P):
    n = max(len(a), len(b))
    return _trim([((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)])


def psub(a, b, p=P):
    n = max(len(a), len(b))
    return _trim([((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)])


def pscalar(a, c, p=P):
    c %= p
    if c == 0:
        return []
    return _trim([(x * c) % p for x in a])


def pmul(a, b, p=P):
    if not a or not b:
        return []
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                res[i + j] = (res[i + j] + ai * bj) % p
    return _trim(res)


def pdivmod(a, b, p=P):
    a = _trim(a)
    b = _trim(b)
    if not b:
        raise ZeroDivisionError("division by zero polynomial")
    if len(a) < len(b):
        return [], a
    inv = pow(b[-1], -1, p)
    a = a[:]
    q = [0] * (len(a) - len(b) + 1)
    for shift in range(len(a) - len(b), -1, -1):
        if len(a) - 1 < len(b) - 1 + shift:
            continue
        coef = (a[len(b) - 1 + shift] * inv) % p
        q[shift] = coef
        if coef:
            for i, bi in enumerate(b):
                a[shift + i] = (a[shift + i] - coef * bi) % p
    return _trim(q), _trim(a)


def pmod(a, b, p=P):
    return pdivmod(a, b, p)[1]


def pgcd(a, b, p=P):
    a, b = _trim(a), _trim(b)
    while b:
        a, b = b, pmod(a, b, p)
    if a:  # make monic
        inv = pow(a[-1], -1, p)
        a = [(c * inv) % p for c in a]
    return a


def peval(f, x, p=P):
    acc = 0
    for c in reversed(f):
        acc = (acc * x + c) % p
    return acc


def pderiv(f, p=P):
    return _trim([(i * f[i]) % p for i in range(1, len(f))])


def ppow_mod(base, e, mod, p=P):
    """base^e mod `mod`, over F_p[x]."""
    result = [1]
    base = pmod(base, mod, p)
    while e:
        if e & 1:
            result = pmod(pmul(result, base, p), mod, p)
        base = pmod(pmul(base, base, p), mod, p)
        e >>= 1
    return result


def presultant(a, b, p=P):
    """Resultant Res(a, b) over F_p via the Euclidean remainder sequence."""
    a, b = _trim(a), _trim(b)
    if not a or not b:
        return 0
    res = 1
    if deg(a) < deg(b):
        a, b = b, a
        if (deg(a) % 2 == 1) and (deg(b) % 2 == 1):
            res = (-res) % p
    while deg(b) > 0:
        da, db = deg(a), deg(b)
        r = pmod(a, b, p)
        if not r:               # b | a  =>  shared factor  =>  resultant 0
            return 0
        if (da * db) % 2 == 1:
            res = (-res) % p
        res = (res * pow(b[-1], da - deg(r), p)) % p
        a, b = b, r
    # deg(b) == 0:  Res(a, const) = const^deg(a)
    res = (res * pow(b[0], deg(a), p)) % p
    return res % p


def _split_roots(g, p, rng):
    """Distinct F_p roots of a squarefree, fully-split poly g (Cantor-Zassenhaus)."""
    g = _trim(g)
    d = deg(g)
    if d <= 0:
        return []
    inv = pow(g[-1], -1, p)
    g = [(c * inv) % p for c in g]      # monic
    if d == 1:
        return [(-g[0]) % p]            # x + g0  ->  root -g0
    while True:
        a = rng.randrange(p)
        h = ppow_mod([a % p, 1], (p - 1) // 2, g, p)
        c = pgcd(g, psub(h, [1], p), p)
        if 0 < deg(c) < d:
            other = pdivmod(g, c, p)[0]
            return _split_roots(c, p, rng) + _split_roots(other, p, rng)


def proots(f, p=P, seed=2026):
    """All distinct roots of f in F_p."""
    f = _trim(f)
    if deg(f) <= 0:
        return []
    # g = gcd(f, x^p - x) is the product of (x - r) over distinct r in F_p
    xp = ppow_mod([0, 1], p, f, p)
    g = pgcd(f, psub(xp, [0, 1], p), p)
    if deg(g) <= 0:
        return []
    roots = _split_roots(g, p, random.Random(seed))
    return sorted(set(roots))


def interpolate(xs, ys, p=P):
    """Newton interpolation: smallest-degree poly through the (xs, ys) points."""
    n = len(xs)
    # divided differences
    coef = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            denom = (xs[i] - xs[i - j]) % p
            coef[i] = ((coef[i] - coef[i - 1]) * pow(denom, -1, p)) % p
    # convert Newton form to monomial coefficients
    poly = [coef[n - 1]]
    for i in range(n - 2, -1, -1):
        poly = padd(pmul(poly, [(-xs[i]) % p, 1], p), [coef[i]], p)
    return _trim(poly)
