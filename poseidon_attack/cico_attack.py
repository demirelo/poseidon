"""
Algebraic CICO attack on Poseidon1 (permutation_plus_linear).

This is the bivariate resultant + GCD pipeline (eprint 2026/150 style), built on
our own primitives:

  - perm_plus_linear_poly: the permutation run over the polynomial ring F_p[var]
    (the "symbolic permutation").  Validated to agree with the numeric one.
  - solve_cico1: 1 free variable, 1 constraint -> univariate root-finding.
  - solve_cico2: 2 free variables, 2 constraints -> Res_Y eliminates Y to a
    univariate R(X); roots of R(X) back-substitute (via gcd) to Y.

Degrees (no skip-first-rounds yet):
  forward deg(output) = 3^(R_F + R_P);  MEASURED resultant deg(R) = 3^(2*R_F + R_P)
  (NOT the naive Bezout product 3^(2(R_F+R_P)); confirmed in tests/test_skip_and_gcd.py
  and core_rs cico_golden).
The skip-first-rounds trick (2026/150) would lower the *solved* degree to the ideal
degree D_I = 3^(2(R_F-1)+R_P) -- a constant 9x refinement, not a prerequisite.
Here we solve the as-built system, which is correct but only tractable in pure
Python at reduced rounds.  Real RP>=6 needs the native core.
"""

import random
from . import poly as pl
from .constants import P, CICO
from .poseidon1 import Poseidon1
from .verifiers import verify_cico


# --- symbolic permutation over F_p[var] ------------------------------------

def _sbox_poly(f, alpha, p):
    if alpha == 3:
        return pl.pmul(pl.pmul(f, f, p), f, p)
    out = [1]
    for _ in range(alpha):
        out = pl.pmul(out, f, p)
    return out


def _mds_poly(state, mds, p):
    t = len(state)
    out = []
    for i in range(t):
        acc = []
        row = mds[i]
        for j in range(t):
            if state[j]:
                acc = pl.padd(acc, pl.pscalar(state[j], row[j], p), p)
        out.append(acc)
    return out


def perm_plus_linear_poly(pos, state):
    """permutation_plus_linear over F_p[var]; `state` is a list of t polynomials."""
    p, t = pos.p, pos.t
    s = [list(c) for c in state]
    s = _mds_poly(s, pos.mds, p)                      # initial linear layer
    half, idx = pos.r_f // 2, 0
    for _ in range(half):
        s = [pl.padd(s[i], [pos.rc[idx][i]], p) for i in range(t)]
        s = [_sbox_poly(v, pos.alpha, p) for v in s]
        s = _mds_poly(s, pos.mds, p); idx += 1
    for _ in range(pos.r_p):
        s = [pl.padd(s[i], [pos.rc[idx][i]], p) for i in range(t)]
        s[0] = _sbox_poly(s[0], pos.alpha, p)
        s = _mds_poly(s, pos.mds, p); idx += 1
    for _ in range(half):
        s = [pl.padd(s[i], [pos.rc[idx][i]], p) for i in range(t)]
        s = [_sbox_poly(v, pos.alpha, p) for v in s]
        s = _mds_poly(s, pos.mds, p); idx += 1
    return s


def _build_state(pos, constants, k, assignment):
    """positions 0..k-1 = constants[:k]; positions k.. = polys from `assignment`."""
    return [[constants[i] % pos.p] if i < k else assignment[i] for i in range(pos.t)]


def forward_degree(pos, k=2):
    """deg_Y of output[0] with a single free variable at position k (scaling probe)."""
    assign = {i: ([0, 1] if i == k else [(i * 7 + 1) % pos.p]) for i in range(k, pos.t)}
    out0 = perm_plus_linear_poly(pos, _build_state(pos, CICO["constants"], k, assign))[0]
    return pl.deg(out0)


# --- CICO-1: one free variable, one constraint -----------------------------

def solve_cico1(pos, constants=None, tries=12, seed=2026, var_pos=1):
    """Find free_inputs (length t-1) with permutation_plus_linear(...)[0]==C2.

    Reduced/representative instances only (degree 3^(R_F+R_P) must be rootable).
    Returns (free_inputs, info) or (None, info).
    """
    if constants is None:
        constants = (CICO["constants"][0], 0)           # (C1, target=0)
    p, t, k = pos.p, pos.t, 1
    rng = random.Random(seed)
    target = constants[1] % p
    for attempt in range(tries):
        fixed = {i: rng.randrange(p) for i in range(k, t) if i != var_pos}
        assign = {i: ([0, 1] if i == var_pos else [fixed[i]]) for i in range(k, t)}
        out0 = perm_plus_linear_poly(pos, _build_state(pos, constants, k, assign))[0]
        roots = pl.proots(pl.psub(out0, [target], p), p)
        for yv in roots:
            free = [yv if i == var_pos else fixed[i] for i in range(k, t)]
            if verify_cico(free, pos.r_p, constants=constants, k=k, t=t,
                           r_f=pos.r_f, mds=pos.mds,
                           round_constants=[c for row in pos.rc for c in row]):
                return free, {"attempts": attempt + 1, "deg": pl.deg(out0), "roots": len(roots)}
    return None, {"attempts": tries, "deg": pl.deg(out0)}


# --- CICO-2: two free variables, two constraints (resultant route) ----------

def solve_cico2(pos, constants=None, tries=8, seed=2026, var_pos=(2, 3)):
    """Find free_inputs (length t-2) with permutation_plus_linear(...)[:2]==(C3,C4).

    Res_Y(out0, out1) -> R(X); roots of R(X) back-substitute via gcd to Y.
    Reduced rounds only.  Returns (free_inputs, info) or (None, info).
    """
    if constants is None:
        constants = CICO["constants"]                   # (C1, C2, 0, 0)
    p, t, k = pos.p, pos.t, 2
    px, py = var_pos
    rng = random.Random(seed)
    rc_flat = [c for row in pos.rc for c in row]
    t0 = constants[2] % p
    t1 = constants[3] % p

    def outs_at_X(xval, fixed):
        assign = {}
        for i in range(k, t):
            if i == px:
                assign[i] = [xval]
            elif i == py:
                assign[i] = [0, 1]                      # Y
            else:
                assign[i] = [fixed[i]]
        s = perm_plus_linear_poly(pos, _build_state(pos, constants, k, assign))
        return pl.psub(s[0], [t0], p), pl.psub(s[1], [t1], p)

    for attempt in range(tries):
        fixed = {i: rng.randrange(p) for i in range(k, t) if i not in var_pos}
        # degree bound for R(X): measured law deg(R) = 3^(2*R_F + R_P); +1 guard node
        degree_bound = 3 ** (2 * pos.r_f + pos.r_p)
        n_nodes = min(p - 1, degree_bound + 1)
        xs, ys = [], []
        xv = 1
        while len(xs) < n_nodes and xv < p:
            f0, f1 = outs_at_X(xv, fixed)
            xs.append(xv)
            ys.append(pl.presultant(f0, f1, p))
            xv += 1
        R = pl.interpolate(xs, ys, p)
        for xroot in pl.proots(R, p):
            f0, f1 = outs_at_X(xroot, fixed)
            g = pl.pgcd(f0, f1, p)
            for yroot in pl.proots(g, p):
                free = []
                for i in range(k, t):
                    free.append(xroot if i == px else (yroot if i == py else fixed[i]))
                if verify_cico(free, pos.r_p, constants=constants, k=k, t=t,
                               r_f=pos.r_f, mds=pos.mds, round_constants=rc_flat):
                    return free, {"attempts": attempt + 1, "deg_R": pl.deg(R),
                                  "nodes": len(xs)}
    return None, {"attempts": tries, "deg_R": pl.deg(R)}


def measure_resultant_degree(pos, constants=None, var_pos=(2, 3), seed=2026, guard=12):
    """True degree of R(X) = Res_Y(out0, out1) for the NAIVE bivariate system.

    MEASURED law (this module):  deg(R) = 3^(2*R_F + R_P).
    Skip-first-rounds (eprint 2026/150) reaches the ideal D_I = 3^(2*(R_F-1)+R_P),
    a CONSTANT factor 3^2 = 9 smaller -- i.e. our plain resultant is already
    within 9x of optimal in degree; the dominant scale lever is the native core,
    not skip.  Adaptive node growth: stop once the interpolant predicts `guard`
    fresh resultant values.
    """
    if constants is None:
        constants = CICO["constants"]
    p, t, k = pos.p, pos.t, 2
    px, py = var_pos
    rng = random.Random(seed)
    fixed = {i: rng.randrange(p) for i in range(k, t) if i not in var_pos}
    t0, t1 = constants[2] % p, constants[3] % p
    cache = {}

    def res_at(xv):
        if xv not in cache:
            assign = {}
            for i in range(k, t):
                assign[i] = [xv] if i == px else ([0, 1] if i == py else [fixed[i]])
            s = perm_plus_linear_poly(pos, _build_state(pos, constants, k, assign))
            cache[xv] = pl.presultant(pl.psub(s[0], [t0], p), pl.psub(s[1], [t1], p), p)
        return cache[xv]

    hard_cap = min(p - 1, 2 * (3 ** (pos.r_f + pos.r_p)) ** 2 + 1)
    n = 16
    while True:
        n = min(n, hard_cap)
        xs = list(range(1, n + 1))
        R = pl.interpolate(xs, [res_at(x) for x in xs], p)
        if n >= hard_cap or all(pl.peval(R, x, p) == res_at(x) for x in range(n + 1, n + 1 + guard)):
            return pl.deg(R)
        n *= 2


# --- CICO-2 GCD route (linear memory; the operative path at scale) ----------

def solve_cico2_gcd(pos, constants=None, lines=4000, seed=2026, var_pos=(2, 3),
                    base=None, direction=None):
    """GCD route: restrict the free inputs to a LINE  free(X) = base + X*direction,
    so out0, out1 become univariate in X; a shared F_p root of gcd(out0,out1)
    solves CICO-2.  Linear memory, embarrassingly parallel.

    For a random line the shared-root probability is ~1/p (a 1-D line meets the
    codimension-2 CICO-2 solution set with prob ~1/p), so ~p random lines are
    needed -- which is why at reduced rounds we plant a line through a known
    solution (base/direction) to unit-test the mechanism, and why a full-strength
    single-machine GCD-route solve is NO-GO (~p lines is cluster scale).
    Returns (free_inputs, info) or (None, info).
    """
    if constants is None:
        constants = CICO["constants"]
    p, t, k = pos.p, pos.t, 2
    rng = random.Random(seed)
    rc_flat = [c for row in pos.rc for c in row]
    t0, t1 = constants[2] % p, constants[3] % p
    nfree = t - k

    for line in range(lines):
        b = list(base) if base is not None else [rng.randrange(p) for _ in range(nfree)]
        d = list(direction) if direction is not None else [rng.randrange(p) for _ in range(nfree)]
        # free position (k+j) carries polynomial [b[j], d[j]]  (= b[j] + d[j]*X)
        assign = {k + j: [b[j] % p, d[j] % p] for j in range(nfree)}
        s = perm_plus_linear_poly(pos, _build_state(pos, constants, k, assign))
        f0, f1 = pl.psub(s[0], [t0], p), pl.psub(s[1], [t1], p)
        g = pl.pgcd(f0, f1, p)
        for xr in pl.proots(g, p):
            free = [(b[j] + xr * d[j]) % p for j in range(nfree)]
            if verify_cico(free, pos.r_p, constants=constants, k=k, t=t,
                           r_f=pos.r_f, mds=pos.mds, round_constants=rc_flat):
                return free, {"lines_tried": line + 1, "deg_gcd_inputs": pl.deg(f0)}
        if base is not None:        # caller pinned a single line; don't loop random
            break
    return None, {"lines_tried": lines}
