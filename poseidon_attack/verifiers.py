"""
Independent verifiers for the four 2026 Poseidon bounty predicates.

These re-implement the *acceptance predicates* from bounty2026.tex / the
reference verifiers, on top of our own Poseidon1 primitive.  They are what we
use to self-check candidate solutions before submission.

Scope note: these helpers target the default Cauchy/Toeplitz poseidon-tools
instance used throughout the paper.  The organizers also accept arbitrary MDS
matrices that satisfy the invariant-subspace requirements; a custom-MDS
submission path must therefore publish and verify that matrix explicitly.  We
construct the default Cauchy matrix here and omit re-deriving the subspace-trail
machinery for custom matrices.
"""

from .constants import P, ALPHA, T, CICO, COLLISION, DENSITY, ZEROTEST
from .poseidon1 import Poseidon1
from .field import f2_eval_poly


# ---------------------------------------------------------------------------
# CICO-2   (bounty2026.tex §3.4)
# ---------------------------------------------------------------------------

def verify_cico(free_inputs, r_p, constants=None, prime=P, alpha=ALPHA, k=2, t=T,
                r_f=None, mds=None, round_constants=None):
    """True iff permutation_plus_linear([C1,C2] + free_inputs)[:k] == [C3,C4].

    free_inputs has length t-k = 14; for the bounty C3 = C4 = 0.
    """
    if r_f is None:
        r_f = CICO["rf"]
    if constants is None:
        constants = CICO["constants"]
    if len(free_inputs) != t - k:
        raise ValueError(f"free_inputs must have length t-k={t-k}")
    state_in = [int(c) % prime for c in constants[:k]] + [int(x) % prime for x in free_inputs]
    pos = Poseidon1(prime, alpha, t, r_f, r_p, mds=mds, round_constants=round_constants)
    out = pos.permutation_plus_linear(state_in)
    return all(out[i] == int(constants[k + i]) % prime for i in range(k))


def cico_residual(free_inputs, r_p, constants=None, prime=P, alpha=ALPHA, k=2, t=T,
                  r_f=None, mds=None, round_constants=None):
    """Return the k constrained output words (target is all-zero).  Profiling aid."""
    if r_f is None:
        r_f = CICO["rf"]
    if constants is None:
        constants = CICO["constants"]
    state_in = [int(c) % prime for c in constants[:k]] + [int(x) % prime for x in free_inputs]
    pos = Poseidon1(prime, alpha, t, r_f, r_p, mds=mds, round_constants=round_constants)
    out = pos.permutation_plus_linear(state_in)
    return [(out[i] - int(constants[k + i]) % prime) % prime for i in range(k)]


# ---------------------------------------------------------------------------
# Partial collision   (bounty2026.tex §3.1)  -- FULL STRENGTH rf=8, rp=20
# ---------------------------------------------------------------------------

def _collision_hash(x, pos, seed, t_perm, ell):
    padded = ([seed] + [v % pos.p for v in x])[:t_perm]
    return pos.compression_mode_hash(padded, out_length=ell)


def verify_collision(x, y, t, prime=P, alpha=ALPHA, ell=None, r_f=None, r_p=None,
                     t_perm=None, seed=None, mds=None, round_constants=None):
    """True iff x != y and H(seed,x)[:t] == H(seed,y)[:t].  x,y have length t_perm-1."""
    ell = ell or COLLISION["ell"]
    r_f = r_f or COLLISION["rf"]
    r_p = r_p or COLLISION["rp"]
    t_perm = t_perm or COLLISION["t_perm"]
    seed = COLLISION["seed"] if seed is None else seed
    if not (1 <= t <= ell):
        raise ValueError("t must satisfy 1 <= t <= ell")
    if len(x) != t_perm - 1 or len(y) != t_perm - 1:
        raise ValueError(f"x,y must have length t_perm-1={t_perm-1}")
    xn = [v % prime for v in x]
    yn = [v % prime for v in y]
    if xn == yn:
        return False
    pos = Poseidon1(prime, alpha, t_perm, r_f, r_p, mds=mds, round_constants=round_constants)
    hx = _collision_hash(xn, pos, seed, t_perm, ell)
    hy = _collision_hash(yn, pos, seed, t_perm, ell)
    return hx[:t] == hy[:t]


# ---------------------------------------------------------------------------
# Density   (bounty2026.tex §3.2)
# ---------------------------------------------------------------------------

def _omega_table(prime, r, omega):
    return {pow(omega, e, prime): e for e in range(r)}


def decode(x, prime=P, r=None, omega=None):
    """Decode(x) = log_omega( x^((p-1)/r) ).  x==0 -> index 0 (special case)."""
    r = r or DENSITY["r"]
    omega = omega or DENSITY["omega"]
    x %= prime
    if x == 0:
        return 0
    y = pow(x, (prime - 1) // r, prime)
    return _omega_table(prime, r, omega)[y]


def verify_density(S, prime=P, alpha=ALPHA, d=None, r=None, ell=None, k=1,
                   r_f=None, r_p=None, t_perm=T, omega=None,
                   mds=None, round_constants=None):
    """True iff <= d zeros in S and all k*ell decoded outputs hit zero positions."""
    d = DENSITY["d"] if d is None else d
    r = r or DENSITY["r"]
    ell = ell or DENSITY["ell"]
    r_f = r_f or DENSITY["rf"]
    r_p = r_p or DENSITY["rp"]
    omega = omega or DENSITY["omega"]
    if len(S) != r:
        raise ValueError(f"S must have length r={r}")
    zero_positions = {i for i, v in enumerate(S) if v % prime == 0}
    if len(zero_positions) > d:
        return False
    padded = ([v % prime for v in S] + [0] * t_perm)[:t_perm]
    pos = Poseidon1(prime, alpha, t_perm, r_f, r_p, mds=mds, round_constants=round_constants)
    out = pos.compression_mode_hash(padded, out_length=ell)
    decoded = [decode(a, prime, r, omega) for a in out for _ in range(k)]
    return all(decoded[j] in zero_positions for j in range(k * ell))


# ---------------------------------------------------------------------------
# Zero-test   (bounty2026.tex §3.3)
# ---------------------------------------------------------------------------

def verify_zerotest(P_hat, prime=P, alpha=ALPHA, r=None, d=None, ell=None, s=1,
                    r_f=None, r_p=None, t_perm=T, ext_beta=None,
                    mds=None, round_constants=None):
    """True iff 1<=deg(P)<=d and P(a_0)=0 in F_{p^2}, where a_0=(out[0],out[1])."""
    r = r or ZEROTEST["r"]
    d = d or ZEROTEST["d"]
    ell = ell or ZEROTEST["ell"]
    r_f = r_f or ZEROTEST["rf"]
    r_p = r_p or ZEROTEST["rp"]
    ext_beta = ZEROTEST["ext_beta"] if ext_beta is None else ext_beta
    if s != 1 or r != 2:
        raise NotImplementedError("only s=1, r=2 supported")
    if len(P_hat) != (d + 1) * r:
        raise ValueError(f"P_hat must have length (d+1)*r={(d+1)*r}")
    coeffs = [(int(P_hat[j * r]) % prime, int(P_hat[j * r + 1]) % prime) for j in range(d + 1)]
    deg = max((j for j in range(d + 1) if coeffs[j] != (0, 0)), default=-1)
    if not (1 <= deg <= d):
        return False
    pos = Poseidon1(prime, alpha, t_perm, r_f, r_p, mds=mds, round_constants=round_constants)
    out = pos.compression_mode_hash([int(v) % prime for v in P_hat], out_length=ell * r)
    a0 = (out[0] % prime, out[1] % prime)
    return f2_eval_poly(coeffs, a0, ext_beta, prime) == (0, 0)
