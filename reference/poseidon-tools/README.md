# poseidon-tools

Pure-Python implementation of the **Poseidon** cryptographic hash function,
plus verifiers and sample challengers for the
[Poseidon Initiative 2026 bounties](bounties/docs/bounty2026.tex) —
no SageMath or other heavy dependencies required.

## References

- Poseidon paper: <https://eprint.iacr.org/2019/458>
- Bounty spec: `bounties/docs/bounty2026.tex`

---

## Requirements

- Python 3.10+ (uses `pow(a, -1, prime)` for modular inverse, `list[int]` type hints)
- No third-party libraries for the core implementation

Running tests requires [pytest](https://pytest.org):

```bash
pip install pytest
```

---

## Installation

Clone the repository and (optionally) create a virtual environment:

```bash
git clone https://github.com/your-org/poseidon-tools.git
cd poseidon-tools
python -m venv .venv && .venv\Scripts\activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/macOS
```

---

## Poseidon API

```python
from poseidon.poseidon import Poseidon

# KoalaBear prime (used in the bounty challenges)
p = 2130706433  # 2^31 - 2^24 + 1

# Instantiate (t=16, R_F=8, R_P=20, alpha=3)
pos = Poseidon(prime=p, alpha=3, t=16, r_f=8, r_p=20)

# Compression-mode hash: inputs must have length t; returns list of out_length elements
outputs = pos.compression_mode_hash(inputs=[0]*16, out_length=16)

# Sponge hash: absorbs arbitrary-length input, returns one field element
digest = pos.hash([1, 2, 3])

# Raw permutation: state must have length t
state_out = pos.permutation([0] * 16)
```

### Constructor parameters

| Parameter | Description |
|-----------|-------------|
| `prime`   | Prime field modulus |
| `alpha`   | S-box exponent (use `-1` for the inverse S-box) |
| `t`       | State width in field elements |
| `r_f`     | Number of full rounds (must be even) |
| `r_p`     | Number of partial rounds |
| `rate`    | Sponge absorb rate (defaults to `t-1`, capacity = 1) |

---

## Bounty Challenges

All four challenges use **KoalaBear** (`p = 2^31 - 2^24 + 1 = 2130706433`, `alpha = 3`).
The `bounties/` package contains a verifier and a sample challenger for each.

- **Partial Collision, Density, Zero-Test:** standard Poseidon1 in **compression mode** (`t_perm = 16`, full-state rate)
- **CICO:** a modified Poseidon permutation with an **upfront linear layer**, exposed as `permutation_plus_linear`

---

### §2.1 — Partial Collision

Find two distinct inputs `x != y` in `F_p^(t_perm-1)` such that the first `t`
output words of `H(x)` and `H(y)` agree, where
`H(inputs) = compression_mode_hash([0] + inputs)`.

**Bounty parameters:** `ell=16, R_F=8, R_P=20, t_perm=16`.
The challenge target is `t = ell` (full output collision).
The sample challenger targets the tractable case `t = 1`.

#### Verifier

```python
from bounties.partial_collision_verifier import verify_collision_solution

# x, y — each a list of t_perm-1 = 15 field elements
ok = verify_collision_solution(
    x, y,
    t=1,                    # number of leading output words that must collide
    prime=2130706433,
    ell=16, r_f=8, r_p=20, t_perm=16, alpha=3,
)
```

`verify_collision_solution` raises `ValueError` if `x` or `y` have the wrong
length, or if `x == y`. Returns `True` on success, `False` on failure.

#### Sample Challenger (Floyd’s rho method)

```bash
python -m bounties.partial_collision_sample_challenger [--seed SEED] [--quiet]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | random | RNG seed for the starting point |
| `--quiet` | off | suppress per-step output |

Expected cost: ~3·√p ≈ 138 000 hash evaluations.

```python
from bounties.partial_collision_sample_challenger import solve

x, y, hash_calls = solve(seed=42, verbose=True)
# x and y are each 15-element lists; hash_calls is the total evaluation count
```

---

### §2.2 — Density

Find a vector `S` in `F_p^r` with at most `d` zero entries such that the
`t = k*ell` indices decoded from `H(S)` all land on zero positions of `S`.

**Bounty parameters:** `r=16, d=2, k=1, ell=16, t=16, R_F=6, R_P=6, t_perm=16`.
The sample challenger targets the tractable case `ell=2` (two decoded indices).

#### Verifier

```python
from bounties.density_verifier import verify_density_solution

# S — list of r = 16 field elements
ok = verify_density_solution(
    S,
    prime=2130706433,
    k=1, d=2, r=16, ell=16,
    r_f=6, r_p=6, t_perm=16, alpha=3,
)
```

Returns `True` on success, `False` on failure.

#### Sample Challenger (random search, simplified `ell=2` instance)

```bash
python -m bounties.density_sample_challenger [--max-attempts N] [--seed SEED] [--quiet]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-attempts` | 10000 | maximum random trials |
| `--seed` | random | RNG seed for reproducibility |
| `--quiet` | off | suppress per-attempt output |

Expected cost: ~64 attempts (probability (d/r)^2 = 1/64 per trial).

```python
from bounties.density_sample_challenger import solve

result = solve(seed=43, verbose=True)
if result is not None:
    S, attempts = result
```

Returns `(S, attempts)` where `S` is a 16-element list, or `None` if no
solution was found within `max_attempts`.

---

### §2.3 — Zero-Test

Find a univariate polynomial `P : F_{p^2} -> F_{p^2}` of degree 1 <= deg <= 7
such that `P` vanishes at the first hash output: `P(a_0) = 0` in `F_{p^2}`,
where `(a_0, ..., a_7) := H(P_hat)` and `P_hat` is the 16-element flat
coefficient vector of `P` over `F_p`.

Extension field: `F_{p^2} = F_p[x] / (x^2 - 3)`. Elements are stored as
`(c0, c1)` pairs representing `c0 + c1*sqrt(3)`.

**Bounty parameters:** `r=2, d=7, ell=8, s=1, R_F=6, R_P=6, t_perm=16`.

#### Verifier

```python
from bounties.zerotest_verifier import verify_zerotest_solution

# P_hat — flat list of 16 base-field coefficients [c0_0, c1_0, c0_1, c1_1, ...]
ok = verify_zerotest_solution(
    P_hat,
    prime=2130706433,
    r=2, d=7, ell=8, s=1,
    r_f=6, r_p=6, t_perm=16, alpha=3,
)
```

A relaxed variant checks only the lowest `k` bits of each component of `P(a_0)`:

```python
from bounties.zerotest_verifier import verify_zerotest_solution_relaxed

ok = verify_zerotest_solution_relaxed(
    P_hat, k=4,
    prime=2130706433,
    r=2, d=7, ell=8, s=1,
    r_f=6, r_p=6, t_perm=16, alpha=3,
)
```

Both functions return `True` on success, `False` on failure.

#### Sample Challenger (random search, relaxed `k`-bit instance)

```bash
python -m bounties.zerotest_sample_challenger [--k K] [--max-attempts N] [--seed SEED] [--quiet]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--k` | 4 | low bits of each component of P(a_0) that must be zero |
| `--max-attempts` | 100000 | maximum random trials |
| `--seed` | random | RNG seed for reproducibility |
| `--quiet` | off | suppress per-attempt output |

Expected cost: ~2^(2k) attempts (e.g. ~256 for k=4).

```python
from bounties.zerotest_sample_challenger import solve

result = solve(k=4, seed=42, verbose=True)
if result is not None:
    P_hat, attempts = result
```

Returns `(P_hat, attempts)` where `P_hat` is a 16-element list, or `None` if
no solution was found within `max_attempts`.

---

### §2.4 — CICO

Find free inputs `(x_{k+1}, ..., x_t)` in `F_p^(t-k)` such that, with
`s = (C_1, ..., C_k, x_{k+1}, ..., x_t)`, the first `k` output words of
`P⁺(s)` match the prescribed constants `C_{k+1}, ..., C_{2k}`. Here `P⁺`
denotes the repository's modified Poseidon permutation that applies an
additional linear layer first via `permutation_plus_linear`.

**Bounty parameters:** `k=2, t=16, R_F=6, R_P in {8, 10, 12}` with fixed
constants `C_1 = 0xC09DE4`, `C_2 = 0xEE6282`, `C_3 = C_4 = 0`.

#### Verifier

```python
from bounties.cico_verifier import verify_cico_solution

# free_inputs — list of t-k = 14 field elements
ok = verify_cico_solution(
  free_inputs,
  prime=2130706433,
  alpha=3,
  k=2, t=16,
  r_f=6, r_p=8,
)
```

A relaxed variant checks only the lowest `m` bits of each constrained output
difference:

```python
from bounties.cico_verifier import verify_cico_solution_relaxed

ok = verify_cico_solution_relaxed(
  free_inputs, m=5,
  prime=2130706433,
  alpha=3,
  k=2, t=16,
  r_f=6, r_p=8,
)
```

Both functions use `Poseidon.permutation_plus_linear()` internally. The exact
verifier returns `True` only for full matches; the relaxed verifier returns
`True` when each constrained difference is divisible by `2^m`.

#### Sample Challenger (random search, relaxed `m`-bit instance)

```bash
python -m bounties.cico_sample_challenger [--m M] [--k K] [--rp RP] [--max-attempts N] [--seed SEED] [--quiet]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--m` | 5 | low bits of each constrained output difference that must be zero |
| `--k` | 2 | number of constrained input/output words |
| `--rp` | 8 | number of partial rounds |
| `--max-attempts` | 10000000 | maximum random trials |
| `--seed` | random | RNG seed for reproducibility |
| `--quiet` | off | suppress per-attempt output |

Expected cost: ~`2^(m*k)` attempts (e.g. ~1024 for `m=5, k=2`).

```python
from bounties.cico_sample_challenger import solve

result = solve(m=5, seed=42, verbose=True)
if result is not None:
  free_inputs, attempts = result
```

Returns `(free_inputs, attempts)` where `free_inputs` is a 14-element list, or
`None` if no solution was found within `max_attempts`.

---

## Repository Layout

```
poseidon/
  grain_lfsr.py      – 80-bit Grain LFSR for round-constant generation
  mds_matrix.py      – Cauchy MDS matrix construction and application
  poseidon.py        – Poseidon permutation, sponge hash, compression-mode hash
bounties/
  docs/
    bounty2026.tex   – Poseidon Initiative 2026 bounty specification
  partial_collision_verifier.py          – §2.1 verifier
  partial_collision_sample_challenger.py – §2.1 sample solver (Floyd rho)
  density_verifier.py                    – §2.2 verifier
  density_sample_challenger.py           – §2.2 sample solver (random search)
  zerotest_verifier.py                   – §2.3 verifier (exact + relaxed)
  zerotest_sample_challenger.py          – §2.3 sample solver (random search)
  cico_verifier.py                       – §2.4 verifier (exact + relaxed, plus-linear Poseidon)
  cico_sample_challenger.py              – §2.4 sample solver (random search)
tests/
  test_poseidon.py
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```
