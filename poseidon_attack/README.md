# poseidon_attack

Independent, verifier-exact reimplementation of the Poseidon1 / KoalaBear
primitive, the four 2026-bounty predicates, and the algebraic CICO attack
pipeline. Built from the written spec and validated against published golden
vectors + an independent oracle (sympy) — **the reference repo is never
executed**.

This is the shared substrate used by the paper's CICO, density, collision, and
zero-test analyses.

## Modules

| Module | Contents |
|---|---|
| `constants.py` | Frozen instance parameters, each annotated with its primary source |
| `field.py` | `F_p` helpers and `F_{p^2}=F_p[x]/(x^2-3)` arithmetic (zero-test) |
| `poseidon1.py` | Grain LFSR, Cauchy & circulant MDS, the permutation (+`permutation_plus_linear`), compression & sponge modes |
| `verifiers.py` | Independent CICO / collision / density / zero-test verifiers (+ `decode`, `cico_residual`) |
| `poly.py` | `F_p[x]` arithmetic: mul/divmod/gcd, **resultant**, **F_p root-finding**, interpolation |
| `cico_attack.py` | Symbolic permutation over `F_p[Y]`; CICO-1 (root-find) and CICO-2 (resultant) solvers; degree probes |

## Quick start

```python
from poseidon_attack import Poseidon1
from poseidon_attack.verifiers import verify_cico, verify_collision

pos = Poseidon1(r_f=6, r_p=10)              # CICO open instance (Cauchy MDS + Grain RC)
out = pos.permutation_plus_linear([0xC09DE4, 0xEE6282] + [0]*14)

# Algebraic CICO solve (reduced rounds — see note below):
from poseidon_attack.cico_attack import solve_cico1
free, info = solve_cico1(Poseidon1(r_f=6, r_p=0))   # -> verifier-accepted free inputs
```

## Tests / reproduction

```bash
python3 tests/extract_golden.py          # parse Plonky3 golden vectors (AST, no exec)
python3 tests/test_golden.py             # primitive vs published vectors  (18/18)
python3 tests/test_poly.py               # poly primitives vs sympy        (14/14)
python3 tests/test_cico_attack.py        # symbolic perm + CICO solves + degree scaling
python3 tests/test_skip_and_gcd.py       # GCD route + measured resultant degree law
python3 tests/profile_density_decode.py  # density decode uniformity
```

## Scope & honesty note

Pure Python, correctness-first. The **pipeline is exact**, but the full-strength
solves are out of reach for pure Python:

- CICO RP∈{6,8,10}: final-solve degree `3^16..3^20`, ≈`2^53..2^66` ops. We
  validate the *pipeline* on reduced rounds (verifier-accepted) and measure the
  degree growth; the scale solve needs the native core.
- Our plain bivariate resultant measures `deg(R) = 3^(2·R_F+R_P)` — only a
  constant **9×** above the **skip-first-rounds** ideal `D_I=3^(2(R_F-1)+R_P)`
  (eprint 2026/150). So skip is a refinement, not a prerequisite; the dominant
  scale lever is a native NTT-capable core (the **GCD route** is the per-line,
  linear-memory path used at scale).
