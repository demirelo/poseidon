# Poseidon1 / KoalaBear Cryptanalysis

Artifact repository for the paper:

*Concrete-Complexity Cryptanalysis of the 2026 Poseidon1 / KoalaBear Challenge Suite*

Author: Ömer Demirel, Snowfall Finance

Contact: <omer@progfi.xyz> · <https://github.com/demirelo>

This repository contains a verifier-exact Poseidon1/KoalaBear implementation,
reduced-round zero-test witnesses, and reproducibility artifacts. It does not
claim a funded full-round RF=6 bounty solve.

## Contents

| Path | Contents |
|---|---|
| `poseidon_attack/` | Python reference implementation and independent predicate verifiers |
| `core_rs/` | Rust KoalaBear/Poseidon core, polynomial routines, zero-test and resultant tooling |
| `tests/` | Python golden-vector, polynomial, CICO, and density tests |
| `artifacts/` | Durable JSON results used by the paper |
| `paper/` | Manuscript source, PDF, and bibliography |
| `submission/` | Official-verifier bridge and example witness files |
| `tests/fixtures/mds/` | Concrete MDS matrices for custom-MDS scout reproducibility |
| `reference/poseidon-tools/` | Vendored official verifier reference pinned for audit/reproducibility |

## Main Claims

- The challenge admits any MDS matrix satisfying the no-invariant-subspace-trail
  conditions (the Plonky3 circulant is one admissible example); the vendored
  `poseidon-tools` repository ships the Cauchy/Toeplitz MDS plus Grain constants
  as its default test vector. This repository's executable claims target that
  default Cauchy/Toeplitz instance unless stated otherwise.
- CICO-2 resultant elimination reproduces the resultant-attack degree law
  `deg R = 3^(2*RF+RP)` on the default KoalaBear/Poseidon1 verifier instance.
- A root-first zero-test resultant model yields the same CICO-like degree order
  and gives exact reduced-round witnesses through RF=4/RP=5.
- The density predicate has corrected baseline `2^-48` and measured
  decode-uniformity at the tested rounds.
- The live Initiative page checked on 2026-06-17 lists the current zero-test
  record as RF=6/RP=8, so RF=6/RP=9 is the first public-record-improving target.
  The amortized resultant port is exact at reduced rounds, but the end-to-end
  default-MDS RF=6 route is build-dominated and remains out of reach under the
  current implementation model.
- The next research lane is custom-MDS scouting: admissible matrices are checked
  through the vendored `verify_mds_matrix` gate, with initial fixtures under
  `tests/fixtures/mds/`.

## Quick Verification

```sh
python3 tests/test_golden.py
python3 tests/test_poly.py
python3 tests/test_cico_attack.py
python3 tests/test_skip_and_gcd.py
python3 tests/profile_density_decode.py
make smoke
```

The reduced-round RF=4/RP=5 paper witness can also be checked through the pinned
official verifier:

```sh
python3 submission/verify_with_official.py submission/examples/good_zerotest_rf4_rp5.json --official
```

Rust checks:

```sh
cd core_rs
cargo build --release --offline
cargo run --release --bin golden
cargo run --release --bin poly_golden
cargo run --release --bin cico_golden
cargo run --release --bin zerotest_golden
cargo run --release --bin zt_resultant_fast -- validate-lite
cargo run --release --bin custom_mds_scout -- --family cauchy-default --count 1 --rf 2 --rp 1
```

Paper build:

```sh
cd paper
pdflatex poseidon2026-kb.tex
pdflatex poseidon2026-kb.tex
```

## License

MIT. See `LICENSE`.
