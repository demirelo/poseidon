# `core_rs` — native verifier-exact KoalaBear / Poseidon1 core

The fast (Rust) port of the hot path. Same instance facts as the Python
reference (`../poseidon_attack/`), cross-checked against the **same golden
anchors** so it is provably bit-identical to `khovratovich/poseidon-tools @ 60075da7`.

This is the native verifier and algebra core used by the paper. The permutation
and predicates, native polynomial stack, CICO GCD route, KoalaBear NTT
multiplication path, and zero-test root-first calibration/solver are implemented
here.
The B2 amortized-resultant spike also lives here: it ports the ePrint 2026/150
evaluation/interpolation idea to the zero-test rootvar route, validates exactness
at reduced rounds, and records the RF=6 memory/compute projection.

## What's here

| File | Contents |
|---|---|
| `src/lib.rs` | KoalaBear Montgomery field (R=2³²), Grain LFSR, Cauchy & circulant MDS, Poseidon1 permutation (+`perm_mont`/`permutation_plus_linear`), compression-mode hash, `verify_cico` / `cico_residual` / `verify_collision` |
| `src/poly.rs` | normal-domain F_p[x] arithmetic: add/sub/mul/divmod/gcd/eval/deriv, resultant, F_p root-finding (Cantor–Zassenhaus), interpolation. `mul` dispatches to KoalaBear NTT for larger products; `divmod` uses Newton/reversal division for large quotients. Mirrors `poseidon_attack/poly.py`. |
| `src/cico.rs` | symbolic permutation over F_p[X]; `solve_cico2` (resultant route), `solve_cico2_gcd` (**GCD route**, the scale path), `measure_resultant_degree`. Mirrors `poseidon_attack/cico_attack.py`. |
| `src/zerotest.rs` | verifier-exact Fp2 arithmetic, official zero-test predicate, relaxed low-bit verifier, and root-first `P(z)=(z-r)G(z)` scaffold |
| `src/bipoly.rs` | bivariate F_p[X,Y] helper used by the B2 amortized-resultant spike |
| `src/golden_data.rs` | auto-generated Plonky3 width-16 cross-check vector (from `../tests/golden_vectors.json`) |
| `src/bin/golden.rs` | **14/14** primitive golden checks — mirrors `../tests/test_golden.py` |
| `src/bin/poly_golden.rs` | **19/19** poly checks — Sylvester-determinant resultant, planted roots, divmod identity, interp round-trip, explicit NTT-vs-schoolbook products, fast-divmod-vs-schoolbook products, and large-GCD checks (same oracles as `test_poly.py`, no sympy needed) |
| `src/bin/cico_golden.rs` | **6/6** CICO checks — symbolic perm exactness, resultant solve, **GCD-route planted recovery**, degree law `deg(R)=3^(2·RF+RP)`, + per-line build-time scaling table |
| `src/bin/zerotest_golden.rs` | **11/11** zero-test checks — Fp2 identities, verifier gates, root-first scaffold, deterministic relaxed k=4 smoke hit |
| `src/bin/zerotest_calibrate.rs` | zero-test root-first degree calibration: confirms `deg Res_Y = 3^(2RF+RP)`, not generic `3^(2RF+2RP)` |
| `src/bin/zerotest_solve.rs` | exact reduced-round zero-test root-first solver: interpolate eliminant, factor X roots, gcd back-substitute Y, verify `P_hat` |
| `src/bin/zerotest_cofactor_scout.rs` | cofactor-family scout: compares degree/root-density for full, sparse, real/imag, monic, geometric, and fixed cofactors |
| `src/bin/bench.rs` | honest single-thread throughput |
| `src/bin/search.rs` | brute-force CICO **calibration baseline** (relaxed-bit search on the real RP=10 instance) |
| `src/bin/calibrate.rs` | RF=6 GCD-route **calibration**: per-line phase breakdown (NTT perm vs schoolbook gcd), `mul_ntt`-vs-`divmod` microbench, NTT-length ceiling, full-solve projection |
| `src/bin/hgcd_bench.rs` | self-contained recursive Half-GCD benchmark, validated 241/241 against `poly::gcd`; wins at RF=6-sized degree |
| `src/bin/gate_calibrate.rs` | RF=6 go/no-go cost gate: measured per-unit costs, GCD/resultant accounting, amortized-resultant requirement |
| `src/bin/zt_gate_calibrate.rs` | paper-grade RF=6 zero-test interpolated-resultant gate; supersedes the older proxy for zero-test economics |
| `src/bin/zt_amortized_spike.rs` | B2 amortized-resultant port: build bivariate rootvar residuals once, evaluate/interpolate, and match the per-X oracle at reduced rounds |
| `src/bin/zt_resultant_fast.rs` | fast resultant correctness/scaling gate used by B2; validates against `poly::resultant` and times real zero-test residuals |

## Build & run

```sh
cargo build --release --offline          # no dependencies, fully offline
./target/release/golden                  # => 14/14  primitive
./target/release/poly_golden             # => 19/19  poly arithmetic + NTT/fast-div paths
./target/release/cico_golden             # =>  6/6   CICO attack + scaling table
./target/release/zerotest_golden         # => 11/11  zero-test verifier + root-first scaffold
./target/release/zerotest_calibrate heavy # => degree law 3^(2RF+RP), guard-stable
./target/release/zerotest_solve ultra     # => exact solves through RF=4/RP=0
./target/release/zerotest_solve case 2 4 8 # => exact RF=2/RP=4 frontier solve
./target/release/zerotest_cofactor_scout  # => cofactor-family/root-density scout
./target/release/zerotest_cofactor_scout 2 4 2 full,real,monic,top,edges
./target/release/zt_resultant_fast validate-lite
./target/release/zt_amortized_spike amortized-rootvar 2 1 0
./target/release/zt_amortized_spike amortized-rootvar 2 2 0
./target/release/bench
./target/release/search 15000000
```

## Verified (golden, 14/14)

1. **Plonky3 width-16 vector** — round schedule (full/partial/S-box/MDS) end-to-end.
2. **Default Cauchy+Grain `perm([0;16])[0] == 1393439926`** — Grain LFSR +
   Cauchy MDS + Montgomery arithmetic all correct simultaneously.
3. **Published t=3 partial collision** — compression mode + collision predicate
   on the full-strength rf=8/rp=20 instance (accepts t≤3, rejects t=4, rejects x==y).
4. **Field identities** — Montgomery round-trips, `mont_mul`/`mont_cube` vs normal domain.
5. **CICO wiring** for rp ∈ {6,8,10}.
6. **Zero-test wiring** — Fp2 arithmetic, default verifier gates, root-first
   `P(z)=(z-r)G(z)` construction, and relaxed k=4 oracle smoke test.
7. **Zero-test root-first solving** — measured degree law and exact reduced
   solves through RF=2/RP=4 and RF=4/RP=0.
8. **Zero-test cofactor scouting** — simple sparse/real/imag/monic families keep
   the same degree; sparse families remain viable; all-ones looked poor. At
   RF=2/RP=4, `edges` matched real-family root yield in a two-seed targeted pass.

## Measured throughput (single thread, this machine)

| Op | rate | ns/op |
|---|---|---|
| `perm` rf=6/rp=10 (CICO) | ~5.0×10⁵ /s | ~1980 |
| `cico_residual` rf=6/rp=10 | ~5.0×10⁵ /s | ~1980 |
| `perm` rf=8/rp=20 (collision) | ~2.3×10⁵ /s | ~4400 |

~500× the pure-Python reference (~10³ perm/s). **Not yet** a
10⁷–10⁸/s target: the MDS is a naive t² Montgomery inner product (256 muls +
16 reductions/round), and that reduction is the bottleneck.

## GCD route: measured scaling with NTT-backed products

`cico_golden` times one GCD-route line (one symbolic `permutation_plus_linear`
over F_p[X] + a gcd) on the real RF=6 instance. Polynomial multiplication is
NTT-backed for larger products:

| RF=6 | forward deg | build + gcd | factor |
|---|---|---|---|
| RP=0 | 729 (3⁶) | 2.4 ms | — |
| RP=1 | 2187 (3⁷) | 11.5 ms | ×4.8 |
| RP=2 | 6561 (3⁸) | 61 ms | ×5.3 |
| RP=3 | 19683 (3⁹) | 393 ms | ×6.4 |

NTT removed the obvious multiplication wall (RP=3 ~0.4 s vs the old schoolbook
~2.2 s).

### RF=6 calibration (`calibrate`) — the wall is now division/GCD

Per-line phase breakdown on the real RF=6 instance (NTT-backed perm; schoolbook gcd):

| RP | fwd deg | perm build (NTT) | gcd (Euclid) | mul_ntt(1) | divmod(1, fast) |
|---|---|---|---|---|---|
| 4 | 59 049 (3⁸) | ~0.28 s | ~2.7 s | ~0.012 s | ~0.07 s |
| 5 | 177 147 (3⁹) | ~0.9 s | ~25 s | ~0.053 s | ~0.3-0.7 s |
| 6 | 531 441 (3¹²) | a few seconds | ~220-230 s (projected from RP5) | ~0.25 s | — |

**Verdict:** NTT made the symbolic permutation cheap, and Newton/reversal
division made single large-quotient `divmod` cheap. The Euclidean GCD step count
was the next per-node wall. `hgcd_bench` now validates a standalone recursive
Half-GCD against the `poly::gcd` oracle (241/241 cases) and shows the crossover:
it loses below about 65k degree, but wins about **7x** at degree 531441, the
RF=6/RP=6 per-node shape. Keep it self-contained until tuning lowers the
crossover; do not blindly promote it into `poly.rs`.

**Full-solve accounting (corrected).** The GCD route needs **~p lines**, not
`p/deg`: a 1-D line meets the codim-2 CICO-2 solution set with probability ~1/p.
So RF=6/RP=6 ≈ `p × ~225 s` ≈ **~15,000 core-years** schoolbook; even with an instant gcd the perm-build floor is
**hundreds of core-years**. **Cluster-scale, not single-machine**
(embarrassingly parallel). For low RP the **resultant route** (deg `D_I=3^(12+RP)`)
is cheaper — RP=6 ~2⁵³ ops, ~0.05 PiB — and the GCD route wins only at high RP
where resultant memory blows up (RP=10 ~355 PiB). An earlier estimate using
`p/deg` line count was too optimistic; the corrected count is the one above.

NTT-length ceiling: RP≤9 fits the field's 2²⁴ max power-of-two root-of-unity
length; **RP=10 (forward degree 3¹⁶≈4.3×10⁷) needs len 2²⁶ > 2²⁴** → Kronecker
substitution or multi-prime CRT NTT.

**RF=6 cost gate:** `zt_gate_calibrate` is the paper-grade gate for the real
zero-test interpolated-resultant route; it supersedes the older `gate_calibrate`
proxy. The GCD line route has about `1/p` hit rate, so it remains cluster-scale
despite Half-GCD. The Initiative page checked on 2026-06-17 lists the current
zero-test record as RF=6/RP=8, so RF=6/RP=9 is the first public-record-improving
target. The current-code resultant route is already roughly **1,460 core-years**
at RP=6, **37,360 core-years** at RP=7, **~9.6×10⁵ core-years** at RP=8,
and **~2.4×10⁷ core-years** at RP=9; even the Half-GCD floor remains far out of
reach. B2 ports the ePrint 2026/150 amortized resultant to the zero-test
rootvar route and validates it at reduced rounds, but priced end-to-end it is
build-dominated: on the order of **10³ core-years** at RP=6, **4×10⁴** at RP=7,
**2×10⁶** at RP=8, and **10⁸** at RP=9, with multi-TiB to hundreds-of-TiB build
memory (≈1.0/9.2/83/745 TiB). So
the next plausible research lane is not another default-Cauchy cluster run; it is
`custom_mds_scout` plus a short custom-MDS vulnerability search gated by the
vendored `verify_mds_matrix`. RP=10 additionally needs the Kronecker/CRT NTT
above.

**Custom-MDS scout:** `custom_mds_scout` makes the MDS matrix a first-class
experiment input. It supports `cauchy-default`, `cauchy-variant`, `circulant`,
`plonky3-circulant`, and `sparse-plus-dense`, calls the vendored
`poseidon.mds_matrix.verify_mds_matrix` admissibility gate, and emits JSON-only
diagnostics. Cheap smoke:

```sh
cargo run --release --bin custom_mds_scout -- --family cauchy-default --count 1 --rf 2 --rp 1
```
