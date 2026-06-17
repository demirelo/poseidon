//! Phase-A baseline benchmark harness for the amortized-resultant feasibility spike.
//!
//! For the zero-test root-first attack, for each X on an arithmetic grid we build
//! two univariate residual polynomials f0(Y), f1(Y) over Fp via a symbolic
//! Poseidon permutation, then compute Res_Y(f0, f1) in Fp. The expensive per-X
//! work is split as:
//!   (a) "build"     = constructing f0, f1 (residual_polys_at_x);
//!   (b) "resultant" = pl::resultant(f0, f1).
//!
//! There are TWO distinct residual constructions ("parameterizations"); this
//! harness exposes each as its own explicit subcommand so the baseline numbers
//! can never silently mix flavors:
//!
//!   * `baseline-rootvar  <rf> <rp> <nodes> <seed_idx>` — the LANDED RF=4/RP=5
//!         route. Fixes a root r and makes ONE coordinate of the cofactor the
//!         variable: G[var_idx] = X + Y*s (var_idx=0). This is byte-for-byte the
//!         same residual construction as `zerotest_rootvar_scout`'s var route, so
//!         the per-X resultant values match the landed shards. deg_y = 3^(2rf+rp).
//!         The seed is derived as scout_seed(rf, rp, seed_idx).
//!
//!   * `baseline-rootgrid <rf> <rp> <nodes> <seed>` — the OLDER flavor: a FIXED
//!         full cofactor g and a root that varies along the grid as r = X + Y*s
//!         (root_mul_g_at_x). This is the construction the single `baseline` mode
//!         used previously; it is preserved verbatim. The last arg is the raw
//!         cofactor seed (default 0), NOT a scout seed_idx.
//!
//! Both modes emit one JSON line with a "mode" field, deg_y reporting, and the
//! build/resultant timing split.
//!
//! Run:
//!   zt_amortized_spike baseline-rootvar  4 5 256 0
//!   zt_amortized_spike baseline-rootgrid 4 4 256 0
//!   zt_amortized_spike baseline-rootgrid 4 5 256 0

// Helpers are copied verbatim from zerotest_cofactor_scout so the baseline and
// future amortized modes share an identical numerical kit; some are unused by
// the baseline path today.
#![allow(dead_code)]

// Bivariate F_p[X][Y] arithmetic helper for the amortized eval-interp resultant
// (eprint 2026/150, Section 3.2.1). Included via #[path] so we touch only this
// bin + the one new helper file (no lib.rs changes).
#[path = "../bipoly.rs"]
mod bipoly;

use bipoly::BiPoly;
use poseidon_core::poly as pl;
use poseidon_core::zerotest::{self, Fp2, ZT_D};
use poseidon_core::{Poseidon1, P};
use std::time::Instant;

const PU64: u64 = P as u64;

#[inline]
fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}

#[inline]
fn fsub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + P - b
    }
}

#[inline]
fn fmul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % PU64) as u32
}

fn fpow(mut a: u64, mut e: u64) -> u32 {
    a %= PU64;
    let mut r: u64 = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = r * a % PU64;
        }
        a = a * a % PU64;
        e >>= 1;
    }
    r as u32
}

fn finv(a: u32) -> u32 {
    fpow(a as u64, (P - 2) as u64)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn field(&mut self) -> u32 {
        (self.next() % PU64) as u32
    }

    fn field_nz(&mut self) -> u32 {
        1 + (self.next() % (PU64 - 1)) as u32
    }

    fn fp2(&mut self) -> Fp2 {
        Fp2::new(self.field(), self.field())
    }

    fn fp2_nonzero(&mut self) -> Fp2 {
        loop {
            let z = self.fp2();
            if !z.is_zero() {
                return z;
            }
        }
    }

    fn fp2_real_nonzero(&mut self) -> Fp2 {
        Fp2::new(self.field_nz(), 0)
    }

    fn fp2_imag_nonzero(&mut self) -> Fp2 {
        Fp2::new(0, self.field_nz())
    }
}

fn sbox_poly(f: &[u32]) -> pl::Poly {
    pl::mul(&pl::mul(f, f), f)
}

fn mds_poly(state: &[pl::Poly], mds_n: &[Vec<u32>]) -> Vec<pl::Poly> {
    let t = state.len();
    let mut out = Vec::with_capacity(t);
    for row in mds_n.iter().take(t) {
        let mut acc: pl::Poly = vec![];
        for j in 0..t {
            if !state[j].is_empty() {
                acc = pl::add(&acc, &pl::scalar(&state[j], row[j]));
            }
        }
        out.push(acc);
    }
    out
}

fn permutation_poly(pos: &Poseidon1, state: &[pl::Poly]) -> Vec<pl::Poly> {
    let t = pos.t;
    let mut s = state.to_vec();
    let half = pos.r_f / 2;
    let mut idx = 0usize;

    for _ in 0..half {
        for i in 0..t {
            s[i] = pl::add(&s[i], &[pos.rc_n[idx][i]]);
        }
        for si in s.iter_mut().take(t) {
            *si = sbox_poly(si);
        }
        s = mds_poly(&s, &pos.mds_n);
        idx += 1;
    }
    for _ in 0..pos.r_p {
        for i in 0..t {
            s[i] = pl::add(&s[i], &[pos.rc_n[idx][i]]);
        }
        s[0] = sbox_poly(&s[0]);
        s = mds_poly(&s, &pos.mds_n);
        idx += 1;
    }
    for _ in 0..half {
        for i in 0..t {
            s[i] = pl::add(&s[i], &[pos.rc_n[idx][i]]);
        }
        for si in s.iter_mut().take(t) {
            *si = sbox_poly(si);
        }
        s = mds_poly(&s, &pos.mds_n);
        idx += 1;
    }
    s
}

fn compression_hash_poly(pos: &Poseidon1, input: &[pl::Poly]) -> Vec<pl::Poly> {
    let mut out = permutation_poly(pos, input);
    for i in 0..pos.t {
        out[i] = pl::add(&out[i], &input[i]);
    }
    out
}

fn root_mul_g_at_x(x: u32, g: Fp2) -> (pl::Poly, pl::Poly) {
    (
        pl::trim(&[fmul(x, g.c0), fmul(3, g.c1)]),
        pl::trim(&[fmul(x, g.c1), g.c0]),
    )
}

fn neg_poly(f: &[u32]) -> pl::Poly {
    pl::scalar(f, P - 1)
}

fn root_first_state_at_x(x: u32, cofactor: &[Fp2; ZT_D]) -> Vec<pl::Poly> {
    let mut coeffs: Vec<(pl::Poly, pl::Poly)> = Vec::with_capacity(ZT_D + 1);

    let (rg0_0, rg0_1) = root_mul_g_at_x(x, cofactor[0]);
    coeffs.push((neg_poly(&rg0_0), neg_poly(&rg0_1)));

    for j in 1..ZT_D {
        let (rg0, rg1) = root_mul_g_at_x(x, cofactor[j]);
        coeffs.push((
            pl::sub(&[cofactor[j - 1].c0], &rg0),
            pl::sub(&[cofactor[j - 1].c1], &rg1),
        ));
    }

    coeffs.push((vec![cofactor[ZT_D - 1].c0], vec![cofactor[ZT_D - 1].c1]));

    let mut flat = Vec::with_capacity(16);
    for (c0, c1) in coeffs {
        flat.push(c0);
        flat.push(c1);
    }
    flat
}

fn residual_polys_at_x(pos: &Poseidon1, cofactor: &[Fp2; ZT_D], x: u32) -> (pl::Poly, pl::Poly) {
    let input = root_first_state_at_x(x, cofactor);
    let out = compression_hash_poly(pos, &input);
    let f0 = pl::sub(&out[0], &[x]);
    let f1 = pl::sub(&out[1], &[0, 1]);
    (pl::trim(&f0), pl::trim(&f1))
}

/// Build one full-random cofactor [Fp2; ZT_D] from `seed` (mimics Family::Full:
/// for each of ZT_D slots, draw a nonzero Fp2 from the seed).
fn full_cofactor(seed: u64) -> [Fp2; ZT_D] {
    let mut rng = Rng::new(seed);
    let mut g = [Fp2::ZERO; ZT_D];
    for slot in &mut g {
        *slot = rng.fp2_nonzero();
    }
    g
}

// ===========================================================================
// Flavor (B): fixed-root, coordinate-variable cofactor G[var_idx] = X + Y*s.
//
// This is an EXACT replica of zerotest_rootvar_scout's var route (seed/root/base
// derivation + residual_polys_at_x), so baseline-rootvar per-X resultants match
// the landed RF=4/RP=5 var0 shards. Do not "simplify" this — value parity with
// the shards is the whole point.
// ===========================================================================
type Fp2Poly = (pl::Poly, pl::Poly);

fn fp2_poly_const(z: Fp2) -> Fp2Poly {
    (pl::trim(&[z.c0]), pl::trim(&[z.c1]))
}

fn fp2_poly_sub(a: &Fp2Poly, b: &Fp2Poly) -> Fp2Poly {
    (pl::sub(&a.0, &b.0), pl::sub(&a.1, &b.1))
}

fn fp2_poly_neg(a: &Fp2Poly) -> Fp2Poly {
    (pl::scalar(&a.0, P - 1), pl::scalar(&a.1, P - 1))
}

fn root_mul_poly(root: Fp2, g: &Fp2Poly) -> Fp2Poly {
    let c0 = pl::add(
        &pl::scalar(&g.0, root.c0),
        &pl::scalar(&g.1, fmul(3, root.c1)),
    );
    let c1 = pl::add(&pl::scalar(&g.1, root.c0), &pl::scalar(&g.0, root.c1));
    (c0, c1)
}

/// scout_seed: the EXACT seed schedule the landed shards used.
fn scout_seed(rf: usize, rp: usize, seed_idx: usize) -> u64 {
    0xA17E_0000 + ((rf as u64) << 24) + ((rp as u64) << 16) + seed_idx as u64
}

fn fixed_root(seed: u64) -> Fp2 {
    Rng::new(seed ^ 0xA11C_E000).fp2_nonzero()
}

fn fixed_cofactor(seed: u64) -> [Fp2; ZT_D] {
    let mut rng = Rng::new(seed);
    let mut g = [Fp2::ZERO; ZT_D];
    for slot in &mut g {
        *slot = rng.fp2_nonzero();
    }
    g
}

fn cofactor_polys_at_x(base: &[Fp2; ZT_D], var_idx: usize, x: u32) -> Vec<Fp2Poly> {
    let mut g = Vec::with_capacity(ZT_D);
    for (j, &z) in base.iter().enumerate() {
        if j == var_idx {
            g.push((vec![x], vec![0, 1]));
        } else {
            g.push(fp2_poly_const(z));
        }
    }
    g
}

fn root_first_state_from_cofactor_polys(root: Fp2, g: &[Fp2Poly]) -> Vec<pl::Poly> {
    assert_eq!(g.len(), ZT_D);
    let mut coeffs = Vec::with_capacity(ZT_D + 1);
    coeffs.push(fp2_poly_neg(&root_mul_poly(root, &g[0])));
    for j in 1..ZT_D {
        coeffs.push(fp2_poly_sub(&g[j - 1], &root_mul_poly(root, &g[j])));
    }
    coeffs.push(g[ZT_D - 1].clone());

    let mut flat = Vec::with_capacity(16);
    for (c0, c1) in coeffs {
        flat.push(c0);
        flat.push(c1);
    }
    flat
}

fn residual_polys_at_x_rootvar(
    pos: &Poseidon1,
    root: Fp2,
    base: &[Fp2; ZT_D],
    var_idx: usize,
    x: u32,
) -> (pl::Poly, pl::Poly) {
    let input = root_first_state_from_cofactor_polys(root, &cofactor_polys_at_x(base, var_idx, x));
    let out = compression_hash_poly(pos, &input);
    let f0 = pl::sub(&out[0], &[root.c0]);
    let f1 = pl::sub(&out[1], &[root.c1]);
    (pl::trim(&f0), pl::trim(&f1))
}

/// Flavor (A) = rootgrid: FIXED full cofactor `full_cofactor(seed)`, root varies
/// along the grid as r = X + Y*s. This is the previous single `baseline` body,
/// preserved verbatim apart from the explicit "baseline_rootgrid" mode label.
fn run_baseline_rootgrid(rf: usize, rp: usize, nodes: usize, seed: u64) {
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let cofactor = full_cofactor(seed);

    let mut build_s: f64 = 0.0;
    let mut resultant_s: f64 = 0.0;
    let mut deg_y: (i64, i64) = (-1, -1);
    let mut resultant_first: u32 = 0;

    for x in 1..=nodes as u32 {
        let build_start = Instant::now();
        let (f0, f1) = residual_polys_at_x(&pos, &cofactor, x);
        build_s += build_start.elapsed().as_secs_f64();

        if x == 1 {
            deg_y = (pl::deg(&f0), pl::deg(&f1));
        }

        let resultant_start = Instant::now();
        let r = pl::resultant(&f0, &f1);
        resultant_s += resultant_start.elapsed().as_secs_f64();

        if x == 1 {
            resultant_first = r;
        }
    }

    let total_s = build_s + resultant_s;
    let ms_per_node = if nodes == 0 {
        0.0
    } else {
        1000.0 * total_s / nodes as f64
    };

    println!(
        "{{\"rf\":{},\"rp\":{},\"mode\":\"baseline_rootgrid\",\"nodes\":{},\"seed\":{},\"build_s\":{:.4},\"resultant_s\":{:.4},\"total_s\":{:.4},\"ms_per_node\":{:.4},\"deg_y\":[{},{}],\"resultant_first\":{}}}",
        rf,
        rp,
        nodes,
        seed,
        build_s,
        resultant_s,
        total_s,
        ms_per_node,
        deg_y.0,
        deg_y.1,
        resultant_first
    );
}

/// Flavor (B) = rootvar: FIXED root, coordinate-variable cofactor G[var_idx] =
/// X + Y*s. The landed RF=4/RP=5 route. Seed is scout_seed(rf, rp, seed_idx);
/// var_idx is fixed to 0 to match the landed var0 shards.
fn run_baseline_rootvar(rf: usize, rp: usize, nodes: usize, seed_idx: usize) {
    let var_idx = 0usize;
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);

    let mut build_s: f64 = 0.0;
    let mut resultant_s: f64 = 0.0;
    let mut deg_y: (i64, i64) = (-1, -1);
    let mut resultant_first: u32 = 0;

    for x in 1..=nodes as u32 {
        let build_start = Instant::now();
        let (f0, f1) = residual_polys_at_x_rootvar(&pos, root, &base, var_idx, x);
        build_s += build_start.elapsed().as_secs_f64();

        if x == 1 {
            deg_y = (pl::deg(&f0), pl::deg(&f1));
        }

        let resultant_start = Instant::now();
        let r = pl::resultant(&f0, &f1);
        resultant_s += resultant_start.elapsed().as_secs_f64();

        if x == 1 {
            resultant_first = r;
        }
    }

    let total_s = build_s + resultant_s;
    let ms_per_node = if nodes == 0 {
        0.0
    } else {
        1000.0 * total_s / nodes as f64
    };

    println!(
        "{{\"rf\":{},\"rp\":{},\"mode\":\"baseline_rootvar\",\"nodes\":{},\"seed_idx\":{},\"seed\":{},\"var_idx\":{},\"root_c0\":{},\"root_c1\":{},\"build_s\":{:.4},\"resultant_s\":{:.4},\"total_s\":{:.4},\"ms_per_node\":{:.4},\"deg_y\":[{},{}],\"resultant_first\":{}}}",
        rf,
        rp,
        nodes,
        seed_idx,
        seed,
        var_idx,
        root.c0,
        root.c1,
        build_s,
        resultant_s,
        total_s,
        ms_per_node,
        deg_y.0,
        deg_y.1,
        resultant_first
    );
}

// ===========================================================================
// B2: amortized evaluation-interpolation resultant (eprint 2026/150 sec 3.2.1).
//
// The per-X oracle (run_baseline_rootvar) rebuilds the symbolic Poseidon
// permutation in F_p[Y] for EACH X. Here we instead build the bivariate system
// f0(X,Y), f1(X,Y) in F_p[X][Y] exactly ONCE (root r fixed, cofactor coordinate
// G[0] = X + Y*sqrt(3) symbolic in BOTH X and Y), multipoint-evaluate each
// Y-coefficient f_i(X) across X, take Res_Y at each X via the validated
// pl::resultant, and interpolate R(X). This must equal the oracle eliminant.
//
// CRITICAL STRUCTURE FACT: the Poseidon residual polynomials are built by
// running the permutation on the FLAT 16-word state as plain F_p elements (the
// verifier's compression_mode_hash treats the 16 words as F_p, not Fp2). The
// Fp2 = F_p[s]/(s^2 - 3) structure (beta = 3) only appears when forming the
// initial root-first state from G[0] = X + Y*s via root_mul_poly, which is an
// F_p-LINEAR combination of the cofactor polys (root is a fixed numeric Fp2).
// So the permutation needs only F_p[X][Y] cube/linear ops; no Fp2 multiply ever
// happens inside it. This mirrors residual_polys_at_x_rootvar exactly.
// ===========================================================================

type Fp2Bi = (BiPoly, BiPoly);

fn fp2_bi_const(z: Fp2) -> Fp2Bi {
    (bipoly::constant(z.c0), bipoly::constant(z.c1))
}

fn fp2_bi_sub(a: &Fp2Bi, b: &Fp2Bi) -> Fp2Bi {
    (bipoly::sub(&a.0, &b.0), bipoly::sub(&a.1, &b.1))
}

fn fp2_bi_neg(a: &Fp2Bi) -> Fp2Bi {
    (bipoly::scalar(&a.0, P - 1), bipoly::scalar(&a.1, P - 1))
}

/// Mirror of root_mul_poly: Fp2 product root * g where g = (g.0, g.1) is an
/// Fp2-of-BiPoly and root is a fixed numeric Fp2. Uses beta = 3:
///   c0 = g.0*root.c0 + g.1*(3*root.c1)
///   c1 = g.1*root.c0 + g.0*root.c1
/// All operations are F_p-linear in the BiPolys (scalar + add), no BiPoly mul.
fn root_mul_bi(root: Fp2, g: &Fp2Bi) -> Fp2Bi {
    let c0 = bipoly::add(
        &bipoly::scalar(&g.0, root.c0),
        &bipoly::scalar(&g.1, fmul(3, root.c1)),
    );
    let c1 = bipoly::add(
        &bipoly::scalar(&g.1, root.c0),
        &bipoly::scalar(&g.0, root.c1),
    );
    (c0, c1)
}

/// Bivariate cofactor polys for the rootvar route: G[var_idx] = X + Y*s, all
/// other coordinates fixed numeric Fp2 constants. Mirror of cofactor_polys_at_x,
/// but with X symbolic. For var_idx the c0 slot carries X and the c1 slot carries
/// Y (matching the oracle's `(vec![x], vec![0, 1])` at fixed x, where c0 = x and
/// c1 = Y).
fn cofactor_bi(base: &[Fp2; ZT_D], var_idx: usize) -> Vec<Fp2Bi> {
    let mut g = Vec::with_capacity(ZT_D);
    for (j, &z) in base.iter().enumerate() {
        if j == var_idx {
            g.push((bipoly::var_x(), bipoly::var_y()));
        } else {
            g.push(fp2_bi_const(z));
        }
    }
    g
}

/// Mirror of root_first_state_from_cofactor_polys over BiPoly. Produces the flat
/// 16-entry state [c0_0, c1_0, c0_1, c1_1, ...] for the root-first reduction
/// P(z) = (z - root) * G(z).
fn root_first_state_bi(root: Fp2, g: &[Fp2Bi]) -> Vec<BiPoly> {
    assert_eq!(g.len(), ZT_D);
    let mut coeffs: Vec<Fp2Bi> = Vec::with_capacity(ZT_D + 1);
    coeffs.push(fp2_bi_neg(&root_mul_bi(root, &g[0])));
    for j in 1..ZT_D {
        coeffs.push(fp2_bi_sub(&g[j - 1], &root_mul_bi(root, &g[j])));
    }
    coeffs.push(g[ZT_D - 1].clone());

    let mut flat = Vec::with_capacity(16);
    for (c0, c1) in coeffs {
        flat.push(c0);
        flat.push(c1);
    }
    flat
}

fn mds_bi(state: &[BiPoly], mds_n: &[Vec<u32>]) -> Vec<BiPoly> {
    let t = state.len();
    let mut out = Vec::with_capacity(t);
    for row in mds_n.iter().take(t) {
        let mut acc: BiPoly = vec![];
        for j in 0..t {
            if !state[j].is_empty() {
                acc = bipoly::add(&acc, &bipoly::scalar(&state[j], row[j]));
            }
        }
        out.push(acc);
    }
    out
}

/// Bivariate symbolic permutation. Byte-for-byte the same control flow as
/// `permutation_poly` (lib's perm structure), with sbox = bivariate cube and
/// MDS/round-constant add = F_p[X][Y] linear ops.
fn permutation_bi(pos: &Poseidon1, state: &[BiPoly]) -> Vec<BiPoly> {
    let t = pos.t;
    let mut s = state.to_vec();
    let half = pos.r_f / 2;
    let mut idx = 0usize;

    for _ in 0..half {
        for i in 0..t {
            s[i] = bipoly::add(&s[i], &bipoly::constant(pos.rc_n[idx][i]));
        }
        for si in s.iter_mut().take(t) {
            *si = bipoly::cube(si);
        }
        s = mds_bi(&s, &pos.mds_n);
        idx += 1;
    }
    for _ in 0..pos.r_p {
        for i in 0..t {
            s[i] = bipoly::add(&s[i], &bipoly::constant(pos.rc_n[idx][i]));
        }
        s[0] = bipoly::cube(&s[0]);
        s = mds_bi(&s, &pos.mds_n);
        idx += 1;
    }
    for _ in 0..half {
        for i in 0..t {
            s[i] = bipoly::add(&s[i], &bipoly::constant(pos.rc_n[idx][i]));
        }
        for si in s.iter_mut().take(t) {
            *si = bipoly::cube(si);
        }
        s = mds_bi(&s, &pos.mds_n);
        idx += 1;
    }
    s
}

fn compression_hash_bi(pos: &Poseidon1, input: &[BiPoly]) -> Vec<BiPoly> {
    let mut out = permutation_bi(pos, input);
    for i in 0..pos.t {
        out[i] = bipoly::add(&out[i], &input[i]);
    }
    out
}

/// Build the bivariate residuals f0(X,Y), f1(X,Y) ONCE. Mirror of
/// residual_polys_at_x_rootvar but with X kept symbolic.
fn residual_bi(pos: &Poseidon1, root: Fp2, base: &[Fp2; ZT_D], var_idx: usize) -> (BiPoly, BiPoly) {
    let input = root_first_state_bi(root, &cofactor_bi(base, var_idx));
    let out = compression_hash_bi(pos, &input);
    let f0 = bipoly::sub(&out[0], &bipoly::constant(root.c0));
    let f1 = bipoly::sub(&out[1], &bipoly::constant(root.c1));
    (bipoly::trim(&f0), bipoly::trim(&f1))
}

fn expected_resultant_degree(rf: usize, rp: usize) -> usize {
    3usize.pow((2 * rf + rp) as u32)
}

/// Amortized eval-interp resultant for the rootvar route. Builds the bivariate
/// system once, evaluates each Y-coefficient at D+1 distinct X points, computes
/// Res_Y via the validated pl::resultant per point, interpolates R(X), and gates
/// it EXACTLY against the per-X oracle (which recomputes Res_Y per X with no
/// amortization). Secondary: try to recover an Fp root -> Y -> verifier witness.
fn run_amortized_rootvar(rf: usize, rp: usize, seed_idx: usize) {
    let var_idx = 0usize;
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);

    let deg_r = expected_resultant_degree(rf, rp); // D = 3^(2rf+rp)
    let n_pts = deg_r + 1; // D + 1 evaluation points

    // (1) Build the bivariate system ONCE.
    let build_start = Instant::now();
    let (f0_bi, f1_bi) = residual_bi(&pos, root, &base, var_idx);
    let build_s = build_start.elapsed().as_secs_f64();

    let dy0 = bipoly::deg_y(&f0_bi);
    let dy1 = bipoly::deg_y(&f1_bi);
    let dx0 = bipoly::deg_x(&f0_bi);
    let dx1 = bipoly::deg_x(&f1_bi);

    // (2)+(3)+(4) Evaluate each f_i(X) at the X grid {1, 2, ..., D+1} (the exact
    // grid the oracle/scout use) and take Res_Y at each point with the validated
    // resultant. Straightforward evaluation at D+1 points (correctness first).
    let xs: Vec<u32> = (1..=n_pts as u32).collect();
    let eval_start = Instant::now();
    let ys: Vec<u32> = xs
        .iter()
        .map(|&x| {
            let f0 = bipoly::eval_x(&f0_bi, x);
            let f1 = bipoly::eval_x(&f1_bi, x);
            pl::resultant(&f0, &f1)
        })
        .collect();
    let eval_s = eval_start.elapsed().as_secs_f64();

    // (5) Interpolate R(X) of degree D from the D+1 values.
    let interp_start = Instant::now();
    let r_poly = pl::interpolate(&xs, &ys);
    let interp_s = interp_start.elapsed().as_secs_f64();
    let deg_r_actual = pl::deg(&r_poly);

    // PRIMARY GATE: compare against the per-X oracle EXACTLY. The oracle rebuilds
    // the symbolic permutation in F_p[Y] per X and takes Res_Y(f0,f1); we compare
    // our interpolated R(X) against that per-X value at every oracle X point
    // (this is degree+coefficient identity: two degree-D polys that agree on D+1
    // points are identical, and we check all D+1).
    let matches_oracle = xs.iter().zip(ys.iter()).all(|(&x, &amort_val)| {
        // Oracle path: independent per-X rebuild (residual_polys_at_x_rootvar).
        let (f0, f1) = residual_polys_at_x_rootvar(&pos, root, &base, var_idx, x);
        let oracle_val = pl::resultant(&f0, &f1);
        let interp_val = pl::eval(&r_poly, x);
        oracle_val == amort_val && interp_val == amort_val
    });

    // SECONDARY (bonus): find Fp roots of R(X), back-substitute Y via gcd(f0,f1)
    // at the root, form p_hat, and report any verifier-accepted witness.
    let mut witness_found = false;
    let mut witness: Option<(u32, u32)> = None;
    let x_roots = pl::roots(&r_poly);
    'outer: for x in x_roots.iter().copied() {
        let f0 = bipoly::eval_x(&f0_bi, x);
        let f1 = bipoly::eval_x(&f1_bi, x);
        let g = pl::gcd(&f0, &f1);
        for y in pl::roots(&g) {
            let cofactor = concrete_cofactor_rootvar(&base, var_idx, x, y);
            let coeffs = zerotest::root_first_coeffs(root, &cofactor);
            let p_hat = zerotest::flatten_coeffs(&coeffs);
            if zerotest::verify_zerotest(&pos, &p_hat) {
                witness_found = true;
                witness = Some((x, y));
                break 'outer;
            }
        }
    }

    let total_s = build_s + eval_s + interp_s;

    let witness_json = match witness {
        Some((x, y)) => format!(",\"witness_x\":{},\"witness_y\":{}", x, y),
        None => String::new(),
    };

    println!(
        "{{\"rf\":{},\"rp\":{},\"mode\":\"amortized_rootvar\",\"seed_idx\":{},\"seed\":{},\"var_idx\":{},\"root_c0\":{},\"root_c1\":{},\"deg_R\":{},\"deg_R_expected\":{},\"deg_y\":[{},{}],\"deg_x\":[{},{}],\"n_points\":{},\"x_roots\":{},\"matches_oracle\":{},\"witness_found\":{}{},\"build_s\":{:.4},\"eval_s\":{:.4},\"interp_s\":{:.4},\"total_s\":{:.4}}}",
        rf,
        rp,
        seed_idx,
        seed,
        var_idx,
        root.c0,
        root.c1,
        deg_r_actual,
        deg_r,
        dy0,
        dy1,
        dx0,
        dx1,
        n_pts,
        x_roots.len(),
        matches_oracle,
        witness_found,
        witness_json,
        build_s,
        eval_s,
        interp_s,
        total_s
    );
}

/// Mirror of the scout's concrete_cofactor for the rootvar route: substitute the
/// recovered (x, y) into coordinate var_idx of the fixed base cofactor.
fn concrete_cofactor_rootvar(base: &[Fp2; ZT_D], var_idx: usize, x: u32, y: u32) -> [Fp2; ZT_D] {
    let mut g = *base;
    g[var_idx] = Fp2::new(x, y);
    g
}

fn usage() {
    eprintln!("usage:");
    eprintln!("  zt_amortized_spike baseline-rootvar  <RF> <RP> <NODES> <SEED_IDX>");
    eprintln!("  zt_amortized_spike baseline-rootgrid <RF> <RP> <NODES> [SEED]");
    eprintln!("  zt_amortized_spike amortized-rootvar <RF> <RP> <SEED_IDX>");
    eprintln!("    rootvar  = fixed root, coordinate-variable cofactor G[0]=X+Y*s (landed RF=4/RP=5 route)");
    eprintln!("    rootgrid = fixed full cofactor, root=X+Y*s grid (older flavor)");
    eprintln!("    amortized-rootvar = build bivariate system once, eval/interp Res_Y across X (2026/150 3.2.1)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("");

    match mode {
        "baseline-rootvar" => {
            let rf = args.get(2).map_or(4, |s| s.parse().expect("RF must parse"));
            let rp = args.get(3).map_or(5, |s| s.parse().expect("RP must parse"));
            let nodes = args
                .get(4)
                .map_or(256, |s| s.parse().expect("NODES must parse"));
            let seed_idx = args
                .get(5)
                .map_or(0, |s| s.parse().expect("SEED_IDX must parse"));
            run_baseline_rootvar(rf, rp, nodes, seed_idx);
        }
        "amortized-rootvar" => {
            let rf = args.get(2).map_or(2, |s| s.parse().expect("RF must parse"));
            let rp = args.get(3).map_or(1, |s| s.parse().expect("RP must parse"));
            let seed_idx = args
                .get(4)
                .map_or(0, |s| s.parse().expect("SEED_IDX must parse"));
            run_amortized_rootvar(rf, rp, seed_idx);
        }
        // `baseline` is kept as a back-compat alias for the older rootgrid flavor.
        "baseline-rootgrid" | "baseline" => {
            let rf = args.get(2).map_or(4, |s| s.parse().expect("RF must parse"));
            let rp = args.get(3).map_or(5, |s| s.parse().expect("RP must parse"));
            let nodes = args
                .get(4)
                .map_or(256, |s| s.parse().expect("NODES must parse"));
            let seed = args.get(5).map_or(0, |s| s.parse().expect("SEED must parse"));
            run_baseline_rootgrid(rf, rp, nodes, seed);
        }
        other => {
            eprintln!("unknown mode: {other:?}");
            usage();
            std::process::exit(2);
        }
    }
}
