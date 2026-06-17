//! Amortized-resultant feasibility spike: a fast univariate resultant over
//! KoalaBear F_p, validated EXACTLY against the schoolbook `poly::resultant`
//! oracle before any speedup is claimed.
//!
//! STATUS: experimental exact-resultant VALIDATION HARNESS — NOT wired into
//! poly.rs and NOT used by any production solver. It exists only to prove the
//! fast path matches pl::resultant exactly; promote only if every gate matches.
//!
//! This binary is self-contained and edits nothing in poly.rs. It provides:
//!   * `resultant_eea`  -- plain Euclidean resultant (O(n^2)), the bookkeeping
//!                          reference that proves the sign / leading-coeff logic.
//!   * `resultant_fast` -- half-GCD driven resultant (~O(M(n) log n)) that
//!                          accumulates the SAME sign and lc^(degdrop) factors
//!                          for every elementary quotient the half-GCD skips.
//!
//! Both are checked against `poly::resultant` on thousands of random + planted
//! + real-Poseidon-residual cases. If anything fails to match the oracle the
//! program exits nonzero; the benchmark speedup is meaningless unless the
//! correctness gates pass.
//!
//! Subcommands:
//!   * `validate-lite`  (default, or any unrecognized arg) -- runs the
//!         shard-independent gates G0..G4 only. No external file is needed; CI
//!         can run this anywhere. Exits nonzero on ANY gate failure.
//!   * `validate-shard --path <csv> --rows <N>` -- the old G5: reproduces the
//!         landed RF=4/RP=5 fixed-root coordinate (var_g0) shard residuals and
//!         checks `resultant_fast` against the recorded value for the first <N>
//!         rows. REQUIRES the shard/fixture; FAILS HARD (nonzero exit) if the
//!         file is missing, the metadata disagrees with the derived seed/root,
//!         or any row mismatches. G5/shard reproduction runs ONLY here.
//!   * `bench` -- the resultant_fast vs schoolbook benchmark (runs after the
//!         lite gates pass).
//!
//! Run:  cargo run --release --bin zt_resultant_fast                 # validate-lite (G0..G4)
//!       cargo run --release --bin zt_resultant_fast validate-lite
//!       cargo run --release --bin zt_resultant_fast validate-shard --path tests/fixtures/rf4_rp5_var0_first_64.csv --rows 64
//!       cargo run --release --bin zt_resultant_fast bench

#![allow(dead_code)]

use poseidon_core::poly as pl;
use poseidon_core::zerotest::{Fp2, ZT_D};
use poseidon_core::{Poseidon1, P};
use std::time::Instant;

const PU64: u64 = P as u64;

// ===========================================================================
// Field helpers (normal domain, matching poly.rs semantics exactly)
// ===========================================================================
#[inline(always)]
fn fmul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % PU64) as u32
}
#[inline(always)]
fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}
#[inline(always)]
fn fsub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + P - b
    }
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

// ===========================================================================
// STEP 1: plain Euclidean resultant with explicit factor bookkeeping.
//
// This is a structural twin of poly.rs `resultant_impl` but written here so the
// sign / leading-coeff accounting is visible and independently checkable. The
// recurrence (coefficients low-to-high, deg of zero poly = -1):
//   res(A,0)=0 (m>0);  res(A,b)=b^m (const b!=0);  res(a,B)=a^n (const a)
//   if m<n: res(A,B) = (-1)^(m*n) res(B,A)
//   else:   R=A mod B, r=deg R, lcB=lead(B);
//           res(A,B) = (-1)^(m*n) * lcB^(m-r) * res(B,R)
// ===========================================================================
fn resultant_eea(a_in: &[u32], b_in: &[u32]) -> u32 {
    let mut a = pl::trim(a_in);
    let mut b = pl::trim(b_in);
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut res = 1u32;
    if pl::deg(&a) < pl::deg(&b) {
        std::mem::swap(&mut a, &mut b);
        if (pl::deg(&a) % 2 == 1) && (pl::deg(&b) % 2 == 1) {
            res = fsub(0, res);
        }
    }
    // Loop invariant: deg(a) >= deg(b). Drive the remainder sequence.
    while pl::deg(&b) > 0 {
        let da = pl::deg(&a);
        let db = pl::deg(&b);
        let lc_b = *b.last().unwrap();
        let r = pl::rem(&a, &b);
        if r.is_empty() {
            return 0; // shared factor
        }
        let dr = pl::deg(&r);
        if (da * db) % 2 == 1 {
            res = fsub(0, res);
        }
        res = fmul(res, pow_small(lc_b, da - dr));
        a = b;
        b = r;
    }
    // deg(b) == 0: b is a nonzero constant, Res(a, c) = c^deg(a).
    fmul(res, fpow(b[0] as u64, pl::deg(&a) as u64))
}

#[inline]
fn pow_small(a: u32, e: i64) -> u32 {
    debug_assert!(e >= 0);
    match e {
        0 => 1,
        1 => a,
        2 => fmul(a, a),
        3 => fmul(fmul(a, a), a),
        _ => fpow(a as u64, e as u64),
    }
}

// ===========================================================================
// Half-GCD with resultant step bookkeeping.
//
// The HGCD core is the validated recursive half-gcd from hgcd_bench.rs /
// gate_calibrate.rs (matched poly::gcd 241/241). We ADD a per-elementary-step
// record so the resultant factors can be reconstructed exactly.
//
// For one elementary Euclidean step (P,Q) -> (Q, R), R = P mod Q, the oracle
// multiplies res by  (-1)^(deg P * deg Q) * lc(Q)^(deg P - deg R).
// Two quantities are shift-invariant (a uniform x^k division of a pair leaves
// quotients and *leading coefficients* unchanged and shifts every degree by k):
//   * lc(Q)                       -- recorded directly
//   * rdrop := deg(Q) - deg(R)    -- recorded directly
// The sign needs ABSOLUTE degrees (deg P * deg Q parity is not shift-invariant),
// so we DO NOT compute the sign inside the recursion. Instead each step records
// (rdrop, lc); a final linear pass rolls absolute degrees forward from the
// known top-level (deg a, deg b) and applies the oracle formula verbatim.
//
// Step ordering matches the Euclidean sequence: within an hgcd call the steps
// are [left-recursion steps, the one middle division, right-recursion steps].
// ===========================================================================

#[derive(Clone, Copy, Debug)]
struct StepRec {
    rdrop: i64, // deg(divisor) - deg(remainder)  (shift-invariant, >= 1)
    lc: u32,    // leading coefficient of the divisor   (shift-invariant)
}

type Mat = [pl::Poly; 4];

fn ident() -> Mat {
    [vec![1], vec![], vec![], vec![1]]
}
fn neg(a: &[u32]) -> pl::Poly {
    pl::scalar(a, P - 1)
}
fn shr(a: &[u32], k: usize) -> pl::Poly {
    if k >= a.len() {
        return vec![];
    }
    pl::trim(&a[k..])
}
fn matvec(m: &Mat, a: &[u32], b: &[u32]) -> (pl::Poly, pl::Poly) {
    (
        pl::add(&pl::mul(&m[0], a), &pl::mul(&m[1], b)),
        pl::add(&pl::mul(&m[2], a), &pl::mul(&m[3], b)),
    )
}
fn matmul(x: &Mat, y: &Mat) -> Mat {
    [
        pl::add(&pl::mul(&x[0], &y[0]), &pl::mul(&x[1], &y[2])),
        pl::add(&pl::mul(&x[0], &y[1]), &pl::mul(&x[1], &y[3])),
        pl::add(&pl::mul(&x[2], &y[0]), &pl::mul(&x[3], &y[2])),
        pl::add(&pl::mul(&x[2], &y[1]), &pl::mul(&x[3], &y[3])),
    ]
}

/// Recursive Half-GCD. Precondition: deg(a) > deg(b) >= 0. Returns M with
/// M*(a,b)=(A,B), deg(A) > deg(a)/2 >= deg(B), and pushes one StepRec per
/// elementary Euclidean division performed, in chronological order, onto `rec`.
fn hgcd(a: &[u32], b: &[u32], rec: &mut Vec<StepRec>) -> Mat {
    let n = pl::deg(a);
    let m = ((n + 1) / 2) as usize; // ceil(n/2)
    if pl::deg(b) < m as i64 {
        return ident();
    }
    let r = hgcd(&shr(a, m), &shr(b, m), rec);
    let (aa, bb) = matvec(&r, a, b);
    if pl::deg(&bb) < m as i64 {
        return r;
    }
    // One real Euclidean step on the (aa,bb) pair. aa,bb are the actual current
    // polynomials (just possibly higher-degree than the local m), so deg(bb) and
    // lc(bb) are the genuine divisor degree/lead; deg(bb)-deg(c) is the genuine
    // remainder drop. Record it.
    let (q, c) = pl::divmod(&aa, &bb); // aa = q*bb + c
    let bdeg = pl::deg(&bb);
    let cdeg = pl::deg(&c);
    rec.push(StepRec {
        rdrop: bdeg - cdeg, // if c==0, cdeg=-1, rdrop = bdeg+1 (handled by consumer)
        lc: *bb.last().unwrap(),
    });
    let qmat: Mat = [vec![], vec![1], vec![1], neg(&q)]; // (aa,bb) -> (bb, c)
    let qr = matmul(&qmat, &r);
    let k = 2 * (m as i64) - bdeg;
    if k < 0 || cdeg < 0 {
        return qr;
    }
    let s = hgcd(&shr(&bb, k as usize), &shr(&c, k as usize), rec);
    matmul(&s, &qr)
}

/// Fast resultant: drives the remainder sequence with the half-GCD, recording
/// every elementary step, then applies the oracle's exact factor formula.
fn resultant_fast(a_in: &[u32], b_in: &[u32]) -> u32 {
    const SMALL: i64 = 48; // below this, the schoolbook EEA is faster + trivially exact

    let mut a = pl::trim(a_in);
    let mut b = pl::trim(b_in);
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut res = 1u32;
    if pl::deg(&a) < pl::deg(&b) {
        std::mem::swap(&mut a, &mut b);
        if (pl::deg(&a) % 2 == 1) && (pl::deg(&b) % 2 == 1) {
            res = fsub(0, res);
        }
    }

    // Running absolute degrees of the current (dividend, divisor) = (a, b).
    // `apply_step` consumes one StepRec, advancing these and folding the
    // sign + lc^(degdrop) factor into `res`. Returns Err if a zero remainder
    // is hit (shared factor => resultant 0).
    let mut da = pl::deg(&a);
    let mut db = pl::deg(&b);

    // Closure-free helper via macro-like inline function:
    // returns None on zero remainder (resultant 0).
    fn apply_step(res: &mut u32, da: &mut i64, db: &mut i64, st: StepRec) -> bool {
        // current step: dividend deg = *da, divisor deg = *db
        let dr = *db - st.rdrop; // remainder degree (-1 if zero)
        if dr < 0 {
            // zero remainder: shared factor => whole resultant is 0
            return false;
        }
        if ((*da) * (*db)) % 2 == 1 {
            *res = fsub(0, *res);
        }
        *res = fmul(*res, pow_small(st.lc, *da - dr));
        // advance sequence: (da, db) -> (db, dr)
        *da = *db;
        *db = dr;
        true
    }

    while db > 0 {
        // Outer loop mirrors gcd_fast: small pairs go straight to schoolbook EEA
        // (exact, cheap); otherwise advance with hgcd, then take explicit
        // division steps to keep deg strictly decreasing.
        if da < SMALL {
            // Finish with the schoolbook remainder sequence, folding factors in.
            while db > 0 {
                let lc_b = *b.last().unwrap();
                let r = pl::rem(&a, &b);
                if r.is_empty() {
                    return 0;
                }
                let dr = pl::deg(&r);
                if (da * db) % 2 == 1 {
                    res = fsub(0, res);
                }
                res = fmul(res, pow_small(lc_b, da - dr));
                a = b;
                b = r;
                da = db;
                db = dr;
            }
            break;
        }

        if da == db {
            // hgcd requires deg(a) > deg(b); take one explicit step first.
            let lc_b = *b.last().unwrap();
            let r = pl::rem(&a, &b);
            if r.is_empty() {
                return 0;
            }
            let dr = pl::deg(&r);
            if (da * db) % 2 == 1 {
                res = fsub(0, res);
            }
            res = fmul(res, pow_small(lc_b, da - dr));
            a = b;
            b = r;
            da = db;
            db = dr;
            continue;
        }

        // da > db, da >= SMALL: advance with hgcd.
        let mut rec: Vec<StepRec> = Vec::new();
        let mm = hgcd(&a, &b, &mut rec);
        let (aa, bb) = matvec(&mm, &a, &b);
        for st in &rec {
            if !apply_step(&mut res, &mut da, &mut db, *st) {
                return 0;
            }
        }
        a = pl::trim(&aa);
        b = pl::trim(&bb);
        // Sanity: running degrees must match the materialized pair. (Validated
        // exhaustively; kept as a debug guard.)
        debug_assert_eq!(da, pl::deg(&a), "running da out of sync with hgcd pair");
        debug_assert_eq!(db, pl::deg(&b), "running db out of sync with hgcd pair");

        if db >= 0 && db > 0 {
            // one explicit step to guarantee progress before the next hgcd
            let lc_b = *b.last().unwrap();
            let r = pl::rem(&a, &b);
            if r.is_empty() {
                return 0;
            }
            let dr = pl::deg(&r);
            if (da * db) % 2 == 1 {
                res = fsub(0, res);
            }
            res = fmul(res, pow_small(lc_b, da - dr));
            a = b;
            b = r;
            da = db;
            db = dr;
        }
    }

    if db < 0 {
        // b became the zero polynomial without ever being a nonzero constant:
        // this means a shared factor (gcd has positive degree). Resultant 0.
        // (Reached only if the last remainder was exactly zero, already handled,
        // so this is defensive.)
        return 0;
    }
    // db == 0: b is a nonzero constant. Res(a, c) = c^deg(a) = c^da.
    fmul(res, fpow(b[0] as u64, da as u64))
}

// ===========================================================================
// Tiny deterministic RNG (SplitMix64) -- same family as the rest of the repo.
// ===========================================================================
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s)
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
    /// Random poly of exact degree d (leading coeff forced nonzero).
    fn poly(&mut self, d: usize) -> pl::Poly {
        let mut v: Vec<u32> = (0..=d).map(|_| self.field()).collect();
        if v[d] == 0 {
            v[d] = 1;
        }
        v
    }
}

// ===========================================================================
// Poseidon residual builders (copied from the spike / scout binaries).
// Two flavors are needed:
//   (A) zt_amortized_spike-style: fixed cofactor, root = X + Y*s grid  -> G4
//   (B) zerotest_rootvar_scout-style: fixed root, coordinate-variable G -> G5
// ===========================================================================

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
fn neg_poly(f: &[u32]) -> pl::Poly {
    pl::scalar(f, P - 1)
}

// ---- Flavor (A): zt_amortized_spike (fixed cofactor, root = X + Y*s) --------
fn root_mul_g_at_x(x: u32, g: Fp2) -> (pl::Poly, pl::Poly) {
    (
        pl::trim(&[fmul(x, g.c0), fmul(3, g.c1)]),
        pl::trim(&[fmul(x, g.c1), g.c0]),
    )
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
fn residual_polys_at_x_spike(
    pos: &Poseidon1,
    cofactor: &[Fp2; ZT_D],
    x: u32,
) -> (pl::Poly, pl::Poly) {
    let input = root_first_state_at_x(x, cofactor);
    let out = compression_hash_poly(pos, &input);
    let f0 = pl::sub(&out[0], &[x]);
    let f1 = pl::sub(&out[1], &[0, 1]);
    (pl::trim(&f0), pl::trim(&f1))
}
fn full_cofactor(seed: u64) -> [Fp2; ZT_D] {
    let mut rng = Rng::new(seed);
    let mut g = [Fp2::ZERO; ZT_D];
    for slot in &mut g {
        *slot = rng.fp2_nonzero();
    }
    g
}

// ---- Flavor (B): zerotest_rootvar_scout (fixed root, coordinate-variable G) -
// EXACT replica of the cofactor derivation that produced the landed shards:
//   seed = 0xA17E_0000 + (rf<<24) + (rp<<16) + seed_idx
//   root = Rng(seed ^ 0xA11C_E000).fp2_nonzero()
//   base = Rng(seed) -> ZT_D nonzero Fp2
//   for var_idx slot: G = (x, Y*s) i.e. (vec![x], vec![0,1]); others constant.
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
fn residual_polys_at_x_scout(
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

// ===========================================================================
// Validation gates
// ===========================================================================
struct GateResult {
    name: &'static str,
    pass: bool,
    detail: String,
}

fn gate_g1(rng: &mut Rng) -> GateResult {
    let mut checked = 0usize;
    let mut mism_fast = 0usize;
    let mut mism_eea = 0usize;
    for _ in 0..1000 {
        let da = 2 + (rng.next() % 19) as usize; // 2..=20
        let db = 2 + (rng.next() % 19) as usize;
        let a = rng.poly(da);
        let b = rng.poly(db);
        let oracle = pl::resultant(&a, &b);
        let eea = resultant_eea(&a, &b);
        let fast = resultant_fast(&a, &b);
        checked += 1;
        if eea != oracle {
            mism_eea += 1;
        }
        if fast != oracle {
            mism_fast += 1;
        }
    }
    let pass = mism_fast == 0 && mism_eea == 0;
    GateResult {
        name: "G1 random deg 2..20 (fast==eea==oracle)",
        pass,
        detail: format!(
            "{} pairs checked; eea mismatches={}, fast mismatches={}",
            checked, mism_eea, mism_fast
        ),
    }
}

fn gate_g2(rng: &mut Rng) -> GateResult {
    let mut detail = String::new();
    let mut pass = true;
    for &d in &[1000usize, 4000, 16000] {
        let a = rng.poly(d);
        let db = d - 1 + (rng.next() % 2) as usize; // d or d-1
        let b = rng.poly(db);
        let oracle = pl::resultant(&a, &b);
        let fast = resultant_fast(&a, &b);
        let ok = oracle == fast;
        pass &= ok;
        detail.push_str(&format!(
            "deg {}: {} ({}={}, oracle={}); ",
            d,
            if ok { "match" } else { "MISMATCH" },
            "fast",
            fast,
            oracle
        ));
    }
    GateResult {
        name: "G2 large random deg 1000/4000/16000 (fast==oracle)",
        pass,
        detail,
    }
}

fn gate_g3(rng: &mut Rng) -> GateResult {
    let mut pass = true;
    let mut detail = String::new();

    // Planted common factor: (x-3)(x-5)*r1  vs  (x-3)(x-9)*r2  => resultant 0.
    // (x - c) in low-to-high coeffs is [-c, 1] = [P-c, 1].
    let xm3 = vec![P - 3, 1];
    let xm5 = vec![P - 5, 1];
    let xm9 = vec![P - 9, 1];
    let mut planted_ok = 0usize;
    let planted_n = 50usize;
    for _ in 0..planted_n {
        let d1 = 30 + (rng.next() % 20) as usize;
        let r1 = rng.poly(d1);
        let d2 = 30 + (rng.next() % 20) as usize;
        let r2 = rng.poly(d2);
        let a = pl::mul(&pl::mul(&xm3, &xm5), &r1);
        let b = pl::mul(&pl::mul(&xm3, &xm9), &r2);
        let fast = resultant_fast(&a, &b);
        let eea = resultant_eea(&a, &b);
        let oracle = pl::resultant(&a, &b);
        if fast == 0 && eea == 0 && oracle == 0 {
            planted_ok += 1;
        }
    }
    let planted_pass = planted_ok == planted_n;
    pass &= planted_pass;
    detail.push_str(&format!(
        "planted shared factor -> 0: {}/{}; ",
        planted_ok, planted_n
    ));

    // Coprime pairs: resultant must be nonzero AND match the oracle.
    let mut coprime_ok = 0usize;
    let mut coprime_match = 0usize;
    let coprime_n = 200usize;
    for _ in 0..coprime_n {
        // distinct linear factors guarantee coprimality with high prob; we
        // additionally only count cases the oracle itself reports nonzero, then
        // require fast to match the oracle (nonzero) on those.
        let a = pl::mul(&vec![P - 11, 1], &rng.poly(20));
        let b = pl::mul(&vec![P - 13, 1], &rng.poly(20));
        let oracle = pl::resultant(&a, &b);
        let fast = resultant_fast(&a, &b);
        if oracle != 0 {
            coprime_ok += 1;
            if fast == oracle {
                coprime_match += 1;
            }
        }
    }
    // Require: at least most are nonzero, and EVERY nonzero-oracle case matches.
    let coprime_pass = coprime_match == coprime_ok && coprime_ok > 0;
    pass &= coprime_pass;
    detail.push_str(&format!(
        "coprime nonzero-oracle cases matched by fast: {}/{} (nonzero seen {}/{})",
        coprime_match, coprime_ok, coprime_ok, coprime_n
    ));

    GateResult {
        name: "G3 planted (->0) and coprime (!=0, matched)",
        pass,
        detail,
    }
}

fn gate_g4() -> GateResult {
    // Real Poseidon residuals (zt_amortized_spike flavor) for RF=2/RP=4 AND
    // (extended) RF=4/RP=5, x=1..200, fast must equal oracle for each.
    let mut pass = true;
    let mut detail = String::new();
    for &(rf, rp) in &[(2usize, 4usize), (4usize, 5usize)] {
        let pos = Poseidon1::new_canonical(3, 16, rf, rp);
        let cofactor = full_cofactor(0);
        let mut checked = 0usize;
        let mut mism = 0usize;
        let mut deg_seen = (-1i64, -1i64);
        for x in 1..=200u32 {
            let (f0, f1) = residual_polys_at_x_spike(&pos, &cofactor, x);
            if x == 1 {
                deg_seen = (pl::deg(&f0), pl::deg(&f1));
            }
            let oracle = pl::resultant(&f0, &f1);
            let fast = resultant_fast(&f0, &f1);
            // also check eea for the small-degree RF=2 case as extra assurance
            let eea = resultant_eea(&f0, &f1);
            checked += 1;
            if fast != oracle || eea != oracle {
                mism += 1;
            }
        }
        pass &= mism == 0;
        detail.push_str(&format!(
            "RF={}/RP={} x=1..200 deg_y={:?}: {}/{} match; ",
            rf,
            rp,
            deg_seen,
            checked - mism,
            checked
        ));
    }
    GateResult {
        name: "G4 real Poseidon residuals (spike flavor) RF2/RP4 + RF4/RP5",
        pass,
        detail,
    }
}

/// G5 (shard-dependent): reproduce the landed RF=4/RP=5 fixed-root coordinate
/// (var_g0) shard residuals EXACTLY (scout flavor) and verify resultant_fast ==
/// recorded resultant for the first `rows` rows of `shard_path`.
///
/// This is gated behind the `validate-shard` subcommand and is a HARD gate:
/// unlike the old soft-skip behavior, a missing file, a metadata mismatch
/// against the derived seed/root, fewer than `rows` data rows, or ANY value
/// mismatch all return `pass=false` (the caller exits nonzero). The residual
/// construction and the `fast != recorded` comparison are byte-for-byte the
/// old G5 logic; only the file path / row count are now explicit parameters.
fn validate_shard(shard_path: &str, rows: usize) -> GateResult {
    const NAME: &str = "G5 reproduce landed shard residuals (scout flavor)";
    let rf = 4usize;
    let rp = 5usize;
    let seed_idx = 0usize;
    let var_idx = 0usize;
    let seed = scout_seed(rf, rp, seed_idx);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);

    let content = match std::fs::read_to_string(shard_path) {
        Ok(c) => c,
        Err(e) => {
            return GateResult {
                name: NAME,
                pass: false,
                detail: format!(
                    "FAIL: cannot read shard/fixture {} ({}). validate-shard requires the file.",
                    shard_path, e
                ),
            };
        }
    };

    // Cross-check shard metadata against our derived seed/root before trusting
    // it. The fixture header carries the same `seed=.. root_c0=.. root_c1=..`
    // tokens as a live shard; if a metadata comment is present it MUST agree.
    for line in content.lines() {
        // The first comment line that carries the seed/root provenance tokens
        // (live shard: "# rf=.. seed=.. root_c0=.."; fixture: "# seed_idx=.. seed=.. root_c0=..").
        if line.starts_with('#') && line.contains("seed=") && line.contains("root_c0=") {
            let want_seed = format!("seed={}", seed);
            let want_r0 = format!("root_c0={}", root.c0);
            let want_r1 = format!("root_c1={}", root.c1);
            if !line.contains(&want_seed) || !line.contains(&want_r0) || !line.contains(&want_r1) {
                return GateResult {
                    name: NAME,
                    pass: false,
                    detail: format!(
                        "FAIL: metadata mismatch: derived seed={} root=({},{}); shard line: {}",
                        seed, root.c0, root.c1, line
                    ),
                };
            }
            break;
        }
    }

    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let mut checked = 0usize;
    let mut mism = 0usize;
    let mut first_mismatch = String::new();
    for line in content.lines() {
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with("x,") {
            continue; // csv header
        }
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split(',');
        let x: u32 = match it.next().and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let recorded: u32 = match it.next().and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let (f0, f1) = residual_polys_at_x_scout(&pos, root, &base, var_idx, x);
        let fast = resultant_fast(&f0, &f1);
        checked += 1;
        if fast != recorded {
            mism += 1;
            if first_mismatch.is_empty() {
                first_mismatch = format!(
                    "x={} fast={} recorded={} (oracle={})",
                    x,
                    fast,
                    recorded,
                    pl::resultant(&f0, &f1)
                );
            }
        }
        if checked >= rows {
            break;
        }
    }
    if checked < rows {
        return GateResult {
            name: NAME,
            pass: false,
            detail: format!(
                "FAIL: requested {} rows but shard/fixture {} only had {} data rows",
                rows, shard_path, checked
            ),
        };
    }
    let pass = mism == 0 && checked > 0;
    GateResult {
        name: NAME,
        pass,
        detail: if pass {
            format!(
                "{}/{} rows from {}: resultant_fast == recorded shard resultant",
                checked, checked, shard_path
            )
        } else {
            format!(
                "{}/{} matched; first mismatch: {}",
                checked - mism,
                checked,
                first_mismatch
            )
        },
    }
}

// Extra correctness: degenerate / edge inputs vs the oracle.
fn gate_edges() -> GateResult {
    let mut pass = true;
    let mut detail = String::new();
    let cases: Vec<(pl::Poly, pl::Poly)> = vec![
        (vec![], vec![5, 1]),       // zero poly
        (vec![7], vec![]),          // const, zero
        (vec![7], vec![3]),         // two constants
        (vec![7], vec![3, 1, 2]),   // const a, deg-2 b
        (vec![3, 1, 2], vec![7]),   // deg-2 a, const b
        (vec![1, 1], vec![1, 1]),   // equal
        (vec![0, 0, 1], vec![0, 1]),// x^2 and x -> share x -> 0
        (vec![5], vec![0]),         // const and zero(trimmed empty)
    ];
    let mut ok = 0usize;
    for (a, b) in &cases {
        let oracle = pl::resultant(a, b);
        let eea = resultant_eea(a, b);
        let fast = resultant_fast(a, b);
        if oracle == eea && oracle == fast {
            ok += 1;
        } else {
            pass = false;
            detail.push_str(&format!(
                "EDGE a={:?} b={:?}: oracle={} eea={} fast={}; ",
                a, b, oracle, eea, fast
            ));
        }
    }
    if pass {
        detail = format!("{}/{} edge cases match oracle", ok, cases.len());
    }
    GateResult {
        name: "G0 degenerate/edge inputs",
        pass,
        detail,
    }
}

// ===========================================================================
// Benchmark
// ===========================================================================
fn bench(rng: &mut Rng) {
    println!("\nBENCHMARK (single-threaded): resultant_fast vs pl::resultant");
    println!(
        "   {:>8}  {:>14}  {:>12}  {:>9}  {:>6}",
        "degree", "schoolbook ms", "fast ms", "speedup", "match"
    );
    // Schoolbook is O(n^2); above this cap we project it from the last measured
    // point so wall-clock stays bounded but the table is still informative.
    const SCHOOL_CAP: usize = 16384;
    let mut last_school: Option<(usize, f64)> = None;
    let mut crossover: Option<usize> = None;
    let mut last_fast_ms = 0.0f64;
    let mut last_school_ms_real: Option<(usize, f64)> = None;

    let degs = [1024usize, 4096, 16384, 65536, 262144];
    for &d in &degs {
        let a = rng.poly(d);
        let b = rng.poly(d - 1);

        let t0 = Instant::now();
        let fast = resultant_fast(&a, &b);
        let fast_s = t0.elapsed().as_secs_f64();
        last_fast_ms = fast_s * 1e3;

        let (school_ms, projected, school_val) = if d <= SCHOOL_CAP {
            let t1 = Instant::now();
            let sb = pl::resultant(&a, &b);
            let s = t1.elapsed().as_secs_f64();
            last_school = Some((d, s));
            last_school_ms_real = Some((d, s * 1e3));
            (s * 1e3, false, Some(sb))
        } else {
            let (d0, t0e) = last_school.unwrap();
            (t0e * (d as f64 / d0 as f64).powi(2) * 1e3, true, None)
        };

        let speedup = school_ms / last_fast_ms;
        if speedup >= 1.0 && crossover.is_none() {
            crossover = Some(d);
        }
        let matched = match school_val {
            Some(v) => {
                if v == fast {
                    "yes"
                } else {
                    "NO!"
                }
            }
            None => "n/a",
        };
        println!(
            "   {:>8}  {:>12.3}{}  {:>12.3}  {:>8.2}x  {:>6}",
            d,
            school_ms,
            if projected { "*" } else { " " },
            last_fast_ms,
            speedup,
            matched
        );
    }
    println!("   (* = O(n^2)-projected schoolbook, not actually run)");

    match crossover {
        Some(c) => println!("   crossover degree (fast overtakes schoolbook): ~{}", c),
        None => println!("   crossover degree: fast did not overtake within tested range"),
    }

    // Extrapolate fast at 3^12 and 3^13 from the two largest fast measurements,
    // assuming ~ n*log2(n) growth (quasi-linear). Also extrapolate schoolbook
    // (n^2) to show the projected speedup ratio at those sizes.
    // Re-measure fast at the two largest sizes for a clean slope.
    let big = [65536usize, 262144usize];
    let mut pts: Vec<(usize, f64)> = Vec::new();
    for &d in &big {
        let a = rng.poly(d);
        let b = rng.poly(d - 1);
        let t0 = Instant::now();
        let _ = resultant_fast(&a, &b);
        pts.push((d, t0.elapsed().as_secs_f64() * 1e3));
    }
    let model_ms = |n: f64, (d0, t0): (usize, f64), (d1, t1): (usize, f64)| -> f64 {
        // fit t = c * n*log2(n); solve c from the larger point, sanity via slope
        let f = |x: f64| x * x.log2();
        // log-log slope between the two points for robustness
        let slope = (t1.ln() - t0.ln()) / ((d1 as f64).ln() - (d0 as f64).ln());
        let c = t1 / (d1 as f64).powf(slope);
        let est_power = c * n.powf(slope);
        let c2 = t1 / f(d1 as f64);
        let est_nlogn = c2 * f(n);
        // report the n*log n model (matches HGCD theory) but keep power-fit handy
        let _ = est_power;
        est_nlogn
    };
    for &target in &[531441usize /*3^12*/, 1594323 /*3^13*/] {
        let fast_est = model_ms(target as f64, pts[0], pts[1]);
        // schoolbook n^2 projection from the real measured small point
        let (sd, sms) = last_school_ms_real.unwrap();
        let school_est = sms * (target as f64 / sd as f64).powi(2);
        println!(
            "   extrapolated @ deg {:>7} (3^{}): fast~{:.1} ms, schoolbook~{:.1} ms (proj), speedup~{:.0}x",
            target,
            if target == 531441 { 12 } else { 13 },
            fast_est,
            school_est,
            school_est / fast_est
        );
    }
    let _ = last_fast_ms;
}

// ===========================================================================
// Measurement mode: time resultant_fast on ONE real Poseidon residual.
//
// `time-real <rf> <rp> [--schoolbook]` builds a single real zero-test residual
// pair (f0,f1) at (rf,rp) using the SAME spike route the G4 gate validates
// exactly (full_cofactor + residual_polys_at_x_spike), then times
// `resultant_fast` on it (wall seconds). The residual degree is deg_y = 3^(rf+rp)
// (empirically confirmed: RF2/RP4->3^6=729, RF4/RP5->3^9=19683). This is a real
// per-node cost at the bounty scale, NOT an extrapolation.
//
// Schoolbook (pl::resultant, O(n^2)) is timed only when `--schoolbook` is passed
// AND deg_y is small enough to finish in reasonable wall-clock; the printed JSON
// records fast_s, and optionally schoolbook_s + match.
// ===========================================================================
fn time_real(rf: usize, rp: usize, run_schoolbook: bool) {
    // Build one real residual at x=1 via the validated G4 (spike) route.
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let cofactor = full_cofactor(0);
    let build_t = Instant::now();
    let (f0, f1) = residual_polys_at_x_spike(&pos, &cofactor, 1u32);
    let build_s = build_t.elapsed().as_secs_f64();
    let deg_y = pl::deg(&f0).max(pl::deg(&f1));

    eprintln!(
        "time-real rf={} rp={}: built residual deg_y0={} deg_y1={} (build {:.3}s); timing resultant_fast...",
        rf,
        rp,
        pl::deg(&f0),
        pl::deg(&f1),
        build_s
    );

    // Time the fast resultant (the measured per-node cost).
    let t0 = Instant::now();
    let fast = resultant_fast(&f0, &f1);
    let fast_s = t0.elapsed().as_secs_f64();
    eprintln!("time-real rf={} rp={}: resultant_fast done in {:.3}s (value={})", rf, rp, fast_s, fast);

    // Optionally cross-check + time schoolbook pl::resultant.
    let mut schoolbook_s: Option<f64> = None;
    let mut matched: Option<bool> = None;
    if run_schoolbook {
        eprintln!("time-real rf={} rp={}: timing schoolbook pl::resultant (O(n^2))...", rf, rp);
        let t1 = Instant::now();
        let sb = pl::resultant(&f0, &f1);
        let s = t1.elapsed().as_secs_f64();
        schoolbook_s = Some(s);
        matched = Some(sb == fast);
        eprintln!(
            "time-real rf={} rp={}: schoolbook done in {:.3}s (value={}, match={})",
            rf, rp, s, sb, sb == fast
        );
    }

    // Emit a single JSON line on stdout.
    let sb_field = match schoolbook_s {
        Some(s) => format!("{:.6}", s),
        None => "null".to_string(),
    };
    let match_field = match matched {
        Some(m) => if m { "true".to_string() } else { "false".to_string() },
        None => "null".to_string(),
    };
    println!(
        "{{\"rf\":{},\"rp\":{},\"deg_y\":{},\"fast_s\":{:.6},\"schoolbook_s\":{},\"match\":{},\"build_s\":{:.6}}}",
        rf, rp, deg_y, fast_s, sb_field, match_field, build_s
    );
}

/// Run the shard-independent gates G0..G4. Returns true iff all passed; prints a
/// per-gate report and a summary. Used by both `validate-lite` and `bench`.
fn run_lite_gates(rng: &mut Rng) -> bool {
    println!("=== zt_resultant_fast: validate-lite gates G0..G4 (oracle = pl::resultant) ===");
    let results: Vec<GateResult> = vec![
        gate_edges(),
        gate_g1(rng),
        gate_g2(rng),
        gate_g3(rng),
        gate_g4(),
    ];
    let mut all_pass = true;
    for r in &results {
        println!(
            "[{}] {}\n        {}",
            if r.pass { "PASS" } else { "FAIL" },
            r.name,
            r.detail
        );
        all_pass &= r.pass;
    }
    println!("\n--- validate-lite summary (G0..G4) ---");
    for r in &results {
        println!("   {:>4}  {}", if r.pass { "PASS" } else { "FAIL" }, r.name);
    }
    all_pass
}

/// Parse `--path <csv> --rows <N>` out of the validate-shard argv tail.
fn parse_shard_args(args: &[String]) -> (Option<String>, Option<usize>) {
    let mut path: Option<String> = None;
    let mut rows: Option<usize> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => {
                path = args.get(i + 1).cloned();
                i += 2;
            }
            "--rows" => {
                rows = args.get(i + 1).and_then(|s| s.parse::<usize>().ok());
                i += 2;
            }
            _ => i += 1,
        }
    }
    (path, rows)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Subcommand: default to validate-lite when absent/unrecognized.
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("validate-lite");

    let mut rng = Rng::new(0x2026_06_04);

    match mode {
        "validate-shard" => {
            let (path, rows) = parse_shard_args(&args[2..]);
            let path = match path {
                Some(p) => p,
                None => {
                    eprintln!("usage: zt_resultant_fast validate-shard --path <csv> --rows <N>");
                    eprintln!("error: --path is required");
                    std::process::exit(2);
                }
            };
            let rows = match rows {
                Some(n) if n > 0 => n,
                _ => {
                    eprintln!("usage: zt_resultant_fast validate-shard --path <csv> --rows <N>");
                    eprintln!("error: --rows <N> is required and must be > 0");
                    std::process::exit(2);
                }
            };
            println!("=== zt_resultant_fast: validate-shard (G5 shard reproduction) ===");
            let r = validate_shard(&path, rows);
            println!(
                "[{}] {}\n        {}",
                if r.pass { "PASS" } else { "FAIL" },
                r.name,
                r.detail
            );
            if !r.pass {
                println!("\nSTATUS: BLOCKED -- G5 shard reproduction failed.");
                std::process::exit(1);
            }
            println!("\nSTATUS: DONE (G5 shard reproduction passed)");
        }
        "time-real" => {
            // Measurement mode: time resultant_fast on ONE real residual at (rf,rp).
            // usage: zt_resultant_fast time-real <rf> <rp> [--schoolbook]
            let rf = args.get(2).and_then(|s| s.parse::<usize>().ok());
            let rp = args.get(3).and_then(|s| s.parse::<usize>().ok());
            let (rf, rp) = match (rf, rp) {
                (Some(rf), Some(rp)) => (rf, rp),
                _ => {
                    eprintln!("usage: zt_resultant_fast time-real <rf> <rp> [--schoolbook]");
                    eprintln!("error: <rf> and <rp> are required positive integers");
                    std::process::exit(2);
                }
            };
            let run_schoolbook = args.iter().any(|a| a == "--schoolbook");
            time_real(rf, rp, run_schoolbook);
        }
        "bench" => {
            // Benchmark path: still requires the lite gates to hold first.
            let all_pass = run_lite_gates(&mut rng);
            if !all_pass {
                println!("\nSTATUS: BLOCKED -- a correctness gate failed; speedup numbers are NOT usable.");
                std::process::exit(1);
            }
            bench(&mut rng);
            println!("\nSTATUS: DONE (G0..G4 pass; benchmark complete)");
        }
        // "validate-lite" and anything unrecognized -> lite gates only.
        other => {
            if other != "validate-lite" {
                eprintln!(
                    "note: unrecognized arg {:?}; running validate-lite (G0..G4). \
                     Known modes: validate-lite | validate-shard --path <csv> --rows <N> | bench | time-real <rf> <rp> [--schoolbook]",
                    other
                );
            }
            let all_pass = run_lite_gates(&mut rng);
            if !all_pass {
                println!("\nSTATUS: BLOCKED -- a correctness gate failed; speedup numbers are NOT usable.");
                std::process::exit(1);
            }
            println!("\nSTATUS: DONE (all G0..G4 correctness gates pass)");
        }
    }
}
