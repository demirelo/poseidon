//! Normal-domain F_p (KoalaBear) univariate polynomial arithmetic.
//!
//! Mirrors `poseidon_attack/poly.py` (validated against sympy). Correctness-first
//! and starts from a schoolbook executable reference. Slice 4 adds KoalaBear NTT
//! multiplication behind the same `mul` interface; callers keep the same logic
//! while large products use the field's 2^24-th roots of unity.
//!
//! Polynomials are `Vec<u32>` of coefficients low-degree-first; the zero
//! polynomial is the empty vec. Coefficients are normal-domain (not Montgomery),
//! matching the Python reference so results are directly comparable.

use crate::P;

const PU64: u64 = P as u64;
const TWO_ADICITY: usize = 24;
const MAX_NTT_LEN: usize = 1 << TWO_ADICITY;
const ROOT_2_24: u32 = 1791270792; // 3^127, primitive 2^24-th root in KoalaBear.
const NTT_MUL_THRESHOLD: usize = 256;
const FAST_DIV_THRESHOLD: usize = 256;

#[inline(always)]
fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b; // 2P < u32::MAX for KoalaBear.
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
#[inline(always)]
fn fmul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % PU64) as u32
}
#[inline(always)]
fn fmul_add2(a0: u32, b0: u32, a1: u32, b1: u32) -> u32 {
    // Two KoalaBear products fit in u64, so q_len=2 tails can reduce once.
    (((a0 as u64 * b0 as u64) + (a1 as u64 * b1 as u64)) % PU64) as u32
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
#[inline(always)]
fn fpow_small_exp(a: u32, e: i64) -> u32 {
    match e {
        0 => 1,
        1 => a,
        2 => fmul(a, a),
        3 => fmul(fmul(a, a), a),
        _ => fpow(a as u64, e as u64),
    }
}
#[inline(always)]
fn finv(a: u32) -> u32 {
    fpow(a as u64, (P - 2) as u64)
}

pub type Poly = Vec<u32>;

pub fn trim(f: &[u32]) -> Poly {
    let mut v = f.to_vec();
    trim_in_place(&mut v);
    v
}

fn trim_in_place(v: &mut Poly) {
    while matches!(v.last(), Some(&0)) {
        v.pop();
    }
}

fn trim_owned(mut v: Poly) -> Poly {
    trim_in_place(&mut v);
    v
}

/// Degree; the zero polynomial has degree -1.
pub fn deg(f: &[u32]) -> i64 {
    let mut n = f.len();
    while n > 0 && f[n - 1] == 0 {
        n -= 1;
    }
    n as i64 - 1
}

pub fn add(a: &[u32], b: &[u32]) -> Poly {
    let n = a.len().max(b.len());
    let mut r = vec![0u32; n];
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        r[i] = fadd(ai, bi);
    }
    trim(&r)
}

pub fn sub(a: &[u32], b: &[u32]) -> Poly {
    let n = a.len().max(b.len());
    let mut r = vec![0u32; n];
    for i in 0..n {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        r[i] = fsub(ai, bi);
    }
    trim(&r)
}

pub fn scalar(a: &[u32], c: u32) -> Poly {
    let c = c % P;
    if c == 0 {
        return vec![];
    }
    trim(&a.iter().map(|&x| fmul(x, c)).collect::<Vec<_>>())
}

/// Schoolbook multiplication. Kept public for goldens and for very small products.
pub fn mul_schoolbook(a: &[u32], b: &[u32]) -> Poly {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut r = vec![0u32; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            r[i + j] = fadd(r[i + j], fmul(ai, bj));
        }
    }
    trim(&r)
}

fn ntt(a: &mut [u32], invert: bool) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(n <= MAX_NTT_LEN);

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let root = if invert { finv(ROOT_2_24) } else { ROOT_2_24 };
    let mut len = 2usize;
    while len <= n {
        let log_len = len.trailing_zeros() as usize;
        let wlen = fpow(root as u64, 1u64 << (TWO_ADICITY - log_len));
        for chunk in a.chunks_exact_mut(len) {
            let mut w = 1u32;
            let half = len >> 1;
            for i in 0..half {
                let u = chunk[i];
                let v = fmul(chunk[i + half], w);
                chunk[i] = fadd(u, v);
                chunk[i + half] = fsub(u, v);
                w = fmul(w, wlen);
            }
        }
        len <<= 1;
    }

    if invert {
        let inv_n = finv(n as u32);
        for x in a.iter_mut() {
            *x = fmul(*x, inv_n);
        }
    }
}

/// NTT convolution over KoalaBear. Supports products whose padded length is at
/// most 2^24, the field's maximum power-of-two root-of-unity length.
pub fn mul_ntt(a: &[u32], b: &[u32]) -> Poly {
    let a = trim(a);
    let b = trim(b);
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let out_len = a.len() + b.len() - 1;
    let n = out_len.next_power_of_two();
    assert!(
        n <= MAX_NTT_LEN,
        "NTT length {} exceeds KoalaBear 2^{} limit",
        n,
        TWO_ADICITY
    );
    let mut fa = vec![0u32; n];
    let mut fb = vec![0u32; n];
    fa[..a.len()].copy_from_slice(&a);
    fb[..b.len()].copy_from_slice(&b);
    ntt(&mut fa, false);
    ntt(&mut fb, false);
    for i in 0..n {
        fa[i] = fmul(fa[i], fb[i]);
    }
    ntt(&mut fa, true);
    fa.truncate(out_len);
    trim(&fa)
}

/// Polynomial multiplication. Small products use schoolbook to avoid NTT setup
/// overhead; larger products use KoalaBear NTT when the padded length fits.
/// Products above the 2^24 NTT limit fail loudly instead of accidentally taking
/// an infeasible schoolbook path; call `mul_schoolbook` explicitly for tiny
/// experiments that intentionally bypass this guard.
pub fn mul(a: &[u32], b: &[u32]) -> Poly {
    let a0 = trim(a);
    let b0 = trim(b);
    if a0.is_empty() || b0.is_empty() {
        return vec![];
    }
    let out_len = a0.len() + b0.len() - 1;
    if out_len < NTT_MUL_THRESHOLD {
        return mul_schoolbook(&a0, &b0);
    }
    let n = out_len.next_power_of_two();
    if n <= MAX_NTT_LEN {
        mul_ntt(&a0, &b0)
    } else {
        panic!(
            "poly::mul product length {} needs NTT length {}, above KoalaBear 2^{} limit; use chunking/CRT or mul_schoolbook explicitly",
            out_len, n, TWO_ADICITY
        );
    }
}

/// Schoolbook long division. Kept public for goldens and tiny quotient cases.
pub fn divmod_schoolbook(a_in: &[u32], b_in: &[u32]) -> (Poly, Poly) {
    let a0 = trim(a_in);
    let b = trim(b_in);
    assert!(!b.is_empty(), "division by zero polynomial");
    if a0.len() < b.len() {
        return (vec![], a0);
    }
    let inv = finv(*b.last().unwrap());
    let mut a = a0;
    let bl = b.len();
    let mut q = vec![0u32; a.len() - bl + 1];
    for shift in (0..=(a.len() - bl)).rev() {
        let coef = fmul(a[bl - 1 + shift], inv);
        q[shift] = coef;
        if coef != 0 {
            for i in 0..bl {
                a[shift + i] = fsub(a[shift + i], fmul(coef, b[i]));
            }
        }
    }
    (trim(&q), trim(&a))
}

/// Schoolbook remainder without allocating the quotient. This is the hot path
/// for Euclidean resultant/GCD loops, where random equal-degree inputs usually
/// produce tiny quotients and the quotient itself is discarded.
pub fn rem_schoolbook(a_in: &[u32], b_in: &[u32]) -> Poly {
    let b = trim(b_in);
    assert!(!b.is_empty(), "division by zero polynomial");
    let a = trim(a_in);
    rem_schoolbook_trimmed(a, &b)
}

fn rem_schoolbook_trimmed(mut a: Poly, b: &[u32]) -> Poly {
    debug_assert!(!b.is_empty());
    if a.len() < b.len() {
        return a;
    }
    let inv = finv(*b.last().unwrap());
    let bl = b.len();
    let q_len = a.len() - bl + 1;
    if q_len == 1 {
        let coef = fmul(a[bl - 1], inv);
        for i in 0..bl - 1 {
            a[i] = fsub(a[i], fmul(coef, b[i]));
        }
        a.truncate(bl - 1);
        return trim_owned(a);
    }
    if q_len == 2 {
        if bl == 1 {
            return vec![];
        }
        let q1 = fmul(a[bl], inv);
        let top = fsub(a[bl - 1], fmul(q1, b[bl - 2]));
        let q0 = fmul(top, inv);
        if q0 == 0 {
            for i in 1..bl - 1 {
                a[i] = fsub(a[i], fmul(q1, b[i - 1]));
            }
        } else {
            a[0] = fsub(a[0], fmul(q0, b[0]));
            for i in 1..bl - 1 {
                a[i] = fsub(a[i], fmul_add2(q0, b[i], q1, b[i - 1]));
            }
        }
        a.truncate(bl - 1);
        return trim_owned(a);
    }
    for shift in (0..=(a.len() - bl)).rev() {
        let coef = fmul(a[bl - 1 + shift], inv);
        if coef != 0 {
            for i in 0..bl {
                a[shift + i] = fsub(a[shift + i], fmul(coef, b[i]));
            }
        }
    }
    trim_owned(a)
}

fn truncate(f: &[u32], n: usize) -> Poly {
    trim(&f[..f.len().min(n)])
}

fn reversed(f: &[u32]) -> Poly {
    let mut r = trim(f);
    r.reverse();
    trim(&r)
}

/// Multiplicative inverse of `f` modulo x^n. Requires f(0) != 0.
fn inv_series(f: &[u32], n: usize) -> Poly {
    assert!(n > 0);
    let f = trim(f);
    assert!(
        !f.is_empty() && f[0] != 0,
        "series inverse needs nonzero constant"
    );
    let mut g = vec![finv(f[0])];
    let mut m = 1usize;
    while m < n {
        let target = (m * 2).min(n);
        let fg = truncate(&mul(&truncate(&f, target), &g), target);
        let mut two_minus_fg = vec![0u32; target];
        two_minus_fg[0] = fsub(2, fg.first().copied().unwrap_or(0));
        for i in 1..fg.len() {
            two_minus_fg[i] = fsub(0, fg[i]);
        }
        g = truncate(&mul(&g, &two_minus_fg), target);
        m = target;
    }
    truncate(&g, n)
}

fn divmod_fast(a: &[u32], b: &[u32]) -> (Poly, Poly) {
    let a = trim(a);
    let b = trim(b);
    assert!(!b.is_empty(), "division by zero polynomial");
    if a.len() < b.len() {
        return (vec![], a);
    }
    let q_len = a.len() - b.len() + 1;
    let ar = truncate(&reversed(&a), q_len);
    let br = reversed(&b);
    let inv = inv_series(&br, q_len);
    let qr = truncate(&mul(&ar, &inv), q_len);
    let q = reversed(&qr);
    let r = sub(&a, &mul(&q, &b));
    (trim(&q), trim(&r))
}

/// Returns (quotient, remainder) with deg(remainder) < deg(divisor). Large
/// quotients use reversal + Newton series inversion, so division benefits from
/// the NTT-backed multiplication path.
pub fn divmod(a_in: &[u32], b_in: &[u32]) -> (Poly, Poly) {
    let a = trim(a_in);
    let b = trim(b_in);
    assert!(!b.is_empty(), "division by zero polynomial");
    if a.len() < b.len() {
        return (vec![], a);
    }
    let q_len = a.len() - b.len() + 1;
    if q_len < FAST_DIV_THRESHOLD {
        divmod_schoolbook(&a, &b)
    } else {
        divmod_fast(&a, &b)
    }
}

pub fn rem(a: &[u32], b: &[u32]) -> Poly {
    let a0 = trim(a);
    let b0 = trim(b);
    assert!(!b0.is_empty(), "division by zero polynomial");
    if a0.len() < b0.len() {
        return a0;
    }
    let q_len = a0.len() - b0.len() + 1;
    if q_len < FAST_DIV_THRESHOLD {
        rem_schoolbook_trimmed(a0, &b0)
    } else {
        divmod_fast(&a0, &b0).1
    }
}

/// Monic GCD over F_p.
pub fn gcd(a_in: &[u32], b_in: &[u32]) -> Poly {
    let mut a = trim(a_in);
    let mut b = trim(b_in);
    while !b.is_empty() {
        // Generic Euclidean GCD over random equal-degree polynomials mostly
        // takes tiny quotient steps; Newton division helps large quotients but
        // adds overhead here. Half-GCD is the real asymptotic fix.
        let r = rem_schoolbook_trimmed(a, &b);
        a = b;
        b = r;
    }
    if !a.is_empty() {
        let inv = finv(*a.last().unwrap());
        a = a.iter().map(|&c| fmul(c, inv)).collect();
    }
    a
}

pub fn eval(f: &[u32], x: u32) -> u32 {
    let mut acc = 0u32;
    for &c in f.iter().rev() {
        acc = fadd(fmul(acc, x), c);
    }
    acc
}

pub fn deriv(f: &[u32]) -> Poly {
    if f.len() <= 1 {
        return vec![];
    }
    trim(
        &(1..f.len())
            .map(|i| fmul((i as u32) % P, f[i]))
            .collect::<Vec<_>>(),
    )
}

/// base^e mod `m`, over F_p[x].
pub fn powmod(base: &[u32], mut e: u64, m: &[u32]) -> Poly {
    let mut result = vec![1u32];
    let mut b = rem(base, m);
    while e > 0 {
        if e & 1 == 1 {
            result = rem(&mul(&result, &b), m);
        }
        b = rem(&mul(&b, &b), m);
        e >>= 1;
    }
    result
}

/// Resultant Res(a, b) over F_p via the Euclidean remainder sequence.
pub fn resultant(a_in: &[u32], b_in: &[u32]) -> u32 {
    resultant_impl::<false>(a_in, b_in).0
}

#[derive(Clone, Debug, Default)]
pub struct ResultantStats {
    pub initial_deg_a: i64,
    pub initial_deg_b: i64,
    pub steps: usize,
    pub schoolbook_steps: usize,
    pub fast_steps: usize,
    pub q_len_1: usize,
    pub q_len_2: usize,
    pub q_len_3: usize,
    pub q_len_2_3: usize,
    pub q_len_4_15: usize,
    pub q_len_16_255: usize,
    pub q_len_fast: usize,
    pub max_q_len: usize,
    pub total_q_len: usize,
    pub total_divisor_len: usize,
    pub total_degree_drop: i64,
    pub max_degree_drop: i64,
    pub jump_events: Vec<ResultantJump>,
    pub zero_remainder: bool,
}

#[derive(Clone, Debug)]
pub struct ResultantJump {
    pub step: usize,
    pub q_len: usize,
    pub deg_a: i64,
    pub deg_b: i64,
    pub deg_r: i64,
    pub degree_drop: i64,
}

impl ResultantStats {
    fn record_step(&mut self, q_len: usize, deg_a: i64, deg_b: i64, deg_r: i64) {
        let divisor_len = (deg_b + 1) as usize;
        let degree_drop = deg_a - deg_r;
        self.steps += 1;
        self.max_q_len = self.max_q_len.max(q_len);
        self.total_q_len += q_len;
        self.total_divisor_len += divisor_len;
        self.total_degree_drop += degree_drop;
        self.max_degree_drop = self.max_degree_drop.max(degree_drop);
        if q_len < FAST_DIV_THRESHOLD {
            self.schoolbook_steps += 1;
        } else {
            self.fast_steps += 1;
        }
        match q_len {
            1 => self.q_len_1 += 1,
            2 => {
                self.q_len_2 += 1;
                self.q_len_2_3 += 1;
            }
            3 => {
                self.q_len_3 += 1;
                self.q_len_2_3 += 1;
            }
            4..=15 => self.q_len_4_15 += 1,
            16..=255 => self.q_len_16_255 += 1,
            _ => self.q_len_fast += 1,
        }
        if q_len >= 16 {
            self.jump_events.push(ResultantJump {
                step: self.steps,
                q_len,
                deg_a,
                deg_b,
                deg_r,
                degree_drop,
            });
        }
    }
}

pub fn resultant_profiled(a_in: &[u32], b_in: &[u32]) -> (u32, ResultantStats) {
    resultant_impl::<true>(a_in, b_in)
}

fn resultant_impl<const PROFILE: bool>(a_in: &[u32], b_in: &[u32]) -> (u32, ResultantStats) {
    let mut a = trim(a_in);
    let mut b = trim(b_in);
    let mut stats = ResultantStats {
        initial_deg_a: deg(&a),
        initial_deg_b: deg(&b),
        ..ResultantStats::default()
    };
    if a.is_empty() || b.is_empty() {
        stats.zero_remainder = true;
        return (0, stats);
    }
    let mut res = 1u32;
    if deg(&a) < deg(&b) {
        std::mem::swap(&mut a, &mut b);
        if (deg(&a) % 2 == 1) && (deg(&b) % 2 == 1) {
            res = fsub(0, res);
        }
    }
    while deg(&b) > 0 {
        let da = deg(&a);
        let db = deg(&b);
        let q_len = a.len() - b.len() + 1;
        let r = if q_len < FAST_DIV_THRESHOLD {
            rem_schoolbook_trimmed(a, &b)
        } else {
            divmod_fast(&a, &b).1
        };
        if r.is_empty() {
            if PROFILE {
                stats.zero_remainder = true;
            }
            return (0, stats); // shared factor
        }
        let dr = deg(&r);
        if PROFILE {
            stats.record_step(q_len, da, db, dr);
        }
        if (da * db) % 2 == 1 {
            res = fsub(0, res);
        }
        res = fmul(res, fpow_small_exp(*b.last().unwrap(), da - dr));
        a = b;
        b = r;
    }
    // deg(b) == 0:  Res(a, c) = c^deg(a)
    (fmul(res, fpow(b[0] as u64, deg(&a) as u64)), stats)
}

/// Deterministic SplitMix64 for reproducible equal-degree splitting.
pub struct SplitMix(u64);
impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn next_field(&mut self) -> u32 {
        (self.next_u64() % PU64) as u32
    }
}

fn split_roots(g_in: &[u32], rng: &mut SplitMix) -> Vec<u32> {
    let g0 = trim(g_in);
    let d = deg(&g0);
    if d <= 0 {
        return vec![];
    }
    let inv = finv(*g0.last().unwrap());
    let g: Vec<u32> = g0.iter().map(|&c| fmul(c, inv)).collect(); // monic
    if d == 1 {
        return vec![fsub(0, g[0])]; // x + g0 -> root -g0
    }
    loop {
        let a = rng.next_field();
        let h = powmod(&[a, 1], (PU64 - 1) / 2, &g);
        let c = gcd(&g, &sub(&h, &[1]));
        let dc = deg(&c);
        if dc > 0 && dc < d {
            let (other, _) = divmod(&g, &c);
            let mut roots = split_roots(&c, rng);
            roots.extend(split_roots(&other, rng));
            return roots;
        }
    }
}

/// All distinct roots of f in F_p (sorted).
pub fn roots(f_in: &[u32]) -> Vec<u32> {
    let f = trim(f_in);
    if deg(&f) <= 0 {
        return vec![];
    }
    // g = gcd(f, x^p - x) = product of (x - r) over distinct r in F_p
    let xp = powmod(&[0, 1], PU64, &f);
    let g = gcd(&f, &sub(&xp, &[0, 1]));
    if deg(&g) <= 0 {
        return vec![];
    }
    let mut rng = SplitMix::new(2026);
    let mut rs = split_roots(&g, &mut rng);
    rs.sort_unstable();
    rs.dedup();
    rs
}

/// Newton interpolation through (xs, ys); xs must be distinct.
pub fn interpolate(xs: &[u32], ys: &[u32]) -> Poly {
    let n = xs.len();
    assert_eq!(n, ys.len());
    let mut coef = ys.to_vec();
    let inv_steps = arithmetic_step(xs).map(|step| {
        let mut inv = vec![0u32; n];
        for (j, slot) in inv.iter_mut().enumerate().take(n).skip(1) {
            *slot = finv(fmul(step, j as u32));
        }
        inv
    });
    for j in 1..n {
        for i in (j..n).rev() {
            let inv = inv_steps
                .as_ref()
                .map_or_else(|| finv(fsub(xs[i], xs[i - j])), |steps| steps[j]);
            coef[i] = fmul(fsub(coef[i], coef[i - 1]), inv);
        }
    }
    let mut poly = vec![coef[n - 1]];
    for i in (0..n - 1).rev() {
        poly = add(&mul(&poly, &[fsub(0, xs[i]), 1]), &[coef[i]]);
    }
    trim(&poly)
}

fn arithmetic_step(xs: &[u32]) -> Option<u32> {
    if xs.len() < 2 {
        return None;
    }
    let step = fsub(xs[1], xs[0]);
    if step == 0 {
        return None;
    }
    xs.windows(2)
        .all(|w| fsub(w[1], w[0]) == step)
        .then_some(step)
}
