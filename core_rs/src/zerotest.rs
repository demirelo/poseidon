//! Native zero-test verifier and attack scaffolding.
//!
//! Mirrors `poseidon_attack.verifiers.verify_zerotest` and the reference
//! `zerotest_verifier.py`: a flat 16-word vector encodes 8 coefficients over
//! Fp2 = Fp[x]/(x^2 - 3), is hashed with Poseidon1 compression mode, and wins
//! iff P((out[0], out[1])) = 0 in Fp2.

use crate::{Poseidon1, P};

const PU64: u64 = P as u64;

pub const ZT_R: usize = 2;
pub const ZT_D: usize = 7;
pub const ZT_ELL: usize = 8;
pub const ZT_RF: usize = 6;
pub const ZT_T: usize = ZT_ELL * ZT_R;
pub const ZT_BETA: u32 = 3;
pub const ZT_COEFF_LEN: usize = (ZT_D + 1) * ZT_R;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Fp2 {
    pub c0: u32,
    pub c1: u32,
}

impl Fp2 {
    pub const ZERO: Fp2 = Fp2 { c0: 0, c1: 0 };
    pub const ONE: Fp2 = Fp2 { c0: 1, c1: 0 };

    #[inline]
    pub fn new(c0: u32, c1: u32) -> Self {
        Fp2 {
            c0: c0 % P,
            c1: c1 % P,
        }
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.c0 == 0 && self.c1 == 0
    }

    #[inline]
    pub fn add(self, rhs: Fp2) -> Fp2 {
        Fp2 {
            c0: fp_add(self.c0, rhs.c0),
            c1: fp_add(self.c1, rhs.c1),
        }
    }

    #[inline]
    pub fn sub(self, rhs: Fp2) -> Fp2 {
        Fp2 {
            c0: fp_sub(self.c0, rhs.c0),
            c1: fp_sub(self.c1, rhs.c1),
        }
    }

    #[inline]
    pub fn neg(self) -> Fp2 {
        Fp2 {
            c0: fp_neg(self.c0),
            c1: fp_neg(self.c1),
        }
    }

    #[inline]
    pub fn mul(self, rhs: Fp2) -> Fp2 {
        let c0 = fp_add(
            fp_mul(self.c0, rhs.c0),
            fp_mul(ZT_BETA, fp_mul(self.c1, rhs.c1)),
        );
        let c1 = fp_add(fp_mul(self.c0, rhs.c1), fp_mul(self.c1, rhs.c0));
        Fp2 { c0, c1 }
    }
}

#[inline]
fn fp_add(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}

#[inline]
fn fp_sub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + P - b
    }
}

#[inline]
fn fp_neg(a: u32) -> u32 {
    if a == 0 {
        0
    } else {
        P - a
    }
}

#[inline]
fn fp_mul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % PU64) as u32
}

pub fn coeffs_from_flat(p_hat: &[u32]) -> Option<[Fp2; ZT_D + 1]> {
    if p_hat.len() != ZT_COEFF_LEN {
        return None;
    }
    let mut coeffs = [Fp2::ZERO; ZT_D + 1];
    for j in 0..=ZT_D {
        coeffs[j] = Fp2::new(p_hat[2 * j], p_hat[2 * j + 1]);
    }
    Some(coeffs)
}

pub fn flatten_coeffs(coeffs: &[Fp2; ZT_D + 1]) -> [u32; ZT_COEFF_LEN] {
    let mut p_hat = [0u32; ZT_COEFF_LEN];
    for j in 0..=ZT_D {
        p_hat[2 * j] = coeffs[j].c0;
        p_hat[2 * j + 1] = coeffs[j].c1;
    }
    p_hat
}

pub fn degree(coeffs: &[Fp2]) -> isize {
    coeffs
        .iter()
        .rposition(|c| !c.is_zero())
        .map_or(-1, |j| j as isize)
}

pub fn eval_poly(coeffs: &[Fp2], x: Fp2) -> Fp2 {
    let Some((&last, rest)) = coeffs.split_last() else {
        return Fp2::ZERO;
    };
    let mut acc = last;
    for c in rest.iter().rev() {
        acc = acc.mul(x).add(*c);
    }
    acc
}

pub fn hash_point(pos: &Poseidon1, p_hat: &[u32]) -> Option<Fp2> {
    if p_hat.len() != ZT_COEFF_LEN {
        return None;
    }
    let mut input = [0u32; ZT_COEFF_LEN];
    for i in 0..ZT_COEFF_LEN {
        input[i] = p_hat[i] % P;
    }
    let out = pos.compression_mode_hash(&input, ZT_COEFF_LEN);
    Some(Fp2::new(out[0], out[1]))
}

pub fn zerotest_residual(pos: &Poseidon1, p_hat: &[u32]) -> Option<Fp2> {
    let coeffs = coeffs_from_flat(p_hat)?;
    let deg = degree(&coeffs);
    if deg < 1 || deg > ZT_D as isize {
        return None;
    }
    let a0 = hash_point(pos, p_hat)?;
    Some(eval_poly(&coeffs, a0))
}

pub fn verify_zerotest(pos: &Poseidon1, p_hat: &[u32]) -> bool {
    zerotest_residual(pos, p_hat).map_or(false, Fp2::is_zero)
}

pub fn verify_zerotest_rp(rp: usize, p_hat: &[u32]) -> bool {
    let pos = Poseidon1::new_canonical(3, ZT_T, ZT_RF, rp);
    verify_zerotest(&pos, p_hat)
}

pub fn verify_zerotest_relaxed(pos: &Poseidon1, p_hat: &[u32], k: u32) -> bool {
    let Some(r) = zerotest_residual(pos, p_hat) else {
        return false;
    };
    let mask = low_mask(k);
    (r.c0 & mask) == 0 && (r.c1 & mask) == 0
}

pub fn verify_zerotest_relaxed_rp(rp: usize, p_hat: &[u32], k: u32) -> bool {
    let pos = Poseidon1::new_canonical(3, ZT_T, ZT_RF, rp);
    verify_zerotest_relaxed(&pos, p_hat, k)
}

#[inline]
fn low_mask(k: u32) -> u32 {
    if k == 0 {
        0
    } else if k >= 31 {
        u32::MAX
    } else {
        (1u32 << k) - 1
    }
}

/// Build P(z) = (z - root) * G(z), where G has degree <= 6.
pub fn root_first_coeffs(root: Fp2, cofactor: &[Fp2; ZT_D]) -> [Fp2; ZT_D + 1] {
    let mut coeffs = [Fp2::ZERO; ZT_D + 1];
    coeffs[0] = root.neg().mul(cofactor[0]);
    for j in 1..ZT_D {
        coeffs[j] = cofactor[j - 1].sub(root.mul(cofactor[j]));
    }
    coeffs[ZT_D] = cofactor[ZT_D - 1];
    coeffs
}

/// For the root-first reduction, the exact target is H(P_hat)[0] == root.
pub fn root_first_hash_residual(pos: &Poseidon1, root: Fp2, cofactor: &[Fp2; ZT_D]) -> Fp2 {
    let coeffs = root_first_coeffs(root, cofactor);
    let p_hat = flatten_coeffs(&coeffs);
    hash_point(pos, &p_hat).unwrap().sub(root)
}

#[derive(Clone, Debug)]
pub struct RelaxedHit {
    pub p_hat: [u32; ZT_COEFF_LEN],
    pub attempt: usize,
    pub degree: isize,
    pub residual: Fp2,
}

pub fn search_relaxed_random(
    pos: &Poseidon1,
    k: u32,
    max_attempts: usize,
    seed: u64,
) -> Option<RelaxedHit> {
    let mut rng = SplitMix64::new(seed);
    for attempt in 1..=max_attempts {
        let (p_hat, deg) = random_poly_hat(&mut rng);
        if verify_zerotest_relaxed(pos, &p_hat, k) {
            let residual = zerotest_residual(pos, &p_hat).unwrap();
            return Some(RelaxedHit {
                p_hat,
                attempt,
                degree: deg,
                residual,
            });
        }
    }
    None
}

fn random_poly_hat(rng: &mut SplitMix64) -> ([u32; ZT_COEFF_LEN], isize) {
    let deg = 1 + (rng.next() as usize % ZT_D);
    let mut coeffs = [Fp2::ZERO; ZT_D + 1];
    for c in coeffs.iter_mut().take(deg) {
        *c = rng.fp2();
    }
    coeffs[deg] = rng.fp2_nonzero();
    (flatten_coeffs(&coeffs), deg as isize)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
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

    fn fp2(&mut self) -> Fp2 {
        Fp2::new(self.field(), self.field())
    }

    fn fp2_nonzero(&mut self) -> Fp2 {
        loop {
            let x = self.fp2();
            if !x.is_zero() {
                return x;
            }
        }
    }
}
