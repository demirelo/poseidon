//! Degree calibration for the zero-test root-first reduction.
//!
//! Fix a random cofactor G and write the root as r = X + Y*sqrt(3). The
//! root-first family P(z)=(z-r)G(z) makes P(r)=0 by construction, so the exact
//! zero-test condition becomes the fixed-point system
//!
//!   H(P_hat(X,Y))[0] - X = 0
//!   H(P_hat(X,Y))[1] - Y = 0
//!
//! This binary measures the degree of the eliminant Res_Y(f0, f1) on reduced
//! rounds. It is the zero-test analog of the CICO degree calibration: before we
//! spend effort on a full solver, measure whether compression/feedforward really
//! collapses the degree or whether the root-first system behaves generically.
//!
//! Run:
//!   cargo run --release --bin zerotest_calibrate
//!   cargo run --release --bin zerotest_calibrate heavy   # also tries RF=2/RP=3

use poseidon_core::poly as pl;
use poseidon_core::zerotest::{Fp2, ZT_D};
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

    fn fp2_nonzero(&mut self) -> Fp2 {
        loop {
            let z = Fp2::new(self.field(), self.field());
            if !z.is_zero() {
                return z;
            }
        }
    }
}

fn fixed_cofactor(seed: u64) -> [Fp2; ZT_D] {
    let mut rng = Rng::new(seed);
    let mut g = [Fp2::ZERO; ZT_D];
    for slot in &mut g {
        *slot = rng.fp2_nonzero();
    }
    g
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
    assert_eq!(pos.alpha, 3, "symbolic S-box specialised to alpha=3");
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
    // (x + Y*s)(a + b*s), s^2=3:
    //   c0 = x*a + 3*b*Y
    //   c1 = x*b + a*Y
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
    let f0 = pl::sub(&out[0], &[x]); // H(P_hat)[0] - X
    let f1 = pl::sub(&out[1], &[0, 1]); // H(P_hat)[1] - Y
    (pl::trim(&f0), pl::trim(&f1))
}

fn newton_coeffs(xs: &[u32], ys: &[u32]) -> Vec<u32> {
    let n = xs.len();
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
    coef
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

fn newton_eval(coef: &[u32], xs: &[u32], x: u32) -> u32 {
    if coef.is_empty() {
        return 0;
    }
    let mut acc = *coef.last().unwrap();
    for i in (0..coef.len() - 1).rev() {
        acc = fadd(fmul(acc, fsub(x, xs[i])), coef[i]);
    }
    acc
}

fn degree_from_newton(coef: &[u32]) -> i64 {
    coef.iter().rposition(|&c| c != 0).map_or(-1, |i| i as i64)
}

struct DegreeReport {
    deg: i64,
    nodes: usize,
    stable: bool,
    deg_y0: i64,
    deg_y1: i64,
    elapsed: f64,
}

fn measure_case(rf: usize, rp: usize, seed: u64, max_nodes: usize) -> DegreeReport {
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let cofactor = fixed_cofactor(seed);
    let guard = 8usize;
    let fwd = 3usize.pow((rf + rp) as u32);
    let hard_cap = (2 * fwd * fwd + 1).min(max_nodes).min((P - 1) as usize);
    let start = Instant::now();

    let (probe0, probe1) = residual_polys_at_x(&pos, &cofactor, 1);
    let deg_y0 = pl::deg(&probe0);
    let deg_y1 = pl::deg(&probe1);

    let mut samples: Vec<u32> = Vec::new();
    let ensure_samples = |samples: &mut Vec<u32>, count: usize| {
        while samples.len() < count {
            let x = (samples.len() + 1) as u32;
            let (f0, f1) = residual_polys_at_x(&pos, &cofactor, x);
            samples.push(pl::resultant(&f0, &f1));
            if samples.len().is_power_of_two() && samples.len() >= 1024 {
                println!("      sampled {} X-values...", samples.len());
            }
        }
    };

    let mut n = 16usize.min(hard_cap);
    loop {
        ensure_samples(&mut samples, (n + guard).min((P - 1) as usize));
        let xs: Vec<u32> = (1..=n as u32).collect();
        let coef = newton_coeffs(&xs, &samples[..n]);
        let deg = degree_from_newton(&coef);
        let stable = (0..guard).all(|g| {
            let x = (n + 1 + g) as u32;
            newton_eval(&coef, &xs, x) == samples[n + g]
        });
        println!(
            "      n={:<6} interpolated_deg={:<8} stable={}",
            n, deg, stable
        );
        if stable || n >= hard_cap {
            return DegreeReport {
                deg,
                nodes: n,
                stable,
                deg_y0,
                deg_y1,
                elapsed: start.elapsed().as_secs_f64(),
            };
        }
        n = (2 * n).min(hard_cap);
    }
}

fn pow3(e: usize) -> u128 {
    let mut x = 1u128;
    for _ in 0..e {
        x *= 3;
    }
    x
}

fn main() {
    let heavy = std::env::args().any(|a| a == "heavy");
    let mut cases = vec![(2usize, 0usize), (2, 1), (2, 2), (4, 0)];
    let max_nodes = if heavy { 140_000 } else { 20_000 };
    if heavy {
        cases.push((2, 3));
    }

    println!("Zero-test root-first degree calibration");
    println!("  system: Res_Y(H(P_hat(X,Y))[0]-X, H(P_hat(X,Y))[1]-Y)");
    println!("  seed: deterministic random cofactor G, degree 6");
    println!("  max interpolation nodes: {}\n", max_nodes);
    println!(
        "  {:>7} {:>8} {:>10} {:>14} {:>14} {:>10} {:>10} {:>9}",
        "RF/RP", "degY", "elim_deg", "3^(2RF+RP)", "3^(2RF+2RP)", "nodes", "stable", "seconds"
    );
    println!("  {}", "-".repeat(99));

    for (rf, rp) in cases {
        println!("\n  measuring RF={} RP={}...", rf, rp);
        let report = measure_case(
            rf,
            rp,
            0x5EED_0000 + ((rf as u64) << 8) + rp as u64,
            max_nodes,
        );
        let cico_like = pow3(2 * rf + rp);
        let generic = pow3(2 * (rf + rp));
        let class = if report.deg as u128 == cico_like {
            "CICO-like"
        } else if report.deg as u128 == generic {
            "generic"
        } else if report.deg >= 0 && (report.deg as u128) < generic {
            "subgeneric"
        } else {
            "other"
        };
        println!(
            "  {:>2}/{:<4} {:>3}/{:<4} {:>10} {:>14} {:>14} {:>10} {:>10} {:>8.2}   {}",
            rf,
            rp,
            report.deg_y0,
            report.deg_y1,
            report.deg,
            cico_like,
            generic,
            report.nodes,
            report.stable,
            report.elapsed,
            class
        );
    }

    println!("\nInterpretation:");
    println!("  * CICO-like scaling would support the claimed feedforward collapse.");
    println!("  * Generic 3^(2(RF+RP)) scaling means the naive root-first exact solve");
    println!("    is materially harder than CICO at fixed RF/RP, so the pivot needs a");
    println!("    stronger structural trick or becomes mainly paper/cluster work.");
}
