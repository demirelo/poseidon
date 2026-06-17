//! Reduced-round zero-test root-first solver.
//!
//! This is the executable counterpart to `zerotest_calibrate`: fix a cofactor
//! G, write r = X + Y*sqrt(3), build P(z)=(z-r)G(z), eliminate Y from
//! H(P_hat)[0]-X and H(P_hat)[1]-Y, then back-substitute roots and verify the
//! resulting P_hat with the native zero-test predicate.
//!
//! Run:
//!   cargo run --release --bin zerotest_solve
//!   cargo run --release --bin zerotest_solve heavy
//!   cargo run --release --bin zerotest_solve ultra   # also tries RF=4/RP=0
//!   cargo run --release --bin zerotest_solve case 2 4 8

use poseidon_core::poly as pl;
use poseidon_core::zerotest::{self, Fp2, ZT_D};
use poseidon_core::{Poseidon1, P};
use std::thread;
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

fn resultant_at_x(pos: &Poseidon1, cofactor: &[Fp2; ZT_D], x: u32) -> u32 {
    let (f0, f1) = residual_polys_at_x(pos, cofactor, x);
    pl::resultant(&f0, &f1)
}

fn sample_resultants(pos: &Poseidon1, cofactor: &[Fp2; ZT_D], xs: &[u32]) -> Vec<u32> {
    if xs.is_empty() {
        return vec![];
    }
    let workers = thread::available_parallelism().map_or(1, |n| n.get());
    let workers = workers.min(xs.len());
    if workers <= 1 || xs.len() < 1024 {
        return xs
            .iter()
            .map(|&x| resultant_at_x(pos, cofactor, x))
            .collect();
    }

    let chunk_len = (xs.len() + workers - 1) / workers;
    let mut chunks = thread::scope(|scope| {
        let handles: Vec<_> = xs
            .chunks(chunk_len)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                scope.spawn(move || {
                    let local = chunk
                        .iter()
                        .map(|&x| resultant_at_x(pos, cofactor, x))
                        .collect::<Vec<_>>();
                    (chunk_idx, local)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("sample worker panicked"))
            .collect::<Vec<_>>()
    });
    chunks.sort_by_key(|(chunk_idx, _)| *chunk_idx);
    chunks
        .into_iter()
        .flat_map(|(_, local)| local)
        .collect::<Vec<_>>()
}

fn expected_degree(rf: usize, rp: usize) -> usize {
    3usize.pow((2 * rf + rp) as u32)
}

fn interpolate_linear(xs: &[u32], ys: &[u32]) -> pl::Poly {
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
        let c = fsub(0, xs[i]); // multiply by (X - xs[i]) = c + X
        let mut next = vec![0u32; poly.len() + 1];
        for (j, &v) in poly.iter().enumerate() {
            next[j] = fadd(next[j], fmul(v, c));
            next[j + 1] = fadd(next[j + 1], v);
        }
        next[0] = fadd(next[0], coef[i]);
        poly = pl::trim(&next);
    }
    poly
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

fn resultant_poly(pos: &Poseidon1, cofactor: &[Fp2; ZT_D], degree: usize) -> Option<pl::Poly> {
    let guard = 8usize;
    let n = degree + 1;
    let xs: Vec<u32> = (1..=n as u32).collect();
    let ys = sample_resultants(pos, cofactor, &xs);
    let r_poly = interpolate_linear(&xs, &ys);
    let stable = (0..guard).all(|g| {
        let x = (n + 1 + g) as u32;
        let (f0, f1) = residual_polys_at_x(pos, cofactor, x);
        pl::eval(&r_poly, x) == pl::resultant(&f0, &f1)
    });
    stable.then_some(r_poly)
}

struct Hit {
    seed: u64,
    root: Fp2,
    p_hat: [u32; 16],
    elim_degree: i64,
    x_roots: usize,
    y_roots_seen: usize,
    elapsed: f64,
}

fn solve_case(rf: usize, rp: usize, max_seeds: usize) -> Option<Hit> {
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let deg = expected_degree(rf, rp);
    let started = Instant::now();

    for seed_idx in 0..max_seeds {
        let seed = 0x501E_0000 + ((rf as u64) << 16) + ((rp as u64) << 8) + seed_idx as u64;
        let cofactor = fixed_cofactor(seed);
        println!(
            "    seed {:>2}: interpolate degree {} eliminant...",
            seed_idx, deg
        );
        let Some(r_poly) = resultant_poly(&pos, &cofactor, deg) else {
            println!("      interpolation did not pass guard checks; trying next seed");
            continue;
        };
        let elim_degree = pl::deg(&r_poly);
        let x_roots = pl::roots(&r_poly);
        println!(
            "      elim_deg={}  Fp x-roots={}",
            elim_degree,
            x_roots.len()
        );

        let mut y_roots_seen = 0usize;
        for x in x_roots.iter().copied() {
            let (f0, f1) = residual_polys_at_x(&pos, &cofactor, x);
            let g = pl::gcd(&f0, &f1);
            for y in pl::roots(&g) {
                y_roots_seen += 1;
                let root = Fp2::new(x, y);
                let coeffs = zerotest::root_first_coeffs(root, &cofactor);
                let p_hat = zerotest::flatten_coeffs(&coeffs);
                if zerotest::verify_zerotest(&pos, &p_hat) {
                    return Some(Hit {
                        seed,
                        root,
                        p_hat,
                        elim_degree,
                        x_roots: x_roots.len(),
                        y_roots_seen,
                        elapsed: started.elapsed().as_secs_f64(),
                    });
                }
            }
        }
        println!(
            "      no verifier-accepted back-substitution (Y candidates seen={})",
            y_roots_seen
        );
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases = if args.get(1).map(|s| s.as_str()) == Some("case") {
        if args.len() != 5 {
            eprintln!("usage: zerotest_solve case <RF> <RP> <MAX_SEEDS>");
            std::process::exit(2);
        }
        let rf = args[2].parse::<usize>().expect("RF must be a usize");
        let rp = args[3].parse::<usize>().expect("RP must be a usize");
        let max_seeds = args[4].parse::<usize>().expect("MAX_SEEDS must be a usize");
        vec![(rf, rp, max_seeds)]
    } else if args.iter().any(|a| a == "ultra") {
        vec![
            (2usize, 0usize, 8usize),
            (2, 1, 8),
            (2, 2, 8),
            (2, 3, 4),
            (4, 0, 2),
        ]
    } else if args.iter().any(|a| a == "heavy") {
        vec![(2usize, 0usize, 8usize), (2, 1, 8), (2, 2, 8), (2, 3, 4)]
    } else {
        vec![(2usize, 0usize, 8usize), (2, 1, 8), (2, 2, 8)]
    };

    println!("Zero-test root-first reduced-round solver");
    println!("  method: interpolate Res_Y, factor X roots, gcd back-substitute Y, verify P_hat");
    println!();

    let total_cases = cases.len();
    let mut solved = 0usize;
    for (rf, rp, max_seeds) in cases {
        println!(
            "  solving RF={} RP={} (expected eliminant degree {})",
            rf,
            rp,
            expected_degree(rf, rp)
        );
        match solve_case(rf, rp, max_seeds) {
            Some(hit) => {
                solved += 1;
                println!(
                    "    SOLVED: root=({}, {}) seed=0x{:x} elim_deg={} x_roots={} y_seen={} elapsed={:.2}s",
                    hit.root.c0,
                    hit.root.c1,
                    hit.seed,
                    hit.elim_degree,
                    hit.x_roots,
                    hit.y_roots_seen,
                    hit.elapsed
                );
                println!("    p_hat={:?}", &hit.p_hat);
            }
            None => {
                println!("    no solution found in configured deterministic seeds");
            }
        }
        println!();
    }

    println!("=== solved {}/{} configured cases ===", solved, total_cases);
    std::process::exit(if solved > 0 { 0 } else { 1 });
}
