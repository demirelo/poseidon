//! Fixed-root / variable-cofactor scout for the zero-test root-first attack.
//!
//! Instead of fixing a cofactor G and solving for a root r = X + Y*s, this
//! experiment fixes r and makes one Fp2 coefficient of G equal to X + Y*s.
//! The equation H((z-r)G)[0..1] = r is again solved by Res_Y, then
//! back-substituted and checked with the native zero-test verifier.
//!
//! Run:
//!   cargo run --release --bin zerotest_rootvar_scout
//!   cargo run --release --bin zerotest_rootvar_scout case 2 4 8
//!   cargo run --release --bin zerotest_rootvar_scout sample 4 5 0 0 8192
//!   cargo run --release --bin zerotest_rootvar_scout profile-x 4 5 0 0 1
//!   ZT_WORKERS=4 cargo run --release --bin zerotest_rootvar_scout sample-range 4 5 0 0 1 8192 scratch/rf4_rp5_var0_000.csv
//!   cargo run --release --bin zerotest_rootvar_scout sample-range 4 5 0 0 1 8192 scratch/rf4_rp5_var0_000.csv
//!   cargo run --release --bin zerotest_rootvar_scout solve-files 4 5 0 0 'scratch/rf4_rp5_var0_*.csv'
//!   cargo run --release --bin zerotest_rootvar_scout samples-status 4 5 8192 'scratch/rf4_rp5_var0_*.csv'
//!   cargo run --release --bin zerotest_rootvar_scout shape 4 5 8
//!   cargo run --release --bin zerotest_rootvar_scout plane-sample 4 5 0 0 2048
//!   cargo run --release --bin zerotest_rootvar_scout profile-plane-x 4 5 0 0 1
//!   cargo run --release --bin zerotest_rootvar_scout plane-shape 4 5 4 8
//!   cargo run --release --bin zerotest_rootvar_scout plane-case 4 3 2 4
//!   cargo run --release --bin zerotest_rootvar_scout plane-sample-range 4 5 0 0 1 8192 scratch/rf4_rp5_p0_000.csv
//!   cargo run --release --bin zerotest_rootvar_scout plane-solve-files 4 5 0 0 'scratch/rf4_rp5_p0_*.csv'
//!   cargo run --release --bin zerotest_rootvar_scout case 4 1 4 0
//!   cargo run --release --bin zerotest_rootvar_scout emit 4 4 0 0 25410007 859909975 166334946 791430512

use poseidon_core::poly as pl;
use poseidon_core::zerotest::{self, Fp2, ZT_D};
use poseidon_core::{Poseidon1, P};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
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

type Fp2Poly = (pl::Poly, pl::Poly);

fn fp2_poly_const(z: Fp2) -> Fp2Poly {
    (pl::trim(&[z.c0]), pl::trim(&[z.c1]))
}

fn fp2_scale(z: Fp2, k: u32) -> Fp2 {
    Fp2::new(fmul(z.c0, k), fmul(z.c1, k))
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

fn plane_dirs(seed: u64, plane_idx: usize) -> ([Fp2; ZT_D], [Fp2; ZT_D]) {
    let mut rng =
        Rng::new(seed ^ 0xC0FA_CE00_D15C_A11E_u64 ^ ((plane_idx as u64).wrapping_mul(0x9E37_79B9)));
    let mut x_dirs = [Fp2::ZERO; ZT_D];
    let mut y_dirs = [Fp2::ZERO; ZT_D];
    for j in 0..ZT_D {
        x_dirs[j] = rng.fp2_nonzero();
        y_dirs[j] = rng.fp2_nonzero();
    }
    (x_dirs, y_dirs)
}

fn scout_seed(rf: usize, rp: usize, seed_idx: usize) -> u64 {
    0xA17E_0000 + ((rf as u64) << 24) + ((rp as u64) << 16) + seed_idx as u64
}

fn checked_var_idx(var_idx: usize) -> usize {
    if var_idx >= ZT_D {
        eprintln!("VAR_IDX must be < {}", ZT_D);
        std::process::exit(2);
    }
    var_idx
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

fn cofactor_plane_polys_at_x(
    base: &[Fp2; ZT_D],
    x_dirs: &[Fp2; ZT_D],
    y_dirs: &[Fp2; ZT_D],
    x: u32,
) -> Vec<Fp2Poly> {
    let mut g = Vec::with_capacity(ZT_D);
    for j in 0..ZT_D {
        let c = base[j].add(fp2_scale(x_dirs[j], x));
        g.push((
            pl::trim(&[c.c0, y_dirs[j].c0]),
            pl::trim(&[c.c1, y_dirs[j].c1]),
        ));
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

fn root_first_state_at_x(root: Fp2, base: &[Fp2; ZT_D], var_idx: usize, x: u32) -> Vec<pl::Poly> {
    let g = cofactor_polys_at_x(base, var_idx, x);
    root_first_state_from_cofactor_polys(root, &g)
}

fn root_first_plane_state_at_x(
    root: Fp2,
    base: &[Fp2; ZT_D],
    x_dirs: &[Fp2; ZT_D],
    y_dirs: &[Fp2; ZT_D],
    x: u32,
) -> Vec<pl::Poly> {
    let g = cofactor_plane_polys_at_x(base, x_dirs, y_dirs, x);
    root_first_state_from_cofactor_polys(root, &g)
}

fn residual_polys_from_state(
    pos: &Poseidon1,
    root: Fp2,
    input: &[pl::Poly],
) -> (pl::Poly, pl::Poly) {
    let out = compression_hash_poly(pos, input);
    let f0 = pl::sub(&out[0], &[root.c0]);
    let f1 = pl::sub(&out[1], &[root.c1]);
    (pl::trim(&f0), pl::trim(&f1))
}

fn residual_polys_at_x(
    pos: &Poseidon1,
    root: Fp2,
    base: &[Fp2; ZT_D],
    var_idx: usize,
    x: u32,
) -> (pl::Poly, pl::Poly) {
    let input = root_first_state_at_x(root, base, var_idx, x);
    residual_polys_from_state(pos, root, &input)
}

fn plane_residual_polys_at_x(
    pos: &Poseidon1,
    root: Fp2,
    base: &[Fp2; ZT_D],
    x_dirs: &[Fp2; ZT_D],
    y_dirs: &[Fp2; ZT_D],
    x: u32,
) -> (pl::Poly, pl::Poly) {
    let input = root_first_plane_state_at_x(root, base, x_dirs, y_dirs, x);
    residual_polys_from_state(pos, root, &input)
}

fn resultant_at_x(pos: &Poseidon1, root: Fp2, base: &[Fp2; ZT_D], var_idx: usize, x: u32) -> u32 {
    let (f0, f1) = residual_polys_at_x(pos, root, base, var_idx, x);
    pl::resultant(&f0, &f1)
}

fn resultant_worker_count(sample_len: usize) -> usize {
    if sample_len < 1024 {
        return 1;
    }
    let available = thread::available_parallelism().map_or(1, |n| n.get());
    let requested = env::var("ZT_WORKERS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(available);
    requested.min(available).min(sample_len).max(1)
}

fn sample_resultants(
    pos: &Poseidon1,
    root: Fp2,
    base: &[Fp2; ZT_D],
    var_idx: usize,
    xs: &[u32],
) -> Vec<u32> {
    if xs.is_empty() {
        return vec![];
    }
    let workers = resultant_worker_count(xs.len());
    if workers <= 1 {
        return xs
            .iter()
            .map(|&x| resultant_at_x(pos, root, base, var_idx, x))
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
                        .map(|&x| resultant_at_x(pos, root, base, var_idx, x))
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

fn plane_resultant_at_x(
    pos: &Poseidon1,
    root: Fp2,
    base: &[Fp2; ZT_D],
    x_dirs: &[Fp2; ZT_D],
    y_dirs: &[Fp2; ZT_D],
    x: u32,
) -> u32 {
    let (f0, f1) = plane_residual_polys_at_x(pos, root, base, x_dirs, y_dirs, x);
    pl::resultant(&f0, &f1)
}

fn sample_plane_resultants(
    pos: &Poseidon1,
    root: Fp2,
    base: &[Fp2; ZT_D],
    x_dirs: &[Fp2; ZT_D],
    y_dirs: &[Fp2; ZT_D],
    xs: &[u32],
) -> Vec<u32> {
    if xs.is_empty() {
        return vec![];
    }
    let workers = resultant_worker_count(xs.len());
    if workers <= 1 {
        return xs
            .iter()
            .map(|&x| plane_resultant_at_x(pos, root, base, x_dirs, y_dirs, x))
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
                        .map(|&x| plane_resultant_at_x(pos, root, base, x_dirs, y_dirs, x))
                        .collect::<Vec<_>>();
                    (chunk_idx, local)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("plane sample worker panicked"))
            .collect::<Vec<_>>()
    });
    chunks.sort_by_key(|(chunk_idx, _)| *chunk_idx);
    chunks
        .into_iter()
        .flat_map(|(_, local)| local)
        .collect::<Vec<_>>()
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
        let c = fsub(0, xs[i]);
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

fn expected_degree(rf: usize, rp: usize) -> usize {
    3usize.pow((2 * rf + rp) as u32)
}

fn concrete_cofactor(base: &[Fp2; ZT_D], var_idx: usize, x: u32, y: u32) -> [Fp2; ZT_D] {
    let mut g = *base;
    g[var_idx] = Fp2::new(x, y);
    g
}

fn concrete_plane_cofactor(
    base: &[Fp2; ZT_D],
    x_dirs: &[Fp2; ZT_D],
    y_dirs: &[Fp2; ZT_D],
    x: u32,
    y: u32,
) -> [Fp2; ZT_D] {
    let mut g = [Fp2::ZERO; ZT_D];
    for j in 0..ZT_D {
        g[j] = base[j]
            .add(fp2_scale(x_dirs[j], x))
            .add(fp2_scale(y_dirs[j], y));
    }
    g
}

struct Hit {
    seed: u64,
    root: Fp2,
    var_idx: usize,
    x: u32,
    y: u32,
    p_hat: [u32; 16],
    elim_degree: i64,
    x_roots: usize,
    elapsed: f64,
}

struct PlaneHit {
    seed: u64,
    root: Fp2,
    plane_idx: usize,
    x: u32,
    y: u32,
    p_hat: [u32; 16],
    elim_degree: i64,
    x_roots: usize,
    elapsed: f64,
}

fn solve_seed(rf: usize, rp: usize, seed_idx: usize, var_idx: usize) -> Option<Hit> {
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let degree = expected_degree(rf, rp);
    let n = degree + 1;
    let guard = 4usize;
    let started = Instant::now();
    let xs = (1..=n as u32).collect::<Vec<_>>();

    let sample_start = Instant::now();
    let ys = sample_resultants(&pos, root, &base, var_idx, &xs);
    let sample_sec = sample_start.elapsed().as_secs_f64();
    let interp_start = Instant::now();
    let r_poly = interpolate_linear(&xs, &ys);
    let interp_sec = interp_start.elapsed().as_secs_f64();
    let stable = (0..guard).all(|g| {
        let x = (n + 1 + g) as u32;
        let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, x);
        pl::eval(&r_poly, x) == pl::resultant(&f0, &f1)
    });
    let roots_start = Instant::now();
    let x_roots = pl::roots(&r_poly);
    let roots_sec = roots_start.elapsed().as_secs_f64();
    let mut y_candidates = 0usize;
    for x in x_roots.iter().copied() {
        let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, x);
        let g = pl::gcd(&f0, &f1);
        for y in pl::roots(&g) {
            y_candidates += 1;
            let cofactor = concrete_cofactor(&base, var_idx, x, y);
            let coeffs = zerotest::root_first_coeffs(root, &cofactor);
            let p_hat = zerotest::flatten_coeffs(&coeffs);
            if zerotest::verify_zerotest(&pos, &p_hat) {
                println!(
                    "    seed {:>2}: HIT root=({}, {}) var_g{}=({}, {}) elim_deg={} xroots={} y_candidates={} sample={:.2}s interp={:.2}s roots={:.2}s total={:.2}s",
                    seed_idx,
                    root.c0,
                    root.c1,
                    var_idx,
                    x,
                    y,
                    pl::deg(&r_poly),
                    x_roots.len(),
                    y_candidates,
                    sample_sec,
                    interp_sec,
                    roots_sec,
                    started.elapsed().as_secs_f64()
                );
                return Some(Hit {
                    seed,
                    root,
                    var_idx,
                    x,
                    y,
                    p_hat,
                    elim_degree: pl::deg(&r_poly),
                    x_roots: x_roots.len(),
                    elapsed: started.elapsed().as_secs_f64(),
                });
            }
        }
    }
    println!(
        "    seed {:>2}: no hit stable={} elim_deg={} xroots={} y_candidates={} sample={:.2}s interp={:.2}s roots={:.2}s total={:.2}s root=({}, {})",
        seed_idx,
        stable,
        pl::deg(&r_poly),
        x_roots.len(),
        y_candidates,
        sample_sec,
        interp_sec,
        roots_sec,
        started.elapsed().as_secs_f64(),
        root.c0,
        root.c1
    );
    None
}

fn solve_plane_seed(rf: usize, rp: usize, seed_idx: usize, plane_idx: usize) -> Option<PlaneHit> {
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
    let degree = expected_degree(rf, rp);
    let n = degree + 1;
    let guard = 4usize;
    let started = Instant::now();
    let xs = (1..=n as u32).collect::<Vec<_>>();

    let sample_start = Instant::now();
    let ys = sample_plane_resultants(&pos, root, &base, &x_dirs, &y_dirs, &xs);
    let sample_sec = sample_start.elapsed().as_secs_f64();
    let interp_start = Instant::now();
    let r_poly = interpolate_linear(&xs, &ys);
    let interp_sec = interp_start.elapsed().as_secs_f64();
    let stable = (0..guard).all(|g| {
        let x = (n + 1 + g) as u32;
        let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, x);
        pl::eval(&r_poly, x) == pl::resultant(&f0, &f1)
    });
    let roots_start = Instant::now();
    let x_roots = pl::roots(&r_poly);
    let roots_sec = roots_start.elapsed().as_secs_f64();
    let mut y_candidates = 0usize;
    for x in x_roots.iter().copied() {
        let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, x);
        let g = pl::gcd(&f0, &f1);
        for y in pl::roots(&g) {
            y_candidates += 1;
            let cofactor = concrete_plane_cofactor(&base, &x_dirs, &y_dirs, x, y);
            let coeffs = zerotest::root_first_coeffs(root, &cofactor);
            let p_hat = zerotest::flatten_coeffs(&coeffs);
            if zerotest::verify_zerotest(&pos, &p_hat) {
                println!(
                    "    seed {:>2} plane {:>2}: HIT root=({}, {}) plane_xy=({}, {}) elim_deg={} xroots={} y_candidates={} sample={:.2}s interp={:.2}s roots={:.2}s total={:.2}s",
                    seed_idx,
                    plane_idx,
                    root.c0,
                    root.c1,
                    x,
                    y,
                    pl::deg(&r_poly),
                    x_roots.len(),
                    y_candidates,
                    sample_sec,
                    interp_sec,
                    roots_sec,
                    started.elapsed().as_secs_f64()
                );
                return Some(PlaneHit {
                    seed,
                    root,
                    plane_idx,
                    x,
                    y,
                    p_hat,
                    elim_degree: pl::deg(&r_poly),
                    x_roots: x_roots.len(),
                    elapsed: started.elapsed().as_secs_f64(),
                });
            }
        }
    }
    println!(
        "    seed {:>2} plane {:>2}: no hit stable={} elim_deg={} xroots={} y_candidates={} sample={:.2}s interp={:.2}s roots={:.2}s total={:.2}s root=({}, {})",
        seed_idx,
        plane_idx,
        stable,
        pl::deg(&r_poly),
        x_roots.len(),
        y_candidates,
        sample_sec,
        interp_sec,
        roots_sec,
        started.elapsed().as_secs_f64(),
        root.c0,
        root.c1
    );
    None
}

fn solve_case(rf: usize, rp: usize, max_seeds: usize, var_idx: usize) -> Option<Hit> {
    let var_idx = checked_var_idx(var_idx);
    println!(
        "  fixed-root variable-cofactor case RF={} RP={} var_g{} expected_degree={} max_seeds={}",
        rf,
        rp,
        var_idx,
        expected_degree(rf, rp),
        max_seeds
    );
    for seed_idx in 0..max_seeds {
        if let Some(hit) = solve_seed(rf, rp, seed_idx, var_idx) {
            return Some(hit);
        }
    }
    None
}

fn solve_plane_case(rf: usize, rp: usize, max_seeds: usize, max_planes: usize) -> Option<PlaneHit> {
    println!(
        "  fixed-root affine-plane cofactor case RF={} RP={} expected_degree={} max_seeds={} max_planes={}",
        rf,
        rp,
        expected_degree(rf, rp),
        max_seeds,
        max_planes
    );
    for seed_idx in 0..max_seeds {
        for plane_idx in 0..max_planes {
            if let Some(hit) = solve_plane_seed(rf, rp, seed_idx, plane_idx) {
                return Some(hit);
            }
        }
    }
    None
}

fn shape_case(args: &[String]) {
    if args.len() < 5 || args.len() > 6 {
        eprintln!("usage: zerotest_rootvar_scout shape <RF> <RP> <SEEDS> [X]");
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seeds = args[4].parse::<usize>().expect("SEEDS must parse");
    let x = args
        .get(5)
        .map_or(1u32, |s| s.parse::<u32>().expect("X must parse"));
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    println!("Zero-test fixed-root / variable-cofactor residual shape scan");
    println!(
        "  RF={} RP={} expected_degree={} x={} seeds={} vars={}",
        rf,
        rp,
        expected_degree(rf, rp),
        x,
        seeds,
        ZT_D
    );
    for seed_idx in 0..seeds {
        let seed = scout_seed(rf, rp, seed_idx);
        let root = fixed_root(seed);
        let base = fixed_cofactor(seed);
        for var_idx in 0..ZT_D {
            let started = Instant::now();
            let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, x);
            println!(
                "  seed={:>2} var_g{} root=({}, {}) degY0={} degY1={} build={:.3}s",
                seed_idx,
                var_idx,
                root.c0,
                root.c1,
                pl::deg(&f0),
                pl::deg(&f1),
                started.elapsed().as_secs_f64()
            );
        }
    }
}

fn plane_shape_case(args: &[String]) {
    if args.len() < 6 || args.len() > 7 {
        eprintln!("usage: zerotest_rootvar_scout plane-shape <RF> <RP> <SEEDS> <PLANES> [X]");
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seeds = args[4].parse::<usize>().expect("SEEDS must parse");
    let planes = args[5].parse::<usize>().expect("PLANES must parse");
    let x = args
        .get(6)
        .map_or(1u32, |s| s.parse::<u32>().expect("X must parse"));
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    println!("Zero-test fixed-root / affine-plane cofactor residual shape scan");
    println!(
        "  RF={} RP={} expected_degree={} x={} seeds={} planes={}",
        rf,
        rp,
        expected_degree(rf, rp),
        x,
        seeds,
        planes
    );
    for seed_idx in 0..seeds {
        let seed = scout_seed(rf, rp, seed_idx);
        let root = fixed_root(seed);
        let base = fixed_cofactor(seed);
        for plane_idx in 0..planes {
            let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
            let started = Instant::now();
            let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, x);
            println!(
                "  seed={:>2} plane={:>2} root=({}, {}) degY0={} degY1={} build={:.3}s",
                seed_idx,
                plane_idx,
                root.c0,
                root.c1,
                pl::deg(&f0),
                pl::deg(&f1),
                started.elapsed().as_secs_f64()
            );
        }
    }
}

fn sample_case(args: &[String]) {
    if args.len() != 7 {
        eprintln!("usage: zerotest_rootvar_scout sample <RF> <RP> <SEED_IDX> <VAR_IDX> <NODES>");
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let var_idx = checked_var_idx(args[5].parse::<usize>().expect("VAR_IDX must parse"));
    let nodes = args[6].parse::<usize>().expect("NODES must parse").max(1);
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let full_nodes = expected_degree(rf, rp) + 1;
    let nodes = nodes.min((P - 1) as usize);
    let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, 1);
    let xs = (1..=nodes as u32).collect::<Vec<_>>();
    let workers = resultant_worker_count(xs.len());
    let started = Instant::now();
    let ys = sample_resultants(&pos, root, &base, var_idx, &xs);
    let sample_sec = started.elapsed().as_secs_f64();
    let zero_resultants = ys.iter().filter(|&&y| y == 0).count();
    let ms_per_node = 1000.0 * sample_sec / nodes as f64;
    let projected_full_sample_sec = sample_sec * full_nodes as f64 / nodes as f64;
    println!("Zero-test fixed-root / variable-cofactor sample-only profile");
    println!(
        "  RF={} RP={} seed_idx={} seed={} var_g{} root=({}, {}) expected_degree={} full_nodes={}",
        rf,
        rp,
        seed_idx,
        seed,
        var_idx,
        root.c0,
        root.c1,
        full_nodes - 1,
        full_nodes
    );
    println!(
        "  nodes={} workers={} degY0={} degY1={} zeros={} sample={:.2}s ms/node={:.3} projected_full_sample={:.2}s projected_days={:.2}",
        nodes,
        workers,
        pl::deg(&f0),
        pl::deg(&f1),
        zero_resultants,
        sample_sec,
        ms_per_node,
        projected_full_sample_sec,
        projected_full_sample_sec / 86400.0
    );
}

fn plane_sample_case(args: &[String]) {
    if args.len() != 7 {
        eprintln!(
            "usage: zerotest_rootvar_scout plane-sample <RF> <RP> <SEED_IDX> <PLANE_IDX> <NODES>"
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let plane_idx = args[5].parse::<usize>().expect("PLANE_IDX must parse");
    let nodes = args[6].parse::<usize>().expect("NODES must parse").max(1);
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
    let full_nodes = expected_degree(rf, rp) + 1;
    let nodes = nodes.min((P - 1) as usize);
    let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, 1);
    let xs = (1..=nodes as u32).collect::<Vec<_>>();
    let workers = resultant_worker_count(xs.len());
    let started = Instant::now();
    let ys = sample_plane_resultants(&pos, root, &base, &x_dirs, &y_dirs, &xs);
    let sample_sec = started.elapsed().as_secs_f64();
    let zero_resultants = ys.iter().filter(|&&y| y == 0).count();
    let ms_per_node = 1000.0 * sample_sec / nodes as f64;
    let projected_full_sample_sec = sample_sec * full_nodes as f64 / nodes as f64;
    println!("Zero-test fixed-root / affine-plane cofactor sample-only profile");
    println!(
        "  RF={} RP={} seed_idx={} seed={} plane={} root=({}, {}) expected_degree={} full_nodes={}",
        rf,
        rp,
        seed_idx,
        seed,
        plane_idx,
        root.c0,
        root.c1,
        full_nodes - 1,
        full_nodes
    );
    println!(
        "  nodes={} workers={} degY0={} degY1={} zeros={} sample={:.2}s ms/node={:.3} projected_full_sample={:.2}s projected_days={:.2}",
        nodes,
        workers,
        pl::deg(&f0),
        pl::deg(&f1),
        zero_resultants,
        sample_sec,
        ms_per_node,
        projected_full_sample_sec,
        projected_full_sample_sec / 86400.0
    );
}

fn print_resultant_profile(res: u32, stats: &pl::ResultantStats, seconds: f64) {
    let avg_q = if stats.steps == 0 {
        0.0
    } else {
        stats.total_q_len as f64 / stats.steps as f64
    };
    let avg_divisor_len = if stats.steps == 0 {
        0.0
    } else {
        stats.total_divisor_len as f64 / stats.steps as f64
    };
    println!(
        "  resultant={} seconds={:.4} initial_deg=({}, {}) steps={} schoolbook={} fast={}",
        res,
        seconds,
        stats.initial_deg_a,
        stats.initial_deg_b,
        stats.steps,
        stats.schoolbook_steps,
        stats.fast_steps
    );
    println!(
        "  q1={} q2={} q3={} q2_3={} q4_15={} q16_255={} qfast={} max_q={} avg_q={:.3} avg_divisor_len={:.1} max_drop={} zero_remainder={}",
        stats.q_len_1,
        stats.q_len_2,
        stats.q_len_3,
        stats.q_len_2_3,
        stats.q_len_4_15,
        stats.q_len_16_255,
        stats.q_len_fast,
        stats.max_q_len,
        avg_q,
        avg_divisor_len,
        stats.max_degree_drop,
        stats.zero_remainder
    );
}

fn profile_x_case(args: &[String]) {
    if args.len() != 7 {
        eprintln!("usage: zerotest_rootvar_scout profile-x <RF> <RP> <SEED_IDX> <VAR_IDX> <X>");
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let var_idx = checked_var_idx(args[5].parse::<usize>().expect("VAR_IDX must parse"));
    let x = args[6].parse::<u32>().expect("X must parse");
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let build_started = Instant::now();
    let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, x);
    let build_sec = build_started.elapsed().as_secs_f64();
    let started = Instant::now();
    let (res, stats) = pl::resultant_profiled(&f0, &f1);
    let seconds = started.elapsed().as_secs_f64();
    println!("Zero-test coordinate variable-cofactor resultant profile");
    println!(
        "  RF={} RP={} seed_idx={} seed={} var_g{} root=({}, {}) x={} degY0={} degY1={}",
        rf,
        rp,
        seed_idx,
        seed,
        var_idx,
        root.c0,
        root.c1,
        x,
        pl::deg(&f0),
        pl::deg(&f1)
    );
    println!(
        "  build_seconds={:.4} resultant_seconds={:.4} total_seconds={:.4}",
        build_sec,
        seconds,
        build_sec + seconds
    );
    print_resultant_profile(res, &stats, seconds);
}

fn profile_plane_x_case(args: &[String]) {
    if args.len() != 7 {
        eprintln!(
            "usage: zerotest_rootvar_scout profile-plane-x <RF> <RP> <SEED_IDX> <PLANE_IDX> <X>"
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let plane_idx = args[5].parse::<usize>().expect("PLANE_IDX must parse");
    let x = args[6].parse::<u32>().expect("X must parse");
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
    let build_started = Instant::now();
    let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, x);
    let build_sec = build_started.elapsed().as_secs_f64();
    let started = Instant::now();
    let (res, stats) = pl::resultant_profiled(&f0, &f1);
    let seconds = started.elapsed().as_secs_f64();
    println!("Zero-test affine-plane cofactor resultant profile");
    println!(
        "  RF={} RP={} seed_idx={} seed={} plane={} root=({}, {}) x={} degY0={} degY1={}",
        rf,
        rp,
        seed_idx,
        seed,
        plane_idx,
        root.c0,
        root.c1,
        x,
        pl::deg(&f0),
        pl::deg(&f1)
    );
    println!(
        "  build_seconds={:.4} resultant_seconds={:.4} total_seconds={:.4}",
        build_sec,
        seconds,
        build_sec + seconds
    );
    print_resultant_profile(res, &stats, seconds);
}

fn checked_x_range(start_x: u32, nodes: usize) -> Vec<u32> {
    let nodes = nodes.max(1);
    if start_x == 0 {
        eprintln!("START_X must be >= 1 for the standard interpolation grid");
        std::process::exit(2);
    }
    let end = start_x as u64 + nodes as u64 - 1;
    if end >= P as u64 {
        eprintln!("sample range must stay below field modulus P={}", P);
        std::process::exit(2);
    }
    (0..nodes).map(|i| start_x + i as u32).collect::<Vec<_>>()
}

fn sample_range_case(args: &[String]) {
    if args.len() != 9 {
        eprintln!(
            "usage: zerotest_rootvar_scout sample-range <RF> <RP> <SEED_IDX> <VAR_IDX> <START_X> <NODES> <OUT>"
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let var_idx = checked_var_idx(args[5].parse::<usize>().expect("VAR_IDX must parse"));
    let start_x = args[6].parse::<u32>().expect("START_X must parse");
    let nodes = args[7].parse::<usize>().expect("NODES must parse");
    let out_path = &args[8];
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let xs = checked_x_range(start_x, nodes);
    let workers = resultant_worker_count(xs.len());
    let started = Instant::now();
    let ys = sample_resultants(&pos, root, &base, var_idx, &xs);
    let sample_sec = started.elapsed().as_secs_f64();
    let file = File::create(out_path).expect("create sample output");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# zerotest_rootvar_coordinate_samples v1").expect("write header");
    writeln!(
        writer,
        "# rf={} rp={} seed_idx={} seed={} var_idx={} root_c0={} root_c1={} start_x={} nodes={} workers={}",
        rf,
        rp,
        seed_idx,
        seed,
        var_idx,
        root.c0,
        root.c1,
        start_x,
        xs.len(),
        workers
    )
    .expect("write metadata");
    writeln!(writer, "x,resultant").expect("write csv header");
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        writeln!(writer, "{},{}", x, y).expect("write sample row");
    }
    writer.flush().expect("flush sample output");
    println!("Zero-test coordinate variable-cofactor sample shard");
    println!(
        "  RF={} RP={} seed_idx={} seed={} var_g{} root=({}, {})",
        rf, rp, seed_idx, seed, var_idx, root.c0, root.c1
    );
    println!(
        "  wrote={} start_x={} nodes={} workers={} sample={:.2}s ms/node={:.3}",
        out_path,
        start_x,
        xs.len(),
        workers,
        sample_sec,
        1000.0 * sample_sec / xs.len() as f64
    );
}

fn plane_sample_range_case(args: &[String]) {
    if args.len() != 9 {
        eprintln!(
            "usage: zerotest_rootvar_scout plane-sample-range <RF> <RP> <SEED_IDX> <PLANE_IDX> <START_X> <NODES> <OUT>"
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let plane_idx = args[5].parse::<usize>().expect("PLANE_IDX must parse");
    let start_x = args[6].parse::<u32>().expect("START_X must parse");
    let nodes = args[7].parse::<usize>().expect("NODES must parse");
    let out_path = &args[8];
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
    let xs = checked_x_range(start_x, nodes);
    let workers = resultant_worker_count(xs.len());
    let started = Instant::now();
    let ys = sample_plane_resultants(&pos, root, &base, &x_dirs, &y_dirs, &xs);
    let sample_sec = started.elapsed().as_secs_f64();
    let file = File::create(out_path).expect("create sample output");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# zerotest_rootvar_plane_samples v1").expect("write header");
    writeln!(
        writer,
        "# rf={} rp={} seed_idx={} seed={} plane_idx={} root_c0={} root_c1={} start_x={} nodes={} workers={}",
        rf,
        rp,
        seed_idx,
        seed,
        plane_idx,
        root.c0,
        root.c1,
        start_x,
        xs.len(),
        workers
    )
    .expect("write metadata");
    writeln!(writer, "x,resultant").expect("write csv header");
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        writeln!(writer, "{},{}", x, y).expect("write sample row");
    }
    writer.flush().expect("flush sample output");
    println!("Zero-test affine-plane cofactor sample shard");
    println!(
        "  RF={} RP={} seed_idx={} seed={} plane={} root=({}, {})",
        rf, rp, seed_idx, seed, plane_idx, root.c0, root.c1
    );
    println!(
        "  wrote={} start_x={} nodes={} workers={} sample={:.2}s ms/node={:.3}",
        out_path,
        start_x,
        xs.len(),
        workers,
        sample_sec,
        1000.0 * sample_sec / xs.len() as f64
    );
}

fn load_sample_file(path: &str) -> Vec<(u32, u32)> {
    let file = File::open(path).expect("open sample file");
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.expect("read sample line");
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') || s == "x,resultant" {
            continue;
        }
        let Some((x, y)) = s.split_once(',') else {
            panic!("{}:{} expected x,resultant CSV row", path, lineno + 1);
        };
        out.push((
            x.parse::<u32>().expect("sample x must parse"),
            y.parse::<u32>().expect("sample resultant must parse"),
        ));
    }
    out
}

fn sorted_unique_samples(mut samples: Vec<(u32, u32)>) -> (Vec<u32>, Vec<u32>) {
    samples.sort_unstable_by_key(|&(x, _)| x);
    let mut xs = Vec::with_capacity(samples.len());
    let mut ys = Vec::with_capacity(samples.len());
    for (x, y) in samples {
        if xs.last().copied() == Some(x) {
            let prev = *ys.last().unwrap();
            if prev != y {
                panic!("conflicting duplicate sample for x={}", x);
            }
            continue;
        }
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

fn summarize_status_samples(mut samples: Vec<(u32, u32)>) -> (Vec<u32>, usize, usize, usize) {
    samples.sort_unstable();
    let mut xs = Vec::with_capacity(samples.len());
    let mut duplicate_rows = 0usize;
    let mut conflicting_xs = 0usize;
    let mut zero_resultants = 0usize;
    let mut i = 0usize;
    while i < samples.len() {
        let x = samples[i].0;
        let y0 = samples[i].1;
        let mut has_conflict = false;
        let mut has_zero = y0 == 0;
        let mut j = i + 1;
        while j < samples.len() && samples[j].0 == x {
            has_conflict |= samples[j].1 != y0;
            has_zero |= samples[j].1 == 0;
            j += 1;
        }
        xs.push(x);
        duplicate_rows += (j - i).saturating_sub(1);
        if has_conflict {
            conflicting_xs += 1;
        }
        if has_zero {
            zero_resultants += 1;
        }
        i = j;
    }
    (xs, duplicate_rows, conflicting_xs, zero_resultants)
}

fn missing_ranges(covered: &[bool], max_ranges: usize) -> (usize, Vec<(u32, usize)>) {
    let mut missing = 0usize;
    let mut ranges = Vec::new();
    let mut i = 0usize;
    while i < covered.len() {
        if covered[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < covered.len() && !covered[i] {
            i += 1;
        }
        let len = i - start;
        missing += len;
        if ranges.len() < max_ranges {
            ranges.push(((start + 1) as u32, len));
        }
    }
    (missing, ranges)
}

fn samples_status_case(args: &[String]) {
    if args.len() < 5 {
        eprintln!("usage: zerotest_rootvar_scout samples-status <RF> <RP> <SHARD_NODES> [FILE]...");
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let shard_nodes = args[4]
        .parse::<usize>()
        .expect("SHARD_NODES must parse")
        .max(1);
    let expected = expected_degree(rf, rp) + 1;
    let mut samples = Vec::new();
    for path in &args[5..] {
        samples.extend(load_sample_file(path));
    }
    let loaded = samples.len();
    let (xs, duplicates, conflicting_xs, zero_resultants) = summarize_status_samples(samples);
    let mut covered = vec![false; expected];
    let mut out_of_grid = 0usize;
    for &x in &xs {
        if x >= 1 && (x as usize) <= expected {
            covered[x as usize - 1] = true;
        } else {
            out_of_grid += 1;
        }
    }
    let (missing, ranges) = missing_ranges(&covered, 8);
    println!("Zero-test sample shard coverage");
    println!(
        "  RF={} RP={} expected_degree={} expected_samples={} files={} loaded_rows={} distinct_samples={} duplicates={} conflicting_xs={} out_of_grid={}",
        rf,
        rp,
        expected - 1,
        expected,
        args.len().saturating_sub(5),
        loaded,
        xs.len(),
        duplicates,
        conflicting_xs,
        out_of_grid
    );
    println!(
        "  covered={} missing={} complete={}",
        expected - missing,
        missing,
        missing == 0 && conflicting_xs == 0 && out_of_grid == 0
    );
    if !ranges.is_empty() {
        println!("  first_missing_ranges:");
        for (start, len) in ranges {
            let todo = len.min(shard_nodes);
            println!(
                "    start_x={} missing_len={} next_nodes={}",
                start, len, todo
            );
        }
    }
    if missing > 0 {
        if let Some((start, len)) = missing_ranges(&covered, 1).1.first().copied() {
            println!(
                "  next_shard_args: START_X={} NODES={}",
                start,
                len.min(shard_nodes)
            );
        }
    }
    if !xs.is_empty() {
        println!("  zero_resultants={}", zero_resultants);
    }
}

fn solve_from_samples(
    rf: usize,
    rp: usize,
    seed_idx: usize,
    var_idx: usize,
    xs: &[u32],
    ys: &[u32],
) -> Option<Hit> {
    let var_idx = checked_var_idx(var_idx);
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let degree = expected_degree(rf, rp);
    let n = degree + 1;
    if xs.len() != n {
        eprintln!(
            "need exactly {} distinct samples for RF={} RP={} degree {}; got {}",
            n,
            rf,
            rp,
            degree,
            xs.len()
        );
        std::process::exit(1);
    }
    let started = Instant::now();
    let interp_start = Instant::now();
    let r_poly = interpolate_linear(xs, ys);
    let interp_sec = interp_start.elapsed().as_secs_f64();
    let guard = 4usize;
    let stable = (0..guard).all(|g| {
        let x = (n + 1 + g) as u32;
        let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, x);
        pl::eval(&r_poly, x) == pl::resultant(&f0, &f1)
    });
    let roots_start = Instant::now();
    let x_roots = pl::roots(&r_poly);
    let roots_sec = roots_start.elapsed().as_secs_f64();
    let mut y_candidates = 0usize;
    for x in x_roots.iter().copied() {
        let (f0, f1) = residual_polys_at_x(&pos, root, &base, var_idx, x);
        let g = pl::gcd(&f0, &f1);
        for y in pl::roots(&g) {
            y_candidates += 1;
            let cofactor = concrete_cofactor(&base, var_idx, x, y);
            let coeffs = zerotest::root_first_coeffs(root, &cofactor);
            let p_hat = zerotest::flatten_coeffs(&coeffs);
            if zerotest::verify_zerotest(&pos, &p_hat) {
                println!(
                    "    samples var_g{}: HIT root=({}, {}) var_g{}=({}, {}) stable={} elim_deg={} xroots={} y_candidates={} interp={:.2}s roots={:.2}s total={:.2}s",
                    var_idx,
                    root.c0,
                    root.c1,
                    var_idx,
                    x,
                    y,
                    stable,
                    pl::deg(&r_poly),
                    x_roots.len(),
                    y_candidates,
                    interp_sec,
                    roots_sec,
                    started.elapsed().as_secs_f64()
                );
                return Some(Hit {
                    seed,
                    root,
                    var_idx,
                    x,
                    y,
                    p_hat,
                    elim_degree: pl::deg(&r_poly),
                    x_roots: x_roots.len(),
                    elapsed: started.elapsed().as_secs_f64(),
                });
            }
        }
    }
    println!(
        "    samples var_g{}: no hit stable={} elim_deg={} xroots={} y_candidates={} interp={:.2}s roots={:.2}s total={:.2}s root=({}, {})",
        var_idx,
        stable,
        pl::deg(&r_poly),
        x_roots.len(),
        y_candidates,
        interp_sec,
        roots_sec,
        started.elapsed().as_secs_f64(),
        root.c0,
        root.c1
    );
    None
}

fn solve_plane_from_samples(
    rf: usize,
    rp: usize,
    seed_idx: usize,
    plane_idx: usize,
    xs: &[u32],
    ys: &[u32],
) -> Option<PlaneHit> {
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let root = fixed_root(seed);
    let base = fixed_cofactor(seed);
    let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
    let degree = expected_degree(rf, rp);
    let n = degree + 1;
    if xs.len() != n {
        eprintln!(
            "need exactly {} distinct samples for RF={} RP={} degree {}; got {}",
            n,
            rf,
            rp,
            degree,
            xs.len()
        );
        std::process::exit(1);
    }
    let started = Instant::now();
    let interp_start = Instant::now();
    let r_poly = interpolate_linear(xs, ys);
    let interp_sec = interp_start.elapsed().as_secs_f64();
    let guard = 4usize;
    let stable = (0..guard).all(|g| {
        let x = (n + 1 + g) as u32;
        let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, x);
        pl::eval(&r_poly, x) == pl::resultant(&f0, &f1)
    });
    let roots_start = Instant::now();
    let x_roots = pl::roots(&r_poly);
    let roots_sec = roots_start.elapsed().as_secs_f64();
    let mut y_candidates = 0usize;
    for x in x_roots.iter().copied() {
        let (f0, f1) = plane_residual_polys_at_x(&pos, root, &base, &x_dirs, &y_dirs, x);
        let g = pl::gcd(&f0, &f1);
        for y in pl::roots(&g) {
            y_candidates += 1;
            let cofactor = concrete_plane_cofactor(&base, &x_dirs, &y_dirs, x, y);
            let coeffs = zerotest::root_first_coeffs(root, &cofactor);
            let p_hat = zerotest::flatten_coeffs(&coeffs);
            if zerotest::verify_zerotest(&pos, &p_hat) {
                println!(
                    "    samples plane {:>2}: HIT root=({}, {}) plane_xy=({}, {}) stable={} elim_deg={} xroots={} y_candidates={} interp={:.2}s roots={:.2}s total={:.2}s",
                    plane_idx,
                    root.c0,
                    root.c1,
                    x,
                    y,
                    stable,
                    pl::deg(&r_poly),
                    x_roots.len(),
                    y_candidates,
                    interp_sec,
                    roots_sec,
                    started.elapsed().as_secs_f64()
                );
                return Some(PlaneHit {
                    seed,
                    root,
                    plane_idx,
                    x,
                    y,
                    p_hat,
                    elim_degree: pl::deg(&r_poly),
                    x_roots: x_roots.len(),
                    elapsed: started.elapsed().as_secs_f64(),
                });
            }
        }
    }
    println!(
        "    samples plane {:>2}: no hit stable={} elim_deg={} xroots={} y_candidates={} interp={:.2}s roots={:.2}s total={:.2}s root=({}, {})",
        plane_idx,
        stable,
        pl::deg(&r_poly),
        x_roots.len(),
        y_candidates,
        interp_sec,
        roots_sec,
        started.elapsed().as_secs_f64(),
        root.c0,
        root.c1
    );
    None
}

fn plane_solve_files_case(args: &[String]) {
    if args.len() < 7 {
        eprintln!(
            "usage: zerotest_rootvar_scout plane-solve-files <RF> <RP> <SEED_IDX> <PLANE_IDX> <FILE>..."
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let plane_idx = args[5].parse::<usize>().expect("PLANE_IDX must parse");
    let mut samples = Vec::new();
    for path in &args[6..] {
        samples.extend(load_sample_file(path));
    }
    let loaded = samples.len();
    let (xs, ys) = sorted_unique_samples(samples);
    println!("Zero-test fixed-root / affine-plane cofactor solve from sample files");
    println!(
        "  RF={} RP={} seed_idx={} plane={} loaded_rows={} distinct_samples={} files={}",
        rf,
        rp,
        seed_idx,
        plane_idx,
        loaded,
        xs.len(),
        args.len() - 6
    );
    let hit = solve_plane_from_samples(rf, rp, seed_idx, plane_idx, &xs, &ys);
    if let Some(hit) = hit {
        println!(
            "    accepted p_hat={:?} seed={} plane={} xy=({}, {}) root=({}, {}) elim_deg={} xroots={} elapsed={:.2}s",
            hit.p_hat,
            hit.seed,
            hit.plane_idx,
            hit.x,
            hit.y,
            hit.root.c0,
            hit.root.c1,
            hit.elim_degree,
            hit.x_roots,
            hit.elapsed
        );
        return;
    }
    std::process::exit(1);
}

fn solve_files_case(args: &[String]) {
    if args.len() < 7 {
        eprintln!(
            "usage: zerotest_rootvar_scout solve-files <RF> <RP> <SEED_IDX> <VAR_IDX> <FILE>..."
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let var_idx = checked_var_idx(args[5].parse::<usize>().expect("VAR_IDX must parse"));
    let mut samples = Vec::new();
    for path in &args[6..] {
        samples.extend(load_sample_file(path));
    }
    let loaded = samples.len();
    let (xs, ys) = sorted_unique_samples(samples);
    println!("Zero-test fixed-root / coordinate variable-cofactor solve from sample files");
    println!(
        "  RF={} RP={} seed_idx={} var_g{} loaded_rows={} distinct_samples={} files={}",
        rf,
        rp,
        seed_idx,
        var_idx,
        loaded,
        xs.len(),
        args.len() - 6
    );
    let hit = solve_from_samples(rf, rp, seed_idx, var_idx, &xs, &ys);
    if let Some(hit) = hit {
        println!(
            "    accepted p_hat={:?} seed={} var_g{}=({}, {}) root=({}, {}) elim_deg={} xroots={} elapsed={:.2}s",
            hit.p_hat,
            hit.seed,
            hit.var_idx,
            hit.x,
            hit.y,
            hit.root.c0,
            hit.root.c1,
            hit.elim_degree,
            hit.x_roots,
            hit.elapsed
        );
        return;
    }
    std::process::exit(1);
}

fn plane_exact_case(args: &[String]) -> Vec<(usize, usize, usize, usize)> {
    if args.len() != 6 {
        eprintln!("usage: zerotest_rootvar_scout plane-case <RF> <RP> <MAX_SEEDS> <MAX_PLANES>");
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let max_seeds = args[4].parse::<usize>().expect("MAX_SEEDS must parse");
    let max_planes = args[5].parse::<usize>().expect("MAX_PLANES must parse");
    vec![(rf, rp, max_seeds, max_planes)]
}

fn emit_hit(args: &[String]) {
    if args.len() != 10 {
        eprintln!(
            "usage: zerotest_rootvar_scout emit <RF> <RP> <SEED_IDX> <VAR_IDX> <ROOT_C0> <ROOT_C1> <X> <Y>"
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let var_idx = checked_var_idx(args[5].parse::<usize>().expect("VAR_IDX must parse"));
    let root = Fp2::new(
        args[6].parse::<u32>().expect("ROOT_C0 must parse"),
        args[7].parse::<u32>().expect("ROOT_C1 must parse"),
    );
    let x = args[8].parse::<u32>().expect("X must parse");
    let y = args[9].parse::<u32>().expect("Y must parse");
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let base = fixed_cofactor(seed);
    let cofactor = concrete_cofactor(&base, var_idx, x, y);
    let coeffs = zerotest::root_first_coeffs(root, &cofactor);
    let p_hat = zerotest::flatten_coeffs(&coeffs);
    let residual = zerotest::zerotest_residual(&pos, &p_hat);
    println!("fixed-root variable-cofactor materialized witness");
    println!(
        "  rf={} rp={} seed_idx={} seed={} var_g{}=({}, {}) root=({}, {})",
        rf, rp, seed_idx, seed, var_idx, x, y, root.c0, root.c1
    );
    println!("  verify={}", zerotest::verify_zerotest(&pos, &p_hat));
    println!("  residual={:?}", residual);
    println!("  p_hat={:?}", p_hat);
}

fn emit_plane_hit(args: &[String]) {
    if args.len() != 10 {
        eprintln!(
            "usage: zerotest_rootvar_scout emit-plane <RF> <RP> <SEED_IDX> <PLANE_IDX> <ROOT_C0> <ROOT_C1> <X> <Y>"
        );
        std::process::exit(2);
    }
    let rf = args[2].parse::<usize>().expect("RF must parse");
    let rp = args[3].parse::<usize>().expect("RP must parse");
    let seed_idx = args[4].parse::<usize>().expect("SEED_IDX must parse");
    let plane_idx = args[5].parse::<usize>().expect("PLANE_IDX must parse");
    let root = Fp2::new(
        args[6].parse::<u32>().expect("ROOT_C0 must parse"),
        args[7].parse::<u32>().expect("ROOT_C1 must parse"),
    );
    let x = args[8].parse::<u32>().expect("X must parse");
    let y = args[9].parse::<u32>().expect("Y must parse");
    let seed = scout_seed(rf, rp, seed_idx);
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let base = fixed_cofactor(seed);
    let (x_dirs, y_dirs) = plane_dirs(seed, plane_idx);
    let cofactor = concrete_plane_cofactor(&base, &x_dirs, &y_dirs, x, y);
    let coeffs = zerotest::root_first_coeffs(root, &cofactor);
    let p_hat = zerotest::flatten_coeffs(&coeffs);
    let residual = zerotest::zerotest_residual(&pos, &p_hat);
    println!("fixed-root affine-plane cofactor materialized witness");
    println!(
        "  rf={} rp={} seed_idx={} seed={} plane={} xy=({}, {}) root=({}, {})",
        rf, rp, seed_idx, seed, plane_idx, x, y, root.c0, root.c1
    );
    println!("  verify={}", zerotest::verify_zerotest(&pos, &p_hat));
    println!("  residual={:?}", residual);
    println!("  p_hat={:?}", p_hat);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("sample") {
        sample_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("profile-x") {
        profile_x_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("sample-range") {
        sample_range_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("solve-files") {
        solve_files_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("samples-status") {
        samples_status_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("plane-sample") {
        plane_sample_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("profile-plane-x") {
        profile_plane_x_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("plane-sample-range") {
        plane_sample_range_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("plane-solve-files") {
        plane_solve_files_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("shape") {
        shape_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("plane-shape") {
        plane_shape_case(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("emit") {
        emit_hit(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("emit-plane") {
        emit_plane_hit(&args);
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("plane-case") {
        let cases = plane_exact_case(&args);
        println!("Zero-test fixed-root / affine-plane cofactor scout");
        println!("  method: fix r, set G = G0 + X*U + Y*V, solve Res_Y(H((z-r)G)-r)");
        println!();

        let mut solved = 0usize;
        for (rf, rp, max_seeds, max_planes) in cases.iter().copied() {
            let hit = solve_plane_case(rf, rp, max_seeds, max_planes);
            if let Some(hit) = hit {
                solved += 1;
                println!(
                    "    accepted p_hat={:?} seed={} plane={} xy=({}, {}) root=({}, {}) elim_deg={} xroots={} elapsed={:.2}s",
                    hit.p_hat,
                    hit.seed,
                    hit.plane_idx,
                    hit.x,
                    hit.y,
                    hit.root.c0,
                    hit.root.c1,
                    hit.elim_degree,
                    hit.x_roots,
                    hit.elapsed
                );
            }
        }

        println!(
            "\n=== solved {}/{} configured plane cases ===",
            solved,
            cases.len()
        );
        std::process::exit(if solved == cases.len() { 0 } else { 1 });
    }
    let cases = if args.get(1).map(|s| s.as_str()) == Some("case") {
        if args.len() < 5 || args.len() > 6 {
            eprintln!("usage: zerotest_rootvar_scout case <RF> <RP> <MAX_SEEDS> [VAR_IDX]");
            std::process::exit(2);
        }
        let rf = args[2].parse::<usize>().expect("RF must parse");
        let rp = args[3].parse::<usize>().expect("RP must parse");
        let max_seeds = args[4].parse::<usize>().expect("MAX_SEEDS must parse");
        let var_idx = args
            .get(5)
            .map_or(0usize, |s| s.parse().expect("VAR_IDX must parse"));
        vec![(rf, rp, max_seeds, var_idx)]
    } else {
        vec![(2usize, 0usize, 8usize, 0usize), (2, 1, 8, 0), (2, 2, 8, 0)]
    };

    println!("Zero-test fixed-root / variable-cofactor scout");
    println!("  method: fix r, set G[var_idx] = X + Y*sqrt(3), solve Res_Y(H((z-r)G)-r)");
    println!();

    let mut solved = 0usize;
    for (rf, rp, max_seeds, var_idx) in cases.iter().copied() {
        let hit = solve_case(rf, rp, max_seeds, var_idx);
        if let Some(hit) = hit {
            solved += 1;
            println!(
                "    accepted p_hat={:?} seed={} var_g{}=({}, {}) root=({}, {}) elim_deg={} xroots={} elapsed={:.2}s",
                hit.p_hat,
                hit.seed,
                hit.var_idx,
                hit.x,
                hit.y,
                hit.root.c0,
                hit.root.c1,
                hit.elim_degree,
                hit.x_roots,
                hit.elapsed
            );
        }
    }

    println!(
        "\n=== solved {}/{} configured cases ===",
        solved,
        cases.len()
    );
    std::process::exit(if solved == cases.len() { 0 } else { 1 });
}
