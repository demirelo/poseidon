//! Cofactor-family scout for the zero-test root-first attack.
//!
//! The RF=2/RP=4 frontier showed the same eliminant degree for several random
//! cofactors, but only some cofactors produced Fp roots. This binary compares
//! simple cofactor families to see whether degree or rational-root density can
//! be biased before we invest in heavier interpolation.
//!
//! Run:
//!   cargo run --release --bin zerotest_cofactor_scout
//!   cargo run --release --bin zerotest_cofactor_scout 2 4 2
//!   cargo run --release --bin zerotest_cofactor_scout 2 4 2 full,real,monic,top,edges
//!   cargo run --release --bin zerotest_cofactor_scout 2 4 2 full,real,edges 2
//!   cargo run --release --bin zerotest_cofactor_scout 4 2 1 full,edges 0 512
//!   cargo run --release --bin zerotest_cofactor_scout 2 8 1 full,edges 0 1 8
//!   cargo run --release --bin zerotest_cofactor_scout 2 8 1 full,edges 0 1 8 100000
//!   cargo run --release --bin zerotest_cofactor_scout 4 4 16 full,edges 0 1 0
//!
//! The per-seed table includes phase timings so higher-RF/RP probes can decide
//! whether interpolation, resultant sampling, root finding, or verification is
//! the next bottleneck.

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

#[derive(Copy, Clone)]
enum Family {
    Full,
    Real,
    Imag,
    Monic,
    Constant,
    Linear,
    Top,
    Edges,
    Geometric,
    Ones,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Full => "full",
            Family::Real => "real",
            Family::Imag => "imag",
            Family::Monic => "monic",
            Family::Constant => "constant",
            Family::Linear => "linear",
            Family::Top => "top",
            Family::Edges => "edges",
            Family::Geometric => "geometric",
            Family::Ones => "ones",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Family::Full),
            "real" => Some(Family::Real),
            "imag" => Some(Family::Imag),
            "monic" => Some(Family::Monic),
            "constant" => Some(Family::Constant),
            "linear" => Some(Family::Linear),
            "top" => Some(Family::Top),
            "edges" => Some(Family::Edges),
            "geometric" => Some(Family::Geometric),
            "ones" => Some(Family::Ones),
            _ => None,
        }
    }
}

fn cofactor(family: Family, seed: u64) -> [Fp2; ZT_D] {
    let mut rng = Rng::new(seed);
    let mut g = [Fp2::ZERO; ZT_D];
    match family {
        Family::Full => {
            for slot in &mut g {
                *slot = rng.fp2_nonzero();
            }
        }
        Family::Real => {
            for slot in &mut g {
                *slot = rng.fp2_real_nonzero();
            }
        }
        Family::Imag => {
            for slot in &mut g {
                *slot = rng.fp2_imag_nonzero();
            }
        }
        Family::Monic => {
            for slot in g.iter_mut().take(ZT_D - 1) {
                *slot = rng.fp2_nonzero();
            }
            g[ZT_D - 1] = Fp2::ONE;
        }
        Family::Constant => {
            g[0] = rng.fp2_nonzero();
        }
        Family::Linear => {
            g[0] = rng.fp2_nonzero();
            g[1] = rng.fp2_nonzero();
        }
        Family::Top => {
            g[ZT_D - 1] = rng.fp2_nonzero();
        }
        Family::Edges => {
            g[0] = rng.fp2_nonzero();
            g[ZT_D - 1] = rng.fp2_nonzero();
        }
        Family::Geometric => {
            let a = rng.fp2_nonzero();
            let q = rng.fp2_nonzero();
            let mut cur = a;
            for slot in &mut g {
                *slot = cur;
                cur = cur.mul(q);
            }
        }
        Family::Ones => {
            g.fill(Fp2::ONE);
        }
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

fn interpolate_linear(xs: &[u32], ys: &[u32]) -> pl::Poly {
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

struct Probe {
    degree: i64,
    x_roots: usize,
    accepted: usize,
    stable: bool,
    shape: PolyShape,
    first_root: Option<Fp2>,
    first_p_hat: Option<[u32; 16]>,
    sample_sec: f64,
    interp_sec: f64,
    guard_sec: f64,
    roots_sec: f64,
    backsub_sec: f64,
    elapsed: f64,
}

#[derive(Clone, Debug)]
struct PolyShape {
    nonzero: usize,
    density: f64,
    valuation: usize,
    support_gcd: usize,
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn poly_shape(f: &[u32]) -> PolyShape {
    let degree = pl::deg(f);
    if degree < 0 {
        return PolyShape {
            nonzero: 0,
            density: 0.0,
            valuation: 0,
            support_gcd: 0,
        };
    }

    let mut nonzero = 0usize;
    let mut first = None;
    let mut support_gcd = 0usize;
    for (i, &c) in f.iter().enumerate() {
        if c == 0 {
            continue;
        }
        nonzero += 1;
        match first {
            Some(base) => {
                support_gcd = gcd_usize(support_gcd, i - base);
            }
            None => {
                first = Some(i);
            }
        }
    }

    PolyShape {
        nonzero,
        density: nonzero as f64 / (degree as usize + 1) as f64,
        valuation: first.unwrap_or(0),
        support_gcd,
    }
}

struct SampleProbe {
    nodes: usize,
    deg_y0: i64,
    deg_y1: i64,
    zero_resultants: usize,
    node_profiles: Vec<NodeProfile>,
    sample_sec: f64,
    ms_per_node: f64,
    projected_full_sample_sec: f64,
}

struct NodeProfile {
    x: u32,
    resultant: u32,
    build_sec: f64,
    resultant_sec: f64,
    steps: usize,
    q_len_1: usize,
    q_len_2: usize,
    q_len_3: usize,
    q_len_2_3: usize,
    q_len_4_15: usize,
    q_len_16_255: usize,
    q_len_fast: usize,
    max_q_len: usize,
    avg_q_len: f64,
    jump_events: Vec<JumpEvent>,
    tail_profile: Option<TailProfile>,
    zero_remainder: bool,
}

struct TailProfile {
    peeled_steps: usize,
    prefix_sec: f64,
    tail_sec: f64,
    tail_resultant: u32,
    tail_deg_a: i64,
    tail_deg_b: i64,
    tail_steps: usize,
    tail_q_len_2: usize,
    tail_jump_events: usize,
}

struct JumpEvent {
    step: usize,
    q_len: usize,
    deg_a: i64,
    deg_b: i64,
    deg_r: i64,
    degree_drop: i64,
}

fn format_jumps(events: &[JumpEvent]) -> String {
    if events.is_empty() {
        return "[]".to_string();
    }
    let parts = events
        .iter()
        .map(|e| {
            format!(
                "{}:q{}:{}>{}>{}:d{}",
                e.step, e.q_len, e.deg_a, e.deg_b, e.deg_r, e.degree_drop
            )
        })
        .collect::<Vec<_>>();
    format!("[{}]", parts.join(","))
}

fn profile_q2_tail(f0: &[u32], f1: &[u32]) -> Option<TailProfile> {
    let mut a = pl::trim(f0);
    let mut b = pl::trim(f1);
    if a.is_empty() || b.is_empty() {
        return None;
    }
    if pl::deg(&a) < pl::deg(&b) {
        std::mem::swap(&mut a, &mut b);
    }

    let prefix_start = Instant::now();
    let mut peeled_steps = 0usize;
    while pl::deg(&b) > 0 {
        let q_len = a.len().saturating_sub(b.len()) + 1;
        if q_len == 2 {
            break;
        }
        let r = pl::rem(&a, &b);
        if r.is_empty() {
            return None;
        }
        a = b;
        b = r;
        peeled_steps += 1;
    }
    let prefix_sec = prefix_start.elapsed().as_secs_f64();
    let tail_deg_a = pl::deg(&a);
    let tail_deg_b = pl::deg(&b);

    let tail_start = Instant::now();
    let (tail_resultant, tail_stats) = pl::resultant_profiled(&a, &b);
    let tail_sec = tail_start.elapsed().as_secs_f64();
    Some(TailProfile {
        peeled_steps,
        prefix_sec,
        tail_sec,
        tail_resultant,
        tail_deg_a,
        tail_deg_b,
        tail_steps: tail_stats.steps,
        tail_q_len_2: tail_stats.q_len_2,
        tail_jump_events: tail_stats.jump_events.len(),
    })
}

fn profile_node(pos: &Poseidon1, cofactor: &[Fp2; ZT_D], x: u32) -> (i64, i64, NodeProfile) {
    let build_start = Instant::now();
    let (f0, f1) = residual_polys_at_x(pos, cofactor, x);
    let build_sec = build_start.elapsed().as_secs_f64();
    let deg_y0 = pl::deg(&f0);
    let deg_y1 = pl::deg(&f1);
    let tail_profile = profile_q2_tail(&f0, &f1);
    let resultant_start = Instant::now();
    let (resultant, stats) = pl::resultant_profiled(&f0, &f1);
    let resultant_sec = resultant_start.elapsed().as_secs_f64();
    let avg_q_len = if stats.steps == 0 {
        0.0
    } else {
        stats.total_q_len as f64 / stats.steps as f64
    };
    (
        deg_y0,
        deg_y1,
        NodeProfile {
            x,
            resultant,
            build_sec,
            resultant_sec,
            steps: stats.steps,
            q_len_1: stats.q_len_1,
            q_len_2: stats.q_len_2,
            q_len_3: stats.q_len_3,
            q_len_2_3: stats.q_len_2_3,
            q_len_4_15: stats.q_len_4_15,
            q_len_16_255: stats.q_len_16_255,
            q_len_fast: stats.q_len_fast,
            max_q_len: stats.max_q_len,
            avg_q_len,
            jump_events: stats
                .jump_events
                .iter()
                .map(|e| JumpEvent {
                    step: e.step,
                    q_len: e.q_len,
                    deg_a: e.deg_a,
                    deg_b: e.deg_b,
                    deg_r: e.deg_r,
                    degree_drop: e.degree_drop,
                })
                .collect(),
            tail_profile,
            zero_remainder: stats.zero_remainder,
        },
    )
}

fn sample_probe(
    pos: &Poseidon1,
    cofactor: &[Fp2; ZT_D],
    sample_nodes: usize,
    full_nodes: usize,
    profile_nodes: usize,
    profile_start: usize,
) -> SampleProbe {
    let nodes = sample_nodes.max(1).min((P - 1) as usize);
    let profile_start = profile_start.max(1).min((P - 1) as usize);
    let mut deg_y0 = -1;
    let mut deg_y1 = -1;
    let mut node_profiles = Vec::with_capacity(profile_nodes);
    if profile_nodes == 0 {
        let (f0, f1) = residual_polys_at_x(pos, cofactor, profile_start as u32);
        deg_y0 = pl::deg(&f0);
        deg_y1 = pl::deg(&f1);
    } else {
        let profile_nodes = profile_nodes.min((P - 1) as usize - profile_start + 1);
        for offset in 0..profile_nodes {
            let x = (profile_start + offset) as u32;
            let (d0, d1, profile) = profile_node(pos, cofactor, x);
            if offset == 0 {
                deg_y0 = d0;
                deg_y1 = d1;
            }
            node_profiles.push(profile);
        }
    }
    let xs = (1..=nodes as u32).collect::<Vec<_>>();
    let start = Instant::now();
    let resultants = sample_resultants(pos, cofactor, &xs);
    let zero_resultants = resultants.iter().filter(|&&y| y == 0).count();
    let sample_sec = start.elapsed().as_secs_f64();
    let ms_per_node = 1000.0 * sample_sec / nodes as f64;
    let projected_full_sample_sec = sample_sec * full_nodes as f64 / nodes as f64;
    SampleProbe {
        nodes,
        deg_y0,
        deg_y1,
        zero_resultants,
        node_profiles,
        sample_sec,
        ms_per_node,
        projected_full_sample_sec,
    }
}

fn probe(pos: &Poseidon1, cofactor: &[Fp2; ZT_D], degree: usize) -> Probe {
    let start = Instant::now();
    let n = degree + 1;
    let guard = 4usize;
    let xs: Vec<u32> = (1..=n as u32).collect();
    let sample_start = Instant::now();
    let ys = sample_resultants(pos, cofactor, &xs);
    let sample_sec = sample_start.elapsed().as_secs_f64();
    let interp_start = Instant::now();
    let r_poly = interpolate_linear(&xs, &ys);
    let shape = poly_shape(&r_poly);
    let interp_sec = interp_start.elapsed().as_secs_f64();
    let guard_start = Instant::now();
    let stable = (0..guard).all(|g| {
        let x = (n + 1 + g) as u32;
        let (f0, f1) = residual_polys_at_x(pos, cofactor, x);
        pl::eval(&r_poly, x) == pl::resultant(&f0, &f1)
    });
    let guard_sec = guard_start.elapsed().as_secs_f64();
    let roots_start = Instant::now();
    let x_roots = pl::roots(&r_poly);
    let roots_sec = roots_start.elapsed().as_secs_f64();
    let mut accepted = 0usize;
    let mut first_root = None;
    let mut first_p_hat = None;
    let backsub_start = Instant::now();
    for x in x_roots.iter().copied() {
        let (f0, f1) = residual_polys_at_x(pos, cofactor, x);
        let g = pl::gcd(&f0, &f1);
        for y in pl::roots(&g) {
            let root = Fp2::new(x, y);
            let coeffs = zerotest::root_first_coeffs(root, cofactor);
            let p_hat = zerotest::flatten_coeffs(&coeffs);
            if zerotest::verify_zerotest(pos, &p_hat) {
                accepted += 1;
                if first_root.is_none() {
                    first_root = Some(root);
                    first_p_hat = Some(p_hat);
                }
            }
        }
    }
    let backsub_sec = backsub_start.elapsed().as_secs_f64();
    Probe {
        degree: pl::deg(&r_poly),
        x_roots: x_roots.len(),
        accepted,
        stable,
        shape,
        first_root,
        first_p_hat,
        sample_sec,
        interp_sec,
        guard_sec,
        roots_sec,
        backsub_sec,
        elapsed: start.elapsed().as_secs_f64(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rf = args.get(1).map_or(2, |s| s.parse().expect("RF must parse"));
    let rp = args.get(2).map_or(3, |s| s.parse().expect("RP must parse"));
    let seeds = args
        .get(3)
        .map_or(4, |s| s.parse().expect("seeds must parse"));
    let seed_offset = args
        .get(5)
        .map_or(0usize, |s| s.parse().expect("seed offset must parse"));
    let sample_only_nodes = args
        .get(6)
        .map(|s| s.parse::<usize>().expect("sample-only nodes must parse"));
    let profile_nodes = args
        .get(7)
        .map_or(1usize, |s| s.parse().expect("profile nodes must parse"));
    let profile_start = args
        .get(8)
        .map_or(1usize, |s| s.parse().expect("profile start must parse"));
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let degree = expected_degree(rf, rp);
    let full_nodes = degree + 1;
    let families: Vec<Family> = match args.get(4) {
        Some(list) => list
            .split(',')
            .map(|name| Family::parse(name).unwrap_or_else(|| panic!("unknown family: {name}")))
            .collect(),
        None => vec![
            Family::Full,
            Family::Real,
            Family::Imag,
            Family::Monic,
            Family::Constant,
            Family::Linear,
            Family::Top,
            Family::Edges,
            Family::Geometric,
            Family::Ones,
        ],
    };

    if let Some(sample_nodes) = sample_only_nodes {
        println!(
            "Zero-test sample-only profile: RF={} RP={} expected_degree={} full_nodes={} sample_nodes={} seeds/family={} seed_offset={} families={}",
            rf,
            rp,
            degree,
            full_nodes,
            sample_nodes,
            seeds,
            seed_offset,
            families.iter().map(|f| f.name()).collect::<Vec<_>>().join(",")
        );
        println!(
            "  {:<10} {:>4} {:>7} {:>7} {:>7} {:>7} {:>9} {:>10} {:>11}",
            "family", "seed", "nodes", "degY0", "degY1", "zeros", "sample", "ms/node", "proj_full"
        );
        println!("  {}", "-".repeat(86));
        for family in families {
            for local_idx in 0..seeds {
                let seed_idx = seed_offset + local_idx;
                let seed = 0xC0FA_0000
                    + ((rf as u64) << 24)
                    + ((rp as u64) << 16)
                    + ((family as u64) << 8)
                    + seed_idx as u64;
                let g = cofactor(family, seed);
                let p = sample_probe(
                    &pos,
                    &g,
                    sample_nodes,
                    full_nodes,
                    profile_nodes,
                    profile_start,
                );
                println!(
                    "  {:<10} {:>4} {:>7} {:>7} {:>7} {:>7} {:>9.2} {:>10.3} {:>11.2}",
                    family.name(),
                    seed_idx,
                    p.nodes,
                    p.deg_y0,
                    p.deg_y1,
                    p.zero_resultants,
                    p.sample_sec,
                    p.ms_per_node,
                    p.projected_full_sample_sec
                );
                for profile in &p.node_profiles {
                    println!(
                        "    node x={} build={:.4}s resultant={:.4}s res={} steps={} q1={} q2={} q3={} q2_3={} q4_15={} q16_255={} qfast={} max_q={} avg_q={:.3} zero_rem={} jumps={}",
                        profile.x,
                        profile.build_sec,
                        profile.resultant_sec,
                        profile.resultant,
                        profile.steps,
                        profile.q_len_1,
                        profile.q_len_2,
                        profile.q_len_3,
                        profile.q_len_2_3,
                        profile.q_len_4_15,
                        profile.q_len_16_255,
                        profile.q_len_fast,
                        profile.max_q_len,
                        profile.avg_q_len,
                        profile.zero_remainder,
                        format_jumps(&profile.jump_events)
                    );
                    if let Some(tail) = &profile.tail_profile {
                        println!(
                            "      tail peeled={} prefix={:.4}s tail={:.4}s tail_res={} tail_deg=({}, {}) tail_steps={} tail_q2={} tail_jumps={}",
                            tail.peeled_steps,
                            tail.prefix_sec,
                            tail.tail_sec,
                            tail.tail_resultant,
                            tail.tail_deg_a,
                            tail.tail_deg_b,
                            tail.tail_steps,
                            tail.tail_q_len_2,
                            tail.tail_jump_events
                        );
                    }
                }
            }
        }
        return;
    }

    println!(
        "Zero-test cofactor-family scout: RF={} RP={} expected_degree={} seeds/family={} seed_offset={} families={}",
        rf,
        rp,
        degree,
        seeds,
        seed_offset,
        families.iter().map(|f| f.name()).collect::<Vec<_>>().join(",")
    );
    println!(
        "  {:<10} {:>4} {:>8} {:>8} {:>8} {:>6} {:>8} {:>8} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "family",
        "seed",
        "degree",
        "xroots",
        "accept",
        "stable",
        "nz",
        "density",
        "sgcd",
        "val",
        "sample",
        "interp",
        "guard",
        "roots",
        "backsub",
        "total"
    );
    println!("  {}", "-".repeat(136));

    for family in families {
        let mut total_roots = 0usize;
        let mut total_accept = 0usize;
        let mut min_deg = i64::MAX;
        let mut max_deg = i64::MIN;
        for local_idx in 0..seeds {
            let seed_idx = seed_offset + local_idx;
            let seed = 0xC0FA_0000
                + ((rf as u64) << 24)
                + ((rp as u64) << 16)
                + ((family as u64) << 8)
                + seed_idx as u64;
            let g = cofactor(family, seed);
            let p = probe(&pos, &g, degree);
            total_roots += p.x_roots;
            total_accept += p.accepted;
            min_deg = min_deg.min(p.degree);
            max_deg = max_deg.max(p.degree);
            println!(
                "  {:<10} {:>4} {:>8} {:>8} {:>8} {:>6} {:>8} {:>8.4} {:>6} {:>6} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                family.name(),
                seed_idx,
                p.degree,
                p.x_roots,
                p.accepted,
                p.stable,
                p.shape.nonzero,
                p.shape.density,
                p.shape.support_gcd,
                p.shape.valuation,
                p.sample_sec,
                p.interp_sec,
                p.guard_sec,
                p.roots_sec,
                p.backsub_sec,
                p.elapsed
            );
            if let (Some(root), Some(p_hat)) = (p.first_root, p.first_p_hat) {
                println!(
                    "    hit {:<10} seed={} root=({}, {}) p_hat={:?}",
                    family.name(),
                    seed_idx,
                    root.c0,
                    root.c1,
                    p_hat
                );
            }
        }
        println!(
            "  {:<10}  SUM deg=[{},{}] roots={} accepted={}",
            family.name(),
            min_deg,
            max_deg,
            total_roots,
            total_accept
        );
    }
}
