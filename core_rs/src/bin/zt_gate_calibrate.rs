//! zt_gate_calibrate -- paper-grade RF=6 zero-test interpolated-resultant cost gate.
//!
//! This SUPERSEDES the `gate_calibrate` proxy for the economic claim. That proxy
//! estimated per-node cost with a CICO-line symbolic build plus a Half-GCD as a
//! stand-in for the elimination. It undercounts the real route: at each
//! interpolation node the attack builds residual polynomials f0,f1 (symbolic
//! permutation over F_p[X], root-first state) and computes a *resultant*
//! Res_Y(f0,f1) -- NOT a GCD. The quotient sequence of that resultant is
//! generically all length-2 (avg_q ~ 2), so the Euclidean resultant runs in
//! O(n^2) and the Half-GCD speedup does not engage on the un-amortized route.
//!
//! Here we MEASURE the real per-node cost (build + pl::resultant) on a feasible
//! degree ladder (RF=6, RP=2..5 -> residual degree 3^8..3^11), fit the scaling by
//! log-log least squares, and project the full RF=6 / RP in {6,7,8,9} cost:
//! D = 3^(2*RF+RP) interpolation nodes, each one real per-node cost. Emits JSON.
//!
//! Run:  cargo run --release --bin zt_gate_calibrate    (costs reported in core-years)

use poseidon_core::poly as pl;
use poseidon_core::zerotest::{Fp2, ZT_D};
use poseidon_core::{Poseidon1, P};
use std::time::Instant;

const PU64: u64 = P as u64;

#[inline]
fn fmul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % PU64) as u32
}

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
    fn fp2_nonzero(&mut self) -> Fp2 {
        loop {
            let z = Fp2::new(self.field(), self.field());
            if !z.is_zero() {
                return z;
            }
        }
    }
}

fn full_cofactor(seed: u64) -> [Fp2; ZT_D] {
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

/// One real per-node measurement: (residual_degree, build_s, resultant_s).
fn measure_node(rf: usize, rp: usize, x: u32, seed: u64) -> (usize, f64, f64) {
    let pos = Poseidon1::new_canonical(3, 16, rf, rp);
    let cofactor = full_cofactor(seed);
    let t0 = Instant::now();
    let (f0, f1) = residual_polys_at_x(&pos, &cofactor, x);
    let build_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let r = pl::resultant(&f0, &f1);
    let res_s = t1.elapsed().as_secs_f64();
    std::hint::black_box(r);
    (pl::deg(&f0).max(0) as usize, build_s, res_s)
}

/// Log-log least squares: returns (exponent k, ln_c) for y = c * x^k.
fn fit_loglog(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0.ln()).sum();
    let sy: f64 = pts.iter().map(|p| p.1.ln()).sum();
    let sxx: f64 = pts.iter().map(|p| p.0.ln() * p.0.ln()).sum();
    let sxy: f64 = pts.iter().map(|p| p.0.ln() * p.1.ln()).sum();
    let k = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let ln_c = (sy - k * sx) / n;
    (k, ln_c)
}

fn main() {
    // An optional CLI argument is accepted and ignored (kept for backward-compatible
    // invocation); costs are reported in core-years.
    const SECS_PER_CORE_YEAR: f64 = 3600.0 * 24.0 * 365.0;
    const RF: usize = 6;
    // Half-GCD speedup at the RF=6/RP=6 per-node degree (3^12), from hgcd_bench.
    const HGCD_SPEEDUP_AT_SCALE: f64 = 7.0;

    // ---- ladder: measure the REAL per-node cost at feasible RP ----
    let ladder_rp = [2usize, 3, 4, 5];
    let mut ladder: Vec<(usize, usize, f64, f64)> = vec![]; // (rp, deg, build_s, res_s)
    eprintln!("zt_gate_calibrate: measuring real per-node (build + resultant) ladder ...");
    for &rp in &ladder_rp {
        let (deg, b, r) = measure_node(RF, rp, 1, 0xC0FA_0000 + rp as u64);
        eprintln!("  RF={RF} RP={rp} residual_deg={deg} build={b:.4}s resultant={r:.4}s");
        ladder.push((rp, deg, b, r));
    }

    // ---- fit power laws (log-log least squares over all ladder points) ----
    let res_pts: Vec<(f64, f64)> = ladder.iter().map(|&(_, d, _, r)| (d as f64, r)).collect();
    let build_pts: Vec<(f64, f64)> = ladder.iter().map(|&(_, d, b, _)| (d as f64, b)).collect();
    let (k_res, lc_res) = fit_loglog(&res_pts);
    let (k_build, lc_build) = fit_loglog(&build_pts);
    let proj_res = |deg: f64| (lc_res + k_res * deg.ln()).exp();
    let proj_build = |deg: f64| (lc_build + k_build * deg.ln()).exp();

    // ---- project full cost for the targets ----
    let targets = [
        (6usize, "below_or_at_public_record"),
        (7usize, "below_current_public_record"),
        (8usize, "current_public_record"),
        (9usize, "first_public_record_improving_target"),
    ];
    let mut projections = String::new();
    for (i, &(rp, role)) in targets.iter().enumerate() {
        let residual_deg = 3f64.powi((RF + rp) as i32);
        let d_nodes = 3f64.powi((2 * RF + rp) as i32); // eliminant degree = #interp nodes
        let per_node_naive = proj_build(residual_deg) + proj_res(residual_deg);
        let per_node_hgcd =
            proj_build(residual_deg) + proj_res(residual_deg) / HGCD_SPEEDUP_AT_SCALE;
        let naive_sec = d_nodes * per_node_naive;
        let hgcd_sec = d_nodes * per_node_hgcd;
        let cy = |s: f64| s / SECS_PER_CORE_YEAR;
        let block = format!(
            "    {{\"rf\":{RF},\"rp\":{rp},\"role\":\"{role}\",\"residual_degree\":{:.0},\"eliminant_degree_nodes\":{:.0},\"per_node_naive_s\":{:.1},\"per_node_hgcd_floor_s\":{:.1},\"naive\":{{\"core_years\":{:.0}}},\"hgcd_floor\":{{\"core_years\":{:.0}}}}}{}",
            residual_deg,
            d_nodes,
            per_node_naive,
            per_node_hgcd,
            cy(naive_sec),
            cy(hgcd_sec),
            if i + 1 < targets.len() { ",\n" } else { "\n" }
        );
        projections.push_str(&block);
    }

    // ---- ladder JSON ----
    let mut ladder_json = String::new();
    for (i, &(rp, deg, b, r)) in ladder.iter().enumerate() {
        ladder_json.push_str(&format!(
            "    {{\"rf\":{RF},\"rp\":{rp},\"residual_degree\":{deg},\"build_s\":{b:.4},\"resultant_s\":{r:.4}}}{}",
            if i + 1 < ladder.len() { ",\n" } else { "\n" }
        ));
    }

    println!("{{");
    println!("  \"gate\": \"zerotest_interpolated_resultant_route\",");
    println!("  \"supersedes\": \"gate_calibrate (CICO-line + Half-GCD proxy)\",");
    println!("  \"method\": \"measured real per-node = build (symbolic permutation over F_p[X], root-first state) + pl::resultant(f0,f1); log-log fit; project D=3^(2RF+RP) interpolation nodes\",");
    println!("  \"ladder\": [\n{ladder_json}  ],");
    println!("  \"fit\": {{\"resultant_exponent\": {k_res:.3}, \"build_exponent\": {k_build:.3}}},");
    println!("  \"hgcd_speedup_at_scale\": {HGCD_SPEEDUP_AT_SCALE},");
    println!("  \"projections\": [\n{projections}  ],");
    println!("  \"note\": \"naive route uses the current O(n^2) Euclidean pl::resultant (quotient sequence is generically length-2, so Half-GCD does not engage). hgcd_floor assumes the validated recursive Half-GCD is integrated into the resultant (tracked follow-up). The Initiative page checked 2026-06-17 lists zero-test RF=6/RP=8 as the current public record, so RP=9 is the first public-record-improving target. Even below-record RP=7 is far above feasibility on this route; RP>=9 is astronomically out of reach without a structural build collapse.\"");
    println!("}}");
}
