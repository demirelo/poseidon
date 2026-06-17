//! RF=6 go/no-go cost gate for zero-test and CICO routes.
//!
//! NOTE (proxy): this estimates per-node cost with a CICO-line symbolic build plus
//! a Half-GCD stand-in for the elimination. The PAPER-GRADE economic gate is
//! `zt_gate_calibrate`, which measures the REAL interpolated-resultant route
//! (residual build + pl::resultant) and writes
//! artifacts/zerotest_rf6_resultant_gate.json. The proxy below undercounts the real
//! per-node cost by ~6x (directly-measured RF=6/RP=6 node: 116.7s vs ~20s here).
//!
//! The resultant route cannot reach RF=6 (eliminant degree 3^(2*6+RP) >= 3^18
//! interpolation nodes -- infeasible). The GCD route CAN: one "line" is a single
//! symbolic permutation over F_p[X] (NTT-backed) + one polynomial GCD. This bin
//! MEASURES that per-line cost at RF=6 and reports it as a per-node core-year
//! projection, so the go/no-go is data, not a guess.
//!
//! The per-line build uses `cico::perm_plus_linear_poly` as a faithful COST proxy
//! for the zero-test per-line hash: same permutation, same degree growth
//! 3^(RF+RP); the zero-test predicate differs only by the (negligible-cost)
//! compression feedforward. GCD uses the Half-GCD validated in `hgcd_bench`
//! (copied here; promote to poly.rs once tuned). NTT caps single-mul degree at
//! ~2^24, so RF=6 single-NTT lines work up to RP=8; CICO RP=10 (fwd 3^16, cube
//! ~3^16*3 > 2^24) needs the Kronecker/CRT path and is out of scope here.
//!
//! Run:  cargo run --release --bin gate_calibrate

use poseidon_core::cico;
use poseidon_core::poly as pl;
use poseidon_core::{Poseidon1, CICO_C1, CICO_C2, P};
use std::time::Instant;

const PU64: u64 = P as u64;

// ---- validated Half-GCD (copy of hgcd_bench's, 241/241 vs poly::gcd oracle) ----
const SMALL: i64 = 48;
fn modpow(mut a: u64, mut e: u64) -> u32 {
    a %= PU64;
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = r * a % PU64;
        }
        a = a * a % PU64;
        e >>= 1;
    }
    r as u32
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
fn monic(a: &[u32]) -> pl::Poly {
    let a = pl::trim(a);
    if a.is_empty() {
        return a;
    }
    pl::scalar(&a, modpow(*a.last().unwrap() as u64, (P - 2) as u64))
}
type Mat = [pl::Poly; 4];
fn ident() -> Mat {
    [vec![1], vec![], vec![], vec![1]]
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
fn hgcd(a: &[u32], b: &[u32]) -> Mat {
    let n = pl::deg(a);
    let m = ((n + 1) / 2) as usize;
    if pl::deg(b) < m as i64 {
        return ident();
    }
    let r = hgcd(&shr(a, m), &shr(b, m));
    let (aa, bb) = matvec(&r, a, b);
    if pl::deg(&bb) < m as i64 {
        return r;
    }
    let (q, c) = pl::divmod(&aa, &bb);
    let qmat: Mat = [vec![], vec![1], vec![1], neg(&q)];
    let qr = matmul(&qmat, &r);
    let k = 2 * (m as i64) - pl::deg(&bb);
    if k < 0 || pl::deg(&c) < 0 {
        return qr;
    }
    let s = hgcd(&shr(&bb, k as usize), &shr(&c, k as usize));
    matmul(&s, &qr)
}
fn gcd_fast(a_in: &[u32], b_in: &[u32]) -> pl::Poly {
    let mut a = pl::trim(a_in);
    let mut b = pl::trim(b_in);
    if pl::deg(&a) < pl::deg(&b) {
        std::mem::swap(&mut a, &mut b);
    }
    loop {
        if pl::deg(&b) < 0 {
            break;
        }
        if pl::deg(&a) == pl::deg(&b) {
            let r = pl::rem(&a, &b);
            a = b;
            b = r;
            continue;
        }
        if pl::deg(&a) < SMALL {
            return monic(&pl::gcd(&a, &b));
        }
        let mm = hgcd(&a, &b);
        let (aa, bb) = matvec(&mm, &a, &b);
        a = pl::trim(&aa);
        b = pl::trim(&bb);
        if pl::deg(&b) >= 0 {
            let r = pl::rem(&a, &b);
            a = b;
            b = r;
        }
    }
    monic(&a)
}

// ---- RNG ----
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
}

fn main() {
    println!("RF=6 GCD-route per-line cost gate (NTT build + Half-GCD)\n");
    println!(
        "  {:>3} {:>10} {:>11} {:>11} {:>11}",
        "RP", "fwd_deg", "build(s)", "gcd(s)", "line(s)"
    );
    let mut rng = Rng::new(0x6A7E);
    let mut results: Vec<(usize, f64)> = vec![];
    for rp in [5usize, 6, 7] {
        let pos = Poseidon1::new_canonical(3, 16, 6, rp);
        let assign: Vec<pl::Poly> = (0..14).map(|_| vec![rng.field(), rng.field_nz()]).collect();
        let mut st: Vec<pl::Poly> = vec![vec![CICO_C1 % P], vec![CICO_C2 % P]];
        st.extend(assign);

        let t0 = Instant::now();
        let s = cico::perm_plus_linear_poly(&pos, &st);
        let tb = t0.elapsed().as_secs_f64();

        let f0 = pl::trim(&s[0]);
        let f1 = pl::trim(&s[1]);
        let t1 = Instant::now();
        let g = gcd_fast(&f0, &f1);
        let tg = t1.elapsed().as_secs_f64();
        std::hint::black_box(&g);

        let line = tb + tg;
        println!(
            "  {:>3} {:>10} {:>11.3} {:>11.3} {:>11.3}",
            rp,
            pl::deg(&s[0]),
            tb,
            tg,
            line
        );
        results.push((rp, line));
    }

    // The GCD route's per-line hit-rate is provably ~1/p (two equations in one
    // variable = codim-2), so it is NOT the favorable path. The resultant route is.
    let cy = |secs: f64| secs / (3600.0 * 24.0 * 365.0);
    let node = results.iter().find(|(rp, _)| *rp == 6).map(|(_, t)| *t).unwrap_or(20.0);
    let d18 = 3f64.powi(18);
    let d19 = 3f64.powi(19);
    let res_rp6_cy = cy(d18 * node);
    let res_rp7_cy = cy(d19 * results.iter().find(|(rp,_)| *rp==7).map(|(_,t)| *t).unwrap_or(52.6));
    println!("\nGo/no-go projection (resultant route is the real path; GCD line is ~1/p codim-2):");
    println!("  GCD route:  ~p (2.1e9) lines x per-line  =>  ~1.3e3 core-years (worse path).");
    println!("  Resultant:  D=3^(2*6+RP) nodes x ~per-node-perm(~{:.0}s):", node);
    println!("    RP=6 (ties record): 3^18 nodes -> {:.0} core-years", res_rp6_cy);
    println!("    RP=7 (below current zero-test record): 3^19 nodes -> {:.0} core-years", res_rp7_cy);
    println!("  => public-record-improving RP>=9 needs a structural build collapse, not a naive cluster run.");
    println!("  CICO RP=10 (fwd 3^16) exceeds the 2^24 NTT cap: needs Kronecker/CRT mul first.");
}
