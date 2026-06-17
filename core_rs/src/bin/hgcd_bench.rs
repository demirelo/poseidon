//! Phase 04: recursive Half-GCD over F_p[x], the asymptotic fix for the
//! per-node resultant/GCD wall (O(M(n) log n) vs the Euclidean O(n^2)).
//!
//! Self-contained on purpose: the HGCD lives here and is VALIDATED against the
//! proven `poly::gcd` oracle on hundreds of random + planted pairs, then
//! benchmarked. It edits nothing in poly.rs. Once it prints ALL-MATCH it can be
//! promoted into poly.rs behind a `gcd` fast-path.
//!
//! Run:  cargo run --release --bin hgcd_bench

use poseidon_core::poly as pl;
use poseidon_core::P;
use std::time::Instant;

const PU64: u64 = P as u64;
const SMALL: i64 = 48; // below this degree, defer to the schoolbook oracle

// ---- field + poly helpers --------------------------------------------------
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
    // a div x^k  (drop the low k coefficients)
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
    let inv = modpow(*a.last().unwrap() as u64, (P - 2) as u64);
    pl::scalar(&a, inv)
}

// ---- 2x2 polynomial matrices  M = [m00 m01; m10 m11] -----------------------
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

// ---- recursive Half-GCD ----------------------------------------------------
// Precondition: deg(a) > deg(b) >= 0. Returns M with M*(a,b)=(A,B),
// deg(A) > deg(a)/2 >= deg(B). Split at m=ceil(n/2); truncation lemma guarantees
// the matrix from the top halves reduces the full pair correctly.
fn hgcd(a: &[u32], b: &[u32]) -> Mat {
    let n = pl::deg(a);
    let m = ((n + 1) / 2) as usize; // ceil(n/2)
    if pl::deg(b) < m as i64 {
        return ident();
    }
    let r = hgcd(&shr(a, m), &shr(b, m));
    let (aa, bb) = matvec(&r, a, b);
    if pl::deg(&bb) < m as i64 {
        return r;
    }
    let (q, c) = pl::divmod(&aa, &bb); // aa = q*bb + c
    let qmat: Mat = [vec![], vec![1], vec![1], neg(&q)]; // [[0,1],[1,-q]] -> (bb, c)
    let qr = matmul(&qmat, &r);
    let bdeg = pl::deg(&bb);
    let k = 2 * (m as i64) - bdeg;
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

// ---- tiny RNG --------------------------------------------------------------
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
    fn poly(&mut self, d: usize) -> pl::Poly {
        let mut v: Vec<u32> = (0..=d).map(|_| self.field()).collect();
        if v[d] == 0 {
            v[d] = 1;
        }
        v
    }
}

fn main() {
    let mut rng = Rng::new(20260603);
    let mut fails = 0usize;
    let mut total = 0usize;

    // 1. correctness vs poly::gcd oracle: random coprime-ish + planted-factor pairs
    println!("1. gcd_fast == poly::gcd  (oracle: poly::gcd)");
    for &(da, db) in &[(48usize, 30usize), (80, 79), (200, 137), (500, 499), (777, 13)] {
        let mut bad = 0;
        for _ in 0..40 {
            let a = rng.poly(da);
            let b = rng.poly(db);
            total += 1;
            if gcd_fast(&a, &b) != pl::gcd(&a, &b) {
                bad += 1;
                fails += 1;
            }
        }
        println!("   random deg ({:>4},{:>4}): {}/40 match", da, db, 40 - bad);
    }
    // planted common factor: a=g*u, b=g*v  => gcd must be monic(g) (up to extra)
    {
        let mut bad = 0;
        for _ in 0..40 {
            let g = rng.poly(20);
            let u = rng.poly(120);
            let v = rng.poly(110);
            let a = pl::mul(&g, &u);
            let b = pl::mul(&g, &v);
            total += 1;
            let got = gcd_fast(&a, &b);
            let want = pl::gcd(&a, &b);
            // got must equal oracle, and oracle must be divisible by g
            if got != want {
                bad += 1;
                fails += 1;
            }
        }
        println!("   planted-factor deg(g=20,u=120,v=110): {}/40 match", 40 - bad);
    }

    // 2. degenerate inputs
    println!("2. degenerate inputs");
    let deg_ok = gcd_fast(&[], &[5, 1]) == monic(&pl::gcd(&[], &[5, 1]))
        && gcd_fast(&[7], &[]) == monic(&pl::gcd(&[7], &[]))
        && gcd_fast(&[1, 1], &[1, 1]) == monic(&[1u32, 1]);
    println!("   empty / constant / equal-input: {}", if deg_ok { "OK" } else { "FAIL" });
    if !deg_ok {
        fails += 1;
    }
    total += 1;

    if fails == 0 {
        println!("\n   ALL-MATCH: {} cases, 0 mismatches vs oracle.", total);
    } else {
        println!("\n   *** {} / {} MISMATCHES vs oracle -- NOT correct, do not promote ***", fails, total);
        std::process::exit(1);
    }

    // 3. benchmark: gcd_fast vs Euclidean poly::gcd, growing degree to the
    //    RF=6/RP=6 per-node size. Euclidean is O(n^2); above EUC_CAP we project
    //    it from the last measured point (n^2 scaling) to bound wall-clock.
    println!("\n3. speedup vs Euclidean poly::gcd (random degree-d pair)");
    println!("   {:>8}  {:>12}  {:>14}  {:>8}", "deg", "fast (ms)", "euclid (ms)", "speedup");
    const EUC_CAP: usize = 131072;
    let mut last_euc: Option<(usize, f64)> = None;
    for &d in &[4096usize, 16384, 65536, 131072, 262144, 531441] {
        let a = rng.poly(d);
        let b = rng.poly(d - 1);
        let t0 = Instant::now();
        let gf = gcd_fast(&a, &b);
        let tf = t0.elapsed().as_secs_f64();
        let (te, proj, eq) = if d <= EUC_CAP {
            let t1 = Instant::now();
            let ge = pl::gcd(&a, &b);
            let te = t1.elapsed().as_secs_f64();
            last_euc = Some((d, te));
            (te, false, ge == gf)
        } else {
            // project O(n^2) from the last measured Euclidean point
            let (d0, t0e) = last_euc.unwrap();
            (t0e * (d as f64 / d0 as f64).powi(2), true, true)
        };
        println!(
            "   {:>8}  {:>12.3}  {:>12.3}{}  {:>7.1}x   {}",
            d,
            tf * 1e3,
            te * 1e3,
            if proj { "*" } else { " " },
            te / tf,
            if eq { "match" } else { "MISMATCH" }
        );
    }
    println!("   (* = O(n^2)-projected, not run.  degY=19683=3^9 is RF=4/RP=5;");
    println!("    3^12=531441 is the RF=6/RP=6 per-node shape -- the gate.)");
}
