//! Validation for the Rust `poly` primitives. Mirrors tests/test_poly.py using
//! the SAME independent oracles (so no sympy needed in Rust):
//!   * mul       -> evaluation-consistency at random points
//!   * divmod    -> the defining identity a == q*b + r, deg(r) < deg(b)
//!   * gcd       -> divides both, monic, and the cofactors are coprime (maximality)
//!   * resultant -> equals the Sylvester determinant (a different algorithm),
//!                  and is 0 iff the inputs share a factor
//!   * roots     -> finds all planted roots, returns no spurious ones
//!   * interp    -> round-trips a known polynomial
//! Plus a few hardcoded vectors matching the sympy-validated Python results.
//!
//! Run:  cargo run --release --bin poly_golden

use poseidon_core::poly::*;
use poseidon_core::P;

const PU64: u64 = P as u64;

fn feval_mul(a: &[u32], b: &[u32], t: u32) -> u32 {
    ((eval(a, t) as u64 * eval(b, t) as u64) % PU64) as u32
}

/// Independent Sylvester-matrix resultant (textbook definition), via Gaussian
/// elimination over F_p. Deliberately a different algorithm from `resultant`.
fn sylvester_res(a_in: &[u32], b_in: &[u32]) -> u32 {
    let a = trim(a_in);
    let b = trim(b_in);
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let m = a.len() - 1;
    let n = b.len() - 1;
    let nn = m + n;
    if nn == 0 {
        return 1;
    }
    let ah: Vec<u32> = a.iter().rev().cloned().collect(); // high -> low
    let bh: Vec<u32> = b.iter().rev().cloned().collect();
    let mut mat = vec![vec![0u32; nn]; nn];
    for i in 0..n {
        for (j, &c) in ah.iter().enumerate() {
            mat[i][i + j] = c % P;
        }
    }
    for i in 0..m {
        for (j, &c) in bh.iter().enumerate() {
            mat[n + i][i + j] = c % P;
        }
    }
    let mut det: u64 = 1;
    for col in 0..nn {
        let piv = (col..nn).find(|&r| mat[r][col] % P != 0);
        let piv = match piv {
            Some(p) => p,
            None => return 0,
        };
        if piv != col {
            mat.swap(col, piv);
            det = (PU64 - det % PU64) % PU64; // negate
        }
        det = det * mat[col][col] as u64 % PU64;
        let inv = inv_fp(mat[col][col]);
        for r in (col + 1)..nn {
            let f = (mat[r][col] as u64 * inv as u64 % PU64) as u32;
            if f != 0 {
                for k in 0..nn {
                    let t =
                        (mat[r][k] as u64 + PU64 - (f as u64 * mat[col][k] as u64 % PU64)) % PU64;
                    mat[r][k] = t as u32;
                }
            }
        }
    }
    (det % PU64) as u32
}

fn inv_fp(a: u32) -> u32 {
    // a^(p-2)
    let mut base = a as u64 % PU64;
    let mut e = (P - 2) as u64;
    let mut r: u64 = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = r * base % PU64;
        }
        base = base * base % PU64;
        e >>= 1;
    }
    r as u32
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
    fn poly(&mut self, dmax: usize) -> Vec<u32> {
        let d = (self.next() as usize) % (dmax + 1);
        let p = trim(&(0..=d).map(|_| self.field()).collect::<Vec<_>>());
        if p.is_empty() {
            vec![self.field().max(1)]
        } else {
            p
        }
    }
}

fn main() {
    let mut passed = 0usize;
    let mut total = 0usize;
    let mut check = |name: &str, cond: bool, detail: String| {
        total += 1;
        if cond {
            passed += 1;
        }
        let mark = if cond { "PASS" } else { "FAIL" };
        if detail.is_empty() {
            println!("  [{}] {}", mark, name);
        } else {
            println!("  [{}] {}  -- {}", mark, name, detail);
        }
    };

    let mut rng = Rng::new(7);

    // 1. mul / divmod / gcd via independent oracles
    println!("1. mul / divmod / gcd (eval-consistency, divmod identity, gcd maximality)");
    let (mut okmul, mut okdiv, mut okgcd) = (true, true, true);
    for _ in 0..400 {
        let a = rng.poly(8);
        let b = rng.poly(6);
        // mul: eval(a*b, t) == eval(a,t)*eval(b,t)
        let ab = mul(&a, &b);
        for _ in 0..3 {
            let t = rng.field();
            if eval(&ab, t) != feval_mul(&a, &b, t) {
                okmul = false;
            }
        }
        // divmod: a == q*b + r and deg(r) < deg(b)
        let (q, r) = divmod(&a, &b);
        let recon = add(&mul(&q, &b), &r);
        if recon != trim(&a) || (!r.is_empty() && deg(&r) >= deg(&b)) {
            okdiv = false;
        }
        // gcd: divides both, monic, cofactors coprime (=> it is the full gcd)
        let g = gcd(&a, &b);
        if !g.is_empty() {
            let divs = rem(&a, &g).is_empty() && rem(&b, &g).is_empty();
            let monic = *g.last().unwrap() == 1;
            let ca = divmod(&a, &g).0;
            let cb = divmod(&b, &g).0;
            let coprime = deg(&gcd(&ca, &cb)) == 0;
            if !(divs && monic && coprime) {
                okgcd = false;
            }
        }
    }
    check("pmul matches eval-consistency", okmul, String::new());
    check(
        "pdivmod satisfies a = q*b + r, deg r < deg b",
        okdiv,
        String::new(),
    );
    check(
        "pgcd divides both, monic, cofactors coprime",
        okgcd,
        String::new(),
    );

    // 2. resultant vs Sylvester determinant + zero-iff-common-factor
    println!("2. resultant vs Sylvester determinant");
    let mut okres = true;
    for _ in 0..300 {
        let a = rng.poly(7);
        let b = rng.poly(5);
        if resultant(&a, &b) != sylvester_res(&a, &b) {
            okres = false;
            break;
        }
    }
    check("presultant == Sylvester determinant", okres, String::new());
    // (x+3)(x+5) and (x+3)(x+9) share root -3  => resultant 0
    let g = mul(&[3, 1], &[5, 1]);
    let h = mul(&[3, 1], &[9, 1]);
    check(
        "presultant == 0 for common-factor polys",
        resultant(&g, &h) == 0,
        String::new(),
    );
    // coprime linear factors => nonzero
    let h2 = mul(&[7, 1], &[9, 1]);
    check(
        "presultant != 0 for coprime polys",
        resultant(&g, &h2) != 0,
        String::new(),
    );

    // 3. root-finding: planted roots found, no spurious roots
    println!("3. F_p root-finding");
    let mut okroots = true;
    let mut okspurious = true;
    for _ in 0..30 {
        let k = 1 + (rng.next() as usize) % 6;
        let mut planted: Vec<u32> = (0..k).map(|_| rng.field()).collect();
        planted.sort_unstable();
        planted.dedup();
        let mut f = vec![1u32];
        for &r in &planted {
            f = mul(&f, &[(P - r % P) % P, 1]); // (x - r)
        }
        let found = roots(&f);
        for r in &planted {
            if !found.contains(r) {
                okroots = false;
            }
        }
        for r in &found {
            if eval(&f, *r) != 0 {
                okspurious = false;
            }
        }
    }
    check("proots finds all planted roots", okroots, String::new());
    check(
        "proots returns no spurious roots",
        okspurious,
        String::new(),
    );
    check(
        "proots(const)/proots(0) == []",
        roots(&[]).is_empty() && roots(&[7]).is_empty(),
        String::new(),
    );

    // 4. interpolation round-trip
    println!("4. interpolation round-trips");
    let mut okint = true;
    for _ in 0..60 {
        let f = rng.poly(10);
        let need = f.len() + 2;
        // distinct xs
        let mut xs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        while xs.len() < need {
            let v = rng.field();
            if seen.insert(v) {
                xs.push(v);
            }
        }
        let ys: Vec<u32> = xs.iter().map(|&xi| eval(&f, xi)).collect();
        if interpolate(&xs, &ys) != trim(&f) {
            okint = false;
            break;
        }
    }
    check("interpolate recovers the polynomial", okint, String::new());

    // 5. NTT multiplication path: explicit NTT vs schoolbook, and the public
    // `mul` dispatcher on products large enough to use the NTT path.
    println!("5. NTT multiplication path");
    let mut okntt = true;
    let mut okdispatch = true;
    for (la, lb) in [(130usize, 151usize), (257, 199), (513, 377), (1024, 733)] {
        let a: Vec<u32> = (0..la).map(|_| rng.field()).collect();
        let b: Vec<u32> = (0..lb).map(|_| rng.field()).collect();
        let sb = mul_schoolbook(&a, &b);
        let nt = mul_ntt(&a, &b);
        if nt != sb {
            okntt = false;
        }
        if mul(&a, &b) != sb {
            okdispatch = false;
        }
    }
    check(
        "mul_ntt matches schoolbook on forced-NTT sizes",
        okntt,
        String::new(),
    );
    check(
        "mul dispatcher returns the same product on NTT sizes",
        okdispatch,
        String::new(),
    );

    // 6. Fast division path: Newton/reversal division vs schoolbook.
    println!("6. fast divmod path");
    let mut okfastdiv = true;
    let mut okfastidentity = true;
    for (la, lb) in [(700usize, 320usize), (1200, 513), (2500, 997)] {
        let mut a: Vec<u32> = (0..la).map(|_| rng.field()).collect();
        let mut b: Vec<u32> = (0..lb).map(|_| rng.field()).collect();
        if *a.last().unwrap() == 0 {
            *a.last_mut().unwrap() = 1;
        }
        if *b.last().unwrap() == 0 {
            *b.last_mut().unwrap() = 1;
        }
        let (q_fast, r_fast) = divmod(&a, &b);
        let (q_slow, r_slow) = divmod_schoolbook(&a, &b);
        if q_fast != q_slow || r_fast != r_slow {
            okfastdiv = false;
        }
        let recon = add(&mul(&q_fast, &b), &r_fast);
        if recon != trim(&a) || (!r_fast.is_empty() && deg(&r_fast) >= deg(&b)) {
            okfastidentity = false;
        }
    }
    check(
        "fast divmod matches schoolbook on forced-fast sizes",
        okfastdiv,
        String::new(),
    );
    check(
        "fast divmod satisfies a = q*b + r",
        okfastidentity,
        String::new(),
    );

    // 7. Large-degree GCD path: forces the Half-GCD dispatch.
    println!("7. large-degree gcd path");
    let mut common = vec![1u32];
    for r in [17u32, 29, 101, 1009, 65537] {
        common = mul(&common, &[(P - r) % P, 1]);
    }
    let mut ca: Vec<u32> = (0..700).map(|_| rng.field()).collect();
    let mut cb: Vec<u32> = (0..680).map(|_| rng.field()).collect();
    if *ca.last().unwrap() == 0 {
        *ca.last_mut().unwrap() = 1;
    }
    if *cb.last().unwrap() == 0 {
        *cb.last_mut().unwrap() = 1;
    }
    let a_large = mul(&common, &ca);
    let b_large = mul(&common, &cb);
    let g_large = gcd(&a_large, &b_large);
    check(
        "large gcd contains the planted factor",
        rem(&g_large, &common).is_empty() && rem(&common, &g_large).is_empty(),
        format!("deg(g)={} deg(common)={}", deg(&g_large), deg(&common)),
    );
    let mut x_large: Vec<u32> = (0..760).map(|_| rng.field()).collect();
    let mut y_large: Vec<u32> = (0..731).map(|_| rng.field()).collect();
    *x_large.last_mut().unwrap() = 1;
    *y_large.last_mut().unwrap() = 1;
    let g_coprime = gcd(&x_large, &y_large);
    check(
        "large random gcd is constant",
        deg(&g_coprime) == 0,
        format!("deg(g)={}", deg(&g_coprime)),
    );

    // 8. hardcoded parity vectors (match sympy-validated Python results)
    println!("8. hardcoded cross-language parity vectors");
    // (x+3)(x+5) = x^2 + 8x + 15
    check(
        "mul((x+3),(x+5)) == [15,8,1]",
        mul(&[3, 1], &[5, 1]) == vec![15u32, 8, 1],
        String::new(),
    );
    // gcd((x+3)(x+5), (x+3)(x+9)) == x+3 (monic)
    check(
        "gcd == x+3 ([3,1])",
        gcd(&mul(&[3, 1], &[5, 1]), &mul(&[3, 1], &[9, 1])) == vec![3u32, 1],
        String::new(),
    );
    // roots of (x-1)(x-2)(x-100) == {1,2,100}
    let f = mul(&mul(&[P - 1, 1], &[P - 2, 1]), &[P - 100, 1]);
    check(
        "roots((x-1)(x-2)(x-100)) == [1,2,100]",
        roots(&f) == vec![1u32, 2, 100],
        format!("got {:?}", roots(&f)),
    );

    println!("\n=== {}/{} checks passed ===", passed, total);
    std::process::exit(if passed == total { 0 } else { 1 });
}
