//! Validation for the native CICO-2 attack (symbolic perm + resultant + GCD route).
//! Mirrors tests/test_cico_attack.py + tests/test_skip_and_gcd.py, and adds a
//! per-line build-time table so the NTT-backed scaling is visible.
//!
//! Run:  cargo run --release --bin cico_golden

use poseidon_core::cico;
use poseidon_core::poly as pl;
use poseidon_core::{verify_cico, Poseidon1, CICO_C1, CICO_C2, P};
use std::time::Instant;

const PU64: u64 = P as u64;

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

    // 1. Symbolic permutation matches the numeric one (eval the symbolic output
    //    polynomials at X=x0 vs permutation_plus_linear of the concrete state).
    println!("1. Symbolic permutation over F_p[X] matches the numeric permutation");
    {
        let pos = Poseidon1::new_canonical(3, 16, 6, 2);
        let mut rng = Rng::new(11);
        let nfree = 14;
        let b: Vec<u32> = (0..nfree).map(|_| rng.field()).collect();
        let d: Vec<u32> = (0..nfree).map(|_| rng.field_nz()).collect();
        let assign: Vec<pl::Poly> = (0..nfree).map(|j| vec![b[j], d[j]]).collect();
        // symbolic
        let mut state_poly: Vec<pl::Poly> = vec![vec![CICO_C1 % P], vec![CICO_C2 % P]];
        state_poly.extend(assign);
        let sym = cico::perm_plus_linear_poly(&pos, &state_poly);
        // concrete at x0
        let x0 = rng.field();
        let mut concrete = [0u32; 16];
        concrete[0] = CICO_C1 % P;
        concrete[1] = CICO_C2 % P;
        for j in 0..nfree {
            concrete[2 + j] = ((b[j] as u64 + x0 as u64 * d[j] as u64) % PU64) as u32;
        }
        let num = pos.permutation_plus_linear(&concrete);
        let matches = (0..16).all(|i| pl::eval(&sym[i], x0) == num[i]);
        check(
            "eval(symbolic[i], x0) == permutation_plus_linear[i] for all 16 lanes",
            matches,
            String::new(),
        );
    }

    // 2. Resultant route finds a verifier-accepted CICO-2 solution (reduced rounds).
    println!("2. Resultant route solves CICO-2 (RF=2/RP=0, reduced)");
    let pos_red = Poseidon1::new_canonical(3, 16, 2, 0);
    let sol = cico::solve_cico2(&pos_red, 8, 5);
    let have_sol = sol.is_some() && verify_cico(&pos_red, sol.as_ref().unwrap());
    check(
        "solve_cico2 returns a verifier-accepted solution",
        have_sol,
        match &sol {
            Some(s) => format!("free[0..3]={:?}", &s[..3]),
            None => "none".into(),
        },
    );

    // 3. GCD route recovers the planted solution on a line through it.
    println!("3. GCD route recovers a planted solution (verifier-accepted)");
    if let Some(sol) = sol.clone() {
        let mut rng = Rng::new(99);
        let nfree = 14;
        let direction: Vec<u32> = (0..nfree).map(|_| rng.field_nz()).collect();
        let x0 = rng.field();
        // base = sol - x0*direction  => line hits sol at X = x0
        let base: Vec<u32> = (0..nfree)
            .map(|j| {
                let sub = (x0 as u64 * direction[j] as u64) % PU64;
                ((sol[j] as u64 + PU64 - sub) % PU64) as u32
            })
            .collect();
        let (got, info) = cico::solve_cico2_gcd(&pos_red, Some(&base), Some(&direction), 1, 0);
        let ok = got.as_ref().map_or(false, |g| verify_cico(&pos_red, g));
        check(
            "GCD route finds a verifier-accepted solution on the planted line",
            ok,
            format!("deg(inputs)={}", info.deg_inputs),
        );
        check(
            "recovered solution equals the planted one",
            got.as_ref() == Some(&sol),
            String::new(),
        );
    } else {
        check(
            "GCD planted recovery (skipped: no solution)",
            false,
            String::new(),
        );
        check("recovered == planted (skipped)", false, String::new());
    }

    // 4. No false positive on a random line.
    println!("4. GCD route does not false-positive on a random line");
    let (none_got, _) = cico::solve_cico2_gcd(&pos_red, None, None, 1, 123456);
    check(
        "random single line yields no spurious solution",
        none_got.is_none(),
        String::new(),
    );

    // 5. Resultant degree law: deg(R) == 3^(2*R_F + R_P).
    println!("5. Measured resultant degree law: deg(R) == 3^(2*R_F + R_P)");
    let mut okdeg = true;
    for (rf, rp) in [(2usize, 0usize), (2, 1), (2, 2)] {
        let pp = Poseidon1::new_canonical(3, 16, rf, rp);
        let d = cico::measure_resultant_degree(&pp, 2026);
        let exp = 3i64.pow((2 * rf + rp) as u32);
        let ideal = 3i64.pow((2 * (rf - 1) + rp) as u32);
        let ok = d == exp;
        okdeg &= ok;
        println!(
            "      RF={} RP={}: measured deg(R)={}  law 3^{}={}  ideal D_I 3^{}={}  (skip x{})  {}",
            rf,
            rp,
            d,
            2 * rf + rp,
            exp,
            2 * (rf - 1) + rp,
            ideal,
            exp / ideal,
            if ok { "ok" } else { "MISMATCH" }
        );
    }
    check(
        "deg(R) == 3^(2*R_F+R_P) on 3 data points",
        okdeg,
        String::new(),
    );

    // Real RF=6 degree target + GCD-route per-line build timing (scaling visibility).
    println!("\n  Real CICO (RF=6): as-built root-find degree 3^(12+RP), post-skip 3^(10+RP):");
    for rp in [6u32, 8, 10] {
        let dnaive = 3f64.powi(12 + rp as i32);
        println!(
            "    RP={:2}: as-built 3^{} ~= 2^{:.1}   post-skip 3^{} ~= 2^{:.1}",
            rp,
            12 + rp,
            dnaive.log2(),
            10 + rp,
            3f64.powi(10 + rp as i32).log2()
        );
    }

    println!("\n  GCD-route single-line build (NTT-backed mul) -- time per line vs RP:");
    println!("    (one line = one symbolic permutation_plus_linear over F_p[X] + gcd)");
    let mut rng = Rng::new(7);
    for rp in [0usize, 1, 2, 3] {
        let pos = Poseidon1::new_canonical(3, 16, 6, rp);
        let nfree = 14;
        let assign: Vec<pl::Poly> = (0..nfree)
            .map(|_| vec![rng.field(), rng.field_nz()])
            .collect();
        let mut state_poly: Vec<pl::Poly> = vec![vec![CICO_C1 % P], vec![CICO_C2 % P]];
        state_poly.extend(assign);
        let t0 = Instant::now();
        let s = cico::perm_plus_linear_poly(&pos, &state_poly);
        let g = pl::gcd(&pl::trim(&s[0]), &pl::trim(&s[1]));
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "    RF=6 RP={:2}: forward deg={:>7} (3^{})  build+gcd {:>8.3} ms  (gcd deg {})",
            rp,
            pl::deg(&s[0]),
            6 + rp,
            dt * 1e3,
            pl::deg(&g)
        );
    }
    println!("    => NTT removes the multiplication wall; large-RP scaling now exposes the next bottleneck(s).");

    println!("\n=== {}/{} checks passed ===", passed, total);
    std::process::exit(if passed == total { 0 } else { 1 });
}
