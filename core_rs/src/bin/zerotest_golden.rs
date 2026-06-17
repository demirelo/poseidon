//! Validation for the native zero-test verifier and relaxed-search scaffold.
//!
//! Run:  cargo run --release --bin zerotest_golden

use poseidon_core::zerotest;
use poseidon_core::zerotest::{Fp2, ZT_D, ZT_RF, ZT_T};
use poseidon_core::Poseidon1;

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

    println!("1. Fp2 arithmetic over Fp[x]/(x^2 - 3)");
    let sqrt3 = Fp2::new(0, 1);
    check(
        "sqrt(3)^2 == 3",
        sqrt3.mul(sqrt3) == Fp2::new(3, 0),
        format!("got {:?}", sqrt3.mul(sqrt3)),
    );
    let a = Fp2::new(11, 7);
    let b = Fp2::new(19, 5);
    let c = Fp2::new(23, 29);
    check(
        "multiplication distributes over addition",
        a.mul(b.add(c)) == a.mul(b).add(a.mul(c)),
        String::new(),
    );

    println!("2. Root-first polynomial construction");
    let root = Fp2::new(12345, 67890);
    let mut cofactor = [Fp2::ZERO; ZT_D];
    for (j, slot) in cofactor.iter_mut().enumerate() {
        *slot = Fp2::new((17 + 31 * j) as u32, (99 + 43 * j) as u32);
    }
    let coeffs = zerotest::root_first_coeffs(root, &cofactor);
    check(
        "P(root) == 0 for P(z)=(z-root)G(z)",
        zerotest::eval_poly(&coeffs, root).is_zero(),
        format!("P(root)={:?}", zerotest::eval_poly(&coeffs, root)),
    );
    check(
        "root-first construction has degree 7 when G degree 6",
        zerotest::degree(&coeffs) == 7,
        format!("deg={}", zerotest::degree(&coeffs)),
    );

    println!("3. Verifier gates match the reference predicate shape");
    let pos = Poseidon1::new_canonical(3, ZT_T, ZT_RF, 6);
    let zero = [0u32; 16];
    check(
        "zero polynomial is rejected",
        !zerotest::verify_zerotest(&pos, &zero),
        String::new(),
    );
    let constant = {
        let mut p = [0u32; 16];
        p[0] = 5;
        p
    };
    check(
        "degree-0 polynomial is rejected",
        !zerotest::verify_zerotest(&pos, &constant),
        String::new(),
    );
    check(
        "wrong-length vector is rejected without panic",
        !zerotest::verify_zerotest(&pos, &zero[..15]),
        String::new(),
    );
    let p_hat = zerotest::flatten_coeffs(&coeffs);
    let residual = zerotest::zerotest_residual(&pos, &p_hat).unwrap();
    check(
        "nonconstant root-first vector is well-formed for residual evaluation",
        zerotest::degree(&coeffs) >= 1 && zerotest::degree(&coeffs) <= 7,
        format!("hash residual P(a0)={:?}", residual),
    );
    check(
        "k=0 relaxed verifier accepts any valid-degree polynomial",
        zerotest::verify_zerotest_relaxed(&pos, &p_hat, 0),
        String::new(),
    );

    println!("4. Root-first hash residual exposes the fixed-point target");
    let hr = zerotest::root_first_hash_residual(&pos, root, &cofactor);
    check(
        "hash residual is computable as H(P_hat)[0] - root",
        hr == zerotest::hash_point(&pos, &p_hat).unwrap().sub(root),
        format!("root residual={:?}", hr),
    );

    println!("5. Deterministic relaxed zero-test random-search smoke test");
    let k = 4;
    let hit = zerotest::search_relaxed_random(&pos, k, 50_000, 20260602);
    let ok = hit.as_ref().map_or(false, |h| {
        zerotest::verify_zerotest_relaxed(&pos, &h.p_hat, k)
    });
    check(
        "finds a verifier-confirmed relaxed k=4 hit on RF=6/RP=6",
        ok,
        match &hit {
            Some(h) => format!(
                "attempt={} degree={} residual={:?}",
                h.attempt, h.degree, h.residual
            ),
            None => "none".into(),
        },
    );

    println!("\n=== {}/{} checks passed ===", passed, total);
    std::process::exit(if passed == total { 0 } else { 1 });
}
