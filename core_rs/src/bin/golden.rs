//! Golden-vector validation for the native core. Mirrors the anchors in
//! tests/test_golden.py so the Rust core is proven bit-identical to the
//! independent Python reference (and thus to khovratovich/poseidon-tools).
//!
//! Run:  cargo run --release --bin golden

#[path = "../golden_data.rs"]
mod golden_data;
use golden_data::{P3_EXPECTED, P3_FIRST_ROW, P3_RC};
use poseidon_core::*;

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

    // 1. Plonky3 width-16 permutation vector (validates round schedule).
    println!("1. Plonky3 width-16 permutation vector (round schedule)");
    let mds = circulant_mds(&P3_FIRST_ROW);
    let pos_p3 = Poseidon1::from_parts(3, 16, 8, 20, &P3_RC, &mds);
    let mut inp = [0u32; 16];
    for i in 0..16 {
        inp[i] = i as u32;
    }
    let got = pos_p3.permutation(&inp);
    check(
        "perm(range(16)) == published Plonky3 vector",
        got == P3_EXPECTED,
        format!("first3 got={:?} exp={:?}", &got[..3], &P3_EXPECTED[..3]),
    );

    // 2. Cauchy + Grain canonical instance (validates Grain LFSR + Cauchy MDS
    //    + Montgomery arithmetic simultaneously).
    println!("2. Cauchy+Grain canonical instance (Grain + Cauchy MDS + Montgomery)");
    let pos = Poseidon1::new_canonical(3, 16, 8, 20);
    let z = pos.permutation(&[0u32; 16]);
    check(
        "perm([0;16])[0] == 1393439926",
        z[0] == 1393439926,
        format!("got={}", z[0]),
    );

    // 3. Published t=3 partial collision (compression mode, rf=8/rp=20).
    println!("3. Published t=3 partial collision (Beltran/Merz/Rodriguez/Scarlata)");
    let x: [u32; 15] = [
        146101246, 585745660, 1080651781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let y: [u32; 15] = [
        310195439, 1632272689, 97247552, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    check(
        "verify_collision(x,y,t=3) == true",
        verify_collision(&pos, &x, &y, 3),
        String::new(),
    );
    check(
        "verify_collision(x,y,t=4) == false (only 3 words collide)",
        !verify_collision(&pos, &x, &y, 4),
        String::new(),
    );
    check(
        "verify_collision(x,x,t=3) == false (x==y rejected)",
        !verify_collision(&pos, &x, &x, 3),
        String::new(),
    );

    // 4. Field sanity (Montgomery round-trips, REDC identities).
    println!("4. Field sanity");
    let round_trip_ok = (0u32..1000).all(|a| from_mont(to_mont(a)) == a);
    check(
        "to_mont/from_mont round-trips",
        round_trip_ok,
        String::new(),
    );
    // (a*b) in normal domain == from_mont(mont_mul(to_mont a, to_mont b))
    let mul_ok = {
        let a = 123456789u64;
        let b = 987654321u64;
        let want = (a * b % (P as u64)) as u32;
        from_mont(mont_mul(to_mont(a as u32), to_mont(b as u32))) == want
    };
    check(
        "mont_mul matches normal-domain product",
        mul_ok,
        String::new(),
    );
    let cube_ok = {
        let a = 1234567u32;
        let want = ((a as u64).pow(3) % (P as u64)) as u32;
        from_mont(mont_cube(to_mont(a))) == want
    };
    check("mont_cube matches x^3", cube_ok, String::new());

    // 5. CICO predicate wiring: a random free input is (a.s.) not a solution,
    //    and a residual of (0,0) would accept. Exercise rp in the funded set.
    println!("5. CICO predicate wiring (rp in {{6,8,10}})");
    for rp in [6usize, 8, 10] {
        let posc = Poseidon1::new_canonical(3, 16, 6, rp);
        let free: Vec<u32> = (2u32..16).collect();
        let is_sol = verify_cico(&posc, &free);
        let (r0, r1) = cico_residual(&posc, &free);
        check(
            &format!("cico rp={}: random input rejected", rp),
            !is_sol,
            format!("residual=({},{})", r0, r1),
        );
        check(
            &format!("cico rp={}: residual==(0,0) iff accept", rp),
            (r0 == 0 && r1 == 0) == is_sol,
            String::new(),
        );
    }

    println!("\n=== {}/{} checks passed ===", passed, total);
    std::process::exit(if passed == total { 0 } else { 1 });
}
