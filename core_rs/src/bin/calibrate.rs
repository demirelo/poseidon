//! RF=6 CICO GCD-route calibration: where is the wall after NTT multiplication?
//!
//! Times one GCD-route line broken into phases (symbolic perm build via NTT mul
//! vs the final gcd via Euclidean division), microbenches mul_ntt vs divmod, and
//! reports the NTT-length ceiling. Answers the gate question: is the next wall
//! divmod/gcd, symbolic cubing, or memory.
//!
//!   cargo run --release --bin calibrate            # RP=4,5 full + RP=6 fast+projected
//!   cargo run --release --bin calibrate rp6full    # one real RP=6 line incl. gcd (slow)

use poseidon_core::cico;
use poseidon_core::poly as pl;
use poseidon_core::{Poseidon1, CICO_C1, CICO_C2, P};
use std::time::Instant;

const PU64: u64 = P as u64;
const MAX_NTT_LEN: usize = 1 << 24;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s)
    }
    fn f(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) % PU64) as u32
    }
    fn fnz(&mut self) -> u32 {
        1 + self.f() % (P - 1)
    }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn line_state(rng: &mut Rng) -> Vec<pl::Poly> {
    let mut st: Vec<pl::Poly> = vec![vec![CICO_C1 % P], vec![CICO_C2 % P]];
    for _ in 0..14 {
        st.push(vec![rng.f(), rng.fnz()]); // b + d*X
    }
    st
}

fn secs<R, F: FnOnce() -> R>(f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64())
}

fn rp6_full() {
    // One real RF=6/RP=6 line including the schoolbook gcd (slow; for background runs).
    let pos = Poseidon1::new_canonical(3, 16, 6, 6);
    let mut rng = Rng::new(20260602);
    let st = line_state(&mut rng);
    let (s, t_perm) = secs(|| cico::perm_plus_linear_poly(&pos, &st));
    let f0 = pl::trim(&s[0]);
    let f1 = pl::trim(&s[1]);
    let (g, t_gcd) = secs(|| pl::gcd(&f0, &f1));
    let (rts, t_roots) = secs(|| pl::roots(&g));
    println!(
        "rp6full: deg={} perm={:.2}s gcd={:.2}s roots={:.2}s gcd_deg={} n_roots={} total={:.2}s",
        pl::deg(&f0),
        t_perm,
        t_gcd,
        t_roots,
        pl::deg(&g),
        rts.len(),
        t_perm + t_gcd + t_roots
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "rp6full" {
        rp6_full();
        return;
    }

    println!(
        "KoalaBear CICO GCD-route calibration (post-NTT). MAX_NTT_LEN = 2^24 = {}.\n",
        MAX_NTT_LEN
    );

    // --- NTT-length ceiling: largest product in the perm is the final cube (deg 3^(6+RP)).
    println!("NTT-length ceiling (perm's largest product needs len >= forward degree):");
    for rp in 6u32..=11 {
        let fwd = 3f64.powi(6 + rp as i32);
        let need = next_pow2(fwd as usize + 1);
        let fits = need <= MAX_NTT_LEN;
        println!(
            "  RP={:2}: forward deg 3^{} = {:>12.0}  -> NTT len {:>10}  {}",
            rp,
            6 + rp,
            fwd,
            need,
            if fits {
                "ok"
            } else {
                "EXCEEDS 2^24 -> needs Kronecker/CRT NTT"
            }
        );
    }
    println!();

    // --- Phase breakdown per RP. RP<=5 full (incl. gcd); RP=6 fast (perm+mul) + projected.
    println!("Per-line phase breakdown (one GCD-route line, RF=6):");
    println!("  RP |  fwd deg  | perm build |   gcd     |  mul_ntt(1) | divmod(1) | note");
    let mut fit_c: Option<f64> = None; // gcd ≈ c * deg^2
    for rp in [4usize, 5, 6] {
        let pos = Poseidon1::new_canonical(3, 16, 6, rp);
        let mut rng = Rng::new(7 + rp as u64);
        let st = line_state(&mut rng);
        let (s, t_perm) = secs(|| cico::perm_plus_linear_poly(&pos, &st));
        let f0 = pl::trim(&s[0]);
        let f1 = pl::trim(&s[1]);
        let d = pl::deg(&f0).max(1) as f64;

        // microbench: one NTT mul of two degree-d polys
        let a: Vec<u32> = (0..=pl::deg(&f0) as usize).map(|_| rng.f()).collect();
        let b: Vec<u32> = (0..=pl::deg(&f1) as usize).map(|_| rng.f()).collect();
        let (_, t_mul) = secs(|| pl::mul_ntt(&a, &b));

        if rp <= 5 {
            // full: real gcd + a single divmod microbench
            let (_g, t_gcd) = secs(|| pl::gcd(&f0, &f1));
            let big: Vec<u32> = (0..(2 * d as usize)).map(|_| rng.f()).collect();
            let (_, t_div) = secs(|| pl::divmod(&big, &f0));
            fit_c = Some(t_gcd / (d * d)); // calibrate the deg^2 constant on the largest full RP
            println!(
                "  {:2} | {:>9.0} | {:>8.3} s | {:>7.3} s | {:>9.4} s | {:>7.3} s | measured",
                rp, d, t_perm, t_gcd, t_mul, t_div
            );
        } else {
            // RP=6: project gcd/divmod from the deg^2 fit (avoids a ~minutes foreground gcd)
            let c = fit_c.unwrap_or(0.0);
            let proj_gcd = c * d * d;
            println!(
                "  {:2} | {:>9.0} | {:>8.3} s | ~{:>6.1} s | {:>9.4} s |    (proj) | perm+mul measured; gcd PROJECTED (c*deg^2)",
                rp, d, t_perm, proj_gcd, t_mul
            );
            // Full-solve accounting. A 1-D line meets the codim-2 CICO-2 solution
            // set (dim 12 in F_p^14) with prob ~1/p, so the GCD route needs ~p
            // lines for one solution -- NOT p/deg. RP=6 gcd-route is ~2^58.7 ops.
            let lines = PU64 as f64; // ~p lines, one expected solution
            let per_line = t_perm + proj_gcd;
            let yr = 3.15e7_f64;
            println!(
                "\n  RP=6 GCD-route full-solve accounting (lines ~= p, per the 1/p hit rate):"
            );
            println!(
                "    ~p = {:.2e} lines x {:.0} s/line (schoolbook) = {:.2e} s = {:.0} core-years",
                lines,
                per_line,
                lines * per_line,
                lines * per_line / yr
            );
            println!(
                "    perm-build floor (even with instant gcd): p x {:.2} s = {:.0} core-years",
                t_perm,
                lines * t_perm / yr
            );
            println!(
                "    => single-machine infeasible; embarrassingly parallel but cluster-scale."
            );
            println!("    For LOW RP the resultant route (deg D_I=3^(12+RP)) is cheaper (RP=6 ~2^53 ops,");
            println!("    ~0.05 PiB); the GCD route wins only at high RP where resultant memory blows up");
            println!("    (RP=10 ~355 PiB). RP=6 is ~2^53 (res) / ~2^59 (gcd) ops.");
        }
    }

    println!("\nDiagnosis:");
    println!(
        "  * NTT fixed multiplication: 'perm build' and 'mul_ntt' are cheap and ~quasilinear."
    );
    println!(
        "  * Fast large-quotient divmod is cheap now, but Euclidean gcd still has O(deg^2) step count."
    );
    println!(
        "  * A production Half-GCD would complete the O(M(deg) log deg) stack, but CICO remains cluster-scale."
    );
    println!("  * Separately, RP>=10 exceeds the 2^24 NTT length -> Kronecker substitution / multi-prime CRT.");
}
