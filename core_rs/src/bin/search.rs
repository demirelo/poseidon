//! Brute-force CICO calibration baseline (plan Phase 1: "profile random search").
//!
//! On the REAL open instance (CICO rf=6/rp=10, Cauchy+Grain, C1=0xC09DE4,
//! C2=0xEE6282) it searches free inputs and tracks the best *relaxed* hit: the
//! largest m such that both constrained output words have their low m bits zero
//! (this is exactly the reference's relaxed CICO predicate). It then extrapolates
//! the cost of an EXACT solve by pure search, which motivates the algebraic
//! attack. Pure search is NOT how we win -- this just pins the baseline number.
//!
//! Run:  cargo run --release --bin search -- <trials>   (default 10_000_000)

use poseidon_core::*;
use std::time::Instant;

#[inline(always)]
fn low_zero_bits(x: u32) -> u32 {
    if x == 0 {
        31
    } else {
        x.trailing_zeros().min(31)
    }
}

fn main() {
    let trials: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    let pos = Poseidon1::new_canonical(3, 16, 6, 10); // CICO open instance
    println!(
        "CICO rf=6/rp=10 brute-force baseline  (C1=0x{:X}, C2=0x{:X}, p=2^31-2^24+1)",
        CICO_C1, CICO_C2
    );
    println!("trials = {}\n", trials);

    let mut free = [0u32; 14];
    // light fixed entropy in the tail so we are not on a structured affine line
    for i in 2..14 {
        free[i] = (0x9E3779B1u32).wrapping_mul(i as u32 + 1) % P;
    }

    let mut best_m = 0u32;
    let mut best_free = free;
    let mut best_res = (u32::MAX, u32::MAX);

    let t0 = Instant::now();
    let mut i: u64 = 0;
    while i < trials {
        // 2-word counter over the field gives > 2^62 distinct inputs.
        free[0] = (i % PU64_HOST) as u32;
        free[1] = ((i / PU64_HOST) % PU64_HOST) as u32;
        let (r0, r1) = cico_residual(&pos, &free);
        let m = low_zero_bits(r0).min(low_zero_bits(r1));
        if m > best_m {
            best_m = m;
            best_free = free;
            best_res = (r0, r1);
            println!(
                "  new best: m={:2} low bits zero in BOTH words   residual=({}, {})",
                m, r0, r1
            );
        }
        i += 1;
    }
    let dt = t0.elapsed().as_secs_f64();
    let rate = trials as f64 / dt;

    println!("\n--- result ---");
    println!(
        "searched {} inputs in {:.2}s  =>  {:.3e} cico_residual/s (1 thread)",
        trials, dt, rate
    );
    println!(
        "best relaxed CICO solution: m = {} low bits zero in BOTH constrained words",
        best_m
    );
    println!("  residual = ({}, {})", best_res.0, best_res.1);
    println!("  free_inputs (x3..x16) = {:?}", best_free);
    // self-check via the relaxed predicate
    let (c0, c1) = cico_residual(&pos, &best_free);
    let m_check = low_zero_bits(c0).min(low_zero_bits(c1));
    println!(
        "  re-verified m = {}  (matches: {})",
        m_check,
        m_check == best_m
    );

    println!("\n--- extrapolation to EXACT solve by brute force ---");
    // exact = both words fully zero = 2*31 = 62 bits = p^2 ~ 2^62 trials
    let exact_trials = 2f64.powi(62);
    let secs = exact_trials / rate;
    println!(
        "  exact solve needs ~p^2 = 2^62 ~= {:.2e} trials  =>  {:.2e} core-years (1 thread)",
        exact_trials,
        secs / (3600.0 * 24.0 * 365.0)
    );
    println!("  CONCLUSION: brute force is infeasible; the algebraic (resultant/GCD)");
    println!("  attack is mandatory. This baseline is calibration only.");
}

const PU64_HOST: u64 = P as u64;
