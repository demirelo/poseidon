//! Honest throughput benchmark for the native Poseidon1 core.
//! Measures single-thread permutation rate on the canonical Cauchy+Grain
//! instance (rf=8/rp=20) and the CICO instance (rf=6/rp=10), plus the CICO
//! residual evaluation that dominates the algebraic-attack inner loop.
//!
//! Run:  cargo run --release --bin bench

use poseidon_core::*;
use std::hint::black_box;
use std::time::Instant;

fn bench<F: FnMut() -> u32>(label: &str, iters: u64, mut f: F) {
    // warmup
    for _ in 0..(iters / 20).max(1) {
        black_box(f());
    }
    let t0 = Instant::now();
    let mut acc = 0u32;
    for _ in 0..iters {
        acc = acc.wrapping_add(f());
    }
    let dt = t0.elapsed().as_secs_f64();
    black_box(acc);
    let rate = iters as f64 / dt;
    println!(
        "  {:<34} {:>10} iters  {:>8.3} s  {:>12.3e} /s  ({:.1} ns/op)",
        label,
        iters,
        dt,
        rate,
        dt / iters as f64 * 1e9
    );
}

fn main() {
    println!("KoalaBear/Poseidon1 native core -- single-thread throughput\n");

    // Permutation, full-strength collision instance (rf=8, rp=20 = 28 rounds).
    let pos_coll = Poseidon1::new_canonical(3, 16, 8, 20);
    let mut st = [0u32; 16];
    for i in 0..16 {
        st[i] = to_mont((i as u32) + 1);
    }
    let mut x = st;
    bench("perm rf=8/rp=20 (collision)", 5_000_000, || {
        x = pos_coll.perm_mont(&x, false);
        x[0]
    });

    // Permutation, CICO instance (rf=6, rp=10 = 16 rounds).
    let pos_cico = Poseidon1::new_canonical(3, 16, 6, 10);
    let mut y = st;
    bench("perm rf=6/rp=10 (CICO open)", 5_000_000, || {
        y = pos_cico.perm_mont(&y, true);
        y[0]
    });

    // CICO residual: the operation the GCD/birthday inner loop calls per trial.
    let mut ctr = 0u32;
    let mut free = [0u32; 14];
    bench("cico_residual rf=6/rp=10", 5_000_000, || {
        ctr = ctr.wrapping_add(1);
        free[0] = ctr;
        let (a, b) = cico_residual(&pos_cico, &free);
        a ^ b
    });

    println!("\nNote: MDS is a naive t^2 mont-mul inner product (256 muls/round).");
    println!("Next speed lever: exploit the Toeplitz MDS (M[i][j]=f(i-j)) as a");
    println!("length-31 NTT convolution + SIMD.");
}
