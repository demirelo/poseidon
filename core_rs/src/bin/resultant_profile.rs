//! Synthetic resultant Euclidean-profile probe.
//!
//! This does not model Poseidon directly; it isolates the F_p[x] resultant wall
//! at the same univariate degrees seen by zero-test per-X resultants.

use poseidon_core::poly::{self, SplitMix};
use poseidon_core::P;
use std::time::Instant;

fn random_poly(degree: usize, rng: &mut SplitMix) -> Vec<u32> {
    let mut f = Vec::with_capacity(degree + 1);
    for _ in 0..degree {
        f.push(rng.next_field());
    }
    f.push(1 + (rng.next_field() % (P - 1)));
    f
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let degrees = if args.is_empty() {
        vec![19683usize, 59049usize]
    } else {
        args.iter()
            .map(|s| s.parse::<usize>().expect("degree must parse"))
            .collect::<Vec<_>>()
    };

    println!("Synthetic resultant profile over KoalaBear F_p[x]");
    println!(
        "  {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "degree", "steps", "q1", "q2_3", "q4_15", "q16_255", "qfast", "max_q", "avg_q", "seconds"
    );
    println!("  {}", "-".repeat(102));

    for degree in degrees {
        let mut rng = SplitMix::new(0x5253_0000 + degree as u64);
        let a = random_poly(degree, &mut rng);
        let b = random_poly(degree, &mut rng);
        let start = Instant::now();
        let (res, stats) = poly::resultant_profiled(&a, &b);
        let seconds = start.elapsed().as_secs_f64();
        let avg_q = if stats.steps == 0 {
            0.0
        } else {
            stats.total_q_len as f64 / stats.steps as f64
        };
        println!(
            "  {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10.3} {:>10.2}",
            degree,
            stats.steps,
            stats.q_len_1,
            stats.q_len_2_3,
            stats.q_len_4_15,
            stats.q_len_16_255,
            stats.q_len_fast,
            stats.max_q_len,
            avg_q,
            seconds
        );
        println!(
            "    res={} initial_deg=({}, {}) schoolbook={} fast={} q2={} q3={} avg_divisor_len={:.1} max_drop={} zero_remainder={}",
            res,
            stats.initial_deg_a,
            stats.initial_deg_b,
            stats.schoolbook_steps,
            stats.fast_steps,
            stats.q_len_2,
            stats.q_len_3,
            if stats.steps == 0 {
                0.0
            } else {
                stats.total_divisor_len as f64 / stats.steps as f64
            },
            stats.max_degree_drop,
            stats.zero_remainder
        );
        if !stats.jump_events.is_empty() {
            let jumps = stats
                .jump_events
                .iter()
                .map(|e| {
                    format!(
                        "{}:q{}:{}>{}>{}:d{}",
                        e.step, e.q_len, e.deg_a, e.deg_b, e.deg_r, e.degree_drop
                    )
                })
                .collect::<Vec<_>>();
            println!("    jumps=[{}]", jumps.join(","));
        }
    }
}
