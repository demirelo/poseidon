//! Lightweight custom-MDS scout.
//!
//! This is an experiment-entry harness, not a bounty solver.  Its job is to
//! make MDS choice explicit, run the organizer/reference admissibility gate, and
//! emit JSON-only diagnostics that can seed a broader reduced-round campaign.
//!
//! Example:
//!   cargo run --release --bin custom_mds_scout -- --family cauchy-default --count 1 --rf 2 --rp 1

use poseidon_core::{cauchy_mds, circulant_mds, grain_round_constants, Poseidon1, P};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

const PU64: u64 = P as u64;
const T: usize = 16;
const PLONKY3_FIRST_ROW_16: [u32; T] = [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];

#[derive(Clone)]
struct Config {
    family: String,
    seed: u64,
    count: usize,
    rf: usize,
    rp: usize,
    admissibility: bool,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn field_nonzero(&mut self) -> u32 {
        1 + (self.next() % (PU64 - 1)) as u32
    }
}

#[inline]
fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}

#[inline]
fn fsub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + P - b
    }
}

fn fpow(mut a: u64, mut e: u64) -> u32 {
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

fn finv(a: u32) -> u32 {
    fpow(a as u64, (P - 2) as u64)
}

fn parse_args() -> Config {
    let mut cfg = Config {
        family: "cauchy-default".to_string(),
        seed: 0,
        count: 1,
        rf: 2,
        rp: 1,
        admissibility: true,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--family" => cfg.family = args.next().expect("--family value missing"),
            "--seed" => cfg.seed = args.next().expect("--seed value missing").parse().unwrap(),
            "--count" => cfg.count = args.next().expect("--count value missing").parse().unwrap(),
            "--rf" => cfg.rf = args.next().expect("--rf value missing").parse().unwrap(),
            "--rp" => cfg.rp = args.next().expect("--rp value missing").parse().unwrap(),
            "--no-admissibility" => cfg.admissibility = false,
            "--help" | "-h" => {
                eprintln!(
                    "usage: custom_mds_scout --family <cauchy-default|cauchy-variant|circulant|plonky3-circulant|sparse-plus-dense> --count N --rf RF --rp RP [--seed S] [--no-admissibility]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    cfg
}

fn cauchy_variant(seed: u64) -> Vec<Vec<u32>> {
    let offset = 19 + ((seed % 10_000) as u32) * 37;
    let x: Vec<u32> = (0..T).map(|i| (1 + 2 * i as u32) % P).collect();
    let y: Vec<u32> = (0..T).map(|j| (offset + 3 * j as u32 + 1000) % P).collect();
    (0..T)
        .map(|i| {
            (0..T)
                .map(|j| {
                    let d = fsub(x[i], y[j]);
                    assert_ne!(d, 0, "bad Cauchy variant collision");
                    finv(d)
                })
                .collect()
        })
        .collect()
}

fn random_circulant(seed: u64) -> Vec<Vec<u32>> {
    if seed == 0 {
        return circulant_mds(&PLONKY3_FIRST_ROW_16);
    }
    let mut rng = Rng::new(seed ^ 0xC1AC_0000_0000_0001);
    let row: Vec<u32> = (0..T).map(|_| rng.field_nonzero()).collect();
    circulant_mds(&row)
}

fn sparse_plus_dense(seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed ^ 0x5A5A_51A5_0000_0001);
    let u: Vec<u32> = (0..T).map(|_| rng.field_nonzero()).collect();
    let v: Vec<u32> = (0..T).map(|_| rng.field_nonzero()).collect();
    let d: Vec<u32> = (0..T).map(|_| rng.field_nonzero()).collect();
    (0..T)
        .map(|i| {
            (0..T)
                .map(|j| {
                    let rank_one = ((u[i] as u64 * v[j] as u64) % PU64) as u32;
                    if i == j {
                        fadd(rank_one, d[i])
                    } else {
                        rank_one
                    }
                })
                .collect()
        })
        .collect()
}

fn matrix_for(family: &str, seed: u64) -> Vec<Vec<u32>> {
    match family {
        "cauchy-default" => cauchy_mds(T),
        "cauchy-variant" => cauchy_variant(seed),
        "circulant" => random_circulant(seed),
        "plonky3-circulant" => circulant_mds(&PLONKY3_FIRST_ROW_16),
        "sparse-plus-dense" => sparse_plus_dense(seed),
        _ => panic!("unknown family {family}"),
    }
}

fn json_matrix(m: &[Vec<u32>]) -> String {
    let mut s = String::from("[");
    for (i, row) in m.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('[');
        for (j, x) in row.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&x.to_string());
        }
        s.push(']');
    }
    s.push(']');
    s
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn reference_path() -> Option<String> {
    if let Ok(p) = std::env::var("POSEIDON_TOOLS_PATH") {
        return Some(p);
    }
    for p in ["../reference/poseidon-tools", "reference/poseidon-tools"] {
        if std::path::Path::new(p)
            .join("poseidon/mds_matrix.py")
            .exists()
        {
            return Some(p.to_string());
        }
    }
    None
}

fn reference_gate(m: &[Vec<u32>]) -> (String, Option<bool>, String) {
    let Some(path) = reference_path() else {
        return (
            "unavailable".to_string(),
            None,
            "reference/poseidon-tools not found".to_string(),
        );
    };
    let code = r#"
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from poseidon.mds_matrix import verify_mds_matrix
p = 2130706433
m = json.load(sys.stdin)
canonical = json.dumps(m, separators=(",", ":"), sort_keys=False).encode()
print("sha256=" + hashlib.sha256(canonical).hexdigest())
print("admissible=" + ("true" if verify_mds_matrix(m, p) else "false"))
"#;
    let mut child = match Command::new("python3")
        .arg("-c")
        .arg(code)
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return (
                "unavailable".to_string(),
                None,
                format!("python3 spawn failed: {e}"),
            )
        }
    };
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(json_matrix(m).as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr).replace('\n', " ");
        return ("unavailable".to_string(), None, err);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut hash = "unavailable".to_string();
    let mut admissible = None;
    for line in stdout.lines() {
        if let Some(v) = line.strip_prefix("sha256=") {
            hash = v.to_string();
        } else if let Some(v) = line.strip_prefix("admissible=") {
            admissible = Some(v == "true");
        }
    }
    (hash, admissible, "verify_mds_matrix".to_string())
}

fn cheap_hash_only(m: &[Vec<u32>]) -> String {
    // Used only when --no-admissibility is requested.  The submission-relevant
    // path above always returns a SHA-256 from Python/hashlib.
    let mut h: u64 = 0xcbf29ce484222325;
    for row in m {
        for &x in row {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("fnv64:{h:016x}")
}

fn matrix_metrics(m: &[Vec<u32>]) -> (usize, usize, usize, usize, usize, f64) {
    let mut nonzero = 0usize;
    let mut row_min = T;
    let mut row_max = 0usize;
    let mut col_counts = [0usize; T];
    for row in m {
        let row_nz = row.iter().filter(|&&x| x != 0).count();
        row_min = row_min.min(row_nz);
        row_max = row_max.max(row_nz);
        nonzero += row_nz;
        for (j, &x) in row.iter().enumerate() {
            if x != 0 {
                col_counts[j] += 1;
            }
        }
    }
    let col_min = *col_counts.iter().min().unwrap();
    let col_max = *col_counts.iter().max().unwrap();
    let density = nonzero as f64 / (T * T) as f64;
    (nonzero, row_min, row_max, col_min, col_max, density)
}

fn support_counts(m: &[Vec<u32>], rounds: usize) -> Vec<usize> {
    let mut active = [false; T];
    active[0] = true;
    let mut out = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut next = [false; T];
        for i in 0..T {
            next[i] = (0..T).any(|j| m[i][j] != 0 && active[j]);
        }
        active = next;
        out.push(active.iter().filter(|&&b| b).count());
    }
    out
}

fn numeric_eval_us(m: &[Vec<u32>], rf: usize, rp: usize) -> f64 {
    let rc = grain_round_constants(3, T as u32, rf as u32, rp as u32);
    let pos = Poseidon1::from_parts(3, T, rf, rp, &rc, m);
    let input: [u32; T] = std::array::from_fn(|i| (17 * i as u32 + 5) % P);
    let reps = 256usize;
    let t0 = Instant::now();
    let mut acc = 0u32;
    for _ in 0..reps {
        let out = pos.compression_mode_hash(&input, T);
        acc ^= out[0];
    }
    std::hint::black_box(acc);
    t0.elapsed().as_secs_f64() * 1_000_000.0 / reps as f64
}

fn pow3(exp: usize) -> u128 {
    let mut x = 1u128;
    for _ in 0..exp {
        x *= 3;
    }
    x
}

fn main() {
    let cfg = parse_args();
    let mut records = Vec::new();
    for i in 0..cfg.count {
        let seed = cfg.seed + i as u64;
        let m = matrix_for(&cfg.family, seed);
        let (hash, admissible, gate_note) = if cfg.admissibility {
            reference_gate(&m)
        } else {
            (
                cheap_hash_only(&m),
                None,
                "admissibility disabled by --no-admissibility".to_string(),
            )
        };
        let (nonzero, row_min, row_max, col_min, col_max, density) = matrix_metrics(&m);
        let support = support_counts(&m, cfg.rf + cfg.rp);
        let eval_us = numeric_eval_us(&m, cfg.rf, cfg.rp);
        records.push(format!(
            concat!(
                "{{",
                "\"candidate_id\":{},",
                "\"family\":{},",
                "\"seed\":{},",
                "\"matrix_hash\":{},",
                "\"admissible\":{},",
                "\"gate\":\"reference/poseidon-tools poseidon.mds_matrix.verify_mds_matrix\",",
                "\"gate_note\":{},",
                "\"nonzero_entries\":{},",
                "\"density\":{:.6},",
                "\"row_nonzero_min\":{},\"row_nonzero_max\":{},",
                "\"col_nonzero_min\":{},\"col_nonzero_max\":{},",
                "\"support_from_lane0_after_mds\":{:?},",
                "\"numeric_eval_us_per_hash\":{:.3},",
                "\"eliminant_degree_measured\":null,",
                "\"eliminant_degree_default_law\":{},",
                "\"verdict\":\"admissibility_and_support_smoke_only\"",
                "}}"
            ),
            json_string(&format!("{}-{}", cfg.family, seed)),
            json_string(&cfg.family),
            seed,
            json_string(&hash),
            admissible
                .map(|b| b.to_string())
                .unwrap_or_else(|| "null".to_string()),
            json_string(&gate_note),
            nonzero,
            density,
            row_min,
            row_max,
            col_min,
            col_max,
            support,
            eval_us,
            pow3(2 * cfg.rf + cfg.rp)
        ));
    }

    println!("{{");
    println!("  \"artifact\": \"custom_mds_scout\",");
    println!("  \"date\": \"2026-06-17\",");
    println!("  \"rf\": {},", cfg.rf);
    println!("  \"rp\": {},", cfg.rp);
    println!("  \"family\": {},", json_string(&cfg.family));
    println!("  \"seed\": {},", cfg.seed);
    println!("  \"count\": {},", cfg.count);
    println!("  \"scope\": \"custom-MDS reduced-round scout entrypoint; not a bounty solve\",");
    println!("  \"candidates\": [");
    for (i, rec) in records.iter().enumerate() {
        println!(
            "    {}{}",
            rec,
            if i + 1 == records.len() { "" } else { "," }
        );
    }
    println!("  ]");
    println!("}}");
}
