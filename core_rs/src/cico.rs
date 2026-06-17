//! Native CICO-2 algebraic attack on Poseidon1 (permutation_plus_linear).
//!
//! Mirrors `poseidon_attack/cico_attack.py`. Three pieces:
//!   * `perm_plus_linear_poly` — the permutation run over F_p[X] (symbolic perm).
//!   * `solve_cico2` — resultant route (2 free vars): Res_Y eliminates Y to R(X),
//!     roots of R(X) back-substitute via gcd. Used here to *find* a solution to plant.
//!   * `solve_cico2_gcd` — the GCD route (1 free var along a line): linear memory,
//!     the operative path at scale.
//!
//! NAIVE degrees (no skip-first-rounds): forward deg = 3^(R_F+R_P);
//! deg(R) = 3^(2*R_F+R_P). The polynomial product path is now NTT-backed for
//! large products; RF=6/RP=6 calibration will reveal the next wall.

use crate::poly as pl;
use crate::{Poseidon1, CICO_C1, CICO_C2, P};

const PU64: u64 = P as u64;

/// x^3 over F_p[X] (alpha = 3).
fn sbox_poly(f: &[u32]) -> pl::Poly {
    pl::mul(&pl::mul(f, f), f)
}

/// Apply the (normal-domain) MDS to a state of polynomials.
fn mds_poly(state: &[pl::Poly], mds_n: &[Vec<u32>]) -> Vec<pl::Poly> {
    let t = state.len();
    let mut out = Vec::with_capacity(t);
    for i in 0..t {
        let mut acc: pl::Poly = vec![];
        for j in 0..t {
            if !state[j].is_empty() {
                acc = pl::add(&acc, &pl::scalar(&state[j], mds_n[i][j]));
            }
        }
        out.push(acc);
    }
    out
}

/// permutation_plus_linear over F_p[X]; `state` is t polynomials.
pub fn perm_plus_linear_poly(pos: &Poseidon1, state: &[pl::Poly]) -> Vec<pl::Poly> {
    assert_eq!(pos.alpha, 3, "symbolic sbox specialised to alpha=3");
    let t = pos.t;
    let mut s: Vec<pl::Poly> = state.to_vec();
    s = mds_poly(&s, &pos.mds_n); // initial linear layer
    let half = pos.r_f / 2;
    let mut idx = 0usize;
    for _ in 0..half {
        for i in 0..t {
            s[i] = pl::add(&s[i], &[pos.rc_n[idx][i]]);
        }
        for i in 0..t {
            s[i] = sbox_poly(&s[i]);
        }
        s = mds_poly(&s, &pos.mds_n);
        idx += 1;
    }
    for _ in 0..pos.r_p {
        for i in 0..t {
            s[i] = pl::add(&s[i], &[pos.rc_n[idx][i]]);
        }
        s[0] = sbox_poly(&s[0]);
        s = mds_poly(&s, &pos.mds_n);
        idx += 1;
    }
    for _ in 0..half {
        for i in 0..t {
            s[i] = pl::add(&s[i], &[pos.rc_n[idx][i]]);
        }
        for i in 0..t {
            s[i] = sbox_poly(&s[i]);
        }
        s = mds_poly(&s, &pos.mds_n);
        idx += 1;
    }
    s
}

/// State with positions 0,1 = (C1,C2) constants, positions 2.. = `assignment` polys.
fn build_state(assignment: &[pl::Poly], t: usize) -> Vec<pl::Poly> {
    let mut st = Vec::with_capacity(t);
    st.push(vec![CICO_C1 % P]);
    st.push(vec![CICO_C2 % P]);
    for j in 0..(t - 2) {
        st.push(assignment[j].clone());
    }
    st
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
    fn field_nz(&mut self) -> u32 {
        1 + (self.next() % (PU64 - 1)) as u32
    }
}

/// Resultant route. var positions px,py (0-based into the 14 free slots: state pos
/// = 2+px). Returns Some(free_inputs[14]) or None. Reduced rounds only.
pub fn solve_cico2(pos: &Poseidon1, tries: usize, seed: u64) -> Option<Vec<u32>> {
    let (px, py) = (0usize, 1usize); // first two free slots
    let t = pos.t;
    let nfree = t - 2;
    let mut rng = Rng::new(seed);

    // out0(Y), out1(Y) with X fixed = xval, the other free slots fixed.
    let outs_at_x = |xval: u32, fixed: &[u32]| -> (pl::Poly, pl::Poly) {
        let mut assign: Vec<pl::Poly> = Vec::with_capacity(nfree);
        for j in 0..nfree {
            if j == px {
                assign.push(vec![xval]);
            } else if j == py {
                assign.push(vec![0, 1]); // Y
            } else {
                assign.push(vec![fixed[j]]);
            }
        }
        let s = perm_plus_linear_poly(pos, &build_state(&assign, t));
        (pl::trim(&s[0]), pl::trim(&s[1])) // targets C3=C4=0
    };

    let fwd = 3usize.pow((pos.r_f + pos.r_p) as u32);
    let n_nodes = ((2 * fwd * fwd + 1) as u64).min(PU64 - 1) as usize;

    for _ in 0..tries {
        let mut fixed = vec![0u32; nfree];
        for j in 0..nfree {
            if j != px && j != py {
                fixed[j] = rng.field();
            }
        }
        let mut xs = Vec::with_capacity(n_nodes);
        let mut ys = Vec::with_capacity(n_nodes);
        let mut xv = 1u32;
        while xs.len() < n_nodes && (xv as u64) < PU64 {
            let (f0, f1) = outs_at_x(xv, &fixed);
            xs.push(xv);
            ys.push(pl::resultant(&f0, &f1));
            xv += 1;
        }
        let r_poly = pl::interpolate(&xs, &ys);
        for xroot in pl::roots(&r_poly) {
            let (f0, f1) = outs_at_x(xroot, &fixed);
            let g = pl::gcd(&f0, &f1);
            for yroot in pl::roots(&g) {
                let mut free = vec![0u32; nfree];
                for j in 0..nfree {
                    free[j] = if j == px {
                        xroot
                    } else if j == py {
                        yroot
                    } else {
                        fixed[j]
                    };
                }
                if crate::verify_cico(pos, &free) {
                    return Some(free);
                }
            }
        }
    }
    None
}

pub struct GcdInfo {
    pub lines_tried: usize,
    pub deg_inputs: i64,
}

/// GCD route: restrict free inputs to a line free(X) = base + X*direction, so
/// out0,out1 become univariate in X; a shared F_p root of gcd(out0,out1) solves
/// CICO-2. If `pinned` is set, only the supplied (base,direction) line is tried
/// (used for planted-solution recovery); otherwise up to `lines` random lines.
pub fn solve_cico2_gcd(
    pos: &Poseidon1,
    base: Option<&[u32]>,
    direction: Option<&[u32]>,
    lines: usize,
    seed: u64,
) -> (Option<Vec<u32>>, GcdInfo) {
    let t = pos.t;
    let nfree = t - 2;
    let mut rng = Rng::new(seed);
    let pinned = base.is_some();

    let mut deg_inputs = -1i64;
    for line in 0..lines.max(1) {
        let b: Vec<u32> = match base {
            Some(v) => v.to_vec(),
            None => (0..nfree).map(|_| rng.field()).collect(),
        };
        let d: Vec<u32> = match direction {
            Some(v) => v.to_vec(),
            None => (0..nfree).map(|_| rng.field_nz()).collect(),
        };
        // free slot j carries the degree-1 polynomial b[j] + d[j]*X
        let assign: Vec<pl::Poly> = (0..nfree).map(|j| vec![b[j] % P, d[j] % P]).collect();
        let s = perm_plus_linear_poly(pos, &build_state(&assign, t));
        let f0 = pl::trim(&s[0]);
        let f1 = pl::trim(&s[1]);
        deg_inputs = pl::deg(&f0);
        let g = pl::gcd(&f0, &f1);
        for xr in pl::roots(&g) {
            let free: Vec<u32> = (0..nfree)
                .map(|j| ((b[j] as u64 + xr as u64 * d[j] as u64) % PU64) as u32)
                .collect();
            if crate::verify_cico(pos, &free) {
                return (
                    Some(free),
                    GcdInfo {
                        lines_tried: line + 1,
                        deg_inputs,
                    },
                );
            }
        }
        if pinned {
            break;
        }
    }
    (
        None,
        GcdInfo {
            lines_tried: lines.max(1),
            deg_inputs,
        },
    )
}

/// True degree of R(X) = Res_Y(out0,out1) for the naive bivariate system,
/// via adaptive interpolation. Measured law: deg(R) = 3^(2*R_F + R_P).
pub fn measure_resultant_degree(pos: &Poseidon1, seed: u64) -> i64 {
    let (px, py) = (0usize, 1usize);
    let t = pos.t;
    let nfree = t - 2;
    let mut rng = Rng::new(seed);
    let mut fixed = vec![0u32; nfree];
    for j in 0..nfree {
        if j != px && j != py {
            fixed[j] = rng.field();
        }
    }

    let res_at = |xv: u32| -> u32 {
        let mut assign: Vec<pl::Poly> = Vec::with_capacity(nfree);
        for j in 0..nfree {
            if j == px {
                assign.push(vec![xv]);
            } else if j == py {
                assign.push(vec![0, 1]);
            } else {
                assign.push(vec![fixed[j]]);
            }
        }
        let s = perm_plus_linear_poly(pos, &build_state(&assign, t));
        pl::resultant(&pl::trim(&s[0]), &pl::trim(&s[1]))
    };

    let fwd = 3u64.pow((pos.r_f + pos.r_p) as u32);
    let hard_cap = (2 * fwd * fwd + 1).min(PU64 - 1) as usize;
    let guard = 12u32;
    let mut n = 16usize;
    loop {
        n = n.min(hard_cap);
        let xs: Vec<u32> = (1..=n as u32).collect();
        let ys: Vec<u32> = xs.iter().map(|&x| res_at(x)).collect();
        let r_poly = pl::interpolate(&xs, &ys);
        let stable = (n as u32 + 1..=n as u32 + guard).all(|x| pl::eval(&r_poly, x) == res_at(x));
        if n >= hard_cap || stable {
            return pl::deg(&r_poly);
        }
        n *= 2;
    }
}
