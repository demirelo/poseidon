//! Verifier-exact KoalaBear / Poseidon1 native core.
//!
//! Mirrors `poseidon_attack/poseidon1.py` + `poseidon_attack/verifiers.py`
//! (themselves cross-checked against khovratovich/poseidon-tools @ 60075da7).
//! Arithmetic is Montgomery form over the whole permutation; round constants and
//! the MDS are converted to Montgomery once at instance-build time.
//!
//! Load-bearing instance facts:
//!   * p = 2^31 - 2^24 + 1 = 2130706433, alpha = 3, t = 16
//!   * canonical MDS is the Cauchy/Toeplitz matrix M[i][j] = (i - t - j)^{-1}
//!     (NOT the Plonky3 circulant; that is golden-test cross-check data only)
//!   * round constants come from the Grain LFSR (eprint 2019/458)
//!   * collision instance is FULL STRENGTH rf=8, rp=20

pub mod cico;
pub mod poly;
pub mod zerotest;

// ===========================================================================
// KoalaBear field, Montgomery arithmetic (R = 2^32)
// ===========================================================================

pub const P: u32 = 2130706433; // 2^31 - 2^24 + 1
const PU64: u64 = P as u64;

/// -p^{-1} mod 2^32 via Newton's iteration (doubles correct bits each step).
const fn compute_pprime(p: u32) -> u32 {
    let mut inv: u32 = 1;
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(p.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}
const PPRIME: u32 = compute_pprime(P);

/// Montgomery multiplication: REDC(a*b). Inputs/outputs are < P (in [0,P)).
#[inline(always)]
pub fn mont_mul(a: u32, b: u32) -> u32 {
    let t = (a as u64) * (b as u64); // < 2^62
    let m = (t as u32).wrapping_mul(PPRIME);
    let u = ((t + (m as u64) * PU64) >> 32) as u32; // < 2P, fits u32
    if u >= P {
        u - P
    } else {
        u
    }
}

#[inline(always)]
pub fn mont_add(a: u32, b: u32) -> u32 {
    let s = a + b; // < 2P < 2^32
    if s >= P {
        s - P
    } else {
        s
    }
}

#[inline(always)]
pub fn mont_sub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + P - b
    }
}

#[inline(always)]
pub fn mont_cube(x: u32) -> u32 {
    mont_mul(mont_mul(x, x), x)
}

/// a -> a*R mod P (Montgomery form). a may be any u32; reduced mod P first.
#[inline(always)]
pub fn to_mont(a: u32) -> u32 {
    (((a as u64 % PU64) << 32) % PU64) as u32
}

/// a*R^{-1} mod P (back to normal form).
#[inline(always)]
pub fn from_mont(a: u32) -> u32 {
    mont_mul(a, 1)
}

/// Normal-domain modular exponentiation (instance setup only, not hot path).
fn modpow(mut a: u64, mut e: u64) -> u32 {
    a %= PU64;
    let mut r: u64 = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = r * a % PU64;
        }
        a = a * a % PU64;
        e >>= 1;
    }
    r as u32
}

/// Modular inverse in the normal domain (P is prime).
fn modinv(a: u32) -> u32 {
    modpow(a as u64, (P - 2) as u64)
}

// ===========================================================================
// Grain LFSR round constants (eprint 2019/458) -- mirrors grain_lfsr.py
// ===========================================================================

pub struct GrainLfsr {
    state: [u8; 80],
    prime_bit_len: u32,
}

impl GrainLfsr {
    pub fn new(prime_bit_len: u32, alpha: u32, t: u32, r_f: u32, r_p: u32) -> Self {
        let mut state = [0u8; 80];
        state[0] = 1; // field type "10" = prime field
        state[1] = 0;
        state[2] = 0; // S-box type (alpha != -1)
        set_bits(&mut state, 3, alpha, 5);
        set_bits(&mut state, 8, prime_bit_len, 10);
        set_bits(&mut state, 18, t, 10);
        set_bits(&mut state, 28, r_f, 10);
        set_bits(&mut state, 38, r_p, 10);
        for i in 48..80 {
            state[i] = 1;
        }
        let mut g = GrainLfsr {
            state,
            prime_bit_len,
        };
        for _ in 0..160 {
            g.clock();
        }
        g
    }

    #[inline]
    fn clock(&mut self) -> u8 {
        let new = self.state[0]
            ^ self.state[13]
            ^ self.state[23]
            ^ self.state[38]
            ^ self.state[51]
            ^ self.state[62];
        let out = self.state[0];
        for i in 0..79 {
            self.state[i] = self.state[i + 1];
        }
        self.state[79] = new;
        out
    }

    /// Next field element by rejection sampling (normal domain).
    pub fn field_element(&mut self) -> u32 {
        loop {
            let mut value: u64 = 0;
            for _ in 0..self.prime_bit_len {
                value = (value << 1) | (self.clock() as u64);
            }
            if value < PU64 {
                return value as u32;
            }
        }
    }
}

fn set_bits(state: &mut [u8; 80], off: usize, value: u32, n: u32) {
    for i in 0..n {
        state[off + i as usize] = ((value >> (n - 1 - i)) & 1) as u8;
    }
}

/// Flat (r_f + r_p) * t Grain round constants, normal domain.
pub fn grain_round_constants(alpha: u32, t: u32, r_f: u32, r_p: u32) -> Vec<u32> {
    let prime_bit_len = 32 - (P).leading_zeros(); // 31 for KoalaBear
    let mut lfsr = GrainLfsr::new(prime_bit_len, alpha, t, r_f, r_p);
    let n = ((r_f + r_p) * t) as usize;
    (0..n).map(|_| lfsr.field_element()).collect()
}

// ===========================================================================
// MDS matrices (normal domain construction -> Montgomery)
// ===========================================================================

/// Canonical bounty MDS: M[i][j] = (i - t - j)^{-1} mod p. Toeplitz in (i-j).
pub fn cauchy_mds(t: usize) -> Vec<Vec<u32>> {
    (0..t)
        .map(|i| {
            (0..t)
                .map(|j| {
                    let diff =
                        (((i as i64) - (t as i64) - (j as i64)).rem_euclid(PU64 as i64)) as u32;
                    modinv(diff)
                })
                .collect()
        })
        .collect()
}

/// Plonky3-style circulant: M[i][j] = first_row[(j - i) mod t]. (golden tests only)
pub fn circulant_mds(first_row: &[u32]) -> Vec<Vec<u32>> {
    let t = first_row.len();
    (0..t)
        .map(|i| (0..t).map(|j| first_row[(j + t - i % t) % t] % P).collect())
        .collect()
}

// ===========================================================================
// Poseidon1 permutation + compression hash
// ===========================================================================

pub struct Poseidon1 {
    pub t: usize,
    pub r_f: usize,
    pub r_p: usize,
    pub alpha: u32,
    rc: Vec<[u32; 16]>,       // Montgomery-domain round constants, t per round
    mds: Vec<[u32; 16]>,      // Montgomery-domain MDS rows
    pub rc_n: Vec<Vec<u32>>,  // normal-domain round constants (for symbolic perm)
    pub mds_n: Vec<Vec<u32>>, // normal-domain MDS rows (for symbolic perm)
}

impl Poseidon1 {
    /// Canonical bounty instance: Cauchy MDS + Grain constants.
    pub fn new_canonical(alpha: u32, t: usize, r_f: usize, r_p: usize) -> Self {
        assert_eq!(t, 16, "this build is specialised to t=16");
        let flat = grain_round_constants(alpha, t as u32, r_f as u32, r_p as u32);
        let mds = cauchy_mds(t);
        Self::from_parts(alpha, t, r_f, r_p, &flat, &mds)
    }

    /// Build from explicit normal-domain round constants + MDS (golden tests).
    pub fn from_parts(
        alpha: u32,
        t: usize,
        r_f: usize,
        r_p: usize,
        flat_rc: &[u32],
        mds_normal: &[Vec<u32>],
    ) -> Self {
        assert_eq!(t, 16);
        let total = r_f + r_p;
        assert_eq!(flat_rc.len(), total * t);
        let mut rc = Vec::with_capacity(total);
        let mut rc_n = Vec::with_capacity(total);
        for r in 0..total {
            let mut row = [0u32; 16];
            let mut row_n = vec![0u32; t];
            for i in 0..t {
                row[i] = to_mont(flat_rc[r * t + i]);
                row_n[i] = flat_rc[r * t + i] % P;
            }
            rc.push(row);
            rc_n.push(row_n);
        }
        let mut mds = Vec::with_capacity(t);
        let mut mds_n = Vec::with_capacity(t);
        for i in 0..t {
            let mut row = [0u32; 16];
            let mut row_n = vec![0u32; t];
            for j in 0..t {
                row[j] = to_mont(mds_normal[i][j]);
                row_n[j] = mds_normal[i][j] % P;
            }
            mds.push(row);
            mds_n.push(row_n);
        }
        Poseidon1 {
            t,
            r_f,
            r_p,
            alpha,
            rc,
            mds,
            rc_n,
            mds_n,
        }
    }

    #[inline(always)]
    fn apply_mds(&self, s: &[u32; 16]) -> [u32; 16] {
        // 16 independent mont_mul products summed in a u64 (each < P, so the sum
        // is < 16P < 2^35), then one reduction. Benchmarks faster than a chained
        // mont_add accumulator, which serialises into a dependency chain.
        let mut out = [0u32; 16];
        for i in 0..16 {
            let row = &self.mds[i];
            let mut acc: u64 = 0;
            for j in 0..16 {
                acc += mont_mul(row[j], s[j]) as u64;
            }
            out[i] = (acc % PU64) as u32;
        }
        out
    }

    /// Permutation on a Montgomery-domain state, optional initial linear layer.
    pub fn perm_mont(&self, state: &[u32; 16], initial_linear: bool) -> [u32; 16] {
        let mut s = *state;
        if initial_linear {
            s = self.apply_mds(&s);
        }
        let half = self.r_f / 2;
        let mut idx = 0usize;
        for _ in 0..half {
            for i in 0..16 {
                s[i] = mont_cube(mont_add(s[i], self.rc[idx][i]));
            }
            s = self.apply_mds(&s);
            idx += 1;
        }
        for _ in 0..self.r_p {
            for i in 0..16 {
                s[i] = mont_add(s[i], self.rc[idx][i]);
            }
            s[0] = mont_cube(s[0]);
            s = self.apply_mds(&s);
            idx += 1;
        }
        for _ in 0..half {
            for i in 0..16 {
                s[i] = mont_cube(mont_add(s[i], self.rc[idx][i]));
            }
            s = self.apply_mds(&s);
            idx += 1;
        }
        s
    }

    /// Normal-domain permutation (converts in/out).
    pub fn permutation(&self, state: &[u32; 16]) -> [u32; 16] {
        let mut sm = [0u32; 16];
        for i in 0..16 {
            sm[i] = to_mont(state[i]);
        }
        let o = self.perm_mont(&sm, false);
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = from_mont(o[i]);
        }
        out
    }

    pub fn permutation_plus_linear(&self, state: &[u32; 16]) -> [u32; 16] {
        let mut sm = [0u32; 16];
        for i in 0..16 {
            sm[i] = to_mont(state[i]);
        }
        let o = self.perm_mont(&sm, true);
        let mut out = [0u32; 16];
        for i in 0..16 {
            out[i] = from_mont(o[i]);
        }
        out
    }

    /// compression_mode_hash: feedforward out[i] = perm(in)[i] + in[i].
    pub fn compression_mode_hash(&self, inputs: &[u32; 16], out_len: usize) -> Vec<u32> {
        let mut sm = [0u32; 16];
        for i in 0..16 {
            sm[i] = to_mont(inputs[i]);
        }
        let o = self.perm_mont(&sm, false);
        (0..out_len)
            .map(|i| from_mont(mont_add(o[i], sm[i])))
            .collect()
    }
}

// ===========================================================================
// Hot-path predicates (CICO-2, partial collision)
// ===========================================================================

pub const CICO_C1: u32 = 0xC09DE4;
pub const CICO_C2: u32 = 0xEE6282;
pub const COLLISION_SEED: u32 = 0xC09DE4;

/// True iff permutation_plus_linear([C1,C2] + free)[:2] == [0,0]. free has len 14.
pub fn verify_cico(pos: &Poseidon1, free: &[u32]) -> bool {
    assert_eq!(free.len(), 14);
    let mut st = [0u32; 16];
    st[0] = CICO_C1;
    st[1] = CICO_C2;
    for i in 0..14 {
        st[2 + i] = free[i] % P;
    }
    let out = pos.permutation_plus_linear(&st);
    out[0] == 0 && out[1] == 0
}

/// The 2 constrained CICO output words (target all-zero). Profiling aid.
pub fn cico_residual(pos: &Poseidon1, free: &[u32]) -> (u32, u32) {
    assert_eq!(free.len(), 14);
    let mut st = [0u32; 16];
    st[0] = CICO_C1;
    st[1] = CICO_C2;
    for i in 0..14 {
        st[2 + i] = free[i] % P;
    }
    let out = pos.permutation_plus_linear(&st);
    (out[0], out[1])
}

/// True iff x != y and H(seed,x)[:t_collide] == H(seed,y)[:t_collide].
/// x,y have length 15 (= t_perm - 1).
pub fn verify_collision(pos: &Poseidon1, x: &[u32], y: &[u32], t_collide: usize) -> bool {
    assert_eq!(x.len(), 15);
    assert_eq!(y.len(), 15);
    let xn: Vec<u32> = x.iter().map(|v| v % P).collect();
    let yn: Vec<u32> = y.iter().map(|v| v % P).collect();
    if xn == yn {
        return false;
    }
    let hx = collision_hash(pos, &xn);
    let hy = collision_hash(pos, &yn);
    hx[..t_collide] == hy[..t_collide]
}

fn collision_hash(pos: &Poseidon1, x: &[u32]) -> Vec<u32> {
    let mut inp = [0u32; 16];
    inp[0] = COLLISION_SEED;
    for i in 0..15 {
        inp[1 + i] = x[i];
    }
    pos.compression_mode_hash(&inp, 16)
}
