"""
Poseidon1 permutation over KoalaBear -- independent reimplementation.

Round schedule (matches reference/poseidon-tools/poseidon/poseidon.py):

    [optional initial MDS]      <- permutation_plus_linear only (CICO)
    RF/2 full rounds            <- AddRC, x^3 on ALL t lanes, MDS
    RP   partial rounds         <- AddRC, x^3 on lane 0 ONLY, MDS
    RF/2 full rounds            <- AddRC, x^3 on ALL t lanes, MDS

Round constants are consumed sequentially, t per round, from the Grain LFSR
(or supplied explicitly, e.g. the Plonky3 vectors used for golden tests).

The default instance uses the Cauchy MDS  M[i][j] = (i - t - j)^{-1} mod p,
the matrix shipped as the default test vector by the vendored poseidon-tools
verifier.  The challenge admits any MDS satisfying the no-invariant-subspace-trail
conditions; the Plonky3 circulant is one admissible example and is used here for
golden-vector cross-checks.
"""

from .constants import P, ALPHA, T


# ---------------------------------------------------------------------------
# Grain LFSR round-constant generator (Poseidon spec, eprint 2019/458).
# Mirrors reference/poseidon-tools/poseidon/grain_lfsr.py.
# ---------------------------------------------------------------------------

class _GrainLFSR:
    def __init__(self, prime_bit_len: int, alpha: int, t: int, r_f: int, r_p: int):
        self.prime_bit_len = prime_bit_len
        self.state = [0] * 80
        # --- initial 80-bit state from instance parameters ---
        self.state[0], self.state[1] = 1, 0          # field type "10" = prime field
        self.state[2] = 1 if alpha == -1 else 0      # S-box type bit
        self._set(3, 0 if alpha == -1 else alpha, 5)  # alpha, 5 bits MSB-first
        self._set(8, prime_bit_len, 10)
        self._set(18, t, 10)
        self._set(28, r_f, 10)
        self._set(38, r_p, 10)
        for i in range(48, 80):
            self.state[i] = 1
        # discard first 160 outputs
        for _ in range(160):
            self._clock()

    def _set(self, off: int, value: int, n: int) -> None:
        for i in range(n):
            self.state[off + i] = (value >> (n - 1 - i)) & 1

    def _clock(self) -> int:
        # feedback taps for x^80 + x^62 + x^51 + x^38 + x^23 + x^13 + 1
        new = (self.state[0] ^ self.state[13] ^ self.state[23]
               ^ self.state[38] ^ self.state[51] ^ self.state[62])
        out = self.state[0]
        self.state = self.state[1:] + [new]
        return out

    def field_element(self, prime: int) -> int:
        while True:                                  # rejection sampling
            value = 0
            for _ in range(self.prime_bit_len):
                value = (value << 1) | self._clock()
            if value < prime:
                return value


def grain_round_constants(prime: int, alpha: int, t: int, r_f: int, r_p: int):
    """Flat list of (r_f + r_p) * t round constants from the Grain LFSR."""
    lfsr = _GrainLFSR(prime.bit_length(), alpha, t, r_f, r_p)
    return [lfsr.field_element(prime) for _ in range((r_f + r_p) * t)]


# ---------------------------------------------------------------------------
# MDS matrices
# ---------------------------------------------------------------------------

def cauchy_mds(t: int = T, p: int = P):
    """Default vendored MDS: M[i][j] = (i - t - j)^{-1} mod p."""
    return [[pow((i - t - j) % p, -1, p) for j in range(t)] for i in range(t)]


def circulant_mds(first_row, p: int = P):
    """Plonky3-style circulant: M[i][j] = first_row[(j - i) mod t]. (golden tests only)"""
    t = len(first_row)
    return [[first_row[(j - i) % t] % p for j in range(t)] for i in range(t)]


def _apply_mds(state, mds, p):
    t = len(state)
    return [sum(mds[i][j] * state[j] for j in range(t)) % p for i in range(t)]


# ---------------------------------------------------------------------------
# Poseidon1 permutation + hash modes
# ---------------------------------------------------------------------------

class Poseidon1:
    def __init__(self, prime=P, alpha=ALPHA, t=T, r_f=8, r_p=20,
                 mds=None, round_constants=None):
        if r_f % 2 != 0:
            raise ValueError("r_f must be even")
        self.p, self.alpha, self.t, self.r_f, self.r_p = prime, alpha, t, r_f, r_p
        self.rate = t - 1
        total = r_f + r_p
        if round_constants is None:
            flat = grain_round_constants(prime, alpha, t, r_f, r_p)
        else:
            flat = list(round_constants)
            if len(flat) != total * t:
                raise ValueError(f"need {total*t} round constants, got {len(flat)}")
        self.rc = [flat[i * t:(i + 1) * t] for i in range(total)]
        self.mds = mds if mds is not None else cauchy_mds(t, prime)

    def _sbox(self, x):
        return pow(x, self.alpha, self.p) if self.alpha != -1 else pow(x, -1, self.p)

    def _full_round(self, s, rc):
        s = [(s[i] + rc[i]) % self.p for i in range(self.t)]
        s = [self._sbox(v) for v in s]
        return _apply_mds(s, self.mds, self.p)

    def _partial_round(self, s, rc):
        s = [(s[i] + rc[i]) % self.p for i in range(self.t)]
        s[0] = self._sbox(s[0])
        return _apply_mds(s, self.mds, self.p)

    def _perm(self, state, initial_linear: bool):
        if len(state) != self.t:
            raise ValueError(f"state must have {self.t} elements")
        s = [v % self.p for v in state]
        if initial_linear:
            s = _apply_mds(s, self.mds, self.p)
        half = self.r_f // 2
        idx = 0
        for _ in range(half):
            s = self._full_round(s, self.rc[idx]); idx += 1
        for _ in range(self.r_p):
            s = self._partial_round(s, self.rc[idx]); idx += 1
        for _ in range(half):
            s = self._full_round(s, self.rc[idx]); idx += 1
        return s

    def permutation(self, state):
        return self._perm(state, initial_linear=False)

    def permutation_plus_linear(self, state):
        """CICO variant: an extra MDS layer before the first round."""
        return self._perm(state, initial_linear=True)

    def compression_mode_hash(self, inputs, out_length):
        """len(inputs) must equal t; feedforward output[i] = perm(in)[i] + in[i]."""
        if len(inputs) != self.t:
            raise ValueError(f"compression input must have length {self.t}")
        s = [v % self.p for v in inputs]
        s = self.permutation(s)
        for i in range(len(inputs)):
            s[i] = (s[i] + inputs[i]) % self.p
        return s[:out_length]

    def sponge_hash(self, inputs, out_length):
        """Capacity holds the input length; absorb rate=t-1 at a time."""
        if not inputs:
            raise ValueError("inputs must be non-empty")
        s = [0] * self.t
        s[self.rate] = len(inputs) % self.p
        for start in range(0, len(inputs), self.rate):
            block = inputs[start:start + self.rate]
            for i, v in enumerate(block):
                s[i] = (s[i] + v) % self.p
            s = self.permutation(s)
        return s[:out_length]
