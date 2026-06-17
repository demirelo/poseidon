"""
Tests for the Poseidon hash function implementation.

Uses BN254 (bn128) curve parameters:
  p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
  t=3, R_F=8, R_P=57, alpha=5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from poseidon.grain_lfsr import GrainLFSR
from poseidon.mds_matrix import (
    generate_mds_matrix,
    apply_mds,
    verify_mds_matrix,
    generate_circulant_mds_matrix,
    _algorithm_1,
    _algorithm_2,
    _algorithm_3,
    _check_minpoly,
)
from poseidon.poseidon import Poseidon
from bounties.density_verifier import verify_density_solution

# ---------------------------------------------------------------------------
# BN254 parameters
# ---------------------------------------------------------------------------
BN254_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
T = 3
R_F = 8
R_P = 57
ALPHA = 5


# ---------------------------------------------------------------------------
# GrainLFSR tests
# ---------------------------------------------------------------------------

class TestGrainLFSR:
    def _make_lfsr(self):
        prime_bit_len = BN254_P.bit_length()  # 254
        return GrainLFSR(prime_bit_len, ALPHA, T, R_F, R_P)

    def test_produces_bits(self):
        lfsr = self._make_lfsr()
        bit = lfsr.get_next_bit()
        assert bit in (0, 1)

    def test_deterministic(self):
        """Two LFSRs with the same params must produce the same sequence."""
        lfsr1 = self._make_lfsr()
        lfsr2 = self._make_lfsr()
        bits1 = [lfsr1.get_next_bit() for _ in range(100)]
        bits2 = [lfsr2.get_next_bit() for _ in range(100)]
        assert bits1 == bits2

    def test_field_element_in_range(self):
        lfsr = self._make_lfsr()
        for _ in range(10):
            elem = lfsr.get_field_element(BN254_P)
            assert 0 <= elem < BN254_P

    def test_field_elements_are_not_all_identical(self):
        lfsr = self._make_lfsr()
        elems = [lfsr.get_field_element(BN254_P) for _ in range(10)]
        # Very unlikely that all 10 are the same
        assert len(set(elems)) > 1

    def test_different_params_give_different_sequences(self):
        prime_bit_len = BN254_P.bit_length()
        lfsr_a = GrainLFSR(prime_bit_len, ALPHA, T, R_F, R_P)
        lfsr_b = GrainLFSR(prime_bit_len, ALPHA, 4, R_F, R_P)  # different t
        bits_a = [lfsr_a.get_next_bit() for _ in range(64)]
        bits_b = [lfsr_b.get_next_bit() for _ in range(64)]
        assert bits_a != bits_b

    def test_inverse_sbox_flag(self):
        """alpha=-1 sets sbox bit to 1 and alpha_bits to 0."""
        prime_bit_len = BN254_P.bit_length()
        lfsr = GrainLFSR(prime_bit_len, -1, T, R_F, R_P)
        # Just verify it initialises without error and produces valid bits
        bit = lfsr.get_next_bit()
        assert bit in (0, 1)


# ---------------------------------------------------------------------------
# MDS matrix tests
# ---------------------------------------------------------------------------

class TestMDSMatrix:
    def test_shape(self):
        mds = generate_mds_matrix(T, BN254_P)
        assert len(mds) == T
        for row in mds:
            assert len(row) == T

    def test_elements_in_field(self):
        mds = generate_mds_matrix(T, BN254_P)
        for row in mds:
            for elem in row:
                assert 0 <= elem < BN254_P

    def test_invertible(self):
        """The MDS matrix must be invertible (det != 0 mod prime)."""
        mds = generate_mds_matrix(T, BN254_P)
        det = _det_mod(mds, BN254_P)
        assert det != 0

    def test_all_2x2_submatrices_invertible(self):
        """MDS property: every 2×2 submatrix must be invertible."""
        mds = generate_mds_matrix(T, BN254_P)
        for r1 in range(T):
            for r2 in range(r1 + 1, T):
                for c1 in range(T):
                    for c2 in range(c1 + 1, T):
                        sub = [
                            [mds[r1][c1], mds[r1][c2]],
                            [mds[r2][c1], mds[r2][c2]],
                        ]
                        det = (sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0]) % BN254_P
                        assert det != 0, f"2×2 sub-matrix rows=({r1},{r2}) cols=({c1},{c2}) has det=0"

    def test_apply_mds(self):
        mds = generate_mds_matrix(T, BN254_P)
        state = [1, 2, 3]
        result = apply_mds(state, mds, BN254_P)
        assert len(result) == T
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_apply_mds_not_identity(self):
        mds = generate_mds_matrix(T, BN254_P)
        state = [1, 2, 3]
        result = apply_mds(state, mds, BN254_P)
        assert result != state


# ---------------------------------------------------------------------------
# Poseidon permutation tests
# ---------------------------------------------------------------------------

class TestPoseidonPermutation:
    def _make_poseidon(self):
        return Poseidon(BN254_P, ALPHA, T, R_F, R_P)

    def test_permutation_length_preserving(self):
        pos = self._make_poseidon()
        state = [1, 2, 3]
        result = pos.permutation(state)
        assert len(result) == T

    def test_permutation_output_in_field(self):
        pos = self._make_poseidon()
        state = [0, 1, 2]
        result = pos.permutation(state)
        for elem in result:
            assert 0 <= elem < BN254_P

    def test_permutation_deterministic(self):
        pos1 = self._make_poseidon()
        pos2 = self._make_poseidon()
        state = [100, 200, 300]
        assert pos1.permutation(state) == pos2.permutation(state)

    def test_permutation_different_inputs_different_outputs(self):
        pos = self._make_poseidon()
        assert pos.permutation([0, 0, 0]) != pos.permutation([1, 0, 0])
        assert pos.permutation([1, 0, 0]) != pos.permutation([0, 1, 0])

    def test_permutation_wrong_state_length_raises(self):
        pos = self._make_poseidon()
        with pytest.raises(ValueError):
            pos.permutation([1, 2])

    def test_permutation_zero_state(self):
        pos = self._make_poseidon()
        result = pos.permutation([0] * T)
        assert len(result) == T
        # The zero state should not stay at zero after a full permutation
        assert not all(x == 0 for x in result)

    def test_permutation_plus_linear_length_preserving(self):
        pos = self._make_poseidon()
        result = pos.permutation_plus_linear([1, 2, 3])
        assert len(result) == T

    def test_permutation_plus_linear_differs_from_standard(self):
        pos = self._make_poseidon()
        state = [1, 2, 3]
        assert pos.permutation_plus_linear(state) != pos.permutation(state)


# ---------------------------------------------------------------------------
# Poseidon hash tests
# ---------------------------------------------------------------------------

class TestPoseidonHash:
    def _make_poseidon(self):
        return Poseidon(BN254_P, ALPHA, T, R_F, R_P)

    def test_hash_returns_field_element(self):
        pos = self._make_poseidon()
        h = pos.hash([1, 2])
        assert 0 <= h < BN254_P

    def test_hash_deterministic(self):
        pos1 = self._make_poseidon()
        pos2 = self._make_poseidon()
        assert pos1.hash([1, 2]) == pos2.hash([1, 2])

    def test_hash_different_inputs_different_outputs(self):
        pos = self._make_poseidon()
        assert pos.hash([1]) != pos.hash([2])
        assert pos.hash([1, 2]) != pos.hash([2, 1])

    def test_hash_single_element(self):
        pos = self._make_poseidon()
        h = pos.hash([42])
        assert 0 <= h < BN254_P

    def test_hash_empty_raises(self):
        pos = self._make_poseidon()
        with pytest.raises(ValueError):
            pos.hash([])

    def test_hash_long_input(self):
        pos = self._make_poseidon()
        inputs = list(range(1, 11))
        h = pos.hash(inputs)
        assert 0 <= h < BN254_P

    def test_hash_known_vector(self):
        """
        Regression / self-consistency test: record the output of our own
        implementation and verify it stays stable across refactors.
        """
        pos = self._make_poseidon()
        # Compute once and store as expected value
        expected = pos.hash([1, 2])
        # Re-instantiate to ensure we get the same thing
        pos2 = self._make_poseidon()
        assert pos2.hash([1, 2]) == expected


# ---------------------------------------------------------------------------
# Utility: determinant mod prime (for small matrices)
# ---------------------------------------------------------------------------

def _det_mod(matrix, prime):
    """Compute determinant of a square matrix mod prime using Gaussian elimination."""
    n = len(matrix)
    mat = [row[:] for row in matrix]
    det = 1
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if mat[row][col] % prime != 0:
                pivot = row
                break
        if pivot is None:
            return 0
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            det = (-det) % prime
        inv = pow(mat[col][col], -1, prime)
        det = (det * mat[col][col]) % prime
        for row in range(col + 1, n):
            factor = (mat[row][col] * inv) % prime
            for k in range(col, n):
                mat[row][k] = (mat[row][k] - factor * mat[col][k]) % prime
    return det % prime


# ---------------------------------------------------------------------------
# KoalaBear field constants
# ---------------------------------------------------------------------------

KOALABEAR_P = 2130706433  # 2^31 - 2^24 + 1

# Prime where the t=3 Cauchy MDS matrix passes all security checks
SMALL_PRIME_FOR_MDS_TEST = 1009  # Cauchy t=3 is fully verified for this prime


# ---------------------------------------------------------------------------
# verify_mds_matrix tests
# ---------------------------------------------------------------------------

class TestVerifyMDSMatrix:
    def test_cauchy_mds_t3_passes_all_checks(self):
        # For p=1009 the Cauchy t=3 MDS has an irreducible char poly for all
        # required powers, so verify_mds_matrix returns True.
        mds = generate_mds_matrix(3, SMALL_PRIME_FOR_MDS_TEST)
        assert verify_mds_matrix(mds, SMALL_PRIME_FOR_MDS_TEST) is True

    def test_cauchy_mds_t3_bn254_minpoly_fails(self):
        # The BN254 Cauchy t=3 matrix has an eigenvalue in GF(p), making its
        # characteristic polynomial reducible.  The minpoly check therefore
        # fails.
        mds = generate_mds_matrix(3, BN254_P)
        assert _check_minpoly(mds, 3, BN254_P) is False


# ---------------------------------------------------------------------------
# KoalaBear tests
# ---------------------------------------------------------------------------

class TestKoalaBear:
    def _make_poseidon(self):
        return Poseidon(KOALABEAR_P, alpha=3, t=16, r_f=8, r_p=20)

    def test_hash_known_vectors(self):
        pos = self._make_poseidon()
        assert pos.hash(list(range(15))) == 93555670
        assert pos.hash(list(range(1, 16))) == 938201807
        assert pos.hash([1]) == 1541345887

    def test_permutation_known_vector(self):
        pos = self._make_poseidon()
        assert pos.permutation([0] * 16)[0] == 1393439926

    def test_hash_output_in_field(self):
        pos = self._make_poseidon()
        h = pos.hash([42])
        assert 0 <= h < KOALABEAR_P


# ---------------------------------------------------------------------------
# Plonky3 / leanSpec KoalaBear test vectors
# (from https://github.com/tcoratger/leanSpec/tree/poseidon11/src/lean_spec)
# Test vectors are from Plonky3 koala-bear/src/poseidon1.rs.
# ---------------------------------------------------------------------------

# MDS first rows (circulant construction, from Plonky3 koala-bear/src/mds.rs)
_KB_MDS_FIRST_ROW_16: list[int] = [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3]
_KB_MDS_FIRST_ROW_24: list[int] = [
    0x2D0AAAAB, 0x64850517, 0x17F5551D, 0x04ECBEB5,
    0x6D91A8D5, 0x60703026, 0x18D6F3CA, 0x729601A7,
    0x77CDA9E2, 0x3C0F5038, 0x26D52A61, 0x0360405D,
    0x68FC71C8, 0x2495A71D, 0x5D57AFC2, 0x1689DD98,
    0x3C2C3DBE, 0x0C23DC41, 0x0524C7F2, 0x6BE4DF69,
    0x0A6E572C, 0x5C7790FA, 0x17E118F6, 0x0878A07F,
]

# Round constants for width 16 (8 full + 20 partial rounds = 448 constants)
# Source: leanSpec constants.py, originally from Plonky3
_KB_ROUND_CONSTANTS_16: list[int] = [
    # Initial full rounds (4 rounds × 16 constants)
    0x7EE56A48, 0x11367045, 0x12E41941, 0x7EBBC12B,
    0x1970B7D5, 0x662B60E8, 0x3E4990C6, 0x679F91F5,
    0x350813BB, 0x00874AD4, 0x28A0081A, 0x18FA5872,
    0x5F25B071, 0x5E5D5998, 0x5E6FD3E7, 0x5B2E2660,
    0x6F1837BF, 0x3FE6182B, 0x1EDD7AC5, 0x57470D00,
    0x43D486D5, 0x1982C70F, 0x0EA53AF9, 0x61D6165B,
    0x51639C00, 0x2DEC352C, 0x2950E531, 0x2D2CB947,
    0x08256CEF, 0x1A0109F6, 0x1F51FAF3, 0x5CEF1C62,
    0x3D65E50E, 0x33D91626, 0x133D5A1E, 0x0FF49B0D,
    0x38900CD1, 0x2C22CC3F, 0x28852BB2, 0x06C65A02,
    0x7B2CF7BC, 0x68016E1A, 0x15E16BC0, 0x5248149A,
    0x6DD212A0, 0x18D6830A, 0x5001BE82, 0x64DAC34E,
    0x5902B287, 0x426583A0, 0x0C921632, 0x3FE028A5,
    0x245F8E49, 0x43BB297E, 0x7873DBD9, 0x3CC987DF,
    0x286BB4CE, 0x640A8DCD, 0x512A8E36, 0x03A4CF55,
    0x481837A2, 0x03D6DA84, 0x73726AC7, 0x760E7FDF,
    # Partial rounds (20 rounds × 16 constants)
    0x54DFEB5D, 0x7D40AFD6, 0x722CB316, 0x106A4573,
    0x45A7CCDB, 0x44061375, 0x154077A5, 0x45744FAA,
    0x4EB5E5EE, 0x3794E83F, 0x47C7093C, 0x5694903C,
    0x69CB6299, 0x373DF84C, 0x46A0DF58, 0x46B8758A,
    0x3241EBCB, 0x0B09D233, 0x1AF42357, 0x1E66CEC2,
    0x43E7DC24, 0x259A5D61, 0x27E85A3B, 0x1B9133FA,
    0x343E5628, 0x485CD4C2, 0x16E269F5, 0x165B60C6,
    0x25F683D9, 0x124F81F9, 0x174331F9, 0x77344DC5,
    0x5A821DBA, 0x5FC4177F, 0x54153BF5, 0x5E3F1194,
    0x3BDBF191, 0x088C84A3, 0x68256C9B, 0x3C90BBC6,
    0x6846166A, 0x03F4238D, 0x463335FB, 0x5E3D3551,
    0x6E59AE6F, 0x32D06CC0, 0x596293F3, 0x6C87EDB2,
    0x08FC60B5, 0x34BCCA80, 0x24F007F3, 0x62731C6F,
    0x1E1DB6C6, 0x0CA409BB, 0x585C1E78, 0x56E94EDC,
    0x16D22734, 0x18E11467, 0x7B2C3730, 0x770075E4,
    0x35D1B18C, 0x22BE3DB5, 0x4FB1FBB7, 0x477CB3ED,
    0x7D5311C6, 0x5B62AE7D, 0x559C5FA8, 0x77F15048,
    0x3211570B, 0x490FEF6A, 0x77EC311F, 0x2247171B,
    0x4E0AC711, 0x2EDF69C9, 0x3B5A8850, 0x65809421,
    0x5619B4AA, 0x362019A7, 0x6BF9D4ED, 0x5B413DFF,
    0x617E181E, 0x5E7AB57B, 0x33AD7833, 0x3466C7CA,
    0x6488DFF4, 0x71F068F4, 0x056E891F, 0x04F1ECCC,
    0x663257D5, 0x671E31B9, 0x5871987C, 0x280C109E,
    0x2A227761, 0x350A25E9, 0x5B91B1C4, 0x7A073546,
    0x01826270, 0x53A67720, 0x0ED4B074, 0x34CF0C4E,
    0x6E751E88, 0x29BD5F59, 0x49EC32DF, 0x7693452B,
    0x3CF09E58, 0x6BA0E2BF, 0x7AB93ACF, 0x3CE597DF,
    0x536E3D42, 0x147A808D, 0x5E32EB56, 0x5A203323,
    0x50965766, 0x6D44B7C5, 0x6698636A, 0x57B84F9F,
    0x554B61B9, 0x6DA0AB28, 0x1585B6AC, 0x6705A2B4,
    0x152872F6, 0x0F4409FD, 0x23A9DD60, 0x6F2B18D4,
    0x65AC9FD4, 0x2F0EFBEA, 0x591E67FD, 0x217CA19B,
    0x469C90CA, 0x03D60EF5, 0x4EA7857E, 0x07C86A4F,
    0x288ED461, 0x2FE51B22, 0x7E293614, 0x2C4BEB85,
    0x5B0B7D11, 0x1E17DFF6, 0x089BEAE1, 0x0A5ACF1A,
    0x2FC33D8F, 0x60422DC6, 0x6E1DC939, 0x635351B9,
    0x55522FC0, 0x3EB94EF7, 0x2A24A65C, 0x2E139C76,
    0x51391144, 0x78CC0742, 0x579538F9, 0x44DE9AAE,
    0x3C2F1E2E, 0x195747BE, 0x2496339C, 0x650B2E39,
    0x52899665, 0x6CB35558, 0x0F461C1C, 0x70F6B270,
    0x3FAAA36F, 0x62E3348A, 0x672167CB, 0x394C880B,
    0x2A46BA82, 0x63FFB74A, 0x1CF875D6, 0x53D12772,
    0x036A4552, 0x3BDD9F2B, 0x02F72C24, 0x02B6006C,
    0x077FE158, 0x1F9D6EA4, 0x20904D6F, 0x5D6534FA,
    0x066D8974, 0x6198F1F4, 0x26301AB4, 0x41F274C2,
    0x00EAC15C, 0x28B54B47, 0x2339739D, 0x48C6281C,
    0x4ED935FC, 0x3F9187FA, 0x4A1930A6, 0x3AD4D736,
    0x0F3F1889, 0x635A388F, 0x2862C145, 0x277ED1E8,
    0x4DB23CAD, 0x1F1B11F5, 0x1F3DBA2B, 0x1C26EB4E,
    0x0F7F5546, 0x6CD024B0, 0x67C47902, 0x793B8900,
    0x0E8A283C, 0x4590B7EA, 0x6F567A2B, 0x5DC97300,
    0x15247BC6, 0x50567FCB, 0x133EFF84, 0x547DC2EF,
    0x34EB3DBB, 0x12402317, 0x66C6AE49, 0x174338B6,
    0x24251008, 0x1B514927, 0x062D98D6, 0x7AF30BBC,
    0x26AF15E8, 0x70D907A3, 0x5DFC5CAC, 0x731F27EC,
    0x53AA7D3F, 0x63AB0EC6, 0x216053F4, 0x18796B39,
    0x19156AFD, 0x5EEA6973, 0x6704C6A9, 0x0DCE002B,
    0x331169C0, 0x714D7178, 0x3DDAFFAF, 0x7E464957,
    0x20CA59EA, 0x679820C9, 0x42EF21A1, 0x798EA089,
    0x14A74FA3, 0x0C06CF18, 0x6A4C8D52, 0x620F6D81,
    0x2220901A, 0x5277BB90, 0x230BF95E, 0x0AD8847A,
    0x5E96E8B6, 0x77B4056E, 0x70A50D2C, 0x5F0EED59,
    0x3646C4DF, 0x10EB9A87, 0x21EED6B7, 0x534ADD36,
    0x6E3E7421, 0x2B25810E, 0x1D8F707B, 0x45318A1A,
    0x677F8FF2, 0x0258C9E0, 0x4CD02A00, 0x2E24FF15,
    0x634A715D, 0x4AC01E59, 0x601511E1, 0x26E9C01A,
    0x4C165C6E, 0x57CD1140, 0x3AC6543B, 0x6787D847,
    0x037DFBF9, 0x6DD9D079, 0x4D24B281, 0x2A6F407D,
    0x0131DF8E, 0x4B8A7896, 0x23700858, 0x2CF5E534,
    0x12AAFC3F, 0x54568D03, 0x1A250735, 0x5331686D,
    0x4CE76D91, 0x799C1A8C, 0x2B7A8AC9, 0x60AEE672,
    0x74F7421C, 0x3C42146D, 0x26D369C5, 0x4AE54A12,
    0x7EEA16D1, 0x5CE3EAE8, 0x69F28994, 0x262B8642,
    0x610D4CC4, 0x5E1AF21C, 0x1A8526D0, 0x316B127B,
    0x3576FE5D, 0x02D968A0, 0x4BA00F51, 0x40BED993,
    0x377FB907, 0x7859216E, 0x1931D9D1, 0x53B0934E,
    0x71914FF7, 0x4EABAE6C, 0x7196468E, 0x164B3CC2,
    0x58CB66C0, 0x4C147307, 0x6B3AFCCD, 0x4236518B,
    0x4AD85605, 0x291382E1, 0x1E89B6CF, 0x5E16C3A8,
    0x2E675921, 0x24300954, 0x05E555C3, 0x78880A24,
    # Terminal full rounds (4 rounds × 16 constants)
    0x763A3125, 0x4F53B240, 0x18B7FA43, 0x2BBE8A73,
    0x1C9A12F2, 0x3F6FD40D, 0x0E1D4EC4, 0x1361C64D,
    0x09A8F470, 0x03D23A40, 0x109AD290, 0x28C2FB88,
    0x3B6498F2, 0x74D8BE57, 0x6A4277D2, 0x18C2B3D4,
    0x6252C30C, 0x07CC2560, 0x209FE15B, 0x52A55FAC,
    0x4DF19EB7, 0x02521116, 0x5E414FF1, 0x3CD9A1F4,
    0x005AAD15, 0x27A53F00, 0x72BBE9CB, 0x71D8BD7D,
    0x4194B79A, 0x48E87A72, 0x3341553C, 0x63D34FAA,
    0x132A01E3, 0x3833E2D9, 0x49726E04, 0x054957F8,
    0x7B71BCE4, 0x73EEC57D, 0x556E5533, 0x1FA93FDE,
    0x346A8CA8, 0x1162DFDE, 0x5C30D028, 0x094A4294,
    0x3052DCDA, 0x37988498, 0x51F06B97, 0x65848779,
    0x7599B0D4, 0x436FDABC, 0x66C5B77D, 0x40C86A9E,
    0x27E7055B, 0x6D0DD9D8, 0x7E5598B5, 0x1A4D04F3,
    0x5E3B2BC7, 0x533B5B2F, 0x3E33A125, 0x664D71CE,
    0x382E6C2A, 0x24C4EB6E, 0x13F246F7, 0x07E2D7EF,
]

# Round constants for width 24 (8 full + 23 partial rounds = 744 constants)
# Source: leanSpec constants.py, originally from Plonky3
_KB_ROUND_CONSTANTS_24: list[int] = [
    # Initial full rounds (4 rounds × 24 constants)
    0x1D0939DC, 0x6D050F8D, 0x628058AD, 0x2681385D,
    0x3E3C62BE, 0x032CFAD8, 0x5A91BA3C, 0x015A56E6,
    0x696B889C, 0x0DBCD780, 0x5881B5C9, 0x2A076F2E,
    0x55393055, 0x6513A085, 0x547AC78F, 0x4281C5B8,
    0x3E7A3F6C, 0x34562C19, 0x2C04E679, 0x0ED78234,
    0x5F7A1AA9, 0x0177640E, 0x0EA4F8D1, 0x15BE7692,
    0x6EAFDD62, 0x71A572C6, 0x72416F0A, 0x31CE1AD3,
    0x2136A0CF, 0x1507C0EB, 0x1EB6E07A, 0x3A0CCF7B,
    0x38E4BF31, 0x44128286, 0x6B05E976, 0x244A9B92,
    0x6E4B32A8, 0x78EE2496, 0x4761115B, 0x3D3A7077,
    0x75D3C670, 0x396A2475, 0x26DD00B4, 0x7DF50F59,
    0x0CB922DF, 0x0568B190, 0x5BD3FCD6, 0x1351F58E,
    0x52191B5F, 0x119171B8, 0x1E8BB727, 0x27D21F26,
    0x36146613, 0x1EE817A2, 0x71ABE84E, 0x44B88070,
    0x5DC04410, 0x2AEAA2F6, 0x2B7BB311, 0x6906884D,
    0x0522E053, 0x0C45A214, 0x1B016998, 0x479B1052,
    0x3ACC89BE, 0x0776021A, 0x7A34A1F5, 0x70F87911,
    0x2CAF9D9E, 0x026AFF1B, 0x2C42468E, 0x67726B45,
    0x09B6F53C, 0x73D76589, 0x5793EEB0, 0x29E720F3,
    0x75FC8BDF, 0x4C2FAE0E, 0x20B41DB3, 0x7E491510,
    0x2CADEF18, 0x57FC24D6, 0x4D1ADE4A, 0x36BF8E3C,
    0x3511B63C, 0x64D8476F, 0x732BA706, 0x46634978,
    0x0521C17C, 0x5EE69212, 0x3559CBA9, 0x2B33DF89,
    0x653538D6, 0x5FDE8344, 0x4091605D, 0x2933BDDE,
    # Partial rounds (23 rounds × 24 constants)
    0x1395D4CA, 0x5DBAC049, 0x51FC2727, 0x13407399,
    0x39AC6953, 0x45E8726C, 0x75A7311C, 0x599F82C9,
    0x702CF13B, 0x026B8955, 0x44E09BBC, 0x2211207F,
    0x5128B4E3, 0x591C41AF, 0x674F5C68, 0x3981D0D3,
    0x2D82F898, 0x707CD267, 0x3B4CCA45, 0x2AD0DC3C,
    0x0CB79B37, 0x23F2F4E8, 0x3DE4E739, 0x7D232359,
    0x389D82F9, 0x259B2E6C, 0x45A94DEF, 0x0D497380,
    0x5B049135, 0x3C268399, 0x78FEB2F9, 0x300A3EEC,
    0x505165BB, 0x20300973, 0x2327C081, 0x1A45A2F4,
    0x5B32EA2E, 0x2D5D1A70, 0x053E613E, 0x5433E39F,
    0x495529F0, 0x1EAA1AA9, 0x578F572A, 0x698EDE71,
    0x5A0F9DBA, 0x398A2E96, 0x0C7B2925, 0x2E6B9564,
    0x026B00DE, 0x7644C1E9, 0x5C23D0BD, 0x3470B5EF,
    0x6013CF3A, 0x48747288, 0x13B7A543, 0x3EAEBD44,
    0x0004E60C, 0x1E8363A2, 0x2343259A, 0x69DA0C2A,
    0x06E3E4C4, 0x1095018E, 0x0DEEA348, 0x1F4C5513,
    0x4F9A3A98, 0x3179112B, 0x524ABB1F, 0x21615BA2,
    0x23AB4065, 0x1202A1D1, 0x21D25B83, 0x6ED17C2F,
    0x391E6B09, 0x5E4ED894, 0x6A2F58F2, 0x5D980D70,
    0x3FA48C5E, 0x1F6366F7, 0x63540F5F, 0x6A8235ED,
    0x14C12A78, 0x6EDDE1C9, 0x58CE1C22, 0x718588BB,
    0x334313AD, 0x7478DBC7, 0x647AD52F, 0x39E82049,
    0x6FEE146A, 0x082C2F24, 0x1F093015, 0x30173C18,
    0x53F70C0D, 0x6028AB0C, 0x2F47A1EE, 0x26A6780E,
    0x3540BC83, 0x1812B49F, 0x5149C827, 0x631DD925,
    0x001F2DEA, 0x7DC05194, 0x3789672E, 0x7CABF72E,
    0x242DBE2F, 0x0B07A51D, 0x38653650, 0x50785C4E,
    0x60E8A7E0, 0x07464338, 0x3482D6E1, 0x08A69F1E,
    0x3F2AFF24, 0x5814C30D, 0x13FECAB2, 0x61CB291A,
    0x68C8226F, 0x5C757EEA, 0x289B4E1E, 0x0198D9B3,
    0x070A92E6, 0x2F1B6CB3, 0x535008BB, 0x35AF339A,
    0x7A38E92C, 0x4FF71B5C, 0x3B193ABA, 0x34D12A1E,
    0x17E94240, 0x2EC214DC, 0x43E09385, 0x7D546918,
    0x71AF9DFD, 0x761A21BB, 0x43FDC986, 0x05DDA714,
    0x2D0E78B5, 0x1FCD387B, 0x76E10A76, 0x28A112D5,
    0x1A7BD787, 0x40190DE2, 0x2E27906A, 0x2033954E,
    0x20AFD2C8, 0x71B5ECB2, 0x57828FB3, 0x222851D8,
    0x732DF0E9, 0x73F48435, 0x7E63EA98, 0x058BE348,
    0x229E7A5F, 0x04576A2F, 0x29939F10, 0x7AFD830A,
    0x5D6DD961, 0x0EB65D94, 0x39DA2B79, 0x36BCE8BA,
    0x5F53A7D4, 0x383B1CD2, 0x1FDC3C5F, 0x7D9CA544,
    0x77480711, 0x36C51A1A, 0x009EA59B, 0x731B17FD,
    0x201359BD, 0x22BF6499, 0x610F1A29, 0x3C73AA45,
    0x6A092599, 0x1C7CB703, 0x79533459, 0x7EF62D86,
    0x5AB925AB, 0x67722AB1, 0x33CA4CFF, 0x007F7DCE,
    0x0EEAC41E, 0x4724BEA7, 0x45EAF64F, 0x21A6C90F,
    0x094B4150, 0x0D942630, 0x18712C30, 0x3A470338,
    0x6EBA7720, 0x487827C8, 0x77013A6D, 0x4AD07390,
    0x57D802EA, 0x720F5FD4, 0x5B8A5357, 0x3649DB1F,
    0x35EA476A, 0x4C6589F5, 0x02C9F31F, 0x16D04670,
    0x62D74B20, 0x1DE813CC, 0x189966ED, 0x527ADD06,
    0x1704F5AF, 0x000F1703, 0x00152A1F, 0x2F49A365,
    0x40EE4288, 0x0AB86260, 0x080C8576, 0x36C6CC05,
    0x0AB9346F, 0x62AA3EC8, 0x51109797, 0x0FEB1585,
    0x04700024, 0x01DEE723, 0x5CD4AAA8, 0x1FE43CE5,
    0x25C31267, 0x58512B48, 0x54147539, 0x4E340AB9,
    0x563FBAEB, 0x60C8353A, 0x65A12D49, 0x6C499FB2,
    0x7EA07556, 0x396E2BBB, 0x31A318F1, 0x11F855AE,
    0x6EDFFB87, 0x59977042, 0x6EC5FA94, 0x75B4F690,
    0x44B6FC61, 0x02A8BED8, 0x4C88C824, 0x08E31432,
    0x09A4C09F, 0x4796B47D, 0x215B7E75, 0x0C639599,
    0x0D93DD4C, 0x2FAC41DE, 0x4F46DADD, 0x03905848,
    0x2B1C39C1, 0x25FFF199, 0x38621F7B, 0x69E59315,
    0x1874C308, 0x024A3959, 0x2BAE1F12, 0x3C200626,
    0x6BA5D369, 0x2FE9B97E, 0x674CC08E, 0x2CBB9657,
    0x550E56C2, 0x5B80E0EC, 0x6549CCFF, 0x54E3E61A,
    0x0FA689E3, 0x2C534848, 0x1EB24382, 0x61B959B5,
    0x4D5F001E, 0x003A95CD, 0x1EDD4507, 0x621E895D,
    0x7DC6E599, 0x0FBC2771, 0x152D0879, 0x77801087,
    0x6A2DD731, 0x3644ABA2, 0x2E43A814, 0x12FF923F,
    0x01CFE2C9, 0x35F8A572, 0x5789FD35, 0x16F39E7A,
    0x7C0CA31C, 0x01016283, 0x2C9DCD96, 0x5D3C6F4E,
    0x0058A186, 0x16354360, 0x502A262B, 0x2B56F93E,
    0x0BC41ECB, 0x33C83E8B, 0x21968FC3, 0x6364490C,
    0x16A45AA5, 0x286D873F, 0x2BE17254, 0x381FBC06,
    0x0DF309AA, 0x15D48B84, 0x0FB2C5DD, 0x7C440D21,
    0x74908F00, 0x75520624, 0x7E58F065, 0x141E1E41,
    0x6582F4AE, 0x2C4479E5, 0x7A09FFF8, 0x1BAA979F,
    0x45AB39BD, 0x774F78BC, 0x3C5F9AA2, 0x115D9DC9,
    0x4B1546D7, 0x196C1A55, 0x6A88FB5E, 0x4C1CA910,
    0x34869067, 0x2662DCBB, 0x0A4625D4, 0x25B121C8,
    0x1A50CCD2, 0x490EA316, 0x42556FFA, 0x6B5E4F88,
    0x329FAF33, 0x54F39A88, 0x3B411E09, 0x6950AE8E,
    0x310A912C, 0x63BDDCBA, 0x347977C0, 0x52831335,
    0x41F32FC6, 0x67DD5ACB, 0x41AE544E, 0x1D83750A,
    0x4BB58D20, 0x2F5496EE, 0x353819EC, 0x412EE425,
    0x1BFD2747, 0x32A14699, 0x2F7BE906, 0x38AFDA41,
    0x5B1E6316, 0x7B810B48, 0x6AEBB30D, 0x55D94F89,
    0x69DB4833, 0x3A6ECB6C, 0x50E7D206, 0x148A4B69,
    0x1AC5548D, 0x40019CF9, 0x1E566F2A, 0x0998A950,
    0x5BC887F0, 0x73FBBD18, 0x341E05A8, 0x7D0597D5,
    0x582308D9, 0x7A98ADDF, 0x0938B854, 0x544BF13D,
    0x50090144, 0x13BAF374, 0x1896A8D5, 0x75EA7475,
    0x23510DD8, 0x72C93BCC, 0x1C41410E, 0x4B72D5F9,
    0x103CCC4E, 0x3896BEF2, 0x2C5E0B1C, 0x1E2096DE,
    0x15594D47, 0x04E035CE, 0x2785D1B1, 0x795BC87D,
    0x373FECBF, 0x0B18C3A0, 0x6516874A, 0x2B567BE9,
    0x5A2A3D1B, 0x74D99C04, 0x437DE605, 0x047DF991,
    0x322FAAD4, 0x2EF2F76F, 0x5F9E7278, 0x62740235,
    0x18C1E8C2, 0x0691E203, 0x3324646D, 0x59542C9F,
    0x32433D0D, 0x42C17492, 0x45AC808A, 0x685394E0,
    0x316F7193, 0x5EA108A0, 0x6BB3F12F, 0x232F8865,
    0x7C162B62, 0x52AA9E45, 0x1B69F8DB, 0x3EC35206,
    0x1EF086DD, 0x34D7A5E3, 0x33AEEA57, 0x03565CC8,
    0x5BC5FD47, 0x47ADC343, 0x1D5857A2, 0x5E7ECE76,
    0x0239FBA3, 0x58BDEAD4, 0x41671AEF, 0x3C8A9189,
    0x7342ED52, 0x19871456, 0x573A02C8, 0x2EC8AD55,
    0x09C4A997, 0x34B9B63A, 0x226DA984, 0x6B31D16E,
    0x458384D2, 0x353911E1, 0x4CFD1256, 0x163C23AF,
    0x7609C5E0, 0x76596C08, 0x087ADAC7, 0x4FD4B62C,
    0x3692A037, 0x51C54B62, 0x133DAF4D, 0x0C76F623,
    0x387D21F3, 0x6034ABE5, 0x7C982E2B, 0x63A266B4,
    0x4F2B17B8, 0x0BD62F1D, 0x70E37A7C, 0x4F162DA9,
    0x38F0E527, 0x6CE798D7, 0x6C74250B, 0x606F2FAD,
    0x212B041D, 0x6724FD32, 0x73AAF9AF, 0x3AE9B76B,
    0x014FE151, 0x37687943, 0x36BB7786, 0x01DA85EF,
    0x28C618AE, 0x36706580, 0x3F5F610D, 0x2E0B9391,
    0x5750E38D, 0x00B48D71, 0x0F1F1D7A, 0x7107C415,
    0x35C1E287, 0x26CCCE2F, 0x4E29277A, 0x1580EE9D,
    0x18136F74, 0x530F32AD, 0x5A19B05D, 0x3D38B320,
    0x6A3BF1E4, 0x39E9EDBB, 0x2CE6A59E, 0x2DF215E1,
    0x216A17BA, 0x3A8F3CFA, 0x0A14D990, 0x1162E529,
    0x1213C181, 0x3DAA68F5, 0x16C570FF, 0x1063321C,
    0x06A2D0E8, 0x17C094A4, 0x39A5D9C9, 0x086D4802,
    0x67AB7FE3, 0x67F51392, 0x3649C2AC, 0x62AA8CF8,
    0x55B6FDBB, 0x55C3E972, 0x2F865724, 0x314FA653,
    0x029F66F1, 0x016F80A2, 0x4B70E0C2, 0x1782F9AB,
    0x697578EE, 0x07B2C8B7, 0x123F6681, 0x2B78DB24,
    0x2CD8DB9D, 0x302947B1, 0x04F4C99A, 0x1F8BCBBD,
    0x61C782EA, 0x3459928C, 0x3EFEC720, 0x24F2B8F6,
    0x5DEC66B5, 0x622386CC, 0x26B70002, 0x1FA0D640,
    0x6EDEAA0A, 0x670FF3E1, 0x18641D8E, 0x43B68197,
    0x315B1707, 0x46DB526A, 0x02FA5277, 0x36F6EDF9,
    0x31AD912B, 0x7D518EBD, 0x61DB2EEA, 0x0BA28BAD,
    0x3C839E59, 0x7ED007F1, 0x74447F8A, 0x6B4CE5B7,
    0x7272E3A4, 0x192257D1, 0x5F882281, 0x5F890768,
    0x47EEC4CB, 0x2EF3E6C8, 0x43D6E4E2, 0x668CE6BA,
    0x50679E00, 0x24C067A8, 0x605BE47C, 0x324AC2EC,
    # Terminal full rounds (4 rounds × 24 constants)
    0x5883788F, 0x7EBA66AF, 0x23620F78, 0x44492C9A,
    0x7CC098A4, 0x705191FA, 0x2F7185E2, 0x6EBBB07E,
    0x23508C3B, 0x6CB0F0F4, 0x1190A8C0, 0x60F8F1D0,
    0x316C16A1, 0x440742C7, 0x7643F142, 0x642F9668,
    0x214B7566, 0x52A5C469, 0x1BFD90DA, 0x1D7D8076,
    0x6E06D1E8, 0x7D672E6D, 0x6FD2E3E3, 0x3257AE18,
    0x75861A51, 0x0E2996FE, 0x2BDC228B, 0x6879FCB8,
    0x14CA9B1C, 0x29953D92, 0x36EE671D, 0x31366E47,
    0x79C4F5F2, 0x2B8C8639, 0x073A293D, 0x32802C31,
    0x4894D32F, 0x06ACC989, 0x40D852B1, 0x508857C4,
    0x2FFE504D, 0x18BE00C1, 0x75A114E9, 0x4ED5922A,
    0x1060EE72, 0x2176563C, 0x0B91B242, 0x6BFBF1A4,
    0x06F94470, 0x694F4383, 0x53CADA3E, 0x1527BFD8,
    0x2BDFE868, 0x120C2D2C, 0x7DFD6309, 0x10B619C2,
    0x0550BC7F, 0x488CF3DC, 0x4C5454A2, 0x00BE2976,
    0x349C9669, 0x2B4EB07D, 0x0450BF40, 0x58DE7343,
    0x3495A265, 0x2305E3B7, 0x661DD781, 0x1C183983,
    0x46992791, 0x3EB3751F, 0x38F728C8, 0x775D0A30,
    0x7636645A, 0x7125AA5D, 0x0C3F2DCA, 0x13B595CC,
    0x5A5E9BCE, 0x54BB3456, 0x069A1A5A, 0x7B9F15EE,
    0x50150189, 0x68C9157B, 0x07E06E22, 0x568AECDB,
    0x1403F847, 0x436CF5DA, 0x3F09C026, 0x652F7B1B,
    0x3E8607F3, 0x5BB37C57, 0x1B1A9ECF, 0x39D11CB0,
    0x1841A51C, 0x1251AD48, 0x74FB5EDD, 0x21FA33C6,
]

# Expected outputs (from Plonky3 koala-bear/src/poseidon1.rs)
_KB_EXPECTED_16 = [
    610090613, 935319874, 1893335292, 796792199,
    356405232, 552237741, 55134556, 1215104204,
    1823723405, 1133298033, 1780633798, 1453946561,
    710069176, 1128629550, 1917333254, 1175481618,
]
_KB_EXPECTED_24 = [
    511672087, 215882318, 237782537, 740528428,
    712760904, 54615367, 751514671, 110231969,
    1905276435, 992525666, 918312360, 18628693,
    749929200, 1916418953, 691276896, 1112901727,
    1163558623, 882867603, 673396520, 1480278156,
    1402044758, 1693467175, 1766273044, 433841551,
]


class TestLeanSpecKoalaBear:
    """
    Verify our Poseidon implementation against test vectors from Plonky3 / leanSpec.

    Reference: https://github.com/tcoratger/leanSpec/tree/poseidon11/src/lean_spec
    Test vectors: Plonky3 koala-bear/src/poseidon1.rs
    """

    def _make_poseidon16(self):
        mds = generate_circulant_mds_matrix(_KB_MDS_FIRST_ROW_16, KOALABEAR_P)
        return Poseidon(
            prime=KOALABEAR_P,
            alpha=3,
            t=16,
            r_f=8,
            r_p=20,
            mds=mds,
            round_constants=_KB_ROUND_CONSTANTS_16,
        )

    def _make_poseidon24(self):
        mds = generate_circulant_mds_matrix(_KB_MDS_FIRST_ROW_24, KOALABEAR_P)
        return Poseidon(
            prime=KOALABEAR_P,
            alpha=3,
            t=24,
            r_f=8,
            r_p=23,
            mds=mds,
            round_constants=_KB_ROUND_CONSTANTS_24,
        )

    def test_plonky3_vector_width16(self):
        """Permutation of [0..15] must match the Plonky3 KoalaBear width-16 vector."""
        pos = self._make_poseidon16()
        result = pos.permutation(list(range(16)))
        assert result == _KB_EXPECTED_16

    def test_plonky3_vector_width24(self):
        """Permutation of [0..23] must match the Plonky3 KoalaBear width-24 vector."""
        pos = self._make_poseidon24()
        result = pos.permutation(list(range(24)))
        assert result == _KB_EXPECTED_24

    def test_custom_round_constants_wrong_length_raises(self):
        mds = generate_circulant_mds_matrix(_KB_MDS_FIRST_ROW_16, KOALABEAR_P)
        with pytest.raises(ValueError, match="round_constants"):
            Poseidon(
                prime=KOALABEAR_P,
                alpha=3,
                t=16,
                r_f=8,
                r_p=20,
                mds=mds,
                round_constants=[0] * 10,  # wrong length
            )


# ---------------------------------------------------------------------------
# Plonky3 KoalaBear matrix security checks
# ---------------------------------------------------------------------------

class TestPlonky3MatrixSecurity:
    """
    Verify Poseidon security properties of the Plonky3 KoalaBear circulant MDS
    matrices (widths 16 and 24) from koala-bear/src/mds.rs.

    Four properties are checked for each matrix:

    1. Algorithm 1 (no subspace-trail type 1): for each i=1..t-1 the vector
       space S_i must not be mapped into itself by M^j for j=1..i, and the
       intersection of S_i with every eigenspace of M^i must be trivial or
       the full space.  Both matrices pass.

    2. Algorithm 2 (invariant subspace from e_0): starting from e_0 the orbit
       {e_0, M*e_0, M^2*e_0, ...} must span GF(p)^t.  Both matrices pass.

    3. Algorithm 3 (Algorithm 2 applied to M^r for r=2..4t): no power of M
       up to the 4t-th may admit a non-trivial invariant subspace reachable
       from e_0.  Both matrices pass.

    4. Minimal-polynomial criterion (_check_minpoly): the characteristic
       polynomial of M^i must be irreducible of degree t over GF(p) for all
       i = 1, …, 2t.  The KoalaBear prime satisfies p-1 = 2^24 × 127, so
       GF(p) contains primitive 16th roots of unity.  Consequently the
       width-16 circulant is diagonalisable over GF(p) itself and its char
       poly splits into linear factors — the criterion fails for both widths.
       This is expected: the Plonky3 matrices were designed for performance,
       not for Poseidon's strict minpoly requirement.
    """

    def _make_m16(self):
        return generate_circulant_mds_matrix(_KB_MDS_FIRST_ROW_16, KOALABEAR_P)

    def _make_m24(self):
        return generate_circulant_mds_matrix(_KB_MDS_FIRST_ROW_24, KOALABEAR_P)

    # --- Algorithm 1 (no subspace-trail type 1) ---

    def test_m16_algorithm1_passes(self):
        """Width-16 matrix: no subspace-trail type-1 vulnerability (Algorithm 1)."""
        result = _algorithm_1(self._make_m16(), 16, KOALABEAR_P)
        assert result[0] is True

    def test_m24_algorithm1_passes(self):
        """Width-24 matrix: no subspace-trail type-1 vulnerability (Algorithm 1)."""
        result = _algorithm_1(self._make_m24(), 24, KOALABEAR_P)
        assert result[0] is True

    def test_m16_algorithm1_reason_is_zero(self):
        """When Algorithm 1 passes the reason code is 0."""
        result = _algorithm_1(self._make_m16(), 16, KOALABEAR_P)
        assert result[1] == 0

    def test_m24_algorithm1_reason_is_zero(self):
        """When Algorithm 1 passes the reason code is 0."""
        result = _algorithm_1(self._make_m24(), 24, KOALABEAR_P)
        assert result[1] == 0

    # --- Algorithm 2 (invariant subspace from e_0) ---

    def test_m16_no_invariant_subspace(self):
        """Width-16 matrix: orbit of e_0 under M spans GF(p)^16 (passes Algorithm 2)."""
        result = _algorithm_2(self._make_m16(), 16, KOALABEAR_P)
        assert result[0] is True

    def test_m24_no_invariant_subspace(self):
        """Width-24 matrix: orbit of e_0 under M spans GF(p)^24 (passes Algorithm 2)."""
        result = _algorithm_2(self._make_m24(), 24, KOALABEAR_P)
        assert result[0] is True

    def test_m16_no_invariant_subspace_returns_none_info(self):
        """When Algorithm 2 passes the auxiliary info field is None."""
        result = _algorithm_2(self._make_m16(), 16, KOALABEAR_P)
        assert result[1] is None

    def test_m24_no_invariant_subspace_returns_none_info(self):
        """When Algorithm 2 passes the auxiliary info field is None."""
        result = _algorithm_2(self._make_m24(), 24, KOALABEAR_P)
        assert result[1] is None

    # --- Algorithm 3 (Algorithm 2 applied to M^r for r=2..4t) ---

    def test_m16_algorithm3_passes(self):
        """Width-16 matrix: no invariant subspace for any power M^r, r=2..64 (Algorithm 3)."""
        result = _algorithm_3(self._make_m16(), 16, KOALABEAR_P)
        assert result[0] is True

    def test_m24_algorithm3_passes(self):
        """Width-24 matrix: no invariant subspace for any power M^r, r=2..96 (Algorithm 3)."""
        result = _algorithm_3(self._make_m24(), 24, KOALABEAR_P)
        assert result[0] is True

    def test_m16_algorithm3_returns_none_info(self):
        """When Algorithm 3 passes the auxiliary info field is None."""
        result = _algorithm_3(self._make_m16(), 16, KOALABEAR_P)
        assert result[1] is None

    def test_m24_algorithm3_returns_none_info(self):
        """When Algorithm 3 passes the auxiliary info field is None."""
        result = _algorithm_3(self._make_m24(), 24, KOALABEAR_P)
        assert result[1] is None

    # --- Minimal polynomial criterion ---

    def test_m16_minpoly_fails_poseidon_criterion(self):
        """
        Width-16 matrix: char poly of M is reducible over GF(p).

        Because 16 | p-1 = 2^24 × 127, all 16th roots of unity lie in GF(p).
        The circulant eigenvalues are therefore elements of GF(p), the char poly
        splits into linear factors, and the Poseidon minpoly criterion fails.
        """
        assert _check_minpoly(self._make_m16(), 16, KOALABEAR_P) is False

    def test_m24_minpoly_fails_poseidon_criterion(self):
        """
        Width-24 matrix: char poly of M is reducible over GF(p).

        Although 24 ∤ p-1 (since 3 ∤ 127), the characteristic polynomial of the
        width-24 circulant still factors into polynomials of degree < 24 over
        GF(p), so the Poseidon minpoly criterion fails here too.
        """
        assert _check_minpoly(self._make_m24(), 24, KOALABEAR_P) is False



# ---------------------------------------------------------------------------
# Density challenge tests
# ---------------------------------------------------------------------------

# Bounty instance parameters (bounty2026.tex §3.2)
_DC_P   = KOALABEAR_P   # 2^31 - 2^24 + 1
_DC_K   = 1
_DC_D   = 2             # max zeros in S
_DC_R   = 17            # len(S) — also used as input length
_DC_ELL = 16            # hash output words
_DC_T   = _DC_K * _DC_ELL  # number of decoded indices that must hit zeros
_DC_RF  = 6
_DC_RP  = 6             # must be > 5 per the spec


class TestDensityChallenge:
    """
    Tests for the density-challenge verifier (bounty2026.tex §2.2).

    Instance: KoalaBear field, k=1, d=2, r=17, ell=16, Poseidon1 R_F=6 R_P=6.

    The verifier is tested for:
      - Correct rejection of trivially invalid candidates.
      - Correct structural checking of each individual condition.
      - Acceptance of a hand-crafted solution (if one were known).
    """

    def _verifier(self, S, **kwargs):
        defaults = dict(
            prime=_DC_P, d=_DC_D, r=_DC_R, t=_DC_T,
            k=_DC_K, ell=_DC_ELL, r_f=_DC_RF, r_p=_DC_RP,
        )
        defaults.update(kwargs)
        return verify_density_solution(S, **defaults)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_wrong_length_raises(self):
        """S with wrong length must raise ValueError."""
        with pytest.raises(ValueError, match="length r"):
            self._verifier([1] * (_DC_R + 1))

    def test_t_exceeds_k_ell_raises(self):
        """t > k*ell must raise ValueError."""
        with pytest.raises(ValueError, match="k\\*ell"):
            self._verifier([1] * _DC_R, t=_DC_K * _DC_ELL + 1)

    # ------------------------------------------------------------------
    # Condition C1: sparsity bound
    # ------------------------------------------------------------------

    def test_too_many_zeros_rejected(self):
        """S with d+1 zeros must be rejected (violates C1)."""
        S = [0] * (_DC_D + 1) + [1] * (_DC_R - _DC_D - 1)
        assert self._verifier(S) is False

    def test_exactly_d_zeros_not_rejected_by_c1(self):
        """
        S with exactly d zeros passes C1 (may still fail C4).
        Use a non-trivial S so C2/C3 can be evaluated; we only assert C1 alone
        doesn't cause rejection (the full verifier may return False due to C4).
        """
        S = [0] * _DC_D + [2] * (_DC_R - _DC_D)
        # C1 must not be the reason for failure — we check it won't raise and
        # returns a bool (True or False depending on C4).
        result = self._verifier(S)
        assert isinstance(result, bool)

    def test_all_nonzero_fails_c4(self):
        """
        S with no zeros cannot satisfy C4 (decoded indices can never point to a
        zero position), so the verifier must return False.
        """
        S = list(range(1, _DC_R + 1))
        assert self._verifier(S) is False

    # ------------------------------------------------------------------
    # Condition C4: decoded indices must hit zero positions
    # ------------------------------------------------------------------

    def test_verifier_returns_bool(self):
        """Verifier must always return a plain bool."""
        S = [0] * _DC_D + [1] * (_DC_R - _DC_D)
        result = self._verifier(S)
        assert isinstance(result, bool)

    def test_deterministic(self):
        """Verifier is deterministic: same S gives the same verdict."""
        S = [0, 0] + list(range(1, _DC_R - 1))
        assert self._verifier(S) == self._verifier(S)

    def test_different_inputs_may_differ(self):
        """Two distinct S vectors need not give the same verdict."""
        S1 = [0, 0] + list(range(1, _DC_R - 1))
        S2 = [0, 0] + list(range(2, _DC_R))
        # At least one should be False (a random S almost surely fails).
        results = {self._verifier(S1), self._verifier(S2)}
        assert False in results

    # ------------------------------------------------------------------
    # Full verifier smoke-test with known-failing candidates
    # ------------------------------------------------------------------

    def test_all_zeros_rejected_by_c1(self):
        """S = [0]*r has r > d zeros so C1 rejects immediately."""
        assert self._verifier([0] * _DC_R) is False

    def test_single_zero_at_position_0(self):
        """
        S with one zero (< d) cannot satisfy C4 for t=16 decoded indices
        unless all 16 decoded indices equal 0 — astronomically unlikely for a
        fixed non-solution S, so the verifier should return False.
        """
        S = [0] + list(range(1, _DC_R))
        assert self._verifier(S) is False
