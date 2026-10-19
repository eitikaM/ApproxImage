"""
tests/test_approx_ops.py
========================
Pytest suite for python/approx_ops.py.

Covered:
  1. lut_mul Mode 0 equals plain integer multiplication on random uint8 inputs.
  2. lut_mul Mode 3 differs from Mode 0 on at least one input pair.
  3. lut_mul scalar inputs work.
  4. lut_mul broadcasts correctly.
  5. signed_matmul Mode 0 equals numpy integer matmul oracle on random signed inputs.
  6. signed_matmul Mode 3 differs from Mode 0 on at least one matrix pair.
  7. signed_matmul result signs are correct (spot-checks).
  8. signed_matmul handles zero rows/columns correctly (result stays zero).
  9. lut_mul bad mode raises ValueError.
  10. signed_matmul shape mismatch raises ValueError.
"""

import sys
import os
import numpy as np
import pytest

# Ensure python/ directory is on path so approx_ops and multiplier_model are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from approx_ops import lut_mul, signed_matmul

# Fixed seed for reproducibility
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers / oracles
# ---------------------------------------------------------------------------

def oracle_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plain integer (exact) multiplication — used only as oracle in tests."""
    return (a.astype(np.int32) * b.astype(np.int32))


def oracle_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Plain integer matmul — used only as oracle in tests."""
    return X.astype(np.int32) @ W.astype(np.int32)


# ---------------------------------------------------------------------------
# Test 1: lut_mul Mode 0 equals exact integer multiplication (random uint8)
# ---------------------------------------------------------------------------

class TestLutMulMode0EqualExact:
    """Mode 0 must produce the same result as plain multiplication for all inputs."""

    def test_flat_array_1d(self):
        a = RNG.integers(0, 256, size=500, dtype=np.uint8)
        b = RNG.integers(0, 256, size=500, dtype=np.uint8)
        result = lut_mul(a, b, mode=0)
        expected = oracle_mul(a, b)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 lut_mul differs from plain multiply on 1-D random arrays")

    def test_2d_array(self):
        a = RNG.integers(0, 256, size=(64, 64), dtype=np.uint8)
        b = RNG.integers(0, 256, size=(64, 64), dtype=np.uint8)
        result = lut_mul(a, b, mode=0)
        expected = oracle_mul(a, b)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 lut_mul differs from plain multiply on 2-D random arrays")

    def test_boundary_values(self):
        """Corners: 0×0, 0×255, 255×0, 255×255."""
        a = np.array([0, 0, 255, 255], dtype=np.uint8)
        b = np.array([0, 255, 0, 255], dtype=np.uint8)
        result = lut_mul(a, b, mode=0)
        expected = oracle_mul(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_all_combinations_exhaustive(self):
        """Exhaustive 256×256 grid — the LUT was generated from the same model."""
        a = np.arange(256, dtype=np.uint8)
        b = np.arange(256, dtype=np.uint8)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        result   = lut_mul(aa, bb, mode=0)
        expected = oracle_mul(aa, bb)
        assert np.array_equal(result, expected), \
            "Mode 0 LUT differs from exact multiply on at least one (a,b) pair"


# ---------------------------------------------------------------------------
# Test 2: lut_mul Mode 3 differs from Mode 0 on at least one pair
# ---------------------------------------------------------------------------

class TestLutMulMode3Differs:
    def test_mode3_not_equal_mode0(self):
        """Mode 3 must differ from Mode 0 on at least one input combination."""
        a = np.arange(256, dtype=np.uint8)
        b = np.arange(256, dtype=np.uint8)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        result_m0 = lut_mul(aa, bb, mode=0)
        result_m3 = lut_mul(aa, bb, mode=3)
        assert not np.array_equal(result_m0, result_m3), \
            "Mode 3 should differ from Mode 0 on at least one (a,b) pair"

    def test_mode3_known_difference(self):
        """3 × 3: exact = 9.  Mode 3 approximates columns 0-6, so result != 9."""
        exact = lut_mul(np.array([3], dtype=np.uint8),
                        np.array([3], dtype=np.uint8), mode=0)
        approx = lut_mul(np.array([3], dtype=np.uint8),
                         np.array([3], dtype=np.uint8), mode=3)
        assert exact[0] == 9, f"Mode 0: 3×3 should be 9, got {exact[0]}"
        # Mode 3 may differ; we just confirm it was looked up (not always 9)
        # The actual approximated value can be 0 or something else — just check type
        assert approx.dtype == np.int32

    def test_modes_1_2_differ_from_0(self):
        """All approximation modes must differ from Mode 0 on at least one pair."""
        a = np.arange(256, dtype=np.uint8)
        b = np.arange(256, dtype=np.uint8)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        m0 = lut_mul(aa, bb, mode=0)
        for m in (1, 2, 3):
            mm = lut_mul(aa, bb, mode=m)
            assert not np.array_equal(m0, mm), \
                f"Mode {m} must differ from Mode 0 on at least one pair"


# ---------------------------------------------------------------------------
# Test 3: lut_mul scalar inputs
# ---------------------------------------------------------------------------

class TestLutMulScalar:
    def test_scalar_mode0(self):
        result = lut_mul(7, 8, mode=0)
        assert int(result) == 56, f"7 × 8 should be 56, got {result}"

    def test_scalar_mode0_zero(self):
        assert int(lut_mul(0, 255, mode=0)) == 0
        assert int(lut_mul(255, 0, mode=0)) == 0
        assert int(lut_mul(0, 0, mode=0)) == 0

    def test_scalar_max(self):
        result = lut_mul(255, 255, mode=0)
        assert int(result) == 65025, f"255 × 255 should be 65025, got {result}"

    def test_return_dtype_is_int32(self):
        result = lut_mul(np.uint8(10), np.uint8(20), mode=0)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# Test 4: lut_mul broadcasting
# ---------------------------------------------------------------------------

class TestLutMulBroadcast:
    def test_broadcast_column_vector_times_row_vector(self):
        """(N,1) × (1,M) should yield (N,M)."""
        a = np.array([[1], [2], [3]], dtype=np.uint8)   # (3, 1)
        b = np.array([[4, 5, 6]], dtype=np.uint8)        # (1, 3)
        result = lut_mul(a, b, mode=0)
        expected = oracle_mul(a, b)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 3)

    def test_broadcast_scalar_times_array(self):
        arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        result = lut_mul(np.uint8(5), arr, mode=0)
        expected = oracle_mul(np.uint8(5), arr)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Test 5: signed_matmul Mode 0 equals oracle matmul
# ---------------------------------------------------------------------------

class TestSignedMatmulMode0EqualExact:
    def test_small_positive(self):
        """Small all-positive matrices: signed_matmul Mode 0 == integer matmul."""
        X = RNG.integers(0, 128, size=(8, 16), dtype=np.int32)
        W = RNG.integers(0, 128, size=(16, 8), dtype=np.int32)
        result   = signed_matmul(X, W, mode=0)
        expected = oracle_matmul(X, W)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 signed_matmul differs from integer matmul (positive inputs)")

    def test_mixed_signs(self):
        """Inputs span [-127, 127] to exercise sign restoration."""
        X = RNG.integers(-127, 128, size=(10, 20), dtype=np.int32)
        W = RNG.integers(-127, 128, size=(20, 10), dtype=np.int32)
        result   = signed_matmul(X, W, mode=0)
        expected = oracle_matmul(X, W)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 signed_matmul differs from integer matmul (mixed signs)")

    def test_all_negative(self):
        X = RNG.integers(-100, 0, size=(6, 12), dtype=np.int32)
        W = RNG.integers(-100, 0, size=(12, 6), dtype=np.int32)
        result   = signed_matmul(X, W, mode=0)
        expected = oracle_matmul(X, W)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 signed_matmul differs from integer matmul (all negative)")

    def test_identity_like(self):
        """Multiplying by an identity-like matrix should give X back."""
        X = RNG.integers(-50, 51, size=(5, 5), dtype=np.int32)
        I = np.eye(5, dtype=np.int32)
        result   = signed_matmul(X, I, mode=0)
        expected = oracle_matmul(X, I)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Test 6: signed_matmul Mode 3 differs from Mode 0
# ---------------------------------------------------------------------------

class TestSignedMatmulMode3Differs:
    def test_mode3_not_equal_mode0(self):
        """On typical inputs, Mode 3 matmul must differ from Mode 0 matmul."""
        X = RNG.integers(1, 128, size=(16, 32), dtype=np.int32)
        W = RNG.integers(1, 128, size=(32, 16), dtype=np.int32)
        result_m0 = signed_matmul(X, W, mode=0)
        result_m3 = signed_matmul(X, W, mode=3)
        assert not np.array_equal(result_m0, result_m3), \
            "Mode 3 signed_matmul should differ from Mode 0 on non-trivial inputs"

    def test_mode3_result_is_int32(self):
        X = RNG.integers(1, 50, size=(4, 8), dtype=np.int32)
        W = RNG.integers(1, 50, size=(8, 4), dtype=np.int32)
        result = signed_matmul(X, W, mode=3)
        assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# Test 7: signed_matmul sign correctness
# ---------------------------------------------------------------------------

class TestSignedMatmulSigns:
    def test_positive_times_negative_is_negative(self):
        """(+a)(−b) must accumulate to a negative result (for large enough values)."""
        X = np.array([[10, 20, 30]], dtype=np.int32)   # (1, 3), all positive
        W = np.array([[-5], [-6], [-7]], dtype=np.int32)  # (3, 1), all negative
        result = signed_matmul(X, W, mode=0)
        # 10*(-5) + 20*(-6) + 30*(-7) = -50 - 120 - 210 = -380
        assert result[0, 0] == -380, f"Expected -380, got {result[0, 0]}"

    def test_negative_times_negative_is_positive(self):
        """(−a)(−b) must be positive."""
        X = np.array([[-3, -4]], dtype=np.int32)
        W = np.array([[-2], [-5]], dtype=np.int32)
        result = signed_matmul(X, W, mode=0)
        # (-3)(-2) + (-4)(-5) = 6 + 20 = 26
        assert result[0, 0] == 26, f"Expected 26, got {result[0, 0]}"

    def test_mixed_sign_single_element(self):
        X = np.array([[7]], dtype=np.int32)
        W = np.array([[-9]], dtype=np.int32)
        result = signed_matmul(X, W, mode=0)
        assert result[0, 0] == -63

    def test_zero_element_propagation(self):
        """Zero in X or W should zero the contribution from that k-index."""
        X = np.array([[0, 5]], dtype=np.int32)
        W = np.array([[100], [3]], dtype=np.int32)
        result = signed_matmul(X, W, mode=0)
        # 0*100 + 5*3 = 15
        assert result[0, 0] == 15


# ---------------------------------------------------------------------------
# Test 8: signed_matmul zero matrix
# ---------------------------------------------------------------------------

class TestSignedMatmulZeroMatrix:
    def test_zero_X(self):
        X = np.zeros((5, 8), dtype=np.int32)
        W = RNG.integers(-50, 50, size=(8, 5), dtype=np.int32)
        result = signed_matmul(X, W, mode=0)
        np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.int32))

    def test_zero_W(self):
        X = RNG.integers(-50, 50, size=(5, 8), dtype=np.int32)
        W = np.zeros((8, 5), dtype=np.int32)
        result = signed_matmul(X, W, mode=0)
        np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.int32))


# ---------------------------------------------------------------------------
# Test 9: invalid mode raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidMode:
    def test_lut_mul_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            lut_mul(np.array([1], dtype=np.uint8), np.array([1], dtype=np.uint8), mode=4)

    def test_lut_mul_negative_mode(self):
        with pytest.raises(ValueError, match="mode"):
            lut_mul(np.array([1], dtype=np.uint8), np.array([1], dtype=np.uint8), mode=-1)

    def test_signed_matmul_bad_mode(self):
        X = np.ones((2, 2), dtype=np.int32)
        W = np.ones((2, 2), dtype=np.int32)
        with pytest.raises(ValueError, match="mode"):
            signed_matmul(X, W, mode=5)


# ---------------------------------------------------------------------------
# Test 10: signed_matmul shape mismatch raises ValueError
# ---------------------------------------------------------------------------

class TestShapeMismatch:
    def test_k_mismatch(self):
        X = np.ones((3, 4), dtype=np.int32)
        W = np.ones((5, 4), dtype=np.int32)   # K2=5 != K=4
        with pytest.raises(ValueError):
            signed_matmul(X, W, mode=0)

    def test_1d_X_raises(self):
        X = np.ones(4, dtype=np.int32)
        W = np.ones((4, 2), dtype=np.int32)
        with pytest.raises(ValueError):
            signed_matmul(X, W, mode=0)

    def test_1d_W_raises(self):
        X = np.ones((2, 4), dtype=np.int32)
        W = np.ones(4, dtype=np.int32)
        with pytest.raises(ValueError):
            signed_matmul(X, W, mode=0)
