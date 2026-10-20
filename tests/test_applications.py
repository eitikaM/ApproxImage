"""
tests/test_applications.py
==========================
Pytest suite for python/applications.py.

Coverage
--------
alpha_blend
  1.  Mode 0 matches independent exact oracle (various alpha values, shapes).
  2.  alpha=255 → output equals A exactly (Mode 0).
  3.  alpha=0   → output equals B exactly (Mode 0).
  4.  alpha=128 → roughly equal mix; result in [0,255] (Mode 0).
  5.  Mode 3 differs from Mode 0 on at least one pixel for non-trivial input.
  6.  Output shape and dtype preserved for (H,W), (H,W,1), (H,W,3) inputs.
  7.  alpha out-of-range raises ValueError.
  8.  Shape mismatch raises ValueError.
  9.  Non-uint8 input raises ValueError.
  10. Invalid mode raises ValueError.
  11. Grayscale (H,W) and (H,W,1) inputs work correctly.
  12. All-zero image blended with non-zero gives scaled B.
  13. Output values always in [0,255].

rgb_to_grayscale
  14. Mode 0 matches independent exact oracle (BT.601 fixed-point).
  15. Pure-red, pure-green, pure-blue images give correct single-channel
      contribution (spot-checked against formula).
  16. All-black image → all-black output for all modes.
  17. All-white image → full-brightness output (Mode 0).
  18. Output shape is (H, W) for (H, W, 3) input.
  19. Output dtype is uint8.
  20. Mode 3 differs from Mode 0 on at least one pixel (real image content).
  21. Non-(H,W,3) input raises ValueError.
  22. Non-uint8 input raises ValueError.
  23. Invalid mode raises ValueError.
  24. Output values always in [0,255].
  25. Random RGB arrays: Mode 0 matches oracle exhaustively.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Add python/ to sys.path so applications, approx_ops, multiplier_model are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from applications import alpha_blend, rgb_to_grayscale, GRAY_COEFFS

RNG = np.random.default_rng(2024)


# ===========================================================================
# Oracle functions (independent exact-arithmetic — no calls to applications.py)
# ===========================================================================

def _oracle_alpha_blend(A: np.ndarray, B: np.ndarray, alpha: int) -> np.ndarray:
    """
    Exact integer alpha blend:  clip((alpha*A + (255-alpha)*B) // 255, 0, 255).
    Uses plain Python integer arithmetic on each pixel — purely a test oracle.
    """
    a = A.astype(np.int32)
    b = B.astype(np.int32)
    result = (alpha * a + (255 - alpha) * b) // 255
    return np.clip(result, 0, 255).astype(np.uint8)


def _oracle_rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """
    Exact BT.601 fixed-point grayscale:
        Y = clip((Kr*R + Kg*G + Kb*B) // 256, 0, 255)
    with Kr=77, Kg=150, Kb=29 (from GRAY_COEFFS).
    Uses plain Python integer arithmetic — purely a test oracle.
    """
    kr = GRAY_COEFFS["kr"]
    kg = GRAY_COEFFS["kg"]
    kb = GRAY_COEFFS["kb"]
    scale = GRAY_COEFFS["scale"]
    R = rgb[:, :, 0].astype(np.int32)
    G = rgb[:, :, 1].astype(np.int32)
    B = rgb[:, :, 2].astype(np.int32)
    acc = kr * R + kg * G + kb * B
    return np.clip(acc // scale, 0, 255).astype(np.uint8)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def random_gray():
    """Random (64, 64) uint8 grayscale image."""
    return RNG.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture(scope="module")
def random_gray_b():
    return RNG.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture(scope="module")
def random_rgb():
    """Random (64, 64, 3) uint8 RGB image."""
    return RNG.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def random_rgb_b():
    return RNG.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


# ===========================================================================
# 1. alpha_blend — Mode 0 matches oracle
# ===========================================================================

class TestAlphaBlendMode0:
    def test_random_gray_alpha128(self, random_gray, random_gray_b):
        result   = alpha_blend(random_gray, random_gray_b, alpha=128, mode=0)
        expected = _oracle_alpha_blend(random_gray, random_gray_b, 128)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 alpha_blend (gray, alpha=128) differs from oracle")

    def test_random_rgb_alpha128(self, random_rgb, random_rgb_b):
        result   = alpha_blend(random_rgb, random_rgb_b, alpha=128, mode=0)
        expected = _oracle_alpha_blend(random_rgb, random_rgb_b, 128)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 alpha_blend (RGB, alpha=128) differs from oracle")

    @pytest.mark.parametrize("alpha_val", [0, 1, 64, 127, 128, 192, 254, 255])
    def test_various_alphas_gray(self, alpha_val, random_gray, random_gray_b):
        result   = alpha_blend(random_gray, random_gray_b, alpha=alpha_val, mode=0)
        expected = _oracle_alpha_blend(random_gray, random_gray_b, alpha_val)
        np.testing.assert_array_equal(result, expected,
            err_msg=f"Mode 0 alpha_blend gray mismatch at alpha={alpha_val}")

    @pytest.mark.parametrize("alpha_val", [0, 64, 128, 192, 255])
    def test_various_alphas_rgb(self, alpha_val, random_rgb, random_rgb_b):
        result   = alpha_blend(random_rgb, random_rgb_b, alpha=alpha_val, mode=0)
        expected = _oracle_alpha_blend(random_rgb, random_rgb_b, alpha_val)
        np.testing.assert_array_equal(result, expected,
            err_msg=f"Mode 0 alpha_blend RGB mismatch at alpha={alpha_val}")


# ===========================================================================
# 2. alpha=255 → output equals A
# ===========================================================================

class TestAlphaBlendBoundaryAlpha:
    def test_alpha_255_gray(self, random_gray, random_gray_b):
        result = alpha_blend(random_gray, random_gray_b, alpha=255, mode=0)
        # (255*A + 0*B) // 255 = A  (exact for any A in [0,255])
        np.testing.assert_array_equal(result, random_gray,
            err_msg="alpha=255 should return A unchanged")

    def test_alpha_0_gray(self, random_gray, random_gray_b):
        result = alpha_blend(random_gray, random_gray_b, alpha=0, mode=0)
        np.testing.assert_array_equal(result, random_gray_b,
            err_msg="alpha=0 should return B unchanged")

    def test_alpha_255_rgb(self, random_rgb, random_rgb_b):
        result = alpha_blend(random_rgb, random_rgb_b, alpha=255, mode=0)
        np.testing.assert_array_equal(result, random_rgb)

    def test_alpha_0_rgb(self, random_rgb, random_rgb_b):
        result = alpha_blend(random_rgb, random_rgb_b, alpha=0, mode=0)
        np.testing.assert_array_equal(result, random_rgb_b)


# ===========================================================================
# 3. Output shape and dtype
# ===========================================================================

class TestAlphaBlendShape:
    def test_shape_hw(self, random_gray, random_gray_b):
        out = alpha_blend(random_gray, random_gray_b, alpha=100, mode=0)
        assert out.shape == random_gray.shape
        assert out.dtype == np.uint8

    def test_shape_hw1(self):
        A = RNG.integers(0, 256, size=(16, 16, 1), dtype=np.uint8)
        B = RNG.integers(0, 256, size=(16, 16, 1), dtype=np.uint8)
        out = alpha_blend(A, B, alpha=100, mode=0)
        assert out.shape == (16, 16, 1)
        assert out.dtype == np.uint8

    def test_shape_hw3(self, random_rgb, random_rgb_b):
        out = alpha_blend(random_rgb, random_rgb_b, alpha=100, mode=0)
        assert out.shape == random_rgb.shape
        assert out.dtype == np.uint8

    def test_output_range(self, random_rgb, random_rgb_b):
        for alpha_val in (0, 128, 255):
            out = alpha_blend(random_rgb, random_rgb_b, alpha=alpha_val, mode=0)
            assert out.min() >= 0 and out.max() <= 255


# ===========================================================================
# 4. Mode 3 differs from Mode 0
# ===========================================================================

class TestAlphaBlendApproximation:
    def test_mode3_differs_from_mode0(self, random_rgb, random_rgb_b):
        m0 = alpha_blend(random_rgb, random_rgb_b, alpha=128, mode=0)
        m3 = alpha_blend(random_rgb, random_rgb_b, alpha=128, mode=3)
        assert not np.array_equal(m0, m3), \
            "Mode 3 alpha_blend should differ from Mode 0 on non-trivial input"

    def test_mode3_result_is_uint8(self, random_rgb, random_rgb_b):
        out = alpha_blend(random_rgb, random_rgb_b, alpha=100, mode=3)
        assert out.dtype == np.uint8

    def test_all_modes_produce_uint8_in_range(self, random_gray, random_gray_b):
        for m in (0, 1, 2, 3):
            out = alpha_blend(random_gray, random_gray_b, alpha=128, mode=m)
            assert out.dtype == np.uint8
            assert 0 <= out.min() and out.max() <= 255


# ===========================================================================
# 5. alpha_blend — all-zero A blended with non-zero B
# ===========================================================================

class TestAlphaBlendZeroImage:
    def test_zero_A_with_constant_B(self):
        A = np.zeros((8, 8), dtype=np.uint8)
        B = np.full((8, 8), 200, dtype=np.uint8)
        result   = alpha_blend(A, B, alpha=0, mode=0)   # alpha=0 → pure B
        expected = _oracle_alpha_blend(A, B, 0)
        np.testing.assert_array_equal(result, expected)

    def test_zero_A_mid_alpha(self):
        A = np.zeros((8, 8), dtype=np.uint8)
        B = np.full((8, 8), 100, dtype=np.uint8)
        result   = alpha_blend(A, B, alpha=128, mode=0)
        expected = _oracle_alpha_blend(A, B, 128)
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 6. alpha_blend — error handling
# ===========================================================================

class TestAlphaBlendErrors:
    def test_invalid_mode_raises(self, random_gray, random_gray_b):
        with pytest.raises(ValueError, match="mode"):
            alpha_blend(random_gray, random_gray_b, alpha=128, mode=5)

    def test_negative_mode_raises(self, random_gray, random_gray_b):
        with pytest.raises(ValueError, match="mode"):
            alpha_blend(random_gray, random_gray_b, alpha=128, mode=-1)

    def test_alpha_negative_raises(self, random_gray, random_gray_b):
        with pytest.raises(ValueError, match="alpha"):
            alpha_blend(random_gray, random_gray_b, alpha=-1, mode=0)

    def test_alpha_too_large_raises(self, random_gray, random_gray_b):
        with pytest.raises(ValueError, match="alpha"):
            alpha_blend(random_gray, random_gray_b, alpha=256, mode=0)

    def test_shape_mismatch_raises(self):
        A = np.zeros((4, 4), dtype=np.uint8)
        B = np.zeros((4, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            alpha_blend(A, B, alpha=128, mode=0)

    def test_non_uint8_A_raises(self, random_gray_b):
        A_f = np.zeros((64, 64), dtype=np.float32)
        with pytest.raises(ValueError):
            alpha_blend(A_f, random_gray_b, alpha=128, mode=0)

    def test_non_uint8_B_raises(self, random_gray):
        B_i = np.zeros((64, 64), dtype=np.int16)
        with pytest.raises(ValueError):
            alpha_blend(random_gray, B_i, alpha=128, mode=0)

    def test_unsupported_channels_raises(self):
        A = np.zeros((4, 4, 4), dtype=np.uint8)   # RGBA — not supported
        B = np.zeros((4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            alpha_blend(A, B, alpha=128, mode=0)

    def test_non_array_input_raises(self):
        with pytest.raises(TypeError):
            alpha_blend([[1, 2], [3, 4]], np.zeros((2, 2), dtype=np.uint8), alpha=128, mode=0)


# ===========================================================================
# 7. rgb_to_grayscale — Mode 0 matches oracle
# ===========================================================================

class TestRGBToGrayscaleMode0:
    def test_random_rgb_vs_oracle(self, random_rgb):
        result   = rgb_to_grayscale(random_rgb, mode=0)
        expected = _oracle_rgb_to_gray(random_rgb)
        np.testing.assert_array_equal(result, expected,
            err_msg="Mode 0 rgb_to_grayscale differs from BT.601 oracle")

    def test_random_rgb_exhaustive_100_images(self):
        """100 random images — Mode 0 must always equal oracle."""
        for _ in range(100):
            img = RNG.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
            result   = rgb_to_grayscale(img, mode=0)
            expected = _oracle_rgb_to_gray(img)
            np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 8. rgb_to_grayscale — pure-channel spot checks
# ===========================================================================

class TestRGBToGrayscalePureChannel:
    def test_pure_red_255(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:, :, 0] = 255    # R=255, G=0, B=0
        result   = rgb_to_grayscale(img, mode=0)
        expected = _oracle_rgb_to_gray(img)
        # Kr*255 // 256 = 77*255 // 256 = 19635 // 256 = 76
        np.testing.assert_array_equal(result, expected)
        assert result[0, 0] == 76, f"Expected 76 for pure red, got {result[0, 0]}"

    def test_pure_green_255(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:, :, 1] = 255    # G=255
        result   = rgb_to_grayscale(img, mode=0)
        expected = _oracle_rgb_to_gray(img)
        # Kg*255 // 256 = 150*255 // 256 = 38250 // 256 = 149
        np.testing.assert_array_equal(result, expected)
        assert result[0, 0] == 149, f"Expected 149 for pure green, got {result[0, 0]}"

    def test_pure_blue_255(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:, :, 2] = 255    # B=255
        result   = rgb_to_grayscale(img, mode=0)
        expected = _oracle_rgb_to_gray(img)
        # Kb*255 // 256 = 29*255 // 256 = 7395 // 256 = 28
        np.testing.assert_array_equal(result, expected)
        assert result[0, 0] == 28, f"Expected 28 for pure blue, got {result[0, 0]}"

    def test_pure_black(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        result = rgb_to_grayscale(img, mode=0)
        np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.uint8))

    def test_pure_white(self):
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        result   = rgb_to_grayscale(img, mode=0)
        expected = _oracle_rgb_to_gray(img)
        # (77+150+29)*255 // 256 = 256*255 // 256 = 255
        np.testing.assert_array_equal(result, expected)
        assert result[0, 0] == 255, f"Expected 255 for white, got {result[0, 0]}"


# ===========================================================================
# 9. rgb_to_grayscale — output shape and dtype
# ===========================================================================

class TestRGBToGrayscaleShape:
    def test_output_is_hw(self, random_rgb):
        H, W, _ = random_rgb.shape
        out = rgb_to_grayscale(random_rgb, mode=0)
        assert out.shape == (H, W), f"Expected shape ({H},{W}), got {out.shape}"

    def test_output_dtype_uint8(self, random_rgb):
        out = rgb_to_grayscale(random_rgb, mode=0)
        assert out.dtype == np.uint8

    def test_output_range(self, random_rgb):
        out = rgb_to_grayscale(random_rgb, mode=0)
        assert out.min() >= 0 and out.max() <= 255


# ===========================================================================
# 10. rgb_to_grayscale — Mode 3 differs from Mode 0
# ===========================================================================

class TestRGBToGrayscaleApproximation:
    def test_mode3_differs_from_mode0(self, random_rgb):
        m0 = rgb_to_grayscale(random_rgb, mode=0)
        m3 = rgb_to_grayscale(random_rgb, mode=3)
        assert not np.array_equal(m0, m3), \
            "Mode 3 grayscale should differ from Mode 0 on non-trivial RGB input"

    def test_all_modes_uint8_in_range(self, random_rgb):
        for m in (0, 1, 2, 3):
            out = rgb_to_grayscale(random_rgb, mode=m)
            assert out.dtype == np.uint8
            assert 0 <= out.min() and out.max() <= 255

    def test_mode1_mode2_differ_from_mode0(self, random_rgb):
        m0 = rgb_to_grayscale(random_rgb, mode=0)
        for m in (1, 2):
            mm = rgb_to_grayscale(random_rgb, mode=m)
            assert not np.array_equal(m0, mm), \
                f"Mode {m} grayscale should differ from Mode 0"


# ===========================================================================
# 11. rgb_to_grayscale — error handling
# ===========================================================================

class TestRGBToGrayscaleErrors:
    def test_invalid_mode_raises(self, random_rgb):
        with pytest.raises(ValueError, match="mode"):
            rgb_to_grayscale(random_rgb, mode=4)

    def test_negative_mode_raises(self, random_rgb):
        with pytest.raises(ValueError, match="mode"):
            rgb_to_grayscale(random_rgb, mode=-1)

    def test_grayscale_2d_input_raises(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(ValueError):
            rgb_to_grayscale(img, mode=0)

    def test_rgba_input_raises(self):
        img = np.zeros((8, 8, 4), dtype=np.uint8)   # RGBA — not (H,W,3)
        with pytest.raises(ValueError):
            rgb_to_grayscale(img, mode=0)

    def test_non_uint8_raises(self):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            rgb_to_grayscale(img, mode=0)

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            rgb_to_grayscale([[[1, 2, 3]]], mode=0)

    def test_1d_input_raises(self):
        img = np.zeros(24, dtype=np.uint8)
        with pytest.raises(ValueError):
            rgb_to_grayscale(img, mode=0)


# ===========================================================================
# 12. Coefficient consistency check (GRAY_COEFFS exported from applications.py)
# ===========================================================================

class TestGrayCoeffs:
    def test_coeffs_sum_to_256(self):
        total = GRAY_COEFFS["kr"] + GRAY_COEFFS["kg"] + GRAY_COEFFS["kb"]
        assert total == 256, f"BT.601 coefficients must sum to 256; got {total}"

    def test_coeffs_are_uint8_range(self):
        for k, v in GRAY_COEFFS.items():
            if k != "scale":
                assert 0 <= v <= 255, f"Coefficient {k}={v} out of uint8 range"

    def test_scale_is_256(self):
        assert GRAY_COEFFS["scale"] == 256
