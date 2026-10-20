"""
applications.py
===============
High-level image processing applications built entirely on top of lut_mul()
from approx_ops.py.  Every pixel multiplication routes through the hardware-
accurate 8×8 LUT; no ordinary NumPy operand multiplication is used in the
processing path.

Applications
------------
alpha_blend(A, B, alpha, mode)  ->  np.ndarray  uint8  shape (H, W[, 3])
rgb_to_grayscale(rgb, mode)     ->  np.ndarray  uint8  shape (H, W)

Supported modes: 0 (exact), 1 (low), 2 (medium), 3 (high approximation).

Fixed-point arithmetic notes
-----------------------------
* Alpha blending uses a 1/255 normalisation implemented as integer divide
  by 255 after accumulation in int32.  Both multiplicands (alpha and
  255-alpha) are uint8 scalars; pixel values are uint8 — all routed through
  lut_mul which returns int32.

* RGB→Grayscale uses BT.601 fixed-point coefficients scaled to sum exactly
  to 256 so that the descale is a right-shift by 8 (integer divide by 256),
  avoiding any floating-point in the processing path:
      Kr = 77   (≈ 0.299 × 256 + rounding)
      Kg = 150  (≈ 0.587 × 256 + rounding)
      Kb = 29   (≈ 0.114 × 256 + rounding)
      sum = 256  → descale by 256 (right-shift 8)
"""

from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from approx_ops import lut_mul


# ---------------------------------------------------------------------------
# BT.601 fixed-point coefficients (scaled × 256, sum = 256)
# ---------------------------------------------------------------------------
_KR = np.uint8(77)   # Red   weight  (≈ 0.2988)
_KG = np.uint8(150)  # Green weight  (≈ 0.5859)
_KB = np.uint8(29)   # Blue  weight  (≈ 0.1133)
_GRAY_SCALE = 256    # descale denominator

# Public constant so tests can import and check it
GRAY_COEFFS = {"kr": int(_KR), "kg": int(_KG), "kb": int(_KB), "scale": _GRAY_SCALE}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_mode(mode: int) -> None:
    if mode not in (0, 1, 2, 3):
        raise ValueError(f"mode must be 0, 1, 2, or 3; got {mode!r}")


def _validate_uint8_image(arr: np.ndarray, name: str) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(arr).__name__}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {arr.dtype}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2-D (H,W) or 3-D (H,W,C), got ndim={arr.ndim}")


def _validate_same_shape(A: np.ndarray, B: np.ndarray) -> None:
    if A.shape != B.shape:
        raise ValueError(
            f"Image shapes must match; got A={A.shape}, B={B.shape}"
        )


# ---------------------------------------------------------------------------
# Application 1: Alpha blending
# ---------------------------------------------------------------------------

def alpha_blend(
    A: np.ndarray,
    B: np.ndarray,
    alpha: int,
    mode: int,
) -> np.ndarray:
    """
    Alpha-composite image A over image B using approximate multiplication.

    Formula per channel per pixel::

        out = clip( ( lut_mul(alpha, A, mode)
                    + lut_mul(255 - alpha, B, mode) ) // 255 ,  0, 255)

    Parameters
    ----------
    A, B : np.ndarray, dtype=uint8
        Source images.  Must have identical shapes.  Accepted layouts:
        * (H, W)       — single-channel / grayscale
        * (H, W, 1)    — single channel, explicit axis
        * (H, W, 3)    — RGB (or any 3-channel image)
        No other channel counts are supported.
    alpha : int
        Blend weight for image A, in the closed range [0, 255].
        alpha=255 → pure A; alpha=0 → pure B.
    mode : int in {0, 1, 2, 3}
        Approximation mode.

    Returns
    -------
    np.ndarray, dtype=uint8, same shape as A and B.

    Notes
    -----
    Both multiplications (alpha×A and (255-alpha)×B) are performed entirely
    through lut_mul; no * operator is applied to pixel values directly.
    The integer division by 255 is exact integer arithmetic on the int32
    accumulator — not a floating-point operation.
    """
    _validate_mode(mode)
    _validate_uint8_image(A, "A")
    _validate_uint8_image(B, "B")
    _validate_same_shape(A, B)

    if not isinstance(alpha, (int, np.integer)):
        raise TypeError(f"alpha must be an integer, got {type(alpha).__name__}")
    alpha = int(alpha)
    if not (0 <= alpha <= 255):
        raise ValueError(f"alpha must be in [0, 255]; got {alpha}")

    ndim_in = A.ndim
    orig_shape = A.shape

    # Flatten to 2-D for uniform processing: (N_pixels, C)
    if A.ndim == 2:
        A_flat = A.reshape(-1, 1)
        B_flat = B.reshape(-1, 1)
    else:
        C = A.shape[2]
        if C not in (1, 3):
            raise ValueError(f"Only 1- or 3-channel images supported; got {C} channels")
        A_flat = A.reshape(-1, C)
        B_flat = B.reshape(-1, C)

    # Alpha scalars as uint8 — these are the multiplier operands for lut_mul
    alpha_u8     = np.uint8(alpha)
    inv_alpha_u8 = np.uint8(255 - alpha)

    # LUT-based multiplications (no * on pixel values)
    # lut_mul broadcasts a scalar against a 2-D array → shape (N_pixels, C)
    prod_A = lut_mul(A_flat, alpha_u8,     mode)   # int32
    prod_B = lut_mul(B_flat, inv_alpha_u8, mode)   # int32

    # Accumulate and descale: integer divide by 255
    acc = prod_A + prod_B                    # int32
    out_flat = (acc // 255).astype(np.int32)

    # Clamp and cast
    out_flat = np.clip(out_flat, 0, 255).astype(np.uint8)
    return out_flat.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Application 2: RGB to Grayscale (BT.601 fixed-point)
# ---------------------------------------------------------------------------

def rgb_to_grayscale(rgb: np.ndarray, mode: int) -> np.ndarray:
    """
    Convert an RGB image to grayscale using BT.601 fixed-point coefficients
    and approximate multiplication.

    Formula per pixel::

        Y = clip( ( lut_mul(R, 77,  mode)
                  + lut_mul(G, 150, mode)
                  + lut_mul(B, 29,  mode) ) // 256 ,  0, 255)

    Coefficients (Kr=77, Kg=150, Kb=29, sum=256) are the standard BT.601
    integer approximation; descaling by 256 keeps all arithmetic in int32.

    Parameters
    ----------
    rgb : np.ndarray, dtype=uint8, shape (H, W, 3)
        RGB image.  Channel order must be R, G, B (i.e. standard RGB, not BGR).
        Convert BGR→RGB with ``rgb = bgr[:, :, ::-1]`` before calling.
    mode : int in {0, 1, 2, 3}
        Approximation mode.

    Returns
    -------
    np.ndarray, dtype=uint8, shape (H, W)
        Grayscale luminance image.

    Notes
    -----
    All three channel multiplications route through lut_mul; no * operator
    is applied to pixel values directly.
    """
    _validate_mode(mode)
    _validate_uint8_image(rgb, "rgb")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"rgb must be a 3-channel (H, W, 3) image; got shape {rgb.shape}"
        )

    R = rgb[:, :, 0]  # uint8  (H, W)
    G = rgb[:, :, 1]  # uint8  (H, W)
    B = rgb[:, :, 2]  # uint8  (H, W)

    # LUT-based channel-weight multiplications — coefficients are uint8 scalars
    prod_r = lut_mul(R, _KR, mode)   # int32 (H, W)
    prod_g = lut_mul(G, _KG, mode)   # int32 (H, W)
    prod_b = lut_mul(B, _KB, mode)   # int32 (H, W)

    # Accumulate and descale by 256 (integer arithmetic, no float)
    acc = prod_r + prod_g + prod_b         # int32 (H, W)
    out = (acc // _GRAY_SCALE).astype(np.int32)

    return np.clip(out, 0, 255).astype(np.uint8)
