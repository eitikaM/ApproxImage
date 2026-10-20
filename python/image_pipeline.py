"""
image_pipeline.py
=================
ApproxImage Convolution Engine for Error-Tolerant Image Processing.

IMPORTANT CONSTRAINTS ENFORCED:
1. No ready-made OpenCV filtering functions (cv2.filter2D, cv2.blur, etc.) are used.
2. Every pixel multiplication is performed strictly through the verified hardware Look-Up Tables (LUTs).
3. 32-bit signed accumulation prevents premature precision loss.
4. Correct signed-magnitude decomposition ensures unsigned 8x8 hardware multiplier compatibility.
"""

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity

from multiplier_model import load_lut, LUT_DIR

# ============================================================================
# Kernel Specifications (Exact values from approved design)
# ============================================================================

KERNELS = {
    "blur": {
        "name": "Gaussian Blur / Smoothing",
        "magnitudes": np.array([
            [3, 7, 9, 7, 3],
            [7, 14, 19, 14, 7],
            [9, 19, 20, 19, 9],
            [7, 14, 19, 14, 7],
            [3, 7, 9, 7, 3]
        ], dtype=np.uint8),
        "signs": np.array([
            [+1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1],
            [+1, +1, +1, +1, +1]
        ], dtype=np.int32),
        "scale": 256.0,
        "bias": 0,
        "is_edge": False
    },
    "edge": {
        "name": "High-Order Laplacian Edge Detection",
        "magnitudes": np.array([
            [16, 32, 16],
            [32, 192, 32],
            [16, 32, 16]
        ], dtype=np.uint8),
        "signs": np.array([
            [-1, -1, -1],
            [-1, +1, -1],
            [-1, -1, -1]
        ], dtype=np.int32),
        "scale": 32.0,
        "bias": 0,
        "is_edge": True
    },
    "sharpen": {
        "name": "High-Boost Sharpening",
        "magnitudes": np.array([
            [11, 21, 11],
            [21, 192, 21],
            [11, 21, 11]
        ], dtype=np.uint8),
        "signs": np.array([
            [-1, -1, -1],
            [-1, +1, -1],
            [-1, -1, -1]
        ], dtype=np.int32),
        "scale": 64.0,
        "bias": 0,
        "is_edge": False
    },
    "emboss": {
        "name": "Directional 45-degree Emboss",
        "magnitudes": np.array([
            [48, 64, 32],
            [64, 0, 64],
            [32, 64, 48]
        ], dtype=np.uint8),
        "signs": np.array([
            [-1, -1, -1],
            [-1, +1, +1],
            [+1, +1, +1]
        ], dtype=np.int32),
        "scale": 128.0,
        "bias": 128,
        "is_edge": False
    }
}


def load_image(image_path: str):
    """
    Loads an image from file using OpenCV only for reading.
    Converts color images to grayscale luminance (uint8, 0-255).
    Returns (gray_image, original_bgr_image).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to read image at: {image_path}")

    if len(bgr.shape) == 3 and bgr.shape[2] == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()

    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray, bgr


def convolve_2d_lut(image: np.ndarray, kernel_key: str, mode: int, lut_dir: str = None) -> np.ndarray:
    """
    Performs 2D convolution over the input image using the verified multiplier LUT.
    
    Dataflow:
      1. Boundary padding with reflection (np.pad mode='reflect').
      2. Shifted slice extractions for the kernel neighborhood (3x3 or 5x5).
      3. Vectorized LUT multiplication: lut[patch, coeff_magnitude].
      4. Sign restoration: sign * lut_result.
      5. Signed 32-bit accumulation.
      6. Descaling (division by scale factor), offset addition, and clamping to [0, 255].
    """
    assert kernel_key in KERNELS, f"Invalid kernel '{kernel_key}'. Choices: {list(KERNELS.keys())}"
    assert mode in (0, 1, 2, 3), "Mode must be 0, 1, 2, or 3."
    assert image.dtype == np.uint8, "Input image must be uint8."

    kernel = KERNELS[kernel_key]
    mags = kernel["magnitudes"]
    signs = kernel["signs"]
    scale = kernel["scale"]
    bias = kernel["bias"]
    is_edge = kernel["is_edge"]

    Kh, Kw = mags.shape
    pad_h = Kh // 2
    pad_w = Kw // 2

    H, W = image.shape
    padded = np.pad(image, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    # Load precomputed verified LUT for the specified mode
    lut = load_lut(mode, lut_dir)

    # 32-bit signed accumulator
    acc = np.zeros((H, W), dtype=np.int32)

    # Iterate through the Kh x Kw window positions
    for r in range(Kh):
        for c in range(Kw):
            mag = mags[r, c]
            sgn = signs[r, c]

            # Slice in padded image corresponding to image[i, j] neighborhood
            patch = padded[r : r + H, c : c + W]

            # STRICT HARDWARE LUT MULTIPLICATION:
            # Multiplies patch (A) by mag (B) via 2D array indexing
            prod = lut[patch, mag].astype(np.int32)

            if sgn == 1:
                acc += prod
            elif sgn == -1:
                acc -= prod

    # Post-processing: Descaling, bias, and clipping
    if is_edge:
        # Edge detection: output = abs(round(Acc / 128)), then clip to [0, 255]
        res = np.abs(np.round(acc / scale))
    else:
        # Blur / Sharpen / Emboss: output = round(Acc / scale) + bias, then clip to [0, 255]
        res = np.round(acc / scale) + bias

    output = np.clip(res, 0, 255).astype(np.uint8)
    return output


def process_image(image: np.ndarray, effect: str, mode: int, lut_dir: str = None) -> np.ndarray:
    """
    High-level entry point for image convolution.
    effect: 'blur', 'edge', 'sharpen', 'emboss'
    mode: 0 (exact), 1 (low approx), 2 (med approx), 3 (high approx)
    """
    return convolve_2d_lut(image, effect, mode, lut_dir)


def calculate_metrics(exact_image: np.ndarray, approx_image: np.ndarray) -> dict:
    """
    Calculates MSE, PSNR (dB), and SSIM between exact processed image and approximate output.
    """
    assert exact_image.shape == approx_image.shape, "Images must have identical dimensions."
    assert exact_image.dtype == np.uint8 and approx_image.dtype == np.uint8, "Images must be uint8."

    # Mean Squared Error
    mse = float(np.mean((exact_image.astype(np.float64) - approx_image.astype(np.float64)) ** 2))

    # Peak Signal-to-Noise Ratio
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))

    # Structural Similarity Index (SSIM)
    ssim = float(structural_similarity(exact_image, approx_image, data_range=255))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }
