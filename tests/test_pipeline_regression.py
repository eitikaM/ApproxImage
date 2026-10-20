import os
import sys
import numpy as np
import pytest

# Ensure python directory is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from image_pipeline import load_image, process_image, calculate_metrics, KERNELS, KERNELS_LEGACY
from multiplier_model import load_lut, LUT_DIR

SAMPLE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "images", "sample_photo.png")


def oracle_convolve_exact(image: np.ndarray, effect_key: str) -> np.ndarray:
    """Independent oracle for exact convolution using standard integer arithmetic."""
    kernel = KERNELS[effect_key] if effect_key in KERNELS else KERNELS_LEGACY[effect_key]
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
    acc = np.zeros((H, W), dtype=np.int32)

    for r in range(Kh):
        for c in range(Kw):
            mag = int(mags[r, c])
            sgn = int(signs[r, c])
            patch = padded[r : r + H, c : c + W].astype(np.int32)
            # Pure integer multiplication oracle
            prod = patch * mag
            if sgn == 1:
                acc += prod
            elif sgn == -1:
                acc -= prod

    if is_edge:
        res = np.abs(np.round(acc / scale))
    else:
        res = np.round(acc / scale) + bias

    return np.clip(res, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("effect", ["blur", "edge", "sharpen", "emboss"])
def test_mode0_matches_independent_oracle(effect):
    """Mode 0 output must exactly equal an independent oracle using plain integer multiplication."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    mode0_out = process_image(img, effect, 0)
    oracle_out = oracle_convolve_exact(img, effect)
    np.testing.assert_array_equal(mode0_out, oracle_out, err_msg=f"Mode 0 diverged from oracle on {effect}")


def test_edge_mode3_differs():
    """For Edge Detection (multi-bit kernel), Mode 3 differs from Mode 0 on sample_photo."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "edge", 0)
    out3 = process_image(img, "edge", 3)
    diff = int(np.sum(out3 != ref))
    assert diff > 0, f"Expected Mode 3 to differ from Mode 0 for edge, got diff={diff}"


def test_legacy_edge_pow2_identical_across_modes():
    """Documents finding: legacy power-of-two edge kernel is bitwise identical across all 4 modes."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "edge_pow2", 0)

    for m in (1, 2, 3):
        out = process_image(img, "edge_pow2", m)
        np.testing.assert_array_equal(out, ref, err_msg=f"Legacy edge Mode {m} differed from Mode 0")


def test_calculate_metrics_identity():
    """calculate_metrics(ref, ref) must return mse=0, psnr=inf, ssim=1.0."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "blur", 0)
    metrics = calculate_metrics(ref, ref)

    assert metrics["mse"] == 0.0, f"Expected MSE 0.0, got {metrics['mse']}"
    assert metrics["psnr"] == float("inf"), f"Expected PSNR inf, got {metrics['psnr']}"
    assert metrics["ssim"] == 1.0, f"Expected SSIM 1.0, got {metrics['ssim']}"
