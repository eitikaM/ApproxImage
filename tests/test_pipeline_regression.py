import os
import sys
import numpy as np
import pytest

# Ensure python directory is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from image_pipeline import load_image, process_image, calculate_metrics, KERNELS
from multiplier_model import load_lut, LUT_DIR

SAMPLE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "images", "sample_photo.png")


def oracle_convolve_exact(image: np.ndarray, effect_key: str) -> np.ndarray:
    """Independent oracle for exact convolution using standard integer arithmetic."""
    kernel = KERNELS[effect_key]
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


@pytest.mark.parametrize("effect", ["blur", "edge", "edge_fine", "sharpen", "emboss"])
def test_mode0_matches_independent_oracle(effect):
    """Mode 0 output must exactly equal an independent oracle using plain integer multiplication."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    mode0_out = process_image(img, effect, 0)
    oracle_out = oracle_convolve_exact(img, effect)
    np.testing.assert_array_equal(mode0_out, oracle_out, err_msg=f"Mode 0 diverged from oracle on {effect}")


def test_edge_fine_mode3_differs():
    """For edge_fine (multi-bit kernel), Mode 3 differs from Mode 0 on sample_photo."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "edge_fine", 0)
    out3 = process_image(img, "edge_fine", 3)
    diff = int(np.sum(out3 != ref))
    assert diff > 0, f"Expected Mode 3 to differ from Mode 0 for edge_fine, got diff={diff}"


def test_load_lut_order_matches_disk():
    """load_lut called in order [3, 0, 2, 1, 3] returns arrays equal to the .npy files on disk."""
    call_order = [3, 0, 2, 1, 3]
    for mode in call_order:
        lut = load_lut(mode)
        disk_file = os.path.join(LUT_DIR, f"lut_mode{mode}.npy")
        assert os.path.exists(disk_file), f"Disk LUT file not found: {disk_file}"
        disk_lut = np.load(disk_file)
        np.testing.assert_array_equal(lut, disk_lut, err_msg=f"LUT in-memory mismatch for Mode {mode}")


def test_blur_modes_divergence_non_decreasing():
    """For Blur, Modes 1, 2, and 3 each differ from Mode 0, and differing pixel count is non-decreasing."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "blur", 0)

    diff1 = int(np.sum(process_image(img, "blur", 1) != ref))
    diff2 = int(np.sum(process_image(img, "blur", 2) != ref))
    diff3 = int(np.sum(process_image(img, "blur", 3) != ref))

    # All modes must differ from exact Mode 0
    assert diff1 > 0, f"Mode 1 produced identical output to Mode 0 for blur (diff count = {diff1})"
    assert diff2 > 0, f"Mode 2 produced identical output to Mode 0 for blur (diff count = {diff2})"
    assert diff3 > 0, f"Mode 3 produced identical output to Mode 0 for blur (diff count = {diff3})"

    # Count of differing pixels must be non-decreasing from Mode 1 to Mode 3
    assert diff1 <= diff2 <= diff3, f"Blur diff counts not non-decreasing: d1={diff1}, d2={diff2}, d3={diff3}"


def test_edge_modes_identical_power_of_two():
    """For Edge, Modes 1, 2, and 3 are identical to Mode 0 due to power-of-two kernel coefficients."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "edge", 0)

    for m in (1, 2, 3):
        out = process_image(img, "edge", m)
        np.testing.assert_array_equal(out, ref, err_msg=f"Edge Mode {m} differed from Mode 0")


def test_calculate_metrics_identity():
    """calculate_metrics(ref, ref) must return mse=0, psnr=inf, ssim=1.0."""
    img, _ = load_image(SAMPLE_IMAGE_PATH)
    ref = process_image(img, "blur", 0)
    metrics = calculate_metrics(ref, ref)

    assert metrics["mse"] == 0.0, f"Expected MSE 0.0, got {metrics['mse']}"
    assert metrics["psnr"] == float("inf"), f"Expected PSNR inf, got {metrics['psnr']}"
    assert metrics["ssim"] == 1.0, f"Expected SSIM 1.0, got {metrics['ssim']}"
