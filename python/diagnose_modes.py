"""
diagnose_modes.py
=================
ApproxImage Multiplier & Image Processing Pipeline Diagnostic Suite.

Comprehensive diagnostics to verify:
1. Multiplier Look-Up Table (LUT) integrity and arithmetic error distributions.
2. Kernel definitions, scaling factors, and approximation sensitivity.
3. End-to-end image convolution across all four modes (0, 1, 2, 3).
4. Real metrics (MSE, PSNR, SSIM) computed directly from actual output arrays.
5. Bit-level output array comparison between Exact (Mode 0) and Approximate (Modes 1-3).
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root and python package are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verified project imports
from multiplier_model import (
    load_lut,
    LUT_DIR,
    approximate_multiplier,
    exact_multiplier
)
from image_pipeline import (
    process_image,
    calculate_metrics,
    KERNELS,
    load_image
)


def diagnose_luts():
    """Diagnose Look-Up Tables across Modes 0, 1, 2, and 3."""
    print("================================================================================")
    print("1. MULTIPLIER LOOK-UP TABLE (LUT) DIAGNOSTIC")
    print("================================================================================")
    print(f"LUT Directory: {LUT_DIR}")
    print(f"Directory exists: {os.path.isdir(LUT_DIR)}")
    
    luts = {}
    for m in range(4):
        lut_file = os.path.join(LUT_DIR, f"lut_mode{m}.npy")
        exists = os.path.isfile(lut_file)
        lut = load_lut(m)
        luts[m] = lut
        print(f"  Mode {m} LUT: file_exists={exists}, shape={lut.shape}, dtype={lut.dtype}")

    # Mode 0 Exactness verification
    lut0 = luts[0]
    exact_match = True
    for a in [0, 1, 15, 64, 128, 255]:
        for b in [0, 1, 15, 64, 128, 255]:
            if int(lut0[a, b]) != (a * b):
                exact_match = False
                break
    print(f"  Mode 0 Golden Match (A * B): {'PASS (100% Exact)' if exact_match else 'FAIL'}")

    # Pairwise comparison vs Mode 0
    total_entries = 256 * 256
    print("\n  Arithmetic Error Statistics vs Mode 0 (across all 65,536 combinations):")
    print("  +--------+------------+------------+-----------+-----------+--------------------+")
    print("  | Mode   | Mismatches | Error Rate | Max Delta | Mean AE   | Unique LUT Output? |")
    print("  +--------+------------+------------+-----------+-----------+--------------------+")
    for m in range(4):
        diff = np.abs(luts[0].astype(np.int32) - luts[m].astype(np.int32))
        mismatches = int(np.count_nonzero(diff))
        err_rate = (mismatches / total_entries) * 100.0
        max_delta = int(np.max(diff))
        mean_ae = float(np.mean(diff))
        arrays_differ = not np.array_equal(luts[0], luts[m])
        status = "Distinct LUT" if arrays_differ else "Identical (Mode 0)"
        print(f"  | Mode {m} | {mismatches:>10} | {err_rate:>9.2f}% | {max_delta:>9} | {mean_ae:>9.4f} | {status:<18} |")
    print("  +--------+------------+------------+-----------+-----------+--------------------+\n")


def diagnose_kernels():
    """Diagnose kernel coefficient definitions and lower-bit approximation exercise."""
    print("================================================================================")
    print("2. KERNEL SPECIFICATIONS & APPROXIMATION SENSITIVITY")
    print("================================================================================")
    print("  +----------+-----------------------+-----------+---------+------+------------------+")
    print("  | Effect   | Kernel Name           | Dimension | Scale   | Bias | Approx Sensitive |")
    print("  +----------+-----------------------+-----------+---------+------+------------------+")
    for key, kspec in KERNELS.items():
        name = kspec["name"]
        mags = kspec["magnitudes"]
        scale = kspec["scale"]
        bias = kspec["bias"]
        dim_str = f"{mags.shape[0]}x{mags.shape[1]}"
        
        # Check if coefficients have non-zero lower bits (columns 0..6)
        lower_bits_present = any(int(v) & 0x7F != 0 for v in mags.flatten() if v > 0)
        sens_str = "High (Non-zero LSBs)" if lower_bits_present else "Low"
        print(f"  | {key:<8} | {name:<21} | {dim_str:<9} | {scale:<7} | {bias:<4} | {sens_str:<16} |")
    print("  +----------+-----------------------+-----------+---------+------+------------------+\n")


def diagnose_image_pipeline(test_image_path: str = None, effect: str = "blur"):
    """Run full convolution across Modes 0, 1, 2, 3 and compute real metrics."""
    print("================================================================================")
    print("3. IMAGE PIPELINE 4-MODE EVALUATION")
    print("================================================================================")
    
    # Locate test image
    if test_image_path is None or not os.path.exists(test_image_path):
        candidates = [
            PROJECT_ROOT / "images" / "sample_photo.png",
            PROJECT_ROOT / "images" / "dataset" / "img_06_detail_circuit.png",
            PROJECT_ROOT / "images" / "dataset" / "img_36_nat_mountain_skyline.png",
            PROJECT_ROOT / "images" / "dataset" / "img_01_smooth_ramp.png"
        ]
        test_image_path = None
        for cand in candidates:
            if cand.exists():
                test_image_path = str(cand)
                break

    if test_image_path is None or not os.path.exists(test_image_path):
        print(f"  [ERROR] No test image found in {PROJECT_ROOT / 'images'}. Aborting pipeline test.")
        return

    # Load image using verified load_image function
    gray_img, bgr_img = load_image(test_image_path)
    h, w = gray_img.shape
    total_pixels = h * w
    print(f"  Test Image Path: {test_image_path}")
    print(f"  Image Dimensions: {w}x{h} ({total_pixels:,} pixels, uint8 grayscale)")
    print(f"  Selected Effect:  {effect} (Kernel: {KERNELS[effect]['name']})")
    print(f"  Evaluating across Modes 0, 1, 2, and 3...\n")

    # Mode 0 Exact Reference
    exact_output = process_image(gray_img, effect=effect, mode=0)

    print("  +--------+----------------------+-----------+----------+---------+-----------+-----------+----------------+")
    print("  | Mode   | Hardware Level       | MSE       | PSNR(dB) | SSIM    | Max Delta | Mean Delta| Arrays Differ? |")
    print("  +--------+----------------------+-----------+----------+---------+-----------+-----------+----------------+")

    mode_labels = {
        0: "Exact Reference",
        1: "Low Approximation",
        2: "Medium Approx",
        3: "High Approximation"
    }

    for m in range(4):
        approx_output = process_image(gray_img, effect=effect, mode=m)
        
        # Real calculated metrics using calculate_metrics
        metrics = calculate_metrics(exact_output, approx_output)
        mse_val = metrics["mse"]
        psnr_val = metrics["psnr"]
        ssim_val = metrics["ssim"]
        psnr_str = f"{psnr_val:>8.2f}" if psnr_val != float("inf") else "     inf"

        # Direct NumPy array comparison
        diff = np.abs(exact_output.astype(np.int32) - approx_output.astype(np.int32))
        max_delta = int(np.max(diff))
        mean_delta = float(np.mean(diff))
        arrays_differ = not np.array_equal(exact_output, approx_output)
        differ_str = "YES (Differ)" if arrays_differ else "NO (Identical)"

        print(f"  | Mode {m} | {mode_labels[m]:<20} | {mse_val:>9.4f} | {psnr_str} | {ssim_val:>7.5f} | {max_delta:>9} | {mean_delta:>9.4f} | {differ_str:<14} |")
    
    print("  +--------+----------------------+-----------+----------+---------+-----------+-----------+----------------+\n")


def main():
    print("################################################################################")
    print("# APPROXIMAGE HARDWARE MULTIPLIER & PIPELINE DIAGNOSTIC SUITE                 #")
    print("################################################################################\n")
    
    # 1. LUT integrity & arithmetic verification
    diagnose_luts()

    # 2. Kernel definitions & sensitivity
    diagnose_kernels()

    # 3. Image convolution test across all 4 modes
    diagnose_image_pipeline(effect="blur")

    print("================================================================================")
    print("DIAGNOSTIC COMPLETE")
    print("================================================================================")


if __name__ == "__main__":
    main()
