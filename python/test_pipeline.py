"""
test_pipeline.py
================
Comprehensive verification and benchmarking of the ApproxImage processing engine:
1. Critical validation test: Manual convolution vs Pipeline.
2. Multiplier usage validation: Verifies LUT integration.
3. Synthetic test image generation (smooth gradients, edges, geometric shapes).
4. Full image convolution across all 4 effects and 4 modes (16 outputs).
5. Quantitative fidelity metrics (MSE, PSNR, SSIM) and runtime benchmarking.
"""

import os
import sys
import time
import cv2
import numpy as np

from image_pipeline import (
    KERNELS,
    load_image,
    process_image,
    calculate_metrics
)
from multiplier_model import (
    load_lut,
    approximate_multiplier,
    exact_multiplier
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "image_results")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# 1. Critical Validation Test: Manual vs Pipeline on Tiny Image
# ============================================================================
def run_manual_validation():
    print("====================================================")
    print("1. CRITICAL VALIDATION TEST: MANUAL CONVOLUTION")
    print("====================================================")

    # 5x5 test image with known gradient pattern
    tiny_img = np.array([
        [ 50,  60,  70,  80,  90],
        [100, 110, 120, 130, 140],
        [150, 160, 170, 180, 190],
        [200, 210, 220, 230, 240],
        [210, 220, 230, 240, 250]
    ], dtype=np.uint8)

    # Target pixel at center: row=2, col=2 (value 170)
    # Window around (2, 2) for 5x5 kernel is the entire 5x5 image:
    blur_mags = KERNELS["blur"]["magnitudes"]
    Kh, Kw = blur_mags.shape
    window = tiny_img[0:Kh, 0:Kw]
    
    # Manual convolution calculation:
    manual_acc = 0
    for r in range(Kh):
        for c in range(Kw):
            manual_acc += int(window[r, c]) * int(blur_mags[r, c])
    manual_pixel = int(np.clip(np.round(manual_acc / 256.0), 0, 255))

    # Pipeline calculation
    pipeline_res = process_image(tiny_img, "blur", mode=0)
    pipeline_pixel = int(pipeline_res[2, 2])

    print(f"  Center pixel window ({Kh}x{Kw}):\n{window}")
    print(f"  Blur kernel magnitudes:\n{blur_mags}")
    print(f"  Manual Convolution Sum Acc: {manual_acc}")
    print(f"  Manual Descaled & Rounded  : {manual_pixel}")
    print(f"  Pipeline Output at (2, 2)  : {pipeline_pixel}")

    assert manual_pixel == pipeline_pixel, f"Mismatch! Manual={manual_pixel}, Pipeline={pipeline_pixel}"
    print("  Status: PASS (Manual calculation matches pipeline exactly)\n")


# ============================================================================
# 2. Multiplier Usage Validation: Confirm LUT Usage
# ============================================================================
def run_multiplier_usage_validation():
    print("====================================================")
    print("2. MULTIPLIER USAGE VALIDATION")
    print("====================================================")

    # 1. Verify Mode 0 LUT matches exact multiplication
    lut0 = load_lut(0)
    test_pairs = [(10, 20), (15, 7), (25, 12), (100, 5), (255, 255), (0, 128)]
    for a, b in test_pairs:
        lut_val = int(lut0[a, b])
        exact_val = a * b
        assert lut_val == exact_val, f"Mode 0 mismatch for {a}x{b}: LUT={lut_val}, Exact={exact_val}"

    # 2. Verify approximate LUTs match verified multiplier model
    for m in [1, 2, 3]:
        lut_m = load_lut(m)
        for a, b in test_pairs:
            lut_val = int(lut_m[a, b])
            model_val = approximate_multiplier(a, b, m)
            assert lut_val == model_val, f"Mode {m} mismatch for {a}x{b}: LUT={lut_val}, Model={model_val}"

    print("  Mode 0: Verified LUT matches exact arithmetic for sample pairs.")
    print("  Modes 1-3: Verified LUT matches approximate RTL model for sample pairs.")
    print("  LUT convolution validation: PASS\n")


# ============================================================================
# 3. Create High-Quality Synthetic Test Image
# ============================================================================
def create_synthetic_test_image(size: int = 512) -> str:
    """
    Creates a deterministic 512x512 synthetic test image containing:
    - Smooth horizontal and radial gradients
    - Sharp vertical/horizontal edges
    - Concentric circles and diagonal bars
    - Checkerboard/textured zones
    """
    img_path = os.path.join(IMAGES_DIR, "synthetic_test.png")
    H = W = size
    img = np.zeros((H, W), dtype=np.float64)

    # 1. Background linear gradient (top to bottom)
    y = np.linspace(30, 220, H)[:, None]
    img += y

    # 2. Left side: smooth horizontal gradient
    x = np.linspace(-40, 40, W)[None, :]
    img += x

    # 3. Geometric Shapes: Sharp squares
    # Top-Left square
    img[40:140, 40:140] = 235.0
    # Center-left dark square
    img[180:260, 40:120] = 20.0

    # 4. Concentric Circles in Center
    center_y, center_x = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    ring1 = (radius >= 40) & (radius <= 60)
    ring2 = (radius >= 80) & (radius <= 100)
    ring3 = (radius >= 120) & (radius <= 135)
    img[ring1] = 250.0
    img[ring2] = 10.0
    img[ring3] = 240.0

    # 5. Diagonal high-contrast edge bars (bottom right)
    for i in range(5):
        offset = 280 + i * 40
        for d in range(20):
            idx = np.arange(H)
            c = (idx + offset + d) % W
            img[idx, c] = 245.0 if (i % 2 == 0) else 15.0

    # 6. Fine-texture checkerboard zone (top right: 60x60 patch)
    chk_size = 6
    patch = np.zeros((90, 90), dtype=np.float64)
    for r in range(90):
        for c in range(90):
            if ((r // chk_size) + (c // chk_size)) % 2 == 0:
                patch[r, c] = 230.0
            else:
                patch[r, c] = 25.0
    img[40:130, W - 130 : W - 40] = patch

    # Normalize to [0, 255] uint8
    final_img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(img_path, final_img)
    print(f"Created synthetic test image: {img_path} ({W}x{H}, uint8)")
    return img_path


# ============================================================================
# 4. Main Processing and Benchmark Pipeline
# ============================================================================
def main():
    print("====================================================")
    print("   APPROXIMAGE: IMAGE-PROCESSING PIPELINE EVALUATION")
    print("====================================================")

    # Step 1: Validation tests
    run_manual_validation()
    run_multiplier_usage_validation()

    # Step 2: Prepare test image
    test_img_path = create_synthetic_test_image(512)
    gray_img, _ = load_image(test_img_path)
    H, W = gray_img.shape
    print(f"Input image loaded successfully: Shape=({H}, {W}), dtype={gray_img.dtype}\n")

    effects = ["blur", "edge", "sharpen", "emboss"]
    benchmark_results = []

    print("====================================================")
    print("PROCESSING 4 EFFECTS ACROSS MODES 0, 1, 2, 3")
    print("====================================================")

    for effect in effects:
        print(f"\nProcessing Effect: {effect.upper()} ({KERNELS[effect]['name']})")
        
        # 1. Generate Mode 0 (Exact reference)
        t0 = time.time()
        exact_img = process_image(gray_img, effect, mode=0)
        t_exact = (time.time() - t0) * 1000.0  # ms
        
        # Save exact output image
        exact_path = os.path.join(RESULTS_DIR, f"{effect}_exact.png")
        cv2.imwrite(exact_path, exact_img)

        # Baseline record for Mode 0
        benchmark_results.append({
            "effect": effect.capitalize(),
            "mode": "Mode 0 (Exact)",
            "mse": 0.0000,
            "psnr": "Infinity",
            "ssim": 1.0000,
            "time_ms": t_exact,
            "filename": f"{effect}_exact.png"
        })
        print(f"  [Mode 0 (Exact)] Processed in {t_exact:.2f} ms -> Saved: {effect}_exact.png")

        # 2. Generate Approximate Modes 1, 2, 3
        for m in [1, 2, 3]:
            t0 = time.time()
            approx_img = process_image(gray_img, effect, mode=m)
            t_approx = (time.time() - t0) * 1000.0  # ms

            # Save approximate output image
            approx_path = os.path.join(RESULTS_DIR, f"{effect}_mode{m}.png")
            cv2.imwrite(approx_path, approx_img)

            # Calculate quality metrics against exact reference
            metrics = calculate_metrics(exact_img, approx_img)

            benchmark_results.append({
                "effect": effect.capitalize(),
                "mode": f"Mode {m}",
                "mse": metrics["mse"],
                "psnr": f"{metrics['psnr']:.2f}" if metrics["psnr"] != float("inf") else "Infinity",
                "ssim": metrics["ssim"],
                "time_ms": t_approx,
                "filename": f"{effect}_mode{m}.png"
            })
            print(f"  [Mode {m}] MSE={metrics['mse']:7.4f} | PSNR={metrics['psnr']:5.2f} dB | SSIM={metrics['ssim']:.4f} | {t_approx:5.2f} ms -> Saved: {effect}_mode{m}.png")

    # ========================================================================
    # 5. Summary Table
    # ========================================================================
    print("\n" + "=" * 88)
    print("                     APPROXIMAGE QUANTITATIVE EVALUATION TABLE")
    print("=" * 88)
    header = f"{'Effect':<12} | {'Mode':<16} | {'MSE':<9} | {'PSNR (dB)':<10} | {'SSIM':<7} | {'Runtime (ms)':<12} | {'Output File':<16}"
    print(header)
    print("-" * 88)

    for r in benchmark_results:
        row = (
            f"{r['effect']:<12} | "
            f"{r['mode']:<16} | "
            f"{r['mse']:<9.4f} | "
            f"{r['psnr']:<10} | "
            f"{r['ssim']:<7.4f} | "
            f"{r['time_ms']:<12.2f} | "
            f"{r['filename']:<16}"
        )
        print(row)
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
