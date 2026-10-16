"""
scratch/visual_blur_check.py
Visual & quantitative verification of the new stronger 5x5 Gaussian blur kernel.
"""
import os
import sys
import numpy as np
import cv2
from skimage.metrics import structural_similarity

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from image_pipeline import process_image, calculate_metrics, KERNELS

test_images = [
    ("Sample Photo (Real Photograph)", "images/sample_photo.png"),
    ("Detail Multigrid", "images/dataset/img_06_detail_multigrid.png"),
    ("Brick Wall Texture", "images/dataset/img_27_text_brick_wall.png"),
    ("Smooth Ramp", "images/dataset/img_01_smooth_ramp.png"),
]

print("================================================================================")
print("PART 6: VISUAL BLUR CHECK — STRONGER 5x5 GAUSSIAN KERNEL")
print("================================================================================")
print("Kernel Magnitudes:")
print(KERNELS["blur"]["magnitudes"])
print("Sum of coefficients:", int(KERNELS["blur"]["magnitudes"].sum()))
print("Scale:", KERNELS["blur"]["scale"])
print("================================================================================\n")

for label, img_path in test_images:
    if not os.path.exists(img_path):
        print(f"Skipping {label} (not found: {img_path})")
        continue

    orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if orig.ndim == 3 and orig.shape[2] == 3:
        # process channel-wise for color
        exact_channels = [process_image(orig[:, :, c], "blur", mode=0) for c in range(3)]
        exact_blur = np.stack(exact_channels, axis=2)
        diff = np.abs(orig.astype(np.float64) - exact_blur.astype(np.float64))
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = float(np.max(diff))
        pct_changed = float(np.count_nonzero(diff > 0) / orig.size * 100.0)
    else:
        gray = orig if orig.ndim == 2 else cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        exact_blur = process_image(gray, "blur", mode=0)
        diff = np.abs(gray.astype(np.float64) - exact_blur.astype(np.float64))
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = float(np.max(diff))
        pct_changed = float(np.count_nonzero(diff > 0) / gray.size * 100.0)

    print(f"[{label}] ({img_path})")
    print(f"  Shape: {orig.shape}")
    print(f"  Original vs Exact Mode 0 Blur:")
    print(f"    - Mean Absolute Pixel Difference : {mean_abs_diff:.3f}")
    print(f"    - Maximum Absolute Difference    : {max_abs_diff:.1f}")
    print(f"    - Percentage of Changed Pixels   : {pct_changed:.2f}%")

    # Now compare Exact Mode 0 vs Modes 1, 2, 3
    if orig.ndim == 3 and orig.shape[2] == 3:
        m1 = np.stack([process_image(orig[:, :, c], "blur", mode=1) for c in range(3)], axis=2)
        m2 = np.stack([process_image(orig[:, :, c], "blur", mode=2) for c in range(3)], axis=2)
        m3 = np.stack([process_image(orig[:, :, c], "blur", mode=3) for c in range(3)], axis=2)
    else:
        gray = orig if orig.ndim == 2 else cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        m1 = process_image(gray, "blur", mode=1)
        m2 = process_image(gray, "blur", mode=2)
        m3 = process_image(gray, "blur", mode=3)

    if exact_blur.ndim == 3:
        def calc_m(ref, app):
            m = float(np.mean((ref.astype(np.float64) - app.astype(np.float64)) ** 2))
            p = float("inf") if m == 0 else float(10.0 * np.log10((255.0 ** 2) / m))
            s = float(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) if ref.ndim==3 and False else 
                      structural_similarity(ref, app, channel_axis=-1, data_range=255))
            return {"mse": m, "psnr": p, "ssim": s}
        metrics_m1 = calc_m(exact_blur, m1)
        metrics_m2 = calc_m(exact_blur, m2)
        metrics_m3 = calc_m(exact_blur, m3)
    else:
        metrics_m1 = calculate_metrics(exact_blur, m1)
        metrics_m2 = calculate_metrics(exact_blur, m2)
        metrics_m3 = calculate_metrics(exact_blur, m3)

    print(f"  Approximate Modes vs Exact Blur (Ground Truth):")
    print(f"    Mode 1 -> SSIM: {metrics_m1['ssim']:.4f}, MSE: {metrics_m1['mse']:.4f}, PSNR: {metrics_m1['psnr']:.2f} dB")
    print(f"    Mode 2 -> SSIM: {metrics_m2['ssim']:.4f}, MSE: {metrics_m2['mse']:.4f}, PSNR: {metrics_m2['psnr']:.2f} dB")
    print(f"    Mode 3 -> SSIM: {metrics_m3['ssim']:.4f}, MSE: {metrics_m3['mse']:.4f}, PSNR: {metrics_m3['psnr']:.2f} dB")
    print()

# Save comparison image for inspection
os.makedirs("scratch", exist_ok=True)
if os.path.exists("images/sample_photo.png"):
    photo = cv2.imread("images/sample_photo.png")
    photo_blur = np.stack([process_image(photo[:, :, c], "blur", mode=0) for c in range(3)], axis=2)
    diff_vis = np.clip(np.abs(photo.astype(np.float64) - photo_blur.astype(np.float64)) * 3.0, 0, 255).astype(np.uint8)
    comparison = np.hstack([photo, photo_blur, diff_vis])
    cv2.imwrite("scratch/photo_blur_comparison.png", comparison)
    print("Saved visual comparison to scratch/photo_blur_comparison.png")
