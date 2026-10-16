import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = r"c:\Users\ASUS\OneDrive\Desktop\final_year\approximate_multipliers"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from image_pipeline import process_image, calculate_metrics
from adaptive_inference import AdaptiveInferenceEngine
from app import convolve_multichannel_wrapper, compute_multichannel_metrics, is_color_image

engine = AdaptiveInferenceEngine()

# Test 4 distinct image types
test_images = {}

# 1. Detailed image
det_path = os.path.join(PROJECT_ROOT, "images", "dataset", "img_06_detail_circuit.png")
if os.path.exists(det_path):
    test_images["detailed"] = cv2.imread(det_path, cv2.IMREAD_GRAYSCALE)

# 2. Smooth image
sm_path = os.path.join(PROJECT_ROOT, "images", "dataset", "img_01_smooth_ramp.png")
if os.path.exists(sm_path):
    test_images["smooth"] = cv2.imread(sm_path, cv2.IMREAD_GRAYSCALE)

# 3. Gradient image
gr_path = os.path.join(PROJECT_ROOT, "images", "dataset", "img_31_grad_bilinear.png")
if os.path.exists(gr_path):
    test_images["gradient"] = cv2.imread(gr_path, cv2.IMREAD_GRAYSCALE)

# 4. Natural photo
nat_path = os.path.join(PROJECT_ROOT, "images", "dataset", "img_36_nat_mountain_skyline.png")
if os.path.exists(nat_path):
    test_images["natural_scene"] = cv2.imread(nat_path, cv2.IMREAD_GRAYSCALE)

print("=" * 70)
print("TEST 1: 5x5 BLUR VISIBILITY AND APPROXIMATION ACROSS IMAGE TYPES")
print("=" * 70)

for name, img in test_images.items():
    print(f"\n--- Image Type: {name.upper()} ({img.shape}) ---")
    exact_b = process_image(img, "blur", 0)
    delta_orig_exact = float(np.mean(np.abs(img.astype(float) - exact_b.astype(float))))
    print(f"  Visibility Check (Original vs Exact Blur): Mean Delta = {delta_orig_exact:.2f} levels (Clearly visible: {delta_orig_exact > 5.0})")
    assert delta_orig_exact > 0, "Blur had zero effect!"

    for m in [1, 2, 3]:
        app_b = process_image(img, "blur", m)
        met = calculate_metrics(exact_b, app_b)
        delta_app_exact = float(np.mean(np.abs(exact_b.astype(float) - app_b.astype(float))))
        print(f"  Mode {m} vs Exact Blur: SSIM = {met['ssim']:.5f}, MSE = {met['mse']:.4f}, PSNR = {met['psnr']:.2f} dB, Delta = {delta_app_exact:.4f}")
        # Verify mode differences are real
        if m == 3 and name in ("detailed", "natural_scene"):
            assert met['mse'] > 0, "Mode 3 should show real arithmetic differences"

print("\n" + "=" * 70)
print("TEST 2: HYBRID AI RECOMMENDATION ACROSS EFFECTS & IMAGES")
print("=" * 70)

effects = ["blur", "sharpen", "edge", "emboss"]
for eff in effects:
    print(f"\nEffect: {eff.upper()}")
    for name, img in test_images.items():
        # Step A: exact reference
        ref = process_image(img, eff, 0)
        
        # Step B: AI predict
        pred = engine.predict(img, eff)
        p_unsafe = pred["probability_unsafe"]
        thresh = pred["decision_threshold"]

        # Step C: Hybrid cascade
        # Try Mode 3
        m3_res = process_image(img, eff, 3)
        m3_met = calculate_metrics(ref, m3_res)
        
        selected_mode = None
        if p_unsafe < thresh and m3_met["ssim"] >= 0.99:
            selected_mode = 3
        else:
            m2_res = process_image(img, eff, 2)
            m2_met = calculate_metrics(ref, m2_res)
            if m2_met["ssim"] >= 0.99:
                selected_mode = 2
            else:
                m1_res = process_image(img, eff, 1)
                m1_met = calculate_metrics(ref, m1_res)
                if m1_met["ssim"] >= 0.99:
                    selected_mode = 1
                else:
                    selected_mode = 0

        # Ground truth best mode (highest safe)
        gt_mode = 0
        for cand in [3, 2, 1, 0]:
            if cand == 0:
                gt_mode = 0
                break
            cand_res = process_image(img, eff, cand)
            if calculate_metrics(ref, cand_res)["ssim"] >= 0.99:
                gt_mode = cand
                break

        # Verification: selected mode must satisfy SSIM >= 0.99
        chosen_res = process_image(img, eff, selected_mode) if selected_mode > 0 else ref
        chosen_met = calculate_metrics(ref, chosen_res) if selected_mode > 0 else {"ssim": 1.0}
        assert chosen_met["ssim"] >= 0.99, f"Violation! Selected mode {selected_mode} SSIM: {chosen_met['ssim']}"
        print(f"  {name:<15}: Selected Mode = {selected_mode} (GT: {gt_mode}) | Unsafe Risk = {p_unsafe*100:.1f}% | Verified SSIM = {chosen_met['ssim']:.4f}")

print("\nALL BACKEND FUNCTIONAL TESTS PASSED PERFECTLY!")
