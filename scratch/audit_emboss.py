import os
import sys
import cv2
import numpy as np

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from image_pipeline import process_image, load_image, KERNELS

def audit_emboss():
    print("=== EMBOSS FUNCTIONAL AUDIT ===")
    
    # 1 & 2. Kernel Inspection
    print("\n1. Kernel Configuration:")
    emboss_kernel = KERNELS.get("emboss")
    print(f"Name: {emboss_kernel['name']}")
    
    mags = emboss_kernel['magnitudes']
    signs = emboss_kernel['signs']
    scale = emboss_kernel['scale']
    bias = emboss_kernel['bias']
    
    actual_kernel = mags.astype(np.float32) * signs
    print("Actual Kernel Matrix (before scale):")
    print(actual_kernel)
    print(f"Sum of kernel coefficients: {np.sum(actual_kernel)}")
    print(f"Scale: {scale}, Bias: {bias}")
    
    # 3. Load image and process
    img_path = os.path.join(PROJECT_ROOT, "scratch", "natural_photo.png")
    if not os.path.exists(img_path):
        print(f"Could not find test image at {img_path}. Creating a dummy synthetic image.")
        dummy = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        cv2.imwrite(img_path, dummy)
    
    gray, _ = load_image(img_path)
    
    print("\nProcessing Exact Emboss (Mode 0)...")
    exact_emboss = process_image(gray, "emboss", 0)
    
    print("Processing Approx Emboss (Mode 3)...")
    approx_emboss = process_image(gray, "emboss", 3)
    
    # 4. Measure
    mad_orig_exact = np.mean(np.abs(gray.astype(float) - exact_emboss.astype(float)))
    diff_pixels = np.sum(np.abs(gray.astype(float) - exact_emboss.astype(float)) > 0)
    pct_changed = (diff_pixels / gray.size) * 100
    
    print("\n2. Measurements (Original vs Exact Mode 0):")
    print(f"Mean Absolute Difference: {mad_orig_exact:.2f}")
    print(f"Percentage of changed pixels: {pct_changed:.2f}%")
    print(f"Output Min Value: {exact_emboss.min()}")
    print(f"Output Max Value: {exact_emboss.max()}")
    print(f"Output Mean Value: {np.mean(exact_emboss):.2f}")
    
    mad_exact_approx = np.mean(np.abs(exact_emboss.astype(float) - approx_emboss.astype(float)))
    print("\n3. Measurements (Exact Mode 0 vs Approx Mode 3):")
    print(f"Mean Absolute Difference: {mad_exact_approx:.2f}")
    
    # Save artifacts for visual inspection
    out_dir = os.path.join(CURRENT_DIR, "emboss_audit_results")
    os.makedirs(out_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(out_dir, "orig.png"), gray)
    cv2.imwrite(os.path.join(out_dir, "exact_emboss.png"), exact_emboss)
    cv2.imwrite(os.path.join(out_dir, "approx_emboss.png"), approx_emboss)
    
    print(f"\nArtifacts saved to: {out_dir}")

if __name__ == "__main__":
    audit_emboss()
