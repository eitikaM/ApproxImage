import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = r"c:\Users\ASUS\OneDrive\Desktop\final_year\approximate_multipliers"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from app import convolve_multichannel_wrapper, compute_multichannel_metrics, is_color_image

photo_path = os.path.join(PROJECT_ROOT, "photo.jpeg")
bgr = cv2.imread(photo_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
is_col = is_color_image(rgb)

print(f"Is color image: {is_col}")

# Run exact reference
ref, is_out_col, lbl = convolve_multichannel_wrapper(rgb, gray, "blur", 0, is_col)
print(f"Ref shape: {ref.shape}, is_out_col: {is_out_col}, lbl: {lbl}")

# Run mode 3
res3, is_out_col3, lbl3 = convolve_multichannel_wrapper(rgb, gray, "blur", 3, is_col)

# Calculate metrics
met = compute_multichannel_metrics(ref, res3)
print(f"Mode 3 color metrics: SSIM={met['ssim']:.5f}, MSE={met['mse']:.3f}, PSNR={met['psnr']:.2f} dB")

diff_orig_color = np.abs(rgb.astype(float) - ref.astype(float)).mean()
print(f"Mean RGB delta between original and Mode 0 blur: {diff_orig_color:.3f}")
