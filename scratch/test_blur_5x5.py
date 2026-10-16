import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = r"c:\Users\ASUS\OneDrive\Desktop\final_year\approximate_multipliers"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from image_pipeline import process_image, calculate_metrics, KERNELS

print("Blur kernel shape:", KERNELS["blur"]["magnitudes"].shape)
print("Blur kernel sum:", int(KERNELS["blur"]["magnitudes"].sum()))
print("Scale:", KERNELS["blur"]["scale"])

# Test on photo.jpeg
photo_path = os.path.join(PROJECT_ROOT, "photo.jpeg")
if os.path.exists(photo_path):
    bgr = cv2.imread(photo_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
else:
    gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

res0 = process_image(gray, "blur", 0)
res1 = process_image(gray, "blur", 1)
res2 = process_image(gray, "blur", 2)
res3 = process_image(gray, "blur", 3)

diff_orig = np.abs(gray.astype(float) - res0.astype(float)).mean()
print(f"Mean delta between original and Mode 0 blur: {diff_orig:.3f}")

for m, r in [(1, res1), (2, res2), (3, res3)]:
    met = calculate_metrics(res0, r)
    print(f"Mode {m} vs Mode 0: SSIM={met['ssim']:.5f}, MSE={met['mse']:.3f}, PSNR={met['psnr']:.2f} dB")
