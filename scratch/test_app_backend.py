import os
import sys
import numpy as np
import cv2

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from app import convolve_multichannel_wrapper, compute_multichannel_metrics

def test_app_logic():
    img_path = os.path.join(PROJECT_ROOT, "scratch", "natural_photo.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(PROJECT_ROOT, "results", "text_crop", "crop_orig.png")
    
    bgr = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    is_col = True
    eff_key = "blur"

    # Exact Mode 0
    exact_ref, exact_is_col, exact_label = convolve_multichannel_wrapper(
        rgb_img, gray_img, eff_key, 0, is_col
    )
    
    # Approx Mode 3
    approx_res, is_out_col, out_lbl = convolve_multichannel_wrapper(
        rgb_img, gray_img, eff_key, 3, is_col
    )
    
    metrics = compute_multichannel_metrics(exact_ref, approx_res)
    print("Direct Backend Mode 3 Metrics:")
    print(metrics)

if __name__ == "__main__":
    test_app_logic()
