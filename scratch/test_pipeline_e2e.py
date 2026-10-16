import os
import sys
import numpy as np
import cv2

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from image_pipeline import process_image, calculate_metrics

def test_pipeline():
    img_path = os.path.join(PROJECT_ROOT, "scratch", "natural_photo.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
    print("Processing Mode 0...")
    res0 = process_image(img, "blur", 0)
    print("Processing Mode 3...")
    res3 = process_image(img, "blur", 3)
    
    diff = np.abs(res0.astype(np.int32) - res3.astype(np.int32))
    print("Max difference:", np.max(diff))
    print("Mismatched pixels:", np.sum(diff > 0))
    print("Total pixels:", img.size)
    
    metrics = calculate_metrics(res0, res3)
    print("Metrics Mode 0 vs 3:", metrics)
    
    # Try with sharpen to see if difference exists
    res0_shp = process_image(img, "sharpen", 0)
    res3_shp = process_image(img, "sharpen", 3)
    diff_shp = np.abs(res0_shp.astype(np.int32) - res3_shp.astype(np.int32))
    print("Max diff sharpen:", np.max(diff_shp))
    print("Mismatched pixels sharpen:", np.sum(diff_shp > 0))

if __name__ == "__main__":
    test_pipeline()
