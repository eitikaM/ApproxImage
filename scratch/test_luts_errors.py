import os
import sys
import numpy as np

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from multiplier_model import load_lut

def test_luts_small_b():
    lut0 = load_lut(0)
    lut3 = load_lut(3)
    
    # Check mismatches when B <= 20
    for B in [3, 7, 9, 14, 19, 20]:
        diff = np.abs(lut0[:, B].astype(np.int32) - lut3[:, B].astype(np.int32))
        mismatches = np.sum(diff > 0)
        max_err = np.max(diff) if mismatches > 0 else 0
        mean_err = np.mean(diff)
        print(f"B={B:2d}, Mismatches: {mismatches}, Max_Err: {max_err}, Mean_Err: {mean_err:.4f}")

if __name__ == "__main__":
    test_luts_small_b()
