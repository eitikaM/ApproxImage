import os
import sys
import numpy as np

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from multiplier_model import load_lut

def test_luts_edge_kernel():
    lut0 = load_lut(0)
    lut3 = load_lut(3)
    
    # Check mismatches when B in [16, 32, 192]
    for B in [16, 32, 192]:
        diff = np.abs(lut0[:, B].astype(np.int32) - lut3[:, B].astype(np.int32))
        mismatches = np.sum(diff > 0)
        print(f"B={B:3d}, Mismatches: {mismatches}")

if __name__ == "__main__":
    test_luts_edge_kernel()
