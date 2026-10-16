import os
import sys
import numpy as np

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from multiplier_model import load_lut

def test_luts():
    print("Loading LUTs...")
    lut0 = load_lut(0)
    lut1 = load_lut(1)
    lut2 = load_lut(2)
    lut3 = load_lut(3)
    
    print("Comparing Mode 0 (Exact) to Mode 3 (Approx):")
    diff = np.abs(lut0.astype(np.int32) - lut3.astype(np.int32))
    
    mismatch_count = np.sum(diff > 0)
    error_rate = (mismatch_count / (256 * 256)) * 100
    mae = np.mean(diff)
    max_ae = np.max(diff)
    
    print(f"Total combinations: 65,536")
    print(f"Mismatches: {mismatch_count}")
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Max Absolute Error: {max_ae}")

    print("\nComparing Mode 0 to Mode 1:")
    diff1 = np.abs(lut0.astype(np.int32) - lut1.astype(np.int32))
    print(f"Mismatches (Mode 1): {np.sum(diff1 > 0)}")

    print("\nComparing Mode 0 to Mode 2:")
    diff2 = np.abs(lut0.astype(np.int32) - lut2.astype(np.int32))
    print(f"Mismatches (Mode 2): {np.sum(diff2 > 0)}")

if __name__ == "__main__":
    test_luts()
