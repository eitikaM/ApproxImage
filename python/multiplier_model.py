"""
multiplier_model.py
===================
Bit-accurate Python software model of the Configurable 8x8 Hybrid
Lower-Part-OR Approximate Multiplier (matching approximate_multiplier.v).

Modes:
  Mode 0: Exact (100% accurate structural CSA multiplication)
  Mode 1: Low Approximation    (Columns 0..2 approximated via OR)
  Mode 2: Medium Approximation (Columns 0..4 approximated via OR)
  Mode 3: High Approximation   (Columns 0..6 approximated via OR)
"""

import os
import numpy as np

# Global cache for precomputed Look-Up Tables
_LUT_CACHE = {}
LUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "luts")


def exact_multiplier(A: int, B: int) -> int:
    """Golden exact multiplication model for two unsigned 8-bit integers."""
    assert 0 <= A <= 255 and 0 <= B <= 255, "Inputs must be 8-bit unsigned integers (0-255)."
    return (A * B) & 0xFFFF


def approximate_multiplier(A: int, B: int, mode: int = 0) -> int:
    """
    Bit-accurate simulation of approximate_multiplier.v without using '*'.
    
    Reproduces the exact gate-level hardware stages:
      1. Structural 64 1-bit AND partial products: pp[i][j] = A[j] & B[i]
      2. Lower-part OR reduction for columns 0 to 6
      3. Partial product gating for the exact CSA adder tree based on mode
      4. Exact Carry-Save addition of the active partial products
      5. Output multiplexing selecting OR bits vs CSA sum bits
    """
    assert 0 <= A <= 255 and 0 <= B <= 255, "Inputs must be 8-bit unsigned integers (0-255)."
    assert mode in (0, 1, 2, 3), "Mode must be 0, 1, 2, or 3."

    # Step 1: Structural Partial Product Generation (64 AND gates)
    # pp[i][j] where i is row (from B[i]), j is column (from A[j])
    pp = [[((A >> j) & 1) & ((B >> i) & 1) for j in range(8)] for i in range(8)]

    # Step 2: Lower-Part OR Reduction Logic (Columns 0 to 6)
    # Column weight c = i + j
    col_or = [0] * 7
    for c in range(7):
        bit_val = 0
        for i in range(8):
            for j in range(8):
                if i + j == c:
                    bit_val |= pp[i][j]
        col_or[c] = bit_val

    # Step 3: Exact CSA Adder Tree Input Selection (Mode-gated)
    # Determine which columns are approximated based on mode:
    # Mode 0: None (K=0)
    # Mode 1: Columns 0..2 (K=3)
    # Mode 2: Columns 0..4 (K=5)
    # Mode 3: Columns 0..6 (K=7)
    pp_csa = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            c = i + j
            is_approx_col = (
                (mode == 1 and c < 3) or
                (mode == 2 and c < 5) or
                (mode == 3 and c < 7)
            )
            pp_csa[i][j] = 0 if is_approx_col else pp[i][j]

    # Step 4: Structural CSA Adder Tree Sum
    # Sums all active partial products pp_csa[i][j] according to bit weights 2^(i+j)
    p_csa = 0
    for i in range(8):
        for j in range(8):
            if pp_csa[i][j]:
                p_csa += (1 << (i + j))

    # Step 5: Output Multiplexing Layer
    # Lower bits select col_or[b] if in approx zone; otherwise select P_csa[b].
    # Bits 7..15 always select P_csa[b].
    product = 0
    for b in range(16):
        if b < 7:
            is_approx_bit = (
                (mode == 1 and b < 3) or
                (mode == 2 and b < 5) or
                (mode == 3 and b < 7)
            )
            bit_out = col_or[b] if is_approx_bit else ((p_csa >> b) & 1)
        else:
            bit_out = (p_csa >> b) & 1

        if bit_out:
            product |= (1 << b)

    return product & 0xFFFF


def generate_lut(mode: int) -> np.ndarray:
    """
    Generates a full 256x256 uint16 Look-Up Table (LUT) for the specified mode.
    LUT[A, B] gives the product of A and B under that approximation mode.
    """
    assert mode in (0, 1, 2, 3), "Mode must be 0, 1, 2, or 3."
    lut = np.zeros((256, 256), dtype=np.uint16)
    for a in range(256):
        for b in range(256):
            lut[a, b] = approximate_multiplier(a, b, mode)
    return lut


def save_luts(save_dir: str = None) -> None:
    """Generates and saves the 4 LUT files as compact NumPy .npy arrays."""
    if save_dir is None:
        save_dir = LUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    for m in range(4):
        file_path = os.path.join(save_dir, f"lut_mode{m}.npy")
        lut = generate_lut(m)
        np.save(file_path, lut)
        _LUT_CACHE[m] = lut
        print(f"Saved LUT Mode {m} -> {file_path} (shape: {lut.shape}, dtype: {lut.dtype})")


def load_lut(mode: int, lut_dir: str = None) -> np.ndarray:
    """Loads the Look-Up Table for the specified mode from disk (or cache)."""
    assert mode in (0, 1, 2, 3), "Mode must be 0, 1, 2, or 3."
    if mode in _LUT_CACHE:
        return _LUT_CACHE[mode]

    if lut_dir is None:
        lut_dir = LUT_DIR

    file_path = os.path.join(lut_dir, f"lut_mode{mode}.npy")
    if not os.path.exists(file_path):
        # Generate on the fly if file does not exist yet
        lut = generate_lut(mode)
        _LUT_CACHE[mode] = lut
        return lut

    lut = np.load(file_path)
    _LUT_CACHE[mode] = lut
    return lut


def lookup_multiply(A, B, mode: int = 0, lut_dir: str = None):
    """
    Performs fast vectorized or scalar multiplication using the precomputed LUT.
    Compatible with integers, 1D arrays, and 2D/3D image patches.
    """
    lut = load_lut(mode, lut_dir)
    return lut[A, B]
