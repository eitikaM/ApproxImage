"""
verify_multiplier_model.py
==========================
Exhaustive verification of the Python multiplier model against actual
Verilog RTL simulation output across all 4 modes (262,144 test vectors).
Generates and validates precomputed LUTs (.npy) upon passing verification.
"""

import os
import sys
import time
import subprocess
import numpy as np

from multiplier_model import (
    exact_multiplier,
    approximate_multiplier,
    generate_lut,
    save_luts,
    load_lut,
    LUT_DIR
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VERILOG_DIR = os.path.join(PROJECT_ROOT, "verilog")
RTL_TABLE_FILE = os.path.join(RESULTS_DIR, "rtl_truth_table.txt")


def ensure_rtl_truth_table() -> np.ndarray:
    """
    Ensures that the RTL truth table exists by running Icarus Verilog if needed,
    then parses all 262,144 values into a NumPy array of shape (4, 256, 256).
    """
    if not os.path.exists(RTL_TABLE_FILE):
        print("RTL truth table file not found. Running Icarus Verilog simulation...")
        sim_out = os.path.join(VERILOG_DIR, "export_sim")
        cmd_compile = [
            r"D:\ucrt64\bin\iverilog.exe",
            "-o", sim_out,
            os.path.join(VERILOG_DIR, "structural_multiplier.v"),
            os.path.join(VERILOG_DIR, "approximate_multiplier.v"),
            os.path.join(VERILOG_DIR, "export_rtl_truth_table.v")
        ]
        cmd_run = [r"D:\ucrt64\bin\vvp.exe", sim_out]

        env = os.environ.copy()
        env["PATH"] = r"D:\ucrt64\bin;D:\usr\bin;" + env.get("PATH", "")

        subprocess.run(cmd_compile, env=env, check=True)
        subprocess.run(cmd_run, env=env, cwd=VERILOG_DIR, check=True)

    print(f"Loading Verilog RTL truth table from: {RTL_TABLE_FILE}")
    with open(RTL_TABLE_FILE, "r") as f:
        lines = f.read().split()

    assert len(lines) == 262144, f"Expected 262,144 entries, got {len(lines)}"
    rtl_data = np.array([int(x) for x in lines], dtype=np.uint16).reshape((4, 256, 256))
    return rtl_data


def run_sanity_checks():
    """Runs designated spot-check vectors from previous RTL simulation runs."""
    print("----------------------------------------------------")
    print("RUNNING SANITY CHECKS:")
    print("----------------------------------------------------")
    
    test_cases = [
        # Mode 0 (Exact)
        (0, 10, 20, 200),
        (0, 15, 7, 105),
        (0, 25, 12, 300),
        (0, 100, 5, 500),
        (0, 255, 255, 65025),
        # Mode 1 (Low)
        (1, 15, 7, 95),
        (1, 255, 255, 65015),
        # Mode 2 (Med)
        (2, 15, 7, 63),
        (2, 255, 255, 64927),
        # Mode 3 (High)
        (3, 10, 20, 168),
        (3, 15, 7, 63),
        (3, 25, 12, 236),
        (3, 255, 255, 64383),
    ]

    all_passed = True
    for mode, a, b, expected in test_cases:
        actual = approximate_multiplier(a, b, mode)
        match = (actual == expected)
        status_str = "PASS" if match else f"FAIL (got {actual})"
        print(f"  Mode {mode} | {a:3d} x {b:3d} -> Expected: {expected:5d} | Actual: {actual:5d} | {status_str}")
        if not match:
            all_passed = False

    if not all_passed:
        print("\nERROR: Sanity checks failed! Stopping execution.")
        sys.exit(1)
    else:
        print("All sanity checks PASSED successfully!\n")


def main():
    start_time = time.time()

    # 1. Sanity Checks
    run_sanity_checks()

    # 2. Load Ground-Truth Verilog RTL Outputs
    rtl_data = ensure_rtl_truth_table()

    # 3. Exhaustive Verification against RTL Truth Table
    print("\n====================================================")
    print("PYTHON MODEL vs VERILOG RTL VERIFICATION")
    print("====================================================")

    overall_pass = True
    mode_results = {}

    for mode in range(4):
        total_cases = 256 * 256
        mismatches = 0
        first_mismatch = None

        for a in range(256):
            for b in range(256):
                py_val = approximate_multiplier(a, b, mode)
                rtl_val = int(rtl_data[mode, a, b])

                # Additional check for Mode 0 against Python A * B
                if mode == 0:
                    assert py_val == (a * b), f"Mode 0 diverged from A*B at {a}x{b}"

                if py_val != rtl_val:
                    mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = (a, b, rtl_val, py_val)

        status = "PASS" if mismatches == 0 else "FAIL"
        if mismatches > 0:
            overall_pass = False

        mode_results[mode] = (total_cases, mismatches, status)

        print(f"\nMode {mode}:")
        print(f"Total cases: {total_cases}")
        print(f"Mismatches: {mismatches}")
        print(f"Status: {status}")
        if first_mismatch:
            a, b, rtl_v, py_v = first_mismatch
            print(f"  First mismatch at ({a}, {b}): RTL={rtl_v}, Python={py_v}")

    print("\n----------------------------------------------------")
    print(f"Overall Status: {'PASS' if overall_pass else 'FAIL'}")
    print("----------------------------------------------------")

    if not overall_pass:
        print("\nVerification FAILED. LUT generation aborted.")
        sys.exit(1)

    # 4. Generate and Save Look-Up Tables (LUTs)
    print("\n====================================================")
    print("GENERATING AND VALIDATING LOOK-UP TABLES (.npy)")
    print("====================================================")
    save_luts(LUT_DIR)

    # 5. Verify the Generated LUTs
    for mode in range(4):
        lut_file = os.path.join(LUT_DIR, f"lut_mode{mode}.npy")
        assert os.path.exists(lut_file), f"Missing {lut_file}"
        lut = np.load(lut_file)

        # Validate shape, dtype, and bounds
        assert lut.shape == (256, 256), f"Mode {mode}: Invalid shape {lut.shape}"
        assert lut.dtype == np.uint16, f"Mode {mode}: Invalid dtype {lut.dtype}"
        assert np.all((lut >= 0) & (lut <= 65025)), f"Mode {mode}: Values out of range!"

        # Validate equivalence with model and RTL
        assert np.array_equal(lut, rtl_data[mode]), f"Mode {mode}: LUT does not match RTL truth table!"
        print(f"  Mode {mode} LUT verified: shape=(256, 256), dtype=uint16, range=[{lut.min()}, {lut.max()}], 100% RTL Match.")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds.")
    print("Bit-accurate fidelity confirmed across all 262,144 input vectors.")


if __name__ == "__main__":
    main()
