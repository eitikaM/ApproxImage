# ApproxImage: Application-Aware Approximate Multiplier Framework

Approximate computing framework exploring runtime-reconfigurable and fixed-mode lower-part-OR approximate multipliers for image processing filters (Blur, Edge Detection, Sharpen, and Emboss) and computer vision applications (Alpha Blending, RGB-to-Grayscale conversion, and Matrix Operations).

---

## 1. Prerequisites & Tool Setup

### A. Python Environment
- **Python**: 3.10+ (tested on Python 3.12 / 3.13)
- **Install Python Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
  Core libraries: `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `joblib`, `streamlit`, `pytest`.

### B. EDA & Hardware Simulation Tools (Optional for RTL verification)
- **Yosys Open SYnthesis Suite**: Tested with Yosys 0.66 (git sha1 `7f8fdfd8d`). Required for generic gate technology-independent synthesis.
- **Icarus Verilog**: Tested with `iverilog` and `vvp` v12.0 (e.g. from MSYS2 UCRT64 / MinGW64). Required for live gate-level netlist and behavioral testbench simulations.
- Ensure `yosys`, `iverilog`, and `vvp` are on your system `PATH` (e.g., `D:\ucrt64\bin` or `C:\msys64\ucrt64\bin`).

---

## 2. Running the Interactive Web Application

Launch the Streamlit graphical interface:

```bash
streamlit run python/app.py
```

The application provides:
- Upload of arbitrary color (RGB) or grayscale images (PNG, JPEG).
- Selection of filter operations: Blur, Edge Detection, Sharpen, and Emboss.
- Multiplier mode selection: Exact Mode 0, Mode 1 (Low), Mode 2 (Medium), Mode 3 (High), or AI Recommended.
- Threshold-aware adaptive mode selector with user-configurable target SSIM constraints (0.99, 0.98, 0.95, 0.90).
- Side-by-side exact reference vs approximate output comparison, 15× amplified pixel difference heatmaps, hardware efficiency metrics, and complete 4-mode sweep tables.

---

## 3. Running Automated Tests

Run the full pytest suite (108 tests passing):

```bash
python -m pytest tests -v
```

The test suite covers:
1. **Pipeline Regression Tests** (`tests/test_pipeline_regression.py`):
   - Exact Mode 0 identity against software integer reference oracles.
   - Precomputed LUT file disk integrity and array ordering.
   - Monotonic error divergence across approximation Modes 0–3.
   - Power-of-two kernel invariance (Edge Detection).
   - Metric identity checks (MSE = 0, PSNR = inf, SSIM = 1.0 on identical inputs).
2. **Elementary Approximate Operators** (`tests/test_approx_ops.py`):
   - Exhaustive unsigned 8×8 LUT multiplication oracle checks (`lut_mul`) in Mode 0.
   - Non-zero approximation divergence in Modes 1–3.
   - Signed matrix multiplication (`signed_matmul`) sign-magnitude reconstruction and int32 accumulation.
   - Input shape validation, broadcasting, and error handling.
3. **Downstream Applications** (`tests/test_applications.py`):
   - Alpha image blending (`alpha_blend`) exact integer rounding oracle matching in Mode 0 across grayscale and RGB.
   - Endpoint boundaries ($\alpha = 0$, $\alpha = 255$) and mid-range blending.
   - RGB-to-grayscale conversion (`rgb_to_grayscale`) fixed-point BT.601 integer coefficients ($77, 150, 29$) oracle checks.
   - Approximation behavior across Modes 1–3 and strict input validation.
4. **Streamlit GUI Smoke Tests** (`tests/test_app_smoke.py`):
   - Headless `AppTest` verification of error-free rendering across configuration, processing, and results screens.
   - Amplified difference heatmap calculation, parameter widgets, and selector recommendations.

---

## 4. Approximate Arithmetic & Applications

In addition to 2D image filtering, the framework provides modular approximate arithmetic operators and downstream applications:

### A. Core Approximate Operators (`python/approx_ops.py`)
- `lut_mul(a, b, mode)`: Elementwise unsigned 8×8 multiplication using verified precomputed LUTs loaded via `multiplier_model.load_lut`. Operands are strictly bounded to $[0, 255]$ with integer output in `int32`.
- `signed_matmul(X, W, mode)`: Sign-magnitude matrix product $X \times W$. Magnitudes $|X|$ and $|W|$ are processed through `lut_mul`, original signs are restored via XOR, and terms are accumulated into an `int32` result.

### B. Image Applications (`python/applications.py`)
- `alpha_blend(img_a, img_b, alpha, mode)`: Blends two images using the integer fixed-point formulation $(A \cdot \alpha + B \cdot (255 - \alpha) + 128) \gg 8$, routing every product through `lut_mul`.
- `rgb_to_grayscale(img_rgb, mode)`: Converts RGB images to grayscale using ITU-R BT.601 fixed-point weights $(R \cdot 77 + G \cdot 150 + B \cdot 29) \gg 8$ via `lut_mul`.

---

## 5. Re-Running Synthesis & Verification Reports

### A. Gate-Level Netlist Verification & Comprehensive Verified Report
Re-runs Icarus Verilog live netlist simulations (`sim_mode*.vvp`, `sim_configurable.vvp`), parses synthesis logs from `results/synthesis_repro/`, computes arithmetic error rates over all 65,536 vector pairs, and generates `results/reports/verified_results.md`:

```bash
python python/make_verified_report.py
```

### B. Adaptive Selector Threshold-Awareness Evaluation
Evaluates old vs. new threshold-aware gate logic across target SSIM thresholds (0.99, 0.98, 0.95, 0.90) on both the 260-task combined dataset and the 80-task independent holdout dataset:

```bash
python python/evaluate_selector_thresholds.py
```
Outputs `results/reports/selector_threshold_eval.md`.

### C. Holdout Dataset Evaluation
Evaluates the trained adaptive selector model on unseen holdout images:

```bash
python python/evaluate_holdout.py
```

### D. RTL Synthesis Reproduction Flow
To reproduce the technology-independent generic gate synthesis using Yosys (`abc -g gates`):

#### 1. Running Reproduction Scripts (`results/synthesis_repro/`)
```bash
yosys results/synthesis_repro/synth_mode0_repro.ys
yosys results/synthesis_repro/synth_mode0_same_rtl.ys
yosys results/synthesis_repro/synth_mode1_repro.ys
yosys results/synthesis_repro/synth_mode2_repro.ys
yosys results/synthesis_repro/synth_mode3_repro.ys
yosys results/synthesis_repro/synth_configurable.ys
```

#### 2. Canonical Synthesis Scripts (`results/`)
Alternatively, the canonical synthesis scripts can be run directly from the project root:
```bash
yosys results/synth_exact.ys
yosys results/synth_mode1.ys
yosys results/synth_mode2.ys
yosys results/synth_mode3.ys
yosys results/synth_configurable.ys
```

To redirect logs to `results/synthesis_repro/` for automated verification reporting:
```bash
yosys -l results/synthesis_repro/synth_mode0_repro.log results/synthesis_repro/synth_mode0_repro.ys
yosys -l results/synthesis_repro/synth_mode1_repro.log results/synthesis_repro/synth_mode1_repro.ys
yosys -l results/synthesis_repro/synth_mode2_repro.log results/synthesis_repro/synth_mode2_repro.ys
yosys -l results/synthesis_repro/synth_mode3_repro.log results/synthesis_repro/synth_mode3_repro.ys
yosys -l results/synthesis_repro/synth_configurable_repro.log results/synthesis_repro/synth_configurable.ys
```

#### 3. Gate-Level Netlist Generation
To synthesize and export gate-level netlists for Icarus Verilog simulation:
```bash
yosys results/synthesis_repro/gen_netlists.ys
yosys results/synthesis_repro/gen_net_configurable.ys
```

#### Expected Synthesis Results Summary (Yosys 0.66)
| Design Top | Multiplier Mode | Total Cells | Scopeinfo | Logic Gates | Cell Savings | LTP (Gate Depth) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `structural_multiplier` | Exact Mode 0 | 430 | 64 | 366 | Baseline (0%) | 40 |
| `synthesis_mode1` | Fixed Mode 1 | 418 | 65 | 353 | -3.55% | 41 |
| `synthesis_mode2` | Fixed Mode 2 | 382 | 65 | 317 | -13.39% | 34 |
| `synthesis_mode3` | Fixed Mode 3 | 327 | 65 | 262 | -28.42% | 29 |
| `approximate_multiplier` | Runtime-Configurable | 495 | 64 | 431 | +17.76% (overhead) | 41 |

---

## 6. Methodological Limitations & Architectural Insights

*(Copied directly from `results/reports/verified_results.md`)*

1. **Generic Gate Counts vs. Physical Implementation**:
   The reported cell counts (`$_AND_`, `$_OR_`, `$_NAND_`, `$_XNOR_`, etc.) represent generic technology-independent gates generated by the Yosys internal mapper (`abc -g gates`). They do not represent actual standard cell library silicon area, parasitic routing capacitance, dynamic switching energy, or power dissipation. Physical ASIC or FPGA implementation under commercial foundry PDKs (e.g. SkyWater 130nm) is required for physical silicon area, power, and clock frequency sign-off.

2. **Longest Topological Path (LTP) vs. Static Timing Analysis (STA)**:
   The LTP length reported by Yosys measures the discrete count of cascaded logic gate levels between primary inputs and outputs. It serves as a rough logic depth proxy, but does not represent actual nanosecond propagation delays, gate fanout loading, wire interconnect delays, or setup/hold timing slacks.

3. **Configurable Design Overhead vs. Fixed-Mode Savings**:
   The fixed-mode cell reductions (-3.55% for Mode 1, -13.39% for Mode 2, -28.42% for Mode 3) are realized ONLY when the mode is permanently tied high/low at synthesis time, enabling Yosys dead-code elimination to prune redundant logic. In contrast, the **Runtime-Configurable** multiplier (`approximate_multiplier`, 431 logic cells, depth 41) requires runtime multiplexing trees and multi-mode partial-product routing, making it **65 cells larger (+17.76% overhead)** than the dedicated 366-cell exact baseline.

4. **Edge Detection Invariance and Arithmetic Overlap in Approximate Columns**:
   OR equals addition unless two or more partial products fall in the same approximated column. Coefficients with a single set bit (16, 32, 64, 128) give one partial-product row, so the result is exact. 192 = 128+64 overlaps only from column 7 upward, outside Mode 3's approximated range (columns 0-6). Emboss coefficient 48 = 32+16 overlaps in columns 5-6, which is why only Mode 3 shows error.

---

## 7. Repository Directory Structure

```
approximate_multipliers/
├── images/
│   ├── dataset/                     # Primary dataset images (img_01 - img_40)
│   ├── dataset_augmented/           # Augmented dataset images (aug_01 - aug_25)
│   ├── dataset_holdout/             # Unseen holdout test images (holdout_01 - holdout_20)
│   └── sample_photo.png             # Canonical benchmark input photo
├── python/
│   ├── adaptive_inference.py        # Standalone ML inference engine for mode prediction
│   ├── app.py                       # Streamlit graphical application and adaptive selector
│   ├── applications.py              # Downstream image applications (Alpha blending, Grayscale)
│   ├── approx_ops.py                # Core LUT multiply and signed matmul operators
│   ├── evaluate_holdout.py          # Holdout validation script
│   ├── evaluate_selector_thresholds.py # Threshold-awareness evaluation script
│   ├── image_pipeline.py            # Convolution, multichannel processing, and metrics
│   ├── make_verified_report.py      # Automated verified report generation script
│   ├── multiplier_model.py          # Software multiplier emulation and LUT loader
│   └── train_adaptive_model.py      # Classifier training pipeline
├── results/
│   ├── luts/                        # Verified 256x256 precomputed multiplier LUTs (.npy, .hex)
│   ├── ml/                          # Trained model bundle, holdout results, and feature metrics
│   ├── reports/                     # Generated verified results & evaluation reports
│   │   ├── verified_results.md      # Verified results report (error, 16 combos, netlists)
│   │   └── selector_threshold_eval.md # Threshold-awareness evaluation report
│   ├── synthesis_repro/             # Reproducible Yosys synthesis scripts, logs, and netlists
│   │   ├── synth_mode0_repro.ys     # Synthesis script for Mode 0 baseline (structural_multiplier)
│   │   ├── synth_mode0_same_rtl.ys  # Synthesis script for Mode 0 wrapper (approximate_multiplier mode 00)
│   │   ├── synth_mode1_repro.ys     # Synthesis script for Mode 1 (synthesis_mode1 wrapper)
│   │   ├── synth_mode2_repro.ys     # Synthesis script for Mode 2 (synthesis_mode2 wrapper)
│   │   ├── synth_mode3_repro.ys     # Synthesis script for Mode 3 (synthesis_mode3 wrapper)
│   │   ├── synth_configurable.ys    # Synthesis script for runtime-configurable approximate_multiplier
│   │   ├── gen_netlists.ys          # Netlist generation script for Modes 0-3
│   │   └── gen_net_configurable.ys  # Netlist generation script for runtime-configurable multiplier
│   ├── synth_exact.ys               # Canonical Mode 0 synthesis script
│   ├── synth_mode1.ys               # Canonical Mode 1 synthesis script
│   ├── synth_mode2.ys               # Canonical Mode 2 synthesis script
│   ├── synth_mode3.ys               # Canonical Mode 3 synthesis script
│   ├── synth_configurable.ys        # Canonical configurable synthesis script
│   └── synthesis_results.csv        # Summary table of hardware synthesis gate counts
├── tests/
│   ├── test_app_smoke.py            # Streamlit AppTest GUI smoke tests
│   ├── test_applications.py         # Tests for alpha blending and grayscale conversion
│   ├── test_approx_ops.py           # Tests for LUT multiply and signed matrix multiplication
│   └── test_pipeline_regression.py  # Image pipeline and multiplier arithmetic regression tests
├── verilog/
│   ├── approximate_multiplier.v     # Synthesizable runtime-configurable 8x8 multiplier RTL
│   ├── approximate_testbench.v      # Exhaustive 65,536-vector Verilog testbench
│   ├── structural_multiplier.v      # Baseline exact 8x8 structural CSA multiplier
│   └── synthesis_mode*.v            # Mode-specific synthesis RTL wrappers (modes 1, 2, 3, exact)
├── README.md                        # Project documentation and reproduction guide
└── requirements.txt                 # Project Python dependencies
```
