"""
app.py
======
ApproxImage: Application-Aware Approximate Computing for Efficient Image Processing
Final-Year Project: Design and Evaluation of a Configurable 8x8 Approximate
Multiplier for Error-Tolerant Image Processing.

Streamlit Interactive Research Demonstration Application.
Redesigned with a warm, cheerful, whimsical pastel engineering aesthetic
and a two-screen configuration-to-results flow.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.metrics import structural_similarity

# Setup module resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import existing verified pipelines (DO NOT MODIFY UNDERLYING MODULES)
try:
    from adaptive_inference import AdaptiveInferenceEngine
    from image_pipeline import process_image, calculate_metrics, KERNELS
    from multiplier_model import LUT_DIR
except Exception as e:
    st.error(f"Failed to load project backend modules: {e}")
    st.stop()

# ============================================================================
# Hardware Synthesis Constants (Verified Yosys Synthesis Results)
# ============================================================================
HARDWARE_SPECS = {
    0: {
        "label": "Mode 0 — Exact",
        "short_label": "Mode 0",
        "level": "Exact Reference",
        "tag": "Maximum Precision",
        "cells": 366,
        "saved_cells": 0,
        "reduction_pct": 0.0,
        "reduction_str": "0.00%",
        "critical_path": 40,
        "timing_change_str": "Baseline (40)",
        "timing_eval": "Baseline reference (40)",
        "desc": "Exact 8×8 multiplier reference (columns 0..7 exact)"
    },
    1: {
        "label": "Mode 1 — Low Approximation",
        "short_label": "Mode 1",
        "level": "Low Approximation",
        "tag": "Small Approximation",
        "cells": 353,
        "saved_cells": 13,
        "reduction_pct": 3.55,
        "reduction_str": "3.55%",
        "critical_path": 41,
        "timing_change_str": "Critical path slightly worse (+2.5%)",
        "timing_eval": "+2.5% slightly longer critical path (41 vs 40)",
        "desc": "Approximate columns 0..2 (OR-gate approximation)"
    },
    2: {
        "label": "Mode 2 — Medium Approximation",
        "short_label": "Mode 2",
        "level": "Medium Approximation",
        "tag": "Balanced Efficiency",
        "cells": 317,
        "saved_cells": 49,
        "reduction_pct": 13.39,
        "reduction_str": "13.39%",
        "critical_path": 34,
        "timing_change_str": "15% shorter critical path",
        "timing_eval": "15% shorter critical path (34 vs 40)",
        "desc": "Approximate columns 0..4 (OR-gate approximation)"
    },
    3: {
        "label": "Mode 3 — High Approximation",
        "short_label": "Mode 3",
        "level": "High Approximation",
        "tag": "Maximum Hardware Efficiency",
        "cells": 262,
        "saved_cells": 104,
        "reduction_pct": 28.42,
        "reduction_str": "28.42%",
        "critical_path": 29,
        "timing_change_str": "27.5% shorter critical path",
        "timing_eval": "27.5% shorter critical path (29 vs 40)",
        "desc": "Approximate columns 0..6 (OR-gate approximation)"
    }
}

QUALITY_THRESHOLDS = [
    (0.999, "Excellent", "badge-excellent"),
    (0.990, "Very Good", "badge-good"),
    (0.950, "Acceptable", "badge-acceptable"),
    (0.000, "Poor", "badge-poor")
]


def classify_quality(ssim: float):
    """Categorizes SSIM value using project quality thresholds."""
    for thresh, label, css_class in QUALITY_THRESHOLDS:
        if ssim >= thresh:
            return label, css_class
    return "Poor", "badge-poor"


@st.cache_resource
def get_inference_engine():
    """Initializes and caches the trained ML inference engine."""
    model_path = os.path.join(PROJECT_ROOT, "results", "ml", "trained_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model bundle not found at: {model_path}")
    return AdaptiveInferenceEngine(model_path=model_path)


def verify_lut_files():
    """Ensures that all 4 verified LUT files are present."""
    for m in range(4):
        lut_path = os.path.join(LUT_DIR, f"lut_mode{m}.npy")
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"Missing multiplier LUT for Mode {m} at {lut_path}. Run multiplier_model.py first.")


# ============================================================================
# Multichannel / Color Application-Level Wrapper
# ============================================================================
def is_color_image(img_rgb: np.ndarray) -> bool:
    """Detects whether an image is multichannel RGB/RGBA with genuine chromatic variance."""
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] >= 3:
        ch0 = img_rgb[:, :, 0]
        ch1 = img_rgb[:, :, 1]
        ch2 = img_rgb[:, :, 2]
        if np.any(ch0 != ch1) or np.any(ch1 != ch2):
            return True
    return False


def convolve_multichannel_wrapper(image_rgb: np.ndarray, gray_img: np.ndarray, effect_key: str, mode: int, is_color: bool):
    """
    Application-level wrapper around the verified image_pipeline.process_image.
    - For color input with Blur or Sharpen:
        Splits into R, G, B channels, convolves each channel independently via the
        verified hardware multiplier LUT pipeline, and recombines into RGB.
    - For Edge Detection:
        Computes on grayscale luminance, outputting a dedicated Grayscale Edge Map.
    - For Emboss:
        Computes on grayscale luminance, outputting a dedicated Grayscale Emboss Map.
    - For Grayscale input:
        Computes on grayscale luminance.
    Returns:
        (result_array, is_output_color, output_label)
    """
    if is_color and effect_key in ("blur", "sharpen"):
        # Split into individual R, G, B channels
        r_ch = image_rgb[:, :, 0]
        g_ch = image_rgb[:, :, 1]
        b_ch = image_rgb[:, :, 2]

        # Execute verified pipeline on each channel independently
        proc_r = process_image(r_ch, effect_key, mode)
        proc_g = process_image(g_ch, effect_key, mode)
        proc_b = process_image(b_ch, effect_key, mode)

        # Recombine into full-color RGB image
        proc_rgb = np.stack([proc_r, proc_g, proc_b], axis=-1)
        return proc_rgb, True, "Color RGB"
    elif effect_key in ("edge", "edge_fine"):
        # Grayscale edge map
        proc_edge = process_image(gray_img, effect_key, mode)
        return proc_edge, False, "Grayscale Edge Map"
    elif effect_key == "emboss":
        # Grayscale emboss map
        proc_emboss = process_image(gray_img, effect_key, mode)
        return proc_emboss, False, "Grayscale Emboss"
    else:
        # Standard grayscale image
        proc_gray = process_image(gray_img, effect_key, mode)
        return proc_gray, False, "Grayscale Output"


def compute_multichannel_metrics(exact_ref: np.ndarray, approx_res: np.ndarray) -> dict:
    """
    Computes MSE, PSNR, and SSIM between exact reference and approximate result.
    Supports both 2D grayscale and 3D RGB arrays.
    """
    assert exact_ref.shape == approx_res.shape, "Images must have identical dimensions."
    assert exact_ref.dtype == np.uint8 and approx_res.dtype == np.uint8, "Images must be uint8."

    mse = float(np.mean((exact_ref.astype(np.float64) - approx_res.astype(np.float64)) ** 2))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))

    if len(exact_ref.shape) == 3 and exact_ref.shape[2] == 3:
        # Multichannel SSIM for RGB
        ssim = float(structural_similarity(exact_ref, approx_res, channel_axis=-1, data_range=255))
    else:
        # 2D Grayscale SSIM
        ssim = float(structural_similarity(exact_ref, approx_res, data_range=255))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }


def compute_pixel_difference_map(exact_img: np.ndarray, approx_img: np.ndarray, amp_factor: float = 15.0):
    """
    Calculates the absolute pixel-by-pixel difference between the exact processed image
    and the approximate processed image.
    
    Amplifies subtle approximation differences for human visualization only.
    The actual processed output is strictly preserved and unmodified.
    
    Returns:
        - diff_gray: uint8 amplified grayscale difference map
        - diff_heatmap_rgb: uint8 high-contrast colored heatmap (Inferno colormap)
        - stats: dict containing raw max_delta, mean_delta, changed_pct, arrays_differ, amp_factor
    """
    raw_diff = np.abs(exact_img.astype(np.int32) - approx_img.astype(np.int32))
    if len(raw_diff.shape) == 3:
        diff_mag = np.max(raw_diff, axis=-1).astype(np.float32)
    else:
        diff_mag = raw_diff.astype(np.float32)

    max_delta = int(diff_mag.max())
    mean_delta = float(diff_mag.mean())
    changed_pct = float((diff_mag > 0).mean() * 100.0)
    arrays_differ = bool(max_delta > 0)

    stats = {
        "max_delta": max_delta,
        "mean_delta": mean_delta,
        "changed_pct": changed_pct,
        "arrays_differ": arrays_differ,
        "amp_factor": amp_factor
    }

    h, w = diff_mag.shape[:2]
    if max_delta == 0:
        zero_gray = np.zeros((h, w), dtype=np.uint8)
        zero_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return zero_gray, zero_rgb, stats

    # Amplify subtle errors (e.g. 1..10 difference values) for visualization
    amplified_gray = np.clip(diff_mag * amp_factor, 0, 255).astype(np.uint8)
    # Convert to RGB heatmap via OpenCV Inferno colormap (BGR to RGB)
    heatmap_bgr = cv2.applyColorMap(amplified_gray, cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return amplified_gray, heatmap_rgb, stats


def to_png_bytes(img_arr: np.ndarray, is_color: bool) -> bytes:
    """
    Converts a numpy image array (RGB or Grayscale uint8) to lossless PNG bytes for download.
    Preserves exact dimensions, channels, and data type.
    """
    if is_color and len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", bgr)
    else:
        success, buf = cv2.imencode(".png", img_arr)
    return buf.tobytes() if success else b""


# ============================================================================
# Streamlit App Configuration & Styling (Warm Ivory / Pastel Whimsical Theme)
# ============================================================================
st.set_page_config(
    page_title="ApproxImage — Application-Aware Approximate Computing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injected Custom CSS for Warm Ivory, Pastel Lilac, Peach, Mint Aesthetics
st.markdown("""
<style>
    /* Warm Ivory Background & Deep Plum Typography */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FFF9F2 !important;
        color: #34263D !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1140px;
    }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFF5EC !important;
        border-right: 1px solid rgba(155, 126, 222, 0.15) !important;
    }

    /* Remove Streamlit default dark header bar and top decoration */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* File Uploader Dropzone & High-Contrast Upload Button */
    [data-testid="stFileUploadDropzone"], 
    [data-testid="stFileUploader"] section,
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed rgba(155, 126, 222, 0.45) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploadDropzone"] *,
    [data-testid="stFileUploader"] section * {
        color: #34263D !important;
    }

    /* High-contrast Upload / Browse Button */
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploader"] button,
    section[data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
        background: #F3EDFC !important;
        color: #34263D !important;
        border: 1.5px solid #9B7EDE !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 8px rgba(155, 126, 222, 0.15) !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover,
    [data-testid="stFileUploader"] button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover,
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
        background: #9B7EDE !important;
        color: #FFFFFF !important;
        border-color: #8A68D6 !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover *,
    [data-testid="stFileUploader"] button:hover *,
    section[data-testid="stFileUploaderDropzone"] button:hover *,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* Ensure button text and internal spans are dark plum and clearly readable */
    [data-testid="stFileUploadDropzone"] button *,
    [data-testid="stFileUploader"] button *,
    section[data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] * {
        color: #34263D !important;
        font-weight: 700 !important;
    }

    /* Upload Icon (SVG) */
    [data-testid="stFileUploadDropzone"] svg,
    [data-testid="stFileUploader"] svg,
    section[data-testid="stFileUploaderDropzone"] svg {
        fill: #6F4DB8 !important;
        color: #6F4DB8 !important;
        stroke: #6F4DB8 !important;
    }
    [data-testid="stFileUploadDropzone"] button svg,
    [data-testid="stFileUploader"] button svg,
    section[data-testid="stFileUploaderDropzone"] button svg {
        fill: #34263D !important;
        color: #34263D !important;
        stroke: #34263D !important;
    }

    /* Prevent expanders from rendering as dark background rectangles */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(155, 126, 222, 0.25) !important;
        border-radius: 14px !important;
    }
    [data-testid="stExpander"] summary {
        color: #34263D !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        color: #34263D !important;
    }

    /* Prevent horizontal rules from rendering as dark bars */
    hr {
        border-color: rgba(155, 126, 222, 0.15) !important;
        background-color: transparent !important;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 1.2rem 1rem 1.6rem 1rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 3.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
        background: linear-gradient(135deg, #8A68D6 0%, #FF8F8F 55%, #F7D774 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-tagline {
        font-size: 1.3rem;
        font-weight: 700;
        color: #4A3557;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #7A6B84;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.5;
    }
    .hero-accent {
        width: 72px;
        height: 4px;
        background: linear-gradient(90deg, #9B7EDE, #FF8F8F, #8FDCC5);
        margin: 1rem auto 0 auto;
        border-radius: 99px;
    }

    /* Pastel Cards */
    .pastel-card {
        background: #FFFFFF;
        border: 1px solid rgba(155, 126, 222, 0.22);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(155, 126, 222, 0.07);
    }
    .pastel-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #34263D;
        margin-bottom: 0.25rem;
    }
    .pastel-card-desc {
        font-size: 0.9rem;
        color: #7A6B84;
        margin-bottom: 0.8rem;
    }

    /* Pill Badges */
    .pill-badge {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        background: rgba(155, 126, 222, 0.12);
        color: #6F4DB8;
        border: 1px solid rgba(155, 126, 222, 0.25);
        margin-right: 0.5rem;
        margin-top: 0.4rem;
    }

    /* Primary Gradient Button */
    .stButton > button {
        background: linear-gradient(135deg, #9B7EDE 0%, #FF8F8F 100%) !important;
        color: #FFFFFF !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2.2rem !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 25px rgba(255, 143, 143, 0.35) !important;
        transition: all 0.25s ease-in-out !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(155, 126, 222, 0.45) !important;
        color: #FFFFFF !important;
    }

    /* Secondary Back Button */
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #8A68D6 !important;
        border: 2px solid #9B7EDE !important;
        border-radius: 999px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        padding: 0.45rem 1.4rem !important;
        box-shadow: 0 4px 15px rgba(155, 126, 222, 0.1) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #9B7EDE !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* Result Comparison Frames */
    .result-frame {
        background: #FFFFFF;
        border: 1.5px solid rgba(155, 126, 222, 0.25);
        border-radius: 18px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 10px 28px rgba(155, 126, 222, 0.08);
    }
    .result-frame-header {
        font-size: 0.98rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }
    .result-frame-sub {
        font-size: 0.85rem;
        color: #7A6B84;
        margin-bottom: 0.8rem;
    }
    .vs-pill {
        display: inline-block;
        background: linear-gradient(135deg, #9B7EDE, #FF8F8F);
        color: white;
        font-weight: 800;
        font-size: 0.95rem;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        box-shadow: 0 4px 15px rgba(155, 126, 222, 0.35);
        margin: 0.5rem auto;
    }

    /* Compact Parameter Boxes Under Images */
    .param-box {
        background: #FFFFFF;
        border: 1px solid rgba(155, 126, 222, 0.22);
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-top: 0.75rem;
        box-shadow: 0 4px 14px rgba(155, 126, 222, 0.05);
    }
    .param-line {
        font-size: 0.88rem;
        color: #4A3557;
        margin: 0.35rem 0;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .param-label {
        color: #523E61;
    }
    .param-label b {
        color: #34263D;
    }
    .param-val {
        font-weight: 800;
        color: #6F4DB8;
        font-size: 0.95rem;
    }
    .param-val-accent {
        font-weight: 800;
        color: #D45B7A;
        font-size: 0.95rem;
    }

    /* Hardware Efficiency Visual Bar Meter */
    .hw-visual-panel {
        background: #FFFFFF;
        border: 1.5px solid rgba(155, 126, 222, 0.22);
        border-radius: 16px;
        padding: 1.3rem;
        margin: 1.2rem 0;
        box-shadow: 0 6px 20px rgba(155, 126, 222, 0.05);
    }
    .meter-bar-bg {
        background: rgba(155, 126, 222, 0.12);
        border-radius: 8px;
        height: 24px;
        width: 100%;
        overflow: hidden;
        margin-top: 0.3rem;
        margin-bottom: 0.8rem;
    }
    .meter-bar-fill-exact {
        background: linear-gradient(90deg, #8A68D6, #A685E2);
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 800;
    }
    .meter-bar-fill-approx {
        background: linear-gradient(90deg, #FF8F8F, #FFB0B0);
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: #34263D;
        font-size: 0.75rem;
        font-weight: 800;
    }
    .hw-stat-badge {
        background: #F8F4FD;
        border: 1px solid rgba(155, 126, 222, 0.3);
        border-radius: 10px;
        padding: 0.5rem 0.9rem;
        font-size: 0.92rem;
        font-weight: 700;
        color: #4A3557;
        display: inline-block;
    }

    /* Metric Cards */
    .metric-card-pastel {
        background: #FFFFFF;
        border: 1px solid rgba(155, 126, 222, 0.2);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(155, 126, 222, 0.06);
    }
    .metric-card-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7A6B84;
    }
    .metric-card-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #34263D;
        margin-top: 0.25rem;
    }
    .metric-card-sub {
        font-size: 0.8rem;
        color: #8C7D96;
        margin-top: 0.2rem;
    }

    /* Quality Badges */
    .badge-excellent {
        color: #1F7A61;
        background: rgba(143, 220, 197, 0.25);
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid rgba(143, 220, 197, 0.5);
    }
    .badge-good {
        color: #6F4DB8;
        background: rgba(155, 126, 222, 0.18);
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid rgba(155, 126, 222, 0.4);
    }
    .badge-acceptable {
        color: #A37A00;
        background: rgba(247, 215, 116, 0.28);
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid rgba(247, 215, 116, 0.5);
    }
    .badge-poor {
        color: #C73838;
        background: rgba(255, 143, 143, 0.25);
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        display: inline-block;
        border: 1px solid rgba(255, 143, 143, 0.5);
    }

    /* Hardware Benefit Cards */
    .hw-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAF5FF 100%);
        border: 1.5px solid rgba(155, 126, 222, 0.25);
        border-radius: 16px;
        padding: 1.3rem 1rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(155, 126, 222, 0.08);
    }
    .hw-val {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8A68D6, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hw-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #523E61;
        margin-top: 0.35rem;
    }

    /* Trade-off Dual Panel */
    .tradeoff-panel {
        background: #FFFFFF;
        border: 1.5px solid rgba(155, 126, 222, 0.22);
        border-radius: 16px;
        padding: 1.3rem;
        margin: 1.2rem 0;
        box-shadow: 0 6px 20px rgba(155, 126, 222, 0.05);
    }

    /* AI Card */
    .ai-pastel-card {
        background: #FFFDF9;
        border-left: 5px solid #9B7EDE;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
        border-top: 1px solid rgba(155, 126, 222, 0.2);
        border-right: 1px solid rgba(155, 126, 222, 0.2);
        border-bottom: 1px solid rgba(155, 126, 222, 0.2);
        box-shadow: 0 6px 20px rgba(155, 126, 222, 0.06);
    }

    /* The Idea Card */
    .idea-pastel-card {
        background: linear-gradient(135deg, #FAF5FF 0%, #FFF5F5 100%);
        border: 1.5px solid rgba(155, 126, 222, 0.25);
        border-radius: 16px;
        padding: 1.2rem 1.6rem;
        margin-top: 2rem;
        box-shadow: 0 6px 20px rgba(155, 126, 222, 0.06);
    }

    /* Form & Radio labels */
    .stSelectbox label, .stRadio label {
        color: #34263D !important;
        font-weight: 700 !important;
    }
    .stMarkdown p, .stMarkdown span {
        color: #4A3557;
    }
</style>
""", unsafe_allow_html=True)

# Verify LUT backend files
verify_lut_files()

# Initialize ML inference engine
try:
    engine = get_inference_engine()
except Exception as e:
    st.error(f"Failed to load AI model: {e}")
    st.stop()


# ============================================================================
# STATE MANAGEMENT
# ============================================================================
if "page" not in st.session_state:
    st.session_state["page"] = "configure"
if "image_data" not in st.session_state:
    st.session_state["image_data"] = None
if "source_name" not in st.session_state:
    st.session_state["source_name"] = None
if "effect_key" not in st.session_state:
    st.session_state["effect_key"] = "blur"
if "effect_label" not in st.session_state:
    st.session_state["effect_label"] = "Blur"
if "mode_str" not in st.session_state:
    st.session_state["mode_str"] = "AI Recommended"
if "ssim_threshold" not in st.session_state:
    st.session_state["ssim_threshold"] = 0.99
if "diff_view_style" not in st.session_state:
    st.session_state["diff_view_style"] = "Heatmap (Inferno)"
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = None
if "sweep_executed" not in st.session_state:
    st.session_state["sweep_executed"] = False


# ============================================================================
# SCREEN 1: CONFIGURATION PAGE
# ============================================================================
def run_adaptive_mode_selection(
    gray_img: np.ndarray,
    eff_key: str,
    target_ssim: float,
    rgb_img: np.ndarray = None,
    is_col: bool = False,
    exact_ref: np.ndarray = None,
    exact_is_col: bool = False,
    exact_label: str = "",
    t_ref_ms: float = 0.0,
    engine: AdaptiveInferenceEngine = None,
    p_unsafe: float = None,
    decision_threshold: float = None,
    precomputed_mode_ssims: dict = None,
    force_old_gate: bool = False
) -> dict:
    """
    Threshold-aware adaptive multiplier mode selector.
    - When target_ssim >= 0.99 (or force_old_gate=True):
        Applies the conservative ML classifier gate (p_unsafe < decision_threshold).
    - When target_ssim < 0.99:
        Bypasses the 0.99-trained classifier gate and directly attempts Mode 3 first with
        real SSIM verification, falling back to Mode 2, Mode 1, then Mode 0.
    """
    if engine is None and (p_unsafe is None or decision_threshold is None):
        engine = get_inference_engine()

    if p_unsafe is None or decision_threshold is None:
        eff_for_ml = "edge" if eff_key == "edge_fine" else eff_key
        rf_pred = engine.predict(gray_img, eff_for_ml)
        p_unsafe = rf_pred["probability_unsafe"]
        threshold = rf_pred["decision_threshold"]
    else:
        threshold = decision_threshold

    if rgb_img is None:
        rgb_img = gray_img

    if exact_ref is None and precomputed_mode_ssims is None:
        t_ref_start = time.perf_counter()
        exact_ref, exact_is_col, exact_label = convolve_multichannel_wrapper(
            rgb_img, gray_img, eff_key, 0, is_col
        )
        t_ref_ms = (time.perf_counter() - t_ref_start) * 1000.0

    target_mode = None
    approx_res = None
    is_out_col = exact_is_col
    out_lbl = exact_label
    proc_ms = 0.0
    metrics = None
    rationale_text = ""
    m3_fail_ssim = None
    m3_recorded_ssim = None

    # Gate logic: classifier gate applies ONLY when target_ssim >= 0.99
    apply_gate = force_old_gate or (target_ssim >= 0.99)
    try_mode3 = (not apply_gate) or (p_unsafe < threshold)

    # Step 1: Check Mode 3
    if try_mode3:
        if precomputed_mode_ssims is not None:
            m3_metrics = {"ssim": float(precomputed_mode_ssims[3]), "mse": 0.0, "psnr": 0.0}
            t_proc = 0.0
            m3_res = None
            m3_col = is_col
            m3_lbl = "Mode 3"
        else:
            t_start = time.perf_counter()
            m3_res, m3_col, m3_lbl = convolve_multichannel_wrapper(
                rgb_img, gray_img, eff_key, 3, is_col
            )
            t_proc = (time.perf_counter() - t_start) * 1000.0
            m3_metrics = compute_multichannel_metrics(exact_ref, m3_res)

        m3_recorded_ssim = m3_metrics["ssim"]

        if m3_metrics["ssim"] >= target_ssim:
            target_mode = 3
            approx_res = m3_res
            is_out_col = m3_col
            out_lbl = m3_lbl
            proc_ms = t_proc
            metrics = m3_metrics
            if not apply_gate:
                rationale_text = (
                    f"Relaxed quality constraint (target SSIM {target_ssim:.2f} < 0.99) bypassed the 0.99-trained classifier gate. "
                    f"Mode 3 directly verified safe (measured SSIM: {m3_metrics['ssim']:.4f} ≥ {target_ssim:.2f}), "
                    f"achieving maximum hardware efficiency (28.42% logic cell reduction)."
                )
            else:
                rationale_text = (
                    f"Mode 3 was predicted safe ({p_unsafe * 100:.1f}% risk < threshold {threshold * 100:.1f}%) "
                    f"via classifier gate (target SSIM ≥ {target_ssim:.2f}) and verified to satisfy the quality constraint "
                    f"(measured SSIM: {m3_metrics['ssim']:.4f}), achieving maximum hardware efficiency (28.42% logic cell reduction)."
                )
        else:
            m3_fail_ssim = m3_metrics["ssim"]

    # Step 2: Evaluate Mode 2
    if target_mode is None:
        if precomputed_mode_ssims is not None:
            m2_metrics = {"ssim": float(precomputed_mode_ssims[2]), "mse": 0.0, "psnr": 0.0}
            t_proc = 0.0
            m2_res = None
            m2_col = is_col
            m2_lbl = "Mode 2"
        else:
            t_start = time.perf_counter()
            m2_res, m2_col, m2_lbl = convolve_multichannel_wrapper(
                rgb_img, gray_img, eff_key, 2, is_col
            )
            t_proc = (time.perf_counter() - t_start) * 1000.0
            m2_metrics = compute_multichannel_metrics(exact_ref, m2_res)

        if m2_metrics["ssim"] >= target_ssim:
            target_mode = 2
            approx_res = m2_res
            is_out_col = m2_col
            out_lbl = m2_lbl
            proc_ms = t_proc
            metrics = m2_metrics
            if apply_gate and p_unsafe >= threshold:
                rationale_text = (
                    f"Classifier gate (target SSIM ≥ {target_ssim:.2f}) predicted Mode 3 to risk violating the quality constraint "
                    f"({p_unsafe * 100:.1f}% risk ≥ threshold {threshold * 100:.1f}%). "
                    f"Mode 2 verified safe (measured SSIM: {m2_metrics['ssim']:.4f}), providing balanced hardware efficiency (13.39% logic cell reduction)."
                )
            elif not apply_gate and m3_fail_ssim is not None:
                rationale_text = (
                    f"Relaxed quality constraint (target SSIM {target_ssim:.2f} < 0.99) bypassed classifier gate. "
                    f"Mode 3 violated target (measured: {m3_fail_ssim:.4f} < {target_ssim:.2f}). "
                    f"Mode 2 directly verified safe (measured SSIM: {m2_metrics['ssim']:.4f}), providing balanced hardware efficiency (13.39% logic cell reduction)."
                )
            else:
                rationale_text = (
                    f"Mode 3 failed the verified SSIM ≥ {target_ssim:.2f} constraint (measured: {m3_fail_ssim:.4f}). "
                    f"Mode 2 verified safe (measured SSIM: {m2_metrics['ssim']:.4f}), providing balanced hardware efficiency (13.39% logic cell reduction)."
                )

    # Step 3: Evaluate Mode 1
    if target_mode is None:
        if precomputed_mode_ssims is not None:
            m1_metrics = {"ssim": float(precomputed_mode_ssims[1]), "mse": 0.0, "psnr": 0.0}
            t_proc = 0.0
            m1_res = None
            m1_col = is_col
            m1_lbl = "Mode 1"
        else:
            t_start = time.perf_counter()
            m1_res, m1_col, m1_lbl = convolve_multichannel_wrapper(
                rgb_img, gray_img, eff_key, 1, is_col
            )
            t_proc = (time.perf_counter() - t_start) * 1000.0
            m1_metrics = compute_multichannel_metrics(exact_ref, m1_res)

        if m1_metrics["ssim"] >= target_ssim:
            target_mode = 1
            approx_res = m1_res
            is_out_col = m1_col
            out_lbl = m1_lbl
            proc_ms = t_proc
            metrics = m1_metrics
            if not apply_gate:
                rationale_text = (
                    f"Relaxed quality constraint path: higher approximation modes (Modes 3 & 2) violated the SSIM ≥ {target_ssim:.2f} requirement. "
                    f"Mode 1 verified safe (measured SSIM: {m1_metrics['ssim']:.4f}), providing low approximation (3.55% logic cell reduction)."
                )
            else:
                rationale_text = (
                    f"Higher approximation modes (Modes 3 & 2) violated the SSIM ≥ {target_ssim:.2f} requirement. "
                    f"Mode 1 verified safe (measured SSIM: {m1_metrics['ssim']:.4f}), providing low approximation (3.55% logic cell reduction)."
                )

    # Step 4: Fallback to Mode 0
    if target_mode is None:
        target_mode = 0
        approx_res = exact_ref
        is_out_col = exact_is_col
        out_lbl = exact_label
        proc_ms = t_ref_ms
        metrics = {"mse": 0.0, "psnr": float("inf"), "ssim": 1.0}
        rationale_text = (
            f"All approximate modes (Modes 1, 2, 3) violated the SSIM ≥ {target_ssim:.2f} quality requirement. "
            f"The system safely selected Exact Mode 0 for maximum precision."
        )

    ai_pred = {
        "selected_mode": target_mode,
        "probability_unsafe": p_unsafe,
        "decision_threshold": threshold,
        "target_ssim_threshold": target_ssim,
        "gate_applied": bool(apply_gate),
        "gate_path": "classifier_gate" if apply_gate else "direct_verification",
        "rationale": rationale_text,
        "mode3_ssim": m3_recorded_ssim
    }

    return {
        "target_mode": target_mode,
        "approx_res": approx_res,
        "is_out_col": is_out_col,
        "out_lbl": out_lbl,
        "proc_ms": proc_ms,
        "metrics": metrics,
        "ai_pred": ai_pred,
        "m3_ssim": m3_recorded_ssim
    }


def render_configuration_screen():
    # 1. Centered Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">ApproxImage</div>
        <div class="hero-tagline">"A little precision. A lot more efficiency."</div>
        <div class="hero-sub">Application-aware approximate computing for efficient image processing.</div>
        <div class="hero-accent"></div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Upload Image Section (Main Page Only)
    st.markdown("""
    <div style="margin-top: 1.2rem; margin-bottom: 0.6rem;">
        <div class="pastel-card-title">Upload an Image</div>
        <div class="pastel-card-desc">Drop your image here or browse • PNG • JPG • JPEG • BMP</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
            raw_decoded = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if raw_decoded is not None:
                if len(raw_decoded.shape) == 3:
                    if raw_decoded.shape[2] == 4:
                        rgb_img = cv2.cvtColor(raw_decoded, cv2.COLOR_BGRA2RGB)
                    else:
                        rgb_img = cv2.cvtColor(raw_decoded, cv2.COLOR_BGR2RGB)
                    is_col = is_color_image(rgb_img)
                    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
                else:
                    is_col = False
                    gray_img = raw_decoded
                    rgb_img = cv2.cvtColor(raw_decoded, cv2.COLOR_GRAY2RGB)

                if rgb_img.dtype != np.uint8:
                    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
                if gray_img.dtype != np.uint8:
                    gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)

                st.session_state["image_data"] = (rgb_img, gray_img, is_col)
                st.session_state["source_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading image file: {e}")

    # Display image metadata pills if image is loaded
    if st.session_state["image_data"] is not None:
        rgb_curr, _, is_col_curr = st.session_state["image_data"]
        name_curr = st.session_state["source_name"] or "image.png"
        format_str = "Color RGB" if is_col_curr else "Grayscale"
        ch_str = "3 channels" if is_col_curr else "1 channel"
        st.markdown(f"""
        <div style="margin-top: 0.6rem; margin-bottom: 0.8rem;">
            <span class="pill-badge">File: {name_curr}</span>
            <span class="pill-badge">Dimensions: {rgb_curr.shape[1]} × {rgb_curr.shape[0]} px</span>
            <span class="pill-badge">Format: {format_str} ({ch_str})</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. Main Processing Section ("Choose Your Processing")
    st.markdown("""
    <div style="margin-top: 1.8rem; margin-bottom: 0.8rem;">
        <div class="pastel-card-title">Choose Your Processing</div>
    </div>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        st.markdown("**Processing Effect**")
        effect_opts = ["Blur", "Edge Detection", "Sharpen", "Emboss"]
        curr_effect_idx = 0
        if st.session_state["effect_label"] in effect_opts:
            curr_effect_idx = effect_opts.index(st.session_state["effect_label"])

        chosen_effect = st.radio(
            "Select Filter Effect",
            options=effect_opts,
            index=curr_effect_idx,
            label_visibility="collapsed"
        )
        st.session_state["effect_label"] = chosen_effect
        EFFECT_KEY_MAP = {
            "Blur": "blur",
            "Edge Detection": "edge",
            "Sharpen": "sharpen",
            "Emboss": "emboss"
        }
        st.session_state["effect_key"] = EFFECT_KEY_MAP[chosen_effect]

    with ctrl_col2:
        st.markdown("**Approximation Mode**")
        mode_opts = [
            "AI Recommended",
            "Exact",
            "Mode 1 — Low Approximation",
            "Mode 2 — Medium Approximation",
            "Mode 3 — High Approximation"
        ]
        curr_mode_idx = 0
        if st.session_state["mode_str"] in mode_opts:
            curr_mode_idx = mode_opts.index(st.session_state["mode_str"])

        mode_captions = [
            "Machine learning dynamic selection",
            "Maximum precision",
            "Small approximation",
            "Balanced efficiency",
            "Maximum hardware efficiency"
        ]
        chosen_mode = st.radio(
            "Select Approximation Level",
            options=mode_opts,
            captions=mode_captions,
            index=curr_mode_idx,
            label_visibility="collapsed"
        )
        st.session_state["mode_str"] = chosen_mode

    st.caption("Higher approximation can reduce hardware complexity, but may increase the difference from the exact result.")

    # Configurable SSIM Quality Constraint (Only shown for AI Recommended mode)
    if chosen_mode == "AI Recommended":
        threshold_options = [0.99, 0.98, 0.95, 0.90]
        curr_thresh = st.session_state.get("ssim_threshold", 0.99)
        curr_t_idx = threshold_options.index(curr_thresh) if curr_thresh in threshold_options else 0
        chosen_thresh = st.selectbox(
            "Target SSIM Quality Threshold",
            options=threshold_options,
            index=curr_t_idx,
            format_func=lambda x: f"SSIM ≥ {x:.2f}" + (" (Strict / Default)" if x == 0.99 else (" (Aggressive)" if x == 0.90 else "")),
            help="Quality threshold used by the adaptive selector to accept an approximate mode."
        )
        st.session_state["ssim_threshold"] = chosen_thresh
        st.caption("Lower threshold allows more aggressive approximation. Does not change the multiplier hardware.")

    st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)

    # 4. Primary Process Button
    process_btn = st.button("Process Image →", type="primary", use_container_width=True)

    if process_btn:
        st.session_state["sweep_executed"] = False
        if st.session_state["image_data"] is None:
            st.warning("Please upload an image above to run approximate processing.")
            return

        rgb_img, gray_img, is_col = st.session_state["image_data"]
        eff_key = st.session_state["effect_key"]
        eff_lbl = st.session_state["effect_label"]
        mode_choice = st.session_state["mode_str"]

        with st.spinner("Computing discrete convolution via verified hardware multiplier LUTs..."):
            # 1. Exact reference (Mode 0) Ground Truth
            t_ref_start = time.perf_counter()
            exact_ref, exact_is_col, exact_label = convolve_multichannel_wrapper(
                rgb_img, gray_img, eff_key, 0, is_col
            )
            t_ref_ms = (time.perf_counter() - t_ref_start) * 1000.0

            # 2. Determine target multiplier mode & run convolution
            ai_pred = None
            target_ssim = st.session_state.get("ssim_threshold", 0.99)
            if mode_choice == "AI Recommended":
                sel = run_adaptive_mode_selection(
                    gray_img=gray_img,
                    eff_key=eff_key,
                    target_ssim=target_ssim,
                    rgb_img=rgb_img,
                    is_col=is_col,
                    exact_ref=exact_ref,
                    exact_is_col=exact_is_col,
                    exact_label=exact_label,
                    t_ref_ms=t_ref_ms,
                    engine=engine
                )
                target_mode = sel["target_mode"]
                approx_res = sel["approx_res"]
                is_out_col = sel["is_out_col"]
                out_lbl = sel["out_lbl"]
                proc_ms = sel["proc_ms"]
                metrics = sel["metrics"]
                ai_pred = sel["ai_pred"]
            else:
                MODE_MAP = {
                    "Exact": 0,
                    "Mode 1 — Low Approximation": 1,
                    "Mode 2 — Medium Approximation": 2,
                    "Mode 3 — High Approximation": 3
                }
                target_mode = MODE_MAP[mode_choice]
                if target_mode == 0:
                    approx_res = exact_ref
                    is_out_col = exact_is_col
                    out_lbl = exact_label
                    proc_ms = t_ref_ms
                    metrics = {"mse": 0.0, "psnr": float("inf"), "ssim": 1.0}
                else:
                    t_proc_start = time.perf_counter()
                    approx_res, is_out_col, out_lbl = convolve_multichannel_wrapper(
                        rgb_img, gray_img, eff_key, target_mode, is_col
                    )
                    proc_ms = (time.perf_counter() - t_proc_start) * 1000.0
                    metrics = compute_multichannel_metrics(exact_ref, approx_res)

            # Store processed payload
            st.session_state["processed_results"] = {
                "source": st.session_state["source_name"],
                "effect_key": eff_key,
                "effect_label": eff_lbl,
                "mode": target_mode,
                "mode_choice": mode_choice,
                "target_ssim_threshold": target_ssim,
                "is_color_input": is_col,
                "is_output_color": is_out_col,
                "output_label": out_lbl,
                "exact_ref": exact_ref,
                "approx_result": approx_res,
                "metrics": metrics,
                "proc_time_ms": proc_ms,
                "exact_proc_time_ms": t_ref_ms,
                "ai_prediction": ai_pred
            }

            # Transition to Screen 2
            st.session_state["page"] = "results"
            st.rerun()


# ============================================================================
# SCREEN 2: RESULTS PAGE
# ============================================================================
def render_results_screen():
    res = st.session_state.get("processed_results")
    if res is None or st.session_state["image_data"] is None:
        st.session_state["page"] = "configure"
        st.rerun()
        return

    rgb_img, gray_img, is_col = st.session_state["image_data"]
    hw_spec = HARDWARE_SPECS[res["mode"]]
    metrics = res["metrics"]
    ssim_val = metrics["ssim"]
    quality_label, quality_class = classify_quality(ssim_val)
    exact_time_ms = res.get("exact_proc_time_ms", 0.0)
    target_ssim = res.get("target_ssim_threshold", 0.99)
    psnr_str = "∞" if metrics["psnr"] == float("inf") else f"{metrics['psnr']:.2f} dB"

    # 1. Back to Configuration Button
    if st.button("← Back to Configuration", type="secondary"):
        st.session_state["page"] = "configure"
        st.session_state["sweep_executed"] = False
        st.rerun()

    # Centered Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem; margin-top: 0.5rem;">
        <div style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #8A68D6, #FF8F8F); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Approximation Results
        </div>
        <div style="color: #7A6B84; font-size: 0.98rem; margin-top: 0.2rem;">
            See what changed — and what we gained.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compute Amplified Difference Map for visualization
    diff_gray, diff_heatmap, diff_stats = compute_pixel_difference_map(res["exact_ref"], res["approx_result"], amp_factor=15.0)

    # Header and difference map style toggle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 0.9rem;">
        <span class="vs-pill">EXACT RESULT vs APPROXIMATE RESULT & AMPLIFIED DIFFERENCE MAP</span>
    </div>
    """, unsafe_allow_html=True)

    c_toggle1, c_toggle2 = st.columns([2, 1])
    with c_toggle1:
        st.caption("Side-by-side comparison with an amplified difference map to reveal subtle approximation errors.")
    with c_toggle2:
        diff_style_choice = st.radio(
            "Difference View Style",
            options=["Heatmap (Inferno)", "Grayscale (15×)"],
            horizontal=True,
            index=0 if "Heatmap" in st.session_state.get("diff_view_style", "Heatmap") else 1,
            label_visibility="collapsed"
        )
        st.session_state["diff_view_style"] = diff_style_choice

    active_diff_img = diff_heatmap if "Heatmap" in diff_style_choice else diff_gray

    # PRIMARY 2-COLUMN DISPLAY: Exact | Approximate (each with param-box below)
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("""
        <div class="result-frame">
            <div class="result-frame-header" style="color: #7B5BC6;">Exact Result</div>
            <div class="result-frame-sub">Mode 0 — Ground-Truth Reference</div>
        </div>
        """, unsafe_allow_html=True)
        if res["is_output_color"]:
            st.image(res["exact_ref"], use_container_width=True, channels="RGB")
        elif res["effect_key"] in ("edge", "edge_fine"):
            ref_disp = cv2.normalize(res["exact_ref"], None, 0, 255, cv2.NORM_MINMAX) if res["exact_ref"].max() > res["exact_ref"].min() else res["exact_ref"]
            st.image(ref_disp, use_container_width=True, clamp=True)
        else:
            st.image(res["exact_ref"], use_container_width=True, clamp=True)
        # Technical info for Mode 0 (exact reference)
        exact_spec = HARDWARE_SPECS[0]
        st.markdown(f"""
        <div class="param-box">
            <div class="param-line"><span class="param-label"><b>Mode:</b></span> <span class="param-val-accent">Mode 0 — Exact</span></div>
            <div class="param-line"><span class="param-label"><b>Logic Cells:</b></span> <span class="param-val">{exact_spec['cells']} cells</span></div>
            <div class="param-line"><span class="param-label"><b>Critical Path:</b></span> <span class="param-val">{exact_spec['critical_path']} logic levels</span></div>
            <div class="param-line"><span class="param-label"><b>Approximation:</b></span> <span class="param-val">None — Full precision</span></div>
            <div class="param-line"><span class="param-label"><b>Description:</b></span> <span class="param-val">{exact_spec['desc']}</span></div>
            <div class="param-line"><span class="param-label"><b>SSIM vs. itself:</b></span> <span class="param-val">1.0000 (Ground Truth)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        mode_header_desc = f"Mode {res['mode']} — {hw_spec['level']}" if res["mode"] != 0 else "Mode 0 — Exact Reference"
        st.markdown(f"""
        <div class="result-frame">
            <div class="result-frame-header" style="color: #D45B7A;">Approximate Result</div>
            <div class="result-frame-sub">{mode_header_desc} (Real Output)</div>
        </div>
        """, unsafe_allow_html=True)
        if res["is_output_color"]:
            st.image(res["approx_result"], use_container_width=True, channels="RGB")
        elif res["effect_key"] in ("edge", "edge_fine"):
            approx_disp = cv2.normalize(res["approx_result"], None, 0, 255, cv2.NORM_MINMAX) if res["approx_result"].max() > res["approx_result"].min() else res["approx_result"]
            st.image(approx_disp, use_container_width=True, clamp=True)
        else:
            st.image(res["approx_result"], use_container_width=True, clamp=True)
        # Technical info for the selected approximate mode
        st.markdown(f"""
        <div class="param-box">
            <div class="param-line"><span class="param-label"><b>Mode:</b></span> <span class="param-val-accent">{hw_spec['label']}</span></div>
            <div class="param-line"><span class="param-label"><b>Logic Cells:</b></span> <span class="param-val">{hw_spec['cells']} cells ({hw_spec['reduction_str']} reduction)</span></div>
            <div class="param-line"><span class="param-label"><b>Critical Path:</b></span> <span class="param-val">{hw_spec['critical_path']} levels — {hw_spec['timing_change_str']}</span></div>
            <div class="param-line"><span class="param-label"><b>Approximation:</b></span> <span class="param-val">{hw_spec['desc']}</span></div>
            <div class="param-line"><span class="param-label"><b>Cells Saved:</b></span> <span class="param-val">{hw_spec['saved_cells']} vs Mode 0</span></div>
            <div class="param-line"><span class="param-label"><b>SSIM vs Mode 0:</b></span> <span class="param-val">{ssim_val:.4f} ({ssim_val * 100:.2f}%)</span></div>
        </div>
        """, unsafe_allow_html=True)

    # AMPLIFIED DIFFERENCE MAP (below the two results)
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    diff_col_img, diff_col_stats = st.columns([3, 2])

    with diff_col_img:
        st.markdown("""
        <div class="result-frame">
            <div class="result-frame-header" style="color: #E06C3B;">Amplified Difference Map</div>
            <div class="result-frame-sub">Visualization Only • 15× Magnified (max across colour channels for RGB)</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(active_diff_img, use_container_width=True, clamp=True)

    with diff_col_stats:
        diff_status_str = f"Differs ({diff_stats['changed_pct']:.1f}% px)" if diff_stats["arrays_differ"] else "Identical (0% diff)"
        st.markdown(f"""
        <div class="param-box" style="margin-top: 2.8rem;">
            <div class="param-line"><span class="param-label"><b>Max Pixel Delta:</b></span> <span class="param-val-accent">{diff_stats['max_delta']} levels</span></div>
            <div class="param-line"><span class="param-label"><b>Mean Abs Delta:</b></span> <span class="param-val">{diff_stats['mean_delta']:.4f} levels</span></div>
            <div class="param-line"><span class="param-label"><b>Modified Pixels:</b></span> <span class="param-val">{diff_stats['changed_pct']:.2f}%</span></div>
            <div class="param-line"><span class="param-label"><b>Output Arrays:</b></span> <span class="param-val" style="color: {'#D45B7A' if diff_stats['arrays_differ'] else '#1F7A61'};">{diff_status_str}</span></div>
            <div class="param-line"><span class="param-label"><b>Amplification:</b></span> <span class="param-val">15× Visualization</span></div>
            <div style="font-size: 0.78rem; color: #8C7D96; margin-top: 0.45rem; font-style: italic; border-top: 1px dashed rgba(155, 126, 222, 0.2); padding-top: 0.35rem;">
                Amplified for visual contrast only; real output is not altered.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ONE compact metrics row (SSIM, PSNR, MSE, logic cells saved, logic depth)
    st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.8rem;"></div>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card-pastel">
            <div class="metric-card-label">SSIM Similarity</div>
            <div class="metric-card-val">{ssim_val * 100:.2f}%</div>
            <div class="metric-card-sub">{ssim_val:.4f} vs Mode 0</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card-pastel">
            <div class="metric-card-label">PSNR Quality</div>
            <div class="metric-card-val">{psnr_str}</div>
            <div class="metric-card-sub">Signal Quality</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card-pastel">
            <div class="metric-card-label">MSE Difference</div>
            <div class="metric-card-val">{metrics['mse']:.4f}</div>
            <div class="metric-card-sub">Pixel MSE</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card-pastel">
            <div class="metric-card-label">Logic Cells Saved</div>
            <div class="metric-card-val" style="color: #7B5BC6;">{hw_spec['saved_cells']}</div>
            <div class="metric-card-sub">{hw_spec['reduction_str']} reduction</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="metric-card-pastel">
            <div class="metric-card-label">Logic Depth</div>
            <div class="metric-card-val" style="color: #D45B7A;">{hw_spec['critical_path']}</div>
            <div class="metric-card-sub">Critical Path (Exact: 40)</div>
        </div>
        """, unsafe_allow_html=True)

    # Visible Download Buttons
    st.markdown('<div style="margin-top: 1rem; margin-bottom: 1.2rem;"></div>', unsafe_allow_html=True)
    approx_bytes = to_png_bytes(res["approx_result"], res["is_output_color"])
    exact_bytes = to_png_bytes(res["exact_ref"], res["is_output_color"])
    diff_bytes = to_png_bytes(active_diff_img, len(active_diff_img.shape) == 3 and active_diff_img.shape[2] == 3)

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            label="Download Exact Reference",
            data=exact_bytes,
            file_name=f"approximage_mode0_exact_{res['effect_key']}.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="Download Approximate Image",
            data=approx_bytes,
            file_name=f"approximage_mode{res['mode']}_{res['effect_key']}.png",
            mime="image/png",
            type="primary",
            use_container_width=True
        )
    with dl_col3:
        st.download_button(
            label="Download Difference Map",
            data=diff_bytes,
            file_name=f"approximage_diff_mode{res['mode']}_{res['effect_key']}.png",
            mime="image/png",
            use_container_width=True
        )

    # AI Recommended card (mode and one sentence; rationale inside expander)
    if res["mode_choice"] == "AI Recommended" and res.get("ai_prediction"):
        ai_p = res["ai_prediction"]
        p_unsafe_pct = ai_p["probability_unsafe"] * 100.0
        thresh_pct = ai_p["decision_threshold"] * 100.0

        MODE_TAGS = {
            3: "Mode 3 — Maximum Hardware Efficiency",
            2: "Mode 2 — Balanced Efficiency",
            1: "Mode 1 — Low Approximation",
            0: "Exact — Maximum Precision"
        }
        mode_tag = MODE_TAGS.get(res["mode"], f"Mode {res['mode']}")
        rationale_text = ai_p.get("rationale", "")

        st.markdown(f"""
        <div class="ai-pastel-card">
            <div style="font-size: 0.85rem; font-weight: 800; color: #7B5BC6; text-transform: uppercase; letter-spacing: 0.06em;">AI Recommended Mode</div>
            <div style="color: #34263D; font-weight: 800; font-size: 1.15rem; margin-top: 0.2rem;">{mode_tag}</div>
            <div style="color: #4A3557; font-size: 0.92rem; margin-top: 0.4rem;">Selected as the highest-efficiency mode that satisfies the SSIM ≥ {target_ssim:.2f} quality requirement.</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Why this mode?", expanded=False):
            st.markdown(f"""
            <div style="font-size: 0.88rem; color: #4A3557; line-height: 1.6;">
                {rationale_text}<br><br>
                <b>Predicted Mode-3 Unsafe Risk:</b> {p_unsafe_pct:.1f}% (Safety Threshold: {thresh_pct:.1f}%)
            </div>
            """, unsafe_allow_html=True)

    # Expanders for secondary/detailed views
    with st.expander("Original input", expanded=False):
        in_c1, in_c2 = st.columns([1, 2])
        with in_c1:
            st.image(rgb_img, use_container_width=True, channels="RGB")
        with in_c2:
            st.markdown(f"**Filename:** `{res['source']}`")
            st.markdown(f"**Dimensions:** `{rgb_img.shape[1]} × {rgb_img.shape[0]} px`")
            st.markdown(f"**Format:** `{'Color RGB' if is_col else 'Grayscale'}`")
            st.caption("The original uploaded image is the input. Approximation quality is evaluated strictly by comparing the approximate output against the exact Mode 0 processed result.")

    with st.expander("Hardware details", expanded=False):
        hw_pct = (hw_spec["cells"] / 366.0) * 100.0

        if res["mode"] == 0:
            cp_summary = "Baseline reference (40)"
            cp_color = "#523E61"
        elif res["mode"] == 3:
            cp_summary = "27.5% shorter (29 vs 40)"
            cp_color = "#1F7A61"
        elif res["mode"] == 2:
            cp_summary = "15% shorter (34 vs 40)"
            cp_color = "#1F7A61"
        elif res["mode"] == 1:
            cp_summary = "2.5% longer (41 vs 40)"
            cp_color = "#A37A00"

        st.markdown(f"""
        <div class="hw-visual-panel">
            <div style="margin-bottom: 0.9rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 700; color: #523E61; margin-bottom: 0.2rem;">
                    <span>EXACT (Mode 0)</span>
                    <span>366 Logic Cells</span>
                </div>
                <div class="meter-bar-bg">
                    <div class="meter-bar-fill-exact" style="width: 100%;">366 Logic Cells (100%)</div>
                </div>
            </div>

            <div style="margin-bottom: 0.9rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 700; color: #523E61; margin-bottom: 0.2rem;">
                    <span>APPROXIMATE ({hw_spec['short_label']})</span>
                    <span>{hw_spec['cells']} Logic Cells</span>
                </div>
                <div class="meter-bar-bg">
                    <div class="meter-bar-fill-approx" style="width: {hw_pct:.1f}%;">{hw_spec['cells']} Logic Cells ({hw_pct:.1f}%)</div>
                </div>
            </div>

            <div style="display: flex; gap: 0.8rem; margin-top: 0.8rem; margin-bottom: 1.1rem; flex-wrap: wrap;">
                <div class="hw-stat-badge">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #7B5BC6;">{hw_spec['saved_cells']}</span> fewer hardware blocks (Cells Saved)
                </div>
                <div class="hw-stat-badge">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #D45B7A;">{hw_spec['reduction_str']}</span> lower logic usage (Logic-Cell Reduction)
                </div>
            </div>

            <div style="border-top: 1px solid rgba(155, 126, 222, 0.2); padding-top: 0.9rem;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #523E61; margin-bottom: 0.35rem;">
                    Processing Delay (Critical Path)
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: #FAF7FD; padding: 0.6rem 1rem; border-radius: 10px; border: 1px solid rgba(155, 126, 222, 0.15);">
                    <div style="font-size: 0.88rem; color: #523E61;">
                        <b>Exact:</b> 40 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Approximate:</b> {hw_spec['critical_path']}
                    </div>
                    <div style="font-size: 0.92rem; font-weight: 800; color: {cp_color};">
                        {cp_summary}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Technical Specifications**")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(f"**Filter Operation:** `{res['effect_label']}` (`{res['effect_key']}`)")
            st.markdown(f"**Selected Approximation Mode:** `Mode {res['mode']}` — {hw_spec['desc']}")
            st.markdown(f"**Output Format:** `{res['output_label']}`")
            st.markdown(f"**Image Dimensions:** `{rgb_img.shape[1]} × {rgb_img.shape[0]} px`")
            st.markdown(f"**Mean Squared Error (MSE):** `{metrics['mse']:.6f}`")
            st.markdown(f"**Peak Signal-to-Noise Ratio (PSNR):** `{psnr_str}`")
            st.markdown(f"**Structural Similarity (SSIM):** `{metrics['ssim']:.6f}`")
            st.markdown(f"**Channels:** `{'3 RGB channels' if res['is_output_color'] else '1 grayscale channel'}`")
        with t_col2:
            if res.get("ai_prediction"):
                st.markdown(f"**AI Predicted Mode-3 Unsafe Risk:** `{res['ai_prediction']['probability_unsafe'] * 100:.2f}%`")
                st.markdown(f"**AI Safety Threshold:** `{res['ai_prediction']['decision_threshold'] * 100:.2f}%`")
                st.markdown(f"**AI Decision Flow:** `ML Mode 3 Risk → Verified SSIM Quality Cascade`")
            else:
                st.markdown("**Mode Selection:** `Manual Selection`")
                st.markdown("**AI Decision Threshold:** `N/A (User chosen)`")
            st.markdown(f"**Exact Logic Cells:** `366 cells`")
            st.markdown(f"**Approximate Logic Cells:** `{hw_spec['cells']} cells`")
            st.markdown(f"**Cells Saved:** `{hw_spec['saved_cells']} cells`")
            st.markdown(f"**Logic Reduction:** `{hw_spec['reduction_str']}`")
            st.markdown(f"**Exact Critical Path:** `40`")
            st.markdown(f"**Approximate Critical Path:** `{hw_spec['critical_path']}` ({hw_spec['timing_eval']})")
            st.markdown(f"**Processing Time:** `{res['proc_time_ms']:.2f} ms`")

    with st.expander("Compare all four modes", expanded=False):
        st.markdown("""
        <div style="color: #7A6B84; font-size: 0.9rem; margin-bottom: 0.8rem;">
            Direct side-by-side comparison of <b>Modes 0, 1, 2, and 3</b> on the same input image. Inspect both the processed outputs and the amplified difference maps vs. Exact Mode 0.
        </div>
        """, unsafe_allow_html=True)

        run_sweep = st.button("Run Comparison Across All 4 Modes", key="btn_run_sweep")
        if run_sweep or st.session_state.get("sweep_executed", False):
            st.session_state["sweep_executed"] = True
            with st.spinner("Executing 4-mode convolution sweep across multiplier LUTs..."):
                sweep_rows = []
                exact_sweep, _, _ = convolve_multichannel_wrapper(
                    rgb_img, gray_img, res["effect_key"], 0, is_col
                )

                sweep_images = []
                sweep_diff_heatmaps = []
                sweep_metrics = []
                sweep_diff_stats = []

                for m in range(4):
                    if m == 0:
                        approx_m = exact_sweep
                        m_met = {"mse": 0.0, "psnr": float("inf"), "ssim": 1.0}
                    else:
                        approx_m, _, _ = convolve_multichannel_wrapper(
                            rgb_img, gray_img, res["effect_key"], m, is_col
                        )
                        m_met = compute_multichannel_metrics(exact_sweep, approx_m)

                    spec = HARDWARE_SPECS[m]
                    q_lbl, _ = classify_quality(m_met["ssim"])
                    _, m_heatmap, m_stats = compute_pixel_difference_map(exact_sweep, approx_m, amp_factor=15.0)

                    sweep_images.append(approx_m)
                    sweep_diff_heatmaps.append(m_heatmap)
                    sweep_metrics.append((m_met, spec, q_lbl))
                    sweep_diff_stats.append(m_stats)

                    diff_status_str = f"Differs ({m_stats['changed_pct']:.1f}% px)" if m_stats["arrays_differ"] else "Identical (0%)"

                    sweep_rows.append({
                        "Mode": f"Mode {m}",
                        "Approximation Level": spec["level"],
                        "SSIM vs Exact": f"{m_met['ssim']:.4f}",
                        "Quality": q_lbl,
                        "MSE": f"{m_met['mse']:.4f}",
                        "PSNR (dB)": "∞" if m_met["psnr"] == float("inf") else f"{m_met['psnr']:.2f}",
                        "Arrays Differ?": diff_status_str,
                        "Max Pixel Delta": m_stats["max_delta"],
                        "Logic Cells": spec["cells"],
                        "Logic Reduction": spec["reduction_str"],
                        "Critical Path": spec["critical_path"]
                    })

                # Show side-by-side images across all 4 modes
                st.markdown("**Processed Images (Real Unaltered Outputs):**")
                cmp_cols = st.columns(4)
                for m in range(4):
                    with cmp_cols[m]:
                        m_met, spec, q_lbl = sweep_metrics[m]
                        m_stats = sweep_diff_stats[m]
                        st.markdown(f"**Mode {m}** ({spec['short_label']})")
                        if res["is_output_color"]:
                            st.image(sweep_images[m], use_container_width=True, channels="RGB")
                        elif res["effect_key"] in ("edge", "edge_fine"):
                            disp = cv2.normalize(sweep_images[m], None, 0, 255, cv2.NORM_MINMAX) if sweep_images[m].max() > sweep_images[m].min() else sweep_images[m]
                            st.image(disp, use_container_width=True, clamp=True)
                        else:
                            st.image(sweep_images[m], use_container_width=True, clamp=True)
                        st.caption(f"SSIM: {m_met['ssim']:.4f} • Cells: {spec['cells']} ({spec['reduction_str']} red.)")

                # Show side-by-side Amplified Difference Maps
                st.markdown("**Amplified Difference Maps vs Exact Mode 0 (15× Magnified):**")
                diff_cols = st.columns(4)
                for m in range(4):
                    with diff_cols[m]:
                        m_met, spec, q_lbl = sweep_metrics[m]
                        m_stats = sweep_diff_stats[m]
                        st.markdown(f"**Mode {m} vs Mode 0**")
                        st.image(sweep_diff_heatmaps[m], use_container_width=True, clamp=True)
                        diff_label = f"Max Δ: {m_stats['max_delta']} | Changed: {m_stats['changed_pct']:.1f}%" if m > 0 else "Baseline: Identical (Δ = 0)"
                        st.caption(diff_label)

                df_sweep = pd.DataFrame(sweep_rows)
                st.dataframe(df_sweep, use_container_width=True, hide_index=True)

    # 11. The Idea Card
    st.markdown("""
    <div class="idea-pastel-card">
        <div style="font-weight: 800; color: #7B5BC6; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;">
            The Idea
        </div>
        <div style="color: #34263D; font-size: 0.93rem; line-height: 1.55;">
            ApproxImage replaces unnecessary arithmetic precision with controlled approximation. Lower-significance multiplier logic is simplified to reduce hardware complexity, while the exact implementation is used as the reference to measure image-quality impact.
        </div>
    </div>
    """, unsafe_allow_html=True)




# ============================================================================
# MINIMAL SIDEBAR (NO CLUTTER, NO EMOJIS)
# ============================================================================
with st.sidebar:
    st.markdown("### ApproxImage")
    st.caption("Application-Aware Approximate Computing")
    st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)
    st.caption("Final-Year Research Project • 2026")


# ============================================================================
# ROUTER
# ============================================================================
if st.session_state.get("page") == "results" and st.session_state.get("processed_results") is not None:
    render_results_screen()
else:
    render_configuration_screen()
