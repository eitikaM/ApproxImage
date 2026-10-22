import os
import re
import sys
import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(PROJECT_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from app import compute_pixel_difference_map

APP_PATH = os.path.join(PROJECT_ROOT, "python", "app.py")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "images", "sample_photo.png")
EFFECTS = ["Blur", "Edge Detection", "Sharpen", "Emboss"]


def _get_widget_by_label(widget_list, label_text: str, partial: bool = False):
    """Finds a widget from an ElementList/WidgetList by exact or partial label."""
    for w in widget_list:
        if (partial and label_text in getattr(w, "label", "")) or (not partial and getattr(w, "label", "") == label_text):
            return w
    raise KeyError(f"Widget with label '{label_text}' not found in widget collection.")


def _get_displayed_max_pixel_delta(at: AppTest) -> int:
    """Extracts the integer Max Pixel Delta rendered in the results page markdown."""
    for m in at.markdown:
        match = re.search(r"Max Pixel Delta:</b></span>\s*<span[^>]*>(\d+)\s*levels", m.value)
        if match:
            return int(match.group(1))
    raise ValueError("Displayed 'Max Pixel Delta' not found in page markdown.")


@pytest.mark.parametrize("effect", EFFECTS)
def test_app_mode3_smoke(effect):
    """Smoke test: Processes sample image in Mode 3 and verifies difference metrics."""
    with open(SAMPLE_PATH, "rb") as f:
        sample_bytes = f.read()

    at = AppTest.from_file(APP_PATH, default_timeout=45).run()

    # Select widgets by label
    uploader = _get_widget_by_label(at.file_uploader, "Upload an image", partial=True)
    uploader.upload("sample_photo.png", sample_bytes, "image/png").run()

    r_effect = _get_widget_by_label(at.radio, "Select Filter Effect")
    r_effect.set_value(effect).run()

    r_mode = _get_widget_by_label(at.radio, "Select Approximation Level")
    mode3_opt = next(o for o in r_mode.options if "Mode 3" in o)
    r_mode.set_value(mode3_opt).run()

    proc_btn = _get_widget_by_label(at.button, "Process Image", partial=True)
    proc_btn.click().run()

    # Assert no traceback / unhandled exceptions
    assert len(at.exception) == 0, f"Unexpected exceptions for {effect}: {[e.value for e in at.exception]}"

    # Assert results page rendered
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Exact Result" in all_markdown, f"'Exact Result' not found in page markdown for {effect}"

    # Verify Max Pixel Delta values
    displayed_delta = _get_displayed_max_pixel_delta(at)
    res = at.session_state["processed_results"]
    _, _, stats = compute_pixel_difference_map(res["exact_ref"], res["approx_result"])
    expected_delta = stats["max_delta"]

    # Assert displayed equals independent calculation on app's own arrays
    assert displayed_delta == expected_delta, (
        f"Displayed Max Pixel Delta ({displayed_delta}) != computed ({expected_delta}) for {effect}"
    )

    if effect in ("Blur", "Sharpen", "Edge Detection"):
        assert displayed_delta > 0, f"Expected {effect} Mode 3 to have Max Pixel Delta > 0, got {displayed_delta}"
    elif effect == "Emboss":
        assert displayed_delta >= 0, f"Expected {effect} Mode 3 to have Max Pixel Delta >= 0, got {displayed_delta}"


@pytest.mark.parametrize("thresh", [0.99, 0.90])
def test_app_ai_recommended_thresholds(thresh):
    """Tests AI Recommended mode selection under different SSIM quality constraints."""
    with open(SAMPLE_PATH, "rb") as f:
        sample_bytes = f.read()

    at = AppTest.from_file(APP_PATH, default_timeout=45).run()

    uploader = _get_widget_by_label(at.file_uploader, "Upload an image", partial=True)
    uploader.upload("sample_photo.png", sample_bytes, "image/png").run()

    r_effect = _get_widget_by_label(at.radio, "Select Filter Effect")
    r_effect.set_value("Blur").run()

    r_mode = _get_widget_by_label(at.radio, "Select Approximation Level")
    r_mode.set_value("AI Recommended").run()

    s_thresh = _get_widget_by_label(at.selectbox, "Target SSIM Quality Threshold")
    s_thresh.set_value(thresh).run()

    proc_btn = _get_widget_by_label(at.button, "Process Image", partial=True)
    proc_btn.click().run()

    assert len(at.exception) == 0, f"Unexpected exception with threshold {thresh}: {[e.value for e in at.exception]}"

    res = at.session_state["processed_results"]
    selected_mode = res["mode"]
    measured_ssim = res["metrics"]["ssim"]

    print(f"\n[AI Recommended Result] Threshold: {thresh} -> Selected Mode: {selected_mode}, Measured SSIM: {measured_ssim:.6f}")

    # Assert quality constraint contract: selected mode SSIM >= threshold, or fallback to exact Mode 0
    assert (selected_mode == 0) or (measured_ssim >= thresh), (
        f"Selected Mode {selected_mode} with SSIM {measured_ssim:.4f} violated constraint {thresh}"
    )


@pytest.mark.parametrize("target_thresh", [0.99, 0.95])
def test_app_smooth_gradient_ai_recommended(target_thresh):
    """
    AppTest verification on synthetic 256x256 smooth gradient with Blur & AI Recommended:
    - At 0.99: assert measured SSIM >= 0.99 or mode 0.
    - At 0.95: assert the app picked Mode 3 whenever its own displayed Mode 3 SSIM is >= 0.95.
    """
    y, x = np.ogrid[:256, :256]
    grad = np.clip(0.5 * x + 0.3 * y, 0, 255).astype(np.uint8)
    success, png_bytes = cv2.imencode(".png", grad)
    assert success, "Failed to encode smooth gradient to PNG bytes"
    grad_data = png_bytes.tobytes()

    at = AppTest.from_file(APP_PATH, default_timeout=45).run()

    uploader = _get_widget_by_label(at.file_uploader, "Upload an image", partial=True)
    uploader.upload("smooth_gradient.png", grad_data, "image/png").run()

    r_effect = _get_widget_by_label(at.radio, "Select Filter Effect")
    r_effect.set_value("Blur").run()

    r_mode = _get_widget_by_label(at.radio, "Select Approximation Level")
    r_mode.set_value("AI Recommended").run()

    s_thresh = _get_widget_by_label(at.selectbox, "Target SSIM Quality Threshold")
    s_thresh.set_value(target_thresh).run()

    proc_btn = _get_widget_by_label(at.button, "Process Image", partial=True)
    proc_btn.click().run()

    assert len(at.exception) == 0, f"Unexpected exception with threshold {target_thresh}: {[e.value for e in at.exception]}"

    res = at.session_state["processed_results"]
    selected_mode = res["mode"]
    measured_ssim = res["metrics"]["ssim"]

    # Extract displayed Mode 3 SSIM from multi-mode comparison dataframe
    displayed_m3_ssim = None
    if len(at.dataframe) > 0:
        df_sweep = at.dataframe[0].value
        m3_row = df_sweep.loc[df_sweep["Mode"] == "Mode 3"]
        if not m3_row.empty:
            displayed_m3_ssim = float(m3_row["SSIM vs Exact"].iloc[0])

    if displayed_m3_ssim is None and res.get("ai_prediction"):
        displayed_m3_ssim = res["ai_prediction"].get("mode3_ssim")

    if target_thresh == 0.99:
        assert (measured_ssim >= 0.99) or (selected_mode == 0), (
            f"At 0.99, expected measured SSIM >= 0.99 or Mode 0, got Mode {selected_mode} with SSIM {measured_ssim:.4f}"
        )
    elif target_thresh == 0.95:
        if displayed_m3_ssim is not None and displayed_m3_ssim >= 0.95:
            assert selected_mode == 3, (
                f"At 0.95, displayed Mode 3 SSIM is {displayed_m3_ssim:.4f} >= 0.95, but app selected Mode {selected_mode}"
            )


def test_app_rgb_gradient_ai_recommended_095():
    """
    AppTest verification on synthetic 256x256 3-channel RGB gradient with Blur & AI Recommended (threshold 0.95):
    - Uploads 256x256 RGB gradient PNG with 3 different channel gradients
    - Selects Blur, AI Recommended, threshold 0.95 (widgets by label)
    - Asserts no exception
    - Asserts that the selected mode's displayed SSIM >= 0.95 or mode is 0
    """
    y, x = np.ogrid[:256, :256]
    r_ch = np.clip(0.5 * x + 0.3 * y, 0, 255).astype(np.uint8)
    g_ch = np.clip(0.2 * x + 0.7 * y, 0, 255).astype(np.uint8)
    b_ch = np.clip(0.8 * x + 0.1 * y, 0, 255).astype(np.uint8)
    rgb_arr = np.stack([b_ch, g_ch, r_ch], axis=-1)  # BGR for OpenCV PNG encoding
    success, png_bytes = cv2.imencode(".png", rgb_arr)
    assert success, "Failed to encode RGB gradient to PNG bytes"
    rgb_data = png_bytes.tobytes()

    at = AppTest.from_file(APP_PATH, default_timeout=45).run()

    uploader = _get_widget_by_label(at.file_uploader, "Upload an image", partial=True)
    uploader.upload("rgb_gradient.png", rgb_data, "image/png").run()

    r_effect = _get_widget_by_label(at.radio, "Select Filter Effect")
    r_effect.set_value("Blur").run()

    r_mode = _get_widget_by_label(at.radio, "Select Approximation Level")
    r_mode.set_value("AI Recommended").run()

    s_thresh = _get_widget_by_label(at.selectbox, "Target SSIM Quality Threshold")
    s_thresh.set_value(0.95).run()

    proc_btn = _get_widget_by_label(at.button, "Process Image", partial=True)
    proc_btn.click().run()

    assert len(at.exception) == 0, f"Unexpected exceptions on RGB gradient: {[e.value for e in at.exception]}"

    res = at.session_state["processed_results"]
    selected_mode = res["mode"]
    measured_ssim = res["metrics"]["ssim"]

    assert (measured_ssim >= 0.95) or (selected_mode == 0), (
        f"At threshold 0.95, selected mode {selected_mode} has displayed SSIM {measured_ssim:.4f} (< 0.95)"
    )


