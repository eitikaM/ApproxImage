import os
import re
import sys
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

    # Assert results page rendered Difference Map
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Amplified Difference Map" in all_markdown, f"'Amplified Difference Map' not found in page markdown for {effect}"

    # Verify Max Pixel Delta values
    displayed_delta = _get_displayed_max_pixel_delta(at)
    res = at.session_state["processed_results"]
    _, _, stats = compute_pixel_difference_map(res["exact_ref"], res["approx_result"])
    expected_delta = stats["max_delta"]

    # Assert displayed equals independent calculation on app's own arrays
    assert displayed_delta == expected_delta, (
        f"Displayed Max Pixel Delta ({displayed_delta}) != computed ({expected_delta}) for {effect}"
    )

    if effect in ("Blur", "Sharpen"):
        assert displayed_delta > 0, f"Expected {effect} Mode 3 to have Max Pixel Delta > 0, got {displayed_delta}"
    elif effect == "Edge Detection":
        assert displayed_delta == 0, f"Expected Edge Detection Mode 3 to have Max Pixel Delta == 0, got {displayed_delta}"


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
