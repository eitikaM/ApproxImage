import os
import sys
import io
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = r"c:\Users\ASUS\OneDrive\Desktop\final_year\approximate_multipliers"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

# Create a test synthetic color image in memory
test_img = np.zeros((128, 128, 3), dtype=np.uint8)
test_img[:64, :, 0] = 200  # Blue
test_img[64:, :, 2] = 200  # Red
test_img[:, 64:, 1] = 150  # Green
_, encoded = cv2.imencode(".png", test_img)
test_bytes = encoded.tobytes()

print("--- 1. TEST FRESH LAUNCH ---")
at = AppTest.from_file(os.path.join(PROJECT_ROOT, "python", "app.py"), default_timeout=30)
at.run()
assert at.session_state["page"] == "configure"
assert at.session_state["image_data"] is None
print("Fresh launch: PASS")

print("--- 2. TEST UPLOAD AND PROCESS (BLUR + AI RECOMMENDED) ---")
# Upload image into session state
bgr = cv2.imdecode(np.frombuffer(test_bytes, np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
at.session_state["image_data"] = (rgb, gray, True)
at.session_state["source_name"] = "synthetic_color.png"
at.session_state["effect_label"] = "Blur"
at.session_state["effect_key"] = "blur"
at.session_state["mode_str"] = "AI Recommended"
at.run()

# Find and click Process Image button
# Buttons on configure page: process_btn is index 0
at.button[0].click()
at.run()

assert at.session_state["page"] == "results", f"Expected results page, got {at.session_state['page']}"
res = at.session_state["processed_results"]
assert res is not None
print(f"Processed Blur with AI Recommended: Mode {res['mode']}")
print(f"Metrics: SSIM={res['metrics']['ssim']:.4f}, MSE={res['metrics']['mse']:.4f}, PSNR={res['metrics']['psnr']:.2f}")
assert res['metrics']['ssim'] >= 0.99, "SSIM must be >= 0.99"
print("Blur + AI: PASS")

print("--- 3. TEST RESULTS PAGE COMPONENTS ---")
# Check download buttons exist
download_buttons = at.download_button
print(f"Found {len(download_buttons)} download buttons")
assert len(download_buttons) >= 1, "Expected at least 1 download button"
for db in download_buttons:
    print(f"  Download Button: '{db.label}'")
assert any("Approximate" in db.label for db in download_buttons), "Download Approximate Image button missing"
print("Download buttons: PASS")

print("--- 4. TEST BACK TO CONFIGURATION ---")
# Click Back to Configuration (first button on results screen)
at.button[0].click()
at.run()
assert at.session_state["page"] == "configure"
print("Back to configuration: PASS")

print("--- 5. TEST SHARPEN + AI RECOMMENDED ---")
at.session_state["effect_label"] = "Sharpen"
at.session_state["effect_key"] = "sharpen"
at.session_state["mode_str"] = "AI Recommended"
at.run()
at.button[0].click()
at.run()
res_sh = at.session_state["processed_results"]
print(f"Processed Sharpen with AI Recommended: Mode {res_sh['mode']}")
print(f"Metrics: SSIM={res_sh['metrics']['ssim']:.4f}, MSE={res_sh['metrics']['mse']:.4f}, PSNR={res_sh['metrics']['psnr']:.2f}")
assert res_sh['metrics']['ssim'] >= 0.99, "SSIM must be >= 0.99"
assert res_sh['effect_key'] == "sharpen", "Stale effect key!"
print("Sharpen + AI: PASS")

print("--- 6. TEST EDGE DETECTION + AI RECOMMENDED ---")
# Back to config
at.button[0].click()
at.run()
at.session_state["effect_label"] = "Edge Detection"
at.session_state["effect_key"] = "edge"
at.session_state["mode_str"] = "AI Recommended"
at.run()
at.button[0].click()
at.run()
res_edge = at.session_state["processed_results"]
print(f"Processed Edge Detection with AI Recommended: Mode {res_edge['mode']}")
print(f"Metrics: SSIM={res_edge['metrics']['ssim']:.4f}, MSE={res_edge['metrics']['mse']:.4f}, PSNR={res_edge['metrics']['psnr']:.2f}")
assert res_edge['metrics']['ssim'] >= 0.99, "SSIM must be >= 0.99"
assert res_edge['effect_key'] == "edge", "Stale effect key!"
print("Edge Detection + AI: PASS")

print("--- 7. TEST EMBOSS + AI RECOMMENDED ---")
# Back to config
at.button[0].click()
at.run()
at.session_state["effect_label"] = "Emboss"
at.session_state["effect_key"] = "emboss"
at.session_state["mode_str"] = "AI Recommended"
at.run()
at.button[0].click()
at.run()
res_emb = at.session_state["processed_results"]
print("--- 8. TEST BLUR ACROSS ALL MODES (0, 1, 2, 3) ---")
for mode_name, expected_mode in [
    ("Exact", 0),
    ("Mode 1 — Low Approximation", 1),
    ("Mode 2 — Medium Approximation", 2),
    ("Mode 3 — High Approximation", 3)
]:
    # Back to config
    at.button[0].click()
    at.run()
    at.session_state["effect_label"] = "Blur"
    at.session_state["effect_key"] = "blur"
    at.session_state["mode_str"] = mode_name
    at.run()
    at.button[0].click()
    at.run()
    res_mode = at.session_state["processed_results"]
    assert res_mode['mode'] == expected_mode, f"Expected mode {expected_mode}, got {res_mode['mode']}"
    assert res_mode['effect_key'] == "blur", "Stale effect key!"
    print(f"  Blur {mode_name}: Mode {res_mode['mode']}, SSIM={res_mode['metrics']['ssim']:.4f}, MSE={res_mode['metrics']['mse']:.4f}")
    if expected_mode == 0:
        assert res_mode['metrics']['ssim'] == 1.0
        assert res_mode['metrics']['mse'] == 0.0

print("\nALL VERIFICATION TESTS (PART 11) PASSED CLEANLY!")
