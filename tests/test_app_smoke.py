import os
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(PROJECT_ROOT, "python", "app.py")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "images", "sample_photo.png")

EFFECTS = ["Blur", "Edge Detection", "Sharpen", "Emboss"]


@pytest.mark.parametrize("effect", EFFECTS)
def test_app_mode3_smoke(effect):
    with open(SAMPLE_PATH, "rb") as f:
        sample_bytes = f.read()

    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()

    # Upload sample image
    at.file_uploader[0].upload("sample_photo.png", sample_bytes, "image/png").run()

    # Set effect
    at.radio[0].set_value(effect).run()

    # Set Mode 3
    mode3_opt = [opt for opt in at.radio[1].options if "Mode 3" in opt][0]
    at.radio[1].set_value(mode3_opt).run()

    # Click Process Image button
    proc_btn = at.button[0]
    proc_btn.click().run()

    # Assert at.exception is empty (no traceback)
    assert len(at.exception) == 0, f"Unexpected exceptions for {effect}: {[e.value for e in at.exception]}"

    # Assert the results page shows the text 'Amplified Difference Map'
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Amplified Difference Map" in all_markdown, f"'Amplified Difference Map' not found in page markdown for {effect}"
