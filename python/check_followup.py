import sys
import numpy as np
sys.path.insert(0, "python")
import multiplier_model as mm
from image_pipeline import process_image, load_image, calculate_metrics
from skimage.metrics import structural_similarity as ssim

print("== A. LUT error at pixel values 0 and 255 for sharpen coefficients ==")
for m in (1, 2, 3):
    lut = mm.load_lut(m).astype(np.int64)
    for c in (10, 19, 244):
        print(f"mode {m} coef {c}: err at A=0: {lut[0, c]}, err at A=255: {int(lut[255, c]) - 255 * c}")

print("\n== B. SSIM on 2D grayscale: pipeline vs channel_axis=-1 ==")
img = load_image("images/sample_photo.png")[0]
for eff in ("sharpen", "emboss"):
    ref = process_image(img, eff, 0)
    out = process_image(img, eff, 3)
    a = calculate_metrics(ref, out)["ssim"]
    try:
        b = ssim(ref, out, channel_axis=-1, data_range=255)
    except Exception as e:
        b = f"ERROR: {e}"
    print(f"{eff}: calculate_metrics SSIM={a:.6f} | channel_axis=-1 on 2D={b}")
