import sys, os, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiplier_model as mm
from image_pipeline import KERNELS, process_image, load_image, calculate_metrics

md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]
exact = np.arange(256, dtype=np.int64)[:, None] * np.arange(256, dtype=np.int64)[None, :]

print("== Scrambled-order load_lut check ==")
disk = [np.load(os.path.join(mm.LUT_DIR, f"lut_mode{m}.npy")) for m in range(4)]
for m in [3, 0, 2, 1, 3, 0]:
    print(f"load_lut({m}) matches disk: {np.array_equal(mm.load_lut(m), disk[m])}")

print("\n== Per-coefficient LUT error (all 256 pixel values) ==")
for key, k in KERNELS.items():
    for mag in sorted({int(v) for v in k['magnitudes'].ravel()}):
        parts = []
        for m in (1, 2, 3):
            e = np.abs(disk[m][:, mag].astype(np.int64) - exact[:, mag])
            parts.append(f"m{m} max={e.max():4d} rate={np.mean(e>0)*100:5.1f}%")
        print(f"{key:8s} coef {mag:3d} | " + " | ".join(parts))

def oracle(img, key):  # test-only independent check using plain multiplication
    k = KERNELS[key]; mg = k['magnitudes'].astype(np.int64); sg = k['signs']
    Kh, Kw = mg.shape; H, W = img.shape
    p = np.pad(img, ((Kh//2,)*2, (Kw//2,)*2), mode='reflect').astype(np.int64)
    acc = np.zeros((H, W), np.int64)
    for r in range(Kh):
        for c in range(Kw):
            acc += sg[r, c] * mg[r, c] * p[r:r+H, c:c+W]
    res = np.abs(np.round(acc / k['scale'])) if k['is_edge'] else np.round(acc / k['scale']) + k['bias']
    return np.clip(res, 0, 255).astype(np.uint8)

rng = np.random.default_rng(0)
y, x = np.mgrid[0:256, 0:256]
images = {"sample_photo": load_image("images/sample_photo.png")[0]}
images["smooth_gradient"] = np.clip(0.5*x + 0.3*y, 0, 255).astype(np.uint8)
images["noisy_texture"] = np.clip(128 + rng.normal(0, 40, (256, 256)), 0, 255).astype(np.uint8)
hc = np.zeros((256, 256), np.uint8); hc[64:192, 64:192] = 255
images["high_contrast"] = hc

for iname, img in images.items():
    print(f"\n===== {iname} {img.shape} =====")
    print("effect  | mode | lut md5  | out md5  | diff px | max | mean   | MSE      | PSNR    | SSIM    | >=0.99")
    for key in KERNELS:
        ref = process_image(img, key, 0)
        ok = np.array_equal(ref, oracle(img, key))
        for m in range(4):
            out = process_image(img, key, m)
            d = np.abs(out.astype(int) - ref.astype(int))
            mt = calculate_metrics(ref, out)
            tag = f"{key}{' (m0==oracle:'+str(ok)+')' if m == 0 else ''}"
            print(f"{tag:8s}| {m}    | {md5(mm.load_lut(m))} | {md5(out)} | {int((d>0).sum()):7d} | "
                  f"{d.max():3d} | {d.mean():6.3f} | {mt['mse']:8.3f} | {mt['psnr']:7.2f} | "
                  f"{mt['ssim']:.5f} | {mt['ssim'] >= 0.99}")
