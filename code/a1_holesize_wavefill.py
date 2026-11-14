"""A1 hole-size test for WaveFill with its benchmark inference script (infer.py): same 40 images and the same extra 64x64
hole as a1_holesize_test.py; PSNR only on the original stroke pixels. Inputs are written to a scratch folder; outputs too.
Usage: python a1_holesize_wavefill.py prepare <scratch> | score <scratch>"""
import os, sys
import numpy as np, pandas as pd
from PIL import Image
W = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(W, ".."); R = os.path.join(V, "..")
DATA = os.path.join(R, "data", "places2")
PI = pd.read_csv(os.path.join(V, "results", "per_image", "final", "SCAT.csv"), index_col="file")
files = PI[PI.bin == "0_10"].sort_values("hole_ratio").index[:40]
mode, S = sys.argv[1], sys.argv[2]


def masks(f):
    m = np.asarray(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127
    ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
    y0, x0 = max([(0, 0), (0, 192), (192, 0), (192, 192)], key=lambda c: (c[0] + 32 - cy) ** 2 + (c[1] + 32 - cx) ** 2)
    m2 = m.copy(); m2[y0:y0 + 64, x0:x0 + 64] = True
    return m, m2


if mode == "prepare":
    for v in ("orig", "extra"):
        for sub in ("img", "mask", "out"): os.makedirs(os.path.join(S, v, sub), exist_ok=True)
    for f in files:
        g = np.asarray(Image.open(os.path.join(DATA, "gt", f)).convert("RGB"))
        for v, mm in zip(("orig", "extra"), masks(f)):
            Image.fromarray(g).save(os.path.join(S, v, "img", f))          # infer.py re-applies the mask itself
            Image.fromarray((mm * 255).astype(np.uint8)).save(os.path.join(S, v, "mask", f))
    print("prepared", len(files))
else:
    rows = []
    for f in files:
        m, _ = masks(f); g = np.asarray(Image.open(os.path.join(DATA, "gt", f)).convert("RGB"), np.float64)
        r = {"file": f}
        for v in ("orig", "extra"):
            o = np.asarray(Image.open(os.path.join(S, v, "out", "WaveFill_NII", f)).convert("RGB"), np.float64)   # composited output
            r[f"psnr_strokes_{v}"] = 10 * np.log10(255 ** 2 / ((o[m] - g[m]) ** 2).mean())
        rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(V, "results", "a1_holesize_wavefill.csv"), index=False)
    d = D.psnr_strokes_extra - D.psnr_strokes_orig
    print(f"WaveFill stroke-pixel PSNR: original {D.psnr_strokes_orig.mean():.2f} dB, with extra hole {D.psnr_strokes_extra.mean():.2f} dB; "
          f"improved on {(d > 0).sum()} of {len(D)}, median {d.median():+.2f} dB")
