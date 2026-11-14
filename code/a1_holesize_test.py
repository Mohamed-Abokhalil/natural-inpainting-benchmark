"""A1 test of the hole-size hypothesis for Spa-former (benchmark pipeline = official pipeline, verified).
For the 40 images of the 0-10% bin with the smallest hole ratio, the original mask is completed (a) as is and (b) with an
additional 64x64 square hole placed in the image corner farthest from the original strokes (the extra hole is excluded from
scoring). PSNR is measured ONLY on the original stroke pixels. If quality there recovers when the total hole grows, the
failure depends on the total hole fraction (genuine behaviour of the checkpoint), not on the strokes themselves.
Seed fixed (Spa-former injects input noise). Writes results/a1_holesize_spaformer.csv."""
import os, sys, json
import numpy as np, pandas as pd, torch
from PIL import Image
W = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(W, ".."); R = os.path.join(V, "..")
DATA = os.path.join(R, "data", "places2")
sys.path.insert(0, os.path.join(W, "runners"))
_src = open(os.path.join(W, "runners", "gan_run.py"), encoding="utf-8").read()
_ns = {"__name__": "gan_run_defs", "__file__": os.path.join(W, "runners", "gan_run.py")}
exec(compile(_src[:_src.index("ap = argparse.ArgumentParser()")], "gan_run.py", "exec"), _ns)   # definitions only
torch.manual_seed(0)
run = _ns["build"]("spa-former")
PI = pd.read_csv(os.path.join(V, "results", "per_image", "final", "SCAT.csv"), index_col="file")
files = PI[PI.bin == "0_10"].sort_values("hole_ratio").index[:40]
dev = "cuda"
rows = []
for f in files:
    img = torch.from_numpy(np.asarray(Image.open(os.path.join(DATA, "gt", f)).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(dev)
    m = (np.asarray(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127)
    ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
    corners = [(0, 0), (0, 192), (192, 0), (192, 192)]
    y0, x0 = max(corners, key=lambda c: (c[0] + 32 - cy) ** 2 + (c[1] + 32 - cx) ** 2)
    m2 = m.copy(); m2[y0:y0 + 64, x0:x0 + 64] = True
    g = img[0].permute(1, 2, 0).cpu().numpy() * 255
    res = dict(file=f, hole_ratio=float(m.mean()), hole_ratio_with_extra=float(m2.mean()))
    for name, mm in (("orig", m), ("extra", m2)):
        hole = torch.from_numpy(mm.astype(np.float32))[None, None].to(dev)
        torch.manual_seed(0)
        with torch.no_grad():
            o = run(img, hole)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        o = np.round(o)
        res[f"psnr_strokes_{name}"] = float(10 * np.log10(255 ** 2 / ((o[m] - g[m]) ** 2).mean()))
    rows.append(res); print(f, {k: round(v, 3) for k, v in res.items() if k != "file"}, flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(V, "results", "a1_holesize_spaformer.csv"), index=False)
d = D.psnr_strokes_extra - D.psnr_strokes_orig
print(f"\nstroke-pixel PSNR: original mask {D.psnr_strokes_orig.mean():.2f} dB, with extra 64x64 hole {D.psnr_strokes_extra.mean():.2f} dB; "
      f"improved on {(d > 0).sum()} of {len(D)} images, median change {d.median():+.2f} dB")
