"""Supplementary Figure S1 (A1): small-hole failure of Spa-former and WaveFill. Rows: three images of the 0-10% bin with
the smallest holes among those in the hole-size test. Columns: masked input, ground truth, SCAT (reference behaviour),
Spa-former, WaveFill (benchmark outputs), and Spa-former with an additional 64x64 hole (the extra hole is greyed out), which
shows that the same strokes are completed correctly once the total hole is larger. Hole PSNR printed under each output.
Writes final/figs/figS1_smallhole.pdf."""
import os, sys
import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
W = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(W, ".."); R = os.path.join(V, "..")
DATA = os.path.join(R, "data", "places2"); OUT = os.path.join(V, "final", "figs"); os.makedirs(OUT, exist_ok=True)
_src = open(os.path.join(W, "runners", "gan_run.py"), encoding="utf-8").read()
_ns = {"__name__": "gan_run_defs", "__file__": os.path.join(W, "runners", "gan_run.py")}
exec(compile(_src[:_src.index("ap = argparse.ArgumentParser()")], "gan_run.py", "exec"), _ns)
run = _ns["build"]("spa-former")
files = ["308470.png", "293715.png", "272087.png"]
ld = lambda p: np.asarray(Image.open(p).convert("RGB"))
psnr = lambda o, g, m: 10 * np.log10(255 ** 2 / ((o[m].astype(float) - g[m].astype(float)) ** 2).mean())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
cols = ["Masked input", "Ground truth", "SCAT", "Spa-former", "WaveFill", "Spa-former,\nextra 64×64 hole"]
fig, ax = plt.subplots(len(files), len(cols), figsize=(7.0, 4.6))
for i, f in enumerate(files):
    g = ld(os.path.join(DATA, "gt", f)); m = ld(os.path.join(DATA, "mask", f))[..., 0] > 127
    ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
    y0, x0 = max([(0, 0), (0, 192), (192, 0), (192, 192)], key=lambda c: (c[0] + 32 - cy) ** 2 + (c[1] + 32 - cx) ** 2)
    m2 = m.copy(); m2[y0:y0 + 64, x0:x0 + 64] = True
    img = torch.from_numpy(g.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
    torch.manual_seed(0)
    with torch.no_grad():
        o2 = np.round(run(img, torch.from_numpy(m2.astype(np.float32))[None, None].cuda())[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    o2 = o2.copy(); o2[y0:y0 + 64, x0:x0 + 64] = (o2[y0:y0 + 64, x0:x0 + 64] * 0.35 + 128 * 0.65).astype(np.uint8)
    outs = [ld(os.path.join(DATA, "masked", f)), g, ld(os.path.join(DATA, "SOTA", "SCAT", f)),
            ld(os.path.join(V, "results", "rerun", "spa-former", f)), ld(os.path.join(DATA, "SOTA", "WaveFill", f)), o2]
    for j, im in enumerate(outs):
        a = ax[i, j]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
        if i == 0: a.set_title(cols[j], fontsize=7)
        if j >= 2: a.set_xlabel(f"{psnr(im, g, m):.1f} dB (strokes)", fontsize=6.5)
        if j == 0: a.set_ylabel(f"{f[:-4]}, hole {m.mean()*100:.1f}%", fontsize=6.5)
plt.tight_layout(pad=0.3, h_pad=1.4)
fig.savefig(os.path.join(OUT, "figS1_smallhole.pdf")); fig.savefig(os.path.join(V, "results", "figS1_preview.png"), dpi=130)
print("written")
