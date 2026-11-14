"""Fig. 7: failure cases selected by fig_failures_select.py (criteria in its docstring) and confirmed visually."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_crops import DATA, OUT, CROP, img, mask, best_window, outline, crop3, masked_input
RES = os.path.join(os.path.dirname(OUT), "rerun")
src = lambda m: os.path.join(RES, m) if m in ("VHII", "spa-former", "Pluralistic", "ZITS", "CTSDG") else os.path.join(DATA, "SOTA", m)
cases = [("(a) G3 DGTS:\nspurious edges", "269126.png", "DGTS", "hole"),
         ("(b) G2 PEN-Net:\npatch-borrowing artefacts", "206874.png", "PEN-Net", "hole"),
         ("(c) G1 MADF, 40-50%:\nover-smoothing", "18142.png", "MADF", "hole"),
         ("(d) G5 StrDiff:\ninserted texture", "18142.png", "StrDiff", "err")]
cols = ["Masked input", "Ground truth", "Output", "GT crop (3×)", "Output crop (3×)"]
fig, ax = plt.subplots(len(cases), 5, figsize=(7.09, 7.09 / 5 * len(cases) + 0.3))
for r, (lab, f, m, crit) in enumerate(cases):
    mi, mk = masked_input(f)
    g = img(os.path.join(DATA, "gt"), f); o = img(src(m), f)
    if crit == "hole": pos = best_window(mk)
    else:  # window with the largest output-vs-GT error inside the hole
        e = (np.abs(o.astype(float) - g.astype(float)).mean(2) * mk)
        import cv2; ii = cv2.integral(e); best = (-1, (0, 0))
        for y in range(0, 256 - CROP + 1, 2):
            for x in range(0, 256 - CROP + 1, 2):
                v = ii[y + CROP, x + CROP] - ii[y, x + CROP] - ii[y + CROP, x] + ii[y, x]
                if v > best[0]: best = (v, (y, x))
        pos = best[1]
    tiles = [mi, outline(g, mk), outline(o, mk), crop3(g, mk, pos), crop3(o, mk, pos)]
    for c, t in enumerate(tiles):
        a = ax[r][c]; a.imshow(t, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if c in (0, 1, 2): a.add_patch(Rectangle((pos[1], pos[0]), CROP, CROP, fill=False, ec="yellow", lw=1.0))
        if c == 0: a.set_ylabel(lab + f"\n{f[:-4]}", fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
        if r == 0: a.set_title(cols[c], fontsize=8.5, fontweight="bold")
fig.subplots_adjust(left=0.2, right=0.999, top=0.95, bottom=0.005, wspace=0.03, hspace=0.04)
for e in ("pdf", "png"): fig.savefig(os.path.join(OUT, "fig_failures." + e), dpi=400, bbox_inches="tight", pad_inches=0.01)
