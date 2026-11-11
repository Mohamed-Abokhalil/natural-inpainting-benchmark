"""Select failure-case candidates for Fig. 7 from per-image metrics (final set), then render a contact sheet
for visual confirmation. Criteria (all on coverage >= 20% unless stated):
 (a) G3 edge-propagation: largest deficit of ZITS hole-PSNR w.r.t. the median hole-PSNR of all 24 methods.
 (b) attention texture repetition: largest deficit of PEN-Net hole-LPIPS (bbox) w.r.t. the median.
 (c) over-smoothing (40-50% bin): lowest ratio of hole gradient energy (output / GT) among G1/G4 methods.
 (d) diffusion colour shift / insertion: largest mean RGB shift inside the hole for RePaint or StrDiff.
"""
import os, json
import numpy as np, pandas as pd
from PIL import Image
R = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
DATA = os.path.join(R, "data", "places2"); RES = os.path.join(R, "revision", "results")
pi = {f[:-4]: pd.read_csv(os.path.join(RES, "per_image", "final", f)) for f in os.listdir(os.path.join(RES, "per_image", "final"))}
src = lambda m: os.path.join(RES, "rerun", m) if m in ("VHII", "spa-former", "Pluralistic") else os.path.join(DATA, "SOTA", m)
files = pi["SCAT"].file.tolist(); bins = pi["SCAT"].bin.tolist()
P = pd.DataFrame({m: d.psnr_hole for m, d in pi.items()}); L = pd.DataFrame({m: d.lpips_hole_bbox for m, d in pi.items()})
big = np.array([b in ("20_30", "30_40", "40_50") for b in bins])
cand = {}
d = (P.median(1) - P["ZITS"]).where(big); cand["a_ZITS"] = [files[i] for i in d.sort_values(ascending=False).index[:6]]
d = (L["PEN-Net"] - L.median(1)).where(big); cand["b_PEN-Net"] = [files[i] for i in d.sort_values(ascending=False).index[:6]]


def load(p): return np.array(Image.open(p).convert("RGB")).astype(np.float32)


def gradE(x, m):
    g = np.abs(np.diff(x.mean(2), axis=0))[:, :-1] + np.abs(np.diff(x.mean(2), axis=1))[:-1, :]
    return g[m[:-1, :-1]].mean()


rows = []
for i, f in enumerate(files):
    if bins[i] != "40_50": continue
    m = np.array(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127; g = load(os.path.join(DATA, "gt", f))
    for meth in ("MADF", "MISF", "MADIN", "LaMa", "CMT", "DeepFillv2"):
        rows.append((gradE(load(os.path.join(src(meth), f)), m) / max(gradE(g, m), 1e-6), f, meth))
rows.sort(); cand["c_smooth"] = [(f, meth, round(float(r), 3)) for r, f, meth in rows[:6]]
rows = []
for i, f in enumerate(files):
    if not big[i]: continue
    m = np.array(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127; g = load(os.path.join(DATA, "gt", f))
    for meth in ("RePaint", "StrDiff"):
        o = load(os.path.join(src(meth), f)); rows.append((float(np.abs(o[m].mean(0) - g[m].mean(0)).sum()), f, meth))
rows.sort(reverse=True); cand["d_diffusion"] = [(f, meth, round(r, 1)) for r, f, meth in rows[:6]]
json.dump(cand, open(os.path.join(RES, "figures", "fig_failures_candidates.json"), "w"), indent=1)
print(json.dumps(cand, indent=1))
# contact sheet: masked | GT | output for each candidate
sheet = []
for k, v in cand.items():
    for it in v[:4]:
        f, meth = (it, k.split("_", 1)[1]) if isinstance(it, str) else (it[0], it[1])
        row = [load(os.path.join(DATA, "masked", f)), load(os.path.join(DATA, "gt", f)), load(os.path.join(src(meth), f))]
        sheet.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(sheet, 0).astype(np.uint8)).resize((768 // 2, 256 * len(sheet) // 2)).save(os.path.join(R, "revision", "work", "fail_sheet.png"))
