"""Round 6: re-select panel (a) of the failure figure after the ZITS re-run (the earlier ZITS case was caused by the
pipeline error). Criterion, as before but over all G3 (structure-texture) methods: largest deficit of a G3 method's
hole-PSNR w.r.t. the median hole-PSNR of all 24 methods, coverage >= 20%. Writes a contact sheet for visual confirmation."""
import os, json
import numpy as np, pandas as pd
from PIL import Image
R = r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"
DATA = os.path.join(R, "data", "places2"); RES = os.path.join(R, "revision", "results")
RERUN = ("VHII", "spa-former", "Pluralistic", "ZITS", "CTSDG")
src = lambda m: os.path.join(RES, "rerun", m) if m in RERUN else os.path.join(DATA, "SOTA", m)
pi = {f[:-4]: pd.read_csv(os.path.join(RES, "per_image", "final", f)) for f in os.listdir(os.path.join(RES, "per_image", "final"))}
files = pi["SCAT"].file.tolist(); bins = pi["SCAT"].bin.tolist()
P = pd.DataFrame({m: d.psnr_hole for m, d in pi.items()})
big = np.array([b in ("20_30", "30_40", "40_50") for b in bins])
G3 = ["EdgeConnect", "StructureFlow", "E2I", "MEDFE", "CTSDG", "DGTS", "ZITS"]
rows = []
for m in G3:
    d = (P.median(1) - P[m]).where(big)
    for i in d.sort_values(ascending=False).index[:3]:
        rows.append((float(d[i]), files[i], m, bins[i]))
rows.sort(reverse=True)
json.dump(rows, open(os.path.join(RES, "figures", "fig_failures_candidates_g3.json"), "w"), indent=1)
for r in rows: print(f"{r[0]:6.2f} dB  {r[1]:>12}  {r[2]:<14} {r[3]}")
ld = lambda p: np.array(Image.open(p).convert("RGB"))
sheet = [np.concatenate([ld(os.path.join(DATA, "masked", f)), ld(os.path.join(DATA, "gt", f)), ld(os.path.join(src(m), f))], 1) for _, f, m, _ in rows[:10]]
Image.fromarray(np.concatenate(sheet, 0)).save(os.path.join(R, "revision", "work", "fail_sheet_g3.png"))
