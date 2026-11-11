"""Diversity subset for Table 10: 100 images stratified by coverage bin, seed 2026.
Bins 30-40% (14 images) and 40-50% (5 images) are taken in full; the remaining 81 are split
equally (27 each) among the three larger bins, because 20 per bin is impossible."""
import os, json, numpy as np
D = os.path.join(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"), r"data\places2\gt_binned")
OUT = os.path.join(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"), r"revision\results\subsets")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)
quota = {"0_10": 27, "10_20": 27, "20_30": 27, "30_40": 14, "40_50": 5}
sel = {}
for b, q in quota.items():
    fs = sorted(os.listdir(os.path.join(D, b)))
    sel[b] = sorted(rng.choice(fs, size=q, replace=False).tolist()) if q < len(fs) else fs
json.dump({"seed": 2026, "quota": quota, "files": sel}, open(os.path.join(OUT, "diversity100.json"), "w"), indent=1)
with open(os.path.join(OUT, "diversity100.txt"), "w") as f:
    for b in quota: f.writelines(x + "\n" for x in sel[b])
print({b: len(v) for b, v in sel.items()})
