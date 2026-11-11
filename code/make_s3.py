"""Supplementary Table S3: Table 8 metrics with every output re-composited onto the known pixels.
Methods whose outputs were not exactly composited (CoModGAN, DGTS, RePaint, and the six with edge
differences) use results/per_image/composited; all others are unchanged (already exactly composited)."""
import os, json
import numpy as np, pandas as pd
from pytorch_fid.fid_score import calculate_frechet_distance
W = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(W, "..", "results")
ORDER = ["DeepFillv2", "MADF", "Pluralistic", "PEN-Net", "CoModGAN", "SCAT", "latent-code", "EdgeConnect",
         "StructureFlow", "E2I", "MEDFE", "CTSDG", "DGTS", "ZITS", "BAT-Fill", "WaveFill", "LaMa", "MISF", "CMT", "VHII",
         "spa-former", "MADIN", "RePaint", "StrDiff"]
NAME = {"latent-code": "Latent-code", "spa-former": "Spa-former"}
COMP = ["CoModGAN", "DGTS", "RePaint", "BAT-Fill", "DeepFillv2", "MEDFE", "PEN-Net", "StructureFlow", "WaveFill"]
gt = np.load(os.path.join(RES, "features", "gt_inception2048.npy")).astype(np.float64)
fid = lambda a: calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False), gt.mean(0), np.cov(gt, rowvar=False))
rows = {}
for v in ("final", "comp"):
    for m in ORDER:
        src = "composited" if (v == "comp" and m in COMP) else "final"
        d = pd.read_csv(os.path.join(RES, "per_image", src, f"{m}.csv"))
        f = np.load(os.path.join(RES, "features", src, f"{m}_inception2048.npy")).astype(np.float64)
        rows[(v, m)] = dict(psnr=d.psnr.mean(), ssim=d.ssim.mean(), fid=fid(f), lpips=d.lpips.mean(), l1=d.l1.mean())
df = pd.DataFrame(rows).T
out = {}
hi = {"psnr": True, "ssim": True, "fid": False, "lpips": False, "l1": False}
for k, h in hi.items():
    r0 = df.loc["final"][k].rank(ascending=not h); r1 = df.loc["comp"][k].rank(ascending=not h)
    out[k] = {"max_rank_change": int((r0 - r1).abs().max()), "changed": [m for m in ORDER if r0[m] != r1[m]],
              "top2_final": list(df.loc["final"][k].sort_values(ascending=not h).index[:2]),
              "top2_comp": list(df.loc["comp"][k].sort_values(ascending=not h).index[:2])}
os.makedirs(os.path.join(RES, "stats", "composited_final"), exist_ok=True)
df.to_csv(os.path.join(RES, "stats", "composited_final", "s3.csv"))
json.dump(out, open(os.path.join(RES, "stats", "composited_final", "s3_ranks.json"), "w"), indent=1)
L = [r"\begin{table}[p]\centering\scriptsize\setlength{\tabcolsep}{3pt}",
     r"\caption{Supplementary Table S3: Table~8 metrics after re-compositing every output onto the known pixels "
     r"($\hat{\mathbf I}=\mathbf M\odot\mathbf O+(\mathbf 1-\mathbf M)\odot\mathbf I$). Only the nine methods marked $^{\circ}$ change; "
     r"all others already return exactly composited outputs. $^{*}$Re-run in the revision.}",
     r"\label{tab:S3}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
     r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & FID$\downarrow$ & LPIPS$\downarrow$ & $\ell_1\downarrow$ \\", r"\midrule"]
for m in ORDER:
    r = df.loc[("comp", m)]
    n = NAME.get(m, m) + ("$^{\circ}$" if m in COMP else "") + ("$^{*}$" if m in ("Pluralistic", "VHII", "spa-former") else "")
    L.append(f"{n} & {r.psnr:.2f} & {r.ssim:.4f} & {r.fid:.2f} & {r.lpips:.4f} & {r.l1:.5f} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(RES, "stats", "composited_final", "s3_rows.tex"), "w", encoding="utf-8").write("\n".join(L))
print(json.dumps(out, indent=1))
print(df.loc["comp"].loc[COMP].round(4))
