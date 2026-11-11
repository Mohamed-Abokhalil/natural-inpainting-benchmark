"""Generate supplementary.tex (Tables S1-S3) from results/stats/final/per_bin.csv and the composited
sensitivity results. Standalone article class; compile with pdflatex."""
import os
import pandas as pd, numpy as np
W = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(W, "..", "results")
ORDER = ["DeepFillv2", "MADF", "Pluralistic", "PEN-Net", "CoModGAN", "SCAT", "latent-code", "EdgeConnect",
         "StructureFlow", "E2I", "MEDFE", "CTSDG", "DGTS", "ZITS", "BAT-Fill", "WaveFill", "LaMa", "MISF", "CMT", "VHII",
         "spa-former", "MADIN", "RePaint", "StrDiff"]
NAME = {"latent-code": "Latent-code", "spa-former": "Spa-former"}
FAM = dict(zip(ORDER, ["G1"] * 2 + ["G2"] * 5 + ["G3"] * 7 + ["G4"] * 8 + ["G5"] * 2))
BINS = [("0_10", "0--10\\%"), ("10_20", "10--20\\%"), ("20_30", "20--30\\%"), ("30_40", "30--40\\%"), ("40_50", "40--50\\%"), ("all", "all")]
pb = pd.read_csv(os.path.join(RES, "stats", "final", "per_bin.csv"))
nb = {b: int(pb[(pb.bin == b)].n.iloc[0]) for b, _ in BINS}
star = {"Pluralistic", "VHII", "spa-former"}


def nm(m): return NAME.get(m, m) + ("$^{*}$" if m in star else "")


def table(caption, label, cols, fmt, binkey):
    out = [r"\begin{table}[p]\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\caption{" + caption + "}", r"\label{" + label + "}",
           r"\begin{tabular}{@{}ll" + "c" * len(cols) + "@{}}", r"\toprule",
           "Fam. & Method & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    sub = pb[pb.bin == binkey].set_index("method")
    for m in ORDER:
        r = sub.loc[m]
        cells = []
        for (k, _, f) in cols:
            v = r[k]
            cells.append("--" if (isinstance(v, float) and np.isnan(v)) else f.format(v))
        out.append(f"{FAM[m]} & {nm(m)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


doc = [r"\documentclass[a4paper,10pt]{article}", r"\usepackage[margin=2cm]{geometry}", r"\usepackage{booktabs}",
       r"\title{Supplementary Material\\[4pt]\large From Convolutions to Diffusion: A Decade of Deep Learning for Natural Image Inpainting}",
       r"\date{}", r"\begin{document}", r"\maketitle",
       r"\noindent All values follow the protocol of Section~12.5 of the main paper (1{,}000 Places2 test images, QD-IMD masks, $256\times256$). "
       r"Pairs per bin: " + ", ".join(f"{lab} {nb[b]}" for b, lab in BINS[:5]) + r". "
       r"$^{*}$Re-run in the revision with the benchmark masks (main paper, Section~12.5). "
       r"FID per bin is given only for bins with at least 100 images; KID (unbiased, $\times10^3$) is given for every bin but is unreliable for the two smallest bins. "
       r"Masked-region metrics: PSNR, SSIM and $\ell_1$ over hole pixels; LPIPS over the hole bounding box (at least $32\times32$). "
       r"Per-image values for every metric are released with the benchmark.", ""]
pb["kid1e3"] = pb["kid"] * 1e3
S1, S2 = [], []
for b, lab in BINS:
    S1.append(table(f"Supplementary Table S1 (bin {lab}, {nb[b]} pairs): full-image metrics.", f"tab:S1-{b}",
                    [("psnr", "PSNR$\\uparrow$", "{:.2f}"), ("ssim", "SSIM$\\uparrow$", "{:.4f}"), ("lpips", "LPIPS$\\downarrow$", "{:.4f}"),
                     ("l1", "$\\ell_1\\downarrow$", "{:.5f}"), ("fid", "FID$\\downarrow$", "{:.2f}"), ("kid1e3", "KID$\\times10^3\\downarrow$", "{:.2f}")], None, b))
    S2.append(table(f"Supplementary Table S2 (bin {lab}, {nb[b]} pairs): masked-region metrics.", f"tab:S2-{b}",
                    [("psnr_hole", "PSNR$_{\\rm hole}\\uparrow$", "{:.2f}"), ("ssim_hole", "SSIM$_{\\rm hole}\\uparrow$", "{:.4f}"),
                     ("l1_hole", "$\\ell_{1,\\rm hole}\\downarrow$", "{:.4f}"), ("lpips_hole_bbox", "LPIPS$_{\\rm bbox}\\downarrow$", "{:.4f}")], None, b))
doc = doc[:7] + [doc[7]] + S1 + S2
s3 = os.path.join(RES, "stats", "composited_final", "s3_rows.tex")
if os.path.exists(s3): doc.append(open(s3, encoding="utf-8").read())
doc.append(r"\end{document}")
open(os.path.join(W, "tex", "supplementary.tex"), "w", encoding="utf-8").write("\n".join(d for d in doc if d is not None))
print("written")
