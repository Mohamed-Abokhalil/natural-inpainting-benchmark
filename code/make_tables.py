"""Write LaTeX rows for Tables 9 (extended), 11 (per-bin hole metrics), 12 (efficiency) and Supplementary
S1-S3 into the working manuscript / supplementary.tex, from results/stats/final, results/efficiency,
results/stats/composited_final. Values are formatted, never edited by hand."""
import os, re, csv, math
import pandas as pd, numpy as np
W = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(W, "..", "results")
TEX = os.path.join(W, "tex", "revised_manuscript.tex")
ORDER = ["DeepFillv2", "MADF", "Pluralistic", "PEN-Net", "VHII", "CoModGAN", "SCAT", "latent-code", "EdgeConnect",
         "StructureFlow", "E2I", "MEDFE", "CTSDG", "DGTS", "ZITS", "BAT-Fill", "WaveFill", "LaMa", "MISF", "CMT",
         "spa-former", "MADIN", "RePaint", "StrDiff"]
TEXNAME = {"latent-code": "Latent-code", "spa-former": "Spa-former"}
BINS = ["0_10", "10_20", "20_30", "30_40", "40_50"]


def row_key(m):  # the manuscript row starts with '    <Name>~\cite{'
    return "    " + TEXNAME.get(m, m) + "~\\cite{"


def fill_rows(s, label, cells_by_method):
    a = s.index("\\label{" + label + "}"); b = s.index("\\end{tabular", a)
    block = s[a:b]; lines = block.split("\n"); n = 0
    for i, l in enumerate(lines):
        for m, cells in cells_by_method.items():
            if l.startswith(row_key(m)) and "\\tbd" in l:
                head = l[: l.index("}", l.index("\\cite{")) + 1]
                if "\\rev{$^{*}$}" in l: head += "\\rev{$^{*}$}"
                lines[i] = head + " & " + " & ".join(cells) + " \\\\"; n += 1
    return s[:a] + "\n".join(lines) + s[b:], n


def best_marks(vals, higher):
    order = sorted(vals, key=lambda k: vals[k], reverse=higher)
    return order[0], order[1]


def main():
    s = open(TEX, encoding="utf-8").read()
    pe = pd.read_csv(os.path.join(RES, "stats", "final", "point_estimates.csv"), index_col=0)
    # ---------------- Table 9 ----------------
    cols = {"fid": False, "kid_sub_mean": False, "pids": True, "uids": True, "niqe": False}
    bm = {c: best_marks(pe[c].to_dict(), h) for c, h in cols.items()}
    def mk(m, c, txt):
        if m == bm[c][0]: return "\\textbf{" + txt + "}"
        if m == bm[c][1]: return "\\underline{" + txt + "}"
        return txt
    t9 = {}
    for m in ORDER:
        r = pe.loc[m]
        t9[m] = [mk(m, "fid", f"{r.fid:.2f}"), f"[{r.fid + r.fid_ci_lo - r.fid_boot_mean:.2f}, {r.fid + r.fid_ci_hi - r.fid_boot_mean:.2f}]",   # bias-corrected percentile interval
                 mk(m, "kid_sub_mean", f"{r.kid_sub_mean*1e3:.2f} $\\pm$ {r.kid_sub_sd*1e3:.2f}"),
                 mk(m, "pids", f"{r.pids*100:.1f}"), mk(m, "uids", f"{r.uids*100:.1f}"), mk(m, "niqe", f"{r.niqe:.2f}")]
    s, n9 = fill_rows(s, "tab:extended", t9)
    # ---------------- Table 11 ----------------
    pb = pd.read_csv(os.path.join(RES, "stats", "final", "per_bin.csv"))
    t11 = {}
    for m in ORDER:
        cells = []
        for b in BINS:
            r = pb[(pb.method == m) & (pb.bin == b)].iloc[0]
            cells += [f"{r.psnr_hole:.2f}", f"{r.lpips_hole_bbox:.3f}"]
        t11[m] = cells
    # bold best per column
    for j, b in enumerate(BINS):
        sub = pb[pb.bin == b].set_index("method")
        bp = sub.psnr_hole.idxmax(); bl = sub.lpips_hole_bbox.idxmin()
        t11[bp][2 * j] = "\\textbf{" + t11[bp][2 * j] + "}"; t11[bl][2 * j + 1] = "\\textbf{" + t11[bl][2 * j + 1] + "}"
    s, n11 = fill_rows(s, "tab:perbin", t11)
    # ---------------- Table 12 ----------------
    n12 = 0
    eff_dir = os.path.join(RES, "efficiency")
    if os.path.isdir(eff_dir):
        t12 = {}
        for m in ORDER:
            f = os.path.join(eff_dir, f"{m}.csv")
            if not os.path.exists(f): continue
            r = next(csv.DictReader(open(f)))
            fmt = lambda v, nd: "--" if v in ("", "nan") or (isinstance(v, str) and v.lower() == "nan") else f"{float(v):.{nd}f}"
            lat = float(r["latency"]); lat_s = f"{lat:,.0f}" if lat >= 100 else f"{lat:.1f}"
            t12[m] = [fmt(r["params"], 1), fmt(r["flops"], 1), fmt(r["memory"], 2), lat_s, str(int(float(r["steps"])))]
        a = s.index("\\label{tab:efficiency}"); b = s.index("\\end{tabular", a); block = s[a:b].split("\n")
        for i, l in enumerate(block):
            for m, cells in t12.items():
                if l.startswith(row_key(m)) and "\\tbd" in l:
                    head = l[: l.index("}", l.index("\\cite{")) + 1]
                    if "\\rev{$^{*}$}" in l: head += "\\rev{$^{*}$}"
                    block[i] = head + " & " + " & ".join(cells) + " \\\\"; n12 += 1
        s = s[:a] + "\n".join(block) + s[b:]
    open(TEX, "w", encoding="utf-8").write(s)
    print("Table 9 rows", n9, "Table 11 rows", n11, "Table 12 rows", n12)


if __name__ == "__main__":
    main()
