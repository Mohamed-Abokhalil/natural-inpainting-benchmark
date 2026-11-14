"""Rebuild the bodies of Tables 8, 9, 11 and 12 from the saved statistics (results/stats/final, results/efficiency):
family grouping with MISF in G1, rows by year within a family, best/second-best marks recomputed, FID printed to three
decimals everywhere (Table 8 prints 2 decimals for PSNR, 4 for SSIM/LPIPS, 5 for l1, as before). Cells that change in this
pass (the re-run ZITS and CTSDG rows, all FID cells of Table 8) are wrapped in \\rev. Values are formatted, never edited."""
import os, re, csv
import pandas as pd
W = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(W, "..", "results")
TEX = os.path.join(W, "tex", "revised_manuscript.tex")
FAMILY = {"G1": ["DeepFillv2", "MADF", "MISF"], "G2": ["Pluralistic", "PEN-Net", "CoModGAN", "SCAT", "latent-code"],
          "G3": ["EdgeConnect", "StructureFlow", "E2I", "MEDFE", "CTSDG", "DGTS", "ZITS"],
          "G4": ["BAT-Fill", "WaveFill", "LaMa", "CMT", "VHII", "spa-former", "MADIN"], "G5": ["RePaint", "StrDiff"]}
NAME = {"latent-code": "Latent-code", "spa-former": "Spa-former"}
CHANGED = {"ZITS", "CTSDG", "DGTS"}
PREV8 = {"Pluralistic", "VHII", "spa-former", "MISF"}  # Table 8 rows marked in earlier rounds, plus the moved MISF row
BINS = ["0_10", "10_20", "20_30", "30_40", "40_50"]
s = open(TEX, encoding="utf-8").read()
pe = pd.read_csv(os.path.join(RES, "stats", "final", "point_estimates.csv"), index_col=0)
pb = pd.read_csv(os.path.join(RES, "stats", "final", "per_bin.csv"))
rv = lambda t, on=True: ("\\rev{" + t + "}") if on else t


def body(label):
    a = s.index("\\label{" + label + "}"); t0 = s.index("\\toprule", a); h = s.index("\\midrule", t0); e = s.index("\\bottomrule", h)
    return a, h + len("\\midrule"), e


def heads(label):
    a, b, e = body(label); blk = s[b:e]
    cite = {}
    for l in blk.split("\n"):
        m = re.search(r"([A-Za-z0-9.\- ]+)~\\cite\{([^}]+)\}", l)
        if m:
            nm = m.group(1).strip()
            key = [k for k, v in {**{x: NAME.get(x, x) for f in FAMILY.values() for x in f}}.items() if v == nm]
            if key: cite[key[0]] = m.group(2)
    grp = {}
    for m in re.finditer(r"(\\multicolumn\{\d+\}\{(?:[^{}]|\{\})*\}\{(?:\\textit\{|\\textbf\{)(G\d)[^\n]*?\\\\)", blk):
        grp[m.group(2)] = m.group(1)
    return cite, grp


def rebuild(label, rowfn, year_order=False):
    global s
    cite, grp = heads(label)
    assert set(grp) == set(FAMILY), (label, grp.keys())
    out = []
    for i, (g, ms) in enumerate(FAMILY.items()):
        if i: out.append("    \\midrule")
        out += ["    " + grp[g], "    \\midrule"]
        order = sorted(ms, key=lambda m: (YEAR[m], ms.index(m))) if year_order else ms
        for m in order:
            out.append("    " + rowfn(m, cite[m]) + " \\\\")
    a, b, e = body(label)
    s = s[:b] + "\n" + "\n".join(out) + "\n    " + s[e:]


# years from Table 8
cite8, _ = heads("tab:benchmark")
YEAR = {}
a, b, e = body("tab:benchmark")
for l in s[b:e].split("\n"):
    m = re.match(r"\s*(?:\\rev\{)?(\d{4})\}?\s*&\s*([A-Za-z0-9.\- ]+)~\\cite", l)
    if m:
        for k, v in {x: NAME.get(x, x) for f in FAMILY.values() for x in f}.items():
            if v == m.group(2).strip(): YEAR[k] = int(m.group(1))
assert len(YEAR) == 24, sorted(set(x for f in FAMILY.values() for x in f) - set(YEAR))

# ---------------- Table 8
cols8 = [("psnr", True, "{:.2f}"), ("ssim", True, "{:.4f}"), ("fid", False, "{:.3f}"), ("lpips", False, "{:.4f}"), ("l1", False, "{:.5f}")]
rank8 = {c: list(pe[c].sort_values(ascending=not h).index[:2]) for c, h, _ in cols8}


def row8(m, key):
    cells = []
    for c, h, f in cols8:
        t = f.format(pe.loc[m, c])
        if m == rank8[c][0]: t = "\\textbf{" + t + "}"
        elif m == rank8[c][1]: t = "\\underline{" + t + "}"
        cells.append(rv(t, m in CHANGED or m in PREV8 or c == "fid"))
    yr = rv(str(YEAR[m]), m in ("VHII", "MISF"))  # VHII year changed earlier; MISF row moved to G1 in this round
    return f"{yr} &{NAME.get(m, m)}~\\cite{{{key}}} & " + " & ".join(cells)


rebuild("tab:benchmark", row8, year_order=True)
# ---------------- Table 9
cols9 = {"fid": False, "kid_sub_mean": False, "pids": True, "uids": True, "niqe": False}
bm9 = {c: list(pe[c].sort_values(ascending=not h).index[:2]) for c, h in cols9.items()}


def mk9(m, c, t):
    return "\\textbf{" + t + "}" if m == bm9[c][0] else ("\\underline{" + t + "}" if m == bm9[c][1] else t)


def row9(m, key):
    r = pe.loc[m]
    cells = [mk9(m, "fid", f"{r.fid:.3f}"), f"[{r.fid + r.fid_ci_lo - r.fid_boot_mean:.2f}, {r.fid + r.fid_ci_hi - r.fid_boot_mean:.2f}]",
             mk9(m, "kid_sub_mean", f"{r.kid_sub_mean*1e3:.2f} $\\pm$ {r.kid_sub_sd*1e3:.2f}".replace("-", "$-$")),
             mk9(m, "pids", f"{r.pids*100:.1f}"), mk9(m, "uids", f"{r.uids*100:.1f}"), mk9(m, "niqe", f"{r.niqe:.2f}")]
    return f"{NAME.get(m, m)}~\\cite{{{key}}} & " + " & ".join(rv(c, m in CHANGED) for c in cells)


rebuild("tab:extended", row9, year_order=True)
# ---------------- Table 11
best11 = {}
for b_ in BINS:
    sub = pb[pb.bin == b_].set_index("method"); best11[b_] = (sub.psnr_hole.idxmax(), sub.lpips_hole_bbox.idxmin())


def row11(m, key):
    cells = []
    for b_ in BINS:
        r = pb[(pb.method == m) & (pb.bin == b_)].iloc[0]
        p_, l_ = f"{r.psnr_hole:.2f}", f"{r.lpips_hole_bbox:.3f}"
        if best11[b_][0] == m: p_ = "\\textbf{" + p_ + "}"
        if best11[b_][1] == m: l_ = "\\textbf{" + l_ + "}"
        cells += [p_, l_]
    return f"{NAME.get(m, m)}~\\cite{{{key}}} & " + " & ".join(rv(c, m in CHANGED) for c in cells)


rebuild("tab:perbin", row11, year_order=True)
# ---------------- Table 12 (values unchanged; regrouped)


def row12(m, key):
    r = next(csv.DictReader(open(os.path.join(RES, "efficiency", f"{m}.csv"))))
    fmt = lambda v, nd: "--" if str(v).lower() in ("", "nan") else f"{float(v):.{nd}f}"
    lat = float(r["latency"]); lat_s = f"{lat:,.0f}" if lat >= 100 else f"{lat:.1f}"
    return f"{NAME.get(m, m)}~\\cite{{{key}}} & " + " & ".join([fmt(r["params"], 1), fmt(r["flops"], 1), fmt(r["memory"], 2), lat_s, str(int(float(r["steps"])))])


rebuild("tab:efficiency", row12, year_order=True)
open(TEX, "w", encoding="utf-8").write(s)
print("Table 8 marks:", rank8)
print(pe.loc[["SCAT", "MADIN", "ZITS", "CTSDG", "MISF"], ["psnr", "ssim", "fid", "lpips", "l1"]].round(4).to_string())
