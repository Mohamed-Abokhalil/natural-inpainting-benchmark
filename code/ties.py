"""Paired-bootstrap tie tests for every Table 8 metric (+KID) from results/stats/final/bootstrap.npz.
Interval for the difference d = best - other: bias-corrected percentile, [d + q2.5 - mean, d + q97.5 - mean]
(for per-image metrics mean == d up to Monte-Carlo error). Candidates: second best and any method within 1% of best.
Also tests the claims of the submitted text. Output: results/stats/final/ties_final.csv"""
import os, numpy as np, pandas as pd
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "stats", "final")
bs = dict(np.load(os.path.join(RES, "bootstrap.npz"))); pe = pd.read_csv(os.path.join(RES, "point_estimates.csv"), index_col=0)
col = {"psnr": "psnr", "ssim": "ssim", "lpips": "lpips", "l1": "l1", "fid": "fid", "kid": "kid_full"}
hi = {"psnr": True, "ssim": True, "lpips": False, "l1": False, "fid": False, "kid": False}
rows = []
def test(k, a, b, why):
    d = pe.loc[a, col[k]] - pe.loc[b, col[k]]; db = bs[f"{a}|{k}"] - bs[f"{b}|{k}"]
    lo, up = d + np.percentile(db, 2.5) - db.mean(), d + np.percentile(db, 97.5) - db.mean()
    rows.append(dict(metric=k, a=a, b=b, a_val=pe.loc[a, col[k]], b_val=pe.loc[b, col[k]], diff=d, ci_lo=lo, ci_hi=up,
                     significant=not (lo <= 0 <= up), why=why))
for k, h in hi.items():
    s = pe[col[k]].sort_values(ascending=not h); best = s.index[0]
    test(k, best, s.index[1], "second best")
    for m in s.index[2:]:
        if abs(s[m] - s.iloc[0]) <= 0.01 * abs(s.iloc[0]): test(k, best, m, "within 1% of best")
for a, b, k in [("SCAT", "MADIN", "fid"), ("MADF", "VHII", "l1"), ("MADF", "MADIN", "l1"), ("VHII", "MADIN", "l1")]:
    test(k, a, b, "claim in submitted text")
for a, b, k in [("SCAT", "ZITS", "fid"), ("MADIN", "ZITS", "fid"), ("MADIN", "ZITS", "psnr"), ("SCAT", "ZITS", "lpips"), ("MADIN", "ZITS", "lpips")]:
    test(k, a, b, "re-run ZITS near the top (round 6)")
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "ties_final.csv"), index=False)
pd.set_option("display.width", 200); print(df.round(6).to_string())
