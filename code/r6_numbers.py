"""Round 6: every number quoted in Sections 12.6-12.9 recomputed from the saved statistics after the ZITS/CTSDG re-runs.
Prints old (results/stats/final_pre_r6) and new values side by side; writes results/r6_text_numbers.json."""
import os, json, csv
import numpy as np, pandas as pd
W = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(W, "..", "results")
B = ["0_10", "10_20", "20_30", "30_40", "40_50"]
out = {}
for tag, d in [("old", "final_pre_r6"), ("new", "final")]:
    pe = pd.read_csv(os.path.join(RES, "stats", d, "point_estimates.csv"), index_col=0)
    pb = pd.read_csv(os.path.join(RES, "stats", d, "per_bin.csv"))
    P = pb.pivot(index="method", columns="bin", values="psnr_hole")[B]; L = pb.pivot(index="method", columns="bin", values="lpips_hole_bbox")[B]
    rp = P.rank(ascending=False).astype(int); rl = L.rank().astype(int)
    sp = lambda X, b: X["0_10"].corr(X[b], "spearman")
    r = {}
    r["kid_fid_rho"] = pe.kid_full.corr(pe.fid, "spearman")
    r["kid_top4"] = list(pe.kid_full.sort_values().index[:4]); r["kid_top4_spread_e3"] = float(np.ptp(pe.kid_full.sort_values()[:4]) * 1e3)
    r["kid_sd_e3_top4"] = [round(float(pe.loc[m, "kid_sub_sd"]) * 1e3, 3) for m in r["kid_top4"]]
    r["repaint_fid_rank"] = int(pe.fid.rank().loc["RePaint"])
    r["pids_rho"] = pe.pids.corr(pe.fid, "spearman"); r["uids_rho"] = pe.uids.corr(pe.fid, "spearman")
    r["pids_top2"] = list(pe.pids.sort_values(ascending=False).index[:2]); r["uids_top2"] = list(pe.uids.sort_values(ascending=False).index[:2])
    r["pids_max_pct"] = float(pe.pids.max() * 100); r["uids_max_pct"] = float(pe.uids.max() * 100)
    r["niqe_rho"] = pe.niqe.corr(pe.fid, "spearman"); r["niqe_top2"] = list(pe.niqe.sort_values().index[:2])
    r["niqe_top2_fidrank"] = [int(pe.fid.rank().loc[m]) for m in r["niqe_top2"]]
    r["sp_psnr"] = (sp(P, "10_20"), sp(P, "40_50")); r["sp_lpips"] = (sp(L, "10_20"), sp(L, "40_50"))
    r["top5_every_bin"] = [m for m in P.index if (rp.loc[m] <= 5).all()]
    r["madin_lpips_ranks"] = list(map(int, rl.loc["MADIN"]))
    for m in ["spa-former", "WaveFill", "RePaint", "StrDiff", "ZITS", "CTSDG"]:
        r[f"rank_psnr_{m}"] = list(map(int, rp.loc[m])); r[f"rank_lpips_{m}"] = list(map(int, rl.loc[m]))
    r["bin0_span"] = (float(P["0_10"].min()), float(P["0_10"].max()), P["0_10"].idxmin(), P["0_10"].idxmax())
    r["best_hole_psnr_per_bin"] = list(P.idxmax()); r["best_hole_lpips_per_bin"] = list(L.idxmin())
    eff = {m: next(csv.DictReader(open(os.path.join(RES, "efficiency", f"{m}.csv")))) for m in pe.index}
    lat = pd.Series({m: float(v["latency"]) for m, v in eff.items()})
    r["lat_psnr_rho"] = lat.corr(pe.psnr, "spearman"); r["lat_fid_rho"] = lat.corr(pe.fid, "spearman")
    r["lat_ratio"] = float(lat.max() / lat.min()); r["fastest"] = lat.idxmin(); r["fid_best"] = pe.fid.idxmin()
    r["psnr_rank"] = list(pe.psnr.sort_values(ascending=False).index)
    out[tag] = r
for k in out["new"]:
    o, n = out["old"][k], out["new"][k]
    f = lambda v: (np.round(v, 3).tolist() if isinstance(v, (float, tuple, list)) and not (isinstance(v, list) and v and isinstance(v[0], str)) and not (isinstance(v, tuple) and isinstance(v[-1], str)) else v)
    print(("   " if str(o) == str(n) else "** ") + f"{k}: {f(o)}  ->  {f(n)}")
json.dump(out, open(os.path.join(RES, "r6_text_numbers.json"), "w"), indent=1, default=str)
