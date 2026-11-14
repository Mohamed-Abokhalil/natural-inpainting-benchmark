"""
stats_tables.py -- Table 8 reproduction check, Table 9 (FID CI, KID, P-/U-IDS, NIQE),
paired-bootstrap ties for every Table 8 metric, Table 11 and Supplementary S1/S2.

Inputs: results/per_image/<variant>/*.csv and results/features/... from eval_core.py
Outputs: results/stats/<variant>/*.csv|json (cached bootstrap arrays in .npz)

FID uses pytorch_fid.fid_score.calculate_frechet_distance for the point estimate
(identical to the original Table 8 pipeline). For the 1,000 bootstrap resamples the
trace term tr(sqrt(S1 S2)) is computed exactly as the nuclear norm of A B^T
(A, B centred feature matrices / sqrt(n-1)), which is valid because n < d; this is
checked against the scipy sqrtm value on the full set before use.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
RES = ROOT / "revision" / "results"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "asis"
OUT = RES / "stats" / VARIANT; OUT.mkdir(parents=True, exist_ok=True)
B = 1000; SEED = 0
DEV = "cuda"
from pytorch_fid.fid_score import calculate_frechet_distance

METHODS = ["DeepFillv2", "MADF", "Pluralistic", "PEN-Net", "VHII", "CoModGAN", "SCAT", "latent-code",
           "EdgeConnect", "StructureFlow", "E2I", "MEDFE", "CTSDG", "DGTS", "ZITS",
           "BAT-Fill", "WaveFill", "LaMa", "MISF", "CMT", "spa-former", "MADIN", "RePaint", "StrDiff"]
METHODS = [m for m in METHODS if (RES / "per_image" / VARIANT / f"{m}.csv").exists()]
BINS = ["0_10", "10_20", "20_30", "30_40", "40_50"]

pi = {m: pd.read_csv(RES / "per_image" / VARIANT / f"{m}.csv") for m in METHODS}
files = pi[METHODS[0]]["file"].tolist()
for m in METHODS: assert pi[m]["file"].tolist() == files
gt_f = np.load(RES / "features" / "gt_inception2048.npy").astype(np.float64)
ft = {m: np.load(RES / "features" / VARIANT / f"{m}_inception2048.npy").astype(np.float64) for m in METHODS}
n = len(files)


def fid_np(a, b):
    return calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def fid_fast(Xa, Xb):  # torch float64, rows = samples
    k = Xa.shape[0]
    A = Xa - Xa.mean(0, keepdim=True); Bm = Xb - Xb.mean(0, keepdim=True)
    mu = ((Xa.mean(0) - Xb.mean(0)) ** 2).sum()
    tr = (A * A).sum() / (k - 1) + (Bm * Bm).sum() / (k - 1)
    C = A @ Bm.T
    nuc = torch.linalg.eigvalsh(C @ C.T).clamp_min(0).sqrt().sum() / (k - 1)   # = sum of singular values of C
    return (mu + tr - 2 * nuc).item()


def kid_unbiased(X, Y):  # polynomial kernel, unbiased MMD^2 (Binkowski et al. 2018)
    d = X.shape[1]
    kxx = (X @ X.T / d + 1) ** 3; kyy = (Y @ Y.T / d + 1) ** 3; kxy = (X @ Y.T / d + 1) ** 3
    m = X.shape[0]
    return ((kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
            - 2 * kxy.mean()).item()


def ids(real, fake):  # exact reference logic of co-mod-gan metrics/inception_discriminative_score.py
    import sklearn.svm
    svm = sklearn.svm.LinearSVC(dual=False)
    X = np.concatenate([real, fake]); y = np.array([1] * real.shape[0] + [0] * fake.shape[0])
    svm.fit(X, y)
    u = 1 - svm.score(X, y)
    p = np.mean(svm.decision_function(fake) > svm.decision_function(real))
    return float(p), float(u)


summary = {}
# ---------- Table 8 reproduction + point estimates ----------
orig = pd.read_csv(ROOT / "results" / "ni_places2" / "summary_by_bin.csv").set_index("method")
rows = []
for m in METHODS:
    d = pi[m]
    r = {"method": m, "psnr": d.psnr.mean(), "ssim": d.ssim.mean(), "lpips": d.lpips.mean(), "l1": d.l1.mean(),
         "mse": d.mse.mean(), "fid": fid_np(ft[m], gt_f), "niqe": d.niqe.mean(),
         "known_meanabs_255": d.known_meanabs_255.mean()}
    if m in orig.index and VARIANT == "asis":
        for k in ["psnr", "ssim", "lpips", "l1", "fid"]:
            r[f"orig_{k}"] = orig.loc[m, k]; r[f"diff_{k}"] = r[k] - orig.loc[m, k]
    rows.append(r)
point = pd.DataFrame(rows).set_index("method"); point.to_csv(OUT / "point_estimates.csv")
print(point[[c for c in point.columns if c.startswith("diff_")]].abs().max())

# ---------- check fast FID vs scipy ----------
G = torch.from_numpy(gt_f).to(DEV)
F = {m: torch.from_numpy(ft[m]).to(DEV) for m in METHODS}
chk = {m: (fid_fast(F[m], G), point.loc[m, "fid"]) for m in METHODS[:3]}
print("fast vs scipy FID", chk)

# ---------- KID (all 1000 + subsets) and IDS ----------
rng = np.random.default_rng(SEED)
sub_idx = [rng.choice(n, 500, replace=False) for _ in range(100)]
for m in METHODS:
    point.loc[m, "kid_full"] = kid_unbiased(F[m], G)
    ks = [kid_unbiased(F[m][s], G[s]) for s in sub_idx]
    point.loc[m, "kid_sub_mean"] = float(np.mean(ks)); point.loc[m, "kid_sub_sd"] = float(np.std(ks, ddof=1))
    p, u = ids(gt_f, ft[m]); point.loc[m, "pids"] = p; point.loc[m, "uids"] = u
    print("ids", m, p, u, flush=True)
point.to_csv(OUT / "point_estimates.csv")

# ---------- bootstrap (common resamples across methods => paired) ----------
bs_path = OUT / "bootstrap.npz"
rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))
FORCE = [x for x in os.environ.get("STATS_RECOMPUTE", "").split(",") if x]   # methods whose outputs changed
bs = dict(np.load(bs_path)) if bs_path.exists() else {}
todo = [m for m in METHODS if m in FORCE or f"{m}|fid" not in bs]
if todo:
    for m in todo:
        d = pi[m]
        for k in ["psnr", "ssim", "lpips", "l1"]:
            v = d[k].to_numpy(); bs[f"{m}|{k}"] = v[idx].mean(1)
        fids, kids = np.empty(B), np.empty(B)
        for b in range(B):
            ii = torch.from_numpy(idx[b]).to(DEV)
            fids[b] = fid_fast(F[m][ii], G[ii])
            kids[b] = kid_unbiased(F[m][ii], G[ii])
        bs[f"{m}|fid"] = fids; bs[f"{m}|kid"] = kids
        print("boot", m, np.percentile(fids, [2.5, 97.5]), flush=True)
    np.savez(bs_path, **bs)

for m in METHODS:
    lo, hi = np.percentile(bs[f"{m}|fid"], [2.5, 97.5]); point.loc[m, "fid_ci_lo"] = lo; point.loc[m, "fid_ci_hi"] = hi
    # bootstrap of FID with duplicated samples is biased upward; report basic percentile interval of the
    # resampled FID and also the bias-shifted (pivot) interval for transparency
    point.loc[m, "fid_boot_mean"] = bs[f"{m}|fid"].mean()
    point.loc[m, "fid_ci_basic_lo"] = 2 * point.loc[m, "fid"] - hi
    point.loc[m, "fid_ci_basic_hi"] = 2 * point.loc[m, "fid"] - lo
point.to_csv(OUT / "point_estimates.csv")

# ---------- ties: best vs second and anything within 1% of best ----------
better_high = {"psnr": True, "ssim": True, "lpips": False, "l1": False, "fid": False, "kid": False}
ties = []
for k, hi_better in better_high.items():
    col = "kid_full" if k == "kid" else k
    s = point[col].sort_values(ascending=not hi_better)
    best = s.index[0]; bval = s.iloc[0]
    cands = [s.index[1]] + [mm for mm in s.index[2:] if abs(s[mm] - bval) <= 0.01 * abs(bval)]
    for c in cands:
        diff = bs[f"{best}|{k}"] - bs[f"{c}|{k}"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        ties.append({"metric": k, "best": best, "best_val": bval, "other": c, "other_val": s[c],
                     "rank_other": int(list(s.index).index(c)) + 1,
                     "diff_point": bval - s[c], "ci_lo": lo, "ci_hi": hi, "significant": not (lo <= 0 <= hi)})
# explicit claims in the manuscript
for (a, b, k) in [("SCAT", "MADIN", "fid"), ("MADF", "VHII", "l1"), ("MADF", "MADIN", "l1"), ("VHII", "MADIN", "l1"),
                  ("SCAT", "MADF", "l1"), ("MADIN", "SCAT", "lpips"), ("SCAT", "MADIN", "kid")]:
    if a in METHODS and b in METHODS:
        diff = bs[f"{a}|{k}"] - bs[f"{b}|{k}"]; lo, hi = np.percentile(diff, [2.5, 97.5])
        col = "kid_full" if k == "kid" else k
        ties.append({"metric": k, "best": a, "best_val": point.loc[a, col], "other": b, "other_val": point.loc[b, col],
                     "rank_other": -1, "diff_point": point.loc[a, col] - point.loc[b, col], "ci_lo": lo, "ci_hi": hi,
                     "significant": not (lo <= 0 <= hi)})
pd.DataFrame(ties).to_csv(OUT / "paired_ties.csv", index=False)

# ---------- per-bin ----------
pb = []
for m in METHODS:
    d = pi[m]
    for b in BINS + ["all"]:
        dd = d if b == "all" else d[d.bin == b]
        r = {"method": m, "bin": b, "n": len(dd)}
        for k in ["psnr", "ssim", "lpips", "l1", "psnr_hole", "ssim_hole", "l1_hole", "lpips_hole_bbox", "niqe"]:
            r[k] = dd[k].mean()
        sel = dd.index.to_numpy()
        r["fid"] = fid_np(ft[m][sel], gt_f[sel]) if len(sel) >= 100 else np.nan
        r["kid"] = kid_unbiased(F[m][torch.from_numpy(sel).to(DEV)], G[torch.from_numpy(sel).to(DEV)]) if len(sel) >= 2 else np.nan
        pb.append(r)
pd.DataFrame(pb).to_csv(OUT / "per_bin.csv", index=False)
print("done")
