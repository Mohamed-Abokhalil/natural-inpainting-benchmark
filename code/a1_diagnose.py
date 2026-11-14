"""A1 diagnosis: why do Spa-former and WaveFill (and ZITS) score low inside the holes of the 0-10% bin?
For every method and bin, from the saved outputs:
  * hole-pixel PSNR split by distance to the nearest known pixel (d = 1, 2, >= 3; Chebyshev distance), i.e. thin-stroke
    pixels vs pixels deep inside a hole;
  * fraction of hole pixels whose output lies within 6 grey levels of the method's hole-fill value (copying the input
    instead of inpainting): white (255) for WaveFill-style inputs, mid-grey (127.5) for [-1,1] zero-filled inputs, black (0);
  * known-region deviation |out - gt| (pipeline sanity);
  * consistency of data/places2/masked with gt and mask (does the masked input match the benchmark mask?).
Writes results/a1_diagnosis.csv and prints a summary."""
import os, glob
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(V, "..")
DATA = os.path.join(R, "data", "places2")
PI = pd.read_csv(os.path.join(V, "results", "per_image", "final", "SCAT.csv"), index_col="file")
SRC = lambda m: os.path.join(V, "results", "rerun", m) if m in ("Pluralistic", "VHII", "spa-former") and os.path.isdir(os.path.join(V, "results", "rerun", m)) else os.path.join(DATA, "SOTA", m)
methods = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, "SOTA", "*")))
files = list(PI.index)
ld = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
cache = {}
for f in files:
    m = np.asarray(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127
    d = ndimage.distance_transform_cdt(m, metric="chessboard")       # distance to nearest known pixel, 1 at the border
    cache[f] = (m, d, ld(os.path.join(DATA, "gt", f)))
# masked-input consistency
bad_masked = 0
for f in files:
    m, _, g = cache[f]; mk = ld(os.path.join(DATA, "masked", f))
    known_dev = np.abs(mk[~m] - g[~m]).mean()
    hole_vals = mk[m]
    if known_dev > 1.0: bad_masked += 1
print("masked inputs whose known region deviates from gt by > 1 grey level:", bad_masked, "of", len(files))
rows = []
for meth in methods:
    src = SRC(meth)
    for f in files:
        m, d, g = cache[f]
        p = os.path.join(src, f)
        if not os.path.exists(p): continue
        o = ld(p)
        if o.shape != g.shape: continue
        err = (o - g) ** 2
        r = dict(method=meth, file=f, bin=PI.loc[f, "bin"], hole_px=int(m.sum()),
                 known_dev=float(np.abs(o[~m] - g[~m]).mean()))
        for name, sel in (("d1", m & (d == 1)), ("d2", m & (d == 2)), ("d3", m & (d >= 3))):
            n = sel.sum(); r[f"n_{name}"] = int(n)
            r[f"mse_{name}"] = float(err[sel].mean()) if n else np.nan
        oh = o[m]
        for nm, val in (("white", 255.0), ("grey", 127.5), ("black", 0.0)):
            r[f"frac_{nm}"] = float((np.abs(oh - val).max(axis=1) <= 6).mean())
        rows.append(r)
D = pd.DataFrame(rows); D.to_csv(os.path.join(V, "results", "a1_diagnosis.csv"), index=False)
psnr = lambda mse: 10 * np.log10(255 ** 2 / mse)
agg = D.groupby(["method", "bin"]).agg(n_d1=("n_d1", "sum"), n_d3=("n_d3", "sum"), mse_d1=("mse_d1", "mean"), mse_d2=("mse_d2", "mean"),
                                      mse_d3=("mse_d3", "mean"), white=("frac_white", "mean"), grey=("frac_grey", "mean"),
                                      black=("frac_black", "mean"), known_dev=("known_dev", "mean")).reset_index()
for c in ("d1", "d2", "d3"): agg[f"psnr_{c}"] = psnr(agg[f"mse_{c}"])
pd.set_option("display.width", 220)
sel = agg[agg.method.isin(["spa-former", "WaveFill", "ZITS", "SCAT", "MISF", "LaMa"])]
print(sel[["method", "bin", "n_d1", "n_d3", "psnr_d1", "psnr_d2", "psnr_d3", "white", "grey", "black", "known_dev"]].round(3).to_string(index=False))
share = D.groupby("bin")[["n_d1", "n_d2", "n_d3"]].sum().drop_duplicates()
tot = D[D.method == methods[0]].groupby("bin")[["n_d1", "n_d2", "n_d3"]].sum()
print("\nshare of hole pixels at distance 1 / 2 / >=3 per bin:\n", (tot.div(tot.sum(1), axis=0)).round(3).to_string())
