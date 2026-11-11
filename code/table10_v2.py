"""Table 10, revised protocol (supersedes the Div./PSNR columns of table10.py; table10.py and its JSONs are kept).
- Common inputs: the 10-image diffusion subset (2 per coverage bin, asserted), restricted to inputs with all 10
  completions for all six sampling methods. VHII is excluded: deterministic at inference (see Section 12.5).
- Div.: mean pairwise LPIPS (AlexNet v0.1) between the 10 completions, on the hole bounding box -- the same region
  (the 'bbox' column of results/per_image/final/<m>.csv, >= 32x32) as the masked-region LPIPS of Table 11.
  Also computed on the 100-image subset for the four non-diffusion methods (Div.100, secondary).
- PSNR mean / best-of-10 / s.d.10: full image, as Table 8, over the same completions.
- Bootstrap over images (B=1000, seed 0, percentile 95% CI): each method's mean Div. and all pairwise differences
  (paired: same images resampled for both methods).
- Prec./Rec. and s.d.5 are taken unchanged from results/stats/table10/<m>.json (1,000 outputs; five full runs).
Writes results/stats/table10/v2/{table10_v2.json, div_pairs_ci.csv, per_image_div.csv}."""
import os, json, itertools
import numpy as np, pandas as pd, torch, lpips
from PIL import Image
import torchvision.transforms.functional as TF
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_f
R = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
DATA = os.path.join(R, "data", "places2"); RES = os.path.join(R, "revision", "results"); S = os.path.join(RES, "samples")
OUT = os.path.join(RES, "stats", "table10", "v2"); os.makedirs(OUT, exist_ok=True)
METH = {"Pluralistic": "div100", "CoModGAN": "div100", "latent-code": "div100", "BAT-Fill": "div100", "RePaint": "div10", "StrDiff": "div10"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
t = lambda p: TF.to_tensor(Image.open(p).convert("RGB"))[None].to(dev)
sp = lambda m, f, k: os.path.join(S, m, METH[m], f"{os.path.splitext(f)[0]}_s0_k{k}.png")
rd = lambda n: [l.strip() for l in open(os.path.join(RES, "subsets", n)) if l.strip()]
PI = pd.read_csv(os.path.join(RES, "per_image", "final", "SCAT.csv"), index_col="file")   # bbox and bin are mask properties
sub10, sub100 = rd("diffusion10.txt"), rd("diversity100.txt")
assert set(sub10) <= set(sub100)
bins10 = PI.loc[sub10, "bin"].value_counts().to_dict(); assert len(bins10) == 5 and set(bins10.values()) == {2}, bins10
common = [f for f in sub10 if all(os.path.exists(sp(m, f, k)) for m in METH for k in range(10))]


def bbox(f):
    a, b = PI.loc[f, "bbox"].split(","); y0, y1 = map(int, a.split(":")); x0, x1 = map(int, b.split(":")); return y0, y1, x0, x1


@torch.no_grad()
def per_image(m, f):
    y0, y1, x0, x1 = bbox(f)
    xs = [t(sp(m, f, k)) for k in range(10)]; g = t(os.path.join(DATA, "gt", f))
    cr = [x[..., y0:y1, x0:x1] * 2 - 1 for x in xs]
    div = float(np.mean([lp(a, b).item() for a, b in itertools.combinations(cr, 2)]))
    ps = [psnr_f(x, g, data_range=1.0).item() for x in xs]
    return dict(method=m, file=f, bin=PI.loc[f, "bin"], div_hole=div, psnr=ps)


cache = os.path.join(OUT, "per_image_div.csv"); done = {}
if os.path.exists(cache):
    c = pd.read_csv(cache); done = {(r.method, r.file): r for r in c.itertuples()}
rows = []
for m in METH:
    files = common + ([f for f in sub100 if f not in common] if METH[m] == "div100" else [])
    for f in files:
        if (m, f) in done:
            r = done[(m, f)]; rows.append(dict(method=m, file=f, bin=r.bin, div_hole=r.div_hole, psnr=json.loads(r.psnr))); continue
        rows.append(per_image(m, f))
D = pd.DataFrame(rows); D2 = D.copy(); D2["psnr"] = D2.psnr.map(json.dumps); D2.to_csv(cache, index=False)

rng = np.random.default_rng(0); B = 1000; n = len(common)
idx = rng.integers(0, n, size=(B, n))
C = D[D.file.isin(common)].set_index(["method", "file"])
V = {m: np.array([C.loc[(m, f), "div_hole"] for f in common]) for m in METH}
res = {"n_inputs": n, "inputs": common, "bins": PI.loc[common, "bin"].value_counts().sort_index().to_dict(), "B": B, "seed": 0, "methods": {}}
for m in METH:
    P = np.array([C.loc[(m, f), "psnr"] for f in common])          # n x 10
    bs = V[m][idx].mean(1)
    _o = os.path.join(RES, "stats", "table10", f"{m}.json"); old = json.load(open(_o)) if os.path.exists(_o) else {"precision": None, "recall": None}
    r = dict(div_hole=float(V[m].mean()), div_hole_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             psnr_mean=float(P.mean()), psnr_best10=float(P.max(1).mean()), sd_psnr_samples=float(np.std(P.mean(0), ddof=1)),
             precision=old["precision"], recall=old["recall"])
    if "sd_fid" in old: r.update(sd_psnr=old["sd_psnr"], sd_fid=old["sd_fid"])
    if METH[m] == "div100":
        d100 = D[(D.method == m) & D.file.isin(sub100)]; assert len(d100) == 100
        r["div_hole_100"] = float(d100.div_hole.mean())
    res["methods"][m] = r
pairs = []
for a, b in itertools.combinations(METH, 2):
    d = V[a] - V[b]; bs = d[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    pairs.append(dict(a=a, b=b, diff=float(d.mean()), lo=float(lo), hi=float(hi), significant=bool(lo > 0 or hi < 0)))
pd.DataFrame(pairs).to_csv(os.path.join(OUT, "div_pairs_ci.csv"), index=False)
json.dump(res, open(os.path.join(OUT, "table10_v2.json"), "w"), indent=1)
for m, r in res["methods"].items(): print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print(pd.DataFrame(pairs).round(4).to_string()); print("n_inputs", n, res["bins"])
