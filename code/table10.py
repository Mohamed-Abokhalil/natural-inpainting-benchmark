"""Table 10: diversity and seed sensitivity of the six stochastic methods.
- Div.: mean pairwise LPIPS (AlexNet, v0.1, full image) between the N=10 samples of each input, averaged over inputs
  (100-image subset; 10-image subset for RePaint and StrDiff).
- PSNR mean / best-of-10 over the same samples (full image, as Table 8).
- Prec./Rec.: improved precision and recall (Kynkaanniemi et al. 2019, k=3) between the 1,000 benchmark outputs
  (one sample per input, the outputs of Table 8) and the 1,000 ground-truth images, VGG-16 fc2 features
  (torchvision ImageNet weights) of the hole bounding box (>=32x32) resized to 224 -- full-image features saturate at 1.0
  because outputs and references share all known pixels.
- s.d. PSNR / FID: standard deviation over five seeds (1-5) of the full 1,000-image run; for RePaint and StrDiff,
  s.d. of the subset-mean PSNR over the 10 samples (FID not computed).
Cached per-method results in results/stats/table10/<method>.json."""
import os, sys, json, itertools
import numpy as np, torch, pandas as pd
from PIL import Image
import torchvision, torchvision.transforms.functional as TF
R = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
DATA = os.path.join(R, "data", "places2"); RES = os.path.join(R, "revision", "results"); S = os.path.join(RES, "samples")
OUT = os.path.join(RES, "stats", "table10"); os.makedirs(OUT, exist_ok=True)
dev = "cuda"
import lpips
from pytorch_fid.inception import InceptionV3
from pytorch_fid import fid_score
from pytorch_fid.fid_score import calculate_frechet_distance
lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_f
SUBSETS = {"div100": [l.strip() for l in open(os.path.join(RES, "subsets", "diversity100.txt")) if l.strip()],
           "div10": [l.strip() for l in open(os.path.join(RES, "subsets", "diffusion10.txt")) if l.strip()]}
METH = {"Pluralistic": "div100", "CoModGAN": "div100", "latent-code": "div100", "BAT-Fill": "div100", "RePaint": "div10", "StrDiff": "div10"}
SRC1000 = lambda m: os.path.join(RES, "rerun", m) if m == "Pluralistic" else os.path.join(DATA, "SOTA", m)
FILES = sorted(os.listdir(os.path.join(DATA, "gt")))


def t(p): return TF.to_tensor(Image.open(p).convert("RGB"))[None].to(dev)


vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(dev).eval()
fc2 = torch.nn.Sequential(*list(vgg.classifier.children())[:5])
norm = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_core import hole_bbox


def crop(p):  # hole bounding box (>= 32x32) of the benchmark mask of this image, as for the masked-region LPIPS
    f = os.path.basename(p).split("_s")[0]; f = f if f.endswith(".png") else f + ".png"
    m = np.array(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127
    y0, y1, x0, x1 = hole_bbox(m); return t(p)[..., y0:y1, x0:x1]


@torch.no_grad()
def vggfeat(paths):
    out = []
    for i in range(0, len(paths), 50):
        x = torch.cat([norm(torch.nn.functional.interpolate(crop(p), size=(224, 224), mode="bilinear", align_corners=False)) for p in paths[i:i + 50]])
        f = vgg.avgpool(vgg.features(x)).flatten(1); out.append(fc2(f).cpu())
    return torch.cat(out).double()


def pr(real, fake, k=3):
    def radii(X):
        d = torch.cdist(X, X); return d.kthvalue(k + 1, dim=1).values
    rr, rf = radii(real), radii(fake)
    d = torch.cdist(fake, real)
    prec = (d <= rr[None, :]).any(1).double().mean().item()
    rec = (d.T <= rf[None, :]).any(1).double().mean().item()
    return prec, rec


def main(methods):
    gtv = os.path.join(OUT, "gt_vgg_holebbox.pt")
    if not os.path.exists(gtv): torch.save(vggfeat([os.path.join(DATA, "gt", f) for f in FILES]), gtv)
    G = torch.load(gtv)
    gt_inc = np.load(os.path.join(RES, "features", "gt_inception2048.npy")).astype(np.float64)
    inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    for m in methods:
        res = {}
        sub = METH[m]; d = os.path.join(S, m, sub)
        # only inputs whose 10 samples are all present (diffusion sampling may be incomplete at hand-off)
        files = [f for f in SUBSETS[sub] if all(os.path.exists(os.path.join(d, f"{os.path.splitext(f)[0]}_s0_k{k}.png")) for k in range(10))]
        res["n_planned"] = len(SUBSETS[sub]); res["inputs_used"] = files
        if not files: print(m, "no complete inputs"); continue
        divs, pm, pb, per_k = [], [], [], {k: [] for k in range(10)}
        with torch.no_grad():
            for f in files:
                st = os.path.splitext(f)[0]
                xs = [t(os.path.join(d, f"{st}_s0_k{k}.png")) for k in range(10)]
                g = t(os.path.join(DATA, "gt", f))
                divs.append(np.mean([lp(a * 2 - 1, b * 2 - 1).item() for a, b in itertools.combinations(xs, 2)]))
                ps = [psnr_f(x, g, data_range=1.0).item() for x in xs]
                pm.append(np.mean(ps)); pb.append(np.max(ps))
                for k in range(10): per_k[k].append(ps[k])
        res.update(n_inputs=len(files), div=float(np.mean(divs)), psnr_mean=float(np.mean(pm)), psnr_best10=float(np.mean(pb)),
                   subset=sub, sd_psnr_samples=float(np.std([np.mean(v) for v in per_k.values()], ddof=1)))
        P, Rr = pr(G, vggfeat([os.path.join(SRC1000(m), f) for f in FILES]))
        res.update(precision=P, recall=Rr)
        prev = os.path.join(OUT, f"{m}.json")
        old = json.load(open(prev)) if os.path.exists(prev) else {}
        if "seed_fid" in old:   # reuse cached 5-seed results
            res.update({k: old[k] for k in ("seed_psnr", "seed_fid", "sd_psnr", "sd_fid")})
        sd = os.path.join(S, m, "seeds")
        if "seed_fid" not in res and os.path.isdir(sd) and len(os.listdir(sd)) >= 5000:
            ps_seed, fid_seed = [], []
            for s in range(1, 6):
                paths = [os.path.join(sd, f"{os.path.splitext(f)[0]}_s{s}_k0.png") for f in FILES]
                vals = []
                with torch.no_grad():
                    for f, p in zip(FILES, paths):
                        vals.append(psnr_f(t(p), t(os.path.join(DATA, "gt", f)), data_range=1.0).item())
                ps_seed.append(float(np.mean(vals)))
                a = fid_score.get_activations(paths, inc, 64, 2048, dev, num_workers=0).astype(np.float64)
                fid_seed.append(float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False), gt_inc.mean(0), np.cov(gt_inc, rowvar=False))))
            res.update(seed_psnr=ps_seed, seed_fid=fid_seed, sd_psnr=float(np.std(ps_seed, ddof=1)), sd_fid=float(np.std(fid_seed, ddof=1)))
        json.dump(res, open(os.path.join(OUT, f"{m}.json"), "w"), indent=1)
        print(m, json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(METH))
