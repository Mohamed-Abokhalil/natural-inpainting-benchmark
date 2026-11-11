"""Corrected, resumable inference for methods whose original benchmark run used mismatched masks
(VHII, Spa-former, Pluralistic) and for the GAN-type stochastic samples of Table 10.
Each image is processed with ITS OWN benchmark mask (data/places2/mask, 255 = hole), at 256x256
(no resize/crop), with the network, checkpoint, normalisation and output compositing of the
official test code. Seeds: torch.manual_seed(seed*100000 + index_in_sorted_list) per image.

Usage: python gan_run.py --method VHII --list FILE|all --seeds 0 [--nsamp 1] --out DIR
Output: DIR/<stem>.png when nsamp==1 and one seed (benchmark format), else DIR/<stem>_s<seed>_k<j>.png
"""
import argparse, os, sys, importlib
import numpy as np, torch
from PIL import Image
ROOT = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
FI = os.path.join(ROOT, "FI", "000_Related work"); NII = os.path.join(ROOT, "NII")
DATA = os.path.join(ROOT, "data", "places2")
dev = "cuda"


def enter(p):
    sys.path.insert(0, p); os.chdir(p)


def build(method):
    if method == "VHII":
        enter(os.path.join(FI, "variable-hyperparameter-image-impainting"))
        g = importlib.import_module("model.VHII_efficient").InpaintGenerator().to(dev)
        g.load_state_dict(torch.load("trained_models/places/places_VHII_efficient/best_model_places.pth", map_location=dev)["netG"]); g.eval()
        def run(img01, hole):  # test_dataset.py: feats in [-1,1], masked = feats*(1-m); composite with input
            x = img01 * 2 - 1
            y = g(x * (1 - hole), hole)
            y = (y + 1) / 2
            return y * hole + img01 * (1 - hole)
        return run
    if method == "spa-former":
        enter(os.path.join(FI, "spa-former"))
        from model import netU_spa_former
        g = netU_spa_former.define_g(gpu_ids=[0])
        sd = torch.load("checkpoints/place2/latest_net_G.pth", map_location=dev)
        (g.module if hasattr(g, "module") else g).load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}); g.eval()
        def run(img01, hole):  # pluralistic_model.test(): mask = 1 keep; img_out = g*(1-mask)+truth*mask
            keep = 1 - hole; t = img01 * 2 - 1
            y = g(keep * t, keep)
            return ((y * (1 - keep) + t * keep) + 1) / 2
        return run
    if method == "Pluralistic":
        enter(os.path.join(NII, "Pluralistic"))
        from model import network
        import torch.nn.functional as F
        E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=5, norm='none', activation='LeakyReLU', init_type='orthogonal', gpu_ids=[0])
        G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=4, init_type='orthogonal', gpu_ids=[0])
        for net, f in ((E, "latest_net_E.pth"), (G, "latest_net_G.pth")):
            net.load_state_dict(torch.load(os.path.join("checkpoints/place2_random", f), map_location=dev))
        E.eval(); G.eval()
        from util import task
        def run(img01, hole):  # pluralistic_model.test() with nsampling=1
            keep = (1 - hole).repeat(1, 3, 1, 1); t = img01 * 2 - 1; im = keep * t
            dist, f = E(im)
            z = torch.distributions.Normal(dist[-1][0], dist[-1][1]).sample()
            sm = task.scale_img(keep, size=[f[2].size(2), f[2].size(3)])
            g, _ = G(z, f_m=f[-1], f_e=f[2], mask=sm.chunk(3, dim=1)[0])
            return ((1 - keep) * g[-1] + keep * im + 1) / 2
        return run
    raise SystemExit("unknown method")


ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--list", default="all")
ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--nsamp", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
allfiles = sorted(os.listdir(os.path.join(DATA, "gt")))
files = allfiles if a.list == "all" else [l.strip() for l in open(a.list) if l.strip()]
run = build(a.method)
bench = (a.nsamp == 1 and len(a.seeds) == 1)
with torch.no_grad():
    for s in a.seeds:
        for f in files:
            stem = os.path.splitext(f)[0]
            outs = [os.path.join(a.out, f) if bench else os.path.join(a.out, f"{stem}_s{s}_k{j}.png") for j in range(a.nsamp)]
            if all(os.path.exists(o) for o in outs): continue
            img = torch.from_numpy(np.array(Image.open(os.path.join(DATA, "gt", f)).convert("RGB"))).float().permute(2, 0, 1)[None].to(dev) / 255
            hole = torch.from_numpy((np.array(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127).astype(np.float32))[None, None].to(dev)
            torch.manual_seed(s * 100000 + allfiles.index(f)); np.random.seed(s)
            for j, o in enumerate(outs):
                y = run(img, hole).clamp(0, 1)
                Image.fromarray((y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(o)
print("done", a.method)
