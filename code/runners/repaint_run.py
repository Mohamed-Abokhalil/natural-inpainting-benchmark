"""Resumable RePaint sampler for Table 10 / timing, reusing the official repo code and the exact
config used for Table 8 (NII/RePaint/confs/face_example.yml: Places2 places_250000.pt, T=250,
jump_length=10, jump_n_sample=10, no classifier). Inputs are the same files the original run used
(NII/RePaint/data/datasets/masked + masks_inverted). Repo files are only read, never modified.

Usage: python repaint_run.py --list subset.txt --seeds 0 --nsamp 10 --out DIR [--limit k]
Writes DIR/<stem>_s<seed>_k<j>.png; skips files that already exist.
Sample j of seed s is drawn in one batch of nsamp after torch.manual_seed(s * 100000 + image_index).
"""
import argparse, os, sys, time
import numpy as np, torch as th
from PIL import Image

REPO = os.path.join(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"), r"NII\RePaint")
sys.path.insert(0, REPO); os.chdir(REPO)
import conf_mgt
from utils import yamlread
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, select_args

ap = argparse.ArgumentParser()
ap.add_argument("--list", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--nsamp", type=int, default=10); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
conf = conf_mgt.conf_base.Default_Conf(); conf.update(yamlread(os.path.join(REPO, "confs", "face_example.yml")))
conf["show_progress"] = False
dev = "cuda"
model, diffusion = create_model_and_diffusion(**select_args(conf, model_and_diffusion_defaults().keys()), conf=conf)
model.load_state_dict(th.load(conf.model_path, map_location="cpu")); model.to(dev).eval()
def model_fn(x, t, y=None, gt=None, **kw): return model(x, t, None, gt=gt)

files = [l.strip() for l in open(a.list) if l.strip()]
if a.limit: files = files[:a.limit]
allfiles = sorted(os.listdir(os.path.join(REPO, "data", "datasets", "masked")))
for s in a.seeds:
    for f in files:
        stem = os.path.splitext(f)[0]
        outs = [os.path.join(a.out, f"{stem}_s{s}_k{j}.png") for j in range(a.nsamp)]
        if all(os.path.exists(o) for o in outs): continue
        gt = np.array(Image.open(os.path.join(REPO, "data/datasets/masked", f)).convert("RGB")).astype(np.float32) / 127.5 - 1
        km = np.array(Image.open(os.path.join(REPO, "data/datasets/masks_inverted", f)).convert("RGB")).astype(np.float32) / 255.0
        gt = th.from_numpy(gt).permute(2, 0, 1)[None].to(dev); km = th.from_numpy(km).permute(2, 0, 1)[None].to(dev)
        th.manual_seed(s * 100000 + allfiles.index(f)); np.random.seed(s)
        t0 = time.time(); res = []
        for j0 in range(0, a.nsamp, a.bs):
            n = min(a.bs, a.nsamp - j0)
            kw = {"gt": gt.repeat(n, 1, 1, 1), "gt_keep_mask": km.repeat(n, 1, 1, 1), "y": th.zeros(n, dtype=th.long, device=dev)}
            with th.no_grad():
                r = diffusion.p_sample_loop(model_fn, (n, 3, 256, 256), clip_denoised=conf.clip_denoised,
                                            model_kwargs=kw, cond_fn=None, device=dev, progress=False, return_all=True, conf=conf)
            res.append(r["sample"])
        x = ((th.cat(res) + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for j, o in enumerate(outs): Image.fromarray(x[j]).save(o)
        print(f"{f} seed {s}: {time.time() - t0:.1f}s for {a.nsamp}", flush=True)
