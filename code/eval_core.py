"""
eval_core.py -- per-image metrics + cached Inception features for the 24 Places2 methods.

Reuses the ORIGINAL evaluation code (<ROOT>/evaluate_inpainting.py: MetricComputer,
load_image) for the full-image metrics so that values are consistent with Table 8.
Adds (new code): masked-region metrics, NIQE (pyiqa), Inception pool3 feature caching
(pytorch-fid InceptionV3, identical to the original FID path).

Resumable: every method's per-image CSV and feature .npy is cached; existing files are skipped.

Usage:
  python eval_core.py --variant asis        # outputs as saved (protocol of Table 8)
  python eval_core.py --variant composited  # outputs re-composited onto known pixels (sensitivity)
"""
import argparse, csv, importlib.util, os, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

ROOT = Path(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"))
DATA = ROOT / "data" / "places2"
RES = ROOT / "revision" / "results"
DEVICE = "cuda"

spec = importlib.util.spec_from_file_location("orig_eval", ROOT / "evaluate_inpainting.py")
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)

METHODS = orig.NI_METHODS  # the 24 methods, original list
BIN_DIRS = ["0_10", "10_20", "20_30", "30_40", "40_50"]


def image_list():
    files = orig.list_images(DATA / "gt")
    binmap = {}
    for b in BIN_DIRS:
        for f in os.listdir(DATA / "gt_binned" / b):
            binmap[f] = b
    return files, binmap


def load_mask(fname):
    m = np.array(Image.open(DATA / "mask" / fname).convert("L")) > 127  # 255 = hole
    return m


def hole_bbox(m, min_side=32):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H, W = m.shape
    def grow(a0, a1, L):
        if a1 - a0 >= min_side: return a0, a1
        c = (a0 + a1) / 2; a0 = int(round(c - min_side / 2)); a0 = max(0, min(a0, L - min_side))
        return a0, a0 + min_side
    y0, y1 = grow(y0, y1, H); x0, x1 = grow(x0, x1, W)
    return y0, y1, x0, x1


class HoleMetrics:
    def __init__(self):
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim_f
        import lpips
        self.ssim_f = ssim_f
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)

    @torch.no_grad()
    def __call__(self, pred, gt, m):
        mt = torch.from_numpy(m).to(DEVICE)[None, None].float()
        pred, gt = pred.to(DEVICE), gt.to(DEVICE)
        n = mt.sum() * 3
        d = (pred - gt) * mt
        mse = (d ** 2).sum() / n
        l1 = d.abs().sum() / n
        _, smap = self.ssim_f(pred, gt, data_range=1.0, return_full_image=True)
        ssim_h = (smap.mean(1, keepdim=True) * mt).sum() / mt.sum()
        y0, y1, x0, x1 = hole_bbox(m)
        lp = self.lp(pred[..., y0:y1, x0:x1] * 2 - 1, gt[..., y0:y1, x0:x1] * 2 - 1).item()
        return {"psnr_hole": (10 * torch.log10(1.0 / mse)).item(), "ssim_hole": ssim_h.item(),
                "l1_hole": l1.item(), "mse_hole": mse.item(), "lpips_hole_bbox": lp,
                "bbox": f"{y0}:{y1},{x0}:{x1}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["asis", "composited", "rerun"], default="asis")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    files, binmap = image_list()
    assert len(files) == 1000
    out_pi = RES / "per_image" / args.variant; out_pi.mkdir(parents=True, exist_ok=True)
    out_ft = RES / "features" / args.variant; out_ft.mkdir(parents=True, exist_ok=True)
    tmp_img = ROOT / "revision" / "work" / "composited"

    mc = orig.MetricComputer(); hm = HoleMetrics()
    import pyiqa
    niqe = pyiqa.create_metric("niqe", device=DEVICE)
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid import fid_score
    inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE).eval()

    gt_feat = RES / "features" / "gt_inception2048.npy"
    if not gt_feat.exists():
        a = fid_score.get_activations([str(DATA / "gt" / f) for f in files], inc, 64, 2048, DEVICE, num_workers=0)
        np.save(gt_feat, a)

    for meth in (args.methods or METHODS):
        csv_p = out_pi / f"{meth}.csv"; ft_p = out_ft / f"{meth}_inception2048.npy"
        if csv_p.exists() and ft_p.exists():
            print("skip", meth); continue
        pred_dir = (RES / "rerun" / meth) if args.variant == "rerun" else (DATA / "SOTA" / meth)
        rows, paths = [], []
        if args.variant == "composited":
            (tmp_img / meth).mkdir(parents=True, exist_ok=True)
        for f in files:
            gt_t = orig.load_image(DATA / "gt" / f)
            pr_t = orig.load_image(pred_dir / f)
            m = load_mask(f)
            if args.variant == "composited":
                mt = torch.from_numpy(m)[None, None].float()
                pr_t = mt * pr_t + (1 - mt) * gt_t
                # save as 8-bit PNG so FID reads the composited image
                arr = (pr_t[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                p = tmp_img / meth / f; Image.fromarray(arr).save(p); paths.append(str(p))
                pr_t = T.ToTensor()(Image.open(p).convert("RGB")).unsqueeze(0)
            else:
                paths.append(str(pred_dir / f))
            r = {"file": f, "bin": binmap[f], "hole_ratio": float(m.mean())}
            r.update(mc.compute(pr_t, gt_t))
            r.update(hm(pr_t, gt_t, m))
            known = ~m
            r["known_meanabs_255"] = float((pr_t[0] - gt_t[0]).abs().permute(1, 2, 0).numpy()[known].mean() * 255)
            with torch.no_grad():
                r["niqe"] = float(niqe(pr_t.to(DEVICE)).item())
            rows.append(r)
        with open(csv_p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        a = fid_score.get_activations(paths, inc, 64, 2048, DEVICE, num_workers=0)
        np.save(ft_p, a)
        print("done", meth, np.mean([r["psnr"] for r in rows]), flush=True)


if __name__ == "__main__":
    main()
