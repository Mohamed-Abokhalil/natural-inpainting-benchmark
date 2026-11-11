#!/usr/bin/env python3
"""
evaluate_inpainting.py  —  Unified evaluation script for all three tasks
==========================================================================
Covers ALL evaluated methods:

  Places2 (NI):
    BAT-Fill, CMT, CoModGAN, CTSDG, DeepFillv2, DGTS, EdgeConnect,
    LaMa, latent-code, MADIN, MEDFE, MISF, PEN-Net, Pluralistic,
    RePaint, SCAT, spa-former, StrDiff, StructureFlow, VHII,
    WaveFill, ZITS
    (E2I and MADF pending — add to list when outputs are ready)

  CelebA FI:
    AOT-GAN, AttrFaceNet, BAT-Fill, CMT, CodeFormer, CoModGAN,
    CoPaint, CTSDG, DeepFillv2, DGTS, EdgeConnect, GPEN, LaFIn,
    LaMa, latent-code, MADIN, MFCL, MISF, MuFIN, PEN-Net, RePaint,
    SCAT, spa-former, StructureFlow, VHII, WaveFill
    (E2I, MADF, TSFormer pending)

  CelebA FU:
    CoModGAN, CoPaint, DeepFillv2, Din2020, EdgeConnect, GANMasker,
    GLCIC, M2UNet, MuFIN, RePaint, SCAT

Data structure:
  <dataset>/
    gt/           ← ground-truth images
    mask/         ← binary mask images
    masked/       ← corrupted input
    SOTA/<Method>/ ← output images (same filenames as gt/)
    gt_binned/    ← bin membership (NI and FI only)
                     subfolder layout: gt_binned/<bin_label>/<image>
                     OR JSON/CSV: {filename: bin_label}

Usage
-----
  python evaluate_inpainting.py --task ni --data_root data/places2
  python evaluate_inpainting.py --task fi --data_root data/CelebA_FI
  python evaluate_inpainting.py --task fu --data_root data/CelebA_FU
  python evaluate_inpainting.py --task all --data_root data
  python evaluate_inpainting.py --plot_mask_dist --data_root data

Output (per task)
------
  results/<task>/summary_by_bin.csv
      columns: method, bin, n_images, psnr, ssim, fid, lpips, l1, mse

Requirements
------------
  pip install torch torchvision Pillow tqdm matplotlib numpy
  pip install pytorch-fid lpips torchmetrics
"""

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("WARNING: lpips not installed — LPIPS metric unavailable.")

try:
    from pytorch_fid import fid_score as fid_lib
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("WARNING: pytorch-fid not installed — FID metric unavailable.")

try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("WARNING: torchmetrics not installed — PSNR/SSIM unavailable.")

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
IMG_SIZE = 256

# ── Method lists ──────────────────────────────────────────────────────────────
NI_METHODS = [
    "BAT-Fill", "CMT", "CoModGAN", "CTSDG", "DeepFillv2", "DGTS", "E2I",
    "EdgeConnect", "LaMa", "latent-code", "MADF", "MADIN", "MEDFE", "MISF",
    "PEN-Net", "Pluralistic", "RePaint", "SCAT", "spa-former", "StrDiff",
    "StructureFlow", "VHII", "WaveFill", "ZITS", 
]

FI_METHODS = [
    "AOT-GAN", "AttrFaceNet", "BAT-Fill", "CMT", "CodeFormer", "CoModGAN",
    "CoPaint", "CTSDG", "DeepFillv2", "DGTS", "E2I", "EdgeConnect", "GPEN",
    "LaFIn", "LaMa", "latent-code", "MADF", "MADIN", "MFCL", "MISF", "MuFIN", "NDMA",
    "PEN-Net", "Pluralistic", "RePaint", "SCAT", "spa-former", "StructureFlow", "TSFormer", "VHII",
    "WaveFill",
]

FU_METHODS = [
    "CoModGAN", "CoPaint", "DeepFillv2", "Din2020", "EdgeConnect",
    "GANMasker", "GLCIC", "M2UNet", "MuFIN", "RePaint", "SCAT",
]

# Default bin definitions (label, min_ratio, max_ratio)
DEFAULT_BINS: List[Tuple[str, float, float]] = [
    ("0-10",   0.00, 0.10),
    ("10-20",  0.10, 0.20),
    ("20-30",  0.20, 0.30),
    ("30-40",  0.30, 0.40),
    ("40-50",  0.40, 0.50),
    ("50-60",  0.50, 0.60),
]

# ── Small helpers ─────────────────────────────────────────────────────────────

def list_images(folder: Path) -> List[str]:
    """Sorted list of image filenames in a folder."""
    if not folder.exists():
        return []
    return sorted(f.name for f in folder.iterdir()
                  if f.suffix.lower() in IMG_EXTS)


def load_gt_binned(dataset_root: Path) -> Optional[Dict[str, str]]:
    """
    Load {filename: bin_label} from gt_binned/.
    Supports three layouts:
      1. gt_binned/<label>/<image>  — one subfolder per bin
      2. gt_binned/*.json           — {"filename": "bin_label"}
      3. gt_binned/*.csv            — filename,bin_label
    Returns None if gt_binned/ does not exist or is empty.
    """
    gt_binned_dir = dataset_root / "gt_binned"
    if not gt_binned_dir.exists():
        return None

    # Layout 1: subfolder per bin
    subdirs = [p for p in gt_binned_dir.iterdir() if p.is_dir()]
    if subdirs:
        mapping: Dict[str, str] = {}
        for bin_dir in subdirs:
            for img_file in bin_dir.iterdir():
                if img_file.suffix.lower() in IMG_EXTS:
                    mapping[img_file.name] = bin_dir.name
        return mapping if mapping else None

    # Layout 2: JSON
    for json_file in gt_binned_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            return json.load(f)

    # Layout 3: CSV
    for csv_file in gt_binned_dir.glob("*.csv"):
        mapping = {}
        with open(csv_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    mapping[parts[0].strip()] = parts[1].strip()
        return mapping if mapping else None

    return None


def _ratio_in_range(bin_label: str, bmin: float, bmax: float) -> bool:
    """
    Return True when bin_label represents the range [bmin, bmax).
    Handles "20-30%", "20_30", "20-30" etc.
    """
    nums = re.findall(r"\d+", bin_label)
    if len(nums) >= 2:
        lo = int(nums[0]) / 100.0
        hi = int(nums[1]) / 100.0
        return abs(lo - bmin) < 0.005 and abs(hi - bmax) < 0.005
    return False


def _avg(rows: List[Dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}")

# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: Path, size: int = IMG_SIZE) -> Optional[torch.Tensor]:
    """Load an image as float32 [1, 3, H, W] in [0, 1]. Returns None if missing."""
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
        return T.ToTensor()(img).unsqueeze(0)
    except Exception as exc:
        print(f"  WARNING: could not load {path}: {exc}")
        return None

# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricComputer:
    def __init__(self):
        if TORCHMETRICS_AVAILABLE:
            self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
            self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips_lib.LPIPS(net="alex").to(DEVICE)

    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
        pred, gt = pred.to(DEVICE), gt.to(DEVICE)
        result: Dict[str, float] = {}
        if TORCHMETRICS_AVAILABLE:
            result["psnr"] = self.psnr_fn(pred, gt).item()
            result["ssim"] = self.ssim_fn(pred, gt).item()
        result["l1"]  = (pred - gt).abs().mean().item()
        result["mse"] = ((pred - gt) ** 2).mean().item()
        if LPIPS_AVAILABLE:
            result["lpips"] = self.lpips_fn(pred * 2 - 1, gt * 2 - 1).item()
        return result


def compute_fid_for_files(
    pred_dir: Path,
    gt_dir: Path,
    filenames: List[str],
) -> Optional[float]:
    """
    Compute FID between a subset of images (given by filenames) in
    pred_dir and gt_dir.
    Copies the subset to temporary directories so pytorch-fid can read
    them without traversing the full dataset folder.
    """
    if not FID_AVAILABLE or not filenames:
        return None
    tmp_pred = Path(tempfile.mkdtemp())
    tmp_gt   = Path(tempfile.mkdtemp())
    try:
        for fname in filenames:
            src_p = pred_dir / fname
            src_g = gt_dir   / fname
            if src_p.exists():
                shutil.copy2(src_p, tmp_pred / fname)
            if src_g.exists():
                shutil.copy2(src_g, tmp_gt / fname)
        return fid_lib.calculate_fid_given_paths(
            [str(tmp_pred), str(tmp_gt)],
            batch_size=64, device=DEVICE, dims=2048,
        )
    except Exception as exc:
        print(f"  WARNING: FID computation failed: {exc}")
        return None
    finally:
        shutil.rmtree(tmp_pred, ignore_errors=True)
        shutil.rmtree(tmp_gt,   ignore_errors=True)

# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_task(
    task_name:    str,
    dataset_root: Path,
    method_names: List[str],
    out_dir:      Path,
    bins:         Optional[List[Tuple[str, float, float]]] = None,
) -> None:
    """
    Evaluate all methods for one task.

    Parameters
    ----------
    bins : list of (label, min_ratio, max_ratio), optional
        When provided (NI and FI tasks), results are broken down by
        mask-ratio bin using gt_binned/.
        When None (FU task or no gt_binned available), all images are
        aggregated into a single "all" bin.

    Output
    ------
    results/<task_name>/summary_by_bin.csv
        One row per (method, bin) combination.
        Columns: method, bin, n_images, psnr, ssim, fid, lpips, l1, mse
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  Dataset: {dataset_root}")
    print(f"{'='*60}")

    gt_dir      = dataset_root / "gt"
    sota_dir    = dataset_root / "SOTA"
    results_dir = out_dir / task_name
    results_dir.mkdir(parents=True, exist_ok=True)

    gt_files = list_images(gt_dir)
    if not gt_files:
        print(f"  ERROR: no images found in {gt_dir}. Skipping task.")
        return
    print(f"  GT images: {len(gt_files)}")

    # Build bin → [filenames] mapping
    bin_map = load_gt_binned(dataset_root) if bins else None

    if bin_map and bins:
        bin_files: Dict[str, List[str]] = {}
        for bl, bmin, bmax in bins:
            matched = [f for f in gt_files
                       if _ratio_in_range(bin_map.get(f, ""), bmin, bmax)]
            if matched:
                bin_files[bl] = matched
                print(f"  Bin {bl}: {len(matched)} images")
            else:
                print(f"  Bin {bl}: empty — skipping.")
    else:
        bin_files = {"all": gt_files}
        if bins and not bin_map:
            print("  WARNING: gt_binned/ not found — aggregating all images.")

    computer     = MetricComputer()
    summary_rows: List[Dict] = []

    for method in method_names:
        pred_dir = sota_dir / method
        if not pred_dir.exists():
            print(f"  SKIP {method}: folder not found at {pred_dir}")
            continue

        print(f"  Evaluating {method} ...")

        for bin_label, files in bin_files.items():
            metrics_list: List[Dict] = []

            for fname in tqdm(files, desc=f"    {method}/{bin_label}", leave=False):
                gt_t   = load_image(gt_dir   / fname)
                pred_t = load_image(pred_dir / fname)
                if gt_t is None or pred_t is None:
                    continue
                metrics_list.append(computer.compute(pred_t, gt_t))

            if not metrics_list:
                continue

            # FID computed at bin level (distribution metric)
            fid_val = compute_fid_for_files(pred_dir, gt_dir, files)

            row: Dict = {
                "method":   method,
                "bin":      bin_label,
                "n_images": len(metrics_list),
                "psnr":     _avg(metrics_list, "psnr"),
                "ssim":     _avg(metrics_list, "ssim"),
                "fid":      fid_val,
                "lpips":    _avg(metrics_list, "lpips"),
                "l1":       _avg(metrics_list, "l1"),
                "mse":      _avg(metrics_list, "mse"),
            }
            summary_rows.append(row)

            psnr_s  = f"{row['psnr']:.2f}"  if row["psnr"]  is not None else "---"
            ssim_s  = f"{row['ssim']:.4f}"  if row["ssim"]  is not None else "---"
            fid_s   = f"{row['fid']:.2f}"   if row["fid"]   is not None else "---"
            lpips_s = f"{row['lpips']:.4f}" if row["lpips"] is not None else "---"
            print(f"    [{bin_label}] PSNR={psnr_s} SSIM={ssim_s} "
                  f"FID={fid_s} LPIPS={lpips_s}")

    _write_csv(summary_rows, results_dir / "summary_by_bin.csv")

# ── Mask-ratio distribution plot ──────────────────────────────────────────────

def plot_mask_distribution(data_root: Path, out_dir: Path) -> None:
    """Generate the two-panel mask-ratio distribution bar chart (Fig 9b)."""
    print("\nGenerating mask-ratio distribution figure ...")

    tasks = [
        ("places2",   "Places2\n(1K images)",                    "#5ba4cf"),
        ("CelebA_FI", "CelebA — Face Inpainting\n(1K images)",   "#e07b54"),
    ]
    BIN_LABELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, (task_dir_name, title, color) in zip(axes, tasks):
        ds_root = data_root / task_dir_name
        bin_map = load_gt_binned(ds_root)

        if bin_map is None:
            gt_binned_dir = ds_root / "gt_binned"
            if gt_binned_dir.exists():
                counts_raw: Dict[str, int] = {}
                for sub in gt_binned_dir.iterdir():
                    if sub.is_dir():
                        n = sum(1 for f in sub.iterdir()
                                if f.suffix.lower() in IMG_EXTS)
                        counts_raw[sub.name] = n
            else:
                print(f"  WARNING: gt_binned not found for {task_dir_name}.")
                counts_raw = {}
        else:
            from collections import Counter
            counts_raw = dict(Counter(bin_map.values()))

        # Match each canonical bin label
        counts: List[int] = []
        for b in BIN_LABELS:
            matched = 0
            for k, v in counts_raw.items():
                nums_b = [int(x) for x in re.findall(r"\d+", b)]
                nums_k = [int(x) for x in re.findall(r"\d+", k)]
                if len(nums_b) >= 2 and len(nums_k) >= 2:
                    if nums_b[0] == nums_k[0] and nums_b[1] == nums_k[1]:
                        matched = v
                        break
            counts.append(matched)

        bars = ax.bar(range(len(BIN_LABELS)), counts, color=color,
                      edgecolor="white", linewidth=0.5, width=0.65)
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Number of images", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2, str(cnt),
                        ha="center", va="bottom", fontsize=7)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"fig9b_mask_distribution.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"  Saved → {p}")
    plt.close(fig)

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified inpainting evaluation script (per-bin output)."
    )
    parser.add_argument("--task", choices=["ni", "fi", "fu", "all"],
                        default="all")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root containing places2/, CelebA_FI/, CelebA_FU/.")
    parser.add_argument("--out_dir",   type=str, default="results")
    parser.add_argument("--plot_mask_dist", action="store_true",
                        help="Generate mask-ratio distribution figure only.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)

    if args.plot_mask_dist:
        plot_mask_distribution(data_root, out_dir)
        return

    if args.task in ("ni", "all"):
        evaluate_task(
            task_name    = "ni_places2",
            dataset_root = data_root / "places2",
            method_names = NI_METHODS,
            out_dir      = out_dir,
            bins         = None, #DEFAULT_BINS,
        )

    if args.task in ("fi", "all"):
        evaluate_task(
            task_name    = "fi_celeba",
            dataset_root = data_root / "CelebA_FI",
            method_names = FI_METHODS,
            out_dir      = out_dir,
            bins         = None, #DEFAULT_BINS,
        )

    if args.task in ("fu", "all"):
        evaluate_task(
            task_name    = "fu_celeba",
            dataset_root = data_root / "CelebA_FU",
            method_names = FU_METHODS,
            out_dir      = out_dir,
            bins         = None,   # FU has no gt_binned — single "all" bin
        )

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()