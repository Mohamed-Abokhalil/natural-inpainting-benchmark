#!/usr/bin/env python3
"""
Qualitative visual-comparison figures for the survey (Sec. 11.5).

LAYOUT
------
Two figures TOTAL (not two per bin).

  rows    = "Masked input", "Ground truth", then 12 methods per figure,
            one method per row, in Table `tab:benchmark` order (by PSNR).
            Figure 1 = methods 1-12, Figure 2 = methods 13-24.
            Both figures repeat the masked/GT rows so each reads standalone.

  columns = 2 randomly chosen scenes from EACH of the 5 coverage bins
            (0-10, 10-20, 20-30, 30-40, 40-50) = 10 columns, grouped and
            titled by bin percentage.

  The left gutter carries the row labels and is headed "Method".

A reader scans a column downward to compare every method on one image, and
scans left-to-right to see how each method degrades as coverage grows.

PAGE GEOMETRY
-------------
sn-jnl A4 text area ~180 x 247 mm. Height binds: 26 rows would force ~9.5 mm
tiles. 14 rows (masked + gt + 12 methods) x 10 columns gives ~16.6 mm tiles
and a ~180 x 239 mm figure -- full text width, fits at width=\\linewidth.

FOLDER LAYOUT (pass the parent with --root)
-------------------------------------------
    ROOT/
      gt_binned/         # drives bin membership
        0_10/   <scene>.png
        10_20/  <scene>.png
        20_30/  <scene>.png
        30_40/  <scene>.png
        40_50/  <scene>.png
      gt/      <scene>.png
      mask/    <scene>.png
      masked/  <scene>.png
      SOTA/
        SCAT/  <scene>.png
        ...              # one folder per method

Bins auto-discovered from gt_binned/ and sorted numerically. Files matched by
stem, so extensions may differ between folders.

OUTPUT
------
    figs/fig_visual_p1.pdf   (masked, gt, methods 1-12)
    figs/fig_visual_p2.pdf   (masked, gt, methods 13-24)

RUN
---
    python3 make_visual_comparison.py --root /path/to/ROOT
"""
import argparse
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "pdf.fonttype": 42,   # embed TrueType, not Type3 (Springer requirement)
    "ps.fonttype": 42,
})

# Method folder names under SOTA/, in Table `tab:benchmark` order (by PSNR).
METHODS = [  # order of corrected Table 8 PSNR (revision)
    "SCAT", "MISF", "MADIN", "MADF", "VHII", "CMT", "LaMa", "E2I",
    "StructureFlow", "DeepFillv2", "BAT-Fill", "RePaint", "Latent-code", "Spa-former",
    "EdgeConnect", "MEDFE", "CTSDG", "WaveFill", "Pluralistic", "CoModGAN",
    "DGTS", "StrDiff", "PEN-Net", "ZITS",
]
RERUN = os.path.join(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"), r"revision/results/rerun")
CORRECTED = {"VHII": "VHII", "Spa-former": "spa-former", "Pluralistic": "Pluralistic"}  # protocol-corrected outputs

PRETTY = {"gt": "Ground truth", "masked": "Masked input", "mask": "Mask"}
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
OUTDIR = os.path.join(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"), r"revision/results/figures")

TEXT_W_IN, TEXT_H_IN = 7.09, 9.72   # sn-jnl A4 text area
LABEL_W_IN = 1.00                   # left gutter for method names
HEADER_IN = 0.34                    # top strip for bin group titles


def resolve(folder, stem):
    for e in EXTS:
        p = os.path.join(folder, stem + e)
        if os.path.isfile(p):
            return p
    return None


def load(path, max_px):
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if max_px and max(im.size) > max_px:
            r = max_px / max(im.size)
            im = im.resize((max(1, int(im.width * r)),
                            max(1, int(im.height * r))), Image.LANCZOS)
        return im
    except ImportError:
        return mpimg.imread(path)


def bin_sort_key(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else 10**6


def bin_label(name):
    m = re.match(r"(\d+)[_\-](\d+)", name)
    return f"{m.group(1)}\u2013{m.group(2)}%" if m else name


def scene_stems(bindir):
    out = []
    for f in sorted(os.listdir(bindir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in EXTS:
            out.append(stem)
    return out


def draw_tile(ax, path, max_px):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if path is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                fontsize=5, color="#bbbbbb", transform=ax.transAxes)
    else:
        ax.imshow(load(path, max_px), interpolation="lanczos", aspect="equal")


def render(rowspec, columns, groups, outstem, max_px, dpi):
    """rowspec: [(label, folder)]; columns: [(binname, stem)];
       groups:   [(binname, first_col_idx, ncols_in_group)]"""
    nrows, ncols = len(rowspec), len(columns)
    tile = min((TEXT_W_IN - LABEL_W_IN) / ncols, (TEXT_H_IN - HEADER_IN) / nrows)
    fig_w = LABEL_W_IN + ncols * tile
    fig_h = nrows * tile + HEADER_IN

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    missing = []
    for i, (label, folder) in enumerate(rowspec):
        for j, (_b, stem) in enumerate(columns):
            ax = axes[i][j]
            p = resolve(folder, stem)
            if p is None:
                missing.append(f"{label}/{stem}")
            draw_tile(ax, p, max_px)
            if j == 0:
                ax.set_ylabel(PRETTY.get(label, label), rotation=0, ha="right",
                              va="center", fontsize=9.0, labelpad=5)

    left = LABEL_W_IN / fig_w
    top = 1.0 - HEADER_IN / fig_h
    fig.subplots_adjust(left=left, right=0.999, top=top, bottom=0.001,
                        wspace=0.02, hspace=0.02)

    # ---- headers: "Method" over the gutter, bin label over each group ----
    y_text = top + 0.055 * (HEADER_IN / fig_h) * 3.0
    fig.text(left / 2.0, y_text, "Method", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold")

    for bname, j0, n in groups:
        p0 = axes[0][j0].get_position()
        p1 = axes[0][j0 + n - 1].get_position()
        xc = (p0.x0 + p1.x1) / 2.0
        fig.text(xc, y_text, bin_label(bname), ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold")
        # thin rule delimiting the group
        fig.add_artist(Line2D([p0.x0, p1.x1], [y_text - 0.004] * 2,
                              transform=fig.transFigure,
                              color="#333333", linewidth=0.6))

    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, outstem + ".pdf"), dpi=dpi,
                bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(OUTDIR, outstem + ".png"), dpi=400,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return nrows, ncols, tile * 25.4, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="parent of gt_binned/, gt/, mask/, masked/, SOTA/")
    ap.add_argument("--per-bin", type=int, default=2,
                    help="scenes sampled per bin (default 2 -> 10 columns)")
    ap.add_argument("--methods-per-fig", type=int, default=12,
                    help="methods per figure (default 12 -> 2 figures)")
    ap.add_argument("--seed", type=int, default=142,
                    help="RNG seed for scene sampling (fixed = reproducible)")
    ap.add_argument("--show-mask", action="store_true",
                    help="add a 'Mask' row above ground truth")
    ap.add_argument("--max-px", type=int, default=256)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    gt_binned = os.path.join(root, "gt_binned")
    if not os.path.isdir(gt_binned):
        sys.exit(f"error: {gt_binned} not found")

    sota = os.path.join(root, "SOTA")
    present = [m for m in METHODS if os.path.isdir(os.path.join(sota, m))]
    absent = [m for m in METHODS if m not in present]
    if absent:
        print(f"[warn] no SOTA/ folder for: {', '.join(absent)}", file=sys.stderr)
    print(f"[info] {len(present)} methods found under SOTA/")

    refs = [("masked", os.path.join(root, "masked"))]
    if args.show_mask:
        refs.append(("mask", os.path.join(root, "mask")))
    refs.append(("gt", os.path.join(root, "gt")))
    meth = [(m, os.path.join(RERUN, CORRECTED[m]) if m in CORRECTED else os.path.join(root, "SOTA", m)) for m in present]
    allf = refs + meth

    bins = sorted([d for d in os.listdir(gt_binned)
                   if os.path.isdir(os.path.join(gt_binned, d))],
                  key=bin_sort_key)
    print(f"[info] bins: {', '.join(bins)}")

    rng = random.Random(args.seed)
    columns, groups = [], []
    for b in bins:
        stems = scene_stems(os.path.join(gt_binned, b))
        ok = [s for s in stems if all(resolve(f, s) for _, f in allf)]
        if len(ok) < args.per_bin:
            print(f"[warn] bin {b}: only {len(ok)} complete scenes "
                  f"(wanted {args.per_bin})", file=sys.stderr)
        pick = rng.sample(ok, min(args.per_bin, len(ok))) if ok else []
        if not pick:
            print(f"[skip] bin {b}: no complete scene", file=sys.stderr)
            continue
        groups.append((b, len(columns), len(pick)))
        columns.extend((b, s) for s in pick)
        print(f"  bin {b}: {len(ok)} complete -> sampled {pick}")

    if not columns:
        sys.exit("error: no usable columns")

    chunks = [meth[i:i + args.methods_per_fig]
              for i in range(0, len(meth), args.methods_per_fig)]
    for k, ch in enumerate(chunks, start=1):
        outstem = f"fig_visual_q{k}"
        nr, nc, mm, miss = render(refs + ch, columns, groups, outstem,
                                  args.max_px, args.dpi)
        note = f"  missing tiles: {len(miss)}" if miss else ""
        print(f"-> {outstem}.pdf  [{nr} rows x {nc} cols, tile {mm:.1f}mm]{note}")


if __name__ == "__main__":
    main()
