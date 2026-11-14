"""Fig. 6 (enlarged crops) and helpers shared with Fig. 7.
Scenes: the first scene drawn with seed 142 (same draw as the qualitative figures) from the
20-30%, 30-40% and 40-50% bins. Crop: the 85x85 window with the largest number of hole pixels
(ties -> top-left), enlarged 3x (nearest-neighbour, so pixels are not smoothed)."""
import os, sys
import numpy as np, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})

DATA = "D:/Face Unmasking/Inpainting_Framwork/comparison/From-Pixel-to-Face-(survey)/data/places2"
OUT = "D:/Face Unmasking/Inpainting_Framwork/comparison/From-Pixel-to-Face-(survey)/revision/results/figures"
CROP = 85


def img(folder, f): return np.array(Image.open(os.path.join(folder, f)).convert("RGB"))
def mask(f): return np.array(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127


def best_window(m, c=CROP):
    ii = cv2.integral(m.astype(np.uint8))
    H, W = m.shape; best, pos = -1, (0, 0)
    for y in range(0, H - c + 1, 2):
        for x in range(0, W - c + 1, 2):
            s = ii[y + c, x + c] - ii[y, x + c] - ii[y + c, x] + ii[y, x]
            if s > best: best, pos = s, (y, x)
    return pos


def outline(im, m, color=(255, 0, 0)):
    im = im.copy()
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnts, -1, color, 1)
    return im


def crop3(im, m, pos, c=CROP):
    y, x = pos
    a = im[y:y + c, x:x + c]; mm = m[y:y + c, x:x + c]
    a = cv2.resize(a, (3 * c, 3 * c), interpolation=cv2.INTER_NEAREST)
    mm = cv2.resize(mm.astype(np.uint8), (3 * c, 3 * c), interpolation=cv2.INTER_NEAREST)
    return outline(a, mm)


def masked_input(f):
    g = img(os.path.join(DATA, "masked"), f); m = mask(f)
    return outline(g, m), m


def main():
    scenes = [("20–30%", "269126.png"), ("30–40%", "203573.png"), ("40–50%", "166396.png")]
    methods = [("MADF", "MADF", "G1"), ("SCAT", "SCAT", "G2"), ("ZITS", "ZITS", "G3"),
               ("MADIN", "MADIN", "G4a"), ("LaMa", "LaMa", "G4b"), ("RePaint", "RePaint", "G5")]
    cols = ["Masked\ninput", "GT\n(3× crop)"] + [f"{n}\n({g})" for n, _, g in methods]
    fig, axes = plt.subplots(len(scenes), len(cols), figsize=(7.09, 7.09 / len(cols) * len(scenes) + 0.45))
    for r, (bl, f) in enumerate(scenes):
        mi, m = masked_input(f); pos = best_window(m)
        tiles = [mi, crop3(img(os.path.join(DATA, "gt"), f), m, pos)]
        tiles += [crop3(img(os.path.join(os.path.dirname(OUT), "rerun", d) if d in ("ZITS", "CTSDG", "DGTS", "VHII", "spa-former", "Pluralistic") else os.path.join(DATA, "SOTA", d), f), m, pos) for _, d, _ in methods]
        for c, t in enumerate(tiles):
            ax = axes[r][c]; ax.imshow(t, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if c == 0:
                ax.add_patch(Rectangle((pos[1], pos[0]), CROP, CROP, fill=False, ec="yellow", lw=1.2))
                ax.set_ylabel(f"{bl}\n{os.path.splitext(f)[0]}", fontsize=8.5, rotation=0, ha="right", va="center", labelpad=4)
            if r == 0: ax.set_title(cols[c], fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.1, right=0.999, top=0.90, bottom=0.005, wspace=0.03, hspace=0.04)
    os.makedirs(OUT, exist_ok=True)
    for e in ("pdf", "png"): fig.savefig(os.path.join(OUT, "fig_crops." + e), dpi=400, bbox_inches="tight", pad_inches=0.01)
    print({f: best_window(mask(f)) for _, f in scenes})


if __name__ == "__main__":
    main()
