"""Recover which QD-IMD mask was paired with each of the 1,000 benchmark images.
Original pairing script used unseeded random.choice; this matches masks by content."""
import os, sys, hashlib, json, csv
import numpy as np, cv2
from PIL import Image
QD = os.environ.get("QDIMD_DIR", "D:/Face Unmasking/qd_imd")   # folder of the QD-IMD release
BM = os.path.join(os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)"), r"data/places2/mask")
OUT = sys.argv[1]
def h(a): return hashlib.md5(np.packbits(a).tobytes()).hexdigest()
targets = {}
for f in sorted(os.listdir(BM)):
    m = np.array(Image.open(os.path.join(BM, f)).convert("L")) > 127
    targets.setdefault(h(m), []).append(f)
found = {}
methods = {"nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}
for split in ["test", "train"]:
    files = sorted(os.listdir(os.path.join(QD, split)))
    for i, f in enumerate(files):
        q = np.array(Image.open(os.path.join(QD, split, f)).convert("L")).astype(np.uint8)
        q = (q > 0).astype(np.uint8) * 255
        for mn, interp in methods.items():
            r = cv2.resize(q, (256, 256), interpolation=interp) > 127
            for pol, a in (("as_is", r), ("inverted", ~r)):
                k = h(a)
                if k in targets:
                    for t in targets[k]:
                        found.setdefault(t, []).append(f"{split}/{f}|{mn}|{pol}")
        if i % 5000 == 0: print(split, i, "matched", len(found), flush=True)
with open(OUT, "w") as fh: json.dump(found, fh, indent=0)
print("TOTAL matched", len(found), "of", sum(len(v) for v in targets.values()))
