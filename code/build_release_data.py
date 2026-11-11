"""Release data files: image list, Places365 source names, recovered QD-IMD pairing, bins, coverage,
seeds, subsets. Copies (never moves) the 1,000 benchmark masks."""
import os, json, csv, shutil
import numpy as np
from PIL import Image
R = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
DATA = os.path.join(R, "data", "places2"); REL = os.path.join(R, "revision", "release")
os.makedirs(os.path.join(REL, "data", "masks"), exist_ok=True)
pair = json.load(open(os.path.join(R, "revision", "results", "pairing_recovered.json")))
binmap = {f: b for b in ["0_10", "10_20", "20_30", "30_40", "40_50"] for f in os.listdir(os.path.join(DATA, "gt_binned", b))}
rows = []
for f in sorted(os.listdir(os.path.join(DATA, "gt"))):
    stem = os.path.splitext(f)[0]
    m = np.array(Image.open(os.path.join(DATA, "mask", f)).convert("L")) > 127
    qd = [x.split("|")[0] for x in pair[f] if x.startswith("test/") and "|nearest|" in x]
    assert len(qd) == 1, (f, pair[f])
    rows.append({"image": f, "places365_test_file": f"Places365_test_{int(stem):08d}.jpg", "qdimd_test_mask": qd[0].split("/")[1],
                 "hole_ratio": f"{m.mean():.6f}", "bin": binmap[f]})
    shutil.copy2(os.path.join(DATA, "mask", f), os.path.join(REL, "data", "masks", f))
with open(os.path.join(REL, "data", "benchmark_pairs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for sub in ("diversity100.txt", "diversity100.json", "diffusion10.txt"):
    shutil.copy2(os.path.join(R, "revision", "results", "subsets", sub), os.path.join(REL, "data", sub))
json.dump({"image_sampling": "random.seed(42); random.sample(os.listdir(<Places365 test_256 folder renamed 00001.png..328500.png>), 1000) -- verified to reproduce the 1,000 images exactly (NII/test_256_places_1K/dataset_subset.py)",
           "mask_pairing": "unseeded random.choice over QD-IMD test masks; realised pairing in benchmark_pairs.csv (recovered by exact matching)",
           "qualitative_scene_seed": 142, "diversity_subset_seed": 2026,
           "stochastic_reruns": "torch.manual_seed(seed*100000 + index of image in sorted list); seeds 0 (10 samples, subsets) and 1-5 (full set)",
           "bootstrap_seed": 0, "kid_subset_seed": 0},
          open(os.path.join(REL, "data", "seeds.json"), "w"), indent=1)
print(len(rows), "pairs written")
