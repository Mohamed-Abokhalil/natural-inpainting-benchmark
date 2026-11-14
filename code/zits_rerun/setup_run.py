"""ZITS re-run with the official input convention: FTR receives the COMPLETE test images (it masks them itself) and
wireframes extracted by the official lsm_hawp_inference.py from those complete images (the benchmark run fed the
white-filled masked images, so Canny edges and wireframes traced the hole boundaries). Everything else is identical to the
benchmark config (ckpt/zits_places2/config.yml). The original repository is not modified: this folder holds the config
copy and a copy of the inpainting checkpoint."""
import os, shutil, re
H = os.path.dirname(os.path.abspath(__file__))
Z = os.path.normpath(os.path.join(H, "..", "..", "..", "NII", "ZITS_inpainting"))
GT = os.path.normpath(os.path.join(Z, "..", "test_256_places_1K", "gt"))
run = os.path.join(H, "run"); os.makedirs(run, exist_ok=True)
names = [l.strip().split("/")[-1] for l in open(os.path.join(Z, "data_list", "TEST_FLIST.txt")) if l.strip()]
flist = os.path.join(H, "TEST_FLIST_gt.txt")
open(flist, "w").write("\n".join(os.path.join(GT, n).replace("\\", "/") for n in names) + "\n")
cfg = open(os.path.join(Z, "ckpt", "zits_places2", "config.yml"), encoding="utf-8").read()
fl = flist.replace("\\", "/"); wf = os.path.join(H, "wireframes_gt").replace("\\", "/")
cfg = re.sub(r"(?m)^(TRAIN_FLIST|VAL_FLIST|TEST_FLIST):.*$", lambda m: f"{m.group(1)}: {fl}", cfg)
cfg = re.sub(r'(?m)^eval_line_path:.*$', f'eval_line_path: "{wf}"', cfg)
open(os.path.join(run, "config.yml"), "w", encoding="utf-8").write(cfg)
for f in os.listdir(os.path.join(Z, "ckpt", "zits_places2")):
    if f.endswith(".pth"): shutil.copy2(os.path.join(Z, "ckpt", "zits_places2", f), os.path.join(run, f))
print("images:", len(names)); print([l for l in cfg.splitlines() if "FLIST" in l or "line_path" in l])
