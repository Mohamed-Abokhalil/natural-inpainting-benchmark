"""DGTS re-run with the official input convention: the dataset loader receives the COMPLETE test images (the network and
the upsampler mask them themselves, test.py lines `mask = 1 - mask_in`, `masked_img = img_r * mask + (1 - mask)`).
The benchmark run had pointed --dataset_dir at the white-filled masked images, so the JPEG round trip and the min/max
normalisation of make_grid saw the white holes. Everything else is the benchmark code: this script copies the locally
modified official code/test folder (paths only) into run/ and points its hard-coded paths at absolute locations.
The original repository is not modified. Run:  cd run && python main.py --dataset_dir <gt> --num_work 0"""
import os, shutil
H = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(H, "..", "..", "..", "NII", "DGTS-Inpainting", "code", "test"))
GT = os.path.normpath(os.path.join(REPO, "..", "..", "..", "test_256_places_1K", "gt"))
MASK = os.path.normpath(os.path.join(REPO, "..", "..", "..", "test_256_places_1K", "mask"))
run = os.path.join(H, "run"); out = os.path.join(H, "out")
if os.path.exists(run): shutil.rmtree(run)
shutil.copytree(REPO, run, ignore=shutil.ignore_patterns("checkpoints", "__pycache__"))
os.makedirs(os.path.join(out, "DGTS"), exist_ok=True)
p = os.path.join(run, "test.py"); s = open(p, encoding="utf-8").read()
fw = lambda x: x.replace("\\", "/")
R = [("log_base_dir = '../../DGTS'", f"log_base_dir = r'{fw(out)}'"),
     ('upsample_path ="./checkpoints/latest_net_G.pth"', f'upsample_path = r"{fw(os.path.join(REPO, "checkpoints", "latest_net_G.pth"))}"'),
     ("dgts_path = './checkpoints/places.pth'", f"dgts_path = r'{fw(os.path.join(REPO, 'checkpoints', 'places.pth'))}'"),
     ("osp.join('../../../test_256_places_1K/mask',", f"osp.join(r'{fw(MASK)}',")]
for a, b in R:
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("run dir ready; images:", GT, len(os.listdir(GT)))
