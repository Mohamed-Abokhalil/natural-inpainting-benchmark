import os, torch
from _common import *
def load(device, size):
    # simple-lama-inpainting 0.1.2: TorchScript export of the official big-lama (Places-Challenge) checkpoint,
    # the exact file used for Table 8 (torch hub cache big-lama.pt)
    p = os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt")
    m = torch.jit.load(p, map_location=device).eval()
    img = torch.rand(1, 3, size, size, device=device); k = square_mask(device, size)
    return Wrap(lambda x, mk: m(x, mk), m), (img, k)
