import os, torch, importlib
import torch.nn.functional as F
from _common import *
def load(device, size):
    enter(os.path.join(NII, "DGTS-Inpainting", "code", "test"))
    tst = importlib.import_module("test")              # repo test.py (defines Options)
    from transformer_64 import Generator
    from models.networks.generator import UpsamplerGenerator
    g = torch.nn.DataParallel(Generator().to(device)); g.load_state_dict(torch.load("checkpoints/places.pth", map_location=device)); g.eval()
    up = UpsamplerGenerator(tst.Options()); up.load_state_dict(torch.load("checkpoints/latest_net_G.pth", map_location=device)); up = up.to(device).eval()
    def run(real, keep):  # Trainer.train(): coarse 64x64 transformer, then upsampler (JPEG round-trips omitted)
        fakes = g(real, keep)
        r1 = F.interpolate(real, scale_factor=0.25); k1 = F.interpolate(keep, scale_factor=0.25)
        f3 = fakes * (1 - k1) + k1 * r1
        s = F.interpolate(f3, size=(32, 32), mode="bicubic", align_corners=False)
        mi = torch.cat([real * keep + (1 - keep), 1 - keep], 1)
        _, out = up([mi, s]); return out
    real = torch.rand(1, 3, size, size, device=device) * 2 - 1; keep = square_mask(device, size, hole_one=False)
    return Wrap(run, g, up), (real, keep)
