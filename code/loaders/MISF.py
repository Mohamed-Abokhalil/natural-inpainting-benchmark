import os, torch, argparse
from _common import *
def load(device, size):
    enter(os.path.join(NII, "misf"))
    from src.networks import InpaintGenerator
    cfg = argparse.Namespace(FILTER_TYPE=None, kernel_size=None)  # repo Config returns None for absent keys
    g = InpaintGenerator(config=cfg).to(device)
    g.load_state_dict(torch.load("checkpoints/MISF/places2_InpaintingModel_gen.pth", map_location=device)["generator"]); g.eval()
    img = torch.rand(1, 3, size, size, device=device); m = square_mask(device, size)
    return Wrap(lambda x, k: g(torch.cat((x * (1 - k), k), 1)), g), (img, m)
