import os, torch
from _common import *
def load(device, size):
    enter(os.path.join(FI, "spa-former"))
    from model import netU_spa_former
    g = netU_spa_former.define_g(gpu_ids=[])
    sd = torch.load("checkpoints/place2/latest_net_G.pth", map_location=device)
    g.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}); g = g.to(device).eval()
    keep = square_mask(device, size, hole_one=False); img = torch.rand(1, 3, size, size, device=device) * 2 - 1
    return Wrap(lambda x, k: g(x * k, k), g), (img, keep)
