import os, torch
from _common import *
def load(device, size):
    enter(os.path.join(FI, "MADIN"))
    from src.network import MADIN
    m = MADIN().to(device)
    ck = torch.load("checkpoints/Places2.pth", map_location=device)
    m.load_state_dict(ck["net_G"] if isinstance(ck, dict) and "net_G" in ck else ck); m.eval()
    keep = square_mask(device, size, hole_one=False); img = torch.rand(1, 3, size, size, device=device)
    return Wrap(lambda x, k: m(x * k, k), m), (img, keep)
