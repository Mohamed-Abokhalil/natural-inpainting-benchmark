import os, torch
from _common import *
def load(device, size):
    enter(os.path.join(FI, "CMT"))
    from network.network_pro import Inpaint
    from utils import load_checkpoint
    m = load_checkpoint("ckpt/Place2.pth", Inpaint()).eval().to(device)
    img = torch.rand(1, 3, size, size, device=device) * 2 - 1
    return Wrap(lambda a, b: m(a, b), m), (img, square_mask(device, size))
