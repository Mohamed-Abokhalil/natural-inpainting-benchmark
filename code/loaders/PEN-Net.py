import os, torch, importlib
from _common import *
def load(device, size):
    enter(os.path.join(FI, "PEN-Net-for-Inpainting"))
    g = importlib.import_module("model.pennet").InpaintGenerator().to(device)
    g.load_state_dict(torch.load("release_model/pennet_places2_square256/gen_00000.pth", map_location=device)["netG"]); g.eval()
    img = torch.rand(1, 3, size, size, device=device) * 2 - 1; m = square_mask(device, size)
    return Wrap(lambda x, k: g(torch.cat([x * (1 - k) + k, k], 1), k), g), (img, m)
