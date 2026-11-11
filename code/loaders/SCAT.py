import os, sys, torch, importlib, argparse
from _common import *
def load(device, size):
    enter(os.path.join(FI, "Generative-Image-Inpainting", "src"))
    a = argparse.Namespace(block_num=8, rates=[1, 2, 4, 8], model="model")
    net = importlib.import_module("model.model")
    m = net.InpaintGenerator(a).to(device).eval()
    m.load_state_dict(torch.load("checkpoints/Places2_G1000000.pt", map_location=device))
    mask = square_mask(device, size); img = torch.rand(1, 3, size, size, device=device) * 2 - 1
    return Wrap(lambda x, k: m(x * (1 - k) + k, k), m), (img, mask)
