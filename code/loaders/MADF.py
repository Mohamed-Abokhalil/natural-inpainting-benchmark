import os, torch, argparse
from _common import *
def load(device, size):
    enter(os.path.join(FI, "Pytorch-MADF"))
    from net import MADFNet
    from util.io import load_ckpt
    a = argparse.Namespace(image_size=size, n_refinement_D=2, gpu=0)
    m = MADFNet(layer_size=7, args=a).to(device)
    load_ckpt("checkpoints/places2.pth", [("model", m)]); m.eval()
    keep = square_mask(device, size, hole_one=False).expand(-1, 3, -1, -1)
    img = torch.randn(1, 3, size, size, device=device)
    return Wrap(lambda x, k: m(x * k, k), m), (img, keep)
