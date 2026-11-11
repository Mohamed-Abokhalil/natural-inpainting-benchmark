import os, torch
from _common import *
def load(device, size):
    enter(os.path.join(NII, "CTSDG"))
    from models.generator.generator import Generator
    g = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3).to(device)
    g.load_state_dict(torch.load("Checkpoints/places2.pt", map_location=device)["generator"]); g.eval()
    img = torch.rand(1, 3, size, size, device=device); keep = square_mask(device, size, hole_one=False)
    eg = torch.rand(1, 2, size, size, device=device)
    return Wrap(lambda x, e, k: g(x * k, e * k, k), g), (img, eg, keep)
