import os, torch
from _common import *
def load(device, size):
    enter(os.path.join(NII, "edge-connect"))
    from src.networks import EdgeGenerator, InpaintGenerator
    e = EdgeGenerator(use_spectral_norm=True).to(device); i = InpaintGenerator().to(device)
    e.load_state_dict(torch.load("checkpoints/places2/EdgeModel_gen.pth", map_location=device)["generator"])
    i.load_state_dict(torch.load("checkpoints/places2/InpaintingModel_gen.pth", map_location=device)["generator"])
    e.eval(); i.eval()
    def run(img, gray, edges, m):  # MODEL=4 (edge-inpaint) path of edge_connect.test(); Canny on CPU not timed
        ed = e(torch.cat((gray * (1 - m) + m, edges * (1 - m), m), 1))
        ed = ed * m + edges * (1 - m)
        return i(torch.cat((img * (1 - m) + m, ed), 1))
    s = size
    return Wrap(run, e, i), (torch.rand(1, 3, s, s, device=device), torch.rand(1, 1, s, s, device=device),
                             (torch.rand(1, 1, s, s, device=device) > 0.9).float(), square_mask(device, s))
