import os, torch, importlib
from _common import *
def load(device, size):
    enter(os.path.join(FI, "variable-hyperparameter-image-impainting"))
    g = importlib.import_module("model.VHII_efficient").InpaintGenerator().to(device)
    g.load_state_dict(torch.load("trained_models/places/places_VHII_efficient/best_model_places.pth", map_location=device)["netG"]); g.eval()
    img = torch.rand(1, 3, size, size, device=device) * 2 - 1; m = square_mask(device, size)
    return Wrap(lambda x, k: g(x * (1 - k), k), g), (img, m)
