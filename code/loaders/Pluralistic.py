import os, sys, torch
from _common import *
def load(device, size):
    enter(os.path.join(NII, "Pluralistic"))
    from model import network
    E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=5, norm='none', activation='LeakyReLU', init_type='orthogonal', gpu_ids=[0])
    G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=4, init_type='orthogonal', gpu_ids=[0])
    for net, f in ((E, "latest_net_E.pth"), (G, "latest_net_G.pth")):
        sd = torch.load(os.path.join("checkpoints/place2_random", f), map_location=device)
        net.load_state_dict(sd if hasattr(net, "module") else {k.replace("module.", "", 1): v for k, v in sd.items()})
    E.eval(); G.eval()
    keep = square_mask(device, size, hole_one=False); img = torch.rand(1, 3, size, size, device=device) * 2 - 1
    import torch.nn.functional as F
    def run(img, keep):  # mirrors PluralisticModel.test() for one sample
        im = keep * img
        dist, f = E(im)
        z = torch.distributions.Normal(dist[-1][0], dist[-1][1]).sample()
        sm = F.interpolate(keep, size=[f[2].size(2), f[2].size(3)], mode="nearest")
        g, _ = G(z, f_m=f[-1], f_e=f[2], mask=sm)
        return (1 - keep) * g[-1] + keep * im
    return Wrap(run, E, G), (img, keep)
