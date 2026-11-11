import os, sys, torch
from _common import *
def load(device, size):
    enter(os.path.join(NII, "Rethinking-Inpainting-MEDFE"))
    saved = sys.argv; sys.argv = ["test.py", "--checkpoints_dir", "./checkpoints/place2", "--gpu_ids", "0"]
    from options.test_options import TestOptions
    from models.models import create_model
    opt = TestOptions().parse(); sys.argv = saved
    m = create_model(opt)
    for net, f in ((m.netEN, "EN.pkl"), (m.netDE, "DE.pkl"), (m.netMEDFE, "MEDEF.pkl")):
        net.to(device); sd = torch.load(os.path.join("checkpoints/place2", f), map_location="cpu")
        (net.module if isinstance(net, torch.nn.DataParallel) else net).load_state_dict(sd); net.eval()
    def run(img, st, hole):  # MEDFE.set_input + MEDFE.forward (test path)
        m.set_input(img.clone(), st.clone(), hole)
        m.forward(); return m.fake_out
    img = torch.rand(1, 3, size, size, device=device) * 2 - 1
    hole = square_mask(device, size)
    return Wrap(run, m.netEN, m.netDE, m.netMEDFE), (img, img.clone(), hole)
