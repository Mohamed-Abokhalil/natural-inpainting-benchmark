import os, sys, torch
from _common import *
def load(device, size):
    # zengxianyu/co-mod-gan-pytorch (PyTorch port, converted official Places 512 weights); network is fixed at 512,
    # so it is measured at its inference resolution 512 (Table 8 run: 256 -> 512 input, Lanczos 512 -> 256 output)
    enter(os.path.join(FI, "co-mod-gan-pytorch"))
    import importlib.util as iu
    sp = iu.spec_from_file_location("models.networks.op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shims", "stylegan2_op_native.py"))
    shim = iu.module_from_spec(sp); sp.loader.exec_module(shim); sys.modules["models.networks.op"] = shim
    saved = sys.argv
    sys.argv = ["test.py", "--mixing", "0", "--batchSize", "1", "--nThreads", "0", "--name", "comod-places-512",
                "--dataset_mode", "testimage", "--image_dir", ".", "--mask_dir", ".", "--output_dir", ".",
                "--load_size", "512", "--crop_size", "512", "--z_dim", "512", "--model", "comod", "--netG", "comodgan",
                "--which_epoch", "co-mod-gan-places2-050000"]
    from options.test_options import TestOptions
    import models
    opt = TestOptions().parse(); sys.argv = saved
    m = models.create_model(opt); m.eval()
    s = 512
    img = torch.rand(1, 3, s, s, device=device) * 2 - 1; k = square_mask(device, s)
    return Wrap(lambda x, mk: m({"image": x, "mask": mk}, mode="inference")[0], m), (img, k)
