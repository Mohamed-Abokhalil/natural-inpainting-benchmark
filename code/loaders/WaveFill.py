import os, sys, torch
from _common import *
def load(device, size):
    enter(os.path.join(FI, "WaveFill")); sys.path.insert(0, os.path.join(FI, "WaveFill", "pytorch_wavelets"))
    saved = sys.argv; sys.argv = ["infer.py", "--name", "places2_wavefill", "--checkpoints_dir", "./checkpoints", "--gpu_ids", "0", "--netG", "WaveletInpaintLv2GCFRAN", "--input_nc", "4", "--output_nc", "3", "--highfreq_nc", "12", "--wavelet_decomp_level", "2", "--ngf", "64", "--use_attention", "--pconv_level", "0", "--mask_type", "1"]
    from options.test_options import TestOptions
    opt = TestOptions().parse(); sys.argv = saved
    from models.wavelet_model import WaveletModel
    m = WaveletModel(opt).eval()
    img = torch.rand(1, 3, size, size, device=device) * 2 - 1; k = square_mask(device, size)
    def run(img, k):
        mi = torch.cat([img * (1 - k) + k, k], 1)
        return m({"image": img, "mask": k, "masked_img": mi, "img_name": ["x"]}, mode="inference")
    return Wrap(run, m), (img, k)
