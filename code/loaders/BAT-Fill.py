import os, sys, torch, importlib
import torch.nn.functional as F
from _common import *
def load(device, size):
    enter(os.path.join(NII, "BAT-Fill"))
    saved = sys.argv; sys.argv = ["bat_sample.py"]
    bs = importlib.import_module("bat_sample"); sys.argv = saved      # reuse official functions (masked_sample, quantize)
    from models.bat_model import GPT, GPTConfig
    from models.networks.generator import UpsamplerGenerator
    gpt = GPT(GPTConfig(512, 1024 * 2, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=24, n_head=8, n_embd=512))
    gpt.load_state_dict(torch.load("checkpoints/places2_bat_pretrain/latest_tran.pth", map_location="cpu")); gpt = gpt.to(device).eval()
    class O: netG='Upsampler'; ngf=64; norm_G='spectralspadeposition3x3'; resnet_n_blocks=6; use_attention=True; input_nc=4; gpu_ids=[0]; semantic_nc=4
    up = UpsamplerGenerator(O()); up.load_state_dict(torch.load("checkpoints/places2_bat_pretrain/latest_net_G.pth", map_location="cpu")); up = up.to(device).eval()
    hole = square_mask(device, size); img = torch.rand(1, 3, size, size, device=device) * 2 - 1
    img32 = F.interpolate(img, size=32, mode="bicubic", align_corners=False).cpu(); m32 = F.interpolate(hole, size=32, mode="nearest")
    q = bs.color_quantize(img32[0]).unsqueeze(0).to(device); keep32 = (1 - m32).view(1, -1)
    def run(q, keep32, img, hole):  # bat_sample.__main__: AR sampling of masked tokens (top-50) + upsampler
        g = bs.masked_sample(gpt, q.clone(), keep32, sample=True, top_k=50)
        s = torch.from_numpy(bs.dequantize(g[0].cpu())).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
        mi = torch.cat([img * (1 - hole) + hole, hole], 1)
        _, out = up([mi, s]); return out
    w = Wrap(run, gpt, up)
    pred = q[:, keep32[0].bool()][:, :-1]
    w.macs_model = Wrap(lambda a, b, c: gpt(a, b, c)[0], gpt); w.macs_inputs = (q, pred, keep32)
    w.nfe = int((1 - keep32).sum().item()) + 1      # one transformer pass per masked token + one upsampler pass
    return w, (q, keep32, img, hole)
