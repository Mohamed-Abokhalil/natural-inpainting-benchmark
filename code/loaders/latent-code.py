import os, sys, torch, importlib
import torch.nn.functional as F
from _common import *
def load(device, size):
    enter(os.path.join(FI, "latent-code-inpainting"))
    saved = sys.argv; sys.argv = ["infer_latent.py"]
    il = importlib.import_module("infer_latent"); sys.argv = saved   # reuse the script's own functions
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/places_inpainting.yaml"); cfg["data"]["params"]["batch_size"] = 1
    model = il.instantiate_from_config(cfg.model).to(device).eval()
    VQ, Enc, Tr, Unet = model.helper_model
    VQ, Enc, Tr = VQ.to(device), Enc.to(device), Tr.to(device); Unet = Unet.to(device) if Unet is not None else None
    def run(x, keep):  # infer_latent.main() for one image, clamp_ratio 0.25
        quant_z, _, info, mask_out = Enc.encode(x * keep, keep, clamp_ratio=0.25)
        mask_out = mask_out.reshape(x.shape[0], -1); z = info[2].reshape(x.shape[0], -1)
        nb = {"image": (x * keep).permute(0, 2, 3, 1)}
        zc, _, _ = il.forward_to_indices(Tr, nb, z, mask_out, device)
        B, C, H, W = quant_z.shape
        q = VQ.quantize.get_codebook_entry(zc.reshape(-1).int(), shape=(B, H, W, C))
        dec, *_ = model.current_model(nb, quant=q, mask_in=keep, mask_out=mask_out.reshape(B, 1, H, W), return_fstg=False, debug=True)
        rec = x * keep + dec * (1 - keep)
        return Unet.refine(rec, keep) if Unet is not None else rec
    x = torch.rand(1, 3, size, size, device=device) * 2 - 1; keep = square_mask(device, size, hole_one=False)
    mods = [m for m in (VQ, Enc, Tr, Unet, model.current_model) if isinstance(m, torch.nn.Module)]
    w = Wrap(run, *mods)
    # count transformer network evaluations (iterative token sampling) with a hook on its inner GPT
    inner = getattr(Tr, "transformer", Tr); cnt = [0]
    h = inner.register_forward_hook(lambda *a: cnt.__setitem__(0, cnt[0] + 1))
    with torch.no_grad(): run(x, keep)
    h.remove(); w.nfe = cnt[0] + 1   # transformer evaluations + one decoder/refinement pass
    print(f"  [latent-code] transformer evaluations per image: {cnt[0]}")
    return w, (x, keep)
