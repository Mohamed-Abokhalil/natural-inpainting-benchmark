import os, sys, torch
from _common import *
def load(device, size):
    d = os.path.join(NII, "StrDiffusion", "test", "texture", "config", "inpainting")
    enter(d)
    import options as option
    from models import create_model
    import str_utils as str_util
    import utils as util
    opt = option.dict_to_nonedict(option.parse(os.path.join(d, "options", "test", "ir-sde.yml"), is_train=False))
    model = create_model(opt)
    T = opt["sde"]["T"]
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=T, schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device); sde.set_model(model.model)
    S = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=T, schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device); S.set_model(model.models)
    cnt = {"texture": 0, "structure": 0, "dis": 0}
    for k, net in (("texture", model.model), ("structure", model.models), ("dis", model.dis)):
        net.register_forward_hook(lambda *a, k=k: cnt.__setitem__(k, cnt[k] + 1))
    def run(Y, Xg, Xe, keep):  # test.py loop body for one image (gray/Canny preprocessing on CPU not timed)
        ns = sde.noise_state(Y * keep); nss = S.noise_state(Xe * keep)
        model.feed_data(ns, Y * keep, Y, keep, S, Xg, Xe * keep)
        model.test(sde, save_states=False, GT=Y, mask=keep, S_sde=S, S_GT=Xg, S_LQ=nss, dis=model.dis)
        model.model.eval(); model.models.eval()
        return model.output
    Y = torch.rand(1, 3, 256, 256, device=device); Xg = Y.mean(1, keepdim=True); Xe = (torch.rand_like(Xg) > 0.9).float()
    keep = square_mask(device, 256, hole_one=False)
    w = Wrap(run, model.model, model.models, model.dis)
    t = torch.tensor([10], device=device)
    w.cnt = cnt; w.T = T
    return w, (Y, Xg, Xe, keep)
