import os, sys, torch
from _common import *
def load(device, size):
    enter(os.path.join(NII, "RePaint"))
    import conf_mgt
    from utils import yamlread
    from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, select_args
    conf = conf_mgt.conf_base.Default_Conf(); conf.update(yamlread("confs/face_example.yml")); conf["show_progress"] = False
    model, diffusion = create_model_and_diffusion(**select_args(conf, model_and_diffusion_defaults().keys()), conf=conf)
    model.load_state_dict(torch.load(conf.model_path, map_location="cpu")); model.to(device).eval()
    cnt = [0]
    def mf(x, t, y=None, gt=None, **kw):
        cnt[0] += 1; return model(x, t, None, gt=gt)
    def run(gt, keep):  # complete RePaint sampling (T=250, jump length 10, 10 resamplings)
        kw = {"gt": gt, "gt_keep_mask": keep, "y": torch.zeros(1, dtype=torch.long, device=device)}
        return diffusion.p_sample_loop(mf, (1, 3, 256, 256), clip_denoised=True, model_kwargs=kw, device=device,
                                       progress=False, return_all=True, conf=conf)["sample"]
    gt = torch.rand(1, 3, 256, 256, device=device) * 2 - 1; keep = square_mask(device, 256, hole_one=False).repeat(1, 3, 1, 1)
    w = Wrap(run, model)
    t = torch.tensor([10], device=device)
    w.macs_model = Wrap(lambda x, tt: model(x, tt), model); w.macs_inputs = (gt, t)
    from guided_diffusion.scheduler import get_schedule_jump
    sched = get_schedule_jump(**conf.schedule_jump_params)
    w.nfe = sum(1 for a, b in zip(sched[:-1], sched[1:]) if b < a)   # denoising steps (t -> t-1) in the jump schedule
    print(f"  [RePaint] schedule length {len(sched)}, denoising NFE {w.nfe}")
    return w, (gt, keep)
