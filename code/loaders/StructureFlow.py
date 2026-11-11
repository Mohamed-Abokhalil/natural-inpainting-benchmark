import os, torch
from _common import *
def load(device, size):
    enter(os.path.join(NII, "StructureFlow"))
    from src.network import StructureGen, FlowGen
    s = StructureGen(input_dim=3, dim=64, n_res=4, activ='relu', norm='in', pad_type='reflect', use_sn=True).to(device)
    f = FlowGen(input_dim=3, dim=64, n_res=2, activ='relu', norm_conv='ln', norm_flow='in', pad_type='reflect', use_sn=False).to(device)
    for net, p in ((s, "latest_net_s_gen.pth"), (f, "latest_net_f_gen.pth")):
        sd = torch.load(os.path.join("checkpoints/places/checkpoints", p), map_location=device)
        sd = sd.get("netG", sd) if isinstance(sd, dict) and not any(k.endswith("weight") for k in sd) else sd
        net.load_state_dict(sd); net.eval()
    def run(inp, smooth, maps):  # MODEL 3 test path: structure_forward then flow_forward (RTV smoothing precomputed, not timed)
        st1 = s(torch.cat((inp, smooth * (1 - maps), maps), 1))
        out, _ = f(torch.cat((inp, st1, maps), 1))
        return out
    x = torch.rand(1, 3, size, size, device=device) * 2 - 1
    return Wrap(run, s, f), (x, x.clone(), square_mask(device, size))
