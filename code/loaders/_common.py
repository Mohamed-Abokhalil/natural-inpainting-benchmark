import os, sys, torch
ROOT = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
FI = os.path.join(ROOT, "FI", "000_Related work"); NII = os.path.join(ROOT, "NII")
def enter(repo, sub=""):
    p = os.path.join(repo, sub) if sub else repo
    if p not in sys.path: sys.path.insert(0, p)
    os.chdir(p)
    return p
def square_mask(device, size, hole_one=True):
    m = torch.zeros(1, 1, size, size, device=device)
    m[..., size // 4:3 * size // 4, size // 4:3 * size // 4] = 1.0
    return m if hole_one else 1 - m
class Wrap(torch.nn.Module):
    """Runs the complete inference forward of one or more networks; returns a tensor."""
    def __init__(self, fn, *mods):
        super().__init__(); self.mods = torch.nn.ModuleList(mods); self.fn = fn
    def forward(self, *x):
        out = self.fn(*x)
        while isinstance(out, (list, tuple)): out = out[-1]
        return out
