import os, sys, types, torch, numpy as np
from _common import *
def load(device, size):
    enter(os.path.join(NII, "ZITS_inpainting"))
    from src.config import Config
    from src.models.FTR_model import DefaultInpaintingTrainingModule
    from src.utils import SampleEdgeLineLogits
    from datasets.dataset_FTR import DynamicDataset
    cfg = Config("ckpt/zits_places2/config.yml")      # PATH = ckpt/zits_places2 (InpaintingModel_best_gen.pth)
    m = DefaultInpaintingTrainingModule(cfg, gpu=0, rank=0, test=True).to(device).eval()
    # masked positional encoding exactly as the dataset computes it
    ds = DynamicDataset.__new__(DynamicDataset); ds.str_size = 256; ds.pos_num = 128
    ds.ones_filter = np.ones((3, 3), dtype=np.float32)
    ds.d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32); ds.d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    ds.d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32); ds.d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    hole = square_mask(device, size)
    rel, _, direct = ds.load_masked_position_encoding(hole[0, 0].cpu().numpy().astype(np.uint8) * 255)
    rel = torch.LongTensor(rel)[None].to(device); direct = torch.LongTensor(direct)[None].to(device)
    def run(img, edge, line, hole):  # FTR_trainer.test(): TSR sampling (5 iter, fp16) -> upsample -> FTR
        e, l = SampleEdgeLineLogits(m.transformer, context=[img, edge, line], mask=hole.clone(), iterations=5, add_v=0.05, mul_v=4)
        e, l = e.float(), l.float()
        if not cfg.fix_256:
            e = torch.sigmoid((m.structure_upsample(e)[0] + 2) * 2); l = torch.sigmoid((m.structure_upsample(l)[0] + 2) * 2)
        b = {"image": img, "mask": hole, "edge": e, "line": l, "rel_pos": rel, "direct": direct}
        return m(b)["predicted_image"]
    img = torch.rand(1, 3, size, size, device=device)
    z = torch.zeros(1, 1, size, size, device=device)
    return Wrap(run, m.transformer, m.structure_upsample, m.str_encoder, m.generator), (img, z, z.clone(), hole)
