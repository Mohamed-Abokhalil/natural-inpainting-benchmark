"""Resumable stochastic sampling (Table 10) for CoModGAN, Latent-code, BAT-Fill and StrDiff, replicating
the inference pipelines that produced the Table 8 outputs (same inputs, preprocessing, checkpoints,
resolution and output handling). Only the random seed is controlled:
torch.manual_seed(seed*100000 + index_in_sorted_list) (+ numpy/python seeds) before each image.

Usage: python stoch_run.py --method M --list FILE|all --seeds 0 1 .. --nsamp N --out DIR
Writes DIR/<stem>_s<seed>_k<j>.png
"""
import argparse, os, sys, random, importlib
import numpy as np, torch, cv2
import torch.nn.functional as F
from PIL import Image
ROOT = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
FI = os.path.join(ROOT, "FI", "000_Related work"); NII = os.path.join(ROOT, "NII")
DATA = os.path.join(ROOT, "data", "places2"); T1K = os.path.join(NII, "test_256_places_1K")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = "cuda"


def enter(p):
    sys.path.insert(0, p); os.chdir(p)


def build(method):
    if method == "CoModGAN":
        enter(os.path.join(FI, "co-mod-gan-pytorch"))
        import importlib.util as iu
        sp = iu.spec_from_file_location("models.networks.op", os.path.join(W, "shims", "stylegan2_op_native.py"))
        shim = iu.module_from_spec(sp); sp.loader.exec_module(shim); sys.modules["models.networks.op"] = shim
        sys.argv = ["test.py", "--mixing", "0", "--batchSize", "1", "--nThreads", "0", "--name", "comod-places-512",
                    "--dataset_mode", "testimage", "--image_dir", ".", "--mask_dir", ".", "--output_dir", ".",
                    "--load_size", "512", "--crop_size", "512", "--z_dim", "512", "--model", "comod", "--netG", "comodgan",
                    "--which_epoch", "co-mod-gan-places2-050000"]
        from options.test_options import TestOptions
        import models
        m = models.create_model(TestOptions().parse()); m.eval()
        import torchvision.transforms as T
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
        def run(f):  # test.py: 512 inputs (512_masked / 512_mask), output Lanczos-resized to 256, no compositing
            img = Image.open(os.path.join(T1K, "512_masked", f)).convert("RGB")
            mk = Image.open(os.path.join(T1K, "512_mask", f)).convert("L").resize(img.size)
            x = tf(img)[None]; k = (T.ToTensor()(mk) > 0).float()[None]
            g, _ = m({"image": x, "mask": k}, mode="inference")
            g = ((torch.clamp(g, -1, 1) + 1) / 2 * 255).cpu().numpy().astype(np.uint8)[0].transpose(1, 2, 0)
            return cv2.resize(g[:, :, ::-1], (256, 256), interpolation=cv2.INTER_LANCZOS4)[:, :, ::-1]
        return run
    if method == "BAT-Fill":
        enter(os.path.join(NII, "BAT-Fill")); sys.argv = ["bat_sample.py"]
        bs = importlib.import_module("bat_sample")
        from models.bat_model import GPT, GPTConfig
        from models.networks.generator import UpsamplerGenerator
        gpt = GPT(GPTConfig(512, 2048, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0, n_layer=24, n_head=8, n_embd=512))
        gpt.load_state_dict(torch.load("checkpoints/places2_bat_pretrain/latest_tran.pth", map_location="cpu")); gpt = gpt.to(dev).eval()
        class O: netG='Upsampler'; ngf=64; norm_G='spectralspadeposition3x3'; resnet_n_blocks=6; use_attention=True; input_nc=4; gpu_ids=[0]; semantic_nc=4
        up = UpsamplerGenerator(O()); up.load_state_dict(torch.load("checkpoints/places2_bat_pretrain/latest_net_G.pth", map_location="cpu")); up = up.to(dev).eval()
        def run(f):  # bat_sample.__main__ with num_sample=1 (input: GT image, benchmark mask)
            p = os.path.join(DATA, "gt", f); md = os.path.join(DATA, "mask")
            img, mask, mimg = bs.imread_torch(p.replace("/", "\\"), md, 256)
            _, m32, i32 = bs.imread_torch(p.replace("/", "\\"), md, 32)
            q = bs.color_quantize(i32[0]).unsqueeze(0); m32 = 1.0 - m32.view(1, -1)
            g = bs.masked_sample(gpt, q.cuda(), m32.cuda(), sample=True, top_k=50)
            mimg = torch.cat([mimg, mask], 1)
            s = torch.from_numpy(bs.dequantize(g[0])).permute(2, 0, 1)[None].float() / 127.5 - 1
            _, o = up([mimg.cuda(), s.cuda()])
            o = o.cpu() * mask + mimg[:, :3] * (1 - mask)
            return ((o[0].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        return run
    if method == "latent-code":
        enter(os.path.join(FI, "latent-code-inpainting")); sys.argv = ["infer_latent.py"]
        il = importlib.import_module("infer_latent")
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/places_inpainting.yaml"); cfg["data"]["params"]["batch_size"] = 1
        model = il.instantiate_from_config(cfg.model).to(dev).eval()
        VQ, Enc, Tr, Un = model.helper_model
        VQ, Enc, Tr = VQ.to(dev), Enc.to(dev), Tr.to(dev); Un = Un.to(dev) if Un is not None else None
        from skimage.io import imread
        def run(f):  # infer_latent.main() for one image, res 256, clamp_ratio 0.25
            g = imread(os.path.join(DATA, "gt", f)); g = il.center_crop(g, s=256)
            x = il.preprocess(g, 256, dev)[:, :3]; k = torch.round(il.readmask(os.path.join(DATA, "mask", f), 256, dev))
            qz, _, info, mo = Enc.encode(x * k, k, clamp_ratio=0.25)
            mo = mo.reshape(1, -1); z = info[2].reshape(1, -1); nb = {"image": (x * k).permute(0, 2, 3, 1)}
            zc, _, _ = il.forward_to_indices(Tr, nb, z, mo, dev)
            B, C, H, Wd = qz.shape
            q = VQ.quantize.get_codebook_entry(zc.reshape(-1).int(), shape=(B, H, Wd, C))
            dec, *_ = model.current_model(nb, quant=q, mask_in=k, mask_out=mo.reshape(B, 1, H, Wd), return_fstg=False, debug=True)
            rec = x * k + dec * (1 - k)
            if Un is not None: rec = Un.refine(rec, k)
            return il.to_img(rec)
        return run
    if method == "StrDiff":
        d = os.path.join(NII, "StrDiffusion", "test", "texture", "config", "inpainting"); enter(d)
        import options as option
        from models import create_model
        import str_utils as str_util
        import utils as util
        from data import create_dataset
        tst_src = open(os.path.join(d, "test.py"), encoding="utf-8").read()
        ns = {}; exec(tst_src[tst_src.index("class Datasetset_mask"):tst_src.index("if __name__")], {**globals(), "transforms": importlib.import_module("torchvision.transforms"), "Image": Image, "os": os, "torch": torch, "glob": importlib.import_module("glob"), "Dataset": torch.utils.data.Dataset, "np": np, "osp": os.path}, ns)
        opt = option.dict_to_nonedict(option.parse(os.path.join(d, "options", "test", "ir-sde.yml"), is_train=False))
        model = create_model(opt)
        sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=dev); sde.set_model(model.model)
        S = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=dev); S.set_model(model.models)
        ds = create_dataset(opt["datasets"]["test1"]); idx = {os.path.basename(p): i for i, p in enumerate(ds.GT_paths)}
        mds = ns["Datasetset_mask"](opt["degradation"]["mask_root"]); midx = {os.path.basename(p): i for i, p in enumerate(mds.data)}
        def run(f):  # test.py loop body
            it = ds[idx[f]]; Y, Xg, Xe = it["GT"][None].cuda(), it["GT_gray"][None].cuda(), it["GT_edge"][None].cuda()
            k, _ = mds[midx[f]]; k = k[None].cuda()
            n1 = sde.noise_state(Y * k); n2 = S.noise_state(Xe * k)
            model.feed_data(n1, Y * k, Y, k, S, Xg, Xe * k)
            model.test(sde, save_states=False, GT=Y, mask=k, S_sde=S, S_GT=Xg, S_LQ=n2, dis=model.dis)
            o = model.get_current_visuals()["Output"]
            return util.tensor2img(o.squeeze())[:, :, ::-1]
        return run
    raise SystemExit("unknown method")


ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--list", default="all")
ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--nsamp", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
allfiles = sorted(os.listdir(os.path.join(DATA, "gt")))
files = allfiles if a.list == "all" else [l.strip() for l in open(a.list) if l.strip()]
run = build(a.method)
import time
with torch.no_grad():
    for s in a.seeds:
        for f in files:
            stem = os.path.splitext(f)[0]
            outs = [os.path.join(a.out, f"{stem}_s{s}_k{j}.png") for j in range(a.nsamp)]
            if all(os.path.exists(o) for o in outs): continue
            seed = s * 100000 + allfiles.index(f)
            torch.manual_seed(seed); np.random.seed(seed % 2**32); random.seed(seed)
            t0 = time.time()
            for o in outs:
                Image.fromarray(np.ascontiguousarray(run(f))).save(o)
            print(f"{f} s{s}: {(time.time()-t0)/len(outs):.2f}s/sample", flush=True)
print("done", a.method)
