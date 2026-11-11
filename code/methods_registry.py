"""Single source of truth for per-method facts (inventory + release configs).
Every field was read from the local repository (git remote / rev-parse, checkpoint file, scripts,
test_opt.txt) or measured on the saved outputs; see inventory.md for the provenance of each field.
Run: python methods_registry.py  -> results/inventory_methods.md, release/configs/<method>.yaml"""
import os, subprocess, json, hashlib
ROOT = os.environ.get("BENCH_ROOT", r"D:\Face Unmasking\Inpainting_Framwork\comparison\From-Pixel-to-Face-(survey)")
FI = "FI/000_Related work"; NII = "NII"

M = [
 # name, family, repo_dir, checkpoint(s) relative to repo, inference script, native, inference, notes
 ("DeepFillv2", "G1", f"{NII}/generative_inpainting", ["model_logs/release_places2_256"], "test_folder.py (untracked, wraps official test.py graph)", "256 (Places2 release_places2_256)", "256", "TensorFlow 1.x; outputs post-composited with a slightly different mask binarisation (mean |known diff| 0.29 grey levels)"),
 ("MADF", "G1", f"{FI}/Pytorch-MADF", ["checkpoints/places2.pth"], "test_madf.py (untracked)", "fully convolutional", "256", "composited at 256 (known pixels exact)"),
 ("Pluralistic", "G2", f"{NII}/Pluralistic", ["checkpoints/place2_random/latest_net_E.pth", "checkpoints/place2_random/latest_net_G.pth"], "test.py (official) with test_opt.txt", "256", "256", "PUBLISHED RUN INVALID: mask_type [3] draws a random mask per image and resize_and_crop 266->256 misaligns; corrected re-run with paired masks (work/runners/gan_run.py)"),
 ("PEN-Net", "G2", f"{FI}/PEN-Net-for-Inpainting", ["release_model/pennet_places2_square256/gen_00000.pth"], "test_pennet.py (untracked)", "256 (only release: Places2, square masks)", "256", "only released Places2 checkpoint was trained with square masks"),
 ("VHII", "G2", f"{FI}/variable-hyperparameter-image-impainting", ["trained_models/places/places_VHII_efficient/best_model_places.pth"], "test_dataset.py (official)", "256", "256", "PUBLISHED RUN INVALID: test_dataset.py pairs each image with maskfile[random.randint(...)]; corrected re-run with paired masks; deterministic (no sampling)"),
 ("CoModGAN", "G2", f"{FI}/co-mod-gan-pytorch", ["checkpoints/comod-places-512/co-mod-gan-places2-050000_net_G_ema.pth"], "test.py (official PyTorch port, modified to also save 256 copies)", "512", "512 (inputs NII/test_256_places_1K/512_masked, 512_mask, cv2 INTER_AREA up; output cv2 INTER_LANCZOS4 down to 256)", "third-party PyTorch port (zengxianyu) of official TF CoModGAN with converted Places 512 weights; not composited"),
 ("SCAT", "G2", f"{FI}/Generative-Image-Inpainting", ["src/checkpoints/Places2_G1000000.pt"], "src/test.py (official, modified output paths)", "256", "256", "composited"),
 ("latent-code", "G2", f"{FI}/latent-code-inpainting", ["ckpts/places256_vqgan1024_BASE.ckpt", "ckpts/places256_partialencoder.ckpt", "ckpts/places256_decoder.ckpt", "ckpts/places256_unet.ckpt", "ckpts/places256_transformer.ckpt"], "infer_latent.py (untracked; reproduces repository evaluation pipeline)", "256", "256", "composited"),
 ("EdgeConnect", "G3", f"{NII}/edge-connect", ["checkpoints/places2/EdgeModel_gen.pth", "checkpoints/places2/InpaintingModel_gen.pth"], "test.py (official, config edits)", "256", "256", "Canny edges (sigma 2) computed on CPU"),
 ("StructureFlow", "G3", f"{NII}/StructureFlow", ["checkpoints/places/checkpoints/latest_net_s_gen.pth", "checkpoints/places/checkpoints/latest_net_f_gen.pth"], "test.py (official, config edits)", "256", "256", "official resample2d CUDA kernel replaced by a pure-PyTorch re-implementation (src/resample2d.py, local modification)"),
 ("E2I", "G3", f"{FI}/E2I-inpainting", ["checkpoints/", "hed.npy"], "pipeline_master_tcsvt.py (official, modified)", "256", "256", "TensorFlow 1.x; HED edge detector"),
 ("MEDFE", "G3", f"{NII}/Rethinking-Inpainting-MEDFE", ["checkpoints/place2/EN.pkl", "checkpoints/place2/DE.pkl", "checkpoints/place2/MEDEF.pkl"], "test.py (official, modified)", "256", "256", "structure image (RTV smoothing) input"),
 ("CTSDG", "G3", f"{NII}/CTSDG", ["Checkpoints/places2.pt"], "test.py (official, modified)", "256", "256", "composited"),
 ("DGTS", "G3", f"{NII}/DGTS-Inpainting", ["code/test/checkpoints/places.pth", "code/test/checkpoints/latest_net_G.pth"], "code/test/test.py (official, modified paths)", "256 (coarse 64x64 + upsampler)", "256", "intermediate JPEG round-trips in official test code; output not composited (mean |known-pixel diff| 1.87 grey levels)"),
 ("ZITS", "G3", f"{NII}/ZITS_inpainting", ["ckpt/zits_places2/InpaintingModel_best_gen.pth", "ckpt/best_transformer_places2.pth", "ckpt/StructureUpsampling.pth", "ckpt/best_lsm_hawp.pth"], "FTR_inference.py (official, modified)", "256", "256", "HAWP wireframes precomputed (places2_val_wireframes)"),
 ("BAT-Fill", "G4", f"{NII}/BAT-Fill", ["checkpoints/places2_bat_pretrain/latest_tran.pth", "checkpoints/places2_bat_pretrain/latest_net_G.pth"], "bat_sample.py (official, modified path handling)", "256 (tokens 32x32)", "256", "top-k=50 autoregressive sampling, num_sample 1"),
 ("WaveFill", "G4", f"{FI}/WaveFill", ["checkpoints/places2_wavefill/latest_net_G.pth"], "infer.py (untracked)", "256", "256", "options of test_wavelet.sh (WaveletInpaintLv2GCFRAN, level 2, attention)"),
 ("LaMa", "G4", f"{NII}", ["~/.cache/torch/hub/checkpoints/big-lama.pt"], "simpleLaMa.py (simple-lama-inpainting 0.1.2, TorchScript big-lama)", "fully convolutional (trained 256)", "256", "Big-LaMa (Places-Challenge); TorchScript export from simple-lama-inpainting"),
 ("MISF", "G4", f"{NII}/misf", ["checkpoints/MISF/places2_InpaintingModel_gen.pth"], "test via main.py (official, config edits)", "256", "256", "composited"),
 ("CMT", "G4", f"{FI}/CMT", ["ckpt/Place2.pth"], "demo.py (official)", "256", "256", "composited"),
 ("spa-former", "G4", f"{FI}/spa-former", ["checkpoints/place2/latest_net_G.pth"], "test.py (official) with test_opt.txt", "256", "256", "PUBLISHED RUN INVALID: mask_type [2,4] generates masks on the fly; benchmark masks never used; corrected re-run with paired masks"),
 ("MADIN", "G4", f"{FI}/MADIN", ["checkpoints/Places2.pth"], "demo .py (official)", "256", "256", "composited"),
 ("RePaint", "G5", f"{NII}/RePaint", ["data/pretrained/places_250000.pt"], "test.py with confs/face_example.yml (model_path -> places_250000.pt)", "256", "256", "T=250 (respaced from 1000, linear), jump length 10, 10 resamplings, no classifier guidance; unseeded; not composited"),
 ("StrDiff", "G5", f"{NII}/StrDiffusion", ["test/texture/Places2/t.pth", "test/texture/Places2/s.pth", "test/texture/Places2/dis.pth"], "test/texture/config/inpainting/test.py with options/test/ir-sde.yml", "256", "256", "IR-SDE reverse-time SDE, T=400, cosine schedule, max_sigma 30, adaptive resampling (u_max 6, u_min 3, jump 5, re 5)"),
]


def git(repo, *a):
    try:
        return subprocess.check_output(["git", "-C", os.path.join(ROOT, repo), *a], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def main():
    out_md = [
        "| Method | Fam. | Repository (remote) | Commit | Checkpoint(s) | Inference script | Native res. | Inference res. | Notes |",
        "|---|---|---|---|---|---|---|---|---|"]
    os.makedirs(os.path.join(ROOT, "revision", "release", "configs"), exist_ok=True)
    for name, fam, repo, ck, script, native, inf, notes in M:
        url = git(repo, "remote", "get-url", "origin") or ("https://github.com/enesmsahin/simple-lama-inpainting (package); big-lama from https://github.com/advimman/lama" if name == "LaMa" else "")
        com = git(repo, "rev-parse", "HEAD") if git(repo, "remote", "get-url", "origin") else "n/a (pip package simple-lama-inpainting==0.1.2)"
        out_md.append(f"| {name} | {fam} | {url} | {com[:12]} | {'; '.join(ck)} | {script} | {native} | {inf} | {notes} |")
        y = [f"method: {name}", f"family: {fam}", f"repository: {url}", f"commit: {com}",
             "checkpoints:"] + [f"  - {c}" for c in ck] + [
             f"inference_script: \"{script}\"", f"native_resolution: \"{native}\"", f"inference_resolution: \"{inf}\"",
             "benchmark_resolution: 256", "mask_convention_input: 255 = hole (data/masks)", f"notes: \"{notes}\""]
        open(os.path.join(ROOT, "revision", "release", "configs", f"{name}.yaml"), "w").write("\n".join(y) + "\n")
    open(os.path.join(ROOT, "revision", "results", "inventory_methods.md"), "w", encoding="utf-8").write("\n".join(out_md) + "\n")
    print("\n".join(out_md))


if __name__ == "__main__":
    main()
