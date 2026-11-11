# Unified Places2 inpainting benchmark

Code, configurations, test-set definition and per-image scores of the unified benchmark in

> M. Mahmoud, M. F. Senussi, M. Abdalla, M. S. Kasem, H.-S. Kang. **From Convolutions to Diffusion: A Decade of Deep Learning for Natural Image Inpainting.** *Neurocomputing* (under review, manuscript NEUCOM-D-26-17181).

Repository: <<REPO_URL>> · Archived snapshot: https://doi.org/<<DOI>> · Generated outputs: the Zenodo record named in `zenodo_record.json` (see below)

Twenty-four inpainting methods (2019–2026) are evaluated with their authors' released Places2 checkpoints under one protocol:
1,000 Places365-Standard test images (official 256×256 release), each paired with one QD-IMD test mask (hole ratio up to 50 %, five coverage bins),
256×256 inference, one sample per input for stochastic methods, and identical metric implementations
(PSNR/SSIM/LPIPS/ℓ1 with TorchMetrics 1.9.0 and lpips 0.1.4, FID with pytorch-fid 0.3.0). The full protocol is in Section 12.5 of the paper.

## Results (Table 8 of the paper)

Full-image metrics on the 1,000 pairs, as returned by each method. **Bold**: best, <u>underline</u>: second best.
\* corrected in the revision: re-run with the paired benchmark masks (the originally published rows came from runs that did not use them).

| Family | Year | Method | PSNR ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ | ℓ1 ↓ |
|---|---|---|---|---|---|---|---|
| G1 | 2019 | DeepFillv2 | 31.38 | 0.9492 | 10.33 | 0.0327 | 0.00812 |
| G1 | 2021 | MADF | 32.71 | 0.9586 | 8.19 | 0.0271 | <u>0.00588</u> |
| G2 | 2019 | Pluralistic* | 29.49 | 0.9251 | 23.26 | 0.0647 | 0.00982 |
| G2 | 2019 | PEN-Net | 27.06 | 0.9109 | 26.46 | 0.0796 | 0.01391 |
| G2 | 2021 | CoModGAN | 29.38 | 0.9330 | 10.94 | 0.0503 | 0.01724 |
| G2 | 2023 | SCAT | **33.46** | **0.9635** | **6.38** | <u>0.0209</u> | **0.00549** |
| G2 | 2024 | Latent-code | 31.11 | 0.9462 | 9.59 | 0.0307 | 0.00756 |
| G3 | 2019 | EdgeConnect | 31.02 | 0.9457 | 10.49 | 0.0317 | 0.00737 |
| G3 | 2019 | StructureFlow | 31.41 | 0.9536 | 7.62 | 0.0282 | 0.00792 |
| G3 | 2020 | E2I | 31.67 | 0.9512 | 8.34 | 0.0272 | 0.00683 |
| G3 | 2020 | MEDFE | 30.31 | 0.9375 | 15.94 | 0.0465 | 0.00934 |
| G3 | 2021 | CTSDG | 30.24 | 0.9311 | 19.78 | 0.0602 | 0.00859 |
| G3 | 2022 | DGTS | 28.58 | 0.9005 | 26.51 | 0.0985 | 0.01587 |
| G3 | 2022 | ZITS | 26.70 | 0.9157 | 31.84 | 0.1010 | 0.01269 |
| G4 | 2021 | BAT-Fill | 31.24 | 0.9520 | 8.78 | 0.0281 | 0.00796 |
| G4 | 2021 | WaveFill | 29.79 | 0.9418 | 14.33 | 0.0448 | 0.00903 |
| G4 | 2022 | LaMa | 32.43 | 0.9565 | 6.77 | 0.0224 | 0.00623 |
| G4 | 2022 | MISF | <u>32.93</u> | 0.9592 | 7.10 | 0.0227 | 0.00590 |
| G4 | 2023 | CMT | 32.45 | 0.9569 | 7.45 | 0.0245 | 0.00610 |
| G4 | 2023 | VHII* | 32.65 | 0.9572 | 6.79 | 0.0236 | 0.00605 |
| G4 | 2024 | Spa-former* | 31.10 | 0.9562 | 10.01 | 0.0326 | 0.00647 |
| G4 | 2026 | MADIN | 32.85 | <u>0.9595</u> | <u>6.39</u> | **0.0200** | 0.00596 |
| G5 | 2022 | RePaint | 31.14 | 0.9491 | 7.49 | 0.0279 | 0.00949 |
| G5 | 2024 | StrDiff | 28.21 | 0.9336 | 14.50 | 0.0618 | 0.00990 |

Confidence intervals, KID, P-/U-IDS, NIQE (Table 9), diversity (Table 10), per-coverage-bin and masked-region scores (Table 11) and efficiency
(Table 12) are in `scores/tables/`; per-image scores of every method are in `scores/per_image/`.

## Contents

| Path | What |
|---|---|
| `data/benchmark_pairs.csv` | the 1,000 image–mask pairs: benchmark file name, Places365-Standard test file, QD-IMD test mask file, hole ratio, coverage bin |
| `data/masks/` | the 1,000 binary masks as used (256×256 PNG, 255 = missing) |
| `data/seeds.json` | every seed (image sampling 42, qualitative scenes 142, diversity subset 2026, stochastic re-runs, bootstrap 0) |
| `data/diversity100.*`, `data/diffusion10.txt` | the subsets of Table 10 |
| `configs/<method>.yaml` | repository URL, commit, checkpoint(s), inference script, native and inference resolution, deviations |
| `environments/<method>/` | `requirements.txt` (and `environment.yml` for the TensorFlow methods) of the environment used for the re-runs and efficiency measurements; `environments/eval/` for the metrics |
| `code/` | evaluation, statistics, re-run runners, efficiency loaders, figure scripts |
| `scores/per_image/`, `scores/per_image_composited/` | per-image scores (full image and masked region) |
| `scores/tables/` | the CSV/JSON behind Tables 8–12 and Supplementary Tables S1–S3, including bootstrap arrays |
| `MANIFEST_outputs.csv` | SHA-256 of every generated output image |
| `zenodo_record.json`, `download_outputs.py` | location and checksums of the archives with the generated outputs (not in git) |

## Getting the generated outputs

The 24×1,000 benchmark outputs (2.5 GB) and the Table 10 samples (1.5 GB) are archived on Zenodo, one zip per method:

```bash
python download_outputs.py            # all archives -> outputs/<method>/, samples_table10/<method>/
python download_outputs.py --only outputs
```

## Reproducing the tables

Set `BENCH_ROOT` to a working folder laid out as follows (the scripts default to the authors' local path):

```
$BENCH_ROOT/data/places2/gt/<id>.png           ground-truth images (rebuild with step 1)
$BENCH_ROOT/data/places2/mask/<id>.png         masks (copy data/masks/)
$BENCH_ROOT/data/places2/SOTA/<method>/<id>.png  outputs (download_outputs.py, then copy outputs/<method>/)
$BENCH_ROOT/revision/results/                  results written by the scripts
$BENCH_ROOT/NII/, $BENCH_ROOT/FI/000_Related work/   official method repositories (configs/<method>.yaml), for re-runs and efficiency only
```

```bash
pip install -r environments/eval/requirements.txt
export BENCH_ROOT=/path/to/workdir
# 1. test set: Places365-Standard test images listed in data/benchmark_pairs.csv (official 256x256 release, no resizing);
#    masks: use data/masks/ (QD-IMD test masks, nearest-neighbour 512->256, threshold 127, inverted)
# 2. per-image metrics and Inception features (imports the original metric code, code/evaluate_inpainting_original.py, unchanged)
python code/eval_core.py --variant asis
# 3. Tables 8, 9, 11 and Supplementary S1-S3 (paired bootstrap, 1,000 resamples, seed 0)
python code/stats_tables.py final
python code/ties.py
python code/make_tables.py
# 4. Table 10 (needs samples_table10/)
python code/table10.py Pluralistic CoModGAN latent-code BAT-Fill RePaint StrDiff
python code/table10_v2.py
# 5. Table 12 (one GPU; official repositories and checkpoints required)
bash code/run_efficiency.sh SCAT LaMa MISF   # any subset of the 24 method names
# 6. re-running a method (the three corrected rows and the stochastic samples)
python code/runners/gan_run.py --method spa-former --list all --seeds 0 --out outputs/spa-former
```

## Adding a new method under the same protocol

1. Run the method with its official checkpoint on the 1,000 pairs of `data/benchmark_pairs.csv`, at 256×256, with the masks of `data/masks/`
   (255 = missing; convert to the method's own convention if needed), one sample per input and no best-of-N selection.
   Save `<id>.png` (256×256 RGB) for every benchmark file name in `$BENCH_ROOT/data/places2/SOTA/<NewMethod>/`.
2. Record the run in `configs/<NewMethod>.yaml` (repository, commit, checkpoint, inference script, resolution, whether the output is composited onto the known pixels)
   and its environment in `environments/<NewMethod>/requirements.txt`.
3. Compute the scores: `python code/eval_core.py --variant asis --methods <NewMethod>`, then rerun `python code/stats_tables.py final`
   (bootstrap intervals and tie tests are recomputed for all methods) and `python code/make_tables.py`.
4. Optional: `code/loaders/<NewMethod>.py` (see the existing loaders) to measure efficiency with `bash code/run_efficiency.sh <NewMethod>`.

## Known deviations from the authors' code (also in the configs)

* CoModGAN: third-party PyTorch port (zengxianyu/co-mod-gan-pytorch) with converted official Places 512 weights; run at 512 and resized.
* StructureFlow: the CUDA `resample2d` kernel replaced by a PyTorch re-implementation.
* DeepFillv2, MADF, PEN-Net, WaveFill, Latent-code, LaMa: short inference scripts calling the official networks; LaMa is the TorchScript Big-LaMa of `simple-lama-inpainting` 0.1.2.
* Efficiency only: StyleGAN2 CUDA ops replaced by their PyTorch reference implementations (`code/shims/stylegan2_op_native.py`); TensorFlow-1 code run on TF 2.10 through `tf.compat.v1` (`code/shims/tf`).
* The environments of the original inference runs were not preserved; `environments/` records those of the re-runs and measurements.

## Data licenses

The code is released under the MIT license (`LICENSE`). The data are not: the test images come from Places365-Standard
(Zhou et al., MIT CSAIL; distributed for non-commercial research and educational use under the Places terms of use) and the masks from the
Quick Draw Irregular Mask Dataset (QD-IMD; derived from the Google Quick, Draw! data). The masks in `data/masks/`, the generated outputs and all
derived images inherit the licenses and terms of use of Places2 and QD-IMD; check those terms before redistributing them. The method outputs were
produced with the authors' released checkpoints, which have their own licenses (see each repository in `configs/`).

## Citation

See `CITATION.cff`, or cite the paper above.
