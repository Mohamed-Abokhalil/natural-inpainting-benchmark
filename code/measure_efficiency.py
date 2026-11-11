#!/usr/bin/env python3
"""
=============================================================================
 measure_efficiency.py -- fill Table `tab:efficiency` of the survey
=============================================================================

WHAT THIS DOES, AND WHAT IT DOES NOT DO
---------------------------------------
It does NOT train anything and does NOT run your benchmark. It loads a model
that is already trained (the authors' released weights, exactly the ones you
used for Table 4) and measures four things about it:

    parameters   how big the model is
    FLOPs        how much arithmetic one completion costs
    latency      how long one completion takes, in milliseconds
    memory       peak GPU memory for one completion

You already have everything needed: the repo and the checkpoint per method.


THE ONE PRACTICAL PROBLEM: 24 REPOS, 24 ENVIRONMENTS
-----------------------------------------------------
These repos need conflicting torch/CUDA versions, so a single environment
usually cannot import all of them. Do NOT try to register all 24 methods at
once. Instead:

    1. Activate the environment for ONE method.
    2. Write its loader in a small file (see LOADER RECIPE below).
    3. Run:   python3 measure_efficiency.py --loader loaders/lama.py \
                                            --name LaMa --gen G4 --year 2022 \
                                            --out results/lama.csv
    4. Repeat for each method.
    5. Merge everything into LaTeX rows:
              python3 measure_efficiency.py --merge results/*.csv

Hardware must be identical across all runs, otherwise the numbers are not
comparable and the table reproduces the very problem Table 3 documents. Same
GPU, same machine, nothing else running.


LOADER RECIPE  (this is the only part you write)
-------------------------------------------------
Open the method's own `test.py` / `predict.py` / `demo.py`. Find the two or
three lines that (a) build the network and (b) load the checkpoint. Copy them
into a file that defines ONE function called `load`:

    # loaders/lama.py
    import torch, yaml
    from omegaconf import OmegaConf
    from saicinpainting.training.trainers import load_checkpoint

    def load(device, size):
        cfg = OmegaConf.load('big-lama/config.yaml')
        model = load_checkpoint(cfg, 'big-lama/models/best.ckpt', strict=False)
        model = model.to(device).eval()

        img  = torch.randn(1, 3, size, size, device=device)
        mask = torch.zeros(1, 1, size, size, device=device)
        mask[..., size//4:3*size//4, size//4:3*size//4] = 1.0   # a square hole

        # Return exactly what the model's forward() expects.
        # LaMa takes a dict, so wrap it:
        return model, ({'image': img, 'mask': mask},)

Three rules for `load`:

  * return  (model, inputs_tuple)  where calling  model(*inputs_tuple)  runs
    one completion. If forward() takes a dict, make inputs_tuple a 1-tuple
    containing that dict, as above.
  * put the model in .eval() and on `device`.
  * return the GENERATOR ONLY. Most checkpoints bundle the discriminator,
    which is not used at inference and would inflate the parameter count by
    a factor of two or more. If the checkpoint is a full training object,
    pass its generator submodule (often .generator, .netG, or .G).

For a DIFFUSION method, `load` should return the single denoising network,
not the whole sampling loop, and you pass --steps <sampler steps>. FLOPs are
then per network evaluation and the Steps column carries the multiplier --
a fused number would hide where the cost actually comes from. Also pass
--latency-full so latency is timed over the complete sampling loop; for that
case have `load` return an object whose forward() runs the full sampler.

=============================================================================
"""
import argparse
import csv
import glob
import importlib.util
import os
import sys
import time

try:
    import torch
except ImportError:                 # --merge must work without torch, so that
    torch = None                    # you can merge on any machine afterwards


def import_loader(path):
    """Import a loader file and return its `load` function."""
    spec = importlib.util.spec_from_file_location("loader_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load"):
        raise SystemExit(f"{path} must define a function named `load(device, size)`")
    return mod.load


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, inputs):
    """MACs (G) per forward = aten-level FLOPs / 2, counted with
    torch.utils.flop_counter.FlopCounterMode (conv, linear, (b)mm, attention).
    Revision change: thop only hooks nn.Modules and misses functional matmuls
    and TorchScript models, so it is kept only as a secondary log value."""
    from torch.utils.flop_counter import FlopCounterMode
    try:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            model(*inputs)
        macs = fc.get_total_flops() / 2 / 1e9
    except Exception as e:
        print(f"  [warn] FlopCounterMode failed ({e})"); macs = float("nan")
    try:
        from thop import profile
        import copy
        t, _ = profile(copy.deepcopy(model), inputs=inputs, verbose=False)
        print(f"  [info] thop MACs (secondary) {t/1e9:.2f} G")
    except Exception as e:
        print(f"  [info] thop failed ({type(e).__name__})")
    return macs


def measure_latency(model, inputs, runs, warmup, device):
    cuda = device.startswith("cuda")
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(*inputs)
        if cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / runs * 1000.0


def measure_memory(model, inputs, device):
    if not device.startswith("cuda"):
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def fmt(v, nd=1):
    return "--" if v != v else f"{v:.{nd}f}"        # NaN -> --


def do_merge(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as fh:
                rows.extend(list(csv.DictReader(fh)))
    if not rows:
        raise SystemExit("no CSV rows found")

    order = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6}
    titles = {
        "G1": "G1 --- CNN and mask-aware convolution",
        "G2": "G2 --- GAN and contextual attention",
        "G3": "G3 --- Structure--texture learning",
        "G4": "G4 --- Transformer and frequency learning",
        "G5": "G5 --- Diffusion models",
        "G6": "G6 --- Multimodal and guided inpainting",
    }
    print("\n%% ---- paste into tab:efficiency ----")
    for g in sorted({r["gen"] for r in rows}, key=lambda g: order.get(g, 9)):
        grp = [r for r in rows if r["gen"] == g]
        print(f"    \\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{titles.get(g, g)}}}}}\\\\")
        for r in sorted(grp, key=lambda r: (int(r["year"]), r["method"])):
            print(f"    {r['year']} & {r['method']} & "
                  f"{fmt(float(r['params']))} & {fmt(float(r['flops']))} & "
                  f"{r['steps']} & {fmt(float(r['latency']))} & "
                  f"{fmt(float(r['memory']), 2)} \\\\")
        print("    \\midrule")
    print(f"\n%% merged {len(rows)} methods")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--loader", help="path to a .py file defining load(device, size)")
    ap.add_argument("--name", help="method name as it should appear in the table")
    ap.add_argument("--gen", default="", choices=["", "G1", "G2", "G3", "G4", "G5", "G6"])
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1,
                    help="network evaluations per completion (sampler steps for diffusion)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None, help="CSV to write for this method")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge CSVs into LaTeX rows and exit")
    # parse_known_args (not parse_args): loaders read their own extra flags,
    # e.g. --snapshot / --config / --checkpoint, straight from sys.argv.
    args, _loader_extra = ap.parse_known_args()

    if args.merge is not None:
        do_merge(args.merge)
        return

    if not (args.loader and args.name):
        raise SystemExit("need --loader and --name (or --merge). See --help.")
    if torch is None:
        raise SystemExit("torch is required to measure; activate the "
                         "method's environment first.")

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; pass --device cpu")
        print(f"[env] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
        print("[env] record this line -- it must be identical for every method")

    load = import_loader(args.loader)
    print(f"[{args.name}] loading...", flush=True)
    torch.manual_seed(0)
    model, inputs = load(args.device, args.size)
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    p = count_params(model)
    # Revision: multi-step samplers expose one network evaluation for MAC counting;
    # latency/memory still time the complete sampling loop (all steps).
    if hasattr(model, "macs_model"):
        f = count_flops(model.macs_model, model.macs_inputs)
    elif hasattr(model, "cnt"):   # adaptive multi-network sampler: mean MACs per network evaluation
        for k in model.cnt: model.cnt[k] = 0
        tot = count_flops(model, inputs); n = sum(model.cnt.values())
        print(f"  [info] network evaluations in one run: {dict(model.cnt)}")
        f = tot / max(n, 1); model.nfe = n
    else:
        f = count_flops(model, inputs)
    if hasattr(model, "nfe"):
        args.steps = model.nfe
    lat = measure_latency(model, inputs, args.runs, args.warmup, args.device)
    mem = measure_memory(model, inputs, args.device)

    print(f"  params  {fmt(p)} M")
    print(f"  FLOPs   {fmt(f)} G  (per network evaluation)")
    print(f"  steps   {args.steps}")
    print(f"  latency {fmt(lat)} ms")
    print(f"  memory  {fmt(mem, 2)} GB")
    if p > 200:
        print("  [check] >200M parameters -- is the discriminator included? "
              "Pass the generator submodule only.")

    out = args.out or f"{args.name.lower().replace(' ', '_')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "gen", "year", "params",
                                           "flops", "steps", "latency", "memory"])
        w.writeheader()
        w.writerow(dict(method=args.name, gen=args.gen, year=args.year,
                        params=p, flops=f, steps=args.steps,
                        latency=lat, memory=mem))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
