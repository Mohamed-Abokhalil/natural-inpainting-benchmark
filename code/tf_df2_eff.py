#!/usr/bin/env python3
"""
=============================================================================
 measure_efficiency_tf.py -- efficiency measurement for TensorFlow 1.x methods
=============================================================================

Companion to measure_efficiency.py (which is PyTorch-only). Use THIS script
for methods whose released code is TensorFlow 1.x with static graphs and
neuralgym -- e.g. the original DeepFillv2 (Generative Inpainting) repo, whose
test.py you provided.

It writes the SAME CSV columns as the PyTorch script, so you merge everything
together at the end with:
    python measure_efficiency.py --merge results/*.csv

WHAT IT MEASURES
    params    total graph variables, in millions
    FLOPs     from tf.profiler (float_ops), in G  -- see note below
    latency   ms per completion, timed on sess.run of the SAME output tensor
              the test script uses
    memory    peak GPU bytes (best-effort; TF1 grabs memory eagerly, so this
              is less precise than PyTorch's reset_peak_memory_stats -- state
              "--" and rely on params/FLOPs/latency if the number looks wrong)

HOW TO RUN  (inside the DeepFillv2 repo, its TF1 environment active)
    Put this file in the repo root, next to inpaint_model.py and inpaint.yml,
    then:

        python measure_efficiency_tf.py --name DeepFillv2 --gen G1 --year 2019 \
               --size 256 --runs 100 --out results/DeepFillv2.csv

    It builds the graph exactly like your test.py (concatenate image+mask on
    axis=2, build_server_graph, load checkpoint), then times sess.run instead
    of writing images.

NOTE ON FLOPs
    tf.profiler counts graph float_ops, which for a static graph includes the
    whole forward pass. It is a MACs-adjacent number, reported in G to match
    prevailing practice; state the convention in the table caption, as for the
    PyTorch methods.
=============================================================================
"""
import argparse
import csv
import os
import subprocess
import threading
import time

import numpy as np

# TF1 / TF2-compat import: works whether the env has TF 1.x or TF 2.x
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception:
    import tensorflow as tf

import neuralgym as ng
from inpaint_model import InpaintCAModel


def build_dummy(size):
    """A single image+mask pair, sized like the benchmark (size x size),
    concatenated on axis=2 exactly as test.py does."""
    grid = 8
    s = size // grid * grid
    img = np.random.randint(0, 256, (s, s, 3)).astype(np.float32)
    mask = np.zeros((s, s, 3), np.float32)
    mask[s // 4:3 * s // 4, s // 4:3 * s // 4, :] = 255.0   # a square hole
    img = np.expand_dims(img, 0)
    mask = np.expand_dims(mask, 0)
    return np.concatenate([img, mask], axis=2)              # (1, s, 2s, 3)


def count_params():
    total = 0
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        total += int(np.prod(v.shape.as_list()))
    return total / 1e6


def count_flops(graph):
    try:
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        flops = tf.profiler.profile(graph, options=opts)
        return (flops.total_float_ops / 1e9) if flops else float("nan")
    except Exception as e:
        print(f"  [warn] FLOP profiling failed ({e}); reporting --")
        return float("nan")


def fmt(v, nd=1):
    return "--" if v != v else f"{v:.{nd}f}"


def query_gpu_mem_mb(gpu_index):
    """Total used memory (MB) on one GPU, via nvidia-smi. This is the sum over
    ALL processes on the card, so it must be baselined (see GPUMemPoller)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL)
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return None


class GPUMemPoller(threading.Thread):
    """Samples GPU memory in a background thread and records the peak.
    Framework-agnostic: works for TF1, TF2, and PyTorch identically, so every
    method's memory is measured the same way.

    Because allow_growth=True, TF only holds what it actually uses, so the
    peak-minus-baseline is a fair estimate of this model's footprint. Nothing
    else should be running on the GPU while you measure.
    """
    def __init__(self, gpu_index=0, interval=0.02):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.baseline = query_gpu_mem_mb(gpu_index) or 0.0
        self.peak = self.baseline
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            m = query_gpu_mem_mb(self.gpu_index)
            if m is not None and m > self.peak:
                self.peak = m
            time.sleep(self.interval)

    def stop_and_report_gb(self):
        self._stop_evt.set()
        self.join(timeout=1.0)
        return max(0.0, (self.peak - self.baseline)) / 1024.0   # MB -> GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gen", default="G1")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--checkpoint", default="model_logs/release_places2_256")
    ap.add_argument("--config", default="inpaint.yml")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index for nvidia-smi memory polling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    FLAGS = ng.Config(args.config)
    model = InpaintCAModel()

    input_image = build_dummy(args.size)
    input_tensor = tf.constant(input_image, dtype=tf.float32)

    print("[build] building server graph...")
    output = model.build_server_graph(FLAGS, input_tensor)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)

    params = count_params()
    flops = count_flops(tf.get_default_graph())

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        # load weights exactly like test.py
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            sess.run(tf.assign(
                var, tf.contrib.framework.load_variable(args.checkpoint, var.name)))
        print("[build] weights loaded.")

        for _ in range(args.warmup):
            sess.run(output)

        # start sampling GPU memory just before the timed loop
        poller = GPUMemPoller(gpu_index=args.gpu, interval=0.02)
        poller.start()

        t0 = time.perf_counter()
        for _ in range(args.runs):
            sess.run(output)
        t1 = time.perf_counter()
        latency = (t1 - t0) / args.runs * 1000.0

        mem = poller.stop_and_report_gb()
        # revision: use TensorFlow's own allocator peak (GPU:0) instead of nvidia-smi polling
        try:
            import tensorflow as _tf2
            mem = _tf2.config.experimental.get_memory_info('GPU:0')['peak'] / 1024 ** 3
        except Exception as _e:
            print('  [warn] get_memory_info failed', _e)
        if mem <= 0:
            mem = float("nan")   # nvidia-smi unavailable or nothing measured

    print(f"[{args.name}]")
    print(f"  params  {fmt(params)} M")
    print(f"  FLOPs   {fmt(flops)} G")
    print(f"  steps   {args.steps}")
    print(f"  latency {fmt(latency)} ms")
    print(f"  memory  {fmt(mem, 2)} GB")
    if params > 200:
        print("  [check] >200M params -- discriminator variables may be in the "
              "graph. For inference-only counts, build only the generator, or "
              "filter vars whose name starts with 'inpaint_net' (generator "
              "scope) vs 'discriminator'.")

    out = args.out or f"results/{args.name.lower()}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "gen", "year", "params",
                                           "flops", "steps", "latency", "memory"])
        w.writeheader()
        w.writerow(dict(method=args.name, gen=args.gen, year=args.year,
                        params=params, flops=flops, steps=args.steps,
                        latency=latency, memory=mem))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
