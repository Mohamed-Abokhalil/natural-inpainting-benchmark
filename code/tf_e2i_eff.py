"""E2I efficiency (Table 12), run through shims/tf/run_tf1.py with workdir = E2I repo.
Builds the graphs of pipeline_master_tcsvt.py (coarse completion net -> HED on the coarse result ->
completion net with edge), loads the official Places2 weights and hed.npy, and times the complete
three-stage pipeline (three sess.run calls, as in the official code) at 256x256.
Params: trainable variables of completion_net (shared by both passes) + HED. FLOPs: tf.profiler float ops
of the three graphs (FLOPs = 2 x MACs). Memory: tf.config.experimental.get_memory_info peak."""
import sys, time, csv, numpy as np
import tensorflow as tf
from network_test import network
from hed import FCN
S = 256; runs, warm = int(sys.argv[1]), int(sys.argv[2]); out = sys.argv[3]
cfg = tf.ConfigProto(allow_soft_placement=True); cfg.gpu_options.allow_growth = True
g = tf.Graph()
with g.as_default(), tf.Session(config=cfg) as sess:
    img = tf.placeholder(tf.float32, [1, S, S, 3]); edge = tf.placeholder(tf.float32, [1, S, S, 1]); mask = tf.placeholder(tf.float32, [1, S, S, 1])
    m1 = network(img, edge, mask, reuse=False); m1.build_model()
    m2 = network(img, edge, mask, reuse=True); m2.build_model()
    hin = tf.placeholder(tf.float32, [1, S, S, 3])
    with tf.variable_scope("HED", reuse=False):
        net = FCN({"data": hin}); outs = net.get_output()
        hed = tf.nn.sigmoid(tf.reduce_mean(tf.concat([o[:, :S, :S, :] for o in outs], -1), -1, True))
    sess.run(tf.global_variables_initializer())
    tf.train.Saver([v for v in tf.global_variables() if "completion_net" in v.name]).restore(sess, "checkpoints/places2/model")
    with tf.variable_scope("HED", reuse=True): net.load("hed.npy", sess)
    n_comp = sum(int(np.prod(v.shape)) for v in tf.global_variables() if "completion_net" in v.name)
    n_hed = sum(int(np.prod(v.shape)) for v in tf.global_variables() if v.name.startswith("HED"))
    rng = np.random.RandomState(0)
    I = rng.uniform(-1, 1, (1, S, S, 3)).astype(np.float32); M = np.zeros((1, S, S, 1), np.float32); M[:, S // 4:3 * S // 4, S // 4:3 * S // 4] = 1
    E = np.ones((1, S, S, 1), np.float32)
    def pipeline():
        c = sess.run(m1.reconstruct_img, {img: I, edge: E, mask: M})
        h = sess.run(hed, {hin: (c * 127.5 + 127.5)[..., ::-1] - np.array((104.00698793, 116.66876762, 122.67891434))})
        e = (((1 - h) * (1 - M) + M) - 0.5) * 2
        return sess.run([m2.reconstruct_edge, m2.reconstruct_img], {img: I, edge: e, mask: M})
    for _ in range(warm): pipeline()
    t0 = time.perf_counter()
    for _ in range(runs): pipeline()
    lat = (time.perf_counter() - t0) / runs * 1000
    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024 ** 3
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    fl = tf.profiler.profile(g, options=opts).total_float_ops / 1e9
    # profile counts the whole graph once: completion net (built twice, shared weights) + HED
    print(f"params completion {n_comp/1e6:.2f}M hed {n_hed/1e6:.2f}M | graph FLOPs {fl:.1f}G | latency {lat:.1f} ms | peak {peak:.2f} GB")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "gen", "year", "params", "flops", "steps", "latency", "memory"])
        w.writerow(["E2I", "G3", 2020, (n_comp + n_hed) / 1e6, fl / 2 / 3, 3, lat, peak])
