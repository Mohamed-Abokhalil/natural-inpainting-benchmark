"""Run unmodified TensorFlow-1.x scripts on TensorFlow 2.10 (GPU on native Windows) through tf.compat.v1.
- `import tensorflow as tf` resolves to tf.compat.v1 with v2 behaviour disabled;
- tf.contrib.{framework,layers} are provided from tf_slim / compat equivalents (only the symbols the
  DeepFillv2 / neuralgym / E2I code uses);
- neuralgym is a verbatim copy from the original TFGAN environment (shims/tf/neuralgym).
Usage: python run_tf1.py <workdir> <script.py> [script args...]
"""
import os, sys, types, runpy
here = os.path.dirname(os.path.abspath(__file__))
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import tf_slim
contrib = types.ModuleType("tensorflow.contrib")
fw = types.ModuleType("tensorflow.contrib.framework")
fw.arg_scope = tf_slim.arg_scope; fw.add_arg_scope = tf_slim.add_arg_scope
fw.load_variable = tf1.train.load_variable
fwp = types.ModuleType("tensorflow.contrib.framework.python"); fwo = types.ModuleType("tensorflow.contrib.framework.python.ops")
fwo.arg_scope = tf_slim.arg_scope; fwo.add_arg_scope = tf_slim.add_arg_scope
layers = types.ModuleType("tensorflow.contrib.layers")
layers.l2_regularizer = lambda s: tf1.keras.regularizers.l2(s)
layers.xavier_initializer_conv2d = lambda *a, **k: tf1.glorot_uniform_initializer()
layers.xavier_initializer = lambda *a, **k: tf1.glorot_uniform_initializer()
layers.flatten = tf_slim.flatten
layers.conv2d = tf_slim.conv2d; layers.conv2d_transpose = tf_slim.conv2d_transpose   # same variable names as contrib.layers
layers.batch_norm = tf_slim.batch_norm; layers.conv = tf_slim.conv2d
contrib.framework = fw; contrib.layers = layers; fw.python = fwp; fwp.ops = fwo
tf1.contrib = contrib
try:
    tf1.compat.v1.contrib = contrib
except Exception:
    pass
sys.modules.update({"tensorflow": tf1, "tensorflow.contrib": contrib, "tensorflow.contrib.framework": fw,
                    "tensorflow.contrib.framework.python": fwp, "tensorflow.contrib.framework.python.ops": fwo,
                    "tensorflow.contrib.layers": layers})
sys.path.insert(0, here)          # neuralgym copy
workdir, script = sys.argv[1], sys.argv[2]
os.chdir(workdir); sys.path.insert(0, workdir)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
