"""tf_df2_eff.py writes tf.profiler float operations in the 'flops' column; Table 12 reports MACs = FLOPs / 2.
Reads results/efficiency/DeepFillv2_raw.csv (kept) and writes DeepFillv2.csv with MACs."""
import os, csv
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "efficiency")
r = next(csv.DictReader(open(os.path.join(E, "DeepFillv2_raw.csv"))))
r["flops"] = float(r["flops"]) / 2
with open(os.path.join(E, "DeepFillv2.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(r)); w.writeheader(); w.writerow(r)
print(r)
