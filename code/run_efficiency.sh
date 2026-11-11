#!/bin/bash
# Table 12: one GPU (RTX 3090), batch 1, FP32 (ZITS transformer runs in FP16 as in its official code), 256x256
# (CoModGAN at its fixed 512). 100 timed runs after 10 warm-ups; RePaint/StrDiff: 3 runs after 1 warm-up.
cd "$(dirname "$0")"
OUT=../results/efficiency; mkdir -p $OUT
PY=envs/methods/Scripts/python.exe; export PYTHONPATH="$(cygpath -w "$PWD/loaders")"
declare -A G=( [DeepFillv2]=G1 [MADF]=G1 [Pluralistic]=G2 [PEN-Net]=G2 [VHII]=G4 [CoModGAN]=G2 [SCAT]=G2 [latent-code]=G2
 [EdgeConnect]=G3 [StructureFlow]=G3 [E2I]=G3 [MEDFE]=G3 [CTSDG]=G3 [DGTS]=G3 [ZITS]=G3
 [BAT-Fill]=G4 [WaveFill]=G4 [LaMa]=G4 [MISF]=G4 [CMT]=G4 [spa-former]=G4 [MADIN]=G4 [RePaint]=G5 [StrDiff]=G5 )
for m in "$@"; do
  [ -f "$OUT/$m.csv" ] && { echo "skip $m"; continue; }
  R="--runs 100 --warmup 10"; case $m in RePaint|StrDiff) R="--runs 3 --warmup 1";; esac
  $PY measure_efficiency.py --loader "loaders/$m.py" --name "$m" --gen ${G[$m]} $R --out "$OUT/$m.csv" > "$OUT/$m.log" 2>&1
  grep -E "params|FLOPs|latency|memory|steps|thop|evaluations" "$OUT/$m.log" | tr '\n' ' '; echo " <- $m"
done
