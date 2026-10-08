#!/usr/bin/env bash
# Offline candidate-reduction comparison (topk_offline). CPU only; fixtures dumped by fixture_dump_p1_*.txt.
# Accuracy columns are deterministic; timing columns are taken as the median over the 3 runs.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=$PWD; OUT=07_results/topk_offline/evidence
cd "$OUT"   # keep the 0-byte hailort.log that libhailort creates out of the repo root
for run in 1 2 3; do
  for n in default mot20_02 mot20_05; do
    echo "[$(date +%T)] run $run $n ($(vcgencmd measure_temp | cut -d= -f2))"
    "$R/build/bin/yvo_postproc_probe" --probe topk --fixture-dir "$R/build/out/topk_fixtures/$n" \
      --scores 0.2,0.05,0.02,0.003 --ks 100,300,1024 > "topk_${n}_run${run}.txt" 2>&1
    grep -E "^\[check\]|^\[RESULT\]" "topk_${n}_run${run}.txt" | tr '\n' ' '; echo
  done
done
[ -f hailort.log ] && [ ! -s hailort.log ] && rm hailort.log || true
