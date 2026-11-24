#!/bin/bash
# Full version-2 pipeline after generate_all_samples.sh.
set -e
for s in A B; do
  python -m v2.run_v2 process --mode mumu --sample $s &
done
wait
for m in ee qq; do for s in A B; do
  python -m v2.run_v2 process --mode $m --sample $s &
done; done
wait
python -m v2.truth_gates samples/zh240_mumu_A samples/zh240_mumu_B samples/zh240_ee_A samples/zh240_qq_A
python -m v2.run_v2 closure
python -m v2.run_v2 whizard
python -m v2.run_v2 analyse --toys 3000
echo PIPELINE_DONE
