#!/usr/bin/env bash
set -euo pipefail
: "${EXODISC_DATA_ROOT:?Set EXODISC_DATA_ROOT to the ExoDisc dataset directory}"
for fold in 1 2 3 4; do
  python train.py --config "configs/final/exodino3_fold${fold}.yaml"
done
