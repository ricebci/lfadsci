#!/bin/bash
set -euo pipefail

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate lfads-gpu
fi

ROOT_OUT="$PWD/models/t19_finger_data_sweep"

# Sweep only data extraction parameters (no training hyperparameter sweep).
EPOCH_TIME_WINDOW="[-400,-200],[-200,0],[0,200],[200,400]"
# Channels passed as arrays [start, stop].
CHANNELS_SWEEP="55b,M1"

echo "Starting Hydra multirun data sweep..."
echo "outputDir=$ROOT_OUT"

python3 src/lfadsci/t19_train_partial.py --multirun \
  outputDir="$ROOT_OUT" \
  mode=train \
  epoch_time_window="$EPOCH_TIME_WINDOW" \
  channels="$CHANNELS_SWEEP"

echo "Sweep completed. Results under: $ROOT_OUT"
