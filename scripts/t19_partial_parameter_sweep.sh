#!/bin/bash
set -euo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate lfads

# T19 task: dual-GPU hyperparameter sweep for partial pipeline (train + results_partial_*.pkl)
# This runs two Hydra multirun workers in parallel, each pinned to one GPU.

ROOT_OUT="$PWD/models/t19_finger_sweep"
COMMON_ARGS=(
    outputDir="$ROOT_OUT"
    mode=train
    n_steps=40000,60000
    model.n_hidden_decode=100,200
    model.generator_type=gru
    model.ic_dim=5,10
    model.bias_dim=5,10
    data_seed=98
    seed=98
)

# Split factors across GPUs so each configuration runs exactly once.
# Total tasks = 2 (n_steps) * 2 (n_hidden_decode) * 2 (factors) * 2 (ic_dim) * 2 (bias_dim) = 32
GPU0_FACTORS="20"
GPU1_FACTORS="80"

echo "Launching GPU 0 worker (factors: $GPU0_FACTORS)"
python3 src/lfadsci/t19_train_partial.py --multirun \
    "${COMMON_ARGS[@]}" \
    gpuNumber=0 \
    model.factors="$GPU0_FACTORS" &
PID0=$!

echo "Launching GPU 1 worker (factors: $GPU1_FACTORS)"
python3 src/lfadsci/t19_train_partial.py --multirun \
    "${COMMON_ARGS[@]}" \
    gpuNumber=1 \
    model.factors="$GPU1_FACTORS" &
PID1=$!

cleanup() {
    echo "Stopping background sweep workers..."
    kill "$PID0" "$PID1" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$PID0"
wait "$PID1"

echo "Dual-GPU sweep completed. Results under: $ROOT_OUT"