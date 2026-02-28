#!/bin/bash
set -euo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate lfads

# Pendulum task: hyperparameter sweep (Hydra multirun)
python3 src/lfadsci/main.py --multirun \
    dataset=pendulum \
    outputDir=$PWD/models/ \
    model.dropout_rate=0.0 \
    model.bias_dim=10 \
    model.ic_dim=10 \
    model.lam_l2=0,1 \
    model.tv_input_dim=1,2,5 \
    model.noise_stddev=0.0 \
    model.n_hidden_decode=200 \
    model.generator_type='gru' \
    seed=98,99,100,101,102,103,104,105,106,107 \
    model.use_bias=True \
    model.use_tv_input=True
