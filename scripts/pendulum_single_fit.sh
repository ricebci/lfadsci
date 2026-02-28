#!/bin/bash

conda activate lfads

# Pendulum task: single model fit
python3 src/lfadsci/main.py \
    dataset=pendulum \
    outputDir=./fits/ \
    model.dropout_rate=0.0 \
    model.bias_dim=5 \
    model.ic_dim=5 \
    model.lam_l2=0 \
    model.tv_input_dim=1 \
    model.noise_stddev=0.0 \
    model.n_hidden_decode=200 \
    model.generator_type='gru' \
    seed=98 \
    model.use_bias=True \
    model.use_tv_input=False \
    mode='train'
