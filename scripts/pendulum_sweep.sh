#!/bin/bash
ml load python/3.6.1 viz cudnn/8.1.1.33 cuda/11.2.0 py-scipy/1.1.0_py36 py-matplotlib/3.1.1_py36 py-pandas/1.0.3_py36 py-sympy/1.1.1_py36 py-numpy/1.17.2_py36 py-scikit-learn/0.24.2_py36 py-tensorflow/2.4.1_py36 py-jupyter/1.0.0_py36
source /oak/stanford/groups/henderj/nishalps/venvs/lfads_ci/bin/activate

# Pendulum, with multiple seeds
python3 /oak/stanford/groups/henderj/nishalps/code/lfads_ci/src/lfadsci/main.py --multirun hydra/launcher=gpu_slurm \
    dataset=pendulum \
    outputDir=/scratch/users/nishalps/seq_models/pendulum/ \
    model.dropout_rate=0.0 \
    model.bias_dim=5 \
    model.ic_dim=5 \
    model.lam_l2=0 \
    model.tv_input_dim=1 \
    model.noise_stddev=0.0 \
    model.n_hidden_decode=200 \
    model.generator_type='gru' \
    seed=98 \
    model.use_bias=True,False \
    model.use_tv_input=False \
    mode='eval'


python3 src/lfadsci/main.py \
    dataset=pendulum \
    outputDir=~/users/nishalps/fits/ \
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


# python3 /oak/stanford/groups/henderj/nishalps/code/lfads_ci/src/lfadsci/main.py --multirun hydra/launcher=gpu_slurm \
#     dataset=pendulum \
#     outputDir=/scratch/users/nishalps/seq_models/pendulum/ \
#     model.dropout_rate=0.0 \
#     model.bias_dim=10 \
#     model.ic_dim=10 \
#     model.lam_l2=0,1 \
#     model.tv_input_dim=1,2,5 \
#     model.noise_stddev=0.0 \
#     model.n_hidden_decode=200 \
#     model.generator_type='gru' \
#     seed=98,99,100,101,102,103,104,105,106,107 \
#     model.use_bias=True \
#     model.use_tv_input=True