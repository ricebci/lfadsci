import os
import sys

from setuptools import setup, find_packages


def _use_gpu() -> bool:
    # Allow GPU selection through CLI flag: python setup.py install --gpu
    gpu_flag = '--gpu'
    if gpu_flag in sys.argv:
        sys.argv.remove(gpu_flag)
        return True

    # Also support env var for pip installs: LFADSCI_GPU=1 pip install .
    return os.getenv('LFADSCI_GPU', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


tensorflow_pkg = 'tensorflow-gpu==2.7.0' if _use_gpu() else 'tensorflow-cpu==2.7.0'

setup(
    name='lfadsci',
    python_requires='>=3.6',
    version='0.0.1',
    package_dir={'': 'src'}, #find_packages(include=['src']),
    install_requires=[
        tensorflow_pkg,
        'hydra-core==1.3.2',
        'hydra-submitit-launcher==1.1.5',
        'hydra-optuna-sweeper==1.2.0',
        'pandas',
        'jupyterlab',
        'ipywidgets',
        'tqdm',
        'seaborn',
        'numpy==1.25.0',
        'scipy==1.11.1',
        'wandb==0.15.5',
        'scikit-learn',
        'protobuf==3.20.1',
        'nlb_tools'
    ]
)
