from setuptools import setup, find_packages

setup(
    name='lfadsci',
    python_requires='>=3.6',
    version='0.0.1',
    package_dir={'': 'src'}, #find_packages(include=['src']),
    install_requires=[
        # 'tensorflow-gpu==2.7.0',
        'tensorflow-cpu==2.7.0',
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
