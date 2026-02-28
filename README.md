# LFADS

LFADS (Latent Factor Analysis via Dynamical Systems) - a framework for learning dynamic latent representations from high-dimensional neural time series data.

*Improved interpretability in LFADS models using a learned, context-dependent per-trial bias* Nishal P. Shah, Benyamin Abramovich Krasa, Erin Kunz, Nick Hahn, Foram Kamdar, Donald Avansino, Leigh R. Hochberg, Jaimie M. Henderson, David Sussillo, bioRxiv 2025
Paper link: https://www.biorxiv.org/content/10.1101/2025.10.03.680303v1


## Installation

### Prerequisites
- Python 3.9 or higher
- Conda (Miniconda or Anaconda)

### Setup Instructions

1. **Create a new conda environment:**
```bash
conda create -n lfads python=3.9 -y
```

2. **Activate the environment:**
```bash
conda activate lfads
```

3. **Install the lfadsci package in development mode:**
```bash
pip install -e .
```

By default, this installs **TensorFlow CPU** (`tensorflow-cpu==2.7.0`).

4. **(Optional) Install with GPU TensorFlow:**
```bash
LFADSCI_GPU=1 pip install -e .
```

If you already installed the CPU version and want to switch to GPU:
```bash
pip uninstall -y tensorflow-cpu tensorflow
LFADSCI_GPU=1 pip install --force-reinstall -e .
```

This will install all required dependencies including:
- TensorFlow (CPU by default, GPU when `LFADSCI_GPU=1`)
- Hydra (configuration framework)
- Jupyter Lab and Jupyter Notebook support
- Scientific computing libraries (NumPy, SciPy, Pandas, Scikit-learn)
- Visualization tools (Seaborn, Matplotlib)
- Weights & Biases (experiment tracking)

### Verify Installation

To verify that the installation was successful:
```bash
python -c "import lfadsci; print('lfadsci successfully installed')"
```

## Usage

Once installed, you can import and use lfadsci in your Python scripts or Jupyter notebooks.

## Development

To work with Jupyter notebooks:
```bash
conda activate lfads
jupyter lab
```

The environment includes JupyterLab and IPywidgets for interactive development.

# Multi-session training


