# LFADS

LFADS (Latent Factor Analysis via Dynamical Systems) - a framework for learning dynamic latent representations from high-dimensional neural time series data.

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

This will install all required dependencies including:
- TensorFlow (CPU version)
- Hydra (configuration framework)
- PyTorch
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
