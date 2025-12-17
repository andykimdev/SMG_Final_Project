# Graph Transformer Cancer Classification

Standalone implementation of a graph-aware transformer for cancer type classification from protein expression data.

## Overview

This project implements a Graph Transformer classifier that uses protein-protein interaction (PPI) network structure from STRING to improve cancer type classification. The model incorporates graph structure through a learnable attention bias mechanism.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Without data**: Open `notebooks/model_demo.ipynb` to view model architecture
2. **With data**: Place data files in `data/` and run notebooks for training and analysis
3. **With pretrained model**: Use `notebooks/architecture_and_results.ipynb` to generate results figures

## Project Structure

### `src/`
Core model implementation:
- `config.py`: Unified configuration for model and training parameters
- `model.py`: GraphTransformerClassifier implementation
- `graph_prior.py`: PPI network processing and diffusion kernel computation
- `dataset.py`: Data loading, preprocessing, and splitting utilities
- `training.py`: Training and evaluation functions

### `interpretability/`
Analysis tools for model interpretation:
- `utils.py`: Shared utilities for loading models and data
- `shap_analysis.py`: SHAP value computation for protein importance
- `attention_analysis.py`: Attention pattern extraction and visualization
- `pca_baseline.py`: PCA95 + Logistic Regression baseline model
- `ppi_heatmap.py`: PPI network visualization

### `notebooks/`
Three Jupyter notebooks for different use cases:

- **`model_demo.ipynb`**: Interactive demo for training and evaluation. Checks data availability, trains model (or displays architecture if no data), and runs basic evaluation.

- **`interpretability_analysis.ipynb`**: Full interpretability pipeline. Computes SHAP values, extracts attention patterns, trains PCA baseline, and generates analysis plots.

- **`architecture_and_results.ipynb`**: Results visualization notebook. Loads pre-computed results and generates publication-ready figures (training curves, model comparison, SHAP vs attention, graph bias scale).

### `data/`
Data storage directory:
- `processed_datasets/`: TCGA RPPA compiled CSV
- `priors/`: STRING PPI prior NPZ file
- See `data/README.md` for format details

### `pretrained/`
Pretrained model weights:
- `best_model.pt`: Model checkpoint
- `model_config.json`: Model configuration

### `results/`
Generated outputs:
- `plots/SHAP_Plots/`: SHAP importance results
- `plots/Attention_Plots/`: Attention pattern outputs
- `plots/PCA_Cox_Plots/`: PCA baseline results
- `plots/Model_Comparison_Plots/`: Comparative analysis figures

## Running the Notebooks

```bash
cd notebooks
jupyter notebook
```

Recommended order:
1. `model_demo.ipynb` - verify setup and train/load model
2. `interpretability_analysis.ipynb` - run full analysis (~10-20 min for SHAP)
3. `architecture_and_results.ipynb` - generate final figures

## Programmatic Usage

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "interpretability"))

from model import GraphTransformerClassifier
from utils import load_trained_model

model, graph_prior, label_info = load_trained_model()
model.eval()
logits = model(protein_expression_tensor)
```

## Model Architecture

- **Embedding**: Expression value + protein ID embedding + graph positional encoding
- **Graph Attention**: Multi-head attention with learnable PPI diffusion kernel bias
- **Classification**: CLS token-based classification head
- **Parameters**: ~830K trainable parameters

## Configuration

Key hyperparameters in `src/config.py`:
- Model: 128-dim embeddings, 4 layers, 8 attention heads
- Training: 1e-4 learning rate, 64 batch size, 100 max epochs
- Graph: Normalized Laplacian, β=0.5 diffusion kernel

## Expected Results

- GaTmCC test accuracy: ~88%
- PCA95+LogReg baseline: ~93-96%
- SHAP protein importance rankings
- Attention pattern visualizations
- Graph bias scale analysis

## Dependencies

See `requirements.txt`. Key packages:
- PyTorch >= 2.0.0
- NumPy, Pandas, Scikit-learn
- SHAP >= 0.41.0
- Matplotlib, Seaborn, Jupyter

## Workaround should GitHub code is insufficient

See 'https://drive.google.com/drive/folders/1Z7KktivWf7Rte6I9dGlv65goffnVoTWQ?usp=sharing',
which contains all necessary code / files. 



