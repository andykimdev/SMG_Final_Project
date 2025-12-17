# Cancer Classifier

Graph-aware transformer model for classifying cancer types from TCGA RPPA protein expression data, incorporating STRING protein-protein interaction network structure.

## Overview

This classifier predicts cancer type from protein expression profiles using:
- **TCGA Pan-Cancer RPPA data**: 198 protein measurements from 7,500+ samples
- **STRING PPI network**: Experimentally-validated protein interactions as graph structure
- **Graph Transformer**: Combines protein expression with network topology via:
  - Graph positional encodings (Laplacian eigenvectors)
  - Diffusion kernel-based attention bias
  - Protein-aware embeddings

## Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# From the Cancer Classifier directory
cd "/path/to/Cancer Classifier"

# Run training
python scripts/train_classifier.py \
  --csv_path ../Prior_Processor/datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../Prior_Processor/data/tcga_string_prior.npz \
  --output_dir outputs \
  --device mps  # or 'cuda' or 'cpu'
```

**Expected output:**
- Training completes in ~10-15 minutes on GPU
- Test accuracy: ~85%
- Model saved to `outputs/checkpoints/best_model.pt`

### Quick Test

```bash
# Verify installation
python -c "from classifier.models import GraphTransformerClassifier; print('✓ Installation OK')"

# Run a quick test
python tests/test_modules.py
```

## Project Structure

```
Cancer Classifier/
├── classifier/                      # Main Python package
│   ├── __init__.py
│   ├── config.py                    # Hyperparameters
│   ├── models/
│   │   ├── __init__.py
│   │   └── graph_transformer.py     # Graph-aware transformer model
│   └── data/
│       ├── __init__.py
│       ├── dataset.py               # TCGA RPPA data loading
│       └── graph_prior.py           # STRING PPI network processing
│
├── scripts/                         # Executable scripts
│   └── train_classifier.py          # Main training script
│
├── tests/                           # Unit tests
│   └── test_modules.py              # Module verification tests
│
├── outputs/                         # Results (created at runtime)
│   ├── checkpoints/                 # Saved models
│   ├── results/                     # Metrics and reports
│   ├── plots/                       # Visualizations
│   └── logs/                        # Training logs
│
├── graph_transformer/               # [LEGACY] Alternative implementation
├── linear/                          # [LEGACY] Baseline models
│
├── config.py                        # [DEPRECATED] Backward compat shim
├── graph_prior.py                   # [DEPRECATED] Backward compat shim
├── graph_transformer_classifier.py  # [DEPRECATED] Backward compat shim
├── dataset_tcga_rppa.py            # [DEPRECATED] Backward compat shim
├── train_and_eval.py               # [LEGACY] Old training script
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── TRAINING_RESULTS_SUMMARY.md     # Latest training results
```

## Installation

### Option 1: Quick Install

```bash
# Clone/navigate to the repository
cd "/path/to/SMG_Final_Project/Cancer Classifier"

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Development Install

```bash
# Install in editable mode with all dependencies
cd "/path/to/SMG_Final_Project"
pip install -e "Cancer Classifier[dev]"
```

### Verify Installation

```bash
python -c "
from classifier.models import GraphTransformerClassifier
from classifier.data import load_graph_prior, load_and_preprocess_data
print('✓ All imports successful')
"
```

## Usage

### 1. Basic Training

Train a model with default hyperparameters:

```bash
python scripts/train_classifier.py \
  --csv_path ../Prior_Processor/datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../Prior_Processor/data/tcga_string_prior.npz \
  --output_dir outputs \
  --device cuda
```

### 2. Custom Configuration

Edit `classifier/config.py` to modify hyperparameters:

```python
# Model architecture
MODEL = {
    'embedding_dim': 128,      # Try 64, 128, 256
    'n_layers': 4,             # Try 2-6 layers
    'n_heads': 8,              # Must divide embedding_dim
    'ffn_dim': 512,            # Feedforward dimension
    'dropout': 0.1,            # Regularization
    'pe_dim': 16,              # Graph positional encoding dim
    'graph_bias_scale': 1.0,   # Graph attention bias scale
}

# Training settings
TRAINING = {
    'batch_size': 64,          # Adjust based on GPU memory
    'learning_rate': 1e-4,     # AdamW learning rate
    'weight_decay': 1e-5,      # L2 regularization
    'max_epochs': 100,         # Maximum training epochs
    'patience': 15,            # Early stopping patience
    'grad_clip': 1.0,          # Gradient clipping
}
```

### 3. Use Trained Model

```python
import torch
from classifier.models import GraphTransformerClassifier
from classifier.data import load_graph_prior

# Load graph prior
prior_path = "../Prior_Processor/data/tcga_string_prior.npz"
graph_prior = load_graph_prior(prior_path)

# Load checkpoint
checkpoint = torch.load("outputs/checkpoints/best_model.pt")

# Initialize model
model = GraphTransformerClassifier(
    n_proteins=198,
    n_classes=32,
    diffusion_kernel=torch.from_numpy(graph_prior['K']),
    positional_encodings=torch.from_numpy(graph_prior['PE'])
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    logits = model(protein_expression)  # Shape: (batch, 198)
    predictions = torch.argmax(logits, dim=1)

# Get cancer type names
cancer_type = checkpoint['label_info']['idx_to_label'][predictions.item()]
print(f"Predicted cancer type: {cancer_type}")
```

### 4. Evaluate Model

```python
from classifier.data import load_and_preprocess_data, create_dataloaders

# Load data
data_splits, label_info, scaler = load_and_preprocess_data(
    csv_path="../Prior_Processor/datasets/tcga_pancan_rppa_compiled.csv",
    protein_cols=graph_prior['protein_cols']
)

# Create test loader
_, _, test_loader = create_dataloaders(data_splits, batch_size=64)

# Evaluate
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for proteins, labels in test_loader:
        outputs = model(proteins)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
```

## Command-Line Arguments

### train_classifier.py

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--csv_path` | str | Yes | - | Path to TCGA RPPA CSV file |
| `--prior_path` | str | Yes | - | Path to STRING prior .npz file |
| `--output_dir` | str | No | `outputs` | Directory for saving results |
| `--device` | str | No | `cpu` | Device: 'cuda', 'mps', or 'cpu' |
| `--num_workers` | int | No | `0` | Number of data loader workers |

**Example:**

```bash
python scripts/train_classifier.py \
  --csv_path ../Prior_Processor/datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../Prior_Processor/data/tcga_string_prior.npz \
  --output_dir my_experiment \
  --device mps \
  --num_workers 4
```

## Model Architecture

### Overview

```
Input: Protein Expression (N=198)
    ↓
[Value Embedding] + [Protein ID Embedding] + [Graph PE]
    ↓
[CLS Token] + [Protein Tokens]
    ↓
Graph Transformer Layers (4 layers)
    • Multi-head Self-Attention with Graph Bias
    • Feed-Forward Network
    • Layer Normalization
    • Residual Connections
    ↓
[CLS Token Representation]
    ↓
Classification Head (MLP)
    ↓
Output: Cancer Type Logits (32 classes)
```

### Graph-Aware Attention

The key innovation is graph-biased attention:

```python
# Standard attention
scores = Q @ K^T / sqrt(d_k)

# Graph-aware attention (our approach)
scores = Q @ K^T / sqrt(d_k) + graph_bias

# Where graph_bias = learnable_scale * DiffusionKernel[i,j]
```

This biases attention toward proteins that interact in the PPI network.

### Graph Features

1. **Diffusion Kernel** (K):
   - K = exp(-β L) where L is graph Laplacian
   - Captures multi-hop connectivity
   - Used as attention bias

2. **Positional Encodings** (PE):
   - Top 16 eigenvectors of Laplacian
   - Encodes graph topology
   - Added to token embeddings

## Data Format

### Input CSV Format

```csv
PATIENT_ID,CANCER_TYPE_ACRONYM,GENE1|PROTEIN1,GENE2|PROTEIN2,...
TCGA-AA-1234,BRCA,0.45,1.23,...
TCGA-BB-5678,LUAD,-0.32,0.89,...
```

**Requirements:**
- Protein columns must contain `|` character
- Gene symbol before `|`, protein ID after
- Numeric values (z-scored expression)
- Missing values allowed (imputed with column mean)

### STRING Prior Format

The `.npz` file contains:

```python
import numpy as np
prior = np.load('tcga_string_prior.npz', allow_pickle=True)

# Contents:
prior['A']             # Adjacency matrix (198 × 198)
prior['protein_cols']  # Protein column names
prior['genes']         # Gene symbols
```

## Hyperparameters

### Model Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `embedding_dim` | 128 | 64-256 | Token embedding dimension |
| `n_layers` | 4 | 2-8 | Number of transformer layers |
| `n_heads` | 8 | 4-16 | Attention heads (must divide embedding_dim) |
| `ffn_dim` | 512 | 256-1024 | Feedforward network dimension |
| `dropout` | 0.1 | 0.0-0.5 | Dropout probability |
| `pe_dim` | 16 | 8-32 | Graph positional encoding dimension |
| `graph_bias_scale` | 1.0 | 0.1-10.0 | Initial scale for graph bias |

### Training

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `batch_size` | 64 | 16-256 | Training batch size |
| `learning_rate` | 1e-4 | 1e-5 to 1e-3 | AdamW learning rate |
| `weight_decay` | 1e-5 | 0-1e-3 | L2 regularization |
| `max_epochs` | 100 | 50-200 | Maximum epochs |
| `patience` | 15 | 5-30 | Early stopping patience |
| `grad_clip` | 1.0 | 0.5-5.0 | Gradient clipping threshold |

### Data Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `missing_threshold` | 0.5 | Drop samples with >50% missing proteins |
| `min_samples_per_class` | 10 | Minimum samples per cancer type |
| `train_ratio` | 0.85 | Training set proportion |
| `val_ratio` | 0.10 | Validation set proportion |
| `test_ratio` | 0.05 | Test set proportion |

## Performance

### Latest Results (2025-12-13)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 85.41% |
| **F1 Score (macro)** | 78.93% |
| **F1 Score (weighted)** | 84.87% |

**Top Performing Cancer Types:**
- PRAD (Prostate): 100% accuracy
- BRCA (Breast): 94.1% F1
- LGG (Low Grade Glioma): 97.8% F1
- THCA (Thyroid): 94.4% F1

See [TRAINING_RESULTS_SUMMARY.md](TRAINING_RESULTS_SUMMARY.md) for detailed results.

### Comparison with Baselines

| Method | Accuracy | F1 (macro) |
|--------|----------|------------|
| Random Forest | ~75% | ~68% |
| Standard CNN | ~82% | ~74% |
| Transformer (no graph) | ~83% | ~75% |
| **Graph Transformer (ours)** | **85.4%** | **78.9%** |

**Improvement:** +3-5% over non-graph methods

## Output Files

After training, the following files are generated:

### Checkpoints
- `outputs/checkpoints/best_model.pt` - Best model (lowest validation loss)
  - Contains: model weights, label mappings, hyperparameters

### Results
- `outputs/results/test_results.json` - Numerical test metrics
- `outputs/results/classification_report.txt` - Per-class precision/recall/F1
- `outputs/results/training_history.json` - Full training history

### Visualizations
- `outputs/plots/training_curves.png` - Loss and accuracy over epochs
- `outputs/plots/confusion_matrix.png` - Test set confusion matrix

### Logs
- `outputs/logs_YYYYMMDD_HHMMSS.txt` - Complete training log

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce batch size
# Edit classifier/config.py:
TRAINING['batch_size'] = 32  # or 16
```

**2. Slow Training**
```bash
# Use GPU if available
python scripts/train_classifier.py --device cuda  # or 'mps' for Apple Silicon
```

**3. Import Errors**
```bash
# Ensure you're in the Cancer Classifier directory
cd "/path/to/Cancer Classifier"

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/SMG_Final_Project"
```

**4. Missing Data Files**
```bash
# Verify data paths
ls ../Prior_Processor/datasets/tcga_pancan_rppa_compiled.csv
ls ../Prior_Processor/data/tcga_string_prior.npz

# Or use absolute paths in commands
```

**5. Poor Performance**
- Increase model capacity: `embedding_dim=256`, `n_layers=6`
- Adjust learning rate: try `3e-4` or `5e-5`
- More epochs: `max_epochs=200`
- Different graph bias: `graph_bias_scale=0.5` or `2.0`

## Advanced Usage

### Custom Graph Prior

Build your own STRING prior:

```bash
cd ../Prior_Processor/scripts

python build_string_prior.py \
  --csv ../../Prior_Processor/datasets/your_data.csv \
  --out ../data/custom
```

This creates `../data/custom_string_prior.npz`

### Ensemble Predictions

```python
# Train multiple models with different seeds
for seed in [42, 123, 456]:
    # Set RANDOM_SEED = seed in config.py
    # Train model
    # Save to different output directory

# Combine predictions
predictions = []
for checkpoint in model_checkpoints:
    model.load_state_dict(checkpoint)
    pred = model(data)
    predictions.append(pred)

ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

### Transfer Learning

Use pre-trained weights for a new dataset:

```python
# Load pre-trained model
checkpoint = torch.load("outputs/checkpoints/best_model.pt")

# Initialize new model
model = GraphTransformerClassifier(
    n_proteins=198,
    n_classes=your_n_classes,  # Different number of classes
    diffusion_kernel=K,
    positional_encodings=PE
)

# Load only transformer weights
state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
              if not k.startswith('classifier')}
model.load_state_dict(state_dict, strict=False)

# Freeze transformer, train only classifier
for param in model.transformer.parameters():
    param.requires_grad = False
```

## Development

### Running Tests

```bash
# Run all tests
python tests/test_modules.py

# Or use pytest
pytest tests/
```

### Code Style

```bash
# Format code
black classifier/ scripts/ tests/

# Check style
flake8 classifier/ scripts/ tests/
```

### Contributing

1. Create a new branch for your feature
2. Make changes and add tests
3. Ensure all tests pass
4. Submit a pull request

## Citation

If you use this code, please cite:

```bibtex
@software{cancer_classifier_2025,
  title={Graph-Aware Transformer for Pan-Cancer Classification},
  author={SMG Final Project Team},
  year={2025},
  url={https://github.com/your-repo/SMG_Final_Project}
}
```

## References

- **TCGA Pan-Cancer Atlas**: https://www.cancer.gov/tcga
- **STRING Database**: https://string-db.org
- **RPPA Core Facility**: https://www.mdanderson.org/research/research-resources/core-facilities/functional-proteomics-rppa-core.html
- **Graph Transformers**: Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs", AAAI 2021

## License

MIT License - See LICENSE file for details

---

**Last Updated:** December 13, 2025
**Model Version:** 1.0
**Python Version:** 3.8+
