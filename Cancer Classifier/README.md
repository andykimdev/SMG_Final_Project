# Graph Transformer Cancer Type Classifier

Graph-aware transformer model for predicting cancer types from TCGA RPPA protein expression data, incorporating STRING protein-protein interaction network structure.

## Overview

This classifier uses:
- **TCGA Pan-Cancer RPPA data**: 198 protein measurements from 7,500+ cancer samples
- **STRING PPI network**: Experimentally-validated protein interactions as graph structure
- **Graph Transformer**: Combines protein expression with network topology via:
  - Graph positional encodings (Laplacian eigenvectors)
  - Diffusion kernel-based attention bias
  - Protein-aware embeddings

## Project Structure

```
Classifier/
├── config.py                          # Hyperparameters
├── graph_prior.py                     # STRING prior loading & processing
├── dataset_tcga_rppa.py              # TCGA data loading & preprocessing
├── graph_transformer_classifier.py   # Model architecture
├── train_and_eval.py                 # Training script
├── outputs/                          # Results (created at runtime)
│   ├── checkpoints/
│   └── results/
└── README.md
```

## Usage

### 1. Train the Model

```bash
python train_and_eval.py \
  --csv_path ../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../priors/tcga_string_prior.npz \
  --output_dir outputs \
  --device cuda
```

**Arguments:**
- `--csv_path`: Path to TCGA RPPA CSV file
- `--prior_path`: Path to STRING prior `.npz` file
- `--output_dir`: Directory for saving checkpoints and results
- `--device`: `cuda` or `cpu`
- `--num_workers`: Number of data loader workers (default: 0)

### 2. Output

The script will create:

**Checkpoints:**
- `outputs/checkpoints/best_model.pt`: Best model based on validation loss

**Results:**
- `outputs/results/training_history.json`: Loss/accuracy curves
- `outputs/results/test_results.json`: Final test metrics
- `outputs/results/classification_report.txt`: Per-class precision/recall/F1
- `outputs/results/confusion_matrix.png`: Confusion matrix visualization

### 3. Monitor Training

The script uses tqdm progress bars and prints:
- Training/validation loss and accuracy per epoch
- Early stopping based on validation loss (patience: 15 epochs)
- Learning rate scheduling (ReduceLROnPlateau)

## Model Architecture

### Token Embeddings
Each protein token is represented as:
```
token_i = LinearProj(expression_i) + ProteinEmbedding(i) + GraphPE(i)
```

### Graph-Aware Attention
For protein tokens i and j:
```
attention_logits[i,j] += learnable_scale * DiffusionKernel[i,j]
```

This biases attention toward proteins that are connected in the PPI network.

### Classification Head
```
[CLS] token → MLP → logits over cancer types
```

## Hyperparameters

Key settings in `config.py`:
- **Embedding dim**: 128
- **Transformer layers**: 4
- **Attention heads**: 8
- **Learning rate**: 1e-4
- **Batch size**: 64
- **Graph PE dim**: 16 eigenvectors
- **Train/Val/Test split**: 70/15/15 (by patient)

## Data Preprocessing

1. **Filter samples**: Drop samples with >50% missing protein values
2. **Patient-level split**: Prevent data leakage
3. **Imputation**: Fill missing values with training set column means
4. **Standardization**: Z-score normalization (fit on train, apply to all)
5. **Alignment**: Protein columns ordered to match STRING prior

## Graph Features

### Laplacian
```
L = I - D^{-1} A  (normalized)
```

### Diffusion Kernel
```
K = exp(-β L)  where β = 0.5
```

### Positional Encodings
Top 16 eigenvectors of L

## Requirements

```
torch>=2.6.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.12.0
tqdm>=4.66.0
```

## Expected Performance

With default hyperparameters on TCGA data (~30 cancer types):
- **Test accuracy**: 85-90%
- **Training time**: ~10-20 minutes on GPU
- **Model size**: ~2-3M parameters

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in `config.py`
- Use `--device cpu` if GPU memory is limited

**Poor Performance:**
- Increase `n_layers` or `embedding_dim`
- Adjust `diffusion_beta` (try 0.1 - 1.0)
- Increase `max_epochs` or reduce `patience`

**Data Issues:**
- Ensure protein columns in CSV match those in STRING prior
- Check for sufficient samples per cancer type (min: 10)
