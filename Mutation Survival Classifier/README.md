# Mutation-Augmented Graph Transformer for Survival Prediction

Graph-based transformer model for cancer survival prediction using RPPA protein expression + genomic mutations.

## Features

- **Multimodal Input**: Combines protein expression (RPPA), clinical features, genomic features, and mutation data
- **Graph-Aware Attention**: Uses STRING PPI network to bias attention mechanism
- **Survival Modeling**: Cox Proportional Hazards loss with concordance index evaluation
- **Flexible Configuration**: Can train with/without mutations and cancer type information

## Project Structure

```
Mutation Survival Classifier/
├── mutation_survival_classifier/      # Python package
│   ├── config.py                     # Hyperparameters
│   ├── data/
│   │   ├── dataset.py                # Data loading with mutation support
│   │   └── graph_prior.py            # STRING PPI network processing
│   └── models/
│       ├── graph_transformer.py      # Main model
│       └── baseline.py               # Baseline models
├── scripts/
│   └── train_and_eval.py            # Training script
└── outputs/                          # Training outputs
```

## Data Format

Expected CSV columns:
- **Proteins**: 198 RPPA features (e.g., `AKT|Akt`, `MTOR|mTOR_pS2448`)
- **Clinical**: Age, sex, race, ancestry, cancer type (optional), tumor stage, etc.
- **Genomic**: TMB, MSI, aneuploidy score
- **Mutations**: Binary features `MUT_TP53`, `MUT_KRAS`, `MUT_PIK3CA`, etc. (55 features)
- **Survival**: `DSS_STATUS` (0=censored, 1=death), `DSS_MONTHS`

## Usage

### Basic Training (with mutations and cancer type)

```bash
cd "/Users/andykim/Documents/2025 Fall/SMG/Project/SMG_Final_Project/Mutation Survival Classifier/scripts"

python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/mutation_survival" \
  --device mps \
  --use_mutations \
  --use_cancer_type
```

### Blinded Model (no mutations, no cancer type)

```bash
python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/blinded" \
  --device mps \
  --no_mutations \
  --no_cancer_type
```

### Mutations Only (no cancer type)

```bash
python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/mutations_only" \
  --device mps \
  --use_mutations \
  --no_cancer_type
```

## Model Architecture

### Input Processing
1. **Protein Tokens** (198): Each protein → 256-dim embedding
   - Expression value projection
   - Protein ID embedding (learned)
   - Graph positional encoding (from PPI network)

2. **Special Tokens**:
   - CLS token (global aggregation)
   - Clinical token (age, sex, race, ±cancer type)
   - Genomic token (TMB, MSI, aneuploidy, ±mutations)

### Transformer Layers (3 layers)
- **Multi-head attention** (8 heads) with graph-biased attention
- **Feed-forward network** (256 → 1024 → 256)
- **Pre-norm** architecture with residual connections
- **Dropout** (0.4) for regularization

### Output
- Risk prediction head extracts CLS token
- MLP: 256 → 128 → 1 (continuous risk score)
- Higher score = worse prognosis

## Training

- **Loss**: Cox Partial Likelihood (handles censored data)
- **Metric**: C-index (concordance index, 0.5=random, 1.0=perfect)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=5e-4)
- **Regularization**: Dropout 0.4, gradient clipping, early stopping (patience=10)
- **Data splits**: 80% train, 10% val, 10% test (stratified by event status)

## Hyperparameters

See `mutation_survival_classifier/config.py`:
- **Model**: 3 layers, 8 heads, 256 dims, dropout 0.4
- **Training**: batch 64, lr 3e-4, max 50 epochs
- **Graph**: β=0.5 diffusion, 16 PE dims

Based on optimized hyperparameters from Survival Classifier project (v7_shallow_wide).

## Expected Performance

Baseline comparisons (from Survival Classifier):
- **Graph Transformer** (unblinded): 0.780 C-index
- **PCA 50 + Cox**: 0.732 C-index
- **MLP**: 0.760 C-index

With mutations added, expect:
- **Mutation-augmented model**: 0.78-0.82 C-index (if mutations provide signal)
- **Blinded model** (no mutations, no cancer type): 0.72-0.74 C-index

## Outputs

Each training run creates:
```
outputs/mutation_survival/run_YYYYMMDD_HHMMSS/
├── config.json          # Training configuration
├── best_model.pth       # Best model checkpoint
├── results.json         # Test set performance
└── history.npz          # Training curves (loss, C-index)
```

## Data Preprocessing

1. **Quality filtering**: Remove samples with >50% missing proteins
2. **Missing imputation**: Mean imputation (proteins, genomic), 0 for mutations
3. **Normalization**: Z-score per feature (fit on training set only)
4. **Splits**: Stratified by event status to balance censoring

## Mutation Features

55 binary mutation features for key cancer genes:
- **Oncogenes**: KRAS, BRAF, PIK3CA, EGFR, ERBB2
- **Tumor suppressors**: TP53, PTEN, RB1, NF1
- **Pathway genes**: AKT1, MTOR, MAP2K1, CTNNB1
- **DNA repair**: BRCA2, MSH2, MSH6

## Notes

- Dataset has **7,523 samples** across 32 cancer types
- **~35% event rate** (deaths)
- Mutations are **sparse** (most genes <5% mutated)
- Model learns if mutations provide survival signal beyond protein expression

## Citation

Based on:
- TCGA Pan-Cancer RPPA data (MD Anderson)
- STRING v11 PPI network
- Graph Transformer architecture adapted for survival prediction
