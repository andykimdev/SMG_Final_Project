# Graph Transformer Classifier

Cancer type classification from TCGA RPPA protein expression data using STRING PPI network structure.

## Usage

### Training

```bash
python train.py \
  --csv_path ../../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../../priors/tcga_string_prior.npz \
  --output_dir ../../results/classifiers/cancer_type_classifiers/transformer \
  --device cuda
```

Or use the convenience script:
```bash
bash run_training.sh
```

### Testing

```bash
python test.py \
  --checkpoint_path ../../results/classifiers/cancer_type_classifiers/transformer/checkpoints/best_model.pt \
  --csv_path ../../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../../priors/tcga_string_prior.npz \
  --split test
```

## Output

Results are saved to `results/classifiers/cancer_type_classifiers/transformer/`:
- `checkpoints/best_model.pt` - Best model checkpoint
- `results/` - Training history, test metrics, classification reports
- `plots/` - Training curves, confusion matrices

## Architecture

- Token embedding: expression value + protein ID + graph positional encoding
- Graph-aware attention using STRING PPI diffusion kernel
- CLS token for classification

## Configuration

See `config.py` for hyperparameters.
