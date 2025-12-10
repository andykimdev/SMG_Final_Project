# Linear Baseline Classifiers

Linear baseline models for cancer type classification using PCA + Logistic Regression/SVM.
Compares performance against the Graph Transformer classifier.

## Usage

```bash
python train.py \
  --csv_path ../../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../../priors/tcga_string_prior.npz \
  --output_dir ../../results/classifiers/cancer_type_classifiers/linear \
  --transformer_dir ../../results/classifiers/cancer_type_classifiers/transformer
```

## Models

- **PCA + Logistic Regression**: Various PCA components (50, 80, 90, 95% variance)
- **PCA + Linear SVM**: Linear SVM on PCA-reduced features
- **Full Features + Logistic Regression**: Logistic regression on all features

## Output

Results are saved to `results/classifiers/cancer_type_classifiers/linear/`:
- `results/` - Classification reports, baseline results JSON
- `plots/` - Comparison plots, PCA visualizations, confusion matrices
- `training_log_*.txt` - Full training logs

## Configuration

See `config.py` for hyperparameters and data splitting configuration.

