#!/bin/bash
# Convenience script to run Linear Baseline training

# Default paths (modify as needed)
CSV_PATH="../../processed_datasets/tcga_pancan_rppa_compiled.csv"
PRIOR_PATH="../../priors/tcga_string_prior.npz"
OUTPUT_DIR="../../results/classifiers/cancer_type_classifiers/linear"
TRANSFORMER_DIR="../../results/classifiers/cancer_type_classifiers/transformer"

echo "Linear Baseline Classifiers for Cancer Type Classification"
echo "=========================================================="
echo "CSV Path:         $CSV_PATH"
echo "Prior Path:      $PRIOR_PATH"
echo "Output Dir:      $OUTPUT_DIR"
echo "Transformer Dir: $TRANSFORMER_DIR"
echo ""

# Run training
python train.py \
  --csv_path "$CSV_PATH" \
  --prior_path "$PRIOR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --transformer_dir "$TRANSFORMER_DIR"

