#!/bin/bash
# Convenience script to run Graph Transformer training

# Default paths (modify as needed)
CSV_PATH="../../processed_datasets/tcga_pancan_rppa_compiled.csv"
PRIOR_PATH="../../priors/tcga_string_prior.npz"
OUTPUT_DIR="../../results/classifiers/cancer_type_classifiers/transformer"
DEVICE="cuda"  # Use "cpu" if no GPU available

echo "Graph Transformer Cancer Type Classifier"
echo "========================================"
echo "CSV Path:   $CSV_PATH"
echo "Prior Path: $PRIOR_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Device:     $DEVICE"
echo ""

# Run training
python train.py \
  --csv_path "$CSV_PATH" \
  --prior_path "$PRIOR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"

