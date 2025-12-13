#!/bin/bash
# Training script for BLINDED Graph Transformer Survival Predictor
# Uses ONLY features available at initial IHC pathology visit

# Default paths (modify as needed)
CSV_PATH="../../processed_datasets/tcga_pancan_rppa_compiled.csv"
PRIOR_PATH="../../priors/tcga_string_prior.npz"
OUTPUT_DIR="../outputs"
DEVICE="mps"  # Use "cuda" for NVIDIA GPU, "cpu" if no GPU available

echo "=========================================="
echo "BLINDED Graph Transformer Survival Predictor"
echo "=========================================="
echo "Features: Protein expression, Age, Sex, Race, Genetic ancestry"
echo "Excluded: Cancer type, tumor stage, genomic features"
echo ""
echo "CSV Path:   $CSV_PATH"
echo "Prior Path: $PRIOR_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Device:     $DEVICE"
echo ""

# Run training
python train_and_eval.py \
  --csv_path "$CSV_PATH" \
  --prior_path "$PRIOR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --use_clinical \
  --no_genomic
