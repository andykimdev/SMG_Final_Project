#!/bin/bash

# Complete comparison pipeline
# Trains all baselines and compares with Graph Transformer

echo "=========================================="
echo "Survival Model Comparison Pipeline"
echo "=========================================="

# Step 1: Train baseline models
echo ""
echo "Step 1: Training baseline models..."
echo "----------------------------------------"

python train_baselines.py \
  --csv_path ../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../priors/tcga_string_prior.npz \
  --output_dir outputs/baselines \
  --device mps \
  --num_workers 0

# Step 2: Compare all models
echo ""
echo "Step 2: Comparing models..."
echo "----------------------------------------"

python compare_models.py \
  --results_dir outputs \
  --output_dir outputs/comparison

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Baseline models: outputs/baselines/"
echo "  - Comparison: outputs/comparison/"
echo "  - Plots: outputs/comparison/*.png"
echo ""