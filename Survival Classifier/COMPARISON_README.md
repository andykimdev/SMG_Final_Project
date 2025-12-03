# Survival Model Comparison Study

## Research Question
**Does stratifying cancer patients by their signaling topology predict survival better than standard molecular screening?**

## Experimental Design

### Models Compared

| Model | Uses PPI Topology? | Uses Clinical? | Uses Genomic? | Architecture |
|-------|-------------------|----------------|---------------|--------------|
| **Graph Transformer** | ✅ Yes | ✅ Yes | ✅ Yes | Transformer with STRING PPI bias |
| **Vanilla Transformer** | ❌ No | ✅ Yes | ✅ Yes | Standard Transformer (no graph) |
| **MLP Baseline** | ❌ No | ✅ Yes | ✅ Yes | Simple feedforward network |
| **Protein-only MLP** | ❌ No | ❌ No | ❌ No | Protein expression only |

### Key Differences

**Graph Transformer (Ours)**:
- Uses STRING protein-protein interaction network
- Graph bias in attention: `attention[i,j] += scale * DiffusionKernel[i,j]`
- Learns coordinated pathway effects

**Baselines**:
- Treat proteins as independent features
- No network structure
- Standard molecular screening approach

## Dataset

- **Source**: TCGA Pan-Cancer RPPA
- **Samples**: 7,201 patients (32 cancer types)
- **Features**:
  - 198 proteins (RPPA)
  - 7 clinical variables
  - 4 genomic features
- **Target**: Disease-Specific Survival (DSS)
  - 1,601 events (22.2% event rate)
  - Median survival: 24.3 months
- **Split**: 80% train, 10% val, 10% test
- **Same random seed** for all models (fair comparison)

## Hyperparameters

All models trained with identical settings:
```python
learning_rate = 5e-4
weight_decay = 2e-4
dropout = 0.4
batch_size = 64
max_epochs = 50
patience = 10  # early stopping
scheduler = ReduceLROnPlateau
```

## Evaluation Metrics

### Primary: C-index (Concordance Index)
- Measures fraction of correctly ordered patient pairs
- **0.50**: Random predictions
- **0.60-0.70**: Acceptable
- **0.70-0.80**: Good
- **0.80-0.90**: Excellent
- **0.90-1.00**: Outstanding

### Secondary:
- Cox partial likelihood loss
- Train-test generalization gap
- Model size (parameters)

## Expected Outcomes

### Hypothesis 1: PPI Topology Improves Prediction
If **TRUE**, we expect:
- Graph Transformer C-index > Baseline C-index
- Improvement: **+3-8%** absolute
- Statistical significance: p < 0.01

**Biological Rationale**:
- Cancer affects signaling networks, not just individual proteins
- Pathway dysregulation (e.g., TP53, PI3K/AKT) better captured by topology
- Coordinated protein expression patterns are prognostic

### Hypothesis 2: Clinical/Genomic Features Add Value
If **TRUE**, we expect:
- MLP (all features) > Protein-only MLP
- Improvement: **+2-5%**

### Hypothesis 3: Transformer Architecture Helps
If **TRUE**, we expect:
- Transformers (vanilla & graph) > MLP
- Self-attention captures protein interactions better than feedforward

## Interpretation Guidelines

### Scenario 1: Strong Improvement (Graph > Baseline by >5%)
**Conclusion**: ✅ **YES**, topology significantly improves survival prediction
- Biological networks matter for prognosis
- Graph-based methods should be standard

### Scenario 2: Moderate Improvement (Graph > Baseline by 2-5%)
**Conclusion**: ⚠️ **MODERATE** benefit from topology
- Some value but not transformative
- May depend on cancer type

### Scenario 3: Minimal Improvement (Graph > Baseline by <2%)
**Conclusion**: ❌ **NO** significant benefit
- Standard screening sufficient
- PPI topology doesn't add predictive value

## Files Generated

### Training Outputs
```
outputs/
├── survival/                  # Graph Transformer results
│   ├── checkpoints/
│   │   └── best_model.pt
│   └── results/
│       ├── test_results.json
│       └── training_history.json
│
├── baselines/                 # Baseline model results
│   ├── mlp_baseline_best.pt
│   ├── mlp_baseline_results.json
│   ├── vanilla_transformer_best.pt
│   ├── vanilla_transformer_results.json
│   ├── protein_only_best.pt
│   ├── protein_only_results.json
│   └── baseline_summary.json
│
└── comparison/                # Statistical comparison
    ├── model_comparison.json
    ├── model_comparison.png
    └── relative_improvement.png
```

### Key Results Files

**test_results.json**: Final test performance
```json
{
  "test_c_index": 0.7399,
  "best_val_c_index": 0.7885,
  "n_params": 1674889,
  "n_events": 160
}
```

**baseline_summary.json**: All baseline performances

**model_comparison.json**: Side-by-side comparison with statistics

## How to Run

### Full Pipeline
```bash
cd "Survival Classifier"
./run_comparison.sh
```

### Individual Steps

**Step 1: Train baselines** (takes ~2-3 hours)
```bash
python train_baselines.py \
  --csv_path ../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../priors/tcga_string_prior.npz \
  --output_dir outputs/baselines \
  --device mps
```

**Step 2: Compare models**
```bash
python compare_models.py \
  --results_dir outputs \
  --output_dir outputs/comparison
```

## Statistical Analysis

The comparison script computes:

1. **Absolute improvement**: Δ C-index
2. **Relative improvement**: (Graph - Baseline) / Baseline × 100%
3. **Effect size**: Cohen's d
   - Small: d < 0.5
   - Medium: 0.5 ≤ d < 0.8
   - Large: d ≥ 0.8

4. **Confidence intervals**: Bootstrap (10,000 samples)

## Visualization

### Generated Plots

1. **model_comparison.png**: Bar chart of C-index across models
   - Green bars: Graph Transformer (with topology)
   - Blue bars: Baselines (no topology)
   - Shows test vs validation C-index

2. **relative_improvement.png**: % improvement over best baseline
   - Positive bars: Better than baseline
   - Negative bars: Worse than baseline

## Current Results Summary

### Graph Transformer (v3)
- **Test C-index**: 0.7399 (Good)
- **Val C-index**: 0.7885 (Good, close to Excellent)
- **Parameters**: 1.67M
- **Epochs**: 18 (early stopped)
- **Train-val gap**: 0.074 (well-controlled overfitting)

### Baseline Results
*To be filled after training completes*

## References

1. STRING database: Protein-protein interaction networks
2. TCGA RPPA: Reverse Phase Protein Array data
3. Cox Proportional Hazards: Survival analysis framework
4. C-index: Harrell et al., Statistics in Medicine, 1996

## Contact

For questions about this comparison study, see:
- Model implementations: `baseline_models.py`, `graph_transformer_survival_classifier.py`
- Training code: `train_baselines.py`, `train_and_eval.py`
- Analysis code: `compare_models.py`
