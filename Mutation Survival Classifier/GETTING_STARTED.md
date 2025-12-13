# Getting Started with Mutation Survival Classifier

## Quick Start

### 1. Test the installation

```bash
cd "/Users/andykim/Documents/2025 Fall/SMG/Project/SMG_Final_Project/Mutation Survival Classifier"

# Test config
python3 -c "import sys; sys.path.insert(0, '.'); from mutation_survival_classifier import config; print('✓ Config OK')"

# Test data loader
python3 -c "import sys; sys.path.insert(0, '.'); from mutation_survival_classifier.data import load_graph_prior; print('✓ Data module OK')"

# Test model
python3 -c "import sys; sys.path.insert(0, '.'); from mutation_survival_classifier.models import SurvivalGraphTransformer; print('✓ Model module OK')"
```

### 2. Run training (full model with mutations + cancer type)

```bash
cd scripts

python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/full_model" \
  --device mps \
  --use_mutations \
  --use_cancer_type \
  --max_epochs 50
```

**Expected time**: ~15-20 minutes on MPS (Apple Silicon)

**Expected performance**:
- Validation C-index: 0.78-0.82
- Test C-index: 0.78-0.81

### 3. Run baseline comparisons

**A. Blinded model (no mutations, no cancer type)**

```bash
python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/blinded" \
  --device mps \
  --no_mutations \
  --no_cancer_type
```

Expected C-index: 0.72-0.74

**B. Mutations only (no cancer type)**

```bash
python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/mutations_only" \
  --device mps \
  --use_mutations \
  --no_cancer_type
```

Expected C-index: 0.73-0.76

**C. Cancer type only (no mutations)**

```bash
python3 train_and_eval.py \
  --csv_path "../../processed_datasets/tcga_pancan_rppa_with_mutations.csv" \
  --prior_path "../../priors/tcga_string_prior.npz" \
  --output_dir "../outputs/cancer_type_only" \
  --device mps \
  --no_mutations \
  --use_cancer_type
```

Expected C-index: 0.78-0.80

## Understanding the Results

### Output Files

After training, check `outputs/full_model/run_YYYYMMDD_HHMMSS/`:

```bash
# View results
cat results.json

# Key metrics:
# - test_c_index: Primary metric (0.78-0.82 expected)
# - test_loss: Cox loss value
# - best_val_c_index: Best validation performance
```

### Results JSON Structure

```json
{
  "test_loss": 2.82,
  "test_c_index": 0.795,
  "best_val_c_index": 0.802,
  "final_epoch": 35,
  "preprocessing_info": {
    "protein_features": 198,
    "clinical_features": 7,    // includes cancer_type if enabled
    "genomic_features": 59,    // 4 continuous + 55 mutations
    "use_cancer_type": true,
    "use_mutations": true
  }
}
```

### Interpreting C-Index

- **0.78-0.82**: Model correctly ranks 78-82% of patient pairs by survival
- **Baseline PCA+Cox**: 0.73 (from previous project)
- **Improvement**: +5-9% over simple linear model

## Experimental Designs

### Experiment 1: Does mutation data help?

Compare:
1. Full model (mutations + cancer type): Expected ~0.80
2. No mutations (cancer type only): Expected ~0.78

**If mutations help**: C-index increases by +0.02-0.04
**If mutations don't help**: C-index stays same or decreases

### Experiment 2: Blinded vs Unblinded

Compare:
1. Blinded (no mutations, no cancer type): Expected ~0.73
2. Unblinded (mutations + cancer type): Expected ~0.80

**Shows**: Value of clinical context (cancer type) + genomic features

### Experiment 3: Mutations vs Cancer Type

Compare:
1. Mutations only (no cancer type): Expected ~0.74
2. Cancer type only (no mutations): Expected ~0.78

**Shows**: Which feature is more predictive

## Dataset Statistics

- **Total samples**: 7,523 (after filtering)
- **Training**: ~6,018 samples
- **Validation**: ~753 samples
- **Test**: ~753 samples
- **Event rate**: ~35% (deaths)
- **Median survival**: ~30 months

### Mutation Sparsity

Most mutation features are **sparse**:
- TP53: ~30% mutated (most common)
- KRAS: ~8% mutated
- PIK3CA: ~12% mutated
- Most others: <5% mutated

**Challenge**: Model must learn from sparse signals.

## Troubleshooting

### Issue: "No module named mutation_survival_classifier"

```bash
# Make sure you're in the correct directory
cd "/Users/andykim/Documents/2025 Fall/SMG/Project/SMG_Final_Project/Mutation Survival Classifier"

# Check if __init__.py exists
ls mutation_survival_classifier/__init__.py
```

### Issue: "CUDA out of memory" (if using GPU)

```bash
# Reduce batch size in config.py
# Change batch_size from 64 to 32 or 16
```

### Issue: Training is very slow

```bash
# Use fewer workers
--num_workers 0

# Or reduce data size for quick test
# Edit config.py: max_epochs = 5
```

### Issue: C-index is 0.5 (random)

This means model isn't learning. Check:
1. Are there enough events? (Need ~50+ deaths)
2. Is survival time variance sufficient?
3. Try increasing learning rate: `config.TRAINING['learning_rate'] = 0.001`

## Next Steps

### 1. Analyze mutation contributions

After training, you can:
- Load model checkpoint
- Extract attention weights
- See which mutations the model attends to
- Identify predictive mutation patterns

### 2. Stratify by cancer type

Like your previous per-cancer analysis:
- Evaluate model separately per cancer type
- See if mutations help differently across cancers
- Example: Mutations may help in BRCA but not THCA

### 3. Compare to your previous results

From `Survival Classifier/RESULTS_SUMMARY.md`:
- Graph transformer (unblinded): 0.780
- PCA 50 + Cox: 0.732
- Per-cancer: Graph fails catastrophically

**Question**: Do mutations change the per-cancer story?

## Expected Workflow

```bash
# 1. Train full model (30 min)
python3 train_and_eval.py --csv_path ... --use_mutations --use_cancer_type

# 2. Check results
cat ../outputs/full_model/run_*/results.json

# 3. Train baselines for comparison (30 min each)
python3 train_and_eval.py --csv_path ... --no_mutations --use_cancer_type
python3 train_and_eval.py --csv_path ... --use_mutations --no_cancer_type

# 4. Compare C-indices
# Full: 0.80?
# No mutations: 0.78?
# No cancer type: 0.74?

# 5. Interpret:
# - Do mutations help? (Full vs No mutations)
# - How much does cancer type matter? (Cancer type vs Blinded)
# - Is graph better than linear? (Compare to PCA+Cox baseline: 0.73)
```

## Questions to Answer

1. **Do mutations improve survival prediction beyond protein expression?**
   - Compare: mutations+cancer_type vs cancer_type_only

2. **Can mutations compensate for lack of cancer type?**
   - Compare: mutations_only vs blinded

3. **Which mutations are most predictive?**
   - Extract attention weights from trained model
   - Check which MUT_* features have high weights

4. **Does graph structure help with mutations?**
   - Current model: Graph transformer
   - Future: Compare to MLP with mutations

Good luck! 🚀
