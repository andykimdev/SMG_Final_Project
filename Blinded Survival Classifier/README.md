# Blinded Survival Classifier

A graph transformer-based survival prediction model that uses **ONLY** features available at the time of initial IHC (immunohistochemistry) pathology visit, before genomic testing.

## Motivation

This "blinded" classifier addresses a critical clinical scenario: **Can we predict patient survival using only information available at initial diagnosis?**

In real clinical practice:
- IHC protein expression is available within days
- Patient demographics are immediately known
- Genomic testing (TMB, MSI, etc.) takes weeks and is expensive
- Cancer staging requires additional imaging and procedures

This model simulates **early-stage clinical decision making** where genomic information is not yet available.

## Features Used

### ✅ INCLUDED (Available at IHC visit)
- **Protein Expression** (~200-300 proteins from RPPA/IHC)
  - Growth factor receptors (EGFR, HER2, etc.)
  - Signaling proteins (AKT, ERK, STAT3, etc.)
  - Cell cycle regulators (p53, Rb, cyclin D1, etc.)
- **Age** - Patient age at diagnosis
- **Sex** - Patient biological sex
- **Race** - Patient race
- **Genetic Ancestry** - Genetic ancestry label

### ❌ EXCLUDED (Not available at IHC visit)
- **Cancer Type/Subtype** - Requires pathology report
- **Tumor Stage** (AJCC staging) - Requires imaging/surgery
- **Cancer Status** - Post-diagnosis information
- **Sample Type** - Technical metadata
- **Genomic Features**:
  - Tumor Mutational Burden (TMB)
  - Microsatellite Instability (MSI)
  - Aneuploidy Score
  - Other genomic biomarkers

## Key Differences from Full Model

| Feature | Full Model | Blinded Model |
|---------|-----------|---------------|
| Protein expression | ✅ | ✅ |
| Age, Sex, Race, Ancestry | ✅ | ✅ |
| Cancer type | ✅ | ❌ |
| Tumor stage | ✅ | ❌ |
| Genomic features | ✅ | ❌ |
| **Clinical scenario** | Post-workup | Initial visit |

## Clinical Significance

This blinded approach tests whether:
1. **Early prediction is possible** - Can we identify high-risk patients before full workup?
2. **Protein signatures are sufficient** - Do protein expression patterns contain enough survival signal?
3. **Genomic testing adds value** - Comparing blinded vs. full models quantifies genomic contribution

## Project Structure

```
Blinded Survival Classifier/
├── blinded_survival_classifier/  # Main package
│   ├── __init__.py
│   ├── config.py                # Hyperparameters
│   ├── models/                  # Model architectures
│   │   ├── baseline.py         # MLP, Vanilla Transformer
│   │   └── graph_transformer.py # Graph-aware transformer
│   └── data/                    # Data loading/preprocessing
│       ├── dataset.py          # BLINDED dataset (filters features)
│       └── graph_prior.py      # PPI network processing
├── scripts/                     # Training/evaluation scripts
│   ├── train_and_eval.py       # Main training script
│   ├── train_baselines.py      # Baseline models
│   ├── compare_models.py       # Model comparison
│   └── run_training.sh         # Convenience script
├── outputs/                     # Model outputs/results
└── README.md                    # This file
```

## Usage

### Training

```bash
cd scripts
./run_training.sh
```

Or manually:

```bash
python train_and_eval.py \
  --csv_path ../../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --prior_path ../../priors/tcga_string_prior.npz \
  --output_dir ../outputs \
  --device mps \
  --use_clinical \
  --no_genomic
```

**Important flags:**
- `--use_clinical` - Include Age, Sex, Race, Ancestry
- `--no_genomic` - Exclude all genomic features

### Comparing with Full Model

Train both models and compare:

```bash
# Blinded model (this folder)
cd "Blinded Survival Classifier/scripts"
python train_and_eval.py --use_clinical --no_genomic

# Full model (parent folder)
cd "../../Survival Classifier/scripts"
python train_and_eval.py --use_clinical --use_genomic

# Compare results
python compare_models.py --blinded ../Blinded\ Survival\ Classifier/outputs --full ../outputs
```

## Expected Performance

Based on clinical literature:
- **Protein-only models**: C-index ~0.60-0.65
- **With genomics**: C-index ~0.65-0.70
- **Graph transformer advantage**: +0.02-0.05 C-index over MLP

The blinded model may perform similarly to the full model if:
1. Protein expression captures most survival signal
2. Cancer type is implicitly encoded in protein patterns
3. Genomic features are redundant with protein state

## Model Architecture

Same as full model:
- Graph Transformer with PPI network structure
- 5 transformer layers, 8 attention heads
- 160-dim embeddings, strong dropout (0.45)
- Cox Proportional Hazards loss
- C-index evaluation metric

See [config.py](blinded_survival_classifier/config.py) for full hyperparameters.

## Data Preprocessing

The blinded dataset loader ([dataset.py](blinded_survival_classifier/data/dataset.py)) explicitly:
1. Identifies protein columns (contain `|`)
2. Allows only: Age, Sex, Race, Genetic_Ancestry_Label
3. **Excludes** all other clinical features
4. **Excludes** all genomic features
5. Prints clear warnings about excluded features

## Citation

If you use this blinded approach, please cite:
- Graph Transformer architecture: [Your publication]
- Clinical rationale: Early prediction without genomic testing
- STRING PPI network: Szklarczyk et al., Nucleic Acids Res. 2021

## License

[Your license]

## Contact

[Your contact]
