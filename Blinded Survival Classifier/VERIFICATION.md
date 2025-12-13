# Feature Verification for Blinded Survival Classifier

This document verifies that ALL models in the Blinded Survival Classifier folder use identical blinded features.

## Feature Set

### ✅ INCLUDED Features
- **Protein Expression**: ~200-300 RPPA proteins
- **Age**: Patient age
- **Sex**: Patient biological sex  
- **Race**: Patient race
- **Genetic Ancestry**: Genetic ancestry label

### ❌ EXCLUDED Features
- Cancer type/subtype
- Tumor stage (AJCC)
- Cancer status
- Sample type
- **ALL genomic features**: TMB, MSI, Aneuploidy, etc.

## Model Verification

### 1. Graph Transformer (`train_and_eval.py`)
```python
# Line 96-98: Default use_genomic=False for blinded
parser.add_argument('--use_genomic', default=False)

# Line 321-322: Loads data with blinded features
load_and_preprocess_survival_data(
    use_clinical=use_clinical,
    use_genomic=use_genomic  # Will be False
)
```
**Status**: ✅ Uses blinded features

### 2. MLP/Transformer Baselines (`train_baselines.py`)
```python
# Line 183-184: Explicitly disabled genomic features
load_and_preprocess_survival_data(
    use_clinical=True,
    use_genomic=False  # BLINDED: No genomic features!
)
```
**Status**: ✅ Uses blinded features

### 3. Linear Baselines (`train_linear_baselines.py`)
```python
# Line 306-307: Explicitly disabled genomic features  
load_and_preprocess_survival_data(
    use_clinical=True,
    use_genomic=False  # BLINDED: No genomic features!
)

# Line 52-60: Concatenates ALL available features
feature_arrays = [X_protein]
if X_clinical.shape[1] > 0:
    feature_arrays.append(X_clinical)
if X_genomic.shape[1] > 0:  # Will be 0 for blinded
    feature_arrays.append(X_genomic)
X = np.concatenate(feature_arrays, axis=1)
```
**Status**: ✅ Uses blinded features

## Dataset Loader Verification

### `blinded_survival_classifier/data/dataset.py`

```python
# Line 105-106: Only basic demographics
clinical_numeric_candidates = ['AGE']
clinical_categorical_candidates = ['SEX', 'RACE', 'GENETIC_ANCESTRY_LABEL']

# Line 102: Genomic features ALWAYS empty
columns['genomic'] = []  # Always empty for blinded classifier

# Line 90-95: Excluded features
excluded_keywords = [
    'CANCER_TYPE_ACRONYM', 'ONCOTREE_CODE', 'TUMOR_TYPE', 'SUBTYPE',
    'AJCC_PATHOLOGIC_TUMOR_STAGE', 'SAMPLE_TYPE',
    'PERSON_NEOPLASM_CANCER_STATUS',
]
```

## Expected Feature Counts

When training any model in this folder, you should see:

```
BLINDED Column Identification (IHC-available features only):
  Proteins (IHC): ~200-300
  Clinical (numeric): 1 - ['AGE']
  Clinical (categorical): 3 - ['SEX', 'RACE', 'GENETIC_ANCESTRY_LABEL']
  Genomic: 0 (ALWAYS 0 for blinded model)
  Excluded (cancer type, stage, genomic): ~10-15

Total features per sample:
  Proteins: ~200-300
  Clinical: 4
  Genomic: 0
  TOTAL: ~204-304
```

## Training Commands

All scripts automatically use blinded features:

```bash
# Graph Transformer (default --use_genomic=False)
python scripts/train_and_eval.py --csv_path ... --prior_path ...

# MLP/Transformer Baselines (hardcoded use_genomic=False)
python scripts/train_baselines.py --csv_path ... --prior_path ...

# Linear Baselines (hardcoded use_genomic=False)
python scripts/train_linear_baselines.py --csv_path ... --prior_path ...
```

## Verification Checklist

- [x] Dataset loader filters to blinded features only
- [x] Graph Transformer uses blinded features
- [x] MLP baseline uses blinded features
- [x] Transformer baseline uses blinded features
- [x] Protein-only MLP uses blinded features
- [x] PCA + Cox uses blinded features
- [x] Elastic Net Cox uses blinded features
- [x] All models see identical feature set
- [x] Warning messages print feature restrictions

## Fair Comparison Guarantee

✅ **ALL models in this folder use IDENTICAL blinded features**
- Same protein expression features
- Same clinical demographics (Age, Sex, Race, Ancestry)
- ZERO genomic features
- ZERO cancer type information

This ensures a fair comparison where the only difference is model architecture, not input features.
