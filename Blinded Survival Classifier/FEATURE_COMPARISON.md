# Feature Comparison: Blinded vs. Full Survival Classifier

## Overview

This document details the exact feature differences between the **Blinded** and **Full** survival classifiers.

## Feature Categories

### Protein Expression Features
| Feature Type | Blinded Model | Full Model |
|--------------|---------------|------------|
| RPPA protein expression | ✅ (~200-300 proteins) | ✅ (~200-300 proteins) |
| **Status** | **IDENTICAL** | **IDENTICAL** |

### Clinical Features (Numeric)
| Feature | Blinded | Full | Availability |
|---------|---------|------|--------------|
| AGE | ✅ | ✅ | Immediate |

### Clinical Features (Categorical)
| Feature | Blinded | Full | Rationale |
|---------|---------|------|-----------|
| SEX | ✅ | ✅ | Known at visit |
| RACE | ✅ | ✅ | Known at visit |
| GENETIC_ANCESTRY_LABEL | ✅ | ✅ | Known at visit |
| AJCC_PATHOLOGIC_TUMOR_STAGE | ❌ | ✅ | Requires staging workup |
| SAMPLE_TYPE | ❌ | ✅ | Technical metadata |
| PERSON_NEOPLASM_CANCER_STATUS | ❌ | ✅ | Post-diagnosis info |
| CANCER_TYPE_ACRONYM | ❌ | ✅ | Requires pathology report |

### Genomic Features
| Feature | Blinded | Full | Rationale |
|---------|---------|------|-----------|
| ANEUPLOIDY_SCORE | ❌ | ✅ | Requires genomic testing |
| TMB (Tumor Mutational Burden) | ❌ | ✅ | Requires NGS sequencing |
| MSI (Microsatellite Instability) | ❌ | ✅ | Requires genomic testing |
| Other genomic biomarkers | ❌ | ✅ | Not available at IHC visit |

## Feature Counts

### Blinded Model
```
Proteins:     ~200-300 (identical to full)
Clinical:     4 features (AGE + 3 categorical)
Genomic:      0 features (ALWAYS excluded)
TOTAL:        ~204-304 features
```

### Full Model
```
Proteins:     ~200-300 (identical to blinded)
Clinical:     ~7-8 features (AGE + 6-7 categorical)
Genomic:      ~3-5 features (TMB, MSI, aneuploidy, etc.)
TOTAL:        ~210-313 features
```

## Implementation Differences

### Dataset Loading (`dataset.py`)

**Blinded version changes:**
```python
# BLINDED: Only basic demographics available at IHC visit
clinical_numeric_candidates = ['AGE']
clinical_categorical_candidates = ['SEX', 'RACE', 'GENETIC_ANCESTRY_LABEL']

# BLINDED: NO genomic features
columns['genomic'] = []  # Always empty

# BLINDED: Exclude cancer-specific info
excluded_keywords = [
    'CANCER_TYPE_ACRONYM', 'ONCOTREE_CODE',
    'AJCC_PATHOLOGIC_TUMOR_STAGE', 'SAMPLE_TYPE',
    'PERSON_NEOPLASM_CANCER_STATUS',
]
```

**Full version:**
```python
# FULL: More clinical features
clinical_categorical_candidates = [
    'SEX', 'RACE', 'GENETIC_ANCESTRY_LABEL',
    'AJCC_PATHOLOGIC_TUMOR_STAGE', 'SAMPLE_TYPE',
    'PERSON_NEOPLASM_CANCER_STATUS'
]

# FULL: Include genomic features
genomic_keywords = ['ANEUPLOIDY', 'TMB', 'MSI', 'TBL']
columns['genomic'] = [col for col in df.columns
                     if any(kw in col for kw in genomic_keywords)]

# FULL: Include cancer type as feature
columns['clinical_categorical'].append('CANCER_TYPE_ACRONYM')
```

## Training Command Differences

### Blinded Model
```bash
python train_and_eval.py \
  --use_clinical \
  --no_genomic  # Key difference!
```

### Full Model
```bash
python train_and_eval.py \
  --use_clinical \
  --use_genomic  # Key difference!
```

## Clinical Interpretation

### What the Blinded Model Tests
1. **Can protein expression alone predict survival?**
   - Tests if IHC biomarkers are sufficient
   - No knowledge of cancer type or stage

2. **How much do genomics add?**
   - Comparison quantifies genomic contribution
   - Informs clinical testing priorities

3. **Early risk stratification**
   - Can we identify high-risk patients immediately?
   - Before expensive genomic tests

### Expected Performance Gap
- **Literature suggests:** 0.03-0.07 C-index difference
- **Protein patterns** may encode cancer type implicitly
- **Genomic features** add independent information

## Use Cases

### Blinded Model Best For:
- Resource-limited settings
- Emergency triage decisions
- Insurance pre-authorization
- Clinical trial screening

### Full Model Best For:
- Treatment planning
- Precision medicine decisions
- Research studies
- When genomic data available

## Key Insight

The blinded model answers: **"What can we predict at the bedside, before any expensive tests?"**

This is a **critically important clinical question** that standard ML survival models don't address.
