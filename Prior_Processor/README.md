# Prior Processor

Central repository for protein-protein interaction (PPI) network priors and prior generation scripts.

## Overview

This directory consolidates all graph prior processing functionality:
- **Building PPI networks** from STRING database
- **Storing precomputed priors** for use across all models
- **Documentation** for prior generation process

## Directory Structure

```
Prior_Processor/
├── data/                           # Precomputed graph priors
│   └── tcga_string_prior.npz      # STRING PPI network for TCGA proteins
├── scripts/                        # Prior generation scripts
│   └── build_string_prior.py      # Build STRING prior from CSV
└── README.md
```

## Quick Start

### Using Existing Priors

All models in the repository reference priors via the `priors/` symlink:
```python
prior_path = "../priors/tcga_string_prior.npz"
```

The symlink `priors/` → `Prior_Processor/data/` ensures backward compatibility.

### Building New Priors

To generate a new STRING prior from your proteomics data:

```bash
cd Prior_Processor/scripts

python build_string_prior.py \
  --csv ../../processed_datasets/tcga_pancan_rppa_compiled.csv \
  --out ../data/tcga
```

**Output:** `Prior_Processor/data/tcga_string_prior.npz`

## STRING Prior Format

The `.npz` file contains:

```python
import numpy as np
prior = np.load('data/tcga_string_prior.npz', allow_pickle=True)

# Contents:
# - A: Adjacency matrix (N × N, float32)
#      Symmetric matrix of PPI weights from STRING
# - protein_cols: Array of protein column names from input CSV
# - genes: Array of gene symbols extracted from protein names
```

### Adjacency Matrix Details

- **Size:** N × N where N = number of proteins
- **Values:** STRING combined confidence scores (0-1000 scale)
- **Symmetry:** A[i,j] = A[j,i] for all i,j
- **Evidence filtering:** Only edges with experimental OR database evidence
- **Sparsity:** Typical density ~1-5% (most proteins unconnected)

## Building Custom Priors

### Requirements

```bash
pip install numpy requests
```

### Usage

```bash
python scripts/build_string_prior.py \
  --csv <path_to_proteomics_csv> \
  --out <output_prefix>
```

**Arguments:**
- `--csv`: Path to CSV with protein columns (format: `GENE|UNIPROT`)
- `--out`: Output prefix (e.g., `data/custom` → `data/custom_string_prior.npz`)

### What It Does

1. **Extract genes** from protein column names
2. **Map to STRING IDs** via STRING API
3. **Query PPI network** (confidence threshold: 700/1000)
4. **Filter edges** to only experimental + database evidence
5. **Build adjacency matrix** aligned with protein columns
6. **Save to .npz** format

### Customization Options

Edit `build_string_prior.py` to change:
- `required_score=700` → Minimum confidence (400=medium, 900=highest)
- `species=9606` → Taxonomy ID (9606 = human)
- Evidence filtering logic (lines 228-230)

## Evidence Types in STRING

The script filters to keep only high-quality evidence:

| Evidence Type | Code | Kept? |
|---------------|------|-------|
| Experimental | escore | ✅ Yes |
| Database | dscore | ✅ Yes |
| Text mining | tscore | ❌ No |
| Co-expression | ascore | ❌ No |
| Neighborhood | nscore | ❌ No |
| Fusion | fscore | ❌ No |
| Co-occurrence | pscore | ❌ No |

This ensures only experimentally validated or curated interactions are included.

## Using Priors in Models

All three model types use the same prior loading pattern:

```python
from classifier.data.graph_prior import load_graph_prior

# Load prior
prior = load_graph_prior("../priors/tcga_string_prior.npz")

# Extract components
A = prior['A']              # Adjacency matrix
K = prior['K']              # Diffusion kernel
PE = prior['PE']            # Positional encodings
protein_cols = prior['protein_cols']
genes = prior['genes']
```

The `load_graph_prior` function automatically computes:
- **Graph Laplacian** (normalized)
- **Diffusion kernel:** K = exp(-β L)
- **Positional encodings:** Top k eigenvectors of L

## Migration Notes

This directory consolidates:
- **Old:** `priors/` (data only)
- **Old:** `getting_prior/` (script only)
- **New:** `Prior_Processor/` (data + scripts + docs)

**Backward Compatibility:**
- The `priors/` symlink ensures all existing code works unchanged
- The `getting_prior/` directory can be safely removed after migration

## Example: TCGA STRING Prior

**Current prior statistics:**
- Proteins: 198
- Mapped to STRING: ~195-198
- Network edges: ~1,500-2,000
- Edge density: ~8-10%
- Evidence: Experimental + Database only

This represents experimentally validated protein-protein interactions from the STRING database, filtered for high confidence (score ≥ 700) and reliable evidence types.

## Troubleshooting

**"No genes could be mapped to STRING IDs"**
- Check gene symbol format in CSV columns
- Verify internet connection (STRING API required)
- Try different species ID if not human

**"STRING returned no edges"**
- Lower `required_score` (e.g., 400 for medium confidence)
- Check that genes are actually known to interact
- Verify species is correct

**"Unmapped genes"**
- Normal to have 1-5% unmapped
- Often due to deprecated gene symbols
- Check STRING database for correct nomenclature

## References

- STRING Database: https://string-db.org
- STRING API: https://string-db.org/help/api/
- TCGA RPPA Data: https://www.mdanderson.org/research/research-resources/core-facilities/functional-proteomics-rppa-core.html
