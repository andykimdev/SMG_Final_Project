# Data Directory

This directory contains the input data files required for training and evaluation.

## Required Files

### 1. Processed TCGA RPPA Dataset
**Location**: `processed_datasets/tcga_pancan_rppa_compiled.csv`

**Format**: CSV file with:
- Rows: Samples (patients)
- Columns: 
  - First column: Sample ID or cancer type label
  - Remaining columns: Protein expression values (normalized)
- Protein column names should match those in the PPI prior

**Schema**:
```
sample_id,cancer_type,PROTEIN1|GENE1,PROTEIN2|GENE2,...
TCGA-XX-XXXX,BRCA,0.5,1.2,...
```

### 2. STRING PPI Prior
**Location**: `priors/tcga_string_prior.npz`

**Format**: NumPy compressed archive containing:
- `A`: Adjacency matrix (N x N, float32)
- `protein_cols`: List of protein names matching CSV columns
- `genes`: List of gene names (optional)

**Generation**: Use `src/graph_prior.py` to generate from STRING network data.

## Alternative Locations

The code will also check:
- `data/tcga_pancan_rppa_compiled.csv` (directly in data/)
- `data/tcga_string_prior.npz` (directly in data/)

## Data Sources

- **TCGA RPPA Data**: Available from TCGA data portal or GDC
- **STRING Network**: Available from STRING database (string-db.org)

## Notes

- Data files are not included in this repository due to size
- Users must obtain data separately
- See project README for data processing instructions

