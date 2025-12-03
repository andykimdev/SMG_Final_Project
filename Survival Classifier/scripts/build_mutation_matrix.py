#!/usr/bin/env python3
"""
Build a binary mutation matrix from MAF file and merge with RPPA data.

Converts long-format MAF (one row per mutation) to wide-format binary matrix
(rows=patients, columns=genes, values=0/1 for mutation presence).
"""

import pandas as pd
import numpy as np
import argparse
from typing import List, Set, Tuple
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Build mutation matrix from MAF file")
    parser.add_argument(
        "--maf",
        type=str,
        default="../processed_datasets/data_mutations.txt",
        help="Path to MAF file"
    )
    parser.add_argument(
        "--rppa_csv",
        type=str,
        default="../processed_datasets/tcga_pancan_rppa_compiled.csv",
        help="Path to RPPA CSV file"
    )
    parser.add_argument(
        "--genes",
        type=str,
        nargs="+",
        default=None,
        help="Specific genes to include (default: pathway genes)"
    )
    parser.add_argument(
        "--variant_types",
        type=str,
        nargs="+",
        default=["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
                 "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
                 "Splice_Site", "Translation_Start_Site"],
        help="Variant types to include (default: protein-altering variants)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../processed_datasets/mutation_matrix.csv",
        help="Output path for mutation matrix"
    )
    parser.add_argument(
        "--merged_output",
        type=str,
        default="../processed_datasets/tcga_pancan_rppa_with_mutations.csv",
        help="Output path for merged RPPA+mutations dataset"
    )
    return parser.parse_args()


def get_pathway_genes() -> dict:
    """
    Define key pathway genes for mutation-vs-signaling analysis.

    Returns:
        Dictionary mapping pathway names to gene lists
    """
    pathways = {
        'PI3K_AKT_MTOR': [
            'PIK3CA', 'PIK3R1', 'PIK3R2',  # PI3K complex
            'PTEN',  # Negative regulator
            'AKT1', 'AKT2', 'AKT3',  # AKT isoforms
            'MTOR', 'RPTOR', 'RICTOR',  # mTOR complex
            'TSC1', 'TSC2',  # Tuberous sclerosis complex
            'RPS6KB1', 'EIF4EBP1',  # Downstream targets
        ],
        'MAPK_ERK': [
            'KRAS', 'NRAS', 'HRAS',  # RAS family
            'BRAF', 'RAF1', 'ARAF',  # RAF family
            'MAP2K1', 'MAP2K2',  # MEK
            'MAPK1', 'MAPK3',  # ERK
            'NF1',  # Negative regulator
        ],
        'RTK_SIGNALING': [
            'EGFR', 'ERBB2', 'ERBB3', 'ERBB4',  # EGFR/HER family
            'MET', 'KIT', 'PDGFRA', 'FGFR1', 'FGFR2',  # Other RTKs
            'IGF1R', 'INSR',  # Insulin/IGF receptors
        ],
        'CELL_CYCLE': [
            'TP53', 'RB1',  # Tumor suppressors
            'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B',  # CDK inhibitors
            'CDK4', 'CDK6', 'CCND1', 'CCNE1',  # Cyclins/CDKs
        ],
        'WNT_BETA_CATENIN': [
            'CTNNB1', 'APC', 'AXIN1', 'AXIN2',  # Core pathway
            'GSK3A', 'GSK3B',  # Regulators
        ],
        'APOPTOSIS': [
            'BAX', 'BAK1', 'BCL2', 'BCL2L1',  # BCL2 family
            'CASP3', 'CASP7', 'CASP9',  # Caspases
            'PARP1',  # DNA repair/apoptosis
        ],
    }

    # Flatten to unique gene list
    all_genes = set()
    for genes in pathways.values():
        all_genes.update(genes)

    return pathways, sorted(list(all_genes))


def load_maf_file(maf_path: str, genes: List[str] = None,
                  variant_types: List[str] = None) -> pd.DataFrame:
    """
    Load MAF file and filter for relevant mutations.

    Args:
        maf_path: Path to MAF file
        genes: List of genes to include (None = all genes)
        variant_types: List of variant types to include

    Returns:
        DataFrame with columns: Hugo_Symbol, Tumor_Sample_Barcode, Variant_Classification
    """
    print(f"Loading MAF file from {maf_path}...")

    # Read only necessary columns to save memory
    cols_to_read = ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification']

    maf_df = pd.read_csv(
        maf_path,
        sep='\t',
        usecols=cols_to_read,
        low_memory=False
    )

    print(f"Loaded {len(maf_df):,} mutation events")
    print(f"  Unique genes: {maf_df['Hugo_Symbol'].nunique():,}")
    print(f"  Unique samples: {maf_df['Tumor_Sample_Barcode'].nunique():,}")

    # Filter for specific genes
    if genes is not None:
        maf_df = maf_df[maf_df['Hugo_Symbol'].isin(genes)]
        print(f"Filtered to {len(genes)} genes: {len(maf_df):,} mutations remain")

    # Filter for protein-altering variants
    if variant_types is not None:
        maf_df = maf_df[maf_df['Variant_Classification'].isin(variant_types)]
        print(f"Filtered to protein-altering variants: {len(maf_df):,} mutations remain")

    return maf_df


def build_mutation_matrix(maf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format MAF to binary mutation matrix.

    Args:
        maf_df: DataFrame with Hugo_Symbol and Tumor_Sample_Barcode columns

    Returns:
        Binary matrix (rows=samples, columns=genes, values=0/1)
    """
    print("\nBuilding binary mutation matrix...")

    # Create binary indicator (any mutation = 1)
    maf_df['has_mutation'] = 1

    # Pivot to wide format
    mutation_matrix = maf_df.pivot_table(
        index='Tumor_Sample_Barcode',
        columns='Hugo_Symbol',
        values='has_mutation',
        aggfunc='max',  # If multiple mutations in same gene, keep as 1
        fill_value=0
    ).astype(int)

    print(f"Mutation matrix shape: {mutation_matrix.shape}")
    print(f"  Samples: {len(mutation_matrix)}")
    print(f"  Genes: {len(mutation_matrix.columns)}")
    print(f"  Mutation rate: {mutation_matrix.sum().sum() / mutation_matrix.size * 100:.2f}%")

    # Show top mutated genes
    print("\nTop 10 mutated genes:")
    gene_counts = mutation_matrix.sum(axis=0).sort_values(ascending=False).head(10)
    for gene, count in gene_counts.items():
        pct = count / len(mutation_matrix) * 100
        print(f"  {gene}: {int(count)} ({pct:.1f}%)")

    return mutation_matrix


def match_sample_ids(tcga_barcode: str) -> str:
    """
    Convert TCGA sample barcode to patient ID.

    MAF uses sample barcodes like: TCGA-3C-AAAU-01
    RPPA uses patient IDs like: TCGA-3C-AAAU

    Args:
        tcga_barcode: Full TCGA sample barcode

    Returns:
        Patient ID (first 3 parts of barcode)
    """
    parts = tcga_barcode.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return tcga_barcode


def merge_with_rppa(mutation_matrix: pd.DataFrame, rppa_path: str) -> pd.DataFrame:
    """
    Merge mutation matrix with RPPA data by patient ID.

    Args:
        mutation_matrix: Binary mutation matrix (index=sample barcodes)
        rppa_path: Path to RPPA CSV file

    Returns:
        Merged DataFrame with RPPA features + mutation features
    """
    print(f"\nLoading RPPA data from {rppa_path}...")
    rppa_df = pd.read_csv(rppa_path)
    print(f"RPPA data: {len(rppa_df)} samples, {len(rppa_df.columns)} columns")

    # Convert MAF sample barcodes to patient IDs
    mutation_matrix.index = mutation_matrix.index.map(match_sample_ids)
    mutation_matrix = mutation_matrix.groupby(level=0).max()  # Merge multiple samples per patient

    # Rename mutation columns to avoid conflicts
    mutation_matrix.columns = [f"MUT_{gene}" for gene in mutation_matrix.columns]

    # Merge on PATIENT_ID
    if 'PATIENT_ID' not in rppa_df.columns:
        print("Warning: PATIENT_ID not in RPPA data, using index")
        rppa_df['PATIENT_ID'] = rppa_df.index

    merged_df = rppa_df.merge(
        mutation_matrix,
        left_on='PATIENT_ID',
        right_index=True,
        how='left'  # Keep all RPPA samples, even without mutation data
    )

    # Fill missing mutations with 0 (no mutation data = assume wild-type)
    mutation_cols = [col for col in merged_df.columns if col.startswith('MUT_')]
    merged_df[mutation_cols] = merged_df[mutation_cols].fillna(0).astype(int)

    print(f"\nMerged dataset: {len(merged_df)} samples, {len(merged_df.columns)} columns")
    print(f"  RPPA features: {len([c for c in merged_df.columns if '|' in c])}")
    print(f"  Mutation features: {len(mutation_cols)}")
    print(f"  Samples with mutation data: {(merged_df[mutation_cols].sum(axis=1) > 0).sum()}")

    return merged_df


def main():
    args = parse_args()

    # Get pathway genes
    pathways, pathway_genes = get_pathway_genes()

    print("="*80)
    print("Mutation Matrix Builder")
    print("="*80)
    print(f"\nDefined {len(pathways)} pathways with {len(pathway_genes)} total genes:")
    for pathway_name, genes in pathways.items():
        print(f"  {pathway_name}: {len(genes)} genes")

    # Use specified genes or pathway genes
    genes_to_include = args.genes if args.genes is not None else pathway_genes

    # Load and filter MAF file
    maf_df = load_maf_file(args.maf, genes=genes_to_include, variant_types=args.variant_types)

    # Build binary mutation matrix
    mutation_matrix = build_mutation_matrix(maf_df)

    # Save mutation matrix
    print(f"\nSaving mutation matrix to {args.output}...")
    mutation_matrix.to_csv(args.output)

    # Merge with RPPA data
    merged_df = merge_with_rppa(mutation_matrix, args.rppa_csv)

    # Save merged dataset
    print(f"Saving merged dataset to {args.merged_output}...")
    merged_df.to_csv(args.merged_output, index=False)

    # Save pathway definitions
    pathway_output = args.output.replace('.csv', '_pathways.json')
    print(f"Saving pathway definitions to {pathway_output}...")
    with open(pathway_output, 'w') as f:
        json.dump(pathways, f, indent=2)

    print("\n" + "="*80)
    print("SUCCESS! Mutation matrix and merged dataset created.")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  1. Mutation matrix: {args.output}")
    print(f"  2. Merged RPPA+mutations: {args.merged_output}")
    print(f"  3. Pathway definitions: {pathway_output}")


if __name__ == "__main__":
    main()
