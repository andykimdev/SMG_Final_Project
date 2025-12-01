#!/usr/bin/env python3
"""
Build a STRING-based prior adjacency matrix for proteomics data.

This script queries the STRING database to construct a protein-protein
interaction network based on gene symbols extracted from proteomics data.
"""

import argparse
import csv
import sys
import time
import requests
import numpy as np
from typing import List, Dict, Tuple


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build STRING-based prior adjacency matrix for proteomics data"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing proteomics data"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output prefix (e.g., data/tcga)"
    )
    return parser.parse_args()


def read_protein_columns(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Read the header row and extract protein columns and gene symbols.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (protein_cols, genes) where protein_cols are the original
        column names and genes are the extracted gene symbols
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    protein_cols = []
    genes = []
    multi_gene_count = 0
    
    for col in header:
        if '|' in col:
            protein_cols.append(col)
            # Extract gene symbol: part before | and before any spaces
            gene_part = col.split('|')[0].strip()
            
            # If multiple genes (space-separated), take the first one
            if ' ' in gene_part:
                gene = gene_part.split()[0].strip()
                multi_gene_count += 1
            else:
                gene = gene_part
            
            genes.append(gene)
    
    if multi_gene_count > 0:
        print(f"Note: {multi_gene_count} columns contain multiple genes; using first gene only")
    
    return protein_cols, genes


def map_genes_to_string(genes: List[str], species: int = 9606) -> Dict[str, str]:
    """
    Map gene symbols to STRING identifiers using the STRING API.
    
    Args:
        genes: List of gene symbols
        species: NCBI taxonomy ID (default: 9606 for human)
        
    Returns:
        Dictionary mapping gene symbols to STRING IDs (best match only)
    """
    string_api_url = "https://string-db.org/api"
    output_format = "tsv"
    method = "get_string_ids"
    
    # STRING API expects newline-separated identifiers
    params = {
        "identifiers": "\r".join(genes),
        "species": species,
        "limit": 1,  # Keep only best match
        "echo_query": 1,
    }
    
    request_url = f"{string_api_url}/{output_format}/{method}"
    
    print(f"Mapping {len(genes)} genes to STRING IDs...")
    response = requests.post(request_url, data=params)
    
    if response.status_code != 200:
        raise RuntimeError(f"STRING API request failed with status {response.status_code}")
    
    # Parse TSV response
    # Format: queryItem, queryIndex, stringId, ncbiTaxonId, taxonName, preferredName, annotation
    gene_to_string_id = {}
    lines = response.text.strip().split('\n')
    
    if len(lines) > 1:  # Skip if only header
        header = lines[0].split('\t')
        for line in lines[1:]:
            fields = line.split('\t')
            if len(fields) >= 3:
                query_gene = fields[0]
                string_id = fields[2]  # stringId is the 3rd column (index 2)
                # Only keep first (best) match per gene
                if query_gene not in gene_to_string_id:
                    gene_to_string_id[query_gene] = string_id
    
    return gene_to_string_id


def get_string_network(string_ids: List[str], required_score: int = 700,
                       species: int = 9606) -> List[Dict]:
    """
    Query the STRING network for protein-protein interactions.
    
    Args:
        string_ids: List of STRING identifiers
        required_score: Minimum confidence score (0-1000, default: 700)
        species: NCBI taxonomy ID (default: 9606 for human)
        
    Returns:
        List of interaction dictionaries with keys like 'stringId_A', 
        'stringId_B', 'score'
    """
    string_api_url = "https://string-db.org/api"
    output_format = "tsv"
    method = "network"
    
    params = {
        "identifiers": "%0d".join(string_ids),
        "species": species,
        "required_score": required_score,
        "network_type": "functional",
        "add_nodes": 0,
    }
    
    request_url = f"{string_api_url}/{output_format}/{method}"
    
    print(f"Querying STRING network for {len(string_ids)} proteins (required_score={required_score})...")
    response = requests.post(request_url, data=params)
    
    if response.status_code != 200:
        raise RuntimeError(f"STRING network API request failed with status {response.status_code}")
    
    # Parse TSV response
    interactions = []
    lines = response.text.strip().split('\n')
    
    if len(lines) <= 1:  # Only header or empty
        return interactions
    
    header = lines[0].split('\t')
    for line in lines[1:]:
        fields = line.split('\t')
        interaction = {}
        for i, key in enumerate(header):
            if i < len(fields):
                interaction[key] = fields[i]
        interactions.append(interaction)
    
    return interactions


def build_adjacency_matrix(protein_cols: List[str], genes: List[str],
                          gene_to_string_id: Dict[str, str],
                          interactions: List[Dict]) -> Tuple[np.ndarray, int, int]:
    """
    Build adjacency matrix from STRING interactions.
    Only includes edges with experimental or database evidence.
    
    Args:
        protein_cols: Original protein column names
        genes: Gene symbols in same order as protein_cols
        gene_to_string_id: Mapping from gene symbols to STRING IDs
        interactions: List of interaction dictionaries from STRING
        
    Returns:
        Tuple of (adjacency_matrix, edges_kept, edges_filtered)
    """
    N = len(protein_cols)
    A = np.zeros((N, N), dtype=np.float32)
    
    # Build reverse mapping: STRING ID -> index
    # Also create a mapping from preferredName to index for fallback
    string_id_to_idx = {}
    preferred_name_to_idx = {}
    
    for i, gene in enumerate(genes):
        if gene in gene_to_string_id:
            string_id = gene_to_string_id[gene]
            string_id_to_idx[string_id] = i
            # Also map the gene name for fallback
            preferred_name_to_idx[gene.upper()] = i
    
    # Process interactions
    edge_count = 0
    filtered_count = 0
    
    for interaction in interactions:
        # Try multiple field names that STRING might use
        string_id_a = interaction.get('stringId_A', interaction.get('ncbiTaxonId', ''))
        string_id_b = interaction.get('stringId_B', interaction.get('ncbiTaxonId', ''))
        preferred_a = interaction.get('preferredName_A', '').upper()
        preferred_b = interaction.get('preferredName_B', '').upper()
        
        # Get individual evidence scores
        escore = float(interaction.get('escore', 0))  # Experimental evidence
        dscore = float(interaction.get('dscore', 0))  # Database evidence
        
        # FILTER: Only keep edges with experimental OR database evidence
        if escore == 0 and dscore == 0:
            filtered_count += 1
            continue
        
        # Use combined score as weight
        score = float(interaction.get('score', 0))
        weight = score
        
        # Try to find indices using STRING IDs first, then preferred names
        idx_a = string_id_to_idx.get(string_id_a, preferred_name_to_idx.get(preferred_a, None))
        idx_b = string_id_to_idx.get(string_id_b, preferred_name_to_idx.get(preferred_b, None))
        
        if idx_a is not None and idx_b is not None:
            # Keep maximum weight if multiple edges
            A[idx_a, idx_b] = max(A[idx_a, idx_b], weight)
            A[idx_b, idx_a] = max(A[idx_b, idx_a], weight)
            edge_count += 1
    
    return A, edge_count, filtered_count


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}\n")
    
    # Read protein columns and extract gene symbols
    print(f"Reading protein columns from {args.csv}...")
    protein_cols, genes = read_protein_columns(args.csv)
    print(f"Found {len(protein_cols)} protein columns")
    
    if len(protein_cols) == 0:
        print("ERROR: No protein columns found (columns with '|' in the name)")
        sys.exit(1)
    
    # Map genes to STRING IDs
    gene_to_string_id = map_genes_to_string(genes, species=9606)
    mapped_count = len(gene_to_string_id)
    print(f"Successfully mapped {mapped_count}/{len(genes)} genes to STRING IDs")
    
    # Print unmapped genes for debugging
    unmapped = [g for g in genes if g not in gene_to_string_id]
    if unmapped and len(unmapped) <= 10:
        print(f"Unmapped genes: {', '.join(unmapped)}")
    elif unmapped:
        print(f"Unmapped genes (showing first 10): {', '.join(unmapped[:10])}")
    
    if mapped_count == 0:
        print("ERROR: No genes could be mapped to STRING IDs")
        sys.exit(1)
    
    # Get STRING network
    string_ids = list(gene_to_string_id.values())
    interactions = get_string_network(string_ids, required_score=700, species=9606)
    print(f"Retrieved {len(interactions)} interactions from STRING")
    
    if len(interactions) == 0:
        print("ERROR: STRING returned no edges for these proteins at required_score=700.")
        print("Try lowering the required_score parameter (e.g., 400 for medium confidence)")
        sys.exit(1)
    
    # Build adjacency matrix (filtering for experimental/database evidence only)
    print("Building adjacency matrix (experimental + database evidence only)...")
    A, edges_kept, edges_filtered = build_adjacency_matrix(protein_cols, genes, gene_to_string_id, interactions)
    
    # Count non-zero edges
    num_edges = np.sum(A > 0) // 2  # Divide by 2 because matrix is symmetric
    print(f"Kept {edges_kept} edges with experimental/database evidence")
    print(f"Filtered {edges_filtered} edges (text-mining/prediction only)")
    print(f"Adjacency matrix built: {len(protein_cols)}×{len(protein_cols)} with {num_edges} edges")
    
    # Save to compressed npz file
    output_path = f"{args.out}_string_prior.npz"
    print(f"Saving to {output_path}...")
    
    np.savez_compressed(
        output_path,
        A=A,
        protein_cols=np.array(protein_cols, dtype=object),
        genes=np.array(genes, dtype=object)
    )
    
    print(f"Done! Saved to {output_path}")
    print(f"\nSummary:")
    print(f"  - Protein features: {len(protein_cols)}")
    print(f"  - Mapped to STRING: {mapped_count}")
    print(f"  - Network edges: {num_edges}")
    print(f"  - Adjacency matrix shape: {A.shape}")
    print(f"  - Edge density: {num_edges / (len(protein_cols) * (len(protein_cols) - 1) / 2):.4f}")


if __name__ == "__main__":
    main()

