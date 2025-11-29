import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class DataLoader:
    """Load and preprocess proteomics data"""
    
    def __init__(self, string_file, hpa_file, min_score=700):
        self.string_file = string_file
        self.hpa_file = hpa_file
        self.min_score = min_score
        
    def load(self):
        """Load and preprocess data into PyTorch Geometric format"""
        # Load STRING interactions
        interactions = pd.read_csv(self.string_file, sep='\t')
        interactions = interactions[interactions['combined_score'] >= self.min_score]
        proteins = sorted(set(interactions['protein1'].tolist() + interactions['protein2'].tolist()))
        
        # Build adjacency matrix
        n = len(proteins)
        protein_idx = {p: i for i, p in enumerate(proteins)}
        adj = np.zeros((n, n))
        
        for _, row in interactions.iterrows():
            if row['protein1'] in protein_idx and row['protein2'] in protein_idx:
                i, j = protein_idx[row['protein1']], protein_idx[row['protein2']]
                score = row['combined_score'] / 1000.0
                adj[i, j] = score
                adj[j, i] = score
        
        # Load HPA features
        hpa_data = pd.read_csv(self.hpa_file)
        hpa_filtered = hpa_data[hpa_data['Gene names'].isin(proteins)].copy()
        
        feature_cols = [col for col in hpa_data.columns 
                       if col not in ['Gene names', 'Protein names', 'Ensembl', 'UniProt']]
        features = hpa_filtered[feature_cols].fillna(0).values
        features = StandardScaler().fit_transform(features)
        
        # Create graph data
        x = torch.tensor(features, dtype=torch.float32)
        rows, cols = np.nonzero(adj)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        adj_tensor = torch.tensor(adj, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, adj=adj_tensor)
        
        print(f"Loaded {len(proteins)} proteins with {len(interactions)} interactions")
        print(f"Features: {features.shape[1]} dimensions")
        
        return data
