"""
Graph prior processing module for STRING protein-protein interaction network.
Loads precomputed STRING prior and computes graph-based features.
"""

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import torch
from typing import Dict, Tuple
from .. import config


def load_graph_prior(prior_path: str) -> Dict:
    """
    Load precomputed STRING prior and compute graph features.
    
    Args:
        prior_path: Path to .npz file containing STRING prior
        
    Returns:
        Dictionary containing:
            - A: adjacency matrix (N×N)
            - K: diffusion kernel (N×N)
            - PE: graph positional encodings (N×k)
            - protein_cols: list of protein column names
            - genes: list of gene symbols
    """
    # Load the prior
    prior_data = np.load(prior_path, allow_pickle=True)
    
    A = prior_data['A'].astype(np.float32)
    protein_cols = prior_data['protein_cols'].tolist()
    genes = prior_data['genes'].tolist()
    
    N = A.shape[0]
    print(f"Loaded STRING prior: {N} proteins, {np.sum(A > 0) // 2} edges")
    
    # Compute graph Laplacian
    L, degrees = compute_laplacian(A, laplacian_type=config.GRAPH_PRIOR['laplacian_type'])
    
    # Compute diffusion kernel
    K = compute_diffusion_kernel(L, beta=config.GRAPH_PRIOR['diffusion_beta'])
    
    # Compute graph positional encodings
    PE = compute_positional_encodings(L, k=config.MODEL['pe_dim'])
    
    print(f"Computed diffusion kernel and {config.MODEL['pe_dim']} positional encodings")
    
    return {
        'A': A,
        'K': K,
        'PE': PE,
        'protein_cols': protein_cols,
        'genes': genes,
        'degrees': degrees,
    }


def compute_laplacian(A: np.ndarray, laplacian_type: str = 'normalized') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute graph Laplacian from adjacency matrix.
    
    Args:
        A: Adjacency matrix (N×N)
        laplacian_type: 'normalized' or 'symmetric'
        
    Returns:
        L: Laplacian matrix (N×N)
        degrees: Node degrees (N,)
    """
    # Compute degree matrix
    degrees = np.sum(A, axis=1)
    
    # Add small epsilon to avoid division by zero
    degrees_safe = degrees + 1e-8
    
    if laplacian_type == 'normalized':
        # L = I - D^{-1} A
        D_inv = np.diag(1.0 / degrees_safe)
        L = np.eye(A.shape[0]) - D_inv @ A
    elif laplacian_type == 'symmetric':
        # L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")
    
    return L.astype(np.float32), degrees.astype(np.float32)


def compute_diffusion_kernel(L: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Compute diffusion kernel K = exp(-beta * L).
    
    Args:
        L: Laplacian matrix (N×N)
        beta: Diffusion parameter
        
    Returns:
        K: Diffusion kernel (N×N)
    """
    # For efficiency, we use eigendecomposition: K = V exp(-beta * Lambda) V^T
    # However, for moderate-sized graphs (N=198), direct matrix exponential works
    from scipy.linalg import expm
    
    K = expm(-beta * L)
    
    # Ensure symmetry and non-negativity
    K = (K + K.T) / 2
    K = np.clip(K, 0, None)
    
    return K.astype(np.float32)


def compute_positional_encodings(L: np.ndarray, k: int = 16) -> np.ndarray:
    """
    Compute graph positional encodings as top k eigenvectors of Laplacian.
    
    Args:
        L: Laplacian matrix (N×N)
        k: Number of eigenvectors to use
        
    Returns:
        PE: Positional encodings (N×k)
    """
    N = L.shape[0]
    
    # For small k relative to N, use sparse eigensolver
    if k < N // 2:
        try:
            # Compute k smallest eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
            # Sort by eigenvalue (should already be sorted, but just to be safe)
            idx = np.argsort(eigenvalues)
            PE = eigenvectors[:, idx]
        except Exception as e:
            print(f"Warning: Sparse eigensolver failed ({e}), using dense solver")
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            PE = eigenvectors[:, :k]
    else:
        # Use dense eigensolver
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        PE = eigenvectors[:, :k]
    
    return PE.astype(np.float32)


def get_graph_features_as_tensors(graph_prior: Dict, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Convert graph features to PyTorch tensors.
    
    Args:
        graph_prior: Dictionary from load_graph_prior()
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Dictionary of tensors
    """
    return {
        'A': torch.from_numpy(graph_prior['A']).to(device),
        'K': torch.from_numpy(graph_prior['K']).to(device),
        'PE': torch.from_numpy(graph_prior['PE']).to(device),
    }
