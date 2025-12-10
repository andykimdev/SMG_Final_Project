"""Graph prior processing for STRING PPI network."""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import torch
from typing import Dict, Tuple
import config


def load_graph_prior(prior_path: str) -> Dict:
    """Load STRING prior and compute graph features."""
    prior_data = np.load(prior_path, allow_pickle=True)
    
    A = prior_data['A'].astype(np.float32)
    protein_cols = prior_data['protein_cols'].tolist()
    genes = prior_data['genes'].tolist()
    
    N = A.shape[0]
    n_edges = int(np.sum(A > 0) // 2)
    print(f"Loaded prior: {N} proteins, {n_edges} edges")
    
    L, degrees = compute_laplacian(A, config.GRAPH_PRIOR['laplacian_type'])
    K = compute_diffusion_kernel(L, config.GRAPH_PRIOR['diffusion_beta'])
    PE = compute_positional_encodings(L, config.GRAPH_PRIOR['pe_dim'])
    
    return {
        'A': A,
        'K': K,
        'PE': PE,
        'protein_cols': protein_cols,
        'genes': genes,
        'degrees': degrees,
    }


def compute_laplacian(A: np.ndarray, laplacian_type: str = 'normalized') -> Tuple[np.ndarray, np.ndarray]:
    """Compute graph Laplacian."""
    degrees = np.sum(A, axis=1)
    degrees_safe = degrees + 1e-8
    
    if laplacian_type == 'normalized':
        D_inv = np.diag(1.0 / degrees_safe)
        L = np.eye(A.shape[0]) - D_inv @ A
    elif laplacian_type == 'symmetric':
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")
    
    return L.astype(np.float32), degrees.astype(np.float32)


def compute_diffusion_kernel(L: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Compute diffusion kernel K = exp(-beta * L)."""
    K = expm(-beta * L)
    K = (K + K.T) / 2
    K = np.clip(K, 0, None)
    return K.astype(np.float32)


def compute_positional_encodings(L: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute positional encodings from Laplacian eigenvectors."""
    N = L.shape[0]
    
    if k < N // 2:
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
            idx = np.argsort(eigenvalues)
            PE = eigenvectors[:, idx]
        except:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            PE = eigenvectors[:, :k]
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        PE = eigenvectors[:, :k]
    
    return PE.astype(np.float32)


def get_graph_features_as_tensors(graph_prior: Dict, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Convert graph features to tensors."""
    return {
        'A': torch.from_numpy(graph_prior['A']).to(device),
        'K': torch.from_numpy(graph_prior['K']).to(device),
        'PE': torch.from_numpy(graph_prior['PE']).to(device),
    }
