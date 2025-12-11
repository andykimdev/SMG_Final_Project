"""
Graph prior utilities for the simple graph-aware transformer.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm

from . import config


def load_graph_prior(prior_path: str) -> Dict:
    """
    Load STRING adjacency data and derive graph features.
    """
    data = np.load(prior_path, allow_pickle=True)
    A = data["A"].astype(np.float32)
    protein_cols = data["protein_cols"].tolist()
    genes = data["genes"].tolist()

    L, degrees = compute_laplacian(A, config.GRAPH["laplacian_type"])
    K = compute_diffusion_kernel(L, beta=config.GRAPH["diffusion_beta"])
    pe_dim = config.MODEL.get("graph_pe_dim", 32)
    PE = compute_positional_encodings(L, k=pe_dim)

    return {
        "A": A,
        "K": K,
        "PE": PE,
        "protein_cols": protein_cols,
        "genes": genes,
        "degrees": degrees,
    }


def compute_laplacian(A: np.ndarray, laplacian_type: str) -> Tuple[np.ndarray, np.ndarray]:
    degrees = np.sum(A, axis=1)
    degrees_safe = degrees + 1e-8

    if laplacian_type == "normalized":
        D_inv = np.diag(1.0 / degrees_safe)
        L = np.eye(A.shape[0]) - D_inv @ A
    elif laplacian_type == "symmetric":
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_safe))
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    return L.astype(np.float32), degrees.astype(np.float32)


def compute_diffusion_kernel(L: np.ndarray, beta: float) -> np.ndarray:
    K = expm(-beta * L)
    K = (K + K.T) / 2
    K = np.clip(K, 0, None)
    return K.astype(np.float32)


def compute_positional_encodings(L: np.ndarray, k: int) -> np.ndarray:
    num_nodes = L.shape[0]
    k = min(k, num_nodes)
    if k == 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    if k < num_nodes // 2:
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
            idx = np.argsort(eigenvalues)
            PE = eigenvectors[:, idx]
        except Exception:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            PE = eigenvectors[:, :k]
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        PE = eigenvectors[:, :k]

    return PE.astype(np.float32)

