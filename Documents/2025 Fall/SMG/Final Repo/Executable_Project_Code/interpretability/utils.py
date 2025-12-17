"""
Shared utilities for interpretability analysis.

Loads trained GraphTransformerClassifier, graph prior, and dataset.
Provides helpers for SHAP/attention workflows.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Path discovery - this file is at Executable_Project_Code/interpretability/utils.py
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # Executable_Project_Code/

# Add src to path for imports
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import model components
from model import GraphTransformerClassifier
from graph_prior import load_graph_prior, get_graph_features_as_tensors
from dataset import load_and_preprocess_data, create_dataloaders


def _first_existing(*candidates: Path) -> Path:
    """Return first existing path, or first candidate if none exist."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DEFAULT_PATHS: Dict[str, Path] = {
    "checkpoint": _first_existing(
        PROJECT_ROOT / "pretrained" / "best_model.pt",
    ),
    "prior": _first_existing(
        PROJECT_ROOT / "data" / "priors" / "tcga_string_prior.npz",
        PROJECT_ROOT / "data" / "tcga_string_prior.npz",
    ),
    "csv": _first_existing(
        PROJECT_ROOT / "data" / "processed_datasets" / "tcga_pancan_rppa_compiled.csv",
        PROJECT_ROOT / "data" / "tcga_pancan_rppa_compiled.csv",
    ),
}


def load_trained_model(
    checkpoint_path: Optional[str] = None,
    prior_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[GraphTransformerClassifier, Dict, Dict]:
    """
    Load trained model, graph prior, and label info.
    
    Returns:
        model: Trained GraphTransformerClassifier
        graph_prior: Dict with 'A', 'K', 'PE', 'protein_cols'
        label_info: Dict with 'n_classes', 'class_names', etc.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_PATHS["checkpoint"]
    prior = Path(prior_path) if prior_path else DEFAULT_PATHS["prior"]
    
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not prior.exists():
        raise FileNotFoundError(f"Prior not found: {prior}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    label_info = checkpoint.get("label_info", {})
    model_config = checkpoint.get("config", {}).get("MODEL", {})
    
    # Load graph prior
    graph_prior = load_graph_prior(str(prior))
    graph_tensors = get_graph_features_as_tensors(graph_prior, device=device)
    
    # Build model
    model = GraphTransformerClassifier(
        n_proteins=graph_prior["A"].shape[0],
        n_classes=label_info.get("n_classes", 32),
        diffusion_kernel=graph_tensors["K"],
        positional_encodings=graph_tensors["PE"],
        embedding_dim=model_config.get("embedding_dim", 128),
        n_layers=model_config.get("n_layers", 4),
        n_heads=model_config.get("n_heads", 8),
        dropout=model_config.get("dropout", 0.1),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, graph_prior, label_info


def load_data(
    csv_path: Optional[str] = None,
    prior_path: Optional[str] = None,
    return_dataloaders: bool = False,
    batch_size: int = 32,
) -> Tuple[Dict, List[str], Optional[Dict[str, DataLoader]]]:
    """
    Load and preprocess data.
    
    Returns:
        data_splits: Dict with 'train', 'val', 'test' tuples of (X, y)
        protein_names: List of protein column names
        dataloaders: Optional dict {'train': loader, 'val': loader, 'test': loader}
    """
    csv = Path(csv_path) if csv_path else DEFAULT_PATHS["csv"]
    prior = Path(prior_path) if prior_path else DEFAULT_PATHS["prior"]
    
    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv}")
    if not prior.exists():
        raise FileNotFoundError(f"Prior not found: {prior}")
    
    # Load prior for protein order
    prior_data = np.load(prior, allow_pickle=True)
    protein_cols = prior_data["protein_cols"].tolist()
    
    # Load and preprocess data
    data_splits, label_info, scaler = load_and_preprocess_data(str(csv), protein_cols)
    
    if return_dataloaders:
        loaders_list = create_dataloaders(data_splits, batch_size=batch_size)
        dataloaders = {
            'train': loaders_list[0],
            'val': loaders_list[1],
            'test': loaders_list[2]
        }
        return data_splits, protein_cols, dataloaders
    
    return data_splits, protein_cols, None


def get_cancer_types_with_min_samples(
    data_splits: Dict,
    label_info: Dict,
    min_samples: int = 250,
) -> List[str]:
    """Get cancer types with at least min_samples in training set."""
    train_y = data_splits["train"][1]
    class_names = label_info.get("class_names", [])
    
    unique, counts = np.unique(train_y, return_counts=True)
    valid_indices = unique[counts >= min_samples]
    
    return [class_names[int(idx)] for idx in valid_indices]


def compute_graph_distances(A: np.ndarray) -> np.ndarray:
    """Compute shortest path distances in graph."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    
    A_sparse = csr_matrix(A)
    dist_matrix = shortest_path(A_sparse, directed=False, unweighted=False)
    dist_matrix[np.isinf(dist_matrix)] = A.shape[0] + 1
    return dist_matrix


def get_output_dirs(base_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Get output directories for plots and results."""
    if base_dir is None:
        base_dir = PROJECT_ROOT / "results"
    
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return plots_dir, base_dir


if __name__ == "__main__":
    # Validation
    print("Testing utils...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Default paths:")
    for key, path in DEFAULT_PATHS.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {key}: {path} {exists}")
