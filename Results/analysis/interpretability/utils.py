"""
Shared utilities for interpretability analysis.

- Discovers repo roots from this file's location
- Adds CleanedProject/classifiers/graph_transformer to sys.path
- Loads trained GraphTransformerClassifier, graph prior, and dataset
- Provides helpers for SHAP/attention workflows and simple validations
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------------------
# Path discovery (based on your structure)
# This file lives at: CleanedProject/Results/analysis/interpretability/utils.py
# parents[0] = interpretability/
# parents[1] = analysis/
# parents[2] = Results/
# parents[3] = CleanedProject/
# parents[4] = graph_transformer_proteomics/   (repo root)
# --------------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
CLEANED_ROOT = THIS_FILE.parents[3]                  # CleanedProject/
REPO_ROOT_PARENT = THIS_FILE.parents[4]              # graph_transformer_proteomics/

# Classifier code location (CleanedProject/classifiers/graph_transformer)
CLASSIFIER_DIR = CLEANED_ROOT / "classifiers" / "graph_transformer"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

# Now imports are safe
from model import GraphTransformerClassifier
from graph_prior import load_graph_prior, get_graph_features_as_tensors
from dataset import load_and_preprocess_data, create_dataloaders


# --------------------------------------------------------------------------------------
# Defaults (prefer data at repo root; fall back to CleanedProject copies if present)
# --------------------------------------------------------------------------------------
def _first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    # If none exist, return the first for clearer error messages
    return candidates[0]


DEFAULT_PATHS: Dict[str, Path] = {
    "checkpoint": _first_existing(
        CLEANED_ROOT / "Results" / "classifiers" / "cancer_type_classifiers"
        / "transformer" / "checkpoints" / "best_model.pt"
    ),
    "prior": _first_existing(
        REPO_ROOT_PARENT / "priors" / "tcga_string_prior.npz",
        CLEANED_ROOT / "priors" / "tcga_string_prior.npz",
    ),
    "csv": _first_existing(
        REPO_ROOT_PARENT / "processed_datasets" / "tcga_pancan_rppa_compiled.csv",
        CLEANED_ROOT / "processed_datasets" / "tcga_pancan_rppa_compiled.csv",
    ),
}


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def load_trained_model(
    checkpoint_path: Optional[Path] = None,
    prior_path: Optional[Path] = None,
    device: str = "cpu",
) -> Tuple[GraphTransformerClassifier, Dict, Dict]:
    """
    Load a trained GraphTransformerClassifier, graph prior, and label info.

    Returns:
        model: eval-mode GraphTransformerClassifier
        graph_prior: dict with A, K, PE, protein_cols, genes, degrees
        label_info: dict with class_names, n_classes, etc.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_PATHS["checkpoint"]
    prior = Path(prior_path) if prior_path else DEFAULT_PATHS["prior"]

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not prior.exists():
        raise FileNotFoundError(f"Graph prior not found: {prior}")

    print(f"Loading checkpoint: {ckpt}")
    print(f"Loading graph prior: {prior}")
    print(f"Device: {device}")

    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})
    label_info = checkpoint.get("label_info", {})
    model_cfg = saved_config.get("MODEL", {})

    # Load prior and tensors
    graph_prior = load_graph_prior(str(prior))
    graph_tensors = get_graph_features_as_tensors(graph_prior, device=device)

    # Build model (do NOT pass ffn_dim; implementation reads it from config internally)
    model = GraphTransformerClassifier(
        n_proteins=graph_prior["A"].shape[0],
        n_classes=label_info["n_classes"],
        diffusion_kernel=graph_tensors["K"],
        positional_encodings=graph_tensors["PE"],
        embedding_dim=model_cfg.get("embedding_dim"),
        n_layers=model_cfg.get("n_layers"),
        n_heads=model_cfg.get("n_heads"),
        dropout=model_cfg.get("dropout"),
    ).to(device)

    # Load weights and set eval
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Brief summary
    print("Model loaded ✓  (eval mode)")
    if model_cfg:
        print(
            f"Config -> d_model={model_cfg.get('embedding_dim')}, "
            f"layers={model_cfg.get('n_layers')}, heads={model_cfg.get('n_heads')}, "
            f"classes={label_info.get('n_classes')}"
        )

    return model, graph_prior, label_info


def load_data(
    csv_path: Optional[Path] = None,
    prior_path: Optional[Path] = None,
    return_dataloaders: bool = False,
    batch_size: int = 64,
) -> Tuple[Dict, Dict, Optional[Tuple[DataLoader, DataLoader, DataLoader]]]:
    """
    Load and preprocess TCGA RPPA dataset consistent with training.
    """
    csv = Path(csv_path) if csv_path else DEFAULT_PATHS["csv"]
    prior = Path(prior_path) if prior_path else DEFAULT_PATHS["prior"]

    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv}")
    if not prior.exists():
        raise FileNotFoundError(f"Graph prior not found: {prior}")

    prior_data = np.load(prior, allow_pickle=True)
    protein_cols = prior_data["protein_cols"].tolist()

    print(f"Loading data CSV: {csv}")
    print(f"Using protein order from prior: {prior}")

    data_splits, label_info, _scaler = load_and_preprocess_data(str(csv), protein_cols=protein_cols)

    loaders = None
    if return_dataloaders:
        loaders = create_dataloaders(data_splits, batch_size=batch_size, num_workers=0)

    return data_splits, label_info, loaders


def get_cancer_types_with_min_samples(
    df: pd.DataFrame,
    min_samples: int = 250,
    cancer_col: str = "CANCER_TYPE_ACRONYM",
) -> List[str]:
    if cancer_col not in df.columns:
        raise ValueError(f"Column '{cancer_col}' not found in DataFrame")
    counts = df[cancer_col].value_counts()
    return sorted(counts[counts >= min_samples].index.tolist())


def compute_graph_distances(A: np.ndarray) -> np.ndarray:
    """
    Shortest path distances between protein pairs (unweighted, undirected).
    dist[i, j] = -1 if no path.
    """
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    A_sparse = csr_matrix(A)
    dist = shortest_path(A_sparse, directed=False, unweighted=True, return_predecessors=False)
    dist = np.where(np.isinf(dist), -1, dist).astype(int)
    np.fill_diagonal(dist, 0)
    return dist


def get_output_dirs() -> Tuple[Path, Path]:
    """
    Returns (plots_dir, data_dir) under interpretability/.
    """
    base_dir = THIS_FILE.parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, base_dir


# --------------------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Validating utils.py functions")
    print("=" * 80)

    # 1) Check default paths
    print("\n1. Default paths:")
    for k, p in DEFAULT_PATHS.items():
        print(f"   {k:10s}: {p}  [{'✓' if p.exists() else '✗'}]")

    # 2) Try loading prior (lightweight)
    prior_data = None
    print("\n2. Loading prior:")
    try:
        prior_path = DEFAULT_PATHS["prior"]
        prior_data = np.load(prior_path, allow_pickle=True)
        A = prior_data["A"]
        print(f"   ✓ A shape: {A.shape}, proteins: {A.shape[0]}")
        print(f"   ✓ protein_cols: {len(prior_data['protein_cols'])}")
    except Exception as e:
        print(f"   ✗ Prior load error: {e}")

    # 3) Distances (only if prior loaded)
    print("\n3. Graph distances:")
    try:
        if prior_data is None:
            raise RuntimeError("Skip: prior not loaded")
        dist = compute_graph_distances(prior_data["A"])
        print(f"   ✓ dist shape: {dist.shape}")
        print(f"   ✓ edges (d=1): {(dist == 1).sum() // 2}")
        print(f"   ✓ 2-hop (d=2): {(dist == 2).sum() // 2}")
        print(f"   ✓ unconnected: {(dist == -1).sum() // 2}")
    except Exception as e:
        print(f"   ✗ Distance error: {e}")

    # 4) CSV probe
    print("\n4. CSV probe:")
    try:
        csv = DEFAULT_PATHS["csv"]
        df = pd.read_csv(csv, nrows=10)
        print(f"   ✓ CSV readable; columns={len(df.columns)}")
        if "CANCER_TYPE_ACRONYM" in df.columns:
            print("   ✓ CANCER_TYPE_ACRONYM present")
    except Exception as e:
        print(f"   ✗ CSV error: {e}")

    # 5) Checkpoint probe
    print("\n5. Checkpoint probe:")
    try:
        ckpt = DEFAULT_PATHS["checkpoint"]
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        print(f"   ✓ Checkpoint loaded")
        print(f"   ✓ model_state_dict: {'model_state_dict' in checkpoint}")
        print(f"   ✓ label_info: {'label_info' in checkpoint}")
    except Exception as e:
        print(f"   ✗ Checkpoint error: {e}")

    print("\n" + "=" * 80)
    print("Validation complete.")
    print("=" * 80)
