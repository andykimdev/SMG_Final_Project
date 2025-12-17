"""
TCGA RPPA dataset module for cancer type classification.
Handles data loading, preprocessing, and splitting.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
try:
    from . import config
except ImportError:
    import config


class TCGARPPADataset(Dataset):
    """PyTorch Dataset for TCGA RPPA protein expression data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Protein expression values (n_samples, n_proteins)
            y: Cancer type class labels (n_samples,)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(
    csv_path: str,
    protein_cols: List[str],
    random_seed: int = None
) -> Tuple[Dict, Dict, StandardScaler]:
    """
    Load TCGA RPPA data and split into train/val/test sets.
    
    Args:
        csv_path: Path to TCGA RPPA CSV file
        protein_cols: List of protein column names (in correct order from graph prior)
        random_seed: Random seed for reproducibility
        
    Returns:
        data_splits: Dictionary with 'train', 'val', 'test' DataFrames
        label_info: Dictionary with label mappings
        scaler: Fitted StandardScaler (on training data only)
    """
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Identify protein columns in the CSV (those containing '|')
    csv_protein_cols = [col for col in df.columns if '|' in col]
    print(f"Found {len(csv_protein_cols)} protein columns in CSV")
    
    # Verify all required protein columns are present
    missing_cols = set(protein_cols) - set(csv_protein_cols)
    if missing_cols:
        raise ValueError(f"Missing protein columns in CSV: {missing_cols}")
    
    # Reorder protein columns to match graph prior
    protein_data = df[protein_cols].copy()
    
    # Filter samples: drop those with >50% missing protein values
    missing_per_sample = protein_data.isnull().sum(axis=1) / len(protein_cols)
    valid_samples = missing_per_sample <= config.DATA['missing_threshold']
    print(f"Filtering samples: {valid_samples.sum()}/{len(df)} have ≤{config.DATA['missing_threshold']*100}% missing values")
    
    df_filtered = df[valid_samples].copy()
    protein_data_filtered = protein_data[valid_samples].copy()
    
    # Extract labels (cancer type)
    if 'CANCER_TYPE_ACRONYM' not in df_filtered.columns:
        raise ValueError("CANCER_TYPE_ACRONYM column not found in CSV")
    
    cancer_types = df_filtered['CANCER_TYPE_ACRONYM'].copy()
    
    # Filter cancer types with too few samples
    type_counts = cancer_types.value_counts()
    valid_types = type_counts[type_counts >= config.DATA['min_samples_per_class']].index
    valid_mask = cancer_types.isin(valid_types)
    
    df_filtered = df_filtered[valid_mask].copy()
    protein_data_filtered = protein_data_filtered[valid_mask].copy()
    cancer_types = cancer_types[valid_mask].copy()
    
    print(f"After filtering: {len(df_filtered)} samples across {len(valid_types)} cancer types")
    print(f"Cancer type distribution:\n{cancer_types.value_counts().sort_index()}")
    
    # Create label mapping
    unique_types = sorted(cancer_types.unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_types)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    y = cancer_types.map(label_to_idx).values
    
    # Extract patient IDs for splitting
    if 'PATIENT_ID' not in df_filtered.columns:
        raise ValueError("PATIENT_ID column not found in CSV")
    
    patient_ids = df_filtered['PATIENT_ID'].values
    
    # Split by patient (no leakage)
    unique_patients = np.unique(patient_ids)
    
    # First split: train vs (val+test)
    train_patients, valtest_patients, train_y_strat, valtest_y_strat = train_test_split(
        unique_patients,
        [cancer_types[patient_ids == p].iloc[0] for p in unique_patients],  # One label per patient for stratification
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=random_seed,
        stratify=[cancer_types[patient_ids == p].iloc[0] for p in unique_patients]
    )
    
    # Second split: val vs test
    val_size_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_patients, test_patients = train_test_split(
        valtest_patients,
        test_size=1 - val_size_adjusted,
        random_state=random_seed,
        stratify=valtest_y_strat
    )
    
    # Create masks for each split
    train_mask = np.isin(patient_ids, train_patients)
    val_mask = np.isin(patient_ids, val_patients)
    test_mask = np.isin(patient_ids, test_patients)
    
    print(f"\nData splits:")
    print(f"  Train: {train_mask.sum()} samples ({train_mask.sum()/len(df_filtered)*100:.1f}%)")
    print(f"  Val:   {val_mask.sum()} samples ({val_mask.sum()/len(df_filtered)*100:.1f}%)")
    print(f"  Test:  {test_mask.sum()} samples ({test_mask.sum()/len(df_filtered)*100:.1f}%)")
    
    # Split data
    X_train = protein_data_filtered[train_mask].values
    X_val = protein_data_filtered[val_mask].values
    X_test = protein_data_filtered[test_mask].values
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    # Handle missing values: impute with column mean (from training set only)
    print("\nHandling missing values...")
    train_means = np.nanmean(X_train, axis=0)
    
    for split_name, X_split in [('train', X_train), ('val', X_val), ('test', X_test)]:
        n_missing = np.sum(np.isnan(X_split))
        if n_missing > 0:
            print(f"  {split_name}: {n_missing} missing values, imputing with train means")
    
    X_train = np.where(np.isnan(X_train), train_means, X_train)
    X_val = np.where(np.isnan(X_val), train_means, X_val)
    X_test = np.where(np.isnan(X_test), train_means, X_test)
    
    # Standardize: fit on training data only
    print("\nStandardizing protein features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Mean: {scaler.mean_[:5]} ... (first 5)")
    print(f"  Std:  {scaler.scale_[:5]} ... (first 5)")
    
    # Package results
    data_splits = {
        'train': (X_train_scaled, y_train),
        'val': (X_val_scaled, y_val),
        'test': (X_test_scaled, y_test),
    }
    
    label_info = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'n_classes': len(unique_types),
        'class_names': unique_types,
    }
    
    return data_splits, label_info, scaler


def create_dataloaders(
    data_splits: Dict,
    batch_size: int = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        data_splits: Dictionary with 'train', 'val', 'test' tuples (X, y)
        batch_size: Batch size (defaults to config.TRAINING['batch_size'])
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.TRAINING['batch_size']
    
    # Create datasets
    train_dataset = TCGARPPADataset(*data_splits['train'])
    val_dataset = TCGARPPADataset(*data_splits['val'])
    test_dataset = TCGARPPADataset(*data_splits['test'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
