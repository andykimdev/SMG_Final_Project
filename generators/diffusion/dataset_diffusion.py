"""
Dataset Module for Protein Expression Diffusion Model.

This module handles:
1. Loading TCGA RPPA data with patient context
2. Preprocessing and normalization
3. Creating context dictionaries for conditioning
4. Data splitting (same as classifier for consistency)

Key Features:
- Returns patient context features (cancer type, stage, age, sex, molecular scores)
- Handles missing/unknown values in context features
- Creates categorical indices for embeddings
- Normalizes continuous features
- Extracts and encodes survival outcomes (OS, PFS, DSS, DFS) - NEW
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, List, Optional
import sys
import os
import re

# Add Classifier directory to path to import data utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Cancer Classifier'))

import config


class ProteinDiffusionDataset(Dataset):
    """
    PyTorch Dataset for diffusion model training.
    
    Returns:
        protein_expr: [N] z-scored protein expression
        context: Dict of patient features for conditioning
    """
    
    def __init__(
        self,
        protein_expr: np.ndarray,
        cancer_types: np.ndarray,
        stages: np.ndarray,
        ages: np.ndarray,
        sexes: np.ndarray,
        molecular_scores: np.ndarray,
        # Survival outcomes - NEW
        os_status: np.ndarray,
        os_months: np.ndarray,
        pfs_status: np.ndarray,
        pfs_months: np.ndarray,
        dss_status: np.ndarray,
        dss_months: np.ndarray,
        dfs_status: np.ndarray,
        dfs_months: np.ndarray,
    ):
        """
        Args:
            protein_expr: [n_samples, n_proteins] z-scored expression
            cancer_types: [n_samples] categorical indices
            stages: [n_samples] categorical indices  
            ages: [n_samples] normalized ages (0-1)
            sexes: [n_samples] categorical indices
            molecular_scores: [n_samples, 4] normalized scores [TMB, aneuploidy, MSI, TBL]
            os_status/months: Overall survival (status indices, normalized months)
            pfs_status/months: Progression-free survival
            dss_status/months: Disease-specific survival
            dfs_status/months: Disease-free survival
        """
        self.protein_expr = torch.from_numpy(protein_expr).float()
        self.cancer_types = torch.from_numpy(cancer_types).long()
        self.stages = torch.from_numpy(stages).long()
        self.ages = torch.from_numpy(ages).float().unsqueeze(-1)  # [n, 1]
        self.sexes = torch.from_numpy(sexes).long()
        self.molecular_scores = torch.from_numpy(molecular_scores).float()
        
        # Survival outcomes - NEW
        self.os_status = torch.from_numpy(os_status).long()
        self.os_months = torch.from_numpy(os_months).float().unsqueeze(-1)
        self.pfs_status = torch.from_numpy(pfs_status).long()
        self.pfs_months = torch.from_numpy(pfs_months).float().unsqueeze(-1)
        self.dss_status = torch.from_numpy(dss_status).long()
        self.dss_months = torch.from_numpy(dss_months).float().unsqueeze(-1)
        self.dfs_status = torch.from_numpy(dfs_status).long()
        self.dfs_months = torch.from_numpy(dfs_months).float().unsqueeze(-1)
        
    def __len__(self):
        return len(self.protein_expr)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            protein_expr: [N] protein expression
            context: Dict with patient features
        """
        context = {
            'cancer_type': self.cancer_types[idx],
            'stage': self.stages[idx],
            'age': self.ages[idx],
            'sex': self.sexes[idx],
            'molecular': self.molecular_scores[idx],
            # Survival outcomes
            'os_status': self.os_status[idx],
            'os_months': self.os_months[idx],
            'pfs_status': self.pfs_status[idx],
            'pfs_months': self.pfs_months[idx],
            'dss_status': self.dss_status[idx],
            'dss_months': self.dss_months[idx],
            'dfs_status': self.dfs_status[idx],
            'dfs_months': self.dfs_months[idx],
        }
        
        return self.protein_expr[idx], context


def parse_survival_status(status_str: str) -> int:
    """
    Parse survival status string to integer.
    
    TCGA status formats:
    - OS: "0:LIVING", "1:DECEASED"
    - DSS: "0:ALIVE OR DEAD TUMOR FREE", "1:DEAD WITH TUMOR"
    - PFS: "0:CENSORED", "1:PROGRESSION"
    - DFS: "0:DiseaseFree", "1:Recurred/Progressed"
    
    Returns:
        0 = censored/alive (no event)
        1 = event occurred (deceased/progressed/recurred)
        2 = unknown/missing
    """
    if pd.isna(status_str) or str(status_str).strip() == '':
        return 2  # Unknown
    
    status = str(status_str).strip().upper()
    
    # Check prefix first (most reliable) - "0:" means censored, "1:" means event
    if status.startswith('0:'):
        return 0  # Censored
    if status.startswith('1:'):
        return 1  # Event
    
    # Fallback pattern matching for non-standard formats
    # Censored/alive patterns (check these FIRST - more specific)
    censored_patterns = ['LIVING', 'ALIVE', 'CENSORED', 'DISEASEFREE', 'DISEASE FREE', 'TUMOR FREE']
    if any(x in status for x in censored_patterns):
        return 0
    
    # Event occurred patterns
    event_patterns = ['DECEASED', 'DEAD', 'PROGRESSION', 'RECURRED', 'PROGRESSED']
    if any(x in status for x in event_patterns):
        return 1
    
    # Default to unknown
    return 2


def load_and_preprocess_diffusion_data(
    csv_path: str,
    protein_cols: List[str],
    random_seed: int = None
) -> Tuple[Dict, Dict, StandardScaler, Dict]:
    """
    Load TCGA RPPA data and prepare for diffusion model training.
    
    This function:
    1. Loads the raw data
    2. Filters samples and cancer types
    3. Extracts and preprocesses patient context features
    4. Extracts survival outcomes - NEW
    5. Splits data (same as classifier)
    6. Normalizes protein expression and context features
    
    Args:
        csv_path: Path to TCGA RPPA CSV
        protein_cols: List of protein column names (from graph prior)
        random_seed: Random seed for reproducibility
        
    Returns:
        data_splits: Dict with 'train', 'val', 'test' datasets
        context_info: Dict with encoders and statistics
        scaler: Fitted StandardScaler for protein expression
        label_info: Dict with cancer type mappings
    """
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # ========================================================================
    # 1. Filter Protein Columns
    # ========================================================================
    csv_protein_cols = [col for col in df.columns if '|' in col]
    missing_cols = set(protein_cols) - set(csv_protein_cols)
    if missing_cols:
        raise ValueError(f"Missing protein columns: {missing_cols}")
    
    protein_data = df[protein_cols].copy()
    
    # ========================================================================
    # 2. Filter Samples
    # ========================================================================
    # Drop samples with too many missing proteins
    missing_per_sample = protein_data.isnull().sum(axis=1) / len(protein_cols)
    valid_samples = missing_per_sample <= config.DATA['missing_threshold']
    print(f"Filtering samples: {valid_samples.sum()}/{len(df)} have ≤{config.DATA['missing_threshold']*100}% missing")
    
    df_filtered = df[valid_samples].copy()
    protein_data_filtered = protein_data[valid_samples].copy()
    
    # ========================================================================
    # 3. Extract Cancer Types and Filter by Count
    # ========================================================================
    if 'CANCER_TYPE_ACRONYM' not in df_filtered.columns:
        raise ValueError("CANCER_TYPE_ACRONYM column not found")
    
    cancer_types = df_filtered['CANCER_TYPE_ACRONYM'].copy()
    type_counts = cancer_types.value_counts()
    valid_types = type_counts[type_counts >= config.DATA['min_samples_per_class']].index
    valid_mask = cancer_types.isin(valid_types)
    
    df_filtered = df_filtered[valid_mask].copy()
    protein_data_filtered = protein_data_filtered[valid_mask].copy()
    cancer_types = cancer_types[valid_mask].copy()
    
    print(f"After filtering: {len(df_filtered)} samples across {len(valid_types)} cancer types")
    print(f"Cancer type distribution:\n{cancer_types.value_counts().sort_index()}")
    
    # ========================================================================
    # 4. Extract and Process Context Features
    # ========================================================================
    print("\nExtracting patient context features...")
    
    # 4a. Cancer Type (primary biological signal)
    cancer_type_encoder = LabelEncoder()
    cancer_type_indices = cancer_type_encoder.fit_transform(cancer_types)
    
    # 4b. Clinical Stage - CLEANED to only allow valid AJCC stages
    stage_col = 'AJCC_PATHOLOGIC_TUMOR_STAGE'
    
    # Valid AJCC stage patterns
    VALID_STAGES = {
        'STAGE 0', 'STAGE 0A', 'STAGE 0B',
        'STAGE I', 'STAGE IA', 'STAGE IB', 'STAGE IC', 'STAGE IS',
        'STAGE II', 'STAGE IIA', 'STAGE IIB', 'STAGE IIC',
        'STAGE III', 'STAGE IIIA', 'STAGE IIIB', 'STAGE IIIC', 'STAGE IIID',
        'STAGE IV', 'STAGE IVA', 'STAGE IVB', 'STAGE IVC',
        'STAGE X', 'STAGE I/II (NOS)',
    }
    
    def clean_stage(raw_stage):
        """Map raw stage to valid AJCC stage or 'Unknown'."""
        if pd.isna(raw_stage) or str(raw_stage).strip() == '':
            return 'Unknown'
        stage_str = str(raw_stage).strip().upper()
        if stage_str in VALID_STAGES:
            return stage_str
        return 'Unknown'
    
    if stage_col in df_filtered.columns:
        raw_stages = df_filtered[stage_col]
        stages = raw_stages.apply(clean_stage)
        valid_count = (stages != 'Unknown').sum()
        print(f"  Stage cleaning: {valid_count}/{len(stages)} valid AJCC stages")
    else:
        print(f"Warning: {stage_col} not found, using 'Unknown' for all samples")
        stages = pd.Series(['Unknown'] * len(df_filtered))
    
    stage_encoder = LabelEncoder()
    stage_indices = stage_encoder.fit_transform(stages)
    print(f"  Stage categories: {list(stage_encoder.classes_)}")
    
    # 4c. Age (continuous, normalize to [0, 1])
    if 'AGE' in df_filtered.columns:
        ages = df_filtered['AGE'].fillna(df_filtered['AGE'].median()).values
        age_min, age_max = ages.min(), ages.max()
        ages_normalized = (ages - age_min) / (age_max - age_min + 1e-8)
    else:
        print("Warning: AGE not found, using 0.5 for all samples")
        ages_normalized = np.full(len(df_filtered), 0.5)
        age_min, age_max = 0, 100
    
    # 4d. Sex
    if 'SEX' in df_filtered.columns:
        sexes = df_filtered['SEX'].fillna('Unknown').astype(str)
    else:
        print("Warning: SEX not found, using 'Unknown' for all samples")
        sexes = pd.Series(['Unknown'] * len(df_filtered))
    
    sex_encoder = LabelEncoder()
    sex_indices = sex_encoder.fit_transform(sexes)
    
    # 4e. Molecular Scores (continuous, standardize)
    molecular_features = ['ANEUPLOIDY_SCORE', 'TMB_NONSYNONYMOUS', 'MSI_SCORE_MANTIS', 'TBL_SCORE']
    molecular_data = []
    
    for feat in molecular_features:
        if feat in df_filtered.columns:
            values = df_filtered[feat].fillna(df_filtered[feat].median()).values
        else:
            print(f"Warning: {feat} not found, using 0.0 for all samples")
            values = np.zeros(len(df_filtered))
        molecular_data.append(values)
    
    molecular_scores = np.stack(molecular_data, axis=1)  # [n_samples, 4]
    
    # Standardize molecular scores
    molecular_scaler = StandardScaler()
    molecular_scores_scaled = molecular_scaler.fit_transform(molecular_scores)
    
    # ========================================================================
    # 4f. Survival Outcomes - NEW
    # ========================================================================
    print("\nExtracting survival outcomes...")
    
    survival_outcomes = [
        ('OS_STATUS', 'OS_MONTHS'),
        ('PFS_STATUS', 'PFS_MONTHS'),
        ('DSS_STATUS', 'DSS_MONTHS'),
        ('DFS_STATUS', 'DFS_MONTHS'),
    ]
    
    survival_data = {}
    survival_months_max = {}  # For normalization
    
    for status_col, months_col in survival_outcomes:
        prefix = status_col.split('_')[0].lower()  # os, pfs, dss, dfs
        
        # Parse status
        if status_col in df_filtered.columns:
            status_values = df_filtered[status_col].apply(parse_survival_status).values
        else:
            print(f"  Warning: {status_col} not found, using 'unknown' for all")
            status_values = np.full(len(df_filtered), 2)  # Unknown
        
        # Parse months
        if months_col in df_filtered.columns:
            months_raw = pd.to_numeric(df_filtered[months_col], errors='coerce')
            # For missing months, use median of available values
            months_median = months_raw.median() if not months_raw.isna().all() else 0
            months_values = months_raw.fillna(months_median).values
            months_max = months_values.max() if len(months_values) > 0 else 1.0
            # Normalize to [0, 1]
            months_normalized = months_values / (months_max + 1e-8)
        else:
            print(f"  Warning: {months_col} not found, using 0.5 for all")
            months_normalized = np.full(len(df_filtered), 0.5)
            months_max = 1.0
        
        survival_data[f'{prefix}_status'] = status_values
        survival_data[f'{prefix}_months'] = months_normalized
        survival_months_max[prefix] = months_max
        
        # Report stats
        event_count = (status_values == 1).sum()
        censored_count = (status_values == 0).sum()
        unknown_count = (status_values == 2).sum()
        print(f"  {status_col}: {event_count} events, {censored_count} censored, {unknown_count} unknown")
    
    print(f"\n  Cancer types: {len(cancer_type_encoder.classes_)} categories")
    print(f"  Stages: {len(stage_encoder.classes_)} categories")
    print(f"  Age range: {age_min:.1f} - {age_max:.1f} years")
    print(f"  Sexes: {len(sex_encoder.classes_)} categories")
    print(f"  Molecular features: {len(molecular_features)}")
    print(f"  Survival outcomes: 4 (OS, PFS, DSS, DFS)")
    
    # ========================================================================
    # 5. Split Data (same as classifier)
    # ========================================================================
    patient_ids = df_filtered['PATIENT_ID'].values
    unique_patients = np.unique(patient_ids)
    
    # First split: train vs (val+test)
    train_patients, valtest_patients, train_y_strat, valtest_y_strat = train_test_split(
        unique_patients,
        [cancer_types[patient_ids == p].iloc[0] for p in unique_patients],
        test_size=config.DATA['val_ratio'] + config.DATA['test_ratio'],
        random_state=random_seed,
        stratify=[cancer_types[patient_ids == p].iloc[0] for p in unique_patients]
    )
    
    # Second split: val vs test
    val_size_adjusted = config.DATA['val_ratio'] / (config.DATA['val_ratio'] + config.DATA['test_ratio'])
    val_patients, test_patients = train_test_split(
        valtest_patients,
        test_size=1 - val_size_adjusted,
        random_state=random_seed,
        stratify=valtest_y_strat
    )
    
    # Create masks
    train_mask = np.isin(patient_ids, train_patients)
    val_mask = np.isin(patient_ids, val_patients)
    test_mask = np.isin(patient_ids, test_patients)
    
    print(f"\nData splits:")
    print(f"  Train: {train_mask.sum()} samples ({train_mask.sum()/len(df_filtered)*100:.1f}%)")
    print(f"  Val:   {val_mask.sum()} samples ({val_mask.sum()/len(df_filtered)*100:.1f}%)")
    print(f"  Test:  {test_mask.sum()} samples ({test_mask.sum()/len(df_filtered)*100:.1f}%)")
    
    # ========================================================================
    # 6. Handle Missing Protein Values and Standardize
    # ========================================================================
    print("\nProcessing protein expression...")
    
    # Convert to numpy
    X = protein_data_filtered.values
    
    # Split
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    # Impute missing with train mean
    train_means = np.nanmean(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), train_means, X_train)
    X_val = np.where(np.isnan(X_val), train_means, X_val)
    X_test = np.where(np.isnan(X_test), train_means, X_test)
    
    # Standardize (z-score)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Mean: {scaler.mean_[:5]} ... (first 5)")
    print(f"  Std:  {scaler.scale_[:5]} ... (first 5)")
    
    # ========================================================================
    # 7. Create Datasets
    # ========================================================================
    train_dataset = ProteinDiffusionDataset(
        X_train_scaled,
        cancer_type_indices[train_mask],
        stage_indices[train_mask],
        ages_normalized[train_mask],
        sex_indices[train_mask],
        molecular_scores_scaled[train_mask],
        # Survival
        survival_data['os_status'][train_mask],
        survival_data['os_months'][train_mask],
        survival_data['pfs_status'][train_mask],
        survival_data['pfs_months'][train_mask],
        survival_data['dss_status'][train_mask],
        survival_data['dss_months'][train_mask],
        survival_data['dfs_status'][train_mask],
        survival_data['dfs_months'][train_mask],
    )
    
    val_dataset = ProteinDiffusionDataset(
        X_val_scaled,
        cancer_type_indices[val_mask],
        stage_indices[val_mask],
        ages_normalized[val_mask],
        sex_indices[val_mask],
        molecular_scores_scaled[val_mask],
        # Survival
        survival_data['os_status'][val_mask],
        survival_data['os_months'][val_mask],
        survival_data['pfs_status'][val_mask],
        survival_data['pfs_months'][val_mask],
        survival_data['dss_status'][val_mask],
        survival_data['dss_months'][val_mask],
        survival_data['dfs_status'][val_mask],
        survival_data['dfs_months'][val_mask],
    )
    
    test_dataset = ProteinDiffusionDataset(
        X_test_scaled,
        cancer_type_indices[test_mask],
        stage_indices[test_mask],
        ages_normalized[test_mask],
        sex_indices[test_mask],
        molecular_scores_scaled[test_mask],
        # Survival
        survival_data['os_status'][test_mask],
        survival_data['os_months'][test_mask],
        survival_data['pfs_status'][test_mask],
        survival_data['pfs_months'][test_mask],
        survival_data['dss_status'][test_mask],
        survival_data['dss_months'][test_mask],
        survival_data['dfs_status'][test_mask],
        survival_data['dfs_months'][test_mask],
    )
    
    # ========================================================================
    # 8. Package Results
    # ========================================================================
    data_splits = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }
    
    context_info = {
        'cancer_type_encoder': cancer_type_encoder,
        'stage_encoder': stage_encoder,
        'sex_encoder': sex_encoder,
        'molecular_scaler': molecular_scaler,
        'age_min': age_min,
        'age_max': age_max,
        'num_cancer_types': len(cancer_type_encoder.classes_),
        'num_stages': len(stage_encoder.classes_),
        'num_sexes': len(sex_encoder.classes_),
        'num_survival_status': 3,  # 0=censored, 1=event, 2=unknown
        # Survival normalization info
        'survival_months_max': survival_months_max,
    }
    
    label_info = {
        'cancer_types': cancer_type_encoder.classes_.tolist(),
        'stages': stage_encoder.classes_.tolist(),
        'sexes': sex_encoder.classes_.tolist(),
    }
    
    return data_splits, context_info, scaler, label_info


def collate_fn(batch):
    """Custom collate function to handle context dictionaries."""
    protein_exprs = torch.stack([item[0] for item in batch])
    
    # Stack context features
    context = {
        'cancer_type': torch.stack([item[1]['cancer_type'] for item in batch]),
        'stage': torch.stack([item[1]['stage'] for item in batch]),
        'age': torch.stack([item[1]['age'] for item in batch]),
        'sex': torch.stack([item[1]['sex'] for item in batch]),
        'molecular': torch.stack([item[1]['molecular'] for item in batch]),
        # Survival
        'os_status': torch.stack([item[1]['os_status'] for item in batch]),
        'os_months': torch.stack([item[1]['os_months'] for item in batch]),
        'pfs_status': torch.stack([item[1]['pfs_status'] for item in batch]),
        'pfs_months': torch.stack([item[1]['pfs_months'] for item in batch]),
        'dss_status': torch.stack([item[1]['dss_status'] for item in batch]),
        'dss_months': torch.stack([item[1]['dss_months'] for item in batch]),
        'dfs_status': torch.stack([item[1]['dfs_status'] for item in batch]),
        'dfs_months': torch.stack([item[1]['dfs_months'] for item in batch]),
    }
    
    return protein_exprs, context


def create_dataloaders(
    data_splits: Dict,
    batch_size: int = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        data_splits: Dictionary with 'train', 'val', 'test' datasets
        batch_size: Batch size (defaults to config.TRAINING['batch_size'])
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.TRAINING['batch_size']
    
    train_loader = DataLoader(
        data_splits['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        data_splits['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        data_splits['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_context(
    cancer_type: str,
    stage: str = 'STAGE II',
    age: float = 60.0,
    sex: str = 'Female',
    molecular_scores: Optional[np.ndarray] = None,
    os_status: int = 0,
    os_months: float = 0.5,
    pfs_status: int = 0,
    pfs_months: float = 0.5,
    dss_status: int = 0,
    dss_months: float = 0.5,
    dfs_status: int = 0,
    dfs_months: float = 0.5,
    context_info: Dict = None
) -> Dict[str, torch.Tensor]:
    """
    Create a synthetic patient context for conditional generation.
    
    This is useful for generating "what-if" scenarios:
    - "What would a 40-year-old male BRCA patient's proteome look like?"
    - "How does proteome change from Stage I to Stage IV in LUAD?"
    - "What proteins differ in patients who survived 5 years vs 1 year?"
    
    Args:
        cancer_type: Cancer type acronym (e.g., 'BRCA', 'LUAD')
        stage: Clinical stage (e.g., 'STAGE II', 'STAGE IIIA')
        age: Patient age in years
        sex: 'Male', 'Female', or 'Unknown'
        molecular_scores: [4] array of [TMB, aneuploidy, MSI, TBL] (if None, uses median)
        os_status: Overall survival status (0=censored, 1=deceased, 2=unknown)
        os_months: Normalized OS months (0-1)
        pfs_status: Progression-free survival status
        pfs_months: Normalized PFS months
        dss_status: Disease-specific survival status
        dss_months: Normalized DSS months
        dfs_status: Disease-free survival status
        dfs_months: Normalized DFS months
        context_info: Context info dict from load_and_preprocess_diffusion_data
        
    Returns:
        context_dict: Dictionary of tensors ready for model input
    """
    if context_info is None:
        raise ValueError("context_info is required for encoding")
    
    # Encode cancer type
    try:
        cancer_idx = context_info['cancer_type_encoder'].transform([cancer_type])[0]
    except ValueError:
        raise ValueError(f"Unknown cancer type: {cancer_type}. "
                        f"Valid types: {list(context_info['cancer_type_encoder'].classes_)}")
    
    # Encode stage
    try:
        stage_idx = context_info['stage_encoder'].transform([stage])[0]
    except ValueError:
        print(f"Warning: Unknown stage '{stage}', using 'Unknown'")
        stage_idx = context_info['stage_encoder'].transform(['Unknown'])[0]
    
    # Normalize age
    age_normalized = (age - context_info['age_min']) / (context_info['age_max'] - context_info['age_min'] + 1e-8)
    age_normalized = np.clip(age_normalized, 0, 1)
    
    # Encode sex
    try:
        sex_idx = context_info['sex_encoder'].transform([sex])[0]
    except ValueError:
        print(f"Warning: Unknown sex '{sex}', using 'Unknown'")
        sex_idx = context_info['sex_encoder'].transform(['Unknown'])[0]
    
    # Molecular scores
    if molecular_scores is None:
        molecular_scores = np.zeros(4)
    else:
        molecular_scores = context_info['molecular_scaler'].transform(molecular_scores.reshape(1, -1))[0]
    
    # Create context dict
    context_dict = {
        'cancer_type': torch.tensor([cancer_idx], dtype=torch.long),
        'stage': torch.tensor([stage_idx], dtype=torch.long),
        'age': torch.tensor([[age_normalized]], dtype=torch.float32),
        'sex': torch.tensor([sex_idx], dtype=torch.long),
        'molecular': torch.tensor([molecular_scores], dtype=torch.float32),
        # Survival
        'os_status': torch.tensor([os_status], dtype=torch.long),
        'os_months': torch.tensor([[os_months]], dtype=torch.float32),
        'pfs_status': torch.tensor([pfs_status], dtype=torch.long),
        'pfs_months': torch.tensor([[pfs_months]], dtype=torch.float32),
        'dss_status': torch.tensor([dss_status], dtype=torch.long),
        'dss_months': torch.tensor([[dss_months]], dtype=torch.float32),
        'dfs_status': torch.tensor([dfs_status], dtype=torch.long),
        'dfs_months': torch.tensor([[dfs_months]], dtype=torch.float32),
    }
    
    return context_dict
