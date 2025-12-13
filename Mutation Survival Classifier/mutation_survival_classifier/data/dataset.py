"""
TCGA RPPA + Mutations dataset module for Disease-Specific Survival (DSS) prediction.
Handles protein, clinical, genomic, and MUTATION features → predicts DSS_STATUS and DSS_MONTHS.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, List, Optional
from .. import config


class TCGASurvivalDataset(Dataset):
    """PyTorch Dataset for survival prediction with censoring."""

    def __init__(self, X_protein: np.ndarray, X_clinical: np.ndarray,
                 X_genomic: np.ndarray, survival_time: np.ndarray,
                 survival_event: np.ndarray):
        """
        Args:
            X_protein: Protein expression (n_samples, n_proteins)
            X_clinical: Clinical features (n_samples, n_clinical_features)
            X_genomic: Genomic features including mutations (n_samples, n_genomic_features)
            survival_time: DSS_MONTHS (n_samples,)
            survival_event: DSS_STATUS as 0/1 (n_samples,) - 1=death, 0=censored
        """
        self.X_protein = torch.from_numpy(X_protein).float()
        self.X_clinical = torch.from_numpy(X_clinical).float()
        self.X_genomic = torch.from_numpy(X_genomic).float()
        self.survival_time = torch.from_numpy(survival_time).float()
        self.survival_event = torch.from_numpy(survival_event).float()

    def __len__(self):
        return len(self.survival_time)

    def __getitem__(self, idx):
        return {
            'protein': self.X_protein[idx],
            'clinical': self.X_clinical[idx],
            'genomic': self.X_genomic[idx],
            'time': self.survival_time[idx],
            'event': self.survival_event[idx]
        }


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically identify column types from the TCGA dataset with mutations.
    Includes MUT_* columns as genomic features.
    """
    columns = {
        'identifiers': [],
        'protein': [],
        'clinical_numeric': [],
        'clinical_categorical': [],
        'genomic': [],
        'mutations': [],
        'survival': [],
        'target_exclusions': [],  # Cancer type info - can include or exclude
    }

    # Protein columns (contain '|' separator)
    columns['protein'] = [col for col in df.columns if '|' in col]

    # Mutation columns (start with 'MUT_')
    columns['mutations'] = [col for col in df.columns if col.startswith('MUT_')]

    # Survival columns (these are our targets, not features)
    survival_keywords = ['STATUS', 'MONTHS']
    columns['survival'] = [col for col in df.columns
                          if any(kw in col for kw in survival_keywords)]

    # Cancer type columns (can be included as features for unblinded model)
    cancer_type_keywords = ['CANCER_TYPE_ACRONYM', 'ONCOTREE_CODE', 'TUMOR_TYPE', 'SUBTYPE']
    columns['target_exclusions'] = [col for col in df.columns
                                     if col in cancer_type_keywords]

    # Genomic features (continuous values like TMB, MSI)
    genomic_keywords = ['ANEUPLOIDY', 'TMB', 'MSI', 'TBL']
    columns['genomic'] = [col for col in df.columns
                         if any(kw in col for kw in genomic_keywords)]

    # Clinical features
    clinical_numeric_candidates = ['AGE']
    clinical_categorical_candidates = ['SEX', 'RACE', 'GENETIC_ANCESTRY_LABEL',
                                      'AJCC_PATHOLOGIC_TUMOR_STAGE', 'SAMPLE_TYPE',
                                      'PERSON_NEOPLASM_CANCER_STATUS']

    # Identifiers
    id_keywords = ['ID']

    for col in df.columns:
        # Skip if already categorized
        if (col in columns['protein'] or col in columns['survival'] or
            col in columns['genomic'] or col in columns['target_exclusions'] or
            col in columns['mutations']):
            continue

        # Check if identifier
        if any(kw in col for kw in id_keywords):
            columns['identifiers'].append(col)
        # Check if known numeric clinical
        elif col in clinical_numeric_candidates:
            columns['clinical_numeric'].append(col)
        # Check if known categorical clinical
        elif col in clinical_categorical_candidates:
            columns['clinical_categorical'].append(col)

    print("\nColumn Identification:")
    print(f"  Identifiers: {len(columns['identifiers'])}")
    print(f"  Proteins: {len(columns['protein'])}")
    print(f"  Clinical (numeric): {len(columns['clinical_numeric'])} - {columns['clinical_numeric']}")
    print(f"  Clinical (categorical): {len(columns['clinical_categorical'])} - {columns['clinical_categorical']}")
    print(f"  Genomic: {len(columns['genomic'])} - {columns['genomic']}")
    print(f"  Mutations: {len(columns['mutations'])}")
    print(f"  Survival (targets): {len(columns['survival'])}")
    print(f"  Cancer type (excluded by default): {len(columns['target_exclusions'])}")

    return columns


def preprocess_clinical_features(df: pd.DataFrame,
                                 numeric_cols: List[str],
                                 categorical_cols: List[str],
                                 fit_encoders: bool = True,
                                 encoders: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """Preprocess clinical features (both numeric and categorical)."""
    if encoders is None:
        encoders = {}

    features_list = []

    # Process numeric features
    if numeric_cols:
        numeric_data = df[numeric_cols].copy()

        # Fill missing numeric values with median
        for col in numeric_cols:
            if numeric_data[col].isnull().any():
                median_val = numeric_data[col].median()
                numeric_data[col].fillna(median_val, inplace=True)

        features_list.append(numeric_data.values)

    # Process categorical features
    if categorical_cols:
        categorical_data = df[categorical_cols].copy()

        for col in categorical_cols:
            # Fill missing categorical values with 'Unknown'
            categorical_data[col].fillna('Unknown', inplace=True)

            if fit_encoders:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(categorical_data[col].astype(str))
                encoders[col] = encoder
            else:
                encoder = encoders[col]
                # Handle unseen categories
                categorical_data[col] = categorical_data[col].apply(
                    lambda x: x if x in encoder.classes_ else 'Unknown'
                )
                if 'Unknown' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'Unknown')

                encoded = encoder.transform(categorical_data[col].astype(str))

            features_list.append(encoded.reshape(-1, 1))

    # Combine all features
    if features_list:
        processed_features = np.hstack(features_list)
    else:
        processed_features = np.zeros((len(df), 0))

    return processed_features, encoders


def load_and_preprocess_survival_data(
    csv_path: str,
    protein_cols: List[str],
    use_clinical: bool = True,
    use_genomic: bool = True,
    use_mutations: bool = True,
    use_cancer_type: bool = True,
    random_seed: int = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Load and preprocess TCGA data with mutations for DSS survival prediction.

    Args:
        csv_path: Path to TCGA RPPA + mutations CSV file
        protein_cols: List of protein column names (in correct order)
        use_clinical: Whether to include clinical features
        use_genomic: Whether to include genomic features (TMB, MSI, etc.)
        use_mutations: Whether to include mutation features (MUT_*)
        use_cancer_type: Whether to include cancer type as a feature
        random_seed: Random seed for reproducibility

    Returns:
        data_splits: Dictionary with train/val/test splits
        survival_info: Dictionary with survival statistics
        preprocessing_info: Dictionary with scalers and encoders
    """
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Identify column types
    column_types = identify_column_types(df)

    # Verify required columns exist
    if 'DSS_STATUS' not in df.columns or 'DSS_MONTHS' not in df.columns:
        raise ValueError("DSS_STATUS and DSS_MONTHS columns are required for survival prediction")

    # Check if columns are swapped (common data issue)
    # DSS_STATUS should contain strings like "0:ALIVE" or "1:DEAD"
    # DSS_MONTHS should contain numeric values
    sample_status = str(df['DSS_STATUS'].iloc[0])
    sample_months = str(df['DSS_MONTHS'].iloc[0])

    if sample_status.replace('.', '').replace('-', '').isdigit() and ':' in sample_months:
        print("\n⚠ WARNING: DSS_STATUS and DSS_MONTHS columns appear to be swapped!")
        print("  Swapping columns automatically...")
        df = df.rename(columns={'DSS_STATUS': 'DSS_MONTHS_TEMP', 'DSS_MONTHS': 'DSS_STATUS'})
        df = df.rename(columns={'DSS_MONTHS_TEMP': 'DSS_MONTHS'})

    # Verify protein columns
    csv_protein_cols = column_types['protein']
    missing_cols = set(protein_cols) - set(csv_protein_cols)
    if missing_cols:
        raise ValueError(f"Missing protein columns in CSV: {missing_cols}")

    protein_data = df[protein_cols].copy()

    # ========================================================================
    # Filter samples based on data quality
    # ========================================================================
    print("\n" + "="*80)
    print("Filtering Samples")
    print("="*80)

    # 1. Filter by missing protein values
    missing_per_sample = protein_data.isnull().sum(axis=1) / len(protein_cols)
    valid_protein = missing_per_sample <= config.DATA['missing_threshold']
    print(f"Protein filter: {valid_protein.sum()}/{len(df)} samples have ≤{config.DATA['missing_threshold']*100}% missing proteins")

    # 2. Filter by valid survival data
    valid_dss_time = df['DSS_MONTHS'].notna() & (df['DSS_MONTHS'] >= 0)
    valid_dss_status = df['DSS_STATUS'].notna()
    valid_survival = valid_dss_time & valid_dss_status
    print(f"Survival filter: {valid_survival.sum()}/{len(df)} samples have valid DSS data")

    # Combine filters
    valid_samples = valid_protein & valid_survival
    print(f"Combined: {valid_samples.sum()}/{len(df)} samples pass all filters")

    df_filtered = df[valid_samples].copy()
    protein_data_filtered = protein_data[valid_samples].copy()

    # ========================================================================
    # Process survival labels
    # ========================================================================
    print("\n" + "="*80)
    print("Processing Survival Labels")
    print("="*80)

    survival_time = df_filtered['DSS_MONTHS'].values.astype(float)

    # Convert DSS_STATUS to binary (1=death, 0=censored)
    dss_status_raw = df_filtered['DSS_STATUS'].astype(str)
    survival_event = np.zeros(len(dss_status_raw), dtype=float)

    for i, status in enumerate(dss_status_raw):
        if '1:' in status or status == '1' or 'DECEASED' in status.upper() or 'DEAD' in status.upper():
            survival_event[i] = 1.0  # Event occurred (death)
        elif '0:' in status or status == '0' or 'LIVING' in status.upper() or 'ALIVE' in status.upper():
            survival_event[i] = 0.0  # Censored
        else:
            try:
                survival_event[i] = float(status)
            except:
                print(f"Warning: Unknown DSS_STATUS value '{status}', treating as censored")
                survival_event[i] = 0.0

    print(f"Survival statistics:")
    print(f"  Events (deaths): {int(survival_event.sum())} ({survival_event.mean()*100:.1f}%)")
    print(f"  Censored: {int((1-survival_event).sum())} ({(1-survival_event.mean())*100:.1f}%)")
    print(f"  Median survival time: {np.median(survival_time):.1f} months")
    print(f"  Time range: {survival_time.min():.1f} - {survival_time.max():.1f} months")

    # Check if we have enough events
    if survival_event.sum() < config.DATA['min_events']:
        print(f"\n⚠ WARNING: Only {int(survival_event.sum())} events. Cox models need ~{config.DATA['min_events']}+ events.")

    # ========================================================================
    # Split data (stratified by event status to balance censoring)
    # ========================================================================
    print("\n" + "="*80)
    print("Splitting Data")
    print("="*80)

    # Extract patient IDs for splitting
    if 'PATIENT_ID' not in df_filtered.columns:
        print("Warning: PATIENT_ID not found, using sample-level split")
        patient_ids = np.arange(len(df_filtered))
        unique_patients = patient_ids
    else:
        patient_ids = df_filtered['PATIENT_ID'].values
        unique_patients = np.unique(patient_ids)

    # Get event status per patient (for stratification)
    patient_events = np.array([survival_event[patient_ids == p][0] for p in unique_patients])

    # First split: train vs (val+test)
    train_patients, valtest_patients = train_test_split(
        unique_patients,
        test_size=config.VAL_RATIO + config.TEST_RATIO,
        random_state=random_seed,
        stratify=patient_events
    )

    # Get event status for valtest patients
    valtest_events = np.array([survival_event[patient_ids == p][0] for p in valtest_patients])

    # Second split: val vs test
    val_size_adjusted = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_patients, test_patients = train_test_split(
        valtest_patients,
        test_size=1 - val_size_adjusted,
        random_state=random_seed,
        stratify=valtest_events
    )

    # Create masks
    train_mask = np.isin(patient_ids, train_patients)
    val_mask = np.isin(patient_ids, val_patients)
    test_mask = np.isin(patient_ids, test_patients)

    print(f"Data splits:")
    print(f"  Train: {train_mask.sum()} samples ({train_mask.sum()/len(df_filtered)*100:.1f}%), "
          f"{int(survival_event[train_mask].sum())} events")
    print(f"  Val:   {val_mask.sum()} samples ({val_mask.sum()/len(df_filtered)*100:.1f}%), "
          f"{int(survival_event[val_mask].sum())} events")
    print(f"  Test:  {test_mask.sum()} samples ({test_mask.sum()/len(df_filtered)*100:.1f}%), "
          f"{int(survival_event[test_mask].sum())} events")

    # ========================================================================
    # Process Protein Data
    # ========================================================================
    print("\n" + "="*80)
    print("Processing Protein Expression Data")
    print("="*80)

    X_protein_train = protein_data_filtered[train_mask].values
    X_protein_val = protein_data_filtered[val_mask].values
    X_protein_test = protein_data_filtered[test_mask].values

    # Impute missing protein values
    train_protein_means = np.nanmean(X_protein_train, axis=0)
    X_protein_train = np.where(np.isnan(X_protein_train), train_protein_means, X_protein_train)
    X_protein_val = np.where(np.isnan(X_protein_val), train_protein_means, X_protein_val)
    X_protein_test = np.where(np.isnan(X_protein_test), train_protein_means, X_protein_test)

    # Standardize proteins
    protein_scaler = StandardScaler()
    X_protein_train = protein_scaler.fit_transform(X_protein_train)
    X_protein_val = protein_scaler.transform(X_protein_val)
    X_protein_test = protein_scaler.transform(X_protein_test)

    print(f"Protein features: {X_protein_train.shape[1]}")

    # ========================================================================
    # Process Clinical Data
    # ========================================================================
    clinical_cols = []
    if use_clinical:
        clinical_cols = column_types['clinical_numeric'] + column_types['clinical_categorical']
        if use_cancer_type:
            # Add cancer type as a clinical feature
            if 'CANCER_TYPE_ACRONYM' in df_filtered.columns:
                clinical_cols.append('CANCER_TYPE_ACRONYM')
                print("\n✓ Including CANCER_TYPE as a feature (unblinded model)")

    if clinical_cols:
        print("\n" + "="*80)
        print("Processing Clinical Features")
        print("="*80)

        # Separate numeric and categorical
        numeric_cols = [c for c in clinical_cols if c in column_types['clinical_numeric']]
        categorical_cols = [c for c in clinical_cols if c not in numeric_cols]

        X_clinical_train, clinical_encoders = preprocess_clinical_features(
            df_filtered[train_mask],
            numeric_cols,
            categorical_cols,
            fit_encoders=True
        )

        X_clinical_val, _ = preprocess_clinical_features(
            df_filtered[val_mask],
            numeric_cols,
            categorical_cols,
            fit_encoders=False,
            encoders=clinical_encoders
        )

        X_clinical_test, _ = preprocess_clinical_features(
            df_filtered[test_mask],
            numeric_cols,
            categorical_cols,
            fit_encoders=False,
            encoders=clinical_encoders
        )

        # Standardize clinical features
        clinical_scaler = StandardScaler()
        X_clinical_train = clinical_scaler.fit_transform(X_clinical_train)
        X_clinical_val = clinical_scaler.transform(X_clinical_val)
        X_clinical_test = clinical_scaler.transform(X_clinical_test)

        print(f"Clinical features: {X_clinical_train.shape[1]}")
    else:
        X_clinical_train = np.zeros((train_mask.sum(), 0))
        X_clinical_val = np.zeros((val_mask.sum(), 0))
        X_clinical_test = np.zeros((test_mask.sum(), 0))
        clinical_scaler = None
        clinical_encoders = {}

    # ========================================================================
    # Process Genomic Data (TMB, MSI, etc.) + Mutations
    # ========================================================================
    genomic_cols = []
    if use_genomic:
        genomic_cols.extend(column_types['genomic'])
    if use_mutations:
        genomic_cols.extend(column_types['mutations'])

    if genomic_cols:
        print("\n" + "="*80)
        print("Processing Genomic Features + Mutations")
        print("="*80)

        genomic_data = df_filtered[genomic_cols].copy()

        X_genomic_train = genomic_data[train_mask].values
        X_genomic_val = genomic_data[val_mask].values
        X_genomic_test = genomic_data[test_mask].values

        # Impute missing genomic values (use median for continuous, 0 for mutations)
        train_genomic_medians = np.nanmedian(X_genomic_train, axis=0)
        # For mutation columns (binary 0/1), replace NaN with 0
        for i, col in enumerate(genomic_cols):
            if col.startswith('MUT_'):
                train_genomic_medians[i] = 0.0

        X_genomic_train = np.where(np.isnan(X_genomic_train), train_genomic_medians, X_genomic_train)
        X_genomic_val = np.where(np.isnan(X_genomic_val), train_genomic_medians, X_genomic_val)
        X_genomic_test = np.where(np.isnan(X_genomic_test), train_genomic_medians, X_genomic_test)

        # Standardize genomic features (including mutations)
        genomic_scaler = StandardScaler()
        X_genomic_train = genomic_scaler.fit_transform(X_genomic_train)
        X_genomic_val = genomic_scaler.transform(X_genomic_val)
        X_genomic_test = genomic_scaler.transform(X_genomic_test)

        print(f"Genomic features (continuous): {len(column_types['genomic'])}")
        print(f"Mutation features (binary): {len(column_types['mutations'])}")
        print(f"Total genomic+mutation features: {X_genomic_train.shape[1]}")
    else:
        X_genomic_train = np.zeros((train_mask.sum(), 0))
        X_genomic_val = np.zeros((val_mask.sum(), 0))
        X_genomic_test = np.zeros((test_mask.sum(), 0))
        genomic_scaler = None

    # ========================================================================
    # Package Results
    # ========================================================================
    data_splits = {
        'train': (
            X_protein_train, X_clinical_train, X_genomic_train,
            survival_time[train_mask], survival_event[train_mask]
        ),
        'val': (
            X_protein_val, X_clinical_val, X_genomic_val,
            survival_time[val_mask], survival_event[val_mask]
        ),
        'test': (
            X_protein_test, X_clinical_test, X_genomic_test,
            survival_time[test_mask], survival_event[test_mask]
        ),
    }

    survival_info = {
        'total_samples': len(df_filtered),
        'total_events': int(survival_event.sum()),
        'total_censored': int((1 - survival_event).sum()),
        'event_rate': survival_event.mean(),
        'median_time': float(np.median(survival_time)),
        'time_range': (float(survival_time.min()), float(survival_time.max())),
        'train_events': int(survival_event[train_mask].sum()),
        'val_events': int(survival_event[val_mask].sum()),
        'test_events': int(survival_event[test_mask].sum()),
    }

    preprocessing_info = {
        'protein_scaler': protein_scaler,
        'clinical_scaler': clinical_scaler,
        'genomic_scaler': genomic_scaler,
        'clinical_encoders': clinical_encoders,
        'column_types': column_types,
        'feature_dims': {
            'protein': X_protein_train.shape[1],
            'clinical': X_clinical_train.shape[1],
            'genomic': X_genomic_train.shape[1],
        },
        'use_cancer_type': use_cancer_type,
        'use_mutations': use_mutations,
    }

    print("\n" + "="*80)
    print("Preprocessing Complete")
    print("="*80)
    print(f"Total features per sample:")
    print(f"  Proteins: {preprocessing_info['feature_dims']['protein']}")
    print(f"  Clinical: {preprocessing_info['feature_dims']['clinical']}")
    print(f"  Genomic+Mutations:  {preprocessing_info['feature_dims']['genomic']}")
    print(f"  TOTAL:    {sum(preprocessing_info['feature_dims'].values())}")
    print(f"\nSurvival task: Predict DSS_STATUS and DSS_MONTHS")
    print(f"  Event rate: {survival_info['event_rate']*100:.1f}%")
    print(f"  Median follow-up: {survival_info['median_time']:.1f} months")

    return data_splits, survival_info, preprocessing_info


def create_survival_dataloaders(
    data_splits: Dict,
    batch_size: int = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for survival prediction."""
    if batch_size is None:
        batch_size = config.TRAINING['batch_size']

    # Create datasets
    train_dataset = TCGASurvivalDataset(*data_splits['train'])
    val_dataset = TCGASurvivalDataset(*data_splits['val'])
    test_dataset = TCGASurvivalDataset(*data_splits['test'])

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
