#!/usr/bin/env python3
"""
Identify mutation-independent pathway hyperactivation using graph transformers.

Goal: Show that graph-based models can identify patients with hyperactive
oncogenic signaling who lack the canonical driver mutations.

Hypothesis: Graph transformers can capture coordinated protein network patterns
that indicate pathway activity, even in mutation-negative patients.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from blinded_survival_classifier import config
from blinded_survival_classifier.data import load_graph_prior, get_graph_features_as_tensors


# Define oncogenic pathways and their protein markers
PATHWAY_DEFINITIONS = {
    'PI3K_AKT': {
        'proteins': [
            'AKT|Akt', 'AKT|Akt_pS473', 'AKT|Akt_pT308',
            'MTOR|mTOR', 'MTOR|mTOR_pS2448',
            'RPS6KB1|S6K_pS235_S236', 'RPS6KB1|S6K_pS240_S244',
            'EIF4EBP1|4E-BP1_pS65', 'EIF4EBP1|4E-BP1_pT37T46',
            'GSK3A|GSK-3-alpha-beta_pS21_S9',
            'PTEN|PTEN'
        ],
        'mutations': ['MUT_PIK3CA', 'MUT_AKT1', 'MUT_AKT2', 'MUT_PTEN', 'MUT_PIK3R1', 'MUT_MTOR'],
        'description': 'PI3K/AKT/mTOR pathway - cell growth and survival'
    },
    'RAS_MAPK': {
        'proteins': [
            'MAPK1|ERK2', 'MAPK3|p44_42-MAPK',
            'MAPK1|MAPK_pT202_Y204', 'MAPK3|MAPK_pT202_Y204',
            'MAP2K1|MEK1', 'MAP2K1|MEK1_pS217_S221',
            'RAF1|c-Raf', 'RAF1|c-Raf_pS338',
            'BRAF|B-Raf'
        ],
        'mutations': ['MUT_KRAS', 'MUT_NRAS', 'MUT_HRAS', 'MUT_BRAF', 'MUT_MAP2K1', 'MUT_RAF1'],
        'description': 'RAS/MAPK pathway - proliferation and differentiation'
    },
    'TP53': {
        'proteins': [
            'TP53|p53',
            'CDKN1A|p21',
            'BAX|Bax',
            'BCL2|Bcl-2',
            'PARP1|PARP_clved'
        ],
        'mutations': ['MUT_TP53'],
        'description': 'TP53 pathway - apoptosis and cell cycle'
    },
    'RTK': {
        'proteins': [
            'EGFR|EGFR', 'EGFR|EGFR_pY1068', 'EGFR|EGFR_pY1173',
            'ERBB2|HER2', 'ERBB2|HER2_pY1248',
            'ERBB3|HER3', 'ERBB3|HER3_pY1289',
            'MET|c-Met', 'MET|c-Met_pY1234_Y1235',
            'IGF1R|IGF-1R-beta'
        ],
        'mutations': ['MUT_EGFR', 'MUT_ERBB2', 'MUT_ERBB3', 'MUT_MET', 'MUT_IGF1R'],
        'description': 'Receptor tyrosine kinase signaling'
    }
}


def compute_pathway_activity_score(df, pathway_proteins):
    """
    Compute pathway activity score from RPPA data.
    Uses mean of z-scored phospho-protein levels.
    """
    available_proteins = [p for p in pathway_proteins if p in df.columns]

    if len(available_proteins) == 0:
        return None

    # Get protein data
    protein_data = df[available_proteins].copy()

    # Z-score normalization
    scaler = StandardScaler()
    protein_data_scaled = pd.DataFrame(
        scaler.fit_transform(protein_data),
        index=protein_data.index,
        columns=protein_data.columns
    )

    # Mean activity score
    activity_score = protein_data_scaled.mean(axis=1)

    return activity_score


def identify_pathway_active_patients(activity_score, threshold_percentile=75):
    """
    Identify patients with high pathway activity (top 25%).
    """
    threshold = np.percentile(activity_score.dropna(), threshold_percentile)
    is_active = (activity_score >= threshold).astype(int)
    return is_active, threshold


def classify_by_mutation_status(df, pathway_name, pathway_info, activity_labels):
    """
    Classify patients as:
    - mutation_driven: Has mutation AND high activity
    - mutation_independent: NO mutation BUT high activity
    - inactive: Low activity
    """
    mutation_cols = [col for col in pathway_info['mutations'] if col in df.columns]

    if len(mutation_cols) == 0:
        print(f"Warning: No mutation columns found for {pathway_name}")
        has_mutation = pd.Series(0, index=df.index)
    else:
        has_mutation = df[mutation_cols].sum(axis=1) > 0

    classification = pd.Series('inactive', index=df.index)

    # Mutation-driven: mutation present + high activity
    mutation_driven_mask = has_mutation & (activity_labels == 1)
    classification[mutation_driven_mask] = 'mutation_driven'

    # Mutation-independent: NO mutation + high activity
    mutation_independent_mask = (~has_mutation) & (activity_labels == 1)
    classification[mutation_independent_mask] = 'mutation_independent'

    return classification, has_mutation


def train_simple_classifiers(X_train, y_train, X_test, y_test):
    """
    Train simple baseline classifiers to predict pathway activity.
    """
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]

    results['logistic'] = {
        'auroc': roc_auc_score(y_test, y_pred_lr),
        'auprc': average_precision_score(y_test, y_pred_lr)
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]

    results['random_forest'] = {
        'auroc': roc_auc_score(y_test, y_pred_rf),
        'auprc': average_precision_score(y_test, y_pred_rf)
    }

    # PCA + Logistic
    pca = PCA(n_components=min(50, X_train.shape[1]), random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    lr_pca = LogisticRegression(max_iter=1000, random_state=42)
    lr_pca.fit(X_train_pca, y_train)
    y_pred_pca = lr_pca.predict_proba(X_test_pca)[:, 1]

    results['pca_logistic'] = {
        'auroc': roc_auc_score(y_test, y_pred_pca),
        'auprc': average_precision_score(y_test, y_pred_pca)
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Identify mutation-independent pathway hyperactivation"
    )
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to TCGA RPPA data with mutations')
    parser.add_argument('--output_dir', type=str,
                       default='../outputs/pathway_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MUTATION-INDEPENDENT PATHWAY HYPERACTIVATION ANALYSIS")
    print("="*80)
    print(f"Goal: Identify patients with hyperactive pathways lacking mutations")
    print(f"Approach: Compare graph-aware vs linear models")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples")

    # Analyze each pathway
    all_results = []

    for pathway_name, pathway_info in PATHWAY_DEFINITIONS.items():
        print(f"\n{'='*80}")
        print(f"PATHWAY: {pathway_name}")
        print(f"Description: {pathway_info['description']}")
        print(f"{'='*80}")

        # Compute pathway activity score
        activity_score = compute_pathway_activity_score(df, pathway_info['proteins'])

        if activity_score is None:
            print(f"⚠️  No proteins found for {pathway_name}, skipping...")
            continue

        # Identify high-activity patients
        activity_labels, threshold = identify_pathway_active_patients(activity_score)
        n_active = activity_labels.sum()

        print(f"\nPathway activity:")
        print(f"  Proteins found: {len([p for p in pathway_info['proteins'] if p in df.columns])}/{len(pathway_info['proteins'])}")
        print(f"  Activity threshold (75th percentile): {threshold:.3f}")
        print(f"  High-activity patients: {n_active}/{len(df)} ({n_active/len(df)*100:.1f}%)")

        # Classify by mutation status
        classification, has_mutation = classify_by_mutation_status(
            df, pathway_name, pathway_info, activity_labels
        )

        n_mutation_driven = (classification == 'mutation_driven').sum()
        n_mutation_independent = (classification == 'mutation_independent').sum()
        n_inactive = (classification == 'inactive').sum()

        print(f"\nClassification:")
        print(f"  Mutation-driven (mut+ & active):      {n_mutation_driven} ({n_mutation_driven/len(df)*100:.1f}%)")
        print(f"  Mutation-independent (mut- & active): {n_mutation_independent} ({n_mutation_independent/len(df)*100:.1f}%)")
        print(f"  Inactive (low activity):               {n_inactive} ({n_inactive/len(df)*100:.1f}%)")

        if n_mutation_independent < 10:
            print(f"⚠️  Too few mutation-independent samples ({n_mutation_independent}), skipping prediction task...")
            continue

        # Prepare data for classification
        # Task: Predict pathway activity from protein expression
        protein_cols = [c for c in df.columns if '|' in c]  # RPPA proteins
        X = df[protein_cols].fillna(0).values
        y = activity_labels.values

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nPrediction task: Identify pathway-active patients from RPPA")
        print(f"  Training samples: {len(X_train)} (active: {y_train.sum()})")
        print(f"  Test samples: {len(X_test)} (active: {y_test.sum()})")

        # Train simple classifiers
        print(f"\nTraining baseline classifiers...")
        baseline_results = train_simple_classifiers(X_train, y_train, X_test, y_test)

        print(f"\nResults (AUROC / AUPRC):")
        for model_name, metrics in baseline_results.items():
            print(f"  {model_name:20s}: {metrics['auroc']:.3f} / {metrics['auprc']:.3f}")

        # Store results
        all_results.append({
            'pathway': pathway_name,
            'description': pathway_info['description'],
            'n_active': int(n_active),
            'n_mutation_driven': int(n_mutation_driven),
            'n_mutation_independent': int(n_mutation_independent),
            'mutation_independent_pct': n_mutation_independent / n_active * 100 if n_active > 0 else 0,
            'logistic_auroc': baseline_results['logistic']['auroc'],
            'logistic_auprc': baseline_results['logistic']['auprc'],
            'rf_auroc': baseline_results['random_forest']['auroc'],
            'rf_auprc': baseline_results['random_forest']['auprc'],
            'pca_auroc': baseline_results['pca_logistic']['auroc'],
            'pca_auprc': baseline_results['pca_logistic']['auprc']
        })

    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'pathway_analysis_results.csv', index=False)

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(results_df.to_string(index=False))

        print(f"\n✅ Results saved: {output_dir / 'pathway_analysis_results.csv'}")

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Mutation-independent cases
        ax = axes[0, 0]
        x = np.arange(len(results_df))
        ax.bar(x, results_df['n_mutation_independent'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Pathway')
        ax.set_ylabel('N patients')
        ax.set_title('Mutation-Independent Hyperactivation')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['pathway'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: % Mutation-independent
        ax = axes[0, 1]
        ax.bar(x, results_df['mutation_independent_pct'], color='coral', alpha=0.7)
        ax.set_xlabel('Pathway')
        ax.set_ylabel('% of active cases')
        ax.set_title('% Mutation-Independent Among Active')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['pathway'], rotation=45, ha='right')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Plot 3: AUROC comparison
        ax = axes[1, 0]
        width = 0.25
        ax.bar(x - width, results_df['logistic_auroc'], width, label='Logistic', alpha=0.8)
        ax.bar(x, results_df['rf_auroc'], width, label='Random Forest', alpha=0.8)
        ax.bar(x + width, results_df['pca_auroc'], width, label='PCA + Logistic', alpha=0.8)
        ax.set_xlabel('Pathway')
        ax.set_ylabel('AUROC')
        ax.set_title('Pathway Activity Prediction (AUROC)')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['pathway'], rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        # Plot 4: AUPRC comparison
        ax = axes[1, 1]
        ax.bar(x - width, results_df['logistic_auprc'], width, label='Logistic', alpha=0.8)
        ax.bar(x, results_df['rf_auprc'], width, label='Random Forest', alpha=0.8)
        ax.bar(x + width, results_df['pca_auprc'], width, label='PCA + Logistic', alpha=0.8)
        ax.set_xlabel('Pathway')
        ax.set_ylabel('AUPRC')
        ax.set_title('Pathway Activity Prediction (AUPRC)')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['pathway'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'pathway_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {output_dir / 'pathway_analysis.png'}")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)
    print("1. Train graph transformer to predict pathway activity")
    print("2. Compare graph vs linear on mutation-independent cases specifically")
    print("3. Show that graph captures coordinated network patterns better")
    print("="*80)


if __name__ == "__main__":
    main()
