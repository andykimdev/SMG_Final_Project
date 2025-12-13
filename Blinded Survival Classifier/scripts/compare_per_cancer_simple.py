#!/usr/bin/env python3
"""
Simple per-cancer-type comparison.
Loads graph transformer, trains linear model, compares by cancer type.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).parent.parent))

from blinded_survival_classifier import config
from blinded_survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors
)
from blinded_survival_classifier.data.dataset import TCGASurvivalDataset
from blinded_survival_classifier.models import (
    SurvivalGraphTransformer, evaluate_survival
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--graph_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../outputs/per_cancer_analysis')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--pca_components', type=int, default=50,
                       help='Number of PCA components for linear baseline')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PER-CANCER-TYPE ANALYSIS")
    print("="*80)
    print(f"Graph model: {args.graph_model}")
    print(f"Linear baseline: PCA {args.pca_components} + Cox")
    print("="*80)

    # Load data
    print("\nLoading data...")
    graph_prior = load_graph_prior(args.prior_path)
    protein_cols = graph_prior['protein_cols']

    # Load data and keep cancer types aligned
    from blinded_survival_classifier.data import load_and_preprocess_survival_data
    from blinded_survival_classifier.data.dataset import identify_column_types
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    # Load raw data with cancer types
    df = pd.read_csv(args.csv_path)

    # Apply same filtering as dataset.py
    protein_data = df[protein_cols].copy()
    missing_per_sample = protein_data.isnull().sum(axis=1) / len(protein_cols)
    valid_protein = missing_per_sample <= config.DATA['missing_threshold']

    valid_dss_time = df['DSS_MONTHS'].notna() & (df['DSS_MONTHS'] >= 0)
    valid_dss_status = df['DSS_STATUS'].notna()
    valid_survival = valid_dss_time & valid_dss_status

    valid_samples = valid_protein & valid_survival
    df_filtered = df[valid_samples].copy()

    # Get cancer types for valid samples
    cancer_types_all = df_filtered['CANCER_TYPE_ACRONYM'].values

    print(f"Filtered data: {len(df_filtered)} samples")
    print(f"Cancer types: {len(set(cancer_types_all))} unique")

    # Now load and preprocess data (will apply same filtering)
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        args.csv_path,
        protein_cols,
        use_clinical=True,
        use_genomic=False
    )

    # Get data from splits (for training models)
    X_protein_train = data_splits['train'][0]
    X_clinical_train = data_splits['train'][1]
    X_genomic_train = data_splits['train'][2]
    event_train = data_splits['train'][3]
    time_train = data_splits['train'][4]

    # Use ALL data for per-cancer evaluation (models are already trained)
    # Combine train + val + test for more reliable per-cancer statistics
    X_protein_all = np.vstack([
        data_splits['train'][0],
        data_splits['val'][0],
        data_splits['test'][0]
    ])
    X_clinical_all = np.vstack([
        data_splits['train'][1],
        data_splits['val'][1],
        data_splits['test'][1]
    ])
    X_genomic_all = np.vstack([
        data_splits['train'][2],
        data_splits['val'][2],
        data_splits['test'][2]
    ])
    event_all = np.concatenate([
        data_splits['train'][3],
        data_splits['val'][3],
        data_splits['test'][3]
    ])
    time_all = np.concatenate([
        data_splits['train'][4],
        data_splits['val'][4],
        data_splits['test'][4]
    ])

    # Use all cancer types for evaluation
    cancer_types_eval = cancer_types_all

    print(f"\nEvaluation set: {len(X_protein_all)} samples (ALL data)")
    print(f"Cancer types: {len(set(cancer_types_eval))} unique")
    print(f"Note: Models trained on train set only, evaluating on full dataset for per-cancer analysis")

    # Load graph transformer
    print("\nLoading graph transformer...")
    n_proteins = X_protein_train.shape[1]
    n_clinical = X_clinical_train.shape[1]
    n_genomic = X_genomic_train.shape[1]

    graph_features = get_graph_features_as_tensors(graph_prior, device=device)

    graph_model = SurvivalGraphTransformer(
        n_proteins=n_proteins,
        n_clinical=n_clinical,
        n_genomic=n_genomic,
        diffusion_kernel=graph_features['K'],
        positional_encodings=graph_features['PE'],
        embedding_dim=config.MODEL['embedding_dim'],
        n_layers=config.MODEL['n_layers'],
        n_heads=config.MODEL['n_heads'],
        dropout=config.MODEL['dropout'],
        use_clinical=True,
        use_genomic=False,
    ).to(device)

    checkpoint = torch.load(args.graph_model, map_location=device)
    graph_model.load_state_dict(checkpoint['model_state_dict'])
    graph_model.eval()
    print("✅ Graph transformer loaded")

    # Train linear baseline (PCA + Cox)
    print(f"\nTraining linear baseline (PCA {args.pca_components} + Cox)...")

    # Combine all features for training
    X_train_combined = np.concatenate([
        X_protein_train,
        X_clinical_train,
        X_genomic_train
    ], axis=1)

    # Combine all features for evaluation
    X_all_combined = np.concatenate([
        X_protein_all,
        X_clinical_all,
        X_genomic_all
    ], axis=1)

    # PCA (fit on train, transform all)
    pca = PCA(n_components=args.pca_components, random_state=config.RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_combined)
    X_all_pca = pca.transform(X_all_combined)

    # Cox model (fit on train)
    y_train = Surv.from_arrays(
        event=event_train.astype(bool),
        time=time_train
    )

    linear_model = CoxPHSurvivalAnalysis()
    linear_model.fit(X_train_pca, y_train)
    print("✅ Linear model trained")

    # Evaluate per cancer type
    print("\n" + "="*80)
    print("PER-CANCER-TYPE RESULTS (ALL DATA)")
    print("="*80)

    results = []
    unique_cancers = sorted(set(cancer_types_eval))

    for cancer_type in unique_cancers:
        mask = cancer_types_eval == cancer_type
        n_samples = mask.sum()

        if n_samples < 10:
            continue

        # Subset data
        X_p = X_protein_all[mask]
        X_c = X_clinical_all[mask]
        X_g = X_genomic_all[mask]
        evt = event_all[mask]
        t = time_all[mask]

        X_combined = X_all_combined[mask]
        X_pca = X_all_pca[mask]

        n_events = evt.sum()
        event_rate = n_events / n_samples

        # Graph transformer
        dataset = TCGASurvivalDataset(X_p, X_c, X_g, t, evt)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            _, graph_c = evaluate_survival(graph_model, loader, device)

        # Linear model
        y_test = Surv.from_arrays(event=evt.astype(bool), time=t)
        try:
            linear_c = linear_model.score(X_pca, y_test)
        except:
            linear_c = np.nan

        advantage = graph_c - linear_c
        winner = "Graph" if advantage > 0 else "Linear"

        print(f"{cancer_type:10s}: n={n_samples:3d} | Graph={graph_c:.3f} | Linear={linear_c:.3f} | Δ={advantage:+.3f} ({winner})")

        results.append({
            'cancer_type': cancer_type,
            'n_samples': n_samples,
            'n_events': int(n_events),
            'event_rate': event_rate,
            'graph_c_index': graph_c,
            'linear_c_index': linear_c,
            'advantage': advantage,
            'winner': winner
        })

    results_df = pd.DataFrame(results).sort_values('advantage', ascending=False)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    n_graph = (results_df['advantage'] > 0).sum()
    n_linear = (results_df['advantage'] < 0).sum()
    print(f"Graph wins: {n_graph}/{len(results_df)}")
    print(f"Linear wins: {n_linear}/{len(results_df)}")
    print(f"Mean advantage: {results_df['advantage'].mean():.4f}")

    # Save
    results_df.to_csv(output_dir / 'per_cancer_results_all.csv', index=False)
    print(f"\n✅ Saved: {output_dir / 'per_cancer_results_all.csv'}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar comparison
    x = np.arange(len(results_df))
    width = 0.35
    ax1.bar(x - width/2, results_df['graph_c_index'], width, label='Graph Transformer', alpha=0.8)
    ax1.bar(x + width/2, results_df['linear_c_index'], width, label='Linear (PCA 50)', alpha=0.8)
    ax1.set_xlabel('Cancer Type')
    ax1.set_ylabel('C-index')
    ax1.set_title('Model Comparison by Cancer Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['cancer_type'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Advantage
    colors = ['g' if a > 0 else 'r' for a in results_df['advantage']]
    ax2.barh(results_df['cancer_type'], results_df['advantage'], color=colors, alpha=0.7)
    ax2.set_xlabel('Advantage (Graph - Linear)')
    ax2.set_title('Graph Transformer Advantage')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_cancer_comparison_all.png', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_dir / 'per_cancer_comparison_all.png'}")


if __name__ == "__main__":
    main()
