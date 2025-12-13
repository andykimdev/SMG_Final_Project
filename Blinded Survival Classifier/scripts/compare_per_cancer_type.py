#!/usr/bin/env python3
"""
Compare model performance across different cancer types.
Evaluates graph transformer vs linear models per cancer type.

NOTE: Cancer type is used ONLY for stratified evaluation, NOT as a model feature.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blinded_survival_classifier import config
from blinded_survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors
)
from blinded_survival_classifier.models import (
    SurvivalGraphTransformer, ConcordanceIndex
)


def load_data_with_cancer_types(csv_path, protein_cols):
    """
    Load data and keep track of cancer types for stratified evaluation.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Filter for valid samples (same logic as dataset.py)
    protein_data = df[protein_cols].copy()
    missing_per_sample = protein_data.isnull().sum(axis=1) / len(protein_cols)
    valid_protein = missing_per_sample <= config.DATA['missing_threshold']

    valid_dss_time = df['DSS_MONTHS'].notna() & (df['DSS_MONTHS'] >= 0)
    valid_dss_status = df['DSS_STATUS'].notna()
    valid_survival = valid_dss_time & valid_dss_status

    valid_samples = valid_protein & valid_survival
    df_filtered = df[valid_samples].copy()

    print(f"Filtered to {len(df_filtered)} valid samples")

    # Get cancer types
    if 'CANCER_TYPE_ACRONYM' in df_filtered.columns:
        cancer_types = df_filtered['CANCER_TYPE_ACRONYM'].values
    else:
        raise ValueError("CANCER_TYPE_ACRONYM not found in data")

    return df_filtered, cancer_types


def evaluate_on_subset(graph_model, linear_model, X_protein, X_clinical, X_genomic,
                       event, time, device):
    """Evaluate both models on a data subset."""
    from blinded_survival_classifier.models import evaluate_survival

    # Graph transformer
    dataset = TensorDataset(
        torch.FloatTensor(X_protein),
        torch.FloatTensor(X_clinical),
        torch.FloatTensor(X_genomic),
        torch.FloatTensor(event),
        torch.FloatTensor(time)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        _, graph_c_index = evaluate_survival(graph_model, loader, device)

    # Linear model
    X_combined = np.concatenate([X_protein, X_clinical, X_genomic], axis=1)
    y_linear = Surv.from_arrays(event=event.astype(bool), time=time)

    try:
        linear_c_index = linear_model.score(X_combined, y_linear)
    except Exception as e:
        print(f"Linear model error: {e}")
        linear_c_index = np.nan

    return graph_c_index, linear_c_index


def main():
    parser = argparse.ArgumentParser(description="Compare models per cancer type")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--graph_model', type=str, required=True,
                       help='Path to graph transformer checkpoint (.pt file)')
    parser.add_argument('--linear_model', type=str, required=True,
                       help='Path to linear model (.pkl or .joblib file)')
    parser.add_argument('--output_dir', type=str, default='../outputs/per_cancer_analysis')
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PER-CANCER-TYPE MODEL COMPARISON (BLINDED)")
    print("="*80)
    print("Models use ONLY: Protein expression + Age/Sex/Race/Ancestry")
    print("Cancer type used ONLY for stratified evaluation (NOT as feature)")
    print("="*80)

    # Load graph prior and data
    print("\nLoading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    protein_cols = graph_prior['protein_cols']

    print("\nLoading data with cancer type labels...")
    df_filtered, cancer_types = load_data_with_cancer_types(args.csv_path, protein_cols)

    # Process data (same as dataset.py)
    from blinded_survival_classifier.data.dataset import identify_column_types, preprocess_data

    column_types = identify_column_types(df_filtered)

    # Process features
    X_protein, X_clinical, X_genomic, survival_time, survival_event, preprocessing_info = \
        preprocess_data(df_filtered, protein_cols, column_types, use_clinical=True, use_genomic=False)

    # Split data (use same random seed as training)
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X_protein))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED, stratify=cancer_types
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED, stratify=cancer_types[temp_idx]
    )

    # Get test data
    X_protein_test = X_protein[test_idx]
    X_clinical_test = X_clinical[test_idx]
    X_genomic_test = X_genomic[test_idx]
    event_test = survival_event[test_idx]
    time_test = survival_time[test_idx]
    cancer_types_test = cancer_types[test_idx]

    print(f"\nTest set: {len(test_idx)} samples")

    # Load models
    print("\nLoading models...")

    # Graph transformer
    n_proteins = X_protein.shape[1]
    n_clinical = X_clinical.shape[1]
    n_genomic = X_genomic.shape[1]

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

    # Linear model
    linear_model = joblib.load(args.linear_model)
    print("✅ Linear model loaded")

    # Evaluate per cancer type
    print("\n" + "="*80)
    print("EVALUATING PER CANCER TYPE")
    print("="*80)

    results = []
    unique_cancers = sorted(set(cancer_types_test))

    for cancer_type in unique_cancers:
        mask = cancer_types_test == cancer_type
        n_samples = mask.sum()

        if n_samples < 10:
            print(f"Skipping {cancer_type}: only {n_samples} samples")
            continue

        # Get subset
        X_p = X_protein_test[mask]
        X_c = X_clinical_test[mask]
        X_g = X_genomic_test[mask]
        evt = event_test[mask]
        t = time_test[mask]

        n_events = evt.sum()
        event_rate = n_events / n_samples

        print(f"\n{cancer_type}: {n_samples} samples, {int(n_events)} events ({event_rate*100:.1f}%)")

        # Evaluate
        graph_c, linear_c = evaluate_on_subset(
            graph_model, linear_model, X_p, X_c, X_g, evt, t, device
        )

        advantage = graph_c - linear_c
        winner = "Graph" if advantage > 0 else "Linear"

        print(f"  Graph: {graph_c:.4f} | Linear: {linear_c:.4f} | Advantage: {advantage:+.4f} ({winner})")

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

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    n_graph_wins = (results_df['advantage'] > 0).sum()
    n_linear_wins = (results_df['advantage'] < 0).sum()
    print(f"Graph Transformer wins: {n_graph_wins}/{len(results_df)} cancer types")
    print(f"Linear model wins: {n_linear_wins}/{len(results_df)} cancer types")
    print(f"Mean advantage: {results_df['advantage'].mean():.4f}")
    print(f"\nBest for Graph Transformer: {results_df.iloc[0]['cancer_type']} (+{results_df.iloc[0]['advantage']:.4f})")
    print(f"Best for Linear: {results_df.iloc[-1]['cancer_type']} ({results_df.iloc[-1]['advantage']:.4f})")

    # Save results
    results_df.to_csv(output_dir / 'per_cancer_results.csv', index=False)
    print(f"\n✅ Results saved to {output_dir / 'per_cancer_results.csv'}")

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Comparison bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width/2, results_df['graph_c_index'], width, label='Graph Transformer', alpha=0.8, color='#2ecc71')
    ax.bar(x + width/2, results_df['linear_c_index'], width, label='Linear (Cox PH)', alpha=0.8, color='#3498db')

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('C-index', fontsize=12, fontweight='bold')
    ax.set_title('Blinded Model Comparison Across Cancer Types\n(Protein + Demographics Only)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['cancer_type'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_cancer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Advantage plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#27ae60' if adv > 0 else '#e74c3c' for adv in results_df['advantage']]
    bars = ax.barh(results_df['cancer_type'], results_df['advantage'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Advantage (Graph Transformer - Linear)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_title('Graph Transformer Advantage Over Linear Model', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, results_df['advantage'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
               ha='left' if val > 0 else 'right', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'transformer_advantage.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plots saved to {output_dir}/")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
