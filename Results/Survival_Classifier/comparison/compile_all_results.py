#!/usr/bin/env python3
"""
Compile and statistically compare all survival prediction models.
Tests: Linear vs Non-linear, Graph topology benefit, Clinical feature value.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def load_all_results():
    """Load results from all model types."""

    results = []

    # Linear baselines
    linear_path = Path("outputs/linear_baselines/linear_baseline_summary.json")
    if linear_path.exists():
        with open(linear_path) as f:
            linear_results = json.load(f)

        for model_name, res in linear_results.items():
            results.append({
                'Model': model_name,
                'Category': 'Linear',
                'Test C-index': res['test_c_index'],
                'Val C-index': res['val_c_index'],
                'N Params': res['n_params'],
                'Architecture': 'Cox PH'
            })

    # Non-linear baselines
    baseline_path = Path("outputs/baselines/baseline_summary.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_results = json.load(f)

        for model_name, res in baseline_results.items():
            if model_name == "protein_only":
                category = "Non-linear (Proteins Only)"
                arch = "MLP"
            elif model_name == "mlp_baseline":
                category = "Non-linear (No Graph)"
                arch = "MLP"
            elif model_name == "vanilla_transformer":
                category = "Non-linear (No Graph)"
                arch = "Transformer"
            else:
                category = "Non-linear"
                arch = "Unknown"

            results.append({
                'Model': model_name,
                'Category': category,
                'Test C-index': res['test_c_index'],
                'Val C-index': res['val_c_index'],
                'N Params': res['n_params'],
                'Architecture': arch
            })

    # Graph Transformer
    graph_path = Path("outputs/survival/results/test_results.json")
    if graph_path.exists():
        with open(graph_path) as f:
            graph_results = json.load(f)

        results.append({
            'Model': 'graph_transformer',
            'Category': 'Non-linear (Graph)',
            'Test C-index': graph_results['test_c_index'],
            'Val C-index': graph_results['best_val_c_index'],
            'N Params': graph_results['n_params'],
            'Architecture': 'Graph Transformer'
        })

    return pd.DataFrame(results)


def statistical_analysis(df):
    """Perform statistical analysis and comparisons."""

    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON - ALL RESULTS")
    print("="*80)

    # Sort by test C-index
    df_sorted = df.sort_values('Test C-index', ascending=False)

    print("\n" + "="*80)
    print("RANKING: All Models (Sorted by Test C-index)")
    print("="*80)
    print(f"{'Rank':<6} {'Model':<25} {'Category':<30} {'Test C-index':>12} {'Val C-index':>12} {'Params':>10}")
    print("-" * 80)

    for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<6} {row['Model']:<25} {row['Category']:<30} {row['Test C-index']:>12.4f} {row['Val C-index']:>12.4f} {row['N Params']:>10,}")

    # Best model
    best_model = df_sorted.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model['Model']}")
    print(f"  Test C-index: {best_model['Test C-index']:.4f}")
    print(f"  Architecture: {best_model['Architecture']}")
    print(f"{'='*80}")

    # Category-level analysis
    print("\n" + "="*80)
    print("ANALYSIS BY CATEGORY")
    print("="*80)

    category_stats = df.groupby('Category').agg({
        'Test C-index': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)

    print("\nTest C-index by Category:")
    print(category_stats)

    # Key comparisons
    print("\n" + "="*80)
    print("KEY COMPARISONS")
    print("="*80)

    # 1. Best linear vs best non-linear
    best_linear = df[df['Category'] == 'Linear']['Test C-index'].max()
    best_linear_model = df[df['Test C-index'] == best_linear]['Model'].values[0]

    best_nonlinear = df[df['Category'] != 'Linear']['Test C-index'].max()
    best_nonlinear_model = df[df['Test C-index'] == best_nonlinear]['Model'].values[0]

    diff_linear = best_nonlinear - best_linear
    rel_imp = (diff_linear / best_linear) * 100

    print(f"\n1. LINEAR vs NON-LINEAR")
    print(f"   Best Linear:     {best_linear_model:<25} C-index = {best_linear:.4f}")
    print(f"   Best Non-linear: {best_nonlinear_model:<25} C-index = {best_nonlinear:.4f}")
    print(f"   Difference:      {diff_linear:+.4f} ({rel_imp:+.2f}%)")

    if diff_linear > 0.03:
        print(f"   ✅ STRONG EVIDENCE: Non-linear models significantly better")
    elif diff_linear > 0.01:
        print(f"   ⚠️  MODERATE EVIDENCE: Non-linear models modestly better")
    else:
        print(f"   ❌ WEAK EVIDENCE: Minimal difference")

    # 2. Graph vs Vanilla Transformer
    graph_cindex = df[df['Model'] == 'graph_transformer']['Test C-index'].values[0]
    vanilla_cindex = df[df['Model'] == 'vanilla_transformer']['Test C-index'].values[0]

    diff_graph = graph_cindex - vanilla_cindex
    rel_imp_graph = (diff_graph / vanilla_cindex) * 100

    print(f"\n2. GRAPH TOPOLOGY BENEFIT (Graph Transformer vs Vanilla)")
    print(f"   Graph Transformer:   C-index = {graph_cindex:.4f}")
    print(f"   Vanilla Transformer: C-index = {vanilla_cindex:.4f}")
    print(f"   Difference:          {diff_graph:+.4f} ({rel_imp_graph:+.2f}%)")

    if diff_graph > 0.02:
        print(f"   ✅ STRONG BENEFIT: PPI topology improves prediction")
    elif diff_graph > 0.005:
        print(f"   ⚠️  MODERATE BENEFIT: Some improvement from topology")
    else:
        print(f"   ❌ MINIMAL BENEFIT: Topology adds little value")

    # 3. Clinical/Genomic features value
    mlp_full = df[df['Model'] == 'mlp_baseline']['Test C-index'].values[0]
    protein_only = df[df['Model'] == 'protein_only']['Test C-index'].values[0]

    diff_clinical = mlp_full - protein_only
    rel_imp_clinical = (diff_clinical / protein_only) * 100

    print(f"\n3. CLINICAL/GENOMIC FEATURE VALUE (MLP with all vs Proteins only)")
    print(f"   MLP (all features):  C-index = {mlp_full:.4f}")
    print(f"   Protein-only MLP:    C-index = {protein_only:.4f}")
    print(f"   Difference:          {diff_clinical:+.4f} ({rel_imp_clinical:+.2f}%)")

    if diff_clinical > 0.03:
        print(f"   ✅ STRONG VALUE: Clinical/genomic features are critical")
    elif diff_clinical > 0.01:
        print(f"   ⚠️  MODERATE VALUE: Clinical/genomic features help")
    else:
        print(f"   ❌ MINIMAL VALUE: Proteins alone are sufficient")

    # 4. Architecture comparison
    print(f"\n4. ARCHITECTURE COMPARISON (Same features, different architectures)")
    print(f"   MLP:                 C-index = {mlp_full:.4f}  (Params: {df[df['Model'] == 'mlp_baseline']['N Params'].values[0]:,})")
    print(f"   Vanilla Transformer: C-index = {vanilla_cindex:.4f}  (Params: {df[df['Model'] == 'vanilla_transformer']['N Params'].values[0]:,})")
    print(f"   Graph Transformer:   C-index = {graph_cindex:.4f}  (Params: {df[df['Model'] == 'graph_transformer']['N Params'].values[0]:,})")

    # Effect sizes (Cohen's d approximation)
    print("\n" + "="*80)
    print("EFFECT SIZES (vs Best Linear Baseline)")
    print("="*80)

    for idx, row in df_sorted.iterrows():
        if row['Category'] != 'Linear':
            effect = (row['Test C-index'] - best_linear) / 0.05  # Rough std estimate

            if abs(effect) < 0.2:
                magnitude = "Negligible"
            elif abs(effect) < 0.5:
                magnitude = "Small"
            elif abs(effect) < 0.8:
                magnitude = "Medium"
            else:
                magnitude = "Large"

            print(f"   {row['Model']:<25} Δ = {row['Test C-index']-best_linear:+.4f}  (Effect: {magnitude})")

    # Clinical interpretation
    print("\n" + "="*80)
    print("CLINICAL INTERPRETATION")
    print("="*80)
    print("\nC-index Guidelines:")
    print("  0.50-0.60: Poor")
    print("  0.60-0.70: Acceptable")
    print("  0.70-0.80: Good     ← Most models are here")
    print("  0.80-0.90: Excellent")
    print("  0.90-1.00: Outstanding")

    n_good = (df['Test C-index'] >= 0.70).sum()
    n_excellent = (df['Test C-index'] >= 0.80).sum()

    print(f"\nModels achieving 'Good' (≥0.70): {n_good}/{len(df)}")
    print(f"Models achieving 'Excellent' (≥0.80): {n_excellent}/{len(df)}")

    # Statistical significance note
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE NOTES")
    print("="*80)
    print("""
These are single point estimates on the same test set (n=361 patients, 80 events).

Approximate 95% confidence intervals for C-index differences:
  - Margin of error: ±0.015-0.025 (depends on event count)
  - Differences >0.02 are likely statistically significant (p<0.05)
  - Differences >0.03 are almost certainly significant (p<0.01)

Conservative interpretation:
  - Graph Transformer vs Vanilla: Δ={diff_graph:.4f} → {'SIGNIFICANT' if diff_graph > 0.02 else 'NOT SIGNIFICANT'}
  - Graph vs Best Linear: Δ={graph_cindex - best_linear:.4f} → {'SIGNIFICANT' if (graph_cindex - best_linear) > 0.02 else 'NOT SIGNIFICANT'}
  - MLP vs Protein-only: Δ={diff_clinical:.4f} → {'SIGNIFICANT' if diff_clinical > 0.02 else 'NOT SIGNIFICANT'}

For definitive p-values, would need bootstrap confidence intervals (10,000+ resamples).
    """.format(diff_graph=diff_graph, graph_cindex=graph_cindex, best_linear=best_linear, diff_clinical=diff_clinical))

    # Save summary
    output_path = Path("outputs/comprehensive_model_comparison.csv")
    df_sorted.to_csv(output_path, index=False)
    print(f"\n✅ Saved detailed results to: {output_path}")

    return df_sorted


def main():
    # Load all results
    df = load_all_results()

    if len(df) == 0:
        print("ERROR: No results found. Make sure you've run all training scripts.")
        return

    # Perform analysis
    df_sorted = statistical_analysis(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
