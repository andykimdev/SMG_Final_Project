"""
Comparative analysis between Transformer (SHAP) and PCA-Logistic models.

Identifies unique patterns learned by each model and compares against PPI network structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
try:
    from matplotlib_venn import venn2
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    print("matplotlib-venn not available, skipping Venn diagrams")

from utils import DEFAULT_PATHS, get_output_dirs


def load_results():
    """Load SHAP, PCA-Logistic, and PPI results."""
    plots_dir, _ = get_output_dirs()
    
    # SHAP
    with open(plots_dir / 'SHAP_Plots' / 'top_proteins.json', 'r') as f:
        shap_data = json.load(f)
    
    # PCA
    with open(plots_dir / 'PCA_Cox_Plots' / 'top_proteins.json', 'r') as f:
        pca_data = json.load(f)
    
    # PPI
    prior_data = np.load(DEFAULT_PATHS['prior'], allow_pickle=True)
    A = prior_data['A']
    all_proteins = prior_data['protein_cols'].tolist()
    
    return shap_data, pca_data, A, all_proteins


def get_network_properties(protein_list, A, all_proteins):
    """Compute PPI network properties for a set of proteins."""
    indices = [all_proteins.index(p) for p in protein_list if p in all_proteins]
    
    degrees = A.sum(axis=1)
    protein_degrees = [degrees[i] for i in indices]
    
    # Graph distances
    A_sparse = csr_matrix(A)
    dist = shortest_path(A_sparse, directed=False, unweighted=True, return_predecessors=False)
    dist = np.where(np.isinf(dist), -1, dist).astype(int)
    
    # Connectivity within set
    if len(indices) > 1:
        subset_dist = dist[np.ix_(indices, indices)]
        np.fill_diagonal(subset_dist, 0)
        connected = (subset_dist > 0) & (subset_dist < 100)
        avg_dist = subset_dist[connected].mean() if connected.sum() > 0 else np.nan
        pct_connected = 100 * connected.sum() / (len(indices) * (len(indices) - 1))
    else:
        avg_dist = np.nan
        pct_connected = np.nan
    
    return {
        'degrees': protein_degrees,
        'mean_degree': np.mean(protein_degrees),
        'median_degree': np.median(protein_degrees),
        'avg_distance': avg_dist,
        'pct_connected': pct_connected,
    }


def plot_venn_diagram(shap_top20, pca_top20, save_path):
    """Venn diagram of top 20 overlap."""
    if not HAS_VENN:
        print("Skipping Venn diagram (matplotlib-venn not installed)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    venn2([shap_top20, pca_top20], 
          set_labels=('Transformer\n(SHAP)', 'PCA-Logistic'),
          set_colors=('steelblue', 'teal'),
          alpha=0.7,
          ax=ax)
    ax.set_title('Top 20 Protein Overlap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_side_by_side(shap_data, pca_data, overlap, save_path):
    """Side-by-side bar charts of top 20 proteins."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    shap_proteins = [p['protein'] for p in shap_data[:20]]
    shap_importance = [p['importance'] for p in shap_data[:20]]
    pca_proteins = [p['protein'] for p in pca_data[:20]]
    pca_importance = [p['importance'] for p in pca_data[:20]]
    
    y_pos = np.arange(20)
    
    # Transformer
    ax = axes[0]
    colors = ['darkred' if p in overlap else 'steelblue' for p in shap_proteins]
    ax.barh(y_pos, shap_importance, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_proteins, fontsize=9)
    ax.set_xlabel('SHAP Importance', fontsize=12)
    ax.set_title('Transformer Top 20', fontsize=13)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    # PCA
    ax = axes[1]
    colors = ['darkred' if p in overlap else 'teal' for p in pca_proteins]
    ax.barh(y_pos, pca_importance, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pca_proteins, fontsize=9)
    ax.set_xlabel('PCA-Logistic Importance', fontsize=12)
    ax.set_title('PCA-Logistic Top 20', fontsize=13)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    plt.suptitle('Top 20 Proteins by Model (Red = Overlap)', fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_network_properties(shap_net, pca_net, overlap_net, save_path):
    """Plot network connectivity comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Boxplot
    ax = axes[0]
    data = [shap_net['degrees'], pca_net['degrees'], overlap_net['degrees']]
    labels = ['Transformer', 'PCA-Logistic', 'Overlap']
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'teal', 'darkred']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('PPI Degree', fontsize=12)
    ax.set_title('Network Connectivity of Top 20 Proteins', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    
    # Bar chart
    ax = axes[1]
    metrics = ['Mean\nDegree', 'Median\nDegree', '% Pairs\nConnected']
    shap_vals = [shap_net['mean_degree'], shap_net['median_degree'], shap_net['pct_connected']]
    pca_vals = [pca_net['mean_degree'], pca_net['median_degree'], pca_net['pct_connected']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, shap_vals, width, label='Transformer', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, pca_vals, width, label='PCA-Logistic', color='teal', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Network Properties Comparison', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_correlation(shap_data, pca_data, shap_top20, pca_top20, overlap, save_path):
    """Scatter plot of importance correlation."""
    shap_dict = {p['protein']: p['importance'] for p in shap_data}
    pca_dict = {p['protein']: p['importance'] for p in pca_data}
    
    common_proteins = list(set(shap_dict.keys()) & set(pca_dict.keys()))
    shap_scores = np.array([shap_dict[p] for p in common_proteins])
    pca_scores = np.array([pca_dict[p] for p in common_proteins])
    
    # Normalize
    shap_norm = (shap_scores - shap_scores.min()) / (shap_scores.max() - shap_scores.min())
    pca_norm = (pca_scores - pca_scores.min()) / (pca_scores.max() - pca_scores.min())
    
    pearson_r, _ = pearsonr(shap_norm, pca_norm)
    spearman_r, _ = spearmanr(shap_norm, pca_norm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['darkred' if p in overlap else 
              'steelblue' if p in shap_top20 else
              'teal' if p in pca_top20 else
              'lightgray' for p in common_proteins]
    
    ax.scatter(shap_norm, pca_norm, c=colors, alpha=0.6, s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Transformer Importance (normalized)', fontsize=12)
    ax.set_ylabel('PCA-Logistic Importance (normalized)', fontsize=12)
    ax.set_title('Model Agreement on Protein Importance', fontsize=14, pad=20)
    ax.grid(alpha=0.3)
    
    ax.text(0.05, 0.95, 
            f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkred', alpha=0.6, label='Both top 20'),
        Patch(facecolor='steelblue', alpha=0.6, label='Transformer top 20'),
        Patch(facecolor='teal', alpha=0.6, label='PCA-Logistic top 20'),
        Patch(facecolor='lightgray', alpha=0.6, label='Neither top 20')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")
    
    return pearson_r, spearman_r


def main():
    print("\n" + "=" * 70)
    print("Model Comparison Analysis")
    print("=" * 70)
    
    # Setup
    plots_dir, _ = get_output_dirs()
    output_dir = plots_dir / 'Model_Comparison_Plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    print("\nLoading results...")
    shap_data, pca_data, A, all_proteins = load_results()
    
    shap_proteins = [p['protein'] for p in shap_data]
    pca_proteins = [p['protein'] for p in pca_data]
    
    shap_top20 = set(shap_proteins[:20])
    pca_top20 = set(pca_proteins[:20])
    overlap = shap_top20 & pca_top20
    
    print(f"  SHAP: {len(shap_data)} proteins")
    print(f"  PCA-Logistic: {len(pca_data)} proteins")
    print(f"  PPI: {A.shape[0]} proteins, {int(A.sum()//2)} edges")
    
    # Overlap analysis
    print("\n" + "=" * 70)
    print("Top 20 Overlap Analysis")
    print("=" * 70)
    print(f"  Both models: {len(overlap)} proteins ({100*len(overlap)/20:.0f}%)")
    print(f"  Transformer only: {len(shap_top20 - pca_top20)} proteins")
    print(f"  PCA-Logistic only: {len(pca_top20 - shap_top20)} proteins")
    print(f"\n  Overlap: {sorted(overlap)}")
    print(f"\n  Transformer unique: {sorted(shap_top20 - pca_top20)}")
    print(f"\n  PCA-Logistic unique: {sorted(pca_top20 - shap_top20)}")
    
    # Network properties
    print("\n" + "=" * 70)
    print("Network Properties")
    print("=" * 70)
    
    shap_net = get_network_properties(shap_proteins[:20], A, all_proteins)
    pca_net = get_network_properties(pca_proteins[:20], A, all_proteins)
    overlap_net = get_network_properties(list(overlap), A, all_proteins)
    
    print(f"\nTransformer top 20:")
    print(f"  Mean degree: {shap_net['mean_degree']:.1f}")
    print(f"  % pairs connected: {shap_net['pct_connected']:.1f}%")
    print(f"  Avg distance: {shap_net['avg_distance']:.2f}")
    
    print(f"\nPCA-Logistic top 20:")
    print(f"  Mean degree: {pca_net['mean_degree']:.1f}")
    print(f"  % pairs connected: {pca_net['pct_connected']:.1f}%")
    print(f"  Avg distance: {pca_net['avg_distance']:.2f}")
    
    print(f"\nOverlap proteins:")
    print(f"  Mean degree: {overlap_net['mean_degree']:.1f}")
    print(f"  % pairs connected: {overlap_net['pct_connected']:.1f}%")
    print(f"  Avg distance: {overlap_net['avg_distance']:.2f}")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    plot_venn_diagram(shap_top20, pca_top20, output_dir / 'venn_top20.png')
    plot_side_by_side(shap_data, pca_data, overlap, output_dir / 'side_by_side_top20.png')
    plot_network_properties(shap_net, pca_net, overlap_net, output_dir / 'network_properties.png')
    pearson_r, spearman_r = plot_correlation(shap_data, pca_data, shap_top20, pca_top20, overlap, 
                                            output_dir / 'importance_correlation.png')
    
    # Unique discoveries
    print("\n" + "=" * 70)
    print("Unique Discoveries")
    print("=" * 70)
    
    pca_top50 = set(pca_proteins[:50])
    transformer_discoveries = [p for p in shap_proteins[:20] if p not in pca_top50]
    
    print(f"\nTransformer discoveries (top 20, not in PCA top 50): {len(transformer_discoveries)}")
    for protein in transformer_discoveries:
        shap_rank = shap_proteins.index(protein) + 1
        pca_rank = pca_proteins.index(protein) + 1 if protein in pca_proteins else '>50'
        idx = all_proteins.index(protein) if protein in all_proteins else -1
        degree = A[idx].sum() if idx >= 0 else 0
        print(f"  {protein:20s} | SHAP: {shap_rank:2d} | PCA: {pca_rank} | Degree: {degree:.0f}")
    
    shap_top50 = set(shap_proteins[:50])
    pca_discoveries = [p for p in pca_proteins[:20] if p not in shap_top50]
    
    print(f"\nPCA-Logistic discoveries (top 20, not in Transformer top 50): {len(pca_discoveries)}")
    for protein in pca_discoveries:
        pca_rank = pca_proteins.index(protein) + 1
        shap_rank = shap_proteins.index(protein) + 1 if protein in shap_proteins else '>50'
        idx = all_proteins.index(protein) if protein in all_proteins else -1
        degree = A[idx].sum() if idx >= 0 else 0
        print(f"  {protein:20s} | PCA: {pca_rank:2d} | SHAP: {shap_rank} | Degree: {degree:.0f}")
    
    # Summary
    summary = f"""
{'='*70}
Model Comparison Summary
{'='*70}

Overlap Analysis (Top 20):
  Both models: {len(overlap)} proteins ({100*len(overlap)/20:.0f}%)
  Transformer only: {len(shap_top20 - pca_top20)} proteins
  PCA-Logistic only: {len(pca_top20 - shap_top20)} proteins

Network Connectivity (Mean PPI Degree):
  Transformer top 20: {shap_net['mean_degree']:.1f}
  PCA-Logistic top 20: {pca_net['mean_degree']:.1f}
  Overlap proteins: {overlap_net['mean_degree']:.1f}

Network Clustering (% Pairs Connected):
  Transformer top 20: {shap_net['pct_connected']:.1f}%
  PCA-Logistic top 20: {pca_net['pct_connected']:.1f}%
  Overlap proteins: {overlap_net['pct_connected']:.1f}%

Importance Correlation:
  Pearson r: {pearson_r:.3f}
  Spearman ρ: {spearman_r:.3f}

Unique Discoveries:
  Transformer (top 20, not in PCA top 50): {len(transformer_discoveries)}
  PCA-Logistic (top 20, not in Transformer top 50): {len(pca_discoveries)}

{'='*70}
"""
    
    print(summary)
    
    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n" + "=" * 70)
    print(f"Model comparison complete! Files saved to:")
    print(f"  {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

