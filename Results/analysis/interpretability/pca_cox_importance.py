"""
Extract protein importance from PCA-Cox model for comparison with transformer.

Trains PCA + logistic regression if no saved model exists, then extracts
which proteins contribute most to predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils import (
    load_data,
    get_output_dirs,
)


def train_pca_logistic(X_train, y_train, X_test, y_test, n_components=50):
    """Train PCA + Logistic Regression model."""
    print(f"Training PCA ({n_components} components) + Logistic Regression...")
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA variance explained: {explained_var*100:.1f}%")
    
    # Logistic Regression (multi-class)
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train_pca, y_train)
    
    # Evaluate
    train_pred = clf.predict(X_train_pca)
    test_pred = clf.predict(X_test_pca)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Train accuracy: {train_acc*100:.1f}%")
    print(f"Test accuracy: {test_acc*100:.1f}%")
    
    return pca, clf, explained_var, train_acc, test_acc


def compute_protein_importance(pca, clf):
    """
    Compute protein importance from PCA loadings and classifier weights.
    
    Importance = |PCA_loading| weighted by |classifier_coefficient|
    """
    n_proteins = pca.components_.shape[1]
    n_components = pca.components_.shape[0]
    
    # Get classifier weights (shape: n_classes x n_components)
    coef = clf.coef_
    
    # Average absolute coefficients across classes for each component
    component_importance = np.abs(coef).mean(axis=0)
    
    # Weight PCA loadings by component importance
    weighted_loadings = np.abs(pca.components_) * component_importance[:, np.newaxis]
    
    # Sum across all components to get protein importance
    protein_importance = weighted_loadings.sum(axis=0)
    
    return protein_importance, component_importance


def plot_top_proteins(importance, protein_names, save_path, top_n=30):
    """Bar plot of most important proteins."""
    top_idx = np.argsort(importance)[-top_n:][::-1]
    top_importance = importance[top_idx]
    top_names = [protein_names[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_importance, color='teal', alpha=0.8)
    
    # Highlight top 20
    for i, bar in enumerate(bars):
        if i < 20:
            bar.set_color('darkred')
            bar.set_alpha(0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Proteins: PCA-Logistic Model', fontsize=14, pad=20)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    if top_n > 20:
        ax.axhline(y=19.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, 19.5, 'Top 20', 
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_pca_components(pca, component_importance, save_path):
    """Plot PCA variance and component importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_comp = len(pca.explained_variance_ratio_)
    
    # Variance explained
    ax = axes[0]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_, 
           alpha=0.7, color='steelblue', label='Individual')
    ax.plot(range(1, n_comp + 1), cumsum, 'r-o', markersize=4, label='Cumulative')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('PCA Variance Explained', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Component importance (from classifier)
    ax = axes[1]
    ax.bar(range(1, len(component_importance) + 1), component_importance,
           alpha=0.7, color='teal')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Classifier Weight', fontsize=12)
    ax.set_title('Component Importance in Classifier', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_importance_distribution(importance, save_path):
    """Distribution of protein importance scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(importance, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Distribution of Protein Importance', fontsize=13)
    ax.axvline(np.median(importance), color='red', linestyle='--', 
              linewidth=2, label=f'Median = {np.median(importance):.3f}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Cumulative importance
    ax = axes[1]
    sorted_importance = np.sort(importance)[::-1]
    cumulative = np.cumsum(sorted_importance) / sorted_importance.sum()
    
    ax.plot(range(1, len(cumulative) + 1), cumulative, linewidth=2, color='darkblue')
    ax.set_xlabel('Number of Top Proteins', fontsize=12)
    ax.set_ylabel('Cumulative Importance', fontsize=12)
    ax.set_title('Cumulative Protein Importance', fontsize=13)
    ax.grid(alpha=0.3)
    
    # Mark thresholds
    for threshold in [0.8, 0.9]:
        idx = np.where(cumulative >= threshold)[0][0]
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5)
        ax.axvline(idx + 1, color='red', linestyle='--', alpha=0.5)
        ax.text(idx + 1, threshold, f'  {idx + 1} proteins\n  = {threshold*100:.0f}%', 
               fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def save_top_proteins(importance, protein_names, save_path, top_n=50):
    """Save top proteins as JSON."""
    top_idx = np.argsort(importance)[-top_n:][::-1]
    
    top_proteins = []
    for rank, idx in enumerate(top_idx, 1):
        top_proteins.append({
            'rank': rank,
            'protein': protein_names[idx],
            'importance': float(importance[idx]),
            'index': int(idx),
        })
    
    with open(save_path, 'w') as f:
        json.dump(top_proteins, f, indent=2)
    
    print(f"Saved: {save_path.name}")
    return top_proteins


def generate_stats(importance, protein_names, train_acc, test_acc, explained_var):
    """Generate statistics summary."""
    top_20_idx = np.argsort(importance)[-20:][::-1]
    
    stats = [
        "=" * 70,
        "PCA-Logistic Regression Analysis",
        "=" * 70,
        "",
        "Model Performance:",
        f"  Train accuracy: {train_acc*100:.2f}%",
        f"  Test accuracy: {test_acc*100:.2f}%",
        f"  PCA variance explained: {explained_var*100:.1f}%",
        "",
        "Protein Importance:",
        f"  Mean: {importance.mean():.6f}",
        f"  Median: {np.median(importance):.6f}",
        f"  Std: {importance.std():.6f}",
        f"  Min: {importance.min():.6f}",
        f"  Max: {importance.max():.6f}",
        "",
        "Concentration of importance:",
    ]
    
    sorted_importance = np.sort(importance)[::-1]
    cumsum = np.cumsum(sorted_importance) / sorted_importance.sum()
    
    for threshold in [0.5, 0.8, 0.9]:
        n_proteins = np.where(cumsum >= threshold)[0][0] + 1
        pct = 100 * n_proteins / len(importance)
        stats.append(f"  Top {n_proteins} proteins ({pct:.1f}%) explain {threshold*100:.0f}% of importance")
    
    stats.extend([
        "",
        "Top 20 Most Important Proteins:",
    ])
    
    for rank, idx in enumerate(top_20_idx, 1):
        stats.append(f"  {rank:2d}. {protein_names[idx]:20s}  (importance = {importance[idx]:.6f})")
    
    stats.extend([
        "",
        "=" * 70,
    ])
    
    return stats


def main():
    print("\n" + "=" * 70)
    print("PCA-Logistic Regression: Protein Importance")
    print("=" * 70)
    
    # Setup
    plots_dir, _ = get_output_dirs()
    pca_plots_dir = plots_dir / "PCA_Cox_Plots"
    pca_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {pca_plots_dir}")
    
    # Load data
    print("\nLoading data...")
    data_splits, label_info, _ = load_data(return_dataloaders=False)
    
    from utils import DEFAULT_PATHS
    prior_data = np.load(DEFAULT_PATHS['prior'], allow_pickle=True)
    protein_names = prior_data['protein_cols'].tolist()
    
    X_train = data_splits['train'][0]
    y_train = data_splits['train'][1]
    X_test = data_splits['test'][0]
    y_test = data_splits['test'][1]
    
    print(f"Train: {len(X_train)} samples, {X_train.shape[1]} proteins")
    print(f"Test: {len(X_test)} samples")
    print(f"Classes: {label_info['n_classes']}")
    
    # Train model
    print("\nTraining PCA-Logistic model...")
    pca, clf, explained_var, train_acc, test_acc = train_pca_logistic(
        X_train, y_train, X_test, y_test, n_components=50
    )
    
    # Compute importance
    print("\nComputing protein importance...")
    protein_importance, component_importance = compute_protein_importance(pca, clf)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_top_proteins(
        protein_importance,
        protein_names,
        pca_plots_dir / "top_proteins.png",
        top_n=30
    )
    
    plot_pca_components(
        pca,
        component_importance,
        pca_plots_dir / "pca_components.png"
    )
    
    plot_importance_distribution(
        protein_importance,
        pca_plots_dir / "importance_distribution.png"
    )
    
    # Save results
    print("\nSaving results...")
    top_proteins = save_top_proteins(
        protein_importance,
        protein_names,
        pca_plots_dir / "top_proteins.json",
        top_n=50
    )
    
    # Save top 20 as text
    with open(pca_plots_dir / "top_20_proteins.txt", 'w') as f:
        f.write("Top 20 Most Important Proteins (PCA-Logistic)\n")
        f.write("=" * 50 + "\n\n")
        for i in range(20):
            f.write(f"{i+1:2d}. {top_proteins[i]['protein']}\n")
    
    # Generate and save statistics
    print("\nGenerating statistics...")
    stats = generate_stats(
        protein_importance,
        protein_names,
        train_acc,
        test_acc,
        explained_var
    )
    
    stats_path = pca_plots_dir / "pca_stats.txt"
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats))
    
    print('\n'.join(stats))
    
    print("\n" + "=" * 70)
    print(f"PCA-Logistic analysis complete! Files saved to:")
    print(f"  {pca_plots_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

