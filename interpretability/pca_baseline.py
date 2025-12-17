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
from sklearn.pipeline import Pipeline
import shap

from utils import (
    load_data,
    get_output_dirs,
    DEFAULT_PATHS,
)


def stratified_sample(X, y, n_samples, random_state=42):
    """
    Sample n_samples proportionally from each class.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Labels (n_samples,)
        n_samples: Total number of samples to select
        random_state: Random seed
        
    Returns:
        X_subset: Sampled features
        y_subset: Sampled labels
        indices: Indices of selected samples
    """
    if n_samples >= len(X):
        return X, y, np.arange(len(X))
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Calculate proportional allocation
    proportions = class_counts / len(y)
    target_counts = (proportions * n_samples).astype(int)
    
    # Adjust for rounding errors to ensure we get exactly n_samples
    diff = n_samples - target_counts.sum()
    if diff > 0:
        largest_indices = np.argsort(class_counts)[-diff:]
        target_counts[largest_indices] += 1
    elif diff < 0:
        smallest_indices = np.argsort(class_counts)[:abs(diff)]
        target_counts[smallest_indices] = np.maximum(1, target_counts[smallest_indices] - 1)
    
    # Sample from each class
    selected_indices = []
    np.random.seed(random_state)
    
    for class_idx, class_label in enumerate(unique_classes):
        class_mask = (y == class_label)
        class_indices = np.where(class_mask)[0]
        n_select = min(target_counts[class_idx], len(class_indices))
        
        if n_select > 0:
            selected = np.random.choice(class_indices, size=n_select, replace=False)
            selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    return X[selected_indices], y[selected_indices], selected_indices


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


def compute_pca_shap_values(pipeline, X_train, y_train, X_test, y_test, 
                            n_background=100, n_test=250):
    """
    Compute SHAP values for PCA-Logistic Regression pipeline.
    
    Args:
        pipeline: sklearn Pipeline (PCA -> LogisticRegression)
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_background: Number of background samples
        n_test: Number of test samples
        
    Returns:
        shap_values: SHAP values array (n_samples, n_proteins, n_classes)
        protein_importance: Aggregated importance per protein
    """
    print(f"\nComputing SHAP values for PCA model...")
    print(f"Stratified sampling: {n_background} background, {n_test} test samples")
    
    # Stratified sampling
    bg_subset, bg_labels_subset, _ = stratified_sample(
        X_train, y_train, n_background, random_state=42
    )
    test_subset, test_labels_subset, _ = stratified_sample(
        X_test, y_test, n_test, random_state=42
    )
    
    # Print class distribution
    bg_unique, bg_counts = np.unique(bg_labels_subset, return_counts=True)
    test_unique, test_counts = np.unique(test_labels_subset, return_counts=True)
    print(f"Background: {len(bg_subset)} samples across {len(bg_unique)} classes")
    print(f"Test: {len(test_subset)} samples across {len(test_unique)} classes")
    
    # Use LinearExplainer for fast computation (PCA+LogReg is effectively linear)
    print("Setting up SHAP LinearExplainer...")
    # Get PCA-transformed background for explainer
    bg_pca = pipeline.named_steps['pca'].transform(bg_subset)
    explainer = shap.LinearExplainer(
        pipeline.named_steps['logisticregression'],
        bg_pca
    )
    
    print(f"Computing SHAP values for {len(test_subset)} test samples...")
    test_pca = pipeline.named_steps['pca'].transform(test_subset)
    shap_values_pca = explainer.shap_values(test_pca)
    
    # Convert PCA-space SHAP values back to protein space
    # SHAP values are in PCA component space, need to map back via PCA.components_
    pca_model = pipeline.named_steps['pca']
    n_classes = pipeline.named_steps['logisticregression'].classes_.shape[0]
    
    print(f"SHAP output type: {type(shap_values_pca)}, shape: {shap_values_pca.shape}")
    
    # Handle different SHAP output formats
    if isinstance(shap_values_pca, list):
        # List format: one array per class, each (n_samples, n_components)
        shap_values_protein = []
        for class_idx, shap_pca_class in enumerate(shap_values_pca):
            # Shape: (n_samples, n_components) @ (n_components, n_proteins) -> (n_samples, n_proteins)
            shap_protein_class = shap_pca_class @ pca_model.components_
            shap_values_protein.append(shap_protein_class)
        shap_values = np.stack(shap_values_protein, axis=-1)  # (n_samples, n_proteins, n_classes)
    elif len(shap_values_pca.shape) == 3:
        # Array format: (n_samples, n_components, n_classes)
        n_samples, n_components, n_classes = shap_values_pca.shape
        shap_values_protein = []
        for class_idx in range(n_classes):
            # Extract (n_samples, n_components) for this class
            shap_pca_class = shap_values_pca[:, :, class_idx]
            # Map to protein space: (n_samples, n_components) @ (n_components, n_proteins)
            shap_protein_class = shap_pca_class @ pca_model.components_
            shap_values_protein.append(shap_protein_class)
        shap_values = np.stack(shap_values_protein, axis=-1)  # (n_samples, n_proteins, n_classes)
    else:
        # Single output: (n_samples, n_components)
        shap_values_protein = shap_values_pca @ pca_model.components_  # (n_samples, n_proteins)
        shap_values = shap_values_protein[..., np.newaxis]  # (n_samples, n_proteins, 1)
    
    # Aggregate importance (mean absolute SHAP value across samples and classes)
    if len(shap_values.shape) == 3:
        importance = np.abs(shap_values).mean(axis=(0, 2))  # Average over samples and classes
    else:
        importance = np.abs(shap_values).mean(axis=0)
    
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Protein importance shape: {importance.shape}")
    
    return shap_values, importance


def save_shap_results(shap_values, importance, protein_names, output_dir, top_n=50):
    """Save SHAP results in same format as transformer SHAP."""
    shap_dir = output_dir / "SHAP_Plots"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # Save top proteins JSON
    top_idx = np.argsort(importance)[-top_n:][::-1]
    top_proteins = []
    for rank, idx in enumerate(top_idx, 1):
        top_proteins.append({
            'rank': rank,
            'protein': protein_names[idx],
            'importance': float(importance[idx]),
            'index': int(idx),
        })
    
    with open(shap_dir / "top_proteins.json", 'w') as f:
        json.dump(top_proteins, f, indent=2)
    
    # Save top 20 as text
    with open(shap_dir / "top_20_proteins.txt", 'w') as f:
        f.write("Top 20 Most Important Proteins (PCA SHAP)\n")
        f.write("=" * 50 + "\n\n")
        for i in range(min(20, len(top_proteins))):
            f.write(f"{i+1:2d}. {top_proteins[i]['protein']}\n")
    
    # Generate stats
    stats = [
        "=" * 70,
        "PCA SHAP Analysis Statistics",
        "=" * 70,
        "",
        "Dataset:",
        f"  Samples analyzed: {shap_values.shape[0]}",
        f"  Total proteins: {shap_values.shape[1]}",
        f"  Cancer types: {shap_values.shape[2] if len(shap_values.shape) > 2 else 1}",
        "",
        "Importance distribution:",
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
    
    for rank, idx in enumerate(top_idx[:20], 1):
        stats.append(f"  {rank:2d}. {protein_names[idx]:20s}  (importance = {importance[idx]:.6f})")
    
    stats.extend([
        "",
        "=" * 70,
    ])
    
    with open(shap_dir / "shap_stats.txt", 'w') as f:
        f.write('\n'.join(stats))
    
    print(f"Saved PCA SHAP results to: {shap_dir}")
    return top_proteins


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
    
    # Create pipeline for SHAP
    pipeline = Pipeline([
        ('pca', pca),
        ('logisticregression', clf)
    ])
    
    # Compute SHAP values
    print("\n" + "=" * 70)
    print("Computing SHAP values for PCA model...")
    print("=" * 70)
    shap_values, shap_importance = compute_pca_shap_values(
        pipeline, X_train, y_train, X_test, y_test,
        n_background=100, n_test=250
    )
    
    # Save SHAP results
    print("\nSaving PCA SHAP results...")
    shap_top_proteins = save_shap_results(
        shap_values, shap_importance, protein_names, pca_plots_dir, top_n=50
    )
    
    # Compute traditional importance (for comparison)
    print("\nComputing traditional protein importance...")
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

