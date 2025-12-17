"""
SHAP (SHapley Additive exPlanations) analysis for protein importance.

Identifies which proteins contribute most to cancer type predictions.
Saves top proteins for focused downstream analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import shap
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import (
    load_trained_model,
    load_data,
    get_output_dirs,
)


def create_model_wrapper(model, device):
    """Wrap model for SHAP compatibility."""
    def predict_fn(x):
        model.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            x = x.to(device)
            logits = model(x)
            return logits.cpu().numpy()
    return predict_fn


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
    n_classes = len(unique_classes)
    
    # Calculate proportional allocation
    proportions = class_counts / len(y)
    target_counts = (proportions * n_samples).astype(int)
    
    # Adjust for rounding errors to ensure we get exactly n_samples
    diff = n_samples - target_counts.sum()
    if diff > 0:
        # Add extra samples to largest classes
        largest_indices = np.argsort(class_counts)[-diff:]
        target_counts[largest_indices] += 1
    elif diff < 0:
        # Remove samples from smallest classes
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
    np.random.shuffle(selected_indices)  # Shuffle to mix classes
    
    return X[selected_indices], y[selected_indices], selected_indices


def compute_shap_values(model, background_data, background_labels, test_data, test_labels, 
                       device='cpu', n_background=100, n_test=100):
    """
    Compute SHAP values using gradient-based explainer.
    
    Args:
        model: Trained model
        background_data: Reference dataset for SHAP (numpy array)
        background_labels: Labels for background data (numpy array)
        test_data: Data to explain (numpy array)
        test_labels: Labels for test data (numpy array)
        device: torch device
        n_background: Number of background samples (default 100)
        n_test: Number of test samples to explain (default 250)
        
    Returns:
        shap_values: SHAP values array (n_samples, n_proteins, n_classes)
        base_values: Expected values (n_samples, n_classes)
    """
    # Stratified sampling to maintain class proportions
    print(f"Stratified sampling: selecting {n_background} background and {n_test} test samples...")
    bg_subset, bg_labels_subset, bg_indices = stratified_sample(
        background_data, background_labels, n_background, random_state=42
    )
    test_subset, test_labels_subset, test_indices = stratified_sample(
        test_data, test_labels, n_test, random_state=42
    )
    
    # Print class distribution
    bg_unique, bg_counts = np.unique(bg_labels_subset, return_counts=True)
    test_unique, test_counts = np.unique(test_labels_subset, return_counts=True)
    print(f"\nBackground set: {len(bg_subset)} samples across {len(bg_unique)} classes")
    print(f"Test set: {len(test_subset)} samples across {len(test_unique)} classes")
    print(f"Class distribution in background: {dict(zip(bg_unique, bg_counts))}")
    print(f"Class distribution in test: {dict(zip(test_unique, test_counts))}")
    print("Setting up SHAP explainer...")
    
    # Convert to torch tensors (keep on CPU for SHAP compatibility)
    bg_tensor = torch.FloatTensor(bg_subset).cpu()
    test_tensor = torch.FloatTensor(test_subset).cpu()
    
    # Move model to CPU for SHAP (it has issues with MPS/CUDA)
    model_cpu = model.cpu()
    model_cpu.eval()
    
    # Use DeepExplainer for neural networks
    explainer = shap.DeepExplainer(model_cpu, bg_tensor)
    
    print(f"\nComputing SHAP values...")
    print(f"Expected time: ~2.5-3 hours for {n_background} bg × {n_test} test samples")
    print("Will process in batches with live progress updates...\n")
    
    # Process in batches with progress tracking
    batch_size = 10
    n_batches = (len(test_subset) + batch_size - 1) // batch_size
    all_shap_values = []
    
    import time
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(test_subset))
        batch_tensor = test_tensor[batch_start:batch_end]
        
        # Pre-batch notification
        print(f"\n→ Starting batch {batch_idx + 1}/{n_batches} (samples {batch_start}-{batch_end-1})...", flush=True)
        
        batch_shap = explainer.shap_values(batch_tensor, check_additivity=False)
        all_shap_values.append(batch_shap)
        
        # Progress update
        elapsed = time.time() - start_time
        progress = (batch_idx + 1) / n_batches
        eta = (elapsed / progress - elapsed) if progress > 0 else 0
        
        print(f"✓ Batch {batch_idx + 1}/{n_batches} complete ({progress*100:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m", flush=True)
    
    # Combine batches
    if isinstance(all_shap_values[0], list):
        # List of classes, each containing batches
        n_classes = len(all_shap_values[0])
        combined = []
        for class_idx in range(n_classes):
            class_batches = [batch[class_idx] for batch in all_shap_values]
            combined.append(np.concatenate(class_batches, axis=0))
        shap_values = np.stack(combined, axis=-1)
    else:
        shap_values = np.concatenate(all_shap_values, axis=0)
    
    print(f"\n✓ Completed in {(time.time() - start_time)/60:.1f} minutes\n")
    
    return shap_values, explainer.expected_value


def aggregate_shap_importance(shap_values, method='mean_abs'):
    """
    Aggregate SHAP values to get per-protein importance scores.
    
    Args:
        shap_values: Array of shape (n_samples, n_proteins, n_classes)
        method: 'mean_abs' or 'max_abs'
        
    Returns:
        importance: Array of shape (n_proteins,)
    """
    if method == 'mean_abs':
        # Average absolute SHAP value across samples and classes
        importance = np.abs(shap_values).mean(axis=(0, 2))
    elif method == 'max_abs':
        # Max absolute SHAP value across samples and classes
        importance = np.abs(shap_values).max(axis=(0, 2))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return importance


def plot_top_proteins(importance, protein_names, save_path, top_n=30):
    """Bar plot of most important proteins."""
    top_idx = np.argsort(importance)[-top_n:][::-1]
    top_importance = importance[top_idx]
    top_names = [protein_names[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_importance, color='steelblue', alpha=0.8)
    
    # Color top 20 differently
    for i, bar in enumerate(bars):
        if i < 20:
            bar.set_color('darkred')
            bar.set_alpha(0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Proteins', fontsize=14, pad=20)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    # Add dividing line at top 20
    if top_n > 20:
        ax.axhline(y=19.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, 19.5, 'Top 20', 
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_shap_heatmap(shap_values, protein_names, save_path, top_n=50, max_samples=50):
    """Heatmap of SHAP values for top proteins across samples."""
    # Aggregate across classes for visualization
    shap_agg = shap_values.mean(axis=2)  # (n_samples, n_proteins)
    
    # Select top proteins by importance
    importance = np.abs(shap_agg).mean(axis=0)
    top_idx = np.argsort(importance)[-top_n:][::-1]
    
    shap_subset = shap_agg[:max_samples, :][:, top_idx]
    protein_subset = [protein_names[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        shap_subset.T,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'SHAP Value'},
        xticklabels=False,
        yticklabels=protein_subset,
        ax=ax,
    )
    
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Protein', fontsize=12)
    ax.set_title(f'SHAP Values: Top {top_n} Proteins × {max_samples} Samples', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_shap_distribution(importance, save_path):
    """Distribution of protein importance scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(importance, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Distribution of Protein Importance', fontsize=13)
    ax.axvline(np.median(importance), color='red', linestyle='--', 
              linewidth=2, label=f'Median = {np.median(importance):.4f}')
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
    
    # Mark 80% and 90% thresholds
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
    """Save top proteins list for downstream analysis."""
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


def generate_shap_stats(shap_values, importance, protein_names):
    """Generate statistics about protein importance."""
    top_20_idx = np.argsort(importance)[-20:][::-1]
    
    stats = [
        "=" * 70,
        "SHAP Analysis Statistics",
        "=" * 70,
        "",
        "Dataset:",
        f"  Samples analyzed: {shap_values.shape[0]}",
        f"  Total proteins: {shap_values.shape[1]}",
        f"  Cancer types: {shap_values.shape[2]}",
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
    
    for rank, idx in enumerate(top_20_idx, 1):
        stats.append(f"  {rank:2d}. {protein_names[idx]:20s}  (importance = {importance[idx]:.6f})")
    
    stats.extend([
        "",
        "=" * 70,
    ])
    
    return stats


def main():
    print("\n" + "=" * 70)
    print("SHAP Analysis: Protein Importance")
    print("=" * 70)
    
    # Optimize for Mac
    torch.set_num_threads(8)  # Use 8 CPU threads
    
    # Setup
    plots_dir, data_dir = get_output_dirs()
    shap_plots_dir = plots_dir / "SHAP_Plots"
    shap_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {shap_plots_dir}")
    
    device = 'cpu'
    print(f"Device: {device} (optimized for SHAP compatibility)")
    print(f"Using {torch.get_num_threads()} CPU threads")
    
    # Load model and data
    print("\nLoading model and data...")
    model, graph_prior, label_info = load_trained_model(device=device)
    data_splits, _, _ = load_data(return_dataloaders=False)
    
    protein_names = graph_prior['protein_cols']
    
    train_x, train_y = data_splits['train']
    test_x, test_y = data_splits['test']
    
    print(f"Background set: {len(train_x)} samples across {len(np.unique(train_y))} classes")
    print(f"Test set: {len(test_x)} samples across {len(np.unique(test_y))} classes")
    print(f"Proteins: {len(protein_names)}")
    
    # Compute SHAP values (use CPU for SHAP compatibility)
    print("\nComputing SHAP values...")
    shap_values, base_values = compute_shap_values(
        model, 
        train_x,
        train_y,
        test_x,
        test_y,
        device='cpu',
        n_background=100,
        n_test=100
    )
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    # Aggregate importance
    print("\nAggregating protein importance...")
    importance = aggregate_shap_importance(shap_values, method='mean_abs')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_top_proteins(
        importance,
        protein_names,
        shap_plots_dir / "top_proteins.png",
        top_n=30
    )
    
    plot_shap_heatmap(
        shap_values,
        protein_names,
        shap_plots_dir / "shap_heatmap.png",
        top_n=50,
        max_samples=50
    )
    
    plot_shap_distribution(
        importance,
        shap_plots_dir / "importance_distribution.png"
    )
    
    # Save top proteins list
    print("\nSaving top proteins...")
    top_proteins = save_top_proteins(
        importance,
        protein_names,
        shap_plots_dir / "top_proteins.json",
        top_n=50
    )
    
    # Also save top 20 as simple text file
    with open(shap_plots_dir / "top_20_proteins.txt", 'w') as f:
        f.write("Top 20 Most Important Proteins\n")
        f.write("=" * 50 + "\n\n")
        for i in range(20):
            f.write(f"{i+1:2d}. {top_proteins[i]['protein']}\n")
    
    # Generate and save statistics
    print("\nGenerating statistics...")
    stats = generate_shap_stats(shap_values, importance, protein_names)
    
    stats_path = shap_plots_dir / "shap_stats.txt"
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats))
    
    print('\n'.join(stats))
    
    print("\n" + "=" * 70)
    print(f"SHAP analysis complete! Files saved to:")
    print(f"  {shap_plots_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

