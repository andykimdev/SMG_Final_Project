"""
Extract and analyze attention patterns from the trained graph transformer.

Compares learned attention with PPI network structure to assess if the model
uses graph information effectively.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

from utils import (
    load_trained_model,
    load_data,
    compute_graph_distances,
    get_output_dirs,
)


class AttentionExtractor:
    """Hook-based extraction of attention weights from transformer layers."""
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self):
        """Attach hooks to capture attention weights during forward pass."""
        for layer_idx, layer in enumerate(self.model.transformer):
            hook = layer.self_attn.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def _make_hook(self, layer_idx):
        """Create hook function for specific layer."""
        def hook_fn(module, input_tuple, output):
            x = input_tuple[0]
            attn_bias = input_tuple[1] if len(input_tuple) > 1 else None
            
            B, L, D = x.shape
            
            # Recompute attention weights (same as in model forward)
            Q = module.q_proj(x).view(B, L, module.n_heads, module.d_head).transpose(1, 2)
            K = module.k_proj(x).view(B, L, module.n_heads, module.d_head).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(module.d_head)
            
            if attn_bias is not None:
                scores = scores + attn_bias
            
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Store: shape (B, n_heads, L, L)
            self.attention_maps[layer_idx].append(attn_weights.detach().cpu())
        
        return hook_fn
    
    def clear(self):
        """Clear stored attention maps."""
        self.attention_maps.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_aggregated_attention(self, exclude_cls=True, avg_heads=True, avg_layers=False):
        """
        Aggregate attention weights across samples, heads, and/or layers.
        
        Args:
            exclude_cls: Remove CLS token (first position)
            avg_heads: Average across attention heads
            avg_layers: Average across transformer layers
            
        Returns:
            Aggregated attention matrix (n_proteins x n_proteins)
        """
        all_attns = []
        
        for layer_idx in sorted(self.attention_maps.keys()):
            layer_attns = torch.cat(self.attention_maps[layer_idx], dim=0)  # (total_samples, n_heads, L, L)
            
            if exclude_cls:
                # Remove CLS token (position 0)
                layer_attns = layer_attns[:, :, 1:, 1:]  # (B, n_heads, n_proteins, n_proteins)
            
            if avg_heads:
                layer_attns = layer_attns.mean(dim=1)  # (B, n_proteins, n_proteins)
            
            # Average across samples
            layer_attns = layer_attns.mean(dim=0)  # (n_proteins, n_proteins) or (n_heads, n_proteins, n_proteins)
            
            all_attns.append(layer_attns)
        
        if avg_layers:
            attn_matrix = torch.stack(all_attns).mean(dim=0)
        else:
            attn_matrix = torch.stack(all_attns)  # (n_layers, n_proteins, n_proteins) or (n_layers, n_heads, n_proteins, n_proteins)
        
        return attn_matrix.numpy()


def extract_attention_from_data(model, data_loader, device='cpu'):
    """Run model on data and extract attention patterns."""
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    model.eval()
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            _ = model(batch_x)
    
    # Get aggregated attention (averaged across samples and heads, per layer)
    attn_by_layer = extractor.get_aggregated_attention(
        exclude_cls=True,
        avg_heads=True,
        avg_layers=False
    )
    
    # Also get fully averaged version
    attn_avg_all = extractor.get_aggregated_attention(
        exclude_cls=True,
        avg_heads=True,
        avg_layers=True
    )
    
    extractor.remove_hooks()
    
    return attn_by_layer, attn_avg_all


def plot_attention_heatmap(attn_matrix, title, save_path, max_proteins=100):
    """Plot attention matrix as heatmap."""
    n = min(attn_matrix.shape[0], max_proteins)
    attn_subset = attn_matrix[:n, :n]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attn_subset,
        cmap='RdYlBu_r',
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=attn_subset.max(),
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Protein (to)', fontsize=12)
    ax.set_ylabel('Protein (from)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def plot_attention_vs_distance(attn_matrix, distances, A, save_path):
    """Compare attention weights with graph distances."""
    triu_idx = np.triu_indices_from(attn_matrix, k=1)
    attn_vals = attn_matrix[triu_idx]
    dist_vals = distances[triu_idx]
    adj_vals = A[triu_idx]
    
    # Separate by distance bins
    distance_bins = {}
    for d in range(1, 7):
        mask = (dist_vals == d)
        if mask.sum() > 0:
            distance_bins[d] = attn_vals[mask]
    
    disconnected_mask = (dist_vals == -1)
    if disconnected_mask.sum() > 0:
        distance_bins['disconnected'] = attn_vals[disconnected_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Box plot of attention by distance
    ax = axes[0]
    positions = []
    data_to_plot = []
    labels = []
    
    for d in sorted([k for k in distance_bins.keys() if k != 'disconnected']):
        positions.append(d)
        data_to_plot.append(distance_bins[d])
        labels.append(f'd={d}')
    
    if 'disconnected' in distance_bins:
        positions.append(max(positions) + 1)
        data_to_plot.append(distance_bins['disconnected'])
        labels.append('disconn.')
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2))
    
    ax.set_xlabel('Graph Distance', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Attention vs Graph Distance', fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis='y')
    
    # Right: Scatter plot with correlation
    ax = axes[1]
    
    # Focus on connected pairs for correlation
    connected_mask = (dist_vals > 0) & (dist_vals < 100)
    dist_connected = dist_vals[connected_mask]
    attn_connected = attn_vals[connected_mask]
    
    # Sample for plotting (too many points otherwise)
    sample_size = min(5000, len(dist_connected))
    sample_idx = np.random.choice(len(dist_connected), sample_size, replace=False)
    
    ax.scatter(dist_connected[sample_idx], attn_connected[sample_idx], 
              alpha=0.3, s=10, c='steelblue')
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(dist_connected, attn_connected)
    spearman_r, spearman_p = spearmanr(dist_connected, attn_connected)
    
    ax.set_xlabel('Graph Distance', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Attention-Distance Correlation', fontsize=13)
    ax.grid(alpha=0.3)
    
    # Add correlation text
    ax.text(0.98, 0.98,
           f'Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\n'
           f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})',
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")
    
    return pearson_r, spearman_r


def plot_layer_attention_comparison(attn_by_layer, A, save_path):
    """Show how attention patterns evolve across layers."""
    n_layers = len(attn_by_layer)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    max_val = max(layer.max() for layer in attn_by_layer)
    
    for layer_idx, attn in enumerate(attn_by_layer):
        ax = axes[layer_idx]
        
        n = min(100, attn.shape[0])
        attn_subset = attn[:n, :n]
        
        im = ax.imshow(attn_subset, cmap='RdYlBu_r', vmin=0, vmax=max_val, aspect='auto')
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=12)
        ax.set_xlabel('Protein', fontsize=10)
        ax.set_ylabel('Protein', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, label='Attention Weight')
    plt.suptitle('Attention Patterns Across Transformer Layers', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path.name}")


def compute_attention_stats(attn_matrix, A, distances):
    """Compute statistics comparing attention with graph structure."""
    triu_idx = np.triu_indices_from(attn_matrix, k=1)
    attn_vals = attn_matrix[triu_idx]
    adj_vals = A[triu_idx]
    dist_vals = distances[triu_idx]
    
    # Direct edges vs non-edges
    edge_mask = (adj_vals > 0)
    nonedge_mask = (adj_vals == 0)
    
    edge_attn = attn_vals[edge_mask]
    nonedge_attn = attn_vals[nonedge_mask]
    
    # Distance-based
    connected_mask = (dist_vals > 0) & (dist_vals < 100)
    dist_connected = dist_vals[connected_mask]
    attn_connected = attn_vals[connected_mask]
    
    pearson_r, _ = pearsonr(dist_connected, attn_connected)
    spearman_r, _ = spearmanr(dist_connected, attn_connected)
    
    stats = [
        "=" * 70,
        "Attention Analysis Statistics",
        "=" * 70,
        "",
        "Attention to PPI edges:",
        f"  Mean attention to direct neighbors: {edge_attn.mean():.6f}",
        f"  Mean attention to non-neighbors: {nonedge_attn.mean():.6f}",
        f"  Ratio (edge/non-edge): {edge_attn.mean() / nonedge_attn.mean():.2f}x",
        "",
        "Attention by graph distance:",
    ]
    
    for d in range(1, 7):
        mask = (dist_vals == d)
        if mask.sum() > 0:
            stats.append(f"  Distance {d}: mean={attn_vals[mask].mean():.6f}, median={np.median(attn_vals[mask]):.6f}")
    
    disconn_mask = (dist_vals == -1)
    if disconn_mask.sum() > 0:
        stats.append(f"  Disconnected: mean={attn_vals[disconn_mask].mean():.6f}, median={np.median(attn_vals[disconn_mask]):.6f}")
    
    stats.extend([
        "",
        "Correlation with graph distance (connected pairs):",
        f"  Pearson r: {pearson_r:.4f}",
        f"  Spearman ρ: {spearman_r:.4f}",
        "",
        "Attention matrix properties:",
        f"  Mean: {attn_matrix.mean():.6f}",
        f"  Std: {attn_matrix.std():.6f}",
        f"  Max: {attn_matrix.max():.6f}",
        f"  Min: {attn_matrix.min():.6f}",
        "",
        "=" * 70,
    ])
    
    return stats


def main():
    print("\n" + "=" * 70)
    print("Attention Pattern Analysis")
    print("=" * 70)
    
    # Setup
    plots_dir, _ = get_output_dirs()
    attn_plots_dir = plots_dir / "Attention_Plots"
    attn_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {attn_plots_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model and data
    print("\nLoading model and data...")
    model, graph_prior, label_info = load_trained_model(device=device)
    data_splits, _, loaders = load_data(return_dataloaders=True, batch_size=64)
    
    _, _, test_loader = loaders
    A = graph_prior['A']
    
    print(f"Model loaded: {len(model.transformer)} layers, {model.n_heads} heads")
    print(f"Test set: {len(test_loader.dataset)} samples")
    
    # Extract attention
    print("\nExtracting attention patterns from test set...")
    attn_by_layer, attn_avg = extract_attention_from_data(model, test_loader, device)
    print(f"Extracted attention: shape={attn_avg.shape}")
    
    # Compute graph distances
    print("\nComputing graph distances...")
    distances = compute_graph_distances(A)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_attention_heatmap(
        attn_avg,
        "Average Attention Across All Layers",
        attn_plots_dir / "attention_avg_all.png",
        max_proteins=100
    )
    
    plot_layer_attention_comparison(
        attn_by_layer,
        A,
        attn_plots_dir / "attention_by_layer.png"
    )
    
    pearson_r, spearman_r = plot_attention_vs_distance(
        attn_avg,
        distances,
        A,
        attn_plots_dir / "attention_vs_distance.png"
    )
    
    # Compute and save statistics
    print("\nComputing statistics...")
    stats = compute_attention_stats(attn_avg, A, distances)
    
    stats_path = attn_plots_dir / "attention_stats.txt"
    with open(stats_path, 'w') as f:
        f.write('\n'.join(stats))
    
    print('\n'.join(stats))
    
    print("\n" + "=" * 70)
    print(f"Attention analysis complete! Files saved to:")
    print(f"  {attn_plots_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

