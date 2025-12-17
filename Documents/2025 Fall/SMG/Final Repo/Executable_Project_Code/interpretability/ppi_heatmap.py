"""
Visualize the Protein-Protein Interaction (PPI) network from the graph prior.

Creates:
- Full adjacency matrix heatmap
- Network statistics summary
- Degree distribution plot
- Distance distribution plot
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import (
    DEFAULT_PATHS,
    compute_graph_distances,
    get_output_dirs,
)


def plot_adjacency_heatmap(A: np.ndarray, save_path: Path, max_proteins: int = 500):
    """
    Plot adjacency matrix as heatmap.
    
    Args:
        A: Adjacency matrix (n_proteins x n_proteins)
        save_path: Path to save the plot
        max_proteins: Maximum proteins to show (for readability)
    """
    n = min(A.shape[0], max_proteins)
    A_subset = A[:n, :n]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        A_subset,
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(f'PPI Adjacency Matrix (first {n} proteins)', fontsize=14, pad=20)
    ax.set_xlabel('Protein Index', fontsize=12)
    ax.set_ylabel('Protein Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved adjacency heatmap: {save_path}")


def plot_degree_distribution(degrees: np.ndarray, save_path: Path):
    """Plot degree distribution of the PPI network."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(degrees, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Degree', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('PPI Network Degree Distribution', fontsize=13)
    axes[0].grid(alpha=0.3)
    
    # Log-log plot
    counts, bins = np.histogram(degrees, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mask = (counts > 0) & (bin_centers > 0)
    axes[1].loglog(bin_centers[mask], counts[mask], 'o-', color='darkblue', markersize=4)
    axes[1].set_xlabel('Degree (log)', fontsize=12)
    axes[1].set_ylabel('Frequency (log)', fontsize=12)
    axes[1].set_title('Degree Distribution (log-log)', fontsize=13)
    axes[1].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved degree distribution: {save_path}")


def plot_distance_distribution(distances: np.ndarray, save_path: Path):
    """Plot distribution of graph distances in the PPI network."""
    # Get upper triangle (avoid double counting)
    triu_idx = np.triu_indices_from(distances, k=1)
    dist_values = distances[triu_idx]
    
    # Filter out disconnected pairs (-1) and self-loops (0)
    connected = dist_values[dist_values > 0]
    disconnected_count = (dist_values == -1).sum()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    unique_dists = np.unique(connected)
    counts = [(connected == d).sum() for d in unique_dists]
    
    ax.bar(unique_dists, counts, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Graph Distance (shortest path length)', fontsize=12)
    ax.set_ylabel('Number of Protein Pairs', fontsize=12)
    ax.set_title('Distribution of Graph Distances in PPI Network', fontsize=13)
    ax.set_xticks(unique_dists)
    ax.grid(alpha=0.3, axis='y')
    
    # Add text with disconnected pairs
    total_pairs = len(dist_values)
    connected_pairs = len(connected)
    ax.text(
        0.98, 0.98,
        f'Connected: {connected_pairs:,} ({100*connected_pairs/total_pairs:.1f}%)\n'
        f'Disconnected: {disconnected_count:,} ({100*disconnected_count/total_pairs:.1f}%)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distance distribution: {save_path}")


def generate_network_stats(A: np.ndarray, distances: np.ndarray, save_path: Path):
    """Generate and save network statistics summary."""
    n_proteins = A.shape[0]
    n_edges = A.sum() // 2  # Undirected, so divide by 2
    degrees = A.sum(axis=1)
    
    # Distance statistics
    triu_idx = np.triu_indices_from(distances, k=1)
    dist_values = distances[triu_idx]
    connected = dist_values[dist_values > 0]
    
    stats = [
        "=" * 70,
        "PPI Network Statistics",
        "=" * 70,
        "",
        f"Number of proteins: {n_proteins:,}",
        f"Number of edges: {n_edges:,}",
        f"Network density: {n_edges / (n_proteins * (n_proteins - 1) / 2):.4f}",
        "",
        "Degree Statistics:",
        f"  Mean degree: {degrees.mean():.2f}",
        f"  Median degree: {np.median(degrees):.2f}",
        f"  Min degree: {degrees.min()}",
        f"  Max degree: {degrees.max()}",
        f"  Std degree: {degrees.std():.2f}",
        "",
        "Connected protein pairs:",
        f"  Total pairs: {len(dist_values):,}",
        f"  Connected: {len(connected):,} ({100*len(connected)/len(dist_values):.2f}%)",
        f"  Disconnected: {(dist_values == -1).sum():,} ({100*(dist_values == -1).sum()/len(dist_values):.2f}%)",
        "",
        "Graph distance distribution (among connected pairs):",
    ]
    
    for d in sorted(np.unique(connected)):
        count = (connected == d).sum()
        pct = 100 * count / len(connected)
        stats.append(f"  Distance {int(d)}: {count:,} pairs ({pct:.2f}%)")
    
    if len(connected) > 0:
        stats.extend([
            "",
            f"Average shortest path (connected): {connected.mean():.2f}",
            f"Diameter (max distance): {connected.max():.0f}",
        ])
    
    stats.extend([
        "",
        "=" * 70,
    ])
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(stats))
    
    # Print to console
    print('\n'.join(stats))


def main():
    """Generate all PPI visualizations and statistics."""
    print("\n" + "=" * 70)
    print("PPI Network Analysis")
    print("=" * 70)
    
    # Setup output directory
    plots_dir, _ = get_output_dirs()
    ppi_plots_dir = plots_dir / "PPI_Plots"
    ppi_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {ppi_plots_dir}")
    
    # Load prior
    print(f"\nLoading graph prior from: {DEFAULT_PATHS['prior']}")
    prior_data = np.load(DEFAULT_PATHS['prior'], allow_pickle=True)
    A = prior_data['A']
    protein_cols = prior_data['protein_cols']
    
    print(f"Loaded {len(protein_cols)} proteins, {A.sum() // 2:.0f} edges")
    
    # Compute distances
    print("\nComputing graph distances...")
    distances = compute_graph_distances(A)
    print("Done.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_adjacency_heatmap(
        A, 
        ppi_plots_dir / "adjacency_matrix.png",
        max_proteins=500
    )
    
    degrees = A.sum(axis=1)
    plot_degree_distribution(
        degrees,
        ppi_plots_dir / "degree_distribution.png"
    )
    
    plot_distance_distribution(
        distances,
        ppi_plots_dir / "distance_distribution.png"
    )
    
    # Generate statistics
    print("\nGenerating network statistics...")
    generate_network_stats(
        A,
        distances,
        ppi_plots_dir / "network_stats.txt"
    )
    
    print("\n" + "=" * 70)
    print(f"PPI analysis complete! All files saved to:")
    print(f"  {ppi_plots_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

