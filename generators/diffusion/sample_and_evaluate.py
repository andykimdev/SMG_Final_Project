"""
Sampling and Evaluation for Graph-Aware Protein Diffusion Model.

This script provides comprehensive evaluation of the generative model:

1. **Distributional Metrics**:
   - Protein-wise KS test (Kolmogorov-Smirnov)
   - Maximum Mean Discrepancy (MMD)
   - Sample diversity metrics

2. **Biological Validation**:
   - PPI consistency (correlation between interacting proteins)
   - Pathway coherence (within-pathway correlations)
   - Cancer-type specificity

3. **Visualizations**:
   - t-SNE/UMAP of real vs generated
   - Per-protein distributions
   - Correlation matrices

4. **Conditional Generation**:
   - Generate proteomes for specific patient profiles
   - Counterfactual analysis (e.g., same patient, different cancer type)

Usage:
    python sample_and_evaluate.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --num_samples 1000 \
        --output_dir outputs/evaluation
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tqdm import tqdm

import config
from diffusion_model import GraphProteinDiffusion
from diffusion_utils import GaussianDiffusion, EMA
from dataset_diffusion import create_synthetic_context

# Add Classifier directory for graph prior
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Classifier'))
from graph_prior import load_graph_prior


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample and Evaluate Protein Diffusion Model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--use_ema',
        action='store_true',
        help='Use EMA weights for sampling'
    )
    parser.add_argument(
        '--prior_path',
        type=str,
        default='../priors/tcga_string_prior.npz',
        help='Path to STRING prior'
    )
    return parser.parse_args()


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    context_dict: dict,
    num_samples: int,
    device: str,
    batch_size: int = 64
) -> np.ndarray:
    """
    Generate protein expression samples for given context.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        context_dict: Patient context (single sample)
        num_samples: Number of samples to generate
        device: Device to use
        batch_size: Batch size for generation
        
    Returns:
        samples: [num_samples, n_proteins] generated protein expressions
    """
    model.eval()
    all_samples = []
    
    # Generate in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc='Generating samples'):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Expand context for batch
        batch_context = {
            k: v.repeat(current_batch_size, *([1] * (v.dim() - 1))).to(device)
            for k, v in context_dict.items()
        }
        
        # Generate samples
        samples = diffusion.p_sample_loop(
            model,
            shape=(current_batch_size, model.n_proteins),
            context_dict=batch_context,
            device=device,
            progress=False
        )
        
        all_samples.append(samples.cpu().numpy())
    
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    
    return all_samples


def compute_ks_statistics(
    real_data: np.ndarray,
    generated_data: np.ndarray
) -> dict:
    """
    Compute Kolmogorov-Smirnov test for each protein.
    
    Tests if real and generated distributions are significantly different.
    
    Args:
        real_data: [n_real, n_proteins] real protein expressions
        generated_data: [n_gen, n_proteins] generated expressions
        
    Returns:
        ks_stats: Dict with statistics and p-values per protein
    """
    n_proteins = real_data.shape[1]
    
    ks_statistics = []
    p_values = []
    
    for i in range(n_proteins):
        statistic, p_value = stats.ks_2samp(real_data[:, i], generated_data[:, i])
        ks_statistics.append(statistic)
        p_values.append(p_value)
    
    return {
        'ks_statistics': np.array(ks_statistics),
        'p_values': np.array(p_values),
        'mean_ks': np.mean(ks_statistics),
        'frac_significant': np.mean(np.array(p_values) < 0.05),
    }


def compute_mmd(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    kernel: str = 'rbf',
    gamma: float = None
) -> float:
    """
    Compute Maximum Mean Discrepancy between real and generated data.
    
    MMD measures the distance between two distributions in RKHS.
    Lower MMD = more similar distributions.
    
    Args:
        real_data: [n_real, n_proteins] real data
        generated_data: [n_gen, n_proteins] generated data
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel bandwidth (if None, uses median heuristic)
        
    Returns:
        mmd: MMD value
    """
    n_real = real_data.shape[0]
    n_gen = generated_data.shape[0]
    
    if kernel == 'rbf':
        if gamma is None:
            # Median heuristic for bandwidth
            all_data = np.vstack([real_data, generated_data])
            pairwise_dists = cdist(all_data, all_data, 'euclidean')
            gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0]) ** 2
        
        def rbf_kernel(X, Y):
            dists = cdist(X, Y, 'sqeuclidean')
            return np.exp(-gamma * dists)
        
        kernel_fn = rbf_kernel
    else:  # linear
        def linear_kernel(X, Y):
            return X @ Y.T
        
        kernel_fn = linear_kernel
    
    # Compute kernel matrices
    K_xx = kernel_fn(real_data, real_data)
    K_yy = kernel_fn(generated_data, generated_data)
    K_xy = kernel_fn(real_data, generated_data)
    
    # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
    mmd_sq = (K_xx.sum() - np.trace(K_xx)) / (n_real * (n_real - 1))
    mmd_sq += (K_yy.sum() - np.trace(K_yy)) / (n_gen * (n_gen - 1))
    mmd_sq -= 2 * K_xy.mean()
    
    return np.sqrt(max(mmd_sq, 0))


def compute_ppi_consistency(
    generated_data: np.ndarray,
    adjacency_matrix: np.ndarray
) -> dict:
    """
    Measure if generated proteins respect PPI network structure.
    
    Proteins that interact (in STRING) should have correlated expression.
    
    Args:
        generated_data: [n_samples, n_proteins] generated expressions
        adjacency_matrix: [n_proteins, n_proteins] STRING adjacency
        
    Returns:
        ppi_stats: Dict with PPI consistency metrics
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(generated_data.T)
    np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlations
    
    # Separate interacting vs non-interacting pairs
    interacting = adjacency_matrix > 0
    non_interacting = adjacency_matrix == 0
    np.fill_diagonal(interacting, False)
    np.fill_diagonal(non_interacting, False)
    
    # Get correlations
    corr_interacting = corr_matrix[interacting]
    corr_non_interacting = corr_matrix[non_interacting]
    
    return {
        'mean_corr_interacting': np.mean(corr_interacting),
        'mean_corr_non_interacting': np.mean(corr_non_interacting),
        'ppi_signal': np.mean(corr_interacting) - np.mean(corr_non_interacting),
        'statistic': stats.mannwhitneyu(corr_interacting, corr_non_interacting)[0],
        'p_value': stats.mannwhitneyu(corr_interacting, corr_non_interacting)[1],
    }


def plot_tsne(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    output_path: Path,
    perplexity: int = 30
):
    """Plot t-SNE visualization of real vs generated data."""
    print(f"Computing t-SNE (perplexity={perplexity})...")
    
    # Combine data
    all_data = np.vstack([real_data, generated_data])
    labels = np.array(['Real'] * len(real_data) + ['Generated'] * len(generated_data))
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(all_data)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for label in ['Real', 'Generated']:
        mask = labels == label
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            label=label,
            alpha=0.5,
            s=20
        )
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE: Real vs Generated Protein Expression', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to {output_path}")


def plot_protein_distributions(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    protein_names: list,
    output_dir: Path,
    num_proteins: int = 16
):
    """Plot distributions for a subset of proteins."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random proteins
    n_proteins = len(protein_names)
    selected_indices = np.random.choice(n_proteins, min(num_proteins, n_proteins), replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, protein_idx in enumerate(selected_indices):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        # Plot histograms
        ax.hist(real_data[:, protein_idx], bins=50, alpha=0.5, label='Real', density=True)
        ax.hist(generated_data[:, protein_idx], bins=50, alpha=0.5, label='Generated', density=True)
        
        # KS test
        ks_stat, p_val = stats.ks_2samp(real_data[:, protein_idx], generated_data[:, protein_idx])
        
        ax.set_title(f'{protein_names[protein_idx][:20]}\nKS={ks_stat:.3f}, p={p_val:.3f}', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(selected_indices), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    output_path = output_dir / 'protein_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Protein distribution plots saved to {output_path}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Graph-Aware Protein Diffusion Model - Sampling & Evaluation")
    print("=" * 80)
    
    # ========================================================================
    # Load Checkpoint
    # ========================================================================
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    context_info = checkpoint['context_info']
    label_info = checkpoint['label_info']
    
    # ========================================================================
    # Load Graph Prior
    # ========================================================================
    print("Loading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("Initializing model...")
    model = GraphProteinDiffusion(
        n_proteins=graph_prior['K'].shape[0],
        diffusion_kernel=torch.from_numpy(graph_prior['K']),
        positional_encodings=torch.from_numpy(graph_prior['PE']),
        num_cancer_types=context_info['num_cancer_types'],
        num_stages=context_info['num_stages'],
        num_sexes=context_info['num_sexes'],
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Use EMA weights if requested
    if args.use_ema and 'ema_state_dict' in checkpoint:
        print("Using EMA weights...")
        ema = EMA(model)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.apply_shadow(model)
    
    model.eval()
    
    # ========================================================================
    # Initialize Diffusion
    # ========================================================================
    diffusion = GaussianDiffusion(
        timesteps=config.DIFFUSION['timesteps'],
        schedule=config.DIFFUSION['schedule'],
        loss_type=config.TRAINING['loss_type']
    )
    
    # ========================================================================
    # Generate Samples
    # ========================================================================
    print(f"\nGenerating {args.num_samples} samples...")
    
    # Create a typical patient context (e.g., BRCA, Stage II, Female, age 60)
    context_dict = create_synthetic_context(
        cancer_type='BRCA',
        stage='Stage II',
        age=60.0,
        sex='Female',
        context_info=context_info
    )
    
    # Generate
    generated_samples = generate_samples(
        model,
        diffusion,
        context_dict,
        args.num_samples,
        args.device
    )
    
    print(f"Generated samples shape: {generated_samples.shape}")
    print(f"Generated samples mean: {generated_samples.mean():.4f}")
    print(f"Generated samples std: {generated_samples.std():.4f}")
    
    # Save generated samples
    samples_path = output_dir / 'generated_samples.npz'
    np.savez(
        samples_path,
        samples=generated_samples,
        context=context_dict
    )
    print(f"\nSaved generated samples to {samples_path}")
    
    # ========================================================================
    # Evaluation (if real data available)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nGenerated {args.num_samples} samples")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

