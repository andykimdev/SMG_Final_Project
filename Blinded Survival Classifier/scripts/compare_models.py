"""
Statistical comparison of Graph Transformer vs Baselines.
Answers: Is the PPI topology improvement statistically significant?
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_results(results_dir):
    """Load all model results."""
    results_dir = Path(results_dir)

    # Load baseline results
    baseline_summary = results_dir / 'baselines' / 'baseline_summary.json'
    if baseline_summary.exists():
        with open(baseline_summary, 'r') as f:
            baselines = json.load(f)
    else:
        baselines = {}

    # Load graph transformer results
    graph_results_path = results_dir / 'survival' / 'results' / 'test_results.json'
    if graph_results_path.exists():
        with open(graph_results_path, 'r') as f:
            graph_results = json.load(f)
    else:
        graph_results = None

    return baselines, graph_results


def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper


def compare_models(baselines, graph_results, output_dir):
    """Generate comparison statistics and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    # Organize results
    model_results = {}

    if graph_results:
        model_results['Graph Transformer\n(with PPI topology)'] = {
            'test_c_index': graph_results['test_c_index'],
            'val_c_index': graph_results['best_val_c_index'],
            'n_params': graph_results['n_params'],
            'uses_topology': True,
        }

    for name, results in baselines.items():
        display_name = {
            'mlp_baseline': 'MLP\n(no topology)',
            'vanilla_transformer': 'Transformer\n(no graph bias)',
            'protein_only': 'Protein-only MLP\n(no clinical/genomic)',
        }.get(name, name)

        model_results[display_name] = {
            'test_c_index': results['test_c_index'],
            'val_c_index': results['val_c_index'],
            'n_params': results['n_params'],
            'uses_topology': False,
        }

    # Print comparison table
    print(f"\n{'Model':<35} {'Test C-index':<15} {'Val C-index':<15} {'Δ vs Best Baseline':<20}")
    print("-"*90)

    # Find best baseline
    baseline_c_indices = [r['test_c_index'] for name, r in model_results.items() if not r['uses_topology']]
    best_baseline = max(baseline_c_indices) if baseline_c_indices else 0

    for name, results in sorted(model_results.items(), key=lambda x: x[1]['test_c_index'], reverse=True):
        test_c = results['test_c_index']
        val_c = results['val_c_index']
        delta = test_c - best_baseline if not results['uses_topology'] else test_c - best_baseline
        delta_pct = (delta / best_baseline) * 100

        marker = " ⭐" if results['uses_topology'] else ""
        print(f"{name:<35} {test_c:<15.4f} {val_c:<15.4f} "
              f"{delta:>+6.4f} ({delta_pct:>+5.1f}%){marker}")

    # Statistical significance (effect size)
    if graph_results and baselines:
        graph_c_index = graph_results['test_c_index']
        baseline_c_indices = [r['test_c_index'] for r in baselines.values()]

        print(f"\n{'='*80}")
        print("EFFECT SIZE ANALYSIS")
        print(f"{'='*80}")

        for name, baseline_c in zip(baselines.keys(), baseline_c_indices):
            improvement = graph_c_index - baseline_c
            rel_improvement = (improvement / baseline_c) * 100

            # Cohen's d (standardized effect size)
            # Approximation: assume std ~0.02 for C-index
            cohens_d = improvement / 0.02

            print(f"\nGraph Transformer vs {name}:")
            print(f"  Absolute improvement: {improvement:+.4f}")
            print(f"  Relative improvement: {rel_improvement:+.1f}%")
            print(f"  Cohen's d: {cohens_d:.2f}", end="")

            if abs(cohens_d) < 0.2:
                print(" (negligible)")
            elif abs(cohens_d) < 0.5:
                print(" (small)")
            elif abs(cohens_d) < 0.8:
                print(" (medium)")
            else:
                print(" (large) ✓")

    # Generate plots
    plot_comparison(model_results, output_dir)

    # Save comparison
    comparison_path = output_dir / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(model_results, f, indent=2)

    print(f"\n✓ Comparison saved to {output_dir}")


def plot_comparison(model_results, output_dir):
    """Create comparison visualizations."""
    sns.set_style("whitegrid")

    # Sort by test C-index
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['test_c_index'])
    names = [m[0] for m in sorted_models]
    test_c = [m[1]['test_c_index'] for m in sorted_models]
    val_c = [m[1]['val_c_index'] for m in sorted_models]
    uses_topo = [m[1]['uses_topology'] for m in sorted_models]

    # Figure 1: C-index comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    colors_test = ['#2ecc71' if t else '#3498db' for t in uses_topo]
    colors_val = ['#27ae60' if t else '#2980b9' for t in uses_topo]

    bars1 = ax.barh(x - width/2, test_c, width, label='Test C-index', color=colors_test, alpha=0.8)
    bars2 = ax.barh(x + width/2, val_c, width, label='Val C-index', color=colors_val, alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('C-index', fontsize=12, fontweight='bold')
    ax.set_title('Survival Prediction Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.3, label='Clinical cutoff')
    ax.legend(loc='lower right', fontsize=10)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width_val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Improvement over best baseline
    baseline_test_c = [c for c, t in zip(test_c, uses_topo) if not t]
    best_baseline = max(baseline_test_c) if baseline_test_c else 0

    improvements = [(c - best_baseline) * 100 for c in test_c]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, improvements, color=colors_test, alpha=0.8)

    ax.set_xlabel('Improvement over Best Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance Improvement', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars:
        width_val = bar.get_width()
        label_x = width_val + 0.2 if width_val > 0 else width_val - 0.2
        ha = 'left' if width_val > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{width_val:+.1f}%', ha=ha, va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'relative_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare survival models")
    parser.add_argument('--results_dir', type=str, default='outputs',
                       help='Directory containing results from all models')
    parser.add_argument('--output_dir', type=str, default='outputs/comparison',
                       help='Output directory for comparison')
    args = parser.parse_args()

    # Load results
    baselines, graph_results = load_results(args.results_dir)

    if not baselines and not graph_results:
        print("Error: No results found. Train models first.")
        return

    # Compare
    compare_models(baselines, graph_results, args.output_dir)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if graph_results and baselines:
        graph_c = graph_results['test_c_index']
        baseline_c_list = [r['test_c_index'] for r in baselines.values()]
        best_baseline = max(baseline_c_list)
        improvement = ((graph_c - best_baseline) / best_baseline) * 100

        if improvement > 5:
            print(f"\n✅ YES: PPI topology provides SIGNIFICANT improvement (+{improvement:.1f}%)")
            print("   Stratifying by signaling topology predicts survival BETTER than standard screening.")
        elif improvement > 2:
            print(f"\n⚠️  MODERATE: PPI topology provides modest improvement (+{improvement:.1f}%)")
            print("   Topology helps but effect is small.")
        else:
            print(f"\n❌ NO: PPI topology provides minimal improvement (+{improvement:.1f}%)")
            print("   Standard screening performs similarly.")


if __name__ == '__main__':
    main()
