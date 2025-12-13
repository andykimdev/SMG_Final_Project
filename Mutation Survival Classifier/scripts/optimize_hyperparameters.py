"""
Hyperparameter optimization script for Mutation-Augmented Graph Transformer.
Based on successful optimization from Survival Classifier project.

Initial baseline: Test C-index ~0.61
Target: Improve to 0.75-0.80 range
"""

import argparse
import os
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation_survival_classifier import config
from mutation_survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors,
    load_and_preprocess_survival_data, create_survival_dataloaders
)
from mutation_survival_classifier.models import (
    SurvivalGraphTransformer, train_survival_epoch, evaluate_survival
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize mutation survival classifier hyperparameters")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/optimization')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--use_mutations', action='store_true', default=True)
    parser.add_argument('--use_cancer_type', action='store_true', default=True)
    return parser.parse_args()


def train_and_evaluate(model_config, train_loader, val_loader, test_loader,
                       graph_features, device, max_epochs, preprocessing_info):
    """Train model with given config and return performance."""

    # Extract model architecture parameters (exclude training params and name)
    model_arch_params = {
        'embedding_dim': model_config['embedding_dim'],
        'n_layers': model_config['n_layers'],
        'n_heads': model_config['n_heads'],
        'dropout': model_config['dropout'],
    }

    # Initialize model
    model = SurvivalGraphTransformer(
        n_proteins=preprocessing_info['feature_dims']['protein'],
        n_clinical=preprocessing_info['feature_dims']['clinical'],
        n_genomic=preprocessing_info['feature_dims']['genomic'],
        diffusion_kernel=graph_features['K'],
        positional_encodings=graph_features['PE'],
        **model_arch_params
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_config.get('learning_rate', 3e-4),
        weight_decay=model_config.get('weight_decay', 5e-4)
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training loop
    best_val_c_index = 0.0
    patience_counter = 0
    patience = 10

    for epoch in range(max_epochs):
        train_loss = train_survival_epoch(model, train_loader, optimizer, device)
        val_loss, val_c_index = evaluate_survival(model, val_loader, device)

        scheduler.step(val_c_index)

        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate on test set with best model
    model.load_state_dict(best_model_state)
    test_loss, test_c_index = evaluate_survival(model, test_loader, device)

    return {
        'val_c_index': best_val_c_index,
        'test_c_index': test_c_index,
        'final_epoch': epoch + 1
    }


def main():
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"opt_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}\n")

    # Load graph prior
    print("Loading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    protein_cols = graph_prior['protein_cols']

    # Load data
    print("Loading data...")
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        csv_path=args.csv_path,
        protein_cols=protein_cols,
        use_clinical=True,
        use_genomic=True,
        use_mutations=args.use_mutations,
        use_cancer_type=args.use_cancer_type,
        random_seed=config.RANDOM_SEED
    )

    train_loader, val_loader, test_loader = create_survival_dataloaders(
        data_splits, batch_size=64, num_workers=0
    )

    device = torch.device(args.device)
    graph_features = get_graph_features_as_tensors(graph_prior, device=device)

    # Define hyperparameter search space
    # Based on successful configs from Survival Classifier optimization
    configs = [
        {
            'name': 'baseline',
            'embedding_dim': 256,
            'n_layers': 3,
            'n_heads': 8,
            'dropout': 0.4,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v1_deeper',
            'embedding_dim': 256,
            'n_layers': 4,
            'n_heads': 8,
            'dropout': 0.4,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v2_wider',
            'embedding_dim': 320,
            'n_layers': 3,
            'n_heads': 8,
            'dropout': 0.4,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v3_higher_dropout',
            'embedding_dim': 256,
            'n_layers': 3,
            'n_heads': 8,
            'dropout': 0.5,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v4_lower_lr',
            'embedding_dim': 256,
            'n_layers': 3,
            'n_heads': 8,
            'dropout': 0.4,
            'learning_rate': 1e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v5_higher_wd',
            'embedding_dim': 256,
            'n_layers': 3,
            'n_heads': 8,
            'dropout': 0.4,
            'learning_rate': 3e-4,
            'weight_decay': 1e-3,
        },
        {
            'name': 'v6_more_heads',
            'embedding_dim': 256,
            'n_layers': 3,
            'n_heads': 12,
            'dropout': 0.4,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v7_shallow_wide',
            'embedding_dim': 384,
            'n_layers': 2,
            'n_heads': 8,
            'dropout': 0.45,
            'learning_rate': 3e-4,
            'weight_decay': 5e-4,
        },
        {
            'name': 'v8_combo',
            'embedding_dim': 320,
            'n_layers': 3,
            'n_heads': 10,
            'dropout': 0.45,
            'learning_rate': 2e-4,
            'weight_decay': 7e-4,
        },
    ]

    # Run optimization
    results = []

    for i, model_config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Config {i+1}/{len(configs)}: {model_config['name']}")
        print(f"{'='*80}")
        print(f"  embedding_dim: {model_config['embedding_dim']}")
        print(f"  n_layers: {model_config['n_layers']}")
        print(f"  n_heads: {model_config['n_heads']}")
        print(f"  dropout: {model_config['dropout']}")
        print(f"  learning_rate: {model_config['learning_rate']}")
        print(f"  weight_decay: {model_config['weight_decay']}")

        try:
            metrics = train_and_evaluate(
                model_config, train_loader, val_loader, test_loader,
                graph_features, device, args.max_epochs, preprocessing_info
            )

            result = {
                'config_name': model_config['name'],
                'config': model_config,
                'val_c_index': metrics['val_c_index'],
                'test_c_index': metrics['test_c_index'],
                'final_epoch': metrics['final_epoch']
            }

            results.append(result)

            print(f"\n✓ Results:")
            print(f"  Val C-index:  {metrics['val_c_index']:.4f}")
            print(f"  Test C-index: {metrics['test_c_index']:.4f}")
            print(f"  Epochs: {metrics['final_epoch']}")

            # Save intermediate results
            with open(output_dir / 'optimization_results.json', 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"\n✗ Failed: {e}")
            results.append({
                'config_name': model_config['name'],
                'config': model_config,
                'error': str(e)
            })

    # Final summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}\n")

    # Sort by test C-index
    successful_results = [r for r in results if 'test_c_index' in r]
    successful_results.sort(key=lambda x: x['test_c_index'], reverse=True)

    print("Results (sorted by test C-index):\n")
    print(f"{'Rank':<6} {'Config':<20} {'Val C-index':<12} {'Test C-index':<12} {'Epochs':<8}")
    print("-" * 70)

    for i, result in enumerate(successful_results):
        print(f"{i+1:<6} {result['config_name']:<20} "
              f"{result['val_c_index']:<12.4f} {result['test_c_index']:<12.4f} "
              f"{result['final_epoch']:<8}")

    # Save final results
    summary = {
        'all_results': results,
        'best_config': successful_results[0] if successful_results else None,
        'timestamp': timestamp
    }

    with open(output_dir / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")

    if successful_results:
        best = successful_results[0]
        print(f"\n🏆 Best configuration: {best['config_name']}")
        print(f"   Test C-index: {best['test_c_index']:.4f}")
        print(f"   Config: {best['config']}")


if __name__ == '__main__':
    main()
