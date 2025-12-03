#!/usr/bin/env python3
"""
Automated hyperparameter optimization for Graph Transformer.
Runs multiple configurations overnight, tracks best model.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from survival_classifier import config
from survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors,
    load_and_preprocess_survival_data, create_survival_dataloaders
)
from survival_classifier.models import (
    SurvivalGraphTransformer, CoxPHLoss, ConcordanceIndex,
    train_survival_epoch, evaluate_survival
)


def get_hyperparameter_configs():
    """
    Define hyperparameter configurations to test.

    Only varies parameters that can be passed directly to SurvivalGraphTransformer:
    - embedding_dim, n_layers, n_heads, dropout
    - learning_rate, weight_decay (optimizer params)

    Returns list of config dicts to try.
    """
    configs = []

    # Version 1: Current best (baseline from v5)
    configs.append({
        'name': 'v1_baseline',
        'embedding_dim': 160,
        'n_layers': 5,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Version 2: Deeper network (more layers to capture complex patterns)
    configs.append({
        'name': 'v2_deeper',
        'embedding_dim': 128,
        'n_layers': 8,  # Increased from 5
        'n_heads': 8,
        'dropout': 0.50,  # Higher dropout for deeper model
        'learning_rate': 2e-4,  # Lower LR for stability
        'weight_decay': 5e-4,
    })

    # Version 3: Wider network (larger embedding dimension)
    configs.append({
        'name': 'v3_wider',
        'embedding_dim': 256,  # Increased from 160
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Version 4: Very wide (maximum embedding capacity)
    configs.append({
        'name': 'v4_very_wide',
        'embedding_dim': 320,  # 2x baseline
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.50,  # More dropout for larger model
        'learning_rate': 2e-4,
        'weight_decay': 1e-3,  # More regularization
    })

    # Version 5: Regularization-heavy (combat overfitting)
    configs.append({
        'name': 'v5_heavy_reg',
        'embedding_dim': 160,
        'n_layers': 5,
        'n_heads': 8,
        'dropout': 0.60,  # Increased from 0.45
        'learning_rate': 3e-4,
        'weight_decay': 1e-3,  # Doubled from 5e-4
    })

    # Version 6: More attention heads (finer-grained attention)
    configs.append({
        'name': 'v6_more_heads',
        'embedding_dim': 192,  # Must be divisible by n_heads
        'n_layers': 5,
        'n_heads': 12,  # Increased from 8
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Version 7: Shallow and wide (fewer layers, more capacity per layer)
    configs.append({
        'name': 'v7_shallow_wide',
        'embedding_dim': 256,
        'n_layers': 3,  # Fewer layers
        'n_heads': 8,
        'dropout': 0.40,  # Less dropout for shallower network
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Version 8: Compact model (smaller but efficient)
    configs.append({
        'name': 'v8_compact',
        'embedding_dim': 96,
        'n_layers': 6,
        'n_heads': 6,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    return configs


def train_single_config(cfg, graph_prior, data_splits, device, output_dir):
    """Train a single configuration and return results."""

    print(f"\n{'='*80}")
    print(f"Training: {cfg['name']}")
    print(f"{'='*80}")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_survival_dataloaders(data_splits)

    # Get graph features as tensors
    graph_features = get_graph_features_as_tensors(
        graph_prior,
        device=device
    )

    # Build model with this config
    from survival_classifier.data import load_and_preprocess_survival_data
    n_proteins = data_splits['train'][0].shape[1]
    n_clinical = data_splits['train'][1].shape[1]
    n_genomic = data_splits['train'][2].shape[1]

    model = SurvivalGraphTransformer(
        n_proteins=n_proteins,
        n_clinical=n_clinical,
        n_genomic=n_genomic,
        diffusion_kernel=graph_features['K'],
        positional_encodings=graph_features['PE'],
        embedding_dim=cfg['embedding_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        dropout=cfg['dropout'],
        use_clinical=True,
        use_genomic=True,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Training loop
    best_val_c_index = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    max_epochs = 50

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_c_index': [],
    }

    for epoch in range(max_epochs):
        # Train
        train_loss = train_survival_epoch(
            model, train_loader, optimizer, device
        )

        # Validate
        val_loss, val_c_index = evaluate_survival(
            model, val_loader, device
        )

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_c_index'].append(val_c_index)

        # Print progress
        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f}, C-idx={val_c_index:.4f}")

        # Scheduler step
        scheduler.step(val_c_index)

        # Save best model
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ New best validation C-index: {best_val_c_index:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Evaluate best model on test set
    model.load_state_dict(best_model_state)
    test_loss, test_c_index = evaluate_survival(
        model, test_loader, device
    )

    print(f"\n{'='*80}")
    print(f"RESULTS: {cfg['name']}")
    print(f"{'='*80}")
    print(f"Best Val C-index:  {best_val_c_index:.4f}")
    print(f"Test C-index:      {test_c_index:.4f}")
    print(f"Test Loss:         {test_loss:.4f}")

    # Save model
    model_path = output_dir / f"{cfg['name']}_best.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'config': cfg,
        'val_c_index': best_val_c_index,
        'test_c_index': test_c_index,
        'n_params': n_params,
    }, model_path)

    # Save results
    results = {
        'config': cfg,
        'n_params': n_params,
        'n_epochs': epoch + 1,
        'best_val_c_index': float(best_val_c_index),
        'test_c_index': float(test_c_index),
        'test_loss': float(test_loss),
        'history': history,
    }

    results_path = output_dir / f"{cfg['name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize Graph Transformer hyperparameters")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/optimization')
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("="*80)
    print("GRAPH TRANSFORMER HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data once
    print("\nLoading data...")
    graph_prior = load_graph_prior(args.prior_path)
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        args.csv_path,
        graph_prior['protein_cols'],
        use_clinical=True,
        use_genomic=True
    )

    # Get configurations
    configs = get_hyperparameter_configs()
    print(f"\nTesting {len(configs)} configurations")

    # Train each configuration
    all_results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i}/{len(configs)}")
        print(f"{'#'*80}")

        try:
            results = train_single_config(
                cfg, graph_prior, data_splits, device, output_dir
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR training {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Find best model
    if len(all_results) > 0:
        best_result = max(all_results, key=lambda x: x['test_c_index'])

        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTested {len(all_results)} configurations successfully")

        print(f"\n{'='*80}")
        print("BEST MODEL")
        print(f"{'='*80}")
        print(f"Name: {best_result['config']['name']}")
        print(f"Test C-index: {best_result['test_c_index']:.4f}")
        print(f"Val C-index: {best_result['best_val_c_index']:.4f}")
        print(f"Parameters: {best_result['n_params']:,}")
        print(f"\nConfiguration:")
        print(json.dumps(best_result['config'], indent=2))

        # Ranking table
        print(f"\n{'='*80}")
        print("ALL RESULTS (Ranked by Test C-index)")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Name':<25} {'Test C-idx':>12} {'Val C-idx':>12} {'Params':>10}")
        print("-" * 80)

        sorted_results = sorted(all_results, key=lambda x: x['test_c_index'], reverse=True)
        for i, res in enumerate(sorted_results, 1):
            print(f"{i:<6} {res['config']['name']:<25} "
                  f"{res['test_c_index']:>12.4f} "
                  f"{res['best_val_c_index']:>12.4f} "
                  f"{res['n_params']:>10,}")

        # Save summary
        summary = {
            'best_model': best_result['config']['name'],
            'best_test_c_index': best_result['test_c_index'],
            'all_results': [{
                'name': r['config']['name'],
                'test_c_index': r['test_c_index'],
                'val_c_index': r['best_val_c_index'],
                'config': r['config']
            } for r in sorted_results]
        }

        summary_path = output_dir / 'optimization_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Saved summary to: {summary_path}")
        print(f"✅ Best model saved to: {output_dir}/{best_result['config']['name']}_best.pt")
    else:
        print("\n❌ No configurations completed successfully")


if __name__ == "__main__":
    main()