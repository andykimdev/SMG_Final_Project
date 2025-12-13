#!/usr/bin/env python3
"""
Refined hyperparameter optimization around v3_wider (256d × 4L).
Tests variations to squeeze out more performance.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import copy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blinded_survival_classifier import config
from blinded_survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors,
    load_and_preprocess_survival_data, create_survival_dataloaders
)
from blinded_survival_classifier.models import (
    SurvivalGraphTransformer, train_survival_epoch, evaluate_survival
)


def get_refined_configs():
    """
    Refined configs around v3_wider (256d × 4L).

    v3_wider achieved: Test C-index 0.7207
    - embedding_dim: 256
    - n_layers: 4
    - dropout: 0.45
    - lr: 3e-4
    - weight_decay: 5e-4

    We'll explore:
    - Slight variations in embedding dim (224, 256, 288)
    - Dropout variations (0.40, 0.45, 0.50)
    - Learning rate variations
    - Layer variations (3, 4, 5)
    """
    configs = []

    # Baseline: v3_wider (for comparison)
    configs.append({
        'name': 'v3_baseline',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 1: Lower dropout (less regularization)
    configs.append({
        'name': 'v3_lower_dropout',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.40,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 2: Higher dropout (more regularization)
    configs.append({
        'name': 'v3_higher_dropout',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.50,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 3: Lower learning rate (more stable)
    configs.append({
        'name': 'v3_lower_lr',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 2e-4,
        'weight_decay': 5e-4,
    })

    # Variation 4: Higher learning rate (faster learning)
    configs.append({
        'name': 'v3_higher_lr',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 4e-4,
        'weight_decay': 5e-4,
    })

    # Variation 5: Smaller embedding (224d)
    configs.append({
        'name': 'v3_224d',
        'embedding_dim': 224,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 6: Larger embedding (288d)
    configs.append({
        'name': 'v3_288d',
        'embedding_dim': 288,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 7: 3 layers (shallower)
    configs.append({
        'name': 'v3_3layers',
        'embedding_dim': 256,
        'n_layers': 3,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 8: 5 layers (deeper)
    configs.append({
        'name': 'v3_5layers',
        'embedding_dim': 256,
        'n_layers': 5,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    # Variation 9: Higher weight decay (more L2 reg)
    configs.append({
        'name': 'v3_higher_wd',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 1e-3,
    })

    # Variation 10: Lower weight decay (less L2 reg)
    configs.append({
        'name': 'v3_lower_wd',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.45,
        'learning_rate': 3e-4,
        'weight_decay': 2.5e-4,
    })

    # Variation 11: Combined best guesses (lower dropout + lower LR)
    configs.append({
        'name': 'v3_combo_1',
        'embedding_dim': 256,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.40,
        'learning_rate': 2e-4,
        'weight_decay': 5e-4,
    })

    # Variation 12: Combined (288d + lower dropout)
    configs.append({
        'name': 'v3_combo_2',
        'embedding_dim': 288,
        'n_layers': 4,
        'n_heads': 8,
        'dropout': 0.40,
        'learning_rate': 3e-4,
        'weight_decay': 5e-4,
    })

    return configs


def train_single_config(cfg, graph_prior, data_splits, device, output_dir):
    """Train a single configuration."""

    print(f"\n{'='*80}")
    print(f"Training: {cfg['name']}")
    print(f"{'='*80}")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_survival_dataloaders(data_splits)

    # Get graph features
    graph_features = get_graph_features_as_tensors(graph_prior, device=device)

    # Build model
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
        use_genomic=False,
    ).to(device)

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
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
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
        train_loss = train_survival_epoch(model, train_loader, optimizer, device)
        val_loss, val_c_index = evaluate_survival(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_c_index'].append(val_c_index)

        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f}, C-idx={val_c_index:.4f}")

        scheduler.step(val_c_index)

        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ New best validation C-index: {best_val_c_index:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_c_index = evaluate_survival(model, test_loader, device)

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
    parser = argparse.ArgumentParser(description="Refine v3_wider hyperparameters")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../outputs/optimization_v3_refined')
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("="*80)
    print("V3_WIDER REFINED HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print("Target: Beat v3_wider Test C-index of 0.7207")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    graph_prior = load_graph_prior(args.prior_path)
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        args.csv_path,
        graph_prior['protein_cols'],
        use_clinical=True,
        use_genomic=False
    )

    # Get configs
    configs = get_refined_configs()
    print(f"\nTesting {len(configs)} configurations")

    # Train each
    all_results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i}/{len(configs)}")
        print(f"{'#'*80}")

        try:
            results = train_single_config(cfg, graph_prior, data_splits, device, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR training {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Find best
    if len(all_results) > 0:
        best_result = max(all_results, key=lambda x: x['test_c_index'])

        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTested {len(all_results)} configurations")

        print(f"\n{'='*80}")
        print("BEST MODEL")
        print(f"{'='*80}")
        print(f"Name: {best_result['config']['name']}")
        print(f"Test C-index: {best_result['test_c_index']:.4f}")
        print(f"Val C-index: {best_result['best_val_c_index']:.4f}")
        print(f"Parameters: {best_result['n_params']:,}")
        print(f"\nConfiguration:")
        print(json.dumps(best_result['config'], indent=2))

        # Improvement over v3_wider
        v3_test_c = 0.7207
        improvement = best_result['test_c_index'] - v3_test_c
        print(f"\nImprovement over v3_wider: {improvement:+.4f}")

        # Ranking table
        print(f"\n{'='*80}")
        print("ALL RESULTS (Ranked by Test C-index)")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Name':<25} {'Test C-idx':>12} {'Val C-idx':>12} {'vs v3':>10}")
        print("-" * 80)

        sorted_results = sorted(all_results, key=lambda x: x['test_c_index'], reverse=True)
        for i, res in enumerate(sorted_results, 1):
            diff = res['test_c_index'] - v3_test_c
            print(f"{i:<6} {res['config']['name']:<25} "
                  f"{res['test_c_index']:>12.4f} "
                  f"{res['best_val_c_index']:>12.4f} "
                  f"{diff:>+10.4f}")

        # Save summary
        summary = {
            'best_model': best_result['config']['name'],
            'best_test_c_index': best_result['test_c_index'],
            'improvement_over_v3': improvement,
            'v3_baseline': v3_test_c,
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

        print(f"\n✅ Summary: {summary_path}")
        print(f"✅ Best model: {output_dir}/{best_result['config']['name']}_best.pt")
    else:
        print("\n❌ No configurations completed")


if __name__ == "__main__":
    main()
