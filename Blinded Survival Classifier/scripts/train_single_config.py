#!/usr/bin/env python3
"""
Train a single configuration for manual hyperparameter tuning.
Easier to test one config at a time.
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


def main():
    parser = argparse.ArgumentParser(description="Train single config")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='mps')

    # Hyperparameters (override config.py)
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--name', type=str, default='custom_run',
                       help='Name for this run')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use config.py values as defaults, override with args
    cfg = {
        'name': args.name,
        'embedding_dim': args.embedding_dim or config.MODEL['embedding_dim'],
        'n_layers': args.n_layers or config.MODEL['n_layers'],
        'n_heads': args.n_heads or config.MODEL['n_heads'],
        'dropout': args.dropout or config.MODEL['dropout'],
        'learning_rate': args.learning_rate or config.TRAINING['learning_rate'],
        'weight_decay': args.weight_decay or config.TRAINING['weight_decay'],
    }

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("="*80)
    print(f"TRAINING: {cfg['name']}")
    print("="*80)
    print(f"Configuration:")
    for k, v in cfg.items():
        if k != 'name':
            print(f"  {k}: {v}")
    print(f"\nDevice: {device}")
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

    # Create dataloaders
    train_loader, val_loader, test_loader = create_survival_dataloaders(data_splits)

    # Get graph features
    graph_features = get_graph_features_as_tensors(graph_prior, device=device)

    # Build model
    n_proteins = data_splits['train'][0].shape[1]
    n_clinical = data_splits['train'][1].shape[1]
    n_genomic = data_splits['train'][2].shape[1]

    print("\nBuilding model...")
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
    print(f"Model parameters: {n_params:,}")

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
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

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
            print(f"  ✅ New best: {best_val_c_index:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Evaluate on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    model.load_state_dict(best_model_state)
    test_loss, test_c_index = evaluate_survival(model, test_loader, device)

    print(f"Best Val C-index:  {best_val_c_index:.4f}")
    print(f"Test C-index:      {test_c_index:.4f}")
    print(f"Test Loss:         {test_loss:.4f}")

    # Compare to v3_wider
    v3_test = 0.7207
    improvement = test_c_index - v3_test
    print(f"\nComparison to v3_wider (0.7207): {improvement:+.4f}")

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
        'improvement_over_v3': float(improvement),
        'history': history,
    }

    results_path = output_dir / f"{cfg['name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Results saved: {results_path}")
    print(f"✅ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
