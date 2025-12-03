"""
Train all baseline models for comparison with Graph Transformer.
Answers: Does PPI network topology improve survival prediction?
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim

# Add parent directory to path so we can import survival_classifier
sys.path.insert(0, str(Path(__file__).parent.parent))

from survival_classifier import config
from survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors,
    load_and_preprocess_survival_data, create_survival_dataloaders
)
from survival_classifier.models import (
    MLPSurvivalModel, VanillaTransformerSurvival, ProteinOnlyMLP,
    CoxPHLoss, ConcordanceIndex, train_survival_epoch, evaluate_survival
)

def train_baseline_model(model_name, model, train_loader, val_loader, test_loader,
                         device, output_dir):
    """Train a single baseline model."""
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Setup
    criterion = CoxPHLoss()
    c_index_metric = ConcordanceIndex()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING['learning_rate'],
        weight_decay=config.TRAINING['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.TRAINING.get('scheduler_factor', 0.5),
        patience=config.TRAINING.get('scheduler_patience', 5),
        min_lr=config.TRAINING.get('scheduler_min_lr', 1e-6)
    )

    # Training loop
    best_val_c_index = 0.0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_c_index': [],
        'val_loss': [],
        'val_c_index': [],
        'learning_rates': [],
    }

    for epoch in range(config.TRAINING['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config.TRAINING['max_epochs']}")
        print("-" * 80)

        # Train
        train_loss = train_survival_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_c_index = evaluate_survival(model, val_loader, device)
        _, train_c_index = evaluate_survival(model, train_loader, device)

        # Update scheduler
        scheduler.step(val_c_index)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history['train_loss'].append(train_loss)
        history['train_c_index'].append(train_c_index)
        history['val_loss'].append(val_loss)
        history['val_c_index'].append(val_c_index)
        history['learning_rates'].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train C-index: {train_c_index:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val C-index:   {val_c_index:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_c_index': val_c_index,
            }

            checkpoint_path = output_dir / f'{model_name}_best.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (val_c_index: {best_val_c_index:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.TRAINING['patience']:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    # Load best model and evaluate on test set
    checkpoint = torch.load(output_dir / f'{model_name}_best.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_c_index = evaluate_survival(model, test_loader, device)

    print(f"\n{model_name} Final Results:")
    print(f"  Best Val C-index: {best_val_c_index:.4f}")
    print(f"  Test C-index:     {test_c_index:.4f}")

    # Save results
    results = {
        'model_name': model_name,
        'n_params': n_params,
        'test_c_index': float(test_c_index),
        'val_c_index': float(best_val_c_index),
        'train_c_index': float(history['train_c_index'][-1]),
        'n_epochs': len(history['train_loss']),
        'history': history,
    }

    results_path = output_dir / f'{model_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train baseline survival models")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/baselines')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    # Set random seeds (same as graph model)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Baseline Model Training")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Load data (same splits as graph model)
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)

    graph_prior = load_graph_prior(args.prior_path)
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        args.csv_path,
        graph_prior['protein_cols'],
        use_clinical=True,
        use_genomic=True
    )

    train_loader, val_loader, test_loader = create_survival_dataloaders(
        data_splits,
        num_workers=args.num_workers
    )

    n_proteins = preprocessing_info['feature_dims']['protein']
    n_clinical = preprocessing_info['feature_dims']['clinical']
    n_genomic = preprocessing_info['feature_dims']['genomic']

    # Train all baseline models
    all_results = {}

    # 1. MLP Baseline (concat all features, no topology)
    print("\n" + "="*80)
    print("BASELINE 1: MLP (no topology)")
    print("="*80)
    mlp_model = MLPSurvivalModel(
        n_proteins=n_proteins,
        n_clinical=n_clinical,
        n_genomic=n_genomic,
        hidden_dims=[512, 256, 128],
        dropout=config.MODEL['dropout']
    ).to(args.device)

    mlp_results = train_baseline_model(
        'mlp_baseline', mlp_model, train_loader, val_loader, test_loader,
        args.device, output_dir
    )
    all_results['mlp_baseline'] = mlp_results

    # 2. Vanilla Transformer (no graph bias)
    print("\n" + "="*80)
    print("BASELINE 2: Vanilla Transformer (no graph bias)")
    print("="*80)
    vanilla_tf = VanillaTransformerSurvival(
        n_proteins=n_proteins,
        n_clinical=n_clinical,
        n_genomic=n_genomic,
    ).to(args.device)

    vanilla_results = train_baseline_model(
        'vanilla_transformer', vanilla_tf, train_loader, val_loader, test_loader,
        args.device, output_dir
    )
    all_results['vanilla_transformer'] = vanilla_results

    # 3. Protein-only MLP (no clinical, no genomic, no topology)
    print("\n" + "="*80)
    print("BASELINE 3: Protein-only MLP")
    print("="*80)
    protein_only = ProteinOnlyMLP(
        n_proteins=n_proteins,
        hidden_dims=[512, 256, 128],
        dropout=config.MODEL['dropout']
    ).to(args.device)

    protein_results = train_baseline_model(
        'protein_only', protein_only, train_loader, val_loader, test_loader,
        args.device, output_dir
    )
    all_results['protein_only'] = protein_results

    # Summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<25} {'Test C-index':<15} {'Val C-index':<15} {'Parameters':<15}")
    print("-"*80)

    for model_name, results in all_results.items():
        print(f"{model_name:<25} {results['test_c_index']:<15.4f} "
              f"{results['val_c_index']:<15.4f} {results['n_params']:<15,}")

    # Save summary
    summary_path = output_dir / 'baseline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ All results saved to {output_dir}")
    print("\nNext: Compare with Graph Transformer results")


if __name__ == '__main__':
    main()