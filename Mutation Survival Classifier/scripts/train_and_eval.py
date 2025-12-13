"""
Training and evaluation script for Mutation-Augmented Graph Transformer Survival Predictor.
Predicts Disease-Specific Survival (DSS) using RPPA + Mutations with Cox Proportional Hazards model.
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
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation_survival_classifier import config
from mutation_survival_classifier.data import (
    load_graph_prior, get_graph_features_as_tensors,
    load_and_preprocess_survival_data, create_survival_dataloaders
)
from mutation_survival_classifier.models import (
    SurvivalGraphTransformer, CoxPHLoss, ConcordanceIndex,
    train_survival_epoch, evaluate_survival
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Mutation-Augmented Graph Transformer for survival prediction"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to TCGA RPPA + Mutations CSV file'
    )
    parser.add_argument(
        '--prior_path',
        type=str,
        required=True,
        help='Path to STRING prior .npz file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/mutation_survival',
        help='Output directory for checkpoints and results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda, cpu, or mps)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=None,
        help='Maximum epochs (overrides config)'
    )
    parser.add_argument(
        '--use_mutations',
        action='store_true',
        default=True,
        help='Include mutation features (default: True)'
    )
    parser.add_argument(
        '--use_cancer_type',
        action='store_true',
        default=True,
        help='Include cancer type as feature (default: True)'
    )
    parser.add_argument(
        '--no_mutations',
        action='store_true',
        help='Disable mutation features'
    )
    parser.add_argument(
        '--no_cancer_type',
        action='store_true',
        help='Disable cancer type feature (blinded model)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Handle argument overrides
    use_mutations = args.use_mutations and not args.no_mutations
    use_cancer_type = args.use_cancer_type and not args.no_cancer_type

    print(f"\nConfiguration:")
    print(f"  Use mutations: {use_mutations}")
    print(f"  Use cancer type: {use_cancer_type}")
    print(f"  Device: {args.device}")

    # Save configuration
    config_dict = {
        'model': config.MODEL,
        'training': config.TRAINING,
        'graph_prior': config.GRAPH_PRIOR,
        'data': config.DATA,
        'use_mutations': use_mutations,
        'use_cancer_type': use_cancer_type,
        'timestamp': timestamp,
    }
    if args.max_epochs:
        config_dict['training']['max_epochs'] = args.max_epochs

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # ========================================================================
    # Load Graph Prior
    # ========================================================================
    print("\n" + "="*80)
    print("Loading Graph Prior")
    print("="*80)

    graph_prior = load_graph_prior(args.prior_path)
    protein_cols = graph_prior['protein_cols']
    print(f"Graph prior: {len(protein_cols)} proteins")

    # ========================================================================
    # Load and Preprocess Data
    # ========================================================================
    print("\n" + "="*80)
    print("Loading and Preprocessing Data")
    print("="*80)

    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        csv_path=args.csv_path,
        protein_cols=protein_cols,
        use_clinical=True,
        use_genomic=True,
        use_mutations=use_mutations,
        use_cancer_type=use_cancer_type,
        random_seed=config.RANDOM_SEED
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_survival_dataloaders(
        data_splits,
        batch_size=config.TRAINING['batch_size'],
        num_workers=0
    )

    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # ========================================================================
    # Initialize Model
    # ========================================================================
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)

    device = torch.device(args.device)
    graph_features = get_graph_features_as_tensors(graph_prior, device=device)

    model = SurvivalGraphTransformer(
        n_proteins=preprocessing_info['feature_dims']['protein'],
        n_clinical=preprocessing_info['feature_dims']['clinical'],
        n_genomic=preprocessing_info['feature_dims']['genomic'],
        diffusion_kernel=graph_features['K'],
        positional_encodings=graph_features['PE'],
        embedding_dim=config.MODEL['embedding_dim'],
        n_layers=config.MODEL['n_layers'],
        n_heads=config.MODEL['n_heads'],
        dropout=config.MODEL['dropout'],
        use_clinical=preprocessing_info['feature_dims']['clinical'] > 0,
        use_genomic=preprocessing_info['feature_dims']['genomic'] > 0,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {n_params:,}")
    print(f"  Trainable: {n_trainable:,}")

    # ========================================================================
    # Training Setup
    # ========================================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING['learning_rate'],
        weight_decay=config.TRAINING['weight_decay']
    )

    if config.TRAINING['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize C-index
            factor=config.TRAINING['scheduler_factor'],
            patience=config.TRAINING['scheduler_patience'],
            min_lr=config.TRAINING['scheduler_min_lr']
        )
    else:
        scheduler = None

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    max_epochs = args.max_epochs if args.max_epochs else config.TRAINING['max_epochs']
    best_val_c_index = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_c_index': [],
        'val_c_index': []
    }

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 40)

        # Train
        train_loss = train_survival_epoch(model, train_loader, optimizer, device)
        train_loss_eval, train_c_index = evaluate_survival(model, train_loader, device)

        # Validate
        val_loss, val_c_index = evaluate_survival(model, val_loader, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_c_index'].append(train_c_index)
        history['val_c_index'].append(val_c_index)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train C-index: {train_c_index:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val C-index: {val_c_index:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_c_index)

        # Early stopping
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_c_index': val_c_index,
                'config': config.MODEL,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"✓ New best model saved (C-index: {val_c_index:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.TRAINING['patience']}")

            if patience_counter >= config.TRAINING['patience']:
                print("\n" + "="*80)
                print("Early stopping triggered")
                print("="*80)
                break

    # ========================================================================
    # Final Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    test_loss, test_c_index = evaluate_survival(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  C-index: {test_c_index:.4f}")

    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_c_index': float(test_c_index),
        'best_val_c_index': float(best_val_c_index),
        'final_epoch': len(history['train_loss']),
        'survival_info': survival_info,
        'preprocessing_info': {
            'protein_features': preprocessing_info['feature_dims']['protein'],
            'clinical_features': preprocessing_info['feature_dims']['clinical'],
            'genomic_features': preprocessing_info['feature_dims']['genomic'],
            'use_cancer_type': preprocessing_info['use_cancer_type'],
            'use_mutations': preprocessing_info['use_mutations'],
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save training history
    np.savez(
        output_dir / 'history.npz',
        train_loss=history['train_loss'],
        val_loss=history['val_loss'],
        train_c_index=history['train_c_index'],
        val_c_index=history['val_c_index']
    )

    print(f"\n✓ Results saved to {output_dir}")
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
