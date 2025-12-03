"""
Training and evaluation script for Graph Transformer Survival Predictor.
Predicts Disease-Specific Survival (DSS) using Cox Proportional Hazards model.
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
import matplotlib.pyplot as plt
import seaborn as sns

import config
from graph_prior import load_graph_prior, get_graph_features_as_tensors
from dataset_survival_classifier import load_and_preprocess_survival_data, create_survival_dataloaders
from graph_transformer_survival_classifier import (
    SurvivalGraphTransformer,
    CoxPHLoss,
    ConcordanceIndex,
    train_survival_epoch,
    evaluate_survival
)


class Logger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_path):
        self.log_file = open(log_path, 'w')
        self.terminal = sys.stdout
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Graph Transformer for survival prediction"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        required=True,
        help='Path to TCGA RPPA CSV file'
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
        default='outputs/survival',
        help='Output directory for checkpoints and results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda, cpu, or mps)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--use_clinical',
        action='store_true',
        default=True,
        help='Use clinical features'
    )
    parser.add_argument(
        '--use_genomic',
        action='store_true',
        default=True,
        help='Use genomic features'
    )
    parser.add_argument(
        '--no_clinical',
        action='store_true',
        help='Disable clinical features'
    )
    parser.add_argument(
        '--no_genomic',
        action='store_true',
        help='Disable genomic features'
    )
    return parser.parse_args()


def plot_training_curves(history, output_path):
    """Plot and save training curves for survival model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Cox Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # C-index curve
    axes[1].plot(history['train_c_index'], label='Train C-index', linewidth=2)
    axes[1].plot(history['val_c_index'], label='Val C-index', linewidth=2)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('C-index', fontsize=12)
    axes[1].set_title('Training and Validation C-index', fontsize=14)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_kaplan_meier(risk_scores, times, events, output_path, n_groups=3):
    """Plot Kaplan-Meier curves stratified by risk groups."""
    from lifelines import KaplanMeierFitter
    
    # Stratify into risk groups
    risk_percentiles = np.percentile(risk_scores, np.linspace(0, 100, n_groups + 1))
    risk_groups = np.digitize(risk_scores, risk_percentiles[1:-1])
    
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    group_names = ['Low Risk', 'Medium Risk', 'High Risk'] if n_groups == 3 else [f'Group {i+1}' for i in range(n_groups)]
    colors = ['green', 'orange', 'red'] if n_groups == 3 else None
    
    for i in range(n_groups):
        mask = risk_groups == i
        if mask.sum() > 0:
            kmf.fit(
                times[mask],
                events[mask],
                label=f'{group_names[i]} (n={mask.sum()})'
            )
            kmf.plot_survival_function(
                ci_show=True,
                color=colors[i] if colors else None,
                linewidth=2
            )
    
    plt.xlabel('Time (months)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.title('Kaplan-Meier Curves by Risk Group', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Kaplan-Meier curves saved to {output_path}")


def plot_risk_distribution(risk_scores, events, output_path):
    """Plot distribution of risk scores by event status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(
        risk_scores[events == 0],
        bins=30,
        alpha=0.6,
        label='Censored',
        color='blue',
        edgecolor='black'
    )
    axes[0].hist(
        risk_scores[events == 1],
        bins=30,
        alpha=0.6,
        label='Events (Deaths)',
        color='red',
        edgecolor='black'
    )
    axes[0].set_xlabel('Risk Score', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Risk Score Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(
        [risk_scores[events == 0], risk_scores[events == 1]],
        labels=['Censored', 'Events (Deaths)'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.6),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[1].set_ylabel('Risk Score', fontsize=12)
    axes[1].set_title('Risk Scores by Event Status', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Risk distribution plot saved to {output_path}")


def evaluate_survival_full(model, loader, criterion, c_index_metric, device):
    """Full evaluation with predictions for plotting."""
    model.eval()
    
    all_risks = []
    all_times = []
    all_events = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            risk_scores = model(batch)
            loss = criterion(risk_scores, batch['time'], batch['event'])
            
            all_risks.append(risk_scores.cpu())
            all_times.append(batch['time'].cpu())
            all_events.append(batch['event'].cpu())
            total_loss += loss.item()
    
    all_risks = torch.cat(all_risks).numpy().squeeze()
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()
    
    avg_loss = total_loss / len(loader)
    c_index = c_index_metric(
        torch.from_numpy(all_risks),
        torch.from_numpy(all_times),
        torch.from_numpy(all_events)
    ).item()
    
    return {
        'loss': avg_loss,
        'c_index': c_index,
        'risk_scores': all_risks,
        'times': all_times,
        'events': all_events,
    }


def main():
    args = parse_args()
    
    # Handle clinical/genomic flags
    use_clinical = args.use_clinical and not args.no_clinical
    use_genomic = args.use_genomic and not args.no_genomic
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f'logs_{timestamp}.txt'
    logger = Logger(log_path)
    sys.stdout = logger
    
    print(f"Logging to: {log_path}")
    
    print("=" * 80)
    print("Graph Transformer Survival Predictor")
    print("=" * 80)
    print(f"\nDevice: {args.device}")
    print(f"Output directory: {output_dir}")
    print(f"Use clinical features: {use_clinical}")
    print(f"Use genomic features: {use_genomic}")
    
    # Load graph prior
    print("\n" + "=" * 80)
    print("Loading Graph Prior")
    print("=" * 80)
    graph_prior = load_graph_prior(args.prior_path)
    graph_tensors = get_graph_features_as_tensors(graph_prior, device=args.device)
    
    # Load and preprocess data
    print("\n" + "=" * 80)
    print("Loading and Preprocessing Data")
    print("=" * 80)
    data_splits, survival_info, preprocessing_info = load_and_preprocess_survival_data(
        args.csv_path,
        graph_prior['protein_cols'],
        use_clinical=use_clinical,
        use_genomic=use_genomic
    )
    
    # Check minimum events requirement
    if survival_info['total_events'] < config.DATA['min_events']:
        print(f"\n⚠ WARNING: Only {survival_info['total_events']} events "
              f"(recommend ≥{config.DATA['min_events']} for reliable Cox model)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_survival_dataloaders(
        data_splits,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    model = SurvivalGraphTransformer(
        n_proteins=preprocessing_info['feature_dims']['protein'],
        n_clinical=preprocessing_info['feature_dims']['clinical'],
        n_genomic=preprocessing_info['feature_dims']['genomic'],
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
        use_clinical=use_clinical,
        use_genomic=use_genomic,
    ).to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Feature dimensions:")
    print(f"  - Proteins: {preprocessing_info['feature_dims']['protein']}")
    print(f"  - Clinical: {preprocessing_info['feature_dims']['clinical']}")
    print(f"  - Genomic:  {preprocessing_info['feature_dims']['genomic']}")
    print(f"  - TOTAL:    {sum(preprocessing_info['feature_dims'].values())}")
    
    # Loss and optimizer
    criterion = CoxPHLoss()
    c_index_metric = ConcordanceIndex()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING['learning_rate'],
        weight_decay=config.TRAINING['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.TRAINING.get('use_scheduler', False):
        scheduler_type = config.TRAINING.get('scheduler_type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # Maximize C-index
                factor=config.TRAINING.get('scheduler_factor', 0.5),
                patience=config.TRAINING.get('scheduler_patience', 5),
                min_lr=config.TRAINING.get('scheduler_min_lr', 1e-6)
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.TRAINING['max_epochs'],
                eta_min=config.TRAINING.get('scheduler_min_lr', 1e-6)
            )
        print(f"\nUsing {scheduler_type} learning rate scheduler")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    best_val_loss = float('inf')
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
        train_loss = train_survival_epoch(model, train_loader, optimizer, args.device)
        
        # Validate
        val_loss, val_c_index = evaluate_survival(model, val_loader, args.device)
        
        # Also get training C-index (more expensive, so done separately)
        _, train_c_index = evaluate_survival(model, train_loader, args.device)
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_c_index)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_c_index'].append(train_c_index)
        history['val_loss'].append(val_loss)
        history['val_c_index'].append(val_c_index)
        history['learning_rates'].append(current_lr)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train C-index: {train_c_index:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val C-index:   {val_c_index:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model (based on C-index)
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_c_index': val_c_index,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__')},
                'survival_info': survival_info,
                'preprocessing_info': {
                    'feature_dims': preprocessing_info['feature_dims'],
                    'column_types': preprocessing_info['column_types'],
                }
            }
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (val_c_index: {best_val_c_index:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.TRAINING['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation C-index: {best_val_c_index:.4f}")
            break
    
    # Save training history
    history_path = results_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot training curves
    curves_path = plots_dir / 'training_curves.png'
    plot_training_curves(history, curves_path)
    
    # Load best model for testing
    print("\n" + "=" * 80)
    print("Testing")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation C-index: {checkpoint['val_c_index']:.4f}")
    
    # Evaluate on test set with full predictions
    test_metrics = evaluate_survival_full(
        model, test_loader, criterion, c_index_metric, args.device
    )
    
    print("\nTest Set Results:")
    print(f"  Cox Loss:  {test_metrics['loss']:.4f}")
    print(f"  C-index:   {test_metrics['c_index']:.4f}")
    print(f"\nC-index interpretation:")
    print(f"  0.50-0.60: Random to weak")
    print(f"  0.60-0.70: Acceptable")
    print(f"  0.70-0.80: Good")
    print(f"  0.80-0.90: Excellent")
    print(f"  0.90-1.00: Outstanding")
    
    # Additional test set statistics
    n_events = int(test_metrics['events'].sum())
    n_censored = len(test_metrics['events']) - n_events
    median_time = np.median(test_metrics['times'])
    
    print(f"\nTest Set Statistics:")
    print(f"  Total samples: {len(test_metrics['times'])}")
    print(f"  Events (deaths): {n_events} ({n_events/len(test_metrics['times'])*100:.1f}%)")
    print(f"  Censored: {n_censored} ({n_censored/len(test_metrics['times'])*100:.1f}%)")
    print(f"  Median time: {median_time:.1f} months")
    print(f"  Time range: [{test_metrics['times'].min():.1f}, {test_metrics['times'].max():.1f}] months")
    
    # Risk score statistics
    print(f"\nRisk Score Statistics:")
    print(f"  Mean: {test_metrics['risk_scores'].mean():.4f}")
    print(f"  Std:  {test_metrics['risk_scores'].std():.4f}")
    print(f"  Min:  {test_metrics['risk_scores'].min():.4f}")
    print(f"  Max:  {test_metrics['risk_scores'].max():.4f}")
    
    # Plot Kaplan-Meier curves
    try:
        km_path = plots_dir / 'kaplan_meier.png'
        plot_kaplan_meier(
            test_metrics['risk_scores'],
            test_metrics['times'],
            test_metrics['events'],
            km_path,
            n_groups=config.METRICS.get('risk_groups', 3)
        )
    except ImportError:
        print("\n⚠ lifelines not installed, skipping Kaplan-Meier plot")
        print("  Install with: pip install lifelines")
    
    # Plot risk distribution
    risk_dist_path = plots_dir / 'risk_distribution.png'
    plot_risk_distribution(
        test_metrics['risk_scores'],
        test_metrics['events'],
        risk_dist_path
    )
    
    # Save test results
    test_results = {
        'test_loss': float(test_metrics['loss']),
        'test_c_index': float(test_metrics['c_index']),
        'best_val_loss': float(best_val_loss),
        'best_val_c_index': float(best_val_c_index),
        'n_epochs': checkpoint['epoch'] + 1,
        'n_params': n_params,
        'n_events': int(n_events),
        'n_censored': int(n_censored),
        'event_rate': float(n_events / len(test_metrics['times'])),
        'median_survival_time': float(median_time),
        'features': {
            'proteins': preprocessing_info['feature_dims']['protein'],
            'clinical': preprocessing_info['feature_dims']['clinical'],
            'genomic': preprocessing_info['feature_dims']['genomic'],
            'total': sum(preprocessing_info['feature_dims'].values()),
        }
    }
    
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to {results_path}")
    
    # Save predictions
    predictions = {
        'risk_scores': test_metrics['risk_scores'].tolist(),
        'times': test_metrics['times'].tolist(),
        'events': test_metrics['events'].tolist(),
    }
    predictions_path = results_dir / 'test_predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Test predictions saved to {predictions_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal Performance:")
    print(f"  Best Val C-index: {best_val_c_index:.4f}")
    print(f"  Test C-index:     {test_metrics['c_index']:.4f}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Logs: {log_path}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Results: {results_dir}")
    print(f"  - Plots: {plots_dir}")
    
    # Close logger
    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()