"""
Training and evaluation script for Graph Transformer Cancer Type Classifier.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from graph_prior import load_graph_prior, get_graph_features_as_tensors
from Classifier.dataset_cancer_classifier import load_and_preprocess_data, create_dataloaders
from Classifier.graph_transformer_cancer_classifier import GraphTransformerClassifier


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
        description="Train Graph Transformer for cancer type classification"
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
        default='outputs',
        help='Output directory for checkpoints and results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loader workers'
    )
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Evaluating'):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    all_logits = np.vstack(all_logits)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels,
        'logits': all_logits,
    }


def plot_training_curves(history, output_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    args = parse_args()
    
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
    print("Graph Transformer Cancer Type Classifier")
    print("=" * 80)
    print(f"\nDevice: {args.device}")
    print(f"Output directory: {output_dir}")
    
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
    data_splits, label_info, scaler = load_and_preprocess_data(
        args.csv_path,
        graph_prior['protein_cols']
    )
    
    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_splits,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    model = GraphTransformerClassifier(
        n_proteins=graph_prior['A'].shape[0],
        n_classes=label_info['n_classes'],
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
    ).to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Number of proteins: {graph_prior['A'].shape[0]}")
    print(f"Number of classes: {label_info['n_classes']}")
    print(f"Classes: {label_info['class_names']}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING['learning_rate'],
        weight_decay=config.TRAINING['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1_macro': [],
    }
    
    for epoch in range(config.TRAINING['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config.TRAINING['max_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            grad_clip=config.TRAINING['grad_clip']
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f} | "
              f"F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'config': config.__dict__,
                'label_info': label_info,
            }
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.TRAINING['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
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
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, args.device)
    
    print("\nTest Set Results:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    
    # Classification report
    print("\nPer-Class Metrics:")
    report = classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        target_names=label_info['class_names'],
        digits=4
    )
    print(report)
    
    # Save classification report
    report_path = results_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nClassification report saved to {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    cm_path = plots_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, label_info['class_names'], cm_path)
    
    # Save test results
    test_results = {
        'accuracy': float(test_metrics['accuracy']),
        'f1_macro': float(test_metrics['f1_macro']),
        'f1_weighted': float(test_metrics['f1_weighted']),
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc),
        'n_epochs': checkpoint['epoch'] + 1,
        'n_params': n_params,
    }
    
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
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
