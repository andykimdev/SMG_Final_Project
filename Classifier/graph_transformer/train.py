"""Training script for Graph Transformer Cancer Type Classifier."""

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
from dataset import load_and_preprocess_data, create_dataloaders
from model import GraphTransformerClassifier


class Logger:
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
    parser = argparse.ArgumentParser(description="Train Graph Transformer classifier")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../../results/classifiers/cancer_type_classifiers/transformer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f'logs_{timestamp}.txt'
    logger = Logger(log_path)
    sys.stdout = logger
    
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    
    # Load graph prior
    print("\nLoading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    graph_tensors = get_graph_features_as_tensors(graph_prior, device=args.device)
    
    # Load data
    print("Loading data...")
    data_splits, label_info, scaler = load_and_preprocess_data(
        args.csv_path, graph_prior['protein_cols']
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_splits, num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = GraphTransformerClassifier(
        n_proteins=graph_prior['A'].shape[0],
        n_classes=label_info['n_classes'],
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
    ).to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Classes: {label_info['n_classes']}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING['learning_rate'],
        weight_decay=config.TRAINING['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1_macro': [],
    }
    
    for epoch in range(config.TRAINING['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config.TRAINING['max_epochs']}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            grad_clip=config.TRAINING['grad_clip']
        )
        
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val_metrics['loss'])
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'config': {'MODEL': config.MODEL, 'TRAINING': config.TRAINING},
                'label_info': label_info,
            }
            
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config.TRAINING['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Save history
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_curves(history, plots_dir / 'training_curves.png')
    
    # Test
    print("\nTesting...")
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, args.device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 macro: {test_metrics['f1_macro']:.4f}")
    
    report = classification_report(
        test_metrics['labels'], test_metrics['predictions'],
        target_names=label_info['class_names'], digits=4
    )
    print(report)
    
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    plot_confusion_matrix(cm, label_info['class_names'], plots_dir / 'confusion_matrix.png')
    
    test_results = {
        'accuracy': float(test_metrics['accuracy']),
        'f1_macro': float(test_metrics['f1_macro']),
        'f1_weighted': float(test_metrics['f1_weighted']),
        'best_val_loss': float(best_val_loss),
    }
    
    with open(results_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nDone. Results saved to {output_dir}")
    
    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()
