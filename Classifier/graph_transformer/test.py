"""
Test script for Graph Transformer Classifier.
Loads trained model weights and evaluates on test data.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from graph_prior import load_graph_prior, get_graph_features_as_tensors
from dataset import load_and_preprocess_data, create_dataloaders
from model import GraphTransformerClassifier


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(model, loader, criterion, device):
    """Evaluate model on data loader."""
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


def main():
    parser = argparse.ArgumentParser(description="Test Graph Transformer classifier")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint (best_model.pt)')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to TCGA RPPA CSV file')
    parser.add_argument('--prior_path', type=str, required=True,
                       help='Path to graph prior .npz file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as checkpoint parent)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    args = parser.parse_args()
    
    # Determine output directory
    checkpoint_path = Path(args.checkpoint_path)
    if args.output_dir is None:
        output_dir = checkpoint_path.parent.parent  # Go up from checkpoints/ to results/
    else:
        output_dir = Path(args.output_dir)
    
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Evaluating on: {args.split} split")
    print(f"Output directory: {output_dir}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    
    # Extract saved info
    saved_config = checkpoint.get('config', {})
    label_info = checkpoint.get('label_info', {})
    model_config = saved_config.get('MODEL', config.MODEL)
    
    print(f"\nModel config from checkpoint:")
    print(f"  Embedding dim: {model_config.get('embedding_dim', 'N/A')}")
    print(f"  Layers: {model_config.get('n_layers', 'N/A')}")
    print(f"  Heads: {model_config.get('n_heads', 'N/A')}")
    print(f"  Classes: {label_info.get('n_classes', 'N/A')}")
    
    # Load graph prior
    print("\nLoading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    graph_tensors = get_graph_features_as_tensors(graph_prior, device=args.device)
    
    # Load and preprocess data
    print("Loading data...")
    data_splits, label_info_loaded, scaler = load_and_preprocess_data(
        args.csv_path, graph_prior['protein_cols']
    )
    
    # Use label_info from checkpoint if available, otherwise use loaded
    if label_info and 'class_names' in label_info:
        label_info = label_info
    else:
        label_info = label_info_loaded
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(data_splits, num_workers=0)
    
    # Select the appropriate loader
    if args.split == 'train':
        loader = train_loader
    elif args.split == 'val':
        loader = val_loader
    else:
        loader = test_loader
    
    # Initialize model
    print("Initializing model...")
    model = GraphTransformerClassifier(
        n_proteins=graph_prior['A'].shape[0],
        n_classes=label_info['n_classes'],
        diffusion_kernel=graph_tensors['K'],
        positional_encodings=graph_tensors['PE'],
        embedding_dim=model_config.get('embedding_dim'),
        n_layers=model_config.get('n_layers'),
        n_heads=model_config.get('n_heads'),
        ffn_dim=model_config.get('ffn_dim'),
        dropout=model_config.get('dropout'),
    ).to(args.device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully")
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, args.device)
    
    # Print results
    print(f"\n{args.split.upper()} Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 macro: {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted: {metrics['f1_weighted']:.4f}")
    
    # Classification report
    report = classification_report(
        metrics['labels'], metrics['predictions'],
        target_names=label_info['class_names'], digits=4
    )
    print(f"\nClassification Report:\n{report}")
    
    # Save results
    report_path = results_dir / f'{args.split}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"{args.split.upper()} Set Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    print(f"\nSaved classification report: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    cm_path = plots_dir / f'confusion_matrix_{args.split}.png'
    plot_confusion_matrix(cm, label_info['class_names'], cm_path)
    print(f"Saved confusion matrix: {cm_path}")
    
    # Save metrics JSON
    results_dict = {
        'split': args.split,
        'loss': float(metrics['loss']),
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'f1_weighted': float(metrics['f1_weighted']),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch', 'N/A'),
        'checkpoint_val_loss': float(checkpoint.get('val_loss', 0.0)),
        'checkpoint_val_acc': float(checkpoint.get('val_acc', 0.0)),
    }
    
    results_json_path = results_dir / f'{args.split}_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved results JSON: {results_json_path}")
    
    print(f"\nDone! All results saved to {output_dir}")


if __name__ == '__main__':
    main()

