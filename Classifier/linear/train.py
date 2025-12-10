"""
Linear baseline models for cancer type classification.
Compares PCA+LogReg, PCA+SVM, and full LogReg against the graph transformer.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch

sys.path.append(str(Path(__file__).parent))
import config
from graph_prior import load_graph_prior
from dataset import load_and_preprocess_data


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


def load_transformer_results(output_dir):
    """Load graph transformer test results for comparison."""
    try:
        results_path = Path(output_dir) / 'results' / 'test_results.json'
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load transformer results: {e}")
    return None


def plot_pca_variance(pca, output_path):
    """Plot explained variance by PCA components."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, 'o-', linewidth=2, color='steelblue')
    axes[0].set_xlabel('Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('PCA Variance per Component', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'o-', linewidth=2, color='steelblue')
    axes[1].axhline(0.80, color='red', linestyle='--', label='80% variance', linewidth=2)
    axes[1].axhline(0.90, color='orange', linestyle='--', label='90% variance', linewidth=2)
    axes[1].axhline(0.95, color='green', linestyle='--', label='95% variance', linewidth=2)
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pca_2d(X_pca, y, class_names, output_path):
    """Plot first 2 PCA components colored by cancer type."""
    fig, ax = plt.subplots(figsize=(14, 11))
    
    n_classes = len(class_names)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    for idx, cancer in enumerate(class_names):
        mask = y == idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=[colors[idx]], label=cancer, alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('PCA: First 2 Components by Cancer Type', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix_single(cm, class_names, title, output_path):
    """Plot single confusion matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def train_linear_baseline(X_train, y_train, X_val, y_val, n_components=None, model_type='logreg'):
    """Train linear baseline with optional PCA."""
    if n_components is not None:
        print(f"  Applying PCA: {n_components} components")
        pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        X_train_transformed = pca.fit_transform(X_train)
        X_val_transformed = pca.transform(X_val)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        pca = None
        X_train_transformed = X_train
        X_val_transformed = X_val
    
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_C = None
    best_val_acc = 0.0
    best_model = None
    
    print(f"  Tuning C over {C_values}...")
    for C in C_values:
        if model_type == 'logreg':
            model = LogisticRegression(
                C=C, max_iter=1000, multi_class='multinomial',
                solver='lbfgs', random_state=config.RANDOM_SEED, n_jobs=-1
            )
        elif model_type == 'svm':
            model = LinearSVC(
                C=C, max_iter=2000, dual=False,
                random_state=config.RANDOM_SEED
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model.fit(X_train_transformed, y_train)
        y_val_pred = model.predict(X_val_transformed)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"    C={C:8.4f}: val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
            best_model = model
    
    print(f"  Best: C={best_C}, val_acc={best_val_acc:.4f}")
    
    return best_model, pca, best_C


def evaluate_model(model, X, y, pca=None):
    """Evaluate model and return metrics."""
    if pca is not None:
        X_transformed = pca.transform(X)
    else:
        X_transformed = X
    
    y_pred = model.predict(X_transformed)
    
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': y_pred,
        'confusion_matrix': cm
    }


def plot_comparison_with_transformer(results, transformer_results, output_path):
    """Create comparison bar chart with transformer results."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = [r['model_name'] for r in results]
    accuracies = [r['test_accuracy'] * 100 for r in results]
    
    # Add transformer result if available
    if transformer_results:
        model_names.append('Graph Transformer')
        accuracies.append(transformer_results['accuracy'] * 100)
    
    x_pos = np.arange(len(model_names))
    colors = ['steelblue'] * len(results) + ['darkgreen']
    
    bars = ax.bar(x_pos, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Highlight transformer bar
    if transformer_results:
        bars[-1].set_color('darkgreen')
        bars[-1].set_edgecolor('darkred')
        bars[-1].set_linewidth(2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison: Linear Baselines vs Graph Transformer', fontsize=14, fontweight='bold')
    
    # Random baseline
    n_classes = 32
    ax.axhline(100 / n_classes, color='red', linestyle='--', linewidth=2,
              label=f'Random baseline ({100/n_classes:.1f}%)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, max(accuracies) + 8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_f1_comparison(results, transformer_results, output_path):
    """Create F1 score comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    model_names = [r['model_name'] for r in results]
    f1_macro = [r['test_f1_macro'] * 100 for r in results]
    f1_weighted = [r['test_f1_weighted'] * 100 for r in results]
    
    if transformer_results:
        model_names.append('Graph Transformer')
        f1_macro.append(transformer_results['f1_macro'] * 100)
        f1_weighted.append(transformer_results['f1_weighted'] * 100)
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, f1_macro, width, label='F1 Macro', 
                   color='steelblue', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x_pos + width/2, f1_weighted, width, label='F1 Weighted',
                   color='coral', edgecolor='black', linewidth=1)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=11)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Linear baselines for cancer classification")
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../../results/classifiers/cancer_type_classifiers/linear')
    parser.add_argument('--transformer_dir', type=str, default='../../results/classifiers/cancer_type_classifiers/transformer',
                       help='Directory with transformer results for comparison')
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f'training_log_{timestamp}.txt'
    logger = Logger(log_path)
    sys.stdout = logger
    
    print("="*80)
    print("Linear Baseline Models for Cancer Type Classification")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Random seed: {config.RANDOM_SEED}")
    
    # Set seeds
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    
    # Load transformer results
    transformer_results = load_transformer_results(args.transformer_dir)
    if transformer_results:
        print(f"\nLoaded transformer results:")
        print(f"  Accuracy: {transformer_results['accuracy']:.4f}")
        print(f"  F1 macro: {transformer_results['f1_macro']:.4f}")
    
    # Load data
    print("\nLoading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    
    print("\nLoading data...")
    data_splits, label_info, scaler = load_and_preprocess_data(
        args.csv_path, graph_prior['protein_cols']
    )
    
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['val']
    X_test, y_test = data_splits['test']
    
    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {label_info['n_classes']}")
    print(f"  Features: {X_train.shape[1]}")
    
    # PCA analysis
    print("\n" + "="*80)
    print("PCA Analysis")
    print("="*80)
    
    pca_full = PCA(random_state=config.RANDOM_SEED)
    pca_full.fit(X_train)
    
    plot_pca_variance(pca_full, plots_dir / 'pca_variance_explained.png')
    
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_80 = np.argmax(cumsum >= 0.80) + 1
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    
    print(f"\nComponents needed:")
    print(f"  80% variance: {n_80} components ({n_80/X_train.shape[1]*100:.1f}% of features)")
    print(f"  90% variance: {n_90} components ({n_90/X_train.shape[1]*100:.1f}% of features)")
    print(f"  95% variance: {n_95} components ({n_95/X_train.shape[1]*100:.1f}% of features)")
    
    # 2D visualization
    print("\nCreating 2D PCA visualization...")
    pca_2d = PCA(n_components=2, random_state=config.RANDOM_SEED)
    X_train_2d = pca_2d.fit_transform(X_train)
    plot_pca_2d(X_train_2d, y_train, label_info['class_names'], 
                plots_dir / 'pca_2d_cancer_types.png')
    print(f"  First 2 PCs explain {pca_2d.explained_variance_ratio_.sum()*100:.2f}% variance")
    
    # Train models
    models_to_test = [
        ('PCA50+LogReg', 50, 'logreg'),
        ('PCA80+LogReg', n_80, 'logreg'),
        ('PCA90+LogReg', n_90, 'logreg'),
        ('PCA95+LogReg', n_95, 'logreg'),
        ('Full+LogReg', None, 'logreg'),
        ('PCA80+SVM', n_80, 'svm'),
    ]
    
    results = []
    
    for name, n_comp, model_type in models_to_test:
        print("\n" + "="*80)
        print(f"Training: {name}")
        print("="*80)
        
        model, pca, best_C = train_linear_baseline(
            X_train, y_train, X_val, y_val,
            n_components=n_comp, model_type=model_type
        )
        
        # Evaluate
        print("\nTest evaluation:")
        test_metrics = evaluate_model(model, X_test, y_test, pca)
        
        print(f"  Accuracy:      {test_metrics['accuracy']:.4f}")
        print(f"  F1 (macro):    {test_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        
        # Store results
        results.append({
            'model_name': name,
            'n_components': int(n_comp) if n_comp else 198,
            'model_type': model_type,
            'best_C': float(best_C),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1_macro': float(test_metrics['f1_macro']),
            'test_f1_weighted': float(test_metrics['f1_weighted']),
        })
        
        # Save confusion matrix
        cm_path = plots_dir / f'confusion_matrix_{name.replace("+", "_")}.png'
        plot_confusion_matrix_single(
            test_metrics['confusion_matrix'],
            label_info['class_names'],
            f'{name} - Confusion Matrix',
            cm_path
        )
        
        # Save classification report
        report = classification_report(
            y_test, test_metrics['predictions'],
            target_names=label_info['class_names'], digits=4
        )
        report_path = results_dir / f'{name.replace("+", "_")}_classification_report.txt'
        report_path.write_text(report)
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Components':<12} {'C':<10} {'Accuracy':<10} {'F1 Macro':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['model_name']:<20} {str(r['n_components']):<12} "
              f"{r['best_C']:<10.4f} {r['test_accuracy']:<10.4f} {r['test_f1_macro']:<10.4f}")
    
    if transformer_results:
        print(f"{'Graph Transformer':<20} {'N/A':<12} {'N/A':<10} "
              f"{transformer_results['accuracy']:<10.4f} {transformer_results['f1_macro']:<10.4f}")
    
    # Save results
    results_json_path = results_dir / 'baseline_results.json'
    with open(results_json_path, 'w') as f:
        json.dump({
            'baseline_models': results,
            'transformer': transformer_results,
            'timestamp': timestamp
        }, f, indent=2)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison_with_transformer(results, transformer_results, 
                                    plots_dir / 'accuracy_comparison.png')
    plot_f1_comparison(results, transformer_results,
                      plots_dir / 'f1_comparison.png')
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  Log file: {log_path}")
    print(f"  Results:  {results_dir}")
    print(f"  Plots:    {plots_dir}")
    
    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()

