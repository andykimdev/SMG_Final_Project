"""Training script for Graph-Aware Protein Diffusion Model."""

import argparse
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from diffusion_model import create_diffusion_model
from diffusion_utils import GaussianDiffusion, EMA
from dataset_diffusion import load_and_preprocess_diffusion_data, create_dataloaders

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Cancer Classifier'))
from graph_prior import load_graph_prior


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
    parser = argparse.ArgumentParser(description="Train protein diffusion model")
    parser.add_argument('--csv_path', type=str, default=config.PATHS['csv_path'])
    parser.add_argument('--prior_path', type=str, default=config.PATHS['prior_path'])
    parser.add_argument('--output_dir', type=str, default=config.PATHS['output_dir'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_transfer', action='store_true')
    return parser.parse_args()


def train_epoch(model, diffusion, loader, optimizer, device, ema=None, grad_clip=None, epoch=0):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (x_0, context_dict) in enumerate(pbar):
        x_0 = x_0.to(device)
        context_dict = {k: v.to(device) for k, v in context_dict.items()}
        
        optimizer.zero_grad()
        loss = diffusion.training_loss(model, x_0, context_dict)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, diffusion, loader, device, epoch=0):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Val]')
    for x_0, context_dict in pbar:
        x_0 = x_0.to(device)
        context_dict = {k: v.to(device) for k, v in context_dict.items()}
        
        loss = diffusion.training_loss(model, x_0, context_dict)
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def plot_training_curves(history, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    config.set_seed(config.RANDOM_SEED)
    
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    plots_dir = output_dir / 'plots'
    results_dir = output_dir / 'results'
    
    for d in [checkpoint_dir, plots_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f'logs_{timestamp}.txt'
    logger = Logger(log_path)
    sys.stdout = logger
    
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    
    # Load graph prior
    print("\nLoading graph prior...")
    graph_prior = load_graph_prior(args.prior_path)
    
    # Load data
    print("Loading data...")
    data_splits, context_info, scaler, label_info = load_and_preprocess_diffusion_data(
        args.csv_path, graph_prior['protein_cols']
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_splits, num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = create_diffusion_model(
        graph_prior=graph_prior,
        num_cancer_types=context_info['num_cancer_types'],
        num_stages=context_info['num_stages'],
        num_sexes=context_info['num_sexes'],
        num_survival_status=context_info.get('num_survival_status', 3),
        load_classifier=not args.no_transfer,
        device=args.device
    )
    
    # Initialize diffusion
    diffusion = GaussianDiffusion(
        timesteps=config.DIFFUSION['timesteps'],
        schedule=config.DIFFUSION['schedule'],
        loss_type=config.TRAINING['loss_type'],
        cosine_s=config.DIFFUSION.get('cosine_s', 0.008)
    )
    
    print(f"Timesteps: {config.DIFFUSION['timesteps']}")
    print(f"Schedule: {config.DIFFUSION['schedule']}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.TRAINING['learning_rate'],
        weight_decay=config.TRAINING['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.TRAINING['scheduler']['factor'],
        patience=config.TRAINING['scheduler']['patience'],
        min_lr=config.TRAINING['scheduler']['min_lr']
    )
    
    ema = None
    if config.TRAINING['use_ema']:
        ema = EMA(model, decay=config.TRAINING['ema_decay'])
    
    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    
    for epoch in range(config.TRAINING['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config.TRAINING['max_epochs']}")
        
        train_loss = train_epoch(
            model, diffusion, train_loader, optimizer, args.device,
            ema=ema, grad_clip=config.TRAINING['grad_clip'], epoch=epoch
        )
        
        val_loss = validate(model, diffusion, val_loader, args.device, epoch=epoch)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        print(f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
        
        improvement = best_val_loss - val_loss
        if improvement > config.TRAINING['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'context_info': context_info,
                'label_info': label_info,
            }
            
            if ema is not None:
                checkpoint['ema_state_dict'] = ema.state_dict()
            
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config.TRAINING['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % config.TRAINING['save_every'] == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_curves(history, plots_dir / 'training_curves.png')
    
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Results saved to {output_dir}")
    
    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()
