import torch
import yaml
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.model import VGAE
from src.train import Trainer


def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(
        config['data']['string_file'],
        config['data']['hpa_file'],
        config['data']['min_score']
    )
    data = data_loader.load()
    
    # Initialize model
    model = VGAE(
        input_dim=data.x.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        dropout=config['model']['dropout']
    )
    
    # Train
    print("Training...")
    trainer = Trainer(
        model,
        device,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    trainer.train(data, epochs=config['training']['epochs'])
    
    # Save results
    trainer.plot_history()
    embeddings = trainer.get_embeddings(data)
    np.save('latent_embeddings.npy', embeddings)
    torch.save(model.state_dict(), 'vgae_model.pth')
    print("Model and embeddings saved!")


if __name__ == '__main__':
    main()

