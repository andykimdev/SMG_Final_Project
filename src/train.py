import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def vgae_loss(mu, logstd, adj_true, adj_recon):
    """VGAE loss: reconstruction + KL divergence"""
    recon_loss = F.binary_cross_entropy(adj_recon, adj_true)
    kl_loss = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp())
    kl_loss = kl_loss / adj_true.numel()
    return recon_loss + kl_loss, recon_loss, kl_loss


class Trainer:
    """Train VGAE model"""
    
    def __init__(self, model, device, lr=0.01, weight_decay=0.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {'loss': [], 'recon': [], 'kl': []}
        
    def train_epoch(self, data):
        self.model.train()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        adj_true = data.adj.to(self.device)
        
        self.optimizer.zero_grad()
        z, mu, logstd, adj_recon = self.model(x, edge_index)
        loss, recon, kl = vgae_loss(mu, logstd, adj_true, adj_recon)
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), recon.item(), kl.item()
    
    def train(self, data, epochs=100):
        for epoch in tqdm(range(epochs)):
            loss, recon, kl = self.train_epoch(data)
            self.history['loss'].append(loss)
            self.history['recon'].append(recon)
            self.history['kl'].append(kl)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Loss={loss:.4f}, Recon={recon:.4f}, KL={kl:.4f}")
    
    def plot_history(self, save_path='training_history.png'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self.history['loss'])
        axes[0].set_title('Total Loss')
        axes[1].plot(self.history['recon'])
        axes[1].set_title('Reconstruction Loss')
        axes[2].plot(self.history['kl'])
        axes[2].set_title('KL Loss')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    
    @torch.no_grad()
    def get_embeddings(self, data):
        self.model.eval()
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        mu, _ = self.model.encode(x, edge_index)
        return mu.cpu().numpy()
