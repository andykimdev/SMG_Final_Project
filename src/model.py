import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """GCN encoder for VGAE"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn_mu = GCNConv(hidden_dim, latent_dim)
        self.gcn_logstd = GCNConv(hidden_dim, latent_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu = self.gcn_mu(x, edge_index)
        logstd = self.gcn_logstd(x, edge_index)
        return mu, logstd


class VGAE(nn.Module):
    """Variational Graph Autoencoder"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.latent_dim = latent_dim
        
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        return mu
    
    def decode(self, z):
        return torch.sigmoid(torch.mm(z, z.t()))
    
    def forward(self, x, edge_index):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        adj_recon = self.decode(z)
        return z, mu, logstd, adj_recon
