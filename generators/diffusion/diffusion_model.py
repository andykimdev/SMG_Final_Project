"""
Graph-Aware Protein Expression Diffusion Model.
Learns p(proteome | patient_context, STRING_PPI_network) using DDPM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Cancer Classifier'))

from graph_transformer_classifier import GraphAwareMultiheadAttention, GraphTransformerLayer
import config
from context_encoder import HierarchicalContextEncoder


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class TimeEmbedding(nn.Module):
    """Time embedding with MLP."""
    
    def __init__(self, dim: int, time_dim: Optional[int] = None):
        super().__init__()
        
        if time_dim is None:
            time_dim = dim * 4
        
        self.sinusoidal = SinusoidalPositionEmbeddings(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, dim)
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(time)
        return self.mlp(emb)


class GraphProteinDiffusion(nn.Module):
    """Graph-aware diffusion model for protein expression generation."""
    
    def __init__(
        self,
        n_proteins: int,
        diffusion_kernel: torch.Tensor,
        positional_encodings: torch.Tensor,
        num_cancer_types: int = 32,
        num_stages: int = 20,
        num_sexes: int = 3,
        num_survival_status: int = 3,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        
        self.d_model = d_model if d_model is not None else config.MODEL['embedding_dim']
        self.n_layers = n_layers if n_layers is not None else config.MODEL['n_layers']
        self.n_heads = n_heads if n_heads is not None else config.MODEL['n_heads']
        self.dropout = dropout if dropout is not None else config.MODEL['dropout']
        self.ffn_dim = config.MODEL['ffn_dim']
        
        self.n_proteins = n_proteins
        
        self.register_buffer('K', diffusion_kernel)
        self.register_buffer('PE', positional_encodings)
        
        self.context_encoder = HierarchicalContextEncoder(
            num_cancer_types=num_cancer_types,
            num_stages=num_stages,
            num_sexes=num_sexes,
            num_survival_status=num_survival_status,
            context_config=config.MODEL['context_encoder']
        )
        
        context_dim = config.MODEL['context_encoder']['context_dim']
        self.ctx_proj = nn.Linear(context_dim, self.d_model)
        
        self.time_embedding = TimeEmbedding(self.d_model)
        
        # FiLM conditioning
        self.time_scale = nn.Linear(self.d_model, self.d_model)
        self.time_shift = nn.Linear(self.d_model, self.d_model)
        
        self.value_projection = nn.Linear(1, self.d_model)
        self.protein_embedding = nn.Embedding(n_proteins, self.d_model)
        self.pe_projection = nn.Linear(positional_encodings.shape[1], self.d_model)
        
        self.graph_bias_scale = nn.Parameter(
            torch.ones(self.n_heads) * config.MODEL['graph_bias_scale']
        )
        
        self.transformer = nn.ModuleList([
            GraphTransformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout
            )
            for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.d_model)
        
        output_hidden_dim = config.MODEL['output_head']['hidden_dim']
        
        self.noise_head = nn.Sequential(
            nn.Linear(self.d_model, output_hidden_dim),
            nn.LayerNorm(output_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(output_hidden_dim, output_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(output_hidden_dim // 2, 1)
        )
        
        self.noise_residual = nn.Linear(self.d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.protein_embedding.weight, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        with torch.no_grad():
            for layer in self.noise_head:
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(0.1)
            self.noise_residual.weight.mul_(0.1)
            
            nn.init.zeros_(self.time_scale.weight)
            nn.init.zeros_(self.time_scale.bias)
            nn.init.zeros_(self.time_shift.weight)
            nn.init.zeros_(self.time_shift.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        batch_size, n_proteins = x_t.shape
        assert n_proteins == self.n_proteins
        
        z_c = self.context_encoder(context_dict)
        ctx_token = self.ctx_proj(z_c).unsqueeze(1)
        
        t_emb = self.time_embedding(t)
        
        value_emb = self.value_projection(x_t.unsqueeze(-1))
        
        protein_ids = torch.arange(self.n_proteins, device=x_t.device)
        protein_id_emb = self.protein_embedding(protein_ids)
        protein_id_emb = protein_id_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        protein_tokens = value_emb + protein_id_emb
        
        # FiLM conditioning with timestep
        time_scale = self.time_scale(t_emb).unsqueeze(1)
        time_shift = self.time_shift(t_emb).unsqueeze(1)
        protein_tokens = protein_tokens * (1.0 + time_scale) + time_shift
        
        tokens = torch.cat([ctx_token, protein_tokens], dim=1)
        
        graph_bias = self._create_graph_attention_bias(batch_size)
        
        for layer in self.transformer:
            tokens = layer(tokens, graph_bias)
        
        tokens = self.norm(tokens)
        
        protein_output = tokens[:, 1:, :]
        
        eps_main = self.noise_head(protein_output).squeeze(-1)
        eps_residual = self.noise_residual(protein_output).squeeze(-1)
        
        eps_pred = eps_main + 0.1 * eps_residual
        
        return eps_pred
    
    def _create_graph_attention_bias(self, batch_size: int) -> torch.Tensor:
        """Graph attention bias (disabled for ablation)."""
        device = self.K.device
        seq_len = self.n_proteins + 1
        return torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=device)


def create_diffusion_model(
    graph_prior: Dict,
    num_cancer_types: int = 32,
    num_stages: int = 20,
    num_sexes: int = 3,
    num_survival_status: int = 3,
    load_classifier: bool = False,
    classifier_path: Optional[str] = None,
    device: str = 'cpu'
) -> GraphProteinDiffusion:
    """Factory function to create diffusion model."""
    
    model = GraphProteinDiffusion(
        n_proteins=graph_prior['K'].shape[0],
        diffusion_kernel=torch.from_numpy(graph_prior['K']),
        positional_encodings=torch.from_numpy(graph_prior['PE']),
        num_cancer_types=num_cancer_types,
        num_stages=num_stages,
        num_sexes=num_sexes,
        num_survival_status=num_survival_status,
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: d={model.d_model}, layers={model.n_layers}, heads={model.n_heads}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model
