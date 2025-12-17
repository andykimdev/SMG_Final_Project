"""
Graph-aware Transformer Classifier for cancer type prediction.
Uses protein expression + STRING PPI network structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class GraphTransformerClassifier(nn.Module):
    """Graph-aware transformer that incorporates PPI network structure."""
    
    def __init__(
        self,
        n_proteins: int,
        n_classes: int,
        diffusion_kernel: torch.Tensor,
        positional_encodings: torch.Tensor,
        embedding_dim: int = None,
        n_layers: int = None,
        n_heads: int = None,
        ffn_dim: int = None,
        dropout: float = None,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim or config.MODEL['embedding_dim']
        self.n_layers = n_layers or config.MODEL['n_layers']
        self.n_heads = n_heads or config.MODEL['n_heads']
        self.dropout = dropout or config.MODEL['dropout']
        self.ffn_dim = ffn_dim or config.MODEL['ffn_dim']
        
        self.n_proteins = n_proteins
        self.n_classes = n_classes
        
        self.register_buffer('K', diffusion_kernel)
        self.register_buffer('PE', positional_encodings)
        
        self.value_projection = nn.Linear(1, self.embedding_dim)
        self.protein_embedding = nn.Embedding(n_proteins, self.embedding_dim)
        self.pe_projection = nn.Linear(positional_encodings.shape[1], self.embedding_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        self.graph_bias_scale = nn.Parameter(
            torch.ones(self.n_heads) * config.MODEL['graph_bias_scale']
        )
        
        encoder_layer = GraphTransformerLayer(
            d_model=self.embedding_dim,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
        )
        self.transformer = nn.ModuleList([
            GraphTransformerLayer(
                d_model=self.embedding_dim,
                n_heads=self.n_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
            ) for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.embedding_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.protein_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        value_emb = self.value_projection(x.unsqueeze(-1))
        
        protein_ids = torch.arange(self.n_proteins, device=x.device)
        protein_emb = self.protein_embedding(protein_ids)
        protein_emb = protein_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        pe_emb = self.pe_projection(self.PE)
        pe_emb = pe_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        protein_tokens = value_emb + protein_emb + pe_emb
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, protein_tokens], dim=1)
        
        graph_bias = self._create_graph_attention_bias(batch_size)
        
        for layer in self.transformer:
            tokens = layer(tokens, graph_bias)
        
        tokens = self.norm(tokens)
        cls_output = tokens[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits
    
    def _create_graph_attention_bias(self, batch_size: int) -> torch.Tensor:
        device = self.K.device
        seq_len = self.n_proteins + 1
        
        bias = torch.zeros(self.n_heads, seq_len, seq_len, device=device)
        
        for h in range(self.n_heads):
            bias[h, 1:, 1:] = self.graph_bias_scale[h] * self.K
        
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return bias


class GraphAwareMultiheadAttention(nn.Module):
    """Multi-head attention with graph structure bias."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if attn_bias is not None:
            scores = scores + attn_bias
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)
        
        return output


class GraphTransformerLayer(nn.Module):
    """Single transformer layer with graph-aware attention."""
    
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = GraphAwareMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_output = self.self_attn(x, attn_bias=attn_bias)
        x = residual + self.dropout1(attn_output)
        
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        
        return x
