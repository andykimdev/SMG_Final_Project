"""
Graph-aware Transformer Classifier for cancer type prediction.
Uses protein expression + STRING PPI network structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

# NOTE FOR REPORT: Fixed graph-aware attention implementation (was previously not using K-bias)
# The original implementation computed graph attention bias but didn't apply it in MultiheadAttention.
# This custom implementation properly incorporates PPI network structure into attention scores.


class GraphTransformerClassifier(nn.Module):
    """
    Graph-aware transformer that incorporates PPI network structure
    via graph positional encodings and attention bias.
    """
    
    def __init__(
        self,
        n_proteins: int,
        n_classes: int,
        diffusion_kernel: torch.Tensor,
        positional_encodings: torch.Tensor,
        embedding_dim: int = None,
        n_layers: int = None,
        n_heads: int = None,
        dropout: float = None,
    ):
        """
        Args:
            n_proteins: Number of protein features (N)
            n_classes: Number of cancer types to classify
            diffusion_kernel: Graph diffusion kernel K (N×N)
            positional_encodings: Graph positional encodings PE (N×k)
            embedding_dim: Dimension of token embeddings
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Use config defaults if not specified
        self.embedding_dim = embedding_dim or config.MODEL['embedding_dim']
        self.n_layers = n_layers or config.MODEL['n_layers']
        self.n_heads = n_heads or config.MODEL['n_heads']
        self.dropout = dropout or config.MODEL['dropout']
        self.ffn_dim = config.MODEL['ffn_dim']
        
        self.n_proteins = n_proteins
        self.n_classes = n_classes
        
        # Register graph features as buffers (not parameters)
        self.register_buffer('K', diffusion_kernel)  # (N, N)
        self.register_buffer('PE', positional_encodings)  # (N, k)
        
        # Token embedding components
        self.value_projection = nn.Linear(1, self.embedding_dim)  # Project scalar expression value
        self.protein_embedding = nn.Embedding(n_proteins, self.embedding_dim)  # Learned protein-ID embeddings
        self.pe_projection = nn.Linear(positional_encodings.shape[1], self.embedding_dim)  # Project graph PE
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        # Graph attention bias (learnable per head)
        self.graph_bias_scale = nn.Parameter(
            torch.ones(self.n_heads) * config.MODEL['graph_bias_scale']
        )
        
        # Transformer encoder
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize embeddings
        nn.init.normal_(self.protein_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Protein expression values (batch_size, n_proteins)
            
        Returns:
            logits: Class logits (batch_size, n_classes)
        """
        batch_size = x.shape[0]
        
        # Create token embeddings for each protein
        # 1. Project expression values
        value_emb = self.value_projection(x.unsqueeze(-1))  # (B, N, d)
        
        # 2. Add protein ID embeddings
        protein_ids = torch.arange(self.n_proteins, device=x.device)
        protein_emb = self.protein_embedding(protein_ids)  # (N, d)
        protein_emb = protein_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d)
        
        # 3. Add graph positional encodings
        pe_emb = self.pe_projection(self.PE)  # (N, d)
        pe_emb = pe_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, d)
        
        # Combine all embeddings
        protein_tokens = value_emb + protein_emb + pe_emb  # (B, N, d)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls_tokens, protein_tokens], dim=1)  # (B, N+1, d)
        
        # Create graph attention bias
        # For protein-protein attention, add bias based on diffusion kernel
        graph_bias = self._create_graph_attention_bias(batch_size)  # (B, n_heads, N+1, N+1)
        
        # Apply transformer layers
        for layer in self.transformer:
            tokens = layer(tokens, graph_bias)
        
        tokens = self.norm(tokens)
        
        # Extract CLS token representation
        cls_output = tokens[:, 0, :]  # (B, d)
        
        # Classification
        logits = self.classifier(cls_output)  # (B, n_classes)
        
        return logits
    
    def _create_graph_attention_bias(self, batch_size: int) -> torch.Tensor:
        """
        Create attention bias matrix incorporating graph structure.
        
        Args:
            batch_size: Current batch size
            
        Returns:
            bias: Attention bias (batch_size, n_heads, N+1, N+1)
        """
        device = self.K.device
        seq_len = self.n_proteins + 1  # +1 for CLS token
        
        # Initialize bias matrix (all zeros)
        bias = torch.zeros(self.n_heads, seq_len, seq_len, device=device)
        
        # Add graph structure bias for protein-protein attention
        # (CLS token has no graph bias)
        for h in range(self.n_heads):
            # Add scaled diffusion kernel to protein-protein block
            bias[h, 1:, 1:] = self.graph_bias_scale[h] * self.K
        
        # Expand for batch dimension
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, n_heads, N+1, N+1)
        
        return bias


class GraphAwareMultiheadAttention(nn.Module):
    """
    Custom multi-head attention that properly applies graph structure bias.
    
    NOTE FOR REPORT: This replaces PyTorch's MultiheadAttention to enable
    graph-aware attention via the STRING PPI diffusion kernel.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attn_bias: Attention bias (batch_size, n_heads, seq_len, seq_len)
                      Added to attention logits before softmax
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V and reshape for multi-head attention
        # (B, L, D) -> (B, L, n_heads, d_head) -> (B, n_heads, L, d_head)
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        # (B, n_heads, L, d_head) @ (B, n_heads, d_head, L) -> (B, n_heads, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # **KEY FIX: Add graph bias to attention scores**
        if attn_bias is not None:
            scores = scores + attn_bias
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (B, n_heads, L, L) @ (B, n_heads, L, d_head) -> (B, n_heads, L, d_head)
        output = torch.matmul(attn_weights, V)
        
        # Reshape back to (B, L, D)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        
        # Final output projection
        output = self.out_proj(output)
        
        return output


class GraphTransformerLayer(nn.Module):
    """Single transformer layer with graph-aware attention."""
    
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Use custom graph-aware attention instead of PyTorch's MultiheadAttention
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
        """
        Forward pass with graph-aware attention bias.
        
        Args:
            x: Input tokens (batch_size, seq_len, d_model)
            attn_bias: Attention bias (batch_size, n_heads, seq_len, seq_len)
                      NOW PROPERLY APPLIED to incorporate PPI network structure
            
        Returns:
            Output tokens (batch_size, seq_len, d_model)
        """
        # Self-attention with residual (pre-norm)
        residual = x
        x = self.norm1(x)
        
        # Graph-aware attention with bias actually applied
        attn_output = self.self_attn(x, attn_bias=attn_bias)
        x = residual + self.dropout1(attn_output)
        
        # FFN with residual (pre-norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        
        return x
