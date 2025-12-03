"""
Baseline survival prediction models for comparison with Graph Transformer.
Tests whether PPI network topology provides predictive value.
"""

import torch
import torch.nn as nn
from .. import config


class MLPSurvivalModel(nn.Module):
    """
    Simple MLP baseline - no network topology.
    Uses concatenated protein + clinical + genomic features.
    """

    def __init__(self, n_proteins: int, n_clinical: int, n_genomic: int,
                 hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()

        self.n_proteins = n_proteins
        self.n_clinical = n_clinical
        self.n_genomic = n_genomic

        # Total input features
        n_features = n_proteins + n_clinical + n_genomic

        # Build MLP
        layers = []
        prev_dim = n_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer (single risk score)
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        """Concatenate all features and pass through MLP."""
        features = torch.cat([
            batch['protein'],
            batch['clinical'],
            batch['genomic']
        ], dim=1)
        return self.model(features)


class VanillaMultiheadAttention(nn.Module):
    """
    Custom multi-head attention WITHOUT graph bias.
    Identical to GraphAwareMultiheadAttention but ignores attn_bias parameter.
    This ensures a fair comparison with the Graph Transformer.
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

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attn_bias: Ignored (kept for API compatibility with graph model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        import math
        B, L, D = x.shape

        # Project to Q, K, V and reshape for multi-head attention
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores (NO graph bias)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape back to (B, L, D)
        output = output.transpose(1, 2).contiguous().view(B, L, D)

        # Final output projection
        output = self.out_proj(output)

        return output


class VanillaTransformerLayer(nn.Module):
    """Single transformer layer WITHOUT graph-aware attention.
    Identical architecture to GraphTransformerLayer but uses vanilla attention."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        # Use vanilla attention (no graph bias)
        self.self_attn = VanillaMultiheadAttention(
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
        Forward pass WITHOUT graph-aware attention bias.

        Args:
            x: Input tokens (batch_size, seq_len, d_model)
            attn_bias: Ignored (kept for API compatibility)

        Returns:
            Output tokens (batch_size, seq_len, d_model)
        """
        # Self-attention with residual (pre-norm)
        residual = x
        x = self.norm1(x)

        # Vanilla attention (attn_bias ignored)
        attn_output = self.self_attn(x, attn_bias=None)
        x = residual + self.dropout1(attn_output)

        # FFN with residual (pre-norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))

        return x


class VanillaTransformerSurvival(nn.Module):
    """
    Vanilla Transformer baseline - NO graph topology.
    Uses EXACT SAME custom architecture as Graph Transformer but WITHOUT:
    - Graph positional encodings (PE)
    - Graph-aware attention bias (diffusion kernel K)

    This provides a fair comparison to test if PPI network structure helps.
    """

    def __init__(
        self,
        n_proteins: int,
        n_clinical: int,
        n_genomic: int,
        embedding_dim: int = None,
        n_layers: int = None,
        n_heads: int = None,
        dropout: float = None,
    ):
        super().__init__()

        # Use config defaults (IDENTICAL to Graph Transformer)
        self.embedding_dim = embedding_dim or config.MODEL['embedding_dim']
        self.n_layers = n_layers or config.MODEL['n_layers']
        self.n_heads = n_heads or config.MODEL['n_heads']
        self.dropout = dropout or config.MODEL['dropout']
        self.ffn_dim = config.MODEL['ffn_dim']

        self.n_proteins = n_proteins
        self.n_clinical = n_clinical
        self.n_genomic = n_genomic

        # Token embeddings (IDENTICAL to graph model, minus PE projection)
        self.value_projection = nn.Linear(1, self.embedding_dim)
        self.protein_embedding = nn.Embedding(n_proteins, self.embedding_dim)

        # Special tokens (IDENTICAL)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        # Clinical and genomic projections (IDENTICAL)
        if n_clinical > 0:
            self.clinical_projection = nn.Sequential(
                nn.Linear(n_clinical, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            self.clinical_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        else:
            self.clinical_projection = None
            self.clinical_token = None

        if n_genomic > 0:
            self.genomic_projection = nn.Sequential(
                nn.Linear(n_genomic, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            self.genomic_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        else:
            self.genomic_projection = None
            self.genomic_token = None

        # Vanilla Transformer encoder (IDENTICAL architecture, NO graph bias)
        self.transformer = nn.ModuleList([
            VanillaTransformerLayer(
                d_model=self.embedding_dim,
                n_heads=self.n_heads,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
            ) for _ in range(self.n_layers)
        ])

        self.norm = nn.LayerNorm(self.embedding_dim)

        # Risk prediction head (IDENTICAL)
        n_special_tokens = 1 + (1 if n_clinical > 0 else 0) + (1 if n_genomic > 0 else 0)
        self.risk_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * n_special_tokens, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.protein_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.clinical_token is not None:
            nn.init.normal_(self.clinical_token, std=0.02)
        if self.genomic_token is not None:
            nn.init.normal_(self.genomic_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch['protein']
        batch_size = x.shape[0]

        # Create protein tokens (NO graph PE)
        value_emb = self.value_projection(x.unsqueeze(-1))
        protein_ids = torch.arange(self.n_proteins, device=x.device)
        protein_emb = self.protein_embedding(protein_ids).unsqueeze(0).expand(batch_size, -1, -1)
        protein_tokens = value_emb + protein_emb

        # Create special tokens (IDENTICAL to graph model)
        special_tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if self.clinical_projection is not None:
            clinical_emb = self.clinical_projection(batch['clinical'])
            clinical_tokens = self.clinical_token.expand(batch_size, -1, -1) + clinical_emb.unsqueeze(1)
            special_tokens.append(clinical_tokens)

        if self.genomic_projection is not None:
            genomic_emb = self.genomic_projection(batch['genomic'])
            genomic_tokens = self.genomic_token.expand(batch_size, -1, -1) + genomic_emb.unsqueeze(1)
            special_tokens.append(genomic_tokens)

        # Concatenate all tokens
        special_tokens.append(protein_tokens)
        tokens = torch.cat(special_tokens, dim=1)

        # Apply vanilla transformer layers (NO graph bias)
        for layer in self.transformer:
            tokens = layer(tokens, attn_bias=None)

        tokens = self.norm(tokens)

        # Extract special tokens for prediction (IDENTICAL)
        n_special = len(special_tokens) - 1  # Exclude protein tokens
        special_outputs = tokens[:, :n_special, :]
        special_outputs_flat = special_outputs.reshape(batch_size, -1)
        risk_scores = self.risk_predictor(special_outputs_flat)

        return risk_scores


class ProteinOnlyMLP(nn.Module):
    """
    Protein-only baseline - tests if clinical/genomic features help.
    Uses only protein expression (no clinical, no genomic, no topology).
    """

    def __init__(self, n_proteins: int, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()

        layers = []
        prev_dim = n_proteins
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch['protein'])