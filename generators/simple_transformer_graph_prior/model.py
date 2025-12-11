from typing import Dict

import torch
import torch.nn as nn

from simple_generator.model import ContextEncoder
from . import config


class GraphFeatureIntegrator(nn.Module):
    """
    Injects fixed graph positional encodings and diffusion mixing into token states.
    """

    def __init__(self, graph_tensors: Dict[str, torch.Tensor]):
        super().__init__()
        model_cfg = config.MODEL
        self.register_buffer("positional_encodings", graph_tensors["PE"])
        self.register_buffer("diffusion_kernel", graph_tensors["K"])

        pe_dim = self.positional_encodings.size(-1)
        d_model = model_cfg["d_model"]
        self.pe_proj = nn.Sequential(
            nn.LayerNorm(pe_dim) if pe_dim > 0 else nn.Identity(),
            nn.Linear(pe_dim, d_model) if pe_dim > 0 else nn.Identity(),
        )
        self.dropout = nn.Dropout(model_cfg.get("graph_feature_dropout", 0.0))
        init = torch.tensor(model_cfg.get("diffusion_blend_init", 0.0))
        self.diffusion_blend = nn.Parameter(init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, d_model]
        """
        if self.positional_encodings.numel() > 0:
            pe = self.pe_proj(self.positional_encodings).unsqueeze(0)
            tokens = tokens + self.dropout(pe)

        if self.diffusion_kernel is not None:
            blend = torch.sigmoid(self.diffusion_blend)
            if blend > 0:
                diffused = torch.einsum("ij,bjd->bid", self.diffusion_kernel, tokens)
                tokens = blend * diffused + (1 - blend) * tokens

        return tokens


class SimpleTransformerGraphPrior(nn.Module):
    def __init__(self, n_proteins, num_cancers, num_stages, num_sexes, graph_tensors):
        super().__init__()
        model_cfg = config.MODEL
        d_model = model_cfg["d_model"]
        self.n_proteins = n_proteins

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=model_cfg["n_heads"],
            dim_feedforward=model_cfg["ffn_dim"],
            dropout=model_cfg["dropout"],
            activation=model_cfg["activation"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg["n_layers"])

        self.context_encoder = ContextEncoder(num_cancers, num_stages, num_sexes)
        self.protein_id_emb = nn.Embedding(n_proteins, d_model)
        self.graph_integrator = GraphFeatureIntegrator(graph_tensors)

        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = context["cancer_type"].size(0)
        device = context["cancer_type"].device
        d_model = config.MODEL["d_model"]

        noise = torch.randn(bsz, self.n_proteins, d_model, device=device)
        tokens = noise + self.protein_id_emb.weight.unsqueeze(0)
        tokens = self.graph_integrator(tokens)

        ctx = self.context_encoder(context).unsqueeze(1)
        tokens = torch.cat([ctx, tokens], dim=1)

        encoded = self.transformer(tokens)
        protein_tokens = encoded[:, 1:, :]
        preds = self.output_head(protein_tokens).squeeze(-1)
        return preds

    @torch.no_grad()
    def generate(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        return self.forward(context)

