from typing import Dict

import torch
import torch.nn as nn

from . import config


class ContextEncoder(nn.Module):
    def __init__(self, num_cancers, num_stages, num_sexes):
        super().__init__()
        d_model = config.MODEL["d_model"]
        ctx_cfg = config.MODEL.get("context_splits", {})
        self.cancer_dim = ctx_cfg.get("cancer", config.MODEL["context_dim"] // 2)
        self.stage_dim = ctx_cfg.get("stage", config.MODEL["context_dim"] // 4)
        self.sex_dim = ctx_cfg.get("sex", config.MODEL["context_dim"] // 4)
        self.age_dim = ctx_cfg.get("age", config.MODEL["context_dim"] // 4)

        self.cancer_emb = nn.Embedding(num_cancers, self.cancer_dim)
        self.stage_emb = nn.Embedding(num_stages, self.stage_dim)
        self.sex_emb = nn.Embedding(num_sexes, self.sex_dim)
        self.age_mlp = nn.Sequential(
            nn.Linear(1, self.age_dim),
            nn.GELU(),
            nn.Linear(self.age_dim, self.age_dim),
        )
        concat_dim = self.cancer_dim + self.stage_dim + self.sex_dim + self.age_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        cancer = self.cancer_emb(context["cancer_type"])
        stage = self.stage_emb(context["stage"])
        sex = self.sex_emb(context["sex"])
        age = self.age_mlp(context["age"])
        ctx = torch.cat([cancer, stage, sex, age], dim=-1)
        return self.proj(ctx)


class SimpleTransformerGenerator(nn.Module):
    def __init__(self, n_proteins, num_cancers, num_stages, num_sexes):
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_cfg["n_layers"],
        )

        self.context_encoder = ContextEncoder(num_cancers, num_stages, num_sexes)
        self.protein_id_emb = nn.Embedding(n_proteins, d_model)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = context["cancer_type"].size(0)
        device = context["cancer_type"].device
        d_model = config.MODEL["d_model"]

        noise = torch.randn(bsz, self.n_proteins, d_model, device=device)
        tokens = noise + self.protein_id_emb.weight.unsqueeze(0)

        ctx = self.context_encoder(context).unsqueeze(1)  # [B,1,d_model]
        tokens = torch.cat([ctx, tokens], dim=1)

        encoded = self.transformer(tokens)
        protein_tokens = encoded[:, 1:, :]
        preds = self.output_head(protein_tokens).squeeze(-1)
        return preds

    @torch.no_grad()
    def generate(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        return self.forward(context)

