"""
Hierarchical Patient Context Encoder for Protein Expression Generation.
Encodes patient covariates respecting biological hierarchy:
    Cancer Type > Stage > Demographics > Molecular > Survival
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import config


class SurvivalEncoder(nn.Module):
    """Encodes survival outcome pairs (status, months)."""
    
    def __init__(self, num_status: int = 3, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim
        
        self.status_embedding = nn.Embedding(num_status, output_dim // 2)
        
        self.months_encoder = nn.Sequential(
            nn.Linear(1, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim // 2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.status_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, status: torch.Tensor, months: torch.Tensor) -> torch.Tensor:
        status_emb = self.status_embedding(status)
        months_emb = self.months_encoder(months)
        combined = torch.cat([status_emb, months_emb], dim=-1)
        return self.fusion(combined)


class HierarchicalContextEncoder(nn.Module):
    """Encodes patient context with hierarchical weighting."""
    
    def __init__(
        self,
        num_cancer_types: int = 32,
        num_stages: int = 20,
        num_sexes: int = 3,
        num_survival_status: int = 3,
        context_config: Optional[Dict] = None
    ):
        super().__init__()
        
        if context_config is None:
            context_config = config.MODEL['context_encoder']
        
        self.cancer_type_dim = context_config['cancer_type_dim']
        self.stage_dim = context_config['stage_dim']
        self.age_dim = context_config['age_dim']
        self.sex_dim = context_config['sex_dim']
        self.molecular_dim = context_config['molecular_dim']
        self.survival_dim = context_config['survival_dim']
        self.survival_per_outcome_dim = context_config['survival_per_outcome_dim']
        self.context_dim = context_config['context_dim']
        
        # Cancer type embedding
        self.cancer_type_embedding = nn.Embedding(num_cancer_types, self.cancer_type_dim)
        
        # Clinical stage
        self.stage_embedding = nn.Embedding(num_stages, self.stage_dim)
        
        # Age encoder
        self.age_encoder = nn.Sequential(
            nn.Linear(1, self.age_dim),
            nn.LayerNorm(self.age_dim),
            nn.GELU(),
            nn.Linear(self.age_dim, self.age_dim)
        )
        
        # Sex embedding
        self.sex_embedding = nn.Embedding(num_sexes, self.sex_dim)
        
        # Molecular features
        self.molecular_encoder = nn.Sequential(
            nn.Linear(4, self.molecular_dim),
            nn.LayerNorm(self.molecular_dim),
            nn.GELU(),
            nn.Linear(self.molecular_dim, self.molecular_dim),
            nn.LayerNorm(self.molecular_dim)
        )
        
        # Survival encoders
        self.os_encoder = SurvivalEncoder(num_survival_status, self.survival_per_outcome_dim)
        self.pfs_encoder = SurvivalEncoder(num_survival_status, self.survival_per_outcome_dim)
        self.dss_encoder = SurvivalEncoder(num_survival_status, self.survival_per_outcome_dim)
        self.dfs_encoder = SurvivalEncoder(num_survival_status, self.survival_per_outcome_dim)
        
        self.survival_fusion = nn.Sequential(
            nn.Linear(self.survival_per_outcome_dim * 4, self.survival_dim),
            nn.LayerNorm(self.survival_dim),
            nn.GELU()
        )
        
        # Fusion network
        total_dim = (
            self.cancer_type_dim + self.stage_dim + self.age_dim +
            self.sex_dim + self.molecular_dim + self.survival_dim
        )
        
        fusion_layers = []
        current_dim = total_dim
        
        for i in range(context_config['fusion_layers']):
            if i == context_config['fusion_layers'] - 1:
                next_dim = self.context_dim
            else:
                next_dim = (current_dim + self.context_dim) // 2
            
            fusion_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
            ])
            
            if i < context_config['fusion_layers'] - 1:
                fusion_layers.append(nn.Dropout(config.MODEL['dropout']))
            
            current_dim = next_dim
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Learnable level weights
        self.level_weights = nn.Parameter(
            torch.tensor([4.0, 2.0, 1.0, 1.0, 1.0, 1.5])
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.cancer_type_embedding.weight, std=0.02)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)
        nn.init.normal_(self.sex_embedding.weight, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, context_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = context_dict['cancer_type'].shape[0]
        
        cancer_emb = self.cancer_type_embedding(context_dict['cancer_type'])
        stage_emb = self.stage_embedding(context_dict['stage'])
        age_emb = self.age_encoder(context_dict['age'])
        sex_emb = self.sex_embedding(context_dict['sex'])
        molecular_emb = self.molecular_encoder(context_dict['molecular'])
        
        os_emb = self.os_encoder(context_dict['os_status'], context_dict['os_months'])
        pfs_emb = self.pfs_encoder(context_dict['pfs_status'], context_dict['pfs_months'])
        dss_emb = self.dss_encoder(context_dict['dss_status'], context_dict['dss_months'])
        dfs_emb = self.dfs_encoder(context_dict['dfs_status'], context_dict['dfs_months'])
        
        survival_combined = torch.cat([os_emb, pfs_emb, dss_emb, dfs_emb], dim=-1)
        survival_emb = self.survival_fusion(survival_combined)
        
        weights = F.softmax(self.level_weights, dim=0)
        
        cancer_emb = cancer_emb * weights[0]
        stage_emb = stage_emb * weights[1]
        age_emb = age_emb * weights[2]
        sex_emb = sex_emb * weights[3]
        molecular_emb = molecular_emb * weights[4]
        survival_emb = survival_emb * weights[5]
        
        combined = torch.cat([
            cancer_emb, stage_emb, age_emb,
            sex_emb, molecular_emb, survival_emb,
        ], dim=-1)
        
        z_c = self.fusion(combined)
        return z_c
    
    def get_level_importance(self) -> Dict[str, float]:
        """Get learned importance weights for each level."""
        weights = F.softmax(self.level_weights, dim=0)
        return {
            'cancer_type': weights[0].item(),
            'stage': weights[1].item(),
            'age': weights[2].item(),
            'sex': weights[3].item(),
            'molecular': weights[4].item(),
            'survival': weights[5].item(),
        }


class SimpleContextEncoder(nn.Module):
    """Non-hierarchical baseline for ablation."""
    
    def __init__(
        self,
        num_cancer_types: int = 32,
        num_stages: int = 20,
        num_sexes: int = 3,
        num_survival_status: int = 3,
        context_dim: int = 256,
    ):
        super().__init__()
        self.context_dim = context_dim
        
        self.cancer_emb = nn.Embedding(num_cancer_types, 32)
        self.stage_emb = nn.Embedding(num_stages, 16)
        self.sex_emb = nn.Embedding(num_sexes, 8)
        
        self.age_proj = nn.Linear(1, 8)
        self.molecular_proj = nn.Linear(4, 16)
        
        self.survival_status_emb = nn.Embedding(num_survival_status, 4)
        self.survival_months_proj = nn.Linear(4, 16)
        
        total_dim = 32 + 16 + 8 + 8 + 16 + 16 + 16
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim)
        )
    
    def forward(self, context_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        cancer_emb = self.cancer_emb(context_dict['cancer_type'])
        stage_emb = self.stage_emb(context_dict['stage'])
        sex_emb = self.sex_emb(context_dict['sex'])
        age_emb = self.age_proj(context_dict['age'])
        molecular_emb = self.molecular_proj(context_dict['molecular'])
        
        os_status_emb = self.survival_status_emb(context_dict['os_status'])
        pfs_status_emb = self.survival_status_emb(context_dict['pfs_status'])
        dss_status_emb = self.survival_status_emb(context_dict['dss_status'])
        dfs_status_emb = self.survival_status_emb(context_dict['dfs_status'])
        survival_status_emb = os_status_emb + pfs_status_emb + dss_status_emb + dfs_status_emb
        
        survival_months = torch.cat([
            context_dict['os_months'],
            context_dict['pfs_months'],
            context_dict['dss_months'],
            context_dict['dfs_months']
        ], dim=-1)
        survival_months_emb = self.survival_months_proj(survival_months)
        
        combined = torch.cat([
            cancer_emb, stage_emb, sex_emb, age_emb, molecular_emb,
            survival_status_emb, survival_months_emb
        ], dim=-1)
        
        return self.fusion(combined)


class ContextAugmentation:
    """Data augmentation for patient context during training."""
    
    def __init__(
        self,
        age_noise_std: float = 0.05,
        molecular_noise_std: float = 0.1,
        survival_months_noise_std: float = 0.05,
        dropout_prob: float = 0.0,
    ):
        self.age_noise_std = age_noise_std
        self.molecular_noise_std = molecular_noise_std
        self.survival_months_noise_std = survival_months_noise_std
        self.dropout_prob = dropout_prob
    
    def __call__(
        self,
        context_dict: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        if not training:
            return context_dict
        
        aug_context = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                       for k, v in context_dict.items()}
        
        if 'age' in aug_context and self.age_noise_std > 0:
            noise = torch.randn_like(aug_context['age']) * self.age_noise_std
            aug_context['age'] = torch.clamp(aug_context['age'] + noise, 0, 1)
        
        if 'molecular' in aug_context and self.molecular_noise_std > 0:
            noise = torch.randn_like(aug_context['molecular']) * self.molecular_noise_std
            aug_context['molecular'] = aug_context['molecular'] + noise
        
        for key in ['os_months', 'pfs_months', 'dss_months', 'dfs_months']:
            if key in aug_context and self.survival_months_noise_std > 0:
                noise = torch.randn_like(aug_context[key]) * self.survival_months_noise_std
                aug_context[key] = torch.clamp(aug_context[key] + noise, 0, 1)
        
        if self.dropout_prob > 0:
            batch_size = aug_context['cancer_type'].shape[0]
            mask = torch.rand(batch_size) < self.dropout_prob
            
            if mask.any():
                device = aug_context['cancer_type'].device
                aug_context['cancer_type'][mask] = 0
                aug_context['stage'][mask] = 0
                aug_context['sex'][mask] = 2
                aug_context['age'][mask] = 0.5
                aug_context['molecular'][mask] = 0
                for key in ['os_status', 'pfs_status', 'dss_status', 'dfs_status']:
                    aug_context[key][mask] = 2
                for key in ['os_months', 'pfs_months', 'dss_months', 'dfs_months']:
                    aug_context[key][mask] = 0.5
        
        return aug_context
