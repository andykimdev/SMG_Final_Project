"""DDPM utilities: noise schedules, forward/reverse processes, loss computation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math
import config


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, max_beta)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from Ho et al. 2020."""
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Quadratic schedule."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def get_beta_schedule(schedule_type: str = 'cosine', timesteps: int = 1000, **kwargs) -> torch.Tensor:
    """Get beta schedule by name."""
    if schedule_type == 'cosine':
        return cosine_beta_schedule(timesteps, s=kwargs.get('s', 0.008))
    elif schedule_type == 'linear':
        return linear_beta_schedule(timesteps, kwargs.get('beta_start', 1e-4), kwargs.get('beta_end', 0.02))
    elif schedule_type == 'quadratic':
        return quadratic_beta_schedule(timesteps, kwargs.get('beta_start', 1e-4), kwargs.get('beta_end', 0.02))
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


class GaussianDiffusion:
    """Forward and reverse diffusion processes."""
    
    def __init__(self, timesteps: int = 1000, schedule: str = 'linear', loss_type: str = 'mae', **kwargs):
        self.timesteps = timesteps
        self.loss_type = loss_type
        
        betas = get_beta_schedule(schedule, timesteps, **kwargs)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward process: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def p_mean_variance(self, model, x_t: torch.Tensor, t: torch.Tensor, context_dict: Dict, clip_denoised: bool = True):
        """Compute mean/variance for reverse step."""
        predicted_noise = model(x_t, t, context_dict)
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, context_dict: Dict, clip_denoised: bool = True) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, context_dict, clip_denoised)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape: Tuple, context_dict: Dict, device: str = 'cpu', progress: bool = True) -> torch.Tensor:
        """Full reverse sampling loop."""
        batch_size = shape[0]
        x_t = torch.randn(shape, device=device)
        
        timesteps = list(range(self.timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc='Sampling')
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, context_dict)
        
        return x_t
    
    def training_loss(self, model, x_start: torch.Tensor, context_dict: Dict, t: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute DDPM training loss."""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_t, t, context_dict)
        
        if self.loss_type == 'mse':
            return F.mse_loss(predicted_noise, noise)
        elif self.loss_type == 'mae':
            return F.l1_loss(predicted_noise, noise)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss: {self.loss_type}")
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract coefficients at timesteps and reshape for broadcasting."""
        batch_size = t.shape[0]
        if a.device != t.device:
            a = a.to(t.device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]
        self.original = {}
    
    def state_dict(self) -> Dict:
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict: Dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']
