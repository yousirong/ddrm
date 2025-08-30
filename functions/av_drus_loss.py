"""
AV-DRUS Loss Functions
Implements the full Variational Lower Bound (VLB) for joint mean and variance learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def kl_divergence_gaussians(mu1, logvar1, mu2, logvar2):
    """
    Compute KL divergence between two multivariate Gaussian distributions
    KL(N(mu1, Σ1) || N(mu2, Σ2)) for diagonal covariances
    
    Args:
        mu1: mean of first distribution [B, C, H, W]
        logvar1: log variance of first distribution [B, C, H, W]
        mu2: mean of second distribution [B, C, H, W]  
        logvar2: log variance of second distribution [B, C, H, W]
        
    Returns:
        kl: KL divergence [B]
    """
    # Convert log variance to variance
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    # KL divergence formula for multivariate Gaussians with diagonal covariance
    # KL = 0.5 * (tr(Σ2^-1 * Σ1) + (μ2-μ1)^T * Σ2^-1 * (μ2-μ1) - k + log(det(Σ2)/det(Σ1)))
    # For diagonal case: KL = 0.5 * sum_i (σ1_i^2/σ2_i^2 + (μ2_i-μ1_i)^2/σ2_i^2 - 1 + log(σ2_i^2/σ1_i^2))
    
    kl = 0.5 * (
        var1 / var2 +  # σ1^2/σ2^2 term
        (mu2 - mu1).pow(2) / var2 +  # (μ2-μ1)^2/σ2^2 term
        logvar2 - logvar1 - 1  # log(σ2^2/σ1^2) - 1 term
    )
    
    # Sum over spatial dimensions and channels, return batch dimension
    kl = kl.view(kl.shape[0], -1).sum(dim=1)
    return kl


class AVDRUSLoss(nn.Module):
    """
    AV-DRUS Loss function implementing full VLB for joint mean and variance learning
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.vlb_weight = getattr(config.loss, 'vlb_weight', 1.0)
        self.simple_weight = getattr(config.loss, 'simple_weight', 0.1)  # Optional simple MSE loss
        self.variance_reg_weight = getattr(config.loss, 'variance_reg_weight', 0.01)  # Variance regularization
        
        # Diffusion schedule parameters
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        self.beta_start = config.diffusion.beta_start
        self.beta_end = config.diffusion.beta_end
        
        # Precompute diffusion schedule
        betas = self._get_beta_schedule(
            config.diffusion.beta_schedule,
            self.beta_start,
            self.beta_end,
            self.num_timesteps
        )
        
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Posterior variance (for ground truth q(x_{t-1}|x_t,x_0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(posterior_variance.clamp(min=1e-20)))
        
        # Posterior mean coefficients
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
    
    def _get_beta_schedule(self, beta_schedule, beta_start, beta_end, num_timesteps):
        """Get beta schedule for diffusion process"""
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        elif beta_schedule == "quad":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float64) ** 2
        else:
            raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")
        
        return betas.float()
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute ground truth posterior q(x_{t-1}|x_t,x_0)
        
        Args:
            x_start: original image x_0 [B, C, H, W]
            x_t: noisy image at timestep t [B, C, H, W]
            t: timestep [B]
            
        Returns:
            mean: posterior mean [B, C, H, W]
            log_variance: posterior log variance [B, C, H, W]
        """
        # Extract coefficients for timestep t
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x_start.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x_start.shape)
        posterior_log_variance_t = extract(self.posterior_log_variance, t, x_start.shape)
        
        # Compute posterior mean
        mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        return mean, posterior_log_variance_t
    
    def p_mean_variance(self, model_output, x_t, t, x_start=None):
        """
        Convert model output to mean and variance for p_θ(x_{t-1}|x_t)
        
        Args:
            model_output: output from model containing 'mean' and 'log_variance'
            x_t: noisy image at timestep t [B, C, H, W]
            t: timestep [B]
            x_start: if provided, used for mean computation
            
        Returns:
            mean: predicted mean [B, C, H, W]
            log_variance: predicted log variance [B, C, H, W]
        """
        # Extract diffusion parameters
        sqrt_alpha_t = extract(torch.sqrt(1.0 - self.betas), t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x_t.shape)
        
        # Convert noise prediction to x_0 prediction
        pred_x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * model_output['mean']) / sqrt_alpha_t
        
        # Clip x_0 prediction
        pred_x_start = torch.clamp(pred_x_start, -1, 1)
        
        # Compute mean using predicted x_0
        model_mean, _ = self.q_posterior_mean_variance(pred_x_start, x_t, t)
        
        # Use predicted log variance
        model_log_variance = model_output['log_variance']
        
        return model_mean, model_log_variance, pred_x_start
    
    def forward(self, model_output, x_start, x_t, t, condition=None):
        """
        Compute AV-DRUS loss
        
        Args:
            model_output: dict with 'mean' and 'log_variance' from model
            x_start: original clean image [B, C, H, W]
            x_t: noisy image at timestep t [B, C, H, W]
            t: timestep [B]
            condition: conditioning signal if any
            
        Returns:
            loss_dict: dictionary containing various loss components
        """
        B = x_start.shape[0]
        
        # Convert model output to p_θ(x_{t-1}|x_t) parameters
        model_mean, model_log_variance, pred_x_start = self.p_mean_variance(model_output, x_t, t, x_start)
        
        # Ground truth posterior q(x_{t-1}|x_t,x_0) parameters
        true_mean, true_log_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        # VLB Loss: KL divergence between q(x_{t-1}|x_t,x_0) and p_θ(x_{t-1}|x_t)
        vlb_loss = kl_divergence_gaussians(
            model_mean, model_log_variance,
            true_mean, true_log_variance
        )
        vlb_loss = vlb_loss.mean()
        
        # Simple MSE loss for noise prediction (optional, for stability)
        # Ground truth noise
        sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x_start.shape)
        noise = (x_t - extract(torch.sqrt(self.alphas_cumprod), t, x_start.shape) * x_start) / sqrt_one_minus_alphas_cumprod_t
        
        simple_loss = F.mse_loss(model_output['mean'], noise)
        
        # Variance regularization (prevent extreme values)
        variance = torch.exp(model_output['log_variance'])
        variance_reg_loss = torch.mean((variance - 1.0).pow(2))  # Encourage variance around 1
        
        # Total loss
        total_loss = (
            self.vlb_weight * vlb_loss +
            self.simple_weight * simple_loss +
            self.variance_reg_weight * variance_reg_loss
        )
        
        return {
            'total_loss': total_loss,
            'vlb_loss': vlb_loss,
            'simple_loss': simple_loss,
            'variance_reg_loss': variance_reg_loss,
            'pred_x_start': pred_x_start
        }


def extract(a, t, x_shape):
    """
    Extract values from tensor a at indices t and reshape to match x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu().long()).float()
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)