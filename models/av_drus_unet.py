"""
AV-DRUS: Adaptive Variance Diffusion Restoration for Ultrasound
UNet with Dual-Head Architecture for Mean and Variance Prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import get_timestep_embedding, nonlinearity, Normalize, Upsample, Downsample, ResnetBlock, AttnBlock


class AdaptiveVarianceUNet(nn.Module):
    """
    U-Net with dual-head architecture for AV-DRUS
    Predicts both mean (noise) and spatially-varying variance
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        
        # Conditioning signal channels (DAS image)
        self.condition_channels = getattr(config.model, 'condition_channels', 3)
        
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        
        # Shared normalization
        self.norm_out = Normalize(block_in)
        
        # Dual-head outputs
        # Mean head (standard noise prediction)
        self.mean_head = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        
        # Variance head with conditioning
        self.condition_proj = torch.nn.Conv2d(self.condition_channels, block_in // 4, kernel_size=3, stride=1, padding=1)
        self.variance_head = nn.Sequential(
            torch.nn.Conv2d(block_in + block_in // 4, block_in // 2, kernel_size=3, stride=1, padding=1),
            Normalize(block_in // 2),
            nn.SiLU(),
            torch.nn.Conv2d(block_in // 2, out_ch, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x, t, condition=None):
        """
        Forward pass with dual-head output
        
        Args:
            x: noisy input [B, C, H, W]
            t: timesteps [B]
            condition: conditioning signal (DAS image) [B, C, H, W]
            
        Returns:
            dict with 'mean' and 'log_variance' keys
        """
        assert x.shape[2] == x.shape[3] == self.resolution
        
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # Shared normalization
        h = self.norm_out(h)
        h = nonlinearity(h)
        
        # Mean head output (standard noise prediction)
        mean_pred = self.mean_head(h)
        
        # Variance head output with conditioning
        if condition is not None:
            # Project conditioning signal
            cond_feat = self.condition_proj(condition)
            # Concatenate with main features
            variance_input = torch.cat([h, cond_feat], dim=1)
        else:
            # If no conditioning, use zero padding
            zero_cond = torch.zeros(h.shape[0], h.shape[1] // 4, h.shape[2], h.shape[3], 
                                  device=h.device, dtype=h.dtype)
            variance_input = torch.cat([h, zero_cond], dim=1)
        
        # Predict log variance (to ensure positive variance)
        log_variance = self.variance_head(variance_input)
        
        return {
            'mean': mean_pred,
            'log_variance': log_variance
        }


class UltrasoundDASOperator(nn.Module):
    """
    Simple DAS (Delay-and-Sum) operator for ultrasound beamforming
    This is a placeholder implementation - replace with actual beamforming logic
    """
    
    def __init__(self, num_elements=128, num_samples=512):
        super().__init__()
        self.num_elements = num_elements
        self.num_samples = num_samples
        
        # Simple averaging weights for DAS approximation
        self.register_buffer('das_weights', torch.ones(1, 1, 1, 1) / num_elements)
    
    def forward(self, rf_data):
        """
        Apply DAS beamforming to RF data
        
        Args:
            rf_data: Raw RF channel data [B, num_elements, num_samples]
            
        Returns:
            das_image: DAS beamformed image [B, 1, H, W]
        """
        # Simple implementation - just average across channels
        # In real implementation, this would include proper delay calculations
        if len(rf_data.shape) == 3:
            das_image = rf_data.mean(dim=1, keepdim=True)  # [B, 1, num_samples]
            # Reshape to 2D image format
            B, C, L = das_image.shape
            H = W = int(math.sqrt(L))
            if H * W != L:
                # Pad or interpolate to square
                das_image = F.interpolate(das_image.unsqueeze(-1), size=(H, W), mode='bilinear', align_corners=False)
            else:
                das_image = das_image.view(B, C, H, W)
        else:
            # Assume already in image format
            das_image = rf_data
        
        return das_image


def create_av_drus_model(config):
    """Factory function to create AV-DRUS model"""
    return AdaptiveVarianceUNet(config)