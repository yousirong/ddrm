"""
AV-DRUS Diffusion Runner
Implements the AV-DRUS framework with adaptive variance scheduling
"""

import os
import logging
import time
import glob
import math

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F

from models.av_drus_unet import create_av_drus_model, UltrasoundDASOperator
from functions.av_drus_loss import AVDRUSLoss, extract
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download

import torchvision.utils as tvu
from PIL import Image
import random


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule for diffusion process"""
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class AVDRUSDiffusion(object):
    """AV-DRUS Diffusion framework for ultrasound reconstruction"""
    
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # Initialize diffusion parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        
        # Initialize DAS operator for conditioning
        self.das_operator = UltrasoundDASOperator().to(device)
        
        # Initialize loss function
        self.loss_fn = AVDRUSLoss(config).to(device)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(torch.sqrt(self.alphas_cumprod), t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample(self):
        """Main sampling/reconstruction function"""
        # Create model
        model = create_av_drus_model(self.config)
        
        # Load pretrained weights if available
        if hasattr(self.config.model, 'pretrained_path') and self.config.model.pretrained_path:
            checkpoint = torch.load(self.config.model.pretrained_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=False)
            logging.info(f"Loaded pretrained model from {self.config.model.pretrained_path}")
        
        model.to(self.device)
        model.eval()
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        self.sample_sequence(model)

    def sample_sequence(self, model):
        """Sample sequence for reconstruction"""
        args, config = self.args, self.config

        # Get dataset
        if hasattr(args, 'use_busi') and args.use_busi:
            # Use BUSI dataset
            dataset, test_dataset = self.get_busi_dataset()
        else:
            # Use original dataset
            dataset, test_dataset = get_dataset(args, config)
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Setup degradation for ultrasound (simplified)
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            # Simulate ultrasound RF data degradation
            # For now, add noise to simulate low-quality measurement
            sigma_0 = args.sigma_0 * 2  # Account for [-1,1] scaling
            y_0 = x_orig + sigma_0 * torch.randn_like(x_orig)
            
            # Generate DAS conditioning signal
            das_condition = self.generate_das_condition(y_0)

            # Save original and degraded images
            for i in range(len(y_0)):
                tvu.save_image(
                    inverse_data_transform(config, y_0[i]), 
                    os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]), 
                    os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, das_condition[i]), 
                    os.path.join(self.args.image_folder, f"das_{idx_so_far + i}.png")
                )

            # AV-DRUS reconstruction
            with torch.no_grad():
                x_recon = self.av_drus_sample(model, y_0, das_condition, sigma_0)

            x_recon_display = [inverse_data_transform(config, y) for y in x_recon]

            # Save reconstructed images
            for i in range(len(x_recon_display)):
                for j in range(x_recon_display[i].size(0)):
                    tvu.save_image(
                        x_recon_display[i][j], 
                        os.path.join(self.args.image_folder, f"{idx_so_far + j}_recon_{i}.png")
                    )
                    
                    if i == len(x_recon_display) - 1:  # Final reconstruction
                        orig = inverse_data_transform(config, x_orig[j])
                        mse = torch.mean((x_recon_display[i][j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr += psnr

            idx_so_far += y_0.shape[0]
            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def generate_das_condition(self, y_0):
        """
        Generate DAS conditioning signal from degraded measurement
        """
        # Simple approach: apply some processing to create DAS-like image
        das_condition = y_0.clone()
        
        # Apply some smoothing to simulate DAS reconstruction
        das_condition = F.avg_pool2d(das_condition, kernel_size=3, stride=1, padding=1)
        
        # Add some noise to make it more realistic
        das_condition = das_condition + 0.1 * torch.randn_like(das_condition)
        
        return das_condition

    def av_drus_sample(self, model, y_0, das_condition, sigma_0, num_steps=None):
        """
        AV-DRUS sampling with adaptive variance
        """
        if num_steps is None:
            num_steps = self.args.timesteps

        # Initialize with noise
        x = torch.randn_like(y_0)
        
        # Sampling timesteps
        skip = self.num_timesteps // num_steps
        seq = list(range(0, self.num_timesteps, skip))
        
        # Reverse process
        x_sequence = [x]
        for i, t in enumerate(reversed(seq)):
            t_batch = torch.tensor([t] * x.shape[0], device=self.device)
            
            # Model prediction with conditioning
            with torch.no_grad():
                model_output = model(x, t_batch, condition=das_condition)
            
            # Extract mean and variance predictions
            predicted_noise = model_output['mean']
            log_variance = model_output['log_variance']
            variance = torch.exp(log_variance)
            
            # Compute x_{t-1} using predicted mean and adaptive variance
            alpha_t = extract(1.0 - self.betas, t_batch, x.shape)
            sqrt_one_minus_alpha_t = extract(torch.sqrt(1.0 - self.alphas_cumprod), t_batch, x.shape)
            
            # Predicted x_0
            pred_x0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            if t > 0:
                # Compute previous step mean
                alpha_prev = extract(self.alphas_cumprod_prev, t_batch, x.shape)
                beta_t = extract(self.betas, t_batch, x.shape)
                
                # Mean of p_θ(x_{t-1}|x_t)
                pred_mean = (
                    beta_t * torch.sqrt(alpha_prev) / (1.0 - extract(self.alphas_cumprod, t_batch, x.shape)) * pred_x0 +
                    (1.0 - alpha_prev) * torch.sqrt(alpha_t) / (1.0 - extract(self.alphas_cumprod, t_batch, x.shape)) * x
                )
                
                # Sample with adaptive variance
                noise = torch.randn_like(x)
                x = pred_mean + torch.sqrt(variance) * noise
                
                # Data consistency step (simplified)
                data_consistency_weight = 0.1
                x = x + data_consistency_weight * (y_0 - x)
            else:
                x = pred_x0
            
            x_sequence.append(x.clone())
        
        return x_sequence

    def get_busi_dataset(self):
        """Load BUSI dataset for ultrasound experiments"""
        try:
            from datasets.busi_dataset import BUSIDataset
            
            # Create BUSI dataset
            train_dataset = BUSIDataset(
                root=self.args.busi_path,
                split='train',
                transform=lambda x: data_transform(self.config, x),
                image_size=self.config.data.image_size
            )
            
            test_dataset = BUSIDataset(
                root=self.args.busi_path,
                split='test', 
                transform=lambda x: data_transform(self.config, x),
                image_size=self.config.data.image_size
            )
            
            return train_dataset, test_dataset
            
        except ImportError:
            logging.warning("BUSI dataset not available, using default dataset")
            return get_dataset(self.args, self.config)


# For compatibility, alias to original Diffusion class name
Diffusion = AVDRUSDiffusion