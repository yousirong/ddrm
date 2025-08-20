import os
import logging
import time
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import torchvision.utils as tvu
from pathlib import Path
from PIL import Image
from diffusers import UNet2DModel

# Import base DDRM components
from runners.diffusion import Diffusion, get_beta_schedule
from models.diffusion import Model
from functions.denoising import efficient_generalized_steps
from functions.ckpt_util import get_ckpt_path, download

# Import ultrasound-specific components
from ultrasound_h_funcs import (
    create_ultrasound_h_funcs, 
    estimate_version_artifacts,
    estimate_degradation_operator
)

# Import guided diffusion components
from guided_diffusion.script_util import create_model, args_to_dict

logger = logging.getLogger(__name__)

class UltrasoundDDRMRunner(Diffusion):
    """
    Enhanced DDRM Runner for ultrasound blind zone removal
    
    Integrates with base DDRM framework while implementing ultrasound-specific methodology:
    - z_est = Average(CY_ON - CN_ON): structural noise estimation
    - H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²: distortion operator estimation  
    - Physics-based modeling: blind zone as physical distortion, not simple masking
    - Version-specific processing (V3-V7) with different blind zone characteristics
    """
    
    def __init__(self, args, config, device=None):
        # Initialize base DDRM components
        super().__init__(args, config, device)
        
        self.args = args
        self.config = config
        
        # Version-specific noise and degradation operators
        self.version_artifacts = {}
        self.version_degradation_ops = {}
        
        logger.info("Initialized UltrasoundDDRMRunner with base DDRM framework")
        
    def estimate_ultrasound_artifacts(self, cn_on_path, cy_on_path, cn_oy_path=None, cy_oy_path=None):
        """
        Estimate ultrasound-specific artifacts for all versions (V3-V7)
        
        Steps:
        1. z_est = Average(CY_ON - CN_ON) for each version
        2. H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||² for each version
        """
        logger.info("=== Estimating Ultrasound Artifacts ===")
        
        versions = ['V3', 'V4', 'V5', 'V6', 'V7']
        
        for version in versions:
            logger.info(f"Processing {version}...")
            
            # Step 1: Structural noise estimation z_est
            z_est, distortion_map = estimate_version_artifacts(cn_on_path, cy_on_path, version)
            
            if z_est is not None:
                self.version_artifacts[version] = {
                    'noise_pattern': z_est,
                    'distortion_map': distortion_map
                }
                logger.info(f"{version} structural noise z_est estimated")
                
                # Step 2: Degradation operator estimation H_est (if OY data available)
                if cn_oy_path and cy_oy_path:
                    H_est = estimate_degradation_operator(cn_oy_path, cy_oy_path, z_est, version)
                    if H_est is not None:
                        self.version_degradation_ops[version] = H_est
                        logger.info(f"{version} degradation operator H_est estimated")
            else:
                logger.warning(f"Failed to estimate artifacts for {version}")
        
        logger.info(f"Artifact estimation completed for {len(self.version_artifacts)} versions")
        
    def detect_version_from_path(self, image_path):
        """Detect version (V3-V7) from image path"""
        image_path = str(image_path)
        for version in ['V3', 'V4', 'V5', 'V6', 'V7']:
            if version in image_path:
                return version
        return None
    
    def create_h_functions(self, version=None):
        """Create version-specific H_functions for ultrasound"""
        noise_pattern = None
        
        if version and version in self.version_artifacts:
            noise_pattern = self.version_artifacts[version]['noise_pattern']
            logger.info(f"Using version-specific artifacts for {version}")
        else:
            # Use combined artifacts
            if self.version_artifacts:
                all_noise = [artifacts['noise_pattern'] for artifacts in self.version_artifacts.values()]
                noise_pattern = np.mean(all_noise, axis=0)
                logger.info("Using combined artifacts from all versions")
        
        # Get distortion and noise factors from args (with defaults)
        distortion_factor = getattr(self.args, 'distortion_factor', 0.05)
        noise_factor = getattr(self.args, 'noise_factor', 0.02)
        
        return create_ultrasound_h_funcs(self.config, version, noise_pattern, distortion_factor, noise_factor)
    
    def sample_ultrasound_sequence(self, test_images_path, output_dir, sigma_0=0.05):
        """
        Enhanced sampling sequence for ultrasound images
        Uses version-specific processing and physics-based modeling
        """
        logger.info("Starting ultrasound DDRM sampling...")
        
        # Load model (use base DDRM model loading)
        model = self._load_ddrm_model()
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test images
        test_images = self._load_test_images(test_images_path)
        logger.info(f"Loaded {len(test_images)} test images")
        
        # Process each image with version-specific handling
        results = []
        
        for i, (image_path, image) in enumerate(test_images):
            logger.info(f"Processing image {i+1}/{len(test_images)}: {image_path.name}")
            
            # Detect version from filename
            version = self.detect_version_from_path(image_path)
            logger.info(f"Detected version: {version}")
            
            # Create version-specific H_functions
            H_funcs = self.create_h_functions(version)
            
            # Prepare image tensor with proper normalization
            img_array = np.array(image) / 255.0  # Normalize to [0, 1]
            x_orig = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Log input image statistics
            logger.info(f"Input image range: Min: {x_orig.min():.3f}, Max: {x_orig.max():.3f}, Mean: {x_orig.mean():.3f}")
            
            # Apply degradation (simulate measurement)
            y_0 = H_funcs.H(x_orig)
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
            
            # Compute pseudo-inverse for initialization  
            pinv_y_0 = H_funcs.H_pinv(y_0)
            
            # Save degraded image
            degraded_path = output_dir / f"{image_path.stem}_degraded.png"
            tvu.save_image(y_0.squeeze(), degraded_path)
            
            # Save original
            orig_path = output_dir / f"{image_path.stem}_original.png"
            tvu.save_image(x_orig.squeeze(), orig_path)
            
            # DDRM restoration using enhanced sampling
            x_T = torch.randn_like(x_orig)
            
            # Use base DDRM sampling with ultrasound H_functions
            with torch.no_grad():
                restored_sequence = self.sample_image_ultrasound(
                    x_T, model, H_funcs, y_0, sigma_0, version=version
                )
            
            # Save restored image (last in sequence)
            if restored_sequence is not None:
                if isinstance(restored_sequence, torch.Tensor):
                    restored = restored_sequence
                else:
                    restored = restored_sequence[-1] if len(restored_sequence) > 0 else restored_sequence
                    
                restored_path = output_dir / f"{image_path.stem}_restored_{version}.png"
                
                # Handle DDPM output normalization with robust scaling
                restored_cpu = restored.squeeze().cpu()
                
                logger.info(f"Raw DDPM output range: Min: {restored_cpu.min():.3f}, Max: {restored_cpu.max():.3f}, Mean: {restored_cpu.mean():.3f}")
                
                # Robust normalization for extreme values
                if torch.abs(restored_cpu).max() > 5.0:  # Extreme values detected
                    logger.warning("Extreme values detected - applying robust normalization")
                    # Use percentile-based normalization
                    p1, p99 = torch.quantile(restored_cpu, torch.tensor([0.01, 0.99]))
                    restored_tensor = (restored_cpu - p1) / (p99 - p1 + 1e-8)
                    restored_tensor = restored_tensor.clamp(0, 1)
                elif restored_cpu.min() < -0.5 or restored_cpu.max() > 1.5:
                    # Standard [-1, 1] to [0, 1] conversion
                    restored_tensor = (restored_cpu + 1.0) / 2.0
                    restored_tensor = restored_tensor.clamp(0, 1)
                    logger.info("Applied standard [-1,1] to [0,1] conversion")
                else:
                    # Already in [0, 1] range
                    restored_tensor = restored_cpu.clamp(0, 1)
                
                # Final adjustment - make sure image isn't too bright
                if restored_tensor.mean() > 0.8:
                    logger.warning(f"Image too bright (mean={restored_tensor.mean():.3f}), applying brightness correction")
                    # Adjust brightness to match input range
                    target_mean = min(0.4, x_orig.mean().cpu().item() * 1.5)  # Target slightly brighter than input
                    current_mean = restored_tensor.mean()
                    restored_tensor = restored_tensor * (target_mean / current_mean)
                    restored_tensor = restored_tensor.clamp(0, 1)
                
                logger.info(f"Final image range: Min: {restored_tensor.min():.3f}, Max: {restored_tensor.max():.3f}, Mean: {restored_tensor.mean():.3f}")
                
                tvu.save_image(restored_tensor, restored_path)
                
                results.append({
                    'original_path': image_path,
                    'version': version,
                    'restored_path': restored_path,
                    'degraded_path': degraded_path
                })
                
                logger.info(f"Saved restored image: {restored_path}")
        
        logger.info(f"Ultrasound DDRM sampling completed. Results saved to {output_dir}")
        return results
    
    def sample_image_ultrasound(self, x, model, H_funcs, y_0, sigma_0, last=True, version=None):
        """
        Enhanced DDRM sampling for ultrasound with physics-based corrections
        Uses base efficient_generalized_steps with ultrasound-specific H_functions
        """
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        logger.info(f"Starting DDRM sampling with {len(seq)} timesteps for {version}")
        
        # Use base DDRM sampling with ultrasound H_functions
        x_sequence = efficient_generalized_steps(
            x, seq, model, self.betas, H_funcs, y_0, sigma_0,
            etaB=self.args.etaB, 
            etaA=self.args.eta, 
            etaC=self.args.eta,
            cls_fn=None, 
            classes=None
        )
        
        if last:
            return x_sequence[0][-1]
        return x_sequence[0]
    
    def _load_ddrm_model(self):
        """Load DDRM model from user-specified diffusers path"""
        model_path = "/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100/best_model/unet"
        model = UNet2DModel.from_pretrained(model_path)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        logger.info(f"Loaded UNet2DModel from: {model_path}")
        return model
    
    def _load_test_images(self, test_path):
        """Load test images with version detection"""
        test_path = Path(test_path)
        images = []
        
        if test_path.is_file():
            # Single image
            img = Image.open(test_path).convert('L').resize((512, 512))
            images.append((test_path, img))
        else:
            # Directory
            for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg']:
                for img_path in test_path.glob(ext):
                    img = Image.open(img_path).convert('L').resize((512, 512))
                    images.append((img_path, img))
        
        return sorted(images, key=lambda x: x[0].name)
    
    def save_artifacts(self, output_dir):
        """Save estimated artifacts for analysis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save version-specific noise patterns
        for version, artifacts in self.version_artifacts.items():
            noise_path = output_dir / f"z_est_{version}.npy"
            np.save(noise_path, artifacts['noise_pattern'])
            
            if 'distortion_map' in artifacts:
                dist_path = output_dir / f"distortion_map_{version}.npy"
                np.save(dist_path, artifacts['distortion_map'])
        
        # Save degradation operators
        for version, H_est in self.version_degradation_ops.items():
            h_path = output_dir / f"H_est_{version}.npy"
            np.save(h_path, H_est)
        
        logger.info(f"Artifacts saved to {output_dir}")


# Utility functions for standalone usage
def create_ultrasound_runner(args, config, device=None):
    """Factory function to create UltrasoundDDRMRunner"""
    return UltrasoundDDRMRunner(args, config, device)

def run_ultrasound_ddrm(args, config):
    """Main function to run ultrasound DDRM restoration"""
    logger.info("Starting Ultrasound DDRM Restoration")
    
    # Create runner
    runner = create_ultrasound_runner(args, config)
    
    # Step 1: Estimate artifacts if training data available
    if hasattr(args, 'cn_on_path') and hasattr(args, 'cy_on_path'):
        cn_oy_path = getattr(args, 'cn_oy_path', None)
        cy_oy_path = getattr(args, 'cy_oy_path', None)
        
        runner.estimate_ultrasound_artifacts(
            args.cn_on_path, args.cy_on_path, cn_oy_path, cy_oy_path
        )
        
        # Save artifacts
        if hasattr(args, 'artifact_save_dir') and args.artifact_save_dir:
            runner.save_artifacts(args.artifact_save_dir)
        else:
            # Default artifacts save location
            default_artifact_dir = os.path.join(args.image_folder, 'artifacts')
            runner.save_artifacts(default_artifact_dir)
    
    # Step 2: Process test images
    if hasattr(args, 'test_images_path'):
        results = runner.sample_ultrasound_sequence(
            args.test_images_path, 
            args.image_folder,
            sigma_0=getattr(args, 'sigma_0', 0.05)
        )
        
        logger.info(f"Processed {len(results)} images successfully")
        return results
    else:
        logger.warning("No test images path provided")
        return []

if __name__ == "__main__":
    # This would be called from main script
    pass