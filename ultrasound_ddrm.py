import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import logging
import gc
from torchvision import transforms

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import diffusers components for 512x512 model
try:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("Diffusers not available, fallback to original DDRM components")

# Import DDRM components as fallback
try:
    from runners.diffusion import Diffusion
    from models.diffusion import DiffusionUNet
    from functions.denoising import efficient_generalized_steps
except ImportError:
    logger.warning("Original DDRM components not available")

class UltrasoundDDRM:
    def __init__(self, config_path, model_path, device='cuda'):
        self.device = device
        self.config = self.load_config(config_path)
        self.model = self.load_model(model_path)
        self.noise_pattern = None
        self.degradation_operator = None
        
    def load_config(self, config_path):
        """Load DDRM configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_model(self, model_path):
        """Load pretrained DDPM model (512x512 diffusers model)"""
        try:
            # Try loading diffusers model first (512x512 trained model)
            if DIFFUSERS_AVAILABLE and Path(model_path).is_dir():
                logger.info(f"Loading diffusers model from {model_path}")
                pipeline = DDPMPipeline.from_pretrained(model_path)
                model = pipeline.unet.to(self.device)
                self.scheduler = pipeline.scheduler
                model.eval()
                logger.info("Successfully loaded 512x512 diffusers model")
                return model
            else:
                # Fallback to original model format
                logger.info(f"Loading original model format from {model_path}")
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                self.scheduler = None
                return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def estimate_blind_zone_artifacts(self, cn_on_path, cy_on_path, save_path=None):
        """
        Estimate blind zone noise pattern z and degradation operator H
        z_est = Average(CY_ON - CN_ON)
        Handles different blind zone sizes for V3-V7 versions with donut-shaped masks
        """
        logger.info("Estimating blind zone artifacts with version-specific processing...")
        
        # Define version-specific donut masks (outer_radius, inner_radius)
        VERSION_RADIUS = {
            "V3": (220, 85),
            "V4": (130, 50),
            "V5": (90, 30),
            "V6": (60, 20),
            "V7": (45, 15)
        }
        
        cn_on_images = self._load_images_by_version(cn_on_path, "CN_ON")
        cy_on_images = self._load_images_by_version(cy_on_path, "CY_ON")
        
        # Process each version separately (V3-V7 have different blind zone sizes)
        version_noise_patterns = {}
        
        for version in ['V3', 'V4', 'V5', 'V6', 'V7']:
            if version in cn_on_images and version in cy_on_images:
                logger.info(f"Processing version {version}...")
                
                cn_imgs = cn_on_images[version]
                cy_imgs = cy_on_images[version]
                
                if len(cn_imgs) != len(cy_imgs):
                    logger.warning(f"Mismatch in {version} image counts: CN_ON={len(cn_imgs)}, CY_ON={len(cy_imgs)}")
                
                # Create donut-shaped mask for this version
                outer_radius, inner_radius = VERSION_RADIUS[version]
                donut_mask = self._create_donut_mask(512, 512, outer_radius, inner_radius)
                
                noise_patterns = []
                
                for cn_img, cy_img in zip(cn_imgs, cy_imgs):
                    # Ensure images are 512x512
                    cn_img = cn_img.resize((512, 512), Image.LANCZOS)
                    cy_img = cy_img.resize((512, 512), Image.LANCZOS)
                    
                    # Convert to numpy arrays
                    cn_array = np.array(cn_img).astype(np.float32) / 255.0
                    cy_array = np.array(cy_img).astype(np.float32) / 255.0
                    
                    # Compute noise pattern
                    noise = cy_array - cn_array
                    
                    # Apply donut mask - only noise in the donut region is valid
                    masked_noise = noise * donut_mask
                    noise_patterns.append(masked_noise)
                
                # Average noise pattern for this version (only in donut region)
                version_noise_patterns[version] = np.mean(noise_patterns, axis=0)
                
                logger.info(f"{version} noise statistics: mean={np.mean(version_noise_patterns[version]):.4f}, "
                           f"std={np.std(version_noise_patterns[version]):.4f}")
        
        # Store version-specific noise patterns
        self.version_noise_patterns = version_noise_patterns
        
        # Create combined noise pattern (weighted average)
        if version_noise_patterns:
            self.noise_pattern = np.mean(list(version_noise_patterns.values()), axis=0)
        else:
            logger.warning("No version-specific patterns found, using fallback method")
            # Fallback to original method
            cn_on_images = self._load_images(cn_on_path, pattern="CN_ON_*.bmp")
            cy_on_images = self._load_images(cy_on_path, pattern="CY_ON_*.bmp")
            
            noise_patterns = []
            for cn_img, cy_img in zip(cn_on_images, cy_on_images):
                cn_img = cn_img.resize((512, 512), Image.LANCZOS)
                cy_img = cy_img.resize((512, 512), Image.LANCZOS)
                cn_array = np.array(cn_img).astype(np.float32) / 255.0
                cy_array = np.array(cy_img).astype(np.float32) / 255.0
                noise = cy_array - cn_array
                noise_patterns.append(noise)
            
            self.noise_pattern = np.mean(noise_patterns, axis=0)
        
        logger.info(f"Combined noise pattern shape: {self.noise_pattern.shape}")
        logger.info(f"Combined noise statistics: mean={np.mean(self.noise_pattern):.4f}, "
                   f"std={np.std(self.noise_pattern):.4f}")
        
        if save_path:
            # Save combined pattern
            np.save(save_path, self.noise_pattern)
            
            # Save version-specific patterns
            save_dir = Path(save_path).parent
            for version, pattern in version_noise_patterns.items():
                version_path = save_dir / f"noise_pattern_{version}.npy"
                np.save(version_path, pattern)
                logger.info(f"Saved {version} noise pattern to {version_path}")
        
        return self.noise_pattern
    
    def _create_donut_mask(self, height, width, outer_radius, inner_radius):
        """
        Create a donut-shaped mask for blind zone detection
        
        Args:
            height, width: Image dimensions
            outer_radius: Outer radius of the donut (blind zone boundary)
            inner_radius: Inner radius of the donut (center area with no noise)
        
        Returns:
            Binary mask where 1 indicates blind zone area
        """
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create donut mask: 1 in the ring between inner and outer radius, 0 elsewhere
        donut_mask = ((distance >= inner_radius) & (distance <= outer_radius)).astype(np.float32)
        
        return donut_mask
    
    def estimate_degradation_operator(self, cn_oy_path, cy_oy_path):
        """
        Estimate degradation operator H using:
        H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²
        Handles different blind zone sizes for V3-V7 versions with donut masks
        """
        logger.info("Estimating degradation operator with version-specific processing...")
        
        if self.noise_pattern is None:
            raise ValueError("Noise pattern not estimated. Run estimate_blind_zone_artifacts first.")
        
        # Define version-specific donut masks (same as in noise estimation)
        VERSION_RADIUS = {
            "V3": (220, 85),
            "V4": (130, 50), 
            "V5": (90, 30),
            "V6": (60, 20),
            "V7": (45, 15)
        }
        
        cn_oy_images = self._load_images_by_version(cn_oy_path, "CN_OY")
        cy_oy_images = self._load_images_by_version(cy_oy_path, "CY_OY")
        
        # Process each version separately (V3-V7 have different blind zone sizes)
        version_degradation_operators = {}
        
        for version in ['V3', 'V4', 'V5', 'V6', 'V7']:
            if version in cn_oy_images and version in cy_oy_images:
                logger.info(f"Estimating degradation operator for version {version}...")
                
                cn_imgs = cn_oy_images[version]
                cy_imgs = cy_oy_images[version]
                
                # Create donut-shaped mask for this version
                outer_radius, inner_radius = VERSION_RADIUS[version]
                donut_mask = self._create_donut_mask(512, 512, outer_radius, inner_radius)
                
                # Use version-specific noise pattern if available
                noise_pattern = self.version_noise_patterns.get(version, self.noise_pattern) \
                    if hasattr(self, 'version_noise_patterns') else self.noise_pattern
                
                degradation_matrices = []
                
                # Use subset for efficiency
                for cn_img, cy_img in zip(cn_imgs[:10], cy_imgs[:10]):
                    # Ensure images are 512x512
                    cn_img = cn_img.resize((512, 512), Image.LANCZOS)
                    cy_img = cy_img.resize((512, 512), Image.LANCZOS)
                    
                    cn_array = np.array(cn_img).astype(np.float32) / 255.0
                    cy_array = np.array(cy_img).astype(np.float32) / 255.0
                    
                    # Remove estimated noise using version-specific pattern
                    cy_denoised = cy_array - noise_pattern
                    
                    # Estimate linear transformation (simplified)
                    # H * cn_array ≈ cy_denoised
                    # For ultrasound, this is often a spatial masking operation
                    
                    # Apply donut mask to focus on blind zone area only
                    masked_difference = np.abs(cy_denoised - cn_array) * donut_mask
                    
                    # Create binary mask where blind zone artifacts occur
                    # Use donut mask as the degradation operator - 1 where artifacts exist, 0 elsewhere
                    # The donut mask itself defines the blind zone area
                    degradation_mask = donut_mask.copy()
                    
                    degradation_matrices.append(degradation_mask)
                
                # Average degradation mask for this version
                version_degradation_operators[version] = np.mean(degradation_matrices, axis=0)
                
                coverage = np.sum(version_degradation_operators[version] > 0.5) / version_degradation_operators[version].size * 100
                logger.info(f"{version} blind zone coverage: {coverage:.2f}%")
        
        # Store version-specific degradation operators
        self.version_degradation_operators = version_degradation_operators
        
        # Create combined degradation operator
        if version_degradation_operators:
            self.degradation_operator = np.mean(list(version_degradation_operators.values()), axis=0)
        else:
            logger.warning("No version-specific operators found, using fallback method")
            # Fallback to original method
            cn_oy_images = self._load_images(cn_oy_path, pattern="CN_OY_*.bmp")
            cy_oy_images = self._load_images(cy_oy_path, pattern="CY_OY_*.bmp")
            
            degradation_matrices = []
            for cn_img, cy_img in zip(cn_oy_images[:10], cy_oy_images[:10]):
                cn_img = cn_img.resize((512, 512), Image.LANCZOS)
                cy_img = cy_img.resize((512, 512), Image.LANCZOS)
                cn_array = np.array(cn_img).astype(np.float32) / 255.0
                cy_array = np.array(cy_img).astype(np.float32) / 255.0
                cy_denoised = cy_array - self.noise_pattern
                diff = np.abs(cy_denoised - cn_array)
                mask = (diff > 0.1).astype(np.float32)
                degradation_matrices.append(mask)
            
            self.degradation_operator = np.mean(degradation_matrices, axis=0)
        
        logger.info(f"Combined degradation operator shape: {self.degradation_operator.shape}")
        logger.info(f"Combined blind zone coverage: {np.sum(self.degradation_operator > 0.5) / self.degradation_operator.size * 100:.2f}%")
        
        return self.degradation_operator
    
    def ddrm_restore(self, corrupted_image_path, output_path, 
                     steps=50, eta=0.85, etaB=1.0, sigma_0=0.05):
        """
        Perform DDRM restoration on corrupted ultrasound image
        Uses version-specific artifacts if available
        """
        logger.info(f"Starting DDRM restoration for {corrupted_image_path}")
        
        if self.noise_pattern is None or self.degradation_operator is None:
            raise ValueError("Artifacts not estimated. Run estimation methods first.")
        
        # Load corrupted image and resize to 512x512
        corrupted_img = Image.open(corrupted_image_path).convert('L')
        corrupted_img = corrupted_img.resize((512, 512), Image.LANCZOS)
        y = np.array(corrupted_img).astype(np.float32) / 255.0
        y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Detect version from filename if possible
        version = self._detect_version_from_filename(corrupted_image_path)
        
        # Use version-specific patterns if available
        noise_pattern = self.noise_pattern
        degradation_operator = self.degradation_operator
        
        if version and hasattr(self, 'version_noise_patterns') and version in self.version_noise_patterns:
            logger.info(f"Using version-specific artifacts for {version}")
            logger.info(f"Version {version} noise pattern stats: mean={np.mean(self.version_noise_patterns[version]):.4f}, "
                       f"std={np.std(self.version_noise_patterns[version]):.4f}")
            noise_pattern = self.version_noise_patterns[version]
            
        if version and hasattr(self, 'version_degradation_operators') and version in self.version_degradation_operators:
            coverage = np.sum(self.version_degradation_operators[version] > 0.5) / self.version_degradation_operators[version].size * 100
            logger.info(f"Using version-specific degradation operator for {version} (coverage: {coverage:.2f}%)")
            degradation_operator = self.version_degradation_operators[version]
        else:
            logger.warning(f"No version-specific degradation operator found for {version}, using combined operator")
        
        # Convert noise pattern and degradation operator to tensors
        z_tensor = torch.from_numpy(noise_pattern).unsqueeze(0).unsqueeze(0).to(self.device)
        H_tensor = torch.from_numpy(degradation_operator).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # DDRM restoration process
        with torch.no_grad():
            # Initialize from noise (original DDRM approach)
            # DDRM works by iterative denoising with data consistency
            x_T = torch.randn_like(y_tensor)
            
            # Use appropriate scheduler
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                # Use diffusers scheduler for 512x512 model
                restored = self._ddrm_sampling_diffusers(
                    x_T, y_tensor, z_tensor, H_tensor,
                    steps=steps, eta=eta, etaB=etaB, sigma_0=sigma_0
                )
            else:
                # Fallback to original DDRM sampling
                restored = self._ddrm_sampling(
                    x_T, y_tensor, z_tensor, H_tensor,
                    steps=steps, eta=eta, etaB=etaB, sigma_0=sigma_0
                )
        
        # Convert back to image
        restored_np = restored.squeeze().cpu().numpy()
        restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
        
        # Save result
        restored_img = Image.fromarray(restored_np, mode='L')
        restored_img.save(output_path)
        
        logger.info(f"Restored image saved to {output_path} (version: {version})")
        return restored_img
    
    def _ddrm_sampling_diffusers(self, x_T, y, z, H, steps=50, eta=0.85, etaB=1.0, sigma_0=0.05):
        """
        DDRM sampling process with diffusers scheduler (for 512x512 model)
        """
        # Set up scheduler
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps
        
        x = x_T.clone()
        
        for i, t in enumerate(tqdm(timesteps, desc="DDRM Sampling (512x512)")):
            # Predict noise with diffusers model
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    noise_pred = self.model(x, t, return_dict=False)[0]
                else:
                    noise_pred = self.model(x, t)
            
            # Get scheduler parameters - handle diffusers indexing safely
            try:
                alpha_t = self.scheduler.alphas_cumprod[t.item() if hasattr(t, 'item') else t]
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i+1]
                    alpha_t_prev = self.scheduler.alphas_cumprod[t_prev.item() if hasattr(t_prev, 'item') else t_prev]
                else:
                    alpha_t_prev = torch.tensor(1.0, device=self.device)
            except (IndexError, KeyError):
                # Fallback to manual calculation
                step_idx = i / len(timesteps)
                alpha_t = torch.tensor(1 - 0.02 * step_idx, device=self.device)  
                alpha_t_prev = torch.tensor(1 - 0.02 * (step_idx + 1/len(timesteps)), device=self.device)
            
            # Compute x_0 prediction
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # DDRM data consistency step - Apply before scheduler step
            if i < len(timesteps) - 1:  # Don't apply at last step
                # Compute expected measurement
                y_pred = self._apply_degradation(x_0_pred, H, z)
                
                # Data consistency correction
                residual = y - y_pred
                
                # Apply correction with eta scheduling and sigma_0 regularization
                correction = self._solve_inverse_problem(residual, H, sigma_0)
                x_0_pred = x_0_pred + eta * correction
                
                # Update x with corrected x_0_pred using DDPM formula
                if t > 0:
                    beta_t = 1 - alpha_t / alpha_t_prev if i > 0 else 1 - alpha_t
                    noise = torch.randn_like(x)
                    x = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev - beta_t**2) * noise_pred + beta_t * etaB * noise
                else:
                    x = x_0_pred
            else:
                # Final step - use scheduler normally
                if t > 0:
                    x = self.scheduler.step(noise_pred, t, x, return_dict=False)[0]
                else:
                    x = x_0_pred
                
            # Aggressive memory cleanup
            if i % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
                gc.collect()
        
        return x
    
    def _ddrm_sampling(self, x_T, y, z, H, steps=50, eta=0.85, etaB=1.0, sigma_0=0.05):
        """
        Original DDRM sampling process with blind zone guidance (fallback)
        """
        # Get diffusion scheduler
        betas = np.linspace(1e-4, 0.02, 1000)
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas)
        
        x = x_T.clone()
        
        # Sampling timesteps
        timesteps = np.linspace(999, 0, steps).astype(int)
        
        for i, t in enumerate(tqdm(timesteps, desc="DDRM Sampling (Fallback)")):
            t_tensor = torch.tensor([t], device=self.device)
            
            # Predict noise with DDPM
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    noise_pred = self.model(x, t_tensor)
                else:
                    noise_pred = self.model(x, t_tensor, return_dict=False)[0]
            
            # DDPM update
            alpha_t = alphas_cumprod[t]
            alpha_t_prev = alphas_cumprod[t-1] if t > 0 else 1.0
            
            # Compute x_0 prediction
            x_0_pred = (x - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
            
            # DDRM data consistency step
            if i < steps - 1:  # Don't apply at last step
                # Compute expected measurement
                y_pred = self._apply_degradation(x_0_pred, H, z)
                
                # Data consistency correction
                residual = y - y_pred
                
                # Apply correction with eta scheduling
                correction = self._solve_inverse_problem(residual, H, alpha_t)
                x_0_pred = x_0_pred + eta * correction
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = etaB * np.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                x = np.sqrt(alpha_t_prev) * x_0_pred + np.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred + sigma_t * noise
            else:
                x = x_0_pred
                
            # Aggressive memory cleanup
            if i % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
                gc.collect()
        
        return x
    
    def _apply_degradation(self, x, H, z):
        """Apply degradation: H*x + z"""
        return H * x + z
    
    def _solve_inverse_problem(self, residual, H, sigma_0=0.05):
        """
        Solve inverse problem for data consistency with sigma_0 regularization
        For blind zone removal: solve H^+ * residual where H is donut mask
        
        Args:
            residual: measurement residual (y - H*x_0 - z)
            H: degradation operator (donut mask for blind zone)
            sigma_0: noise level parameter for regularization
        """
        # Convert to proper shape for computation
        H_flat = H.view(H.shape[0], -1)
        residual_flat = residual.view(residual.shape[0], -1)
        
        # Compute singular values (for donut mask, these are just the mask values)
        mask_values = torch.abs(H_flat)
        
        # Apply sigma_0 regularization as in original DDRM
        # Only invert singular values that are large enough compared to sigma_0
        large_singulars = mask_values > sigma_0
        
        # Regularized pseudo-inverse
        H_pinv_flat = torch.zeros_like(H_flat)
        H_pinv_flat[large_singulars] = sigma_0 / mask_values[large_singulars]
        
        # Compute correction
        correction_flat = H_pinv_flat * residual_flat
        correction = correction_flat.view_as(residual)
        
        # Apply only where mask is active and above sigma_0 threshold
        active_mask = (torch.abs(H) > sigma_0).float()
        correction = correction * active_mask
        
        return correction
    
    def _load_images(self, path, pattern="*.bmp"):
        """Load images from directory with pattern"""
        path = Path(path)
        images = []
        
        if path.is_file():
            # Single file
            img = Image.open(path).convert('L')
            img = img.resize((512, 512), Image.LANCZOS)  # Ensure 512x512
            images.append(img)
        else:
            # Directory with pattern
            from glob import glob
            files = sorted(glob(str(path / pattern)))
            for file_path in files:
                img = Image.open(file_path).convert('L')
                img = img.resize((512, 512), Image.LANCZOS)  # Ensure 512x512
                images.append(img)
        
        logger.info(f"Loaded {len(images)} images from {path}")
        return images
    
    def _load_images_by_version(self, path, prefix):
        """Load images organized by version (V3-V7)"""
        path = Path(path)
        version_images = {}
        
        from glob import glob
        
        # Load images for each version
        for version in ['V3', 'V4', 'V5', 'V6', 'V7']:
            pattern = f"{prefix}*{version}*.bmp"
            files = sorted(glob(str(path / pattern)))
            
            if not files:
                # Try alternative directory structure
                for subdir in ['data/train_CN_CY_ALL', 'test_CY_ON', 'train_gt']:
                    alt_path = path.parent / subdir
                    if alt_path.exists():
                        files = sorted(glob(str(alt_path / pattern)))
                        if files:
                            break
            
            if files:
                images = []
                for file_path in files:
                    img = Image.open(file_path).convert('L')
                    img = img.resize((512, 512), Image.LANCZOS)  # Ensure 512x512
                    images.append(img)
                
                version_images[version] = images
                logger.info(f"Loaded {len(images)} {prefix} images for {version} from {path}")
        
        return version_images
    
    def _get_version_threshold(self, version):
        """Get version-specific threshold for blind zone detection"""
        # Different versions may have different blind zone characteristics
        version_thresholds = {
            'V3': 0.08,  # Smaller blind zones, lower threshold
            'V4': 0.10,  # Medium blind zones
            'V5': 0.12,  # Medium-large blind zones
            'V6': 0.15,  # Large blind zones
            'V7': 0.18   # Largest blind zones, higher threshold
        }
        return version_thresholds.get(version, 0.1)  # Default threshold
    
    def _detect_version_from_filename(self, filepath):
        """Detect version (V3-V7) from filename"""
        filepath = str(filepath)
        for version in ['V3', 'V4', 'V5', 'V6', 'V7']:
            if version in filepath:
                return version
        return None

def main():
    parser = argparse.ArgumentParser(description="Ultrasound DDRM Blind Zone Removal")
    parser.add_argument("--config", type=str, required=True, help="Path to DDRM config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained DDPM model")
    parser.add_argument("--cn_on_path", type=str, required=True, help="Path to CN_ON images")
    parser.add_argument("--cy_on_path", type=str, required=True, help="Path to CY_ON images")
    parser.add_argument("--cn_oy_path", type=str, required=True, help="Path to CN_OY images")
    parser.add_argument("--cy_oy_path", type=str, required=True, help="Path to CY_OY images")
    parser.add_argument("--test_image", type=str, required=True, help="Path to test image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save restored image")
    parser.add_argument("--noise_save_path", type=str, help="Path to save noise pattern")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDRM steps")
    parser.add_argument("--eta", type=float, default=0.85, help="DDRM eta parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize DDRM
    ddrm = UltrasoundDDRM(args.config, args.model_path, args.device)
    
    # Step 1: Estimate blind zone artifacts
    logger.info("=== Step 1: Estimating blind zone artifacts ===")
    ddrm.estimate_blind_zone_artifacts(
        args.cn_on_path, 
        args.cy_on_path,
        args.noise_save_path
    )
    
    # Step 2: Estimate degradation operator
    logger.info("=== Step 2: Estimating degradation operator ===")
    ddrm.estimate_degradation_operator(args.cn_oy_path, args.cy_oy_path)
    
    # Step 3: Perform DDRM restoration
    logger.info("=== Step 3: DDRM restoration ===")
    ddrm.ddrm_restore(
        args.test_image,
        args.output_path,
        steps=args.steps,
        eta=args.eta
    )
    
    logger.info("DDRM restoration completed!")

if __name__ == "__main__":
    main()