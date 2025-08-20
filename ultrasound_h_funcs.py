import torch
import numpy as np
from functions.svd_replacement import H_functions
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class UltrasoundBlindZone(H_functions):
    """
    Enhanced H_functions implementation for ultrasound blind zone removal
    Handles version-specific (V3-V7) donut-shaped blind zones with physical modeling
    
    Based on DDRM methodology with ultrasound-specific modifications:
    - z_est = Average(CY_ON - CN_ON): structural noise estimation 
    - H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²: distortion operator
    - Physics-based modeling: blind zone as physical distortion, not masking
    """
    
    def __init__(self, channels, img_size, device, version=None, noise_pattern=None, distortion_factor=0.05, noise_factor=0.02):
        self.channels = channels
        self.img_size = img_size
        self.device = device
        self.distortion_factor = distortion_factor
        self.noise_factor = noise_factor
        self.version = version
        
        # Enhanced version-specific parameters based on blind zone physics
        # (outer_radius, inner_radius, distortion_strength, noise_level)
        self.VERSION_PARAMS = {
            "V3": {"outer_r": 220, "inner_r": 85, "strength": 0.8, "noise": 0.05},
            "V4": {"outer_r": 130, "inner_r": 50, "strength": 0.9, "noise": 0.06}, 
            "V5": {"outer_r": 90, "inner_r": 30, "strength": 1.0, "noise": 0.07},
            "V6": {"outer_r": 60, "inner_r": 20, "strength": 1.1, "noise": 0.08},
            "V7": {"outer_r": 45, "inner_r": 15, "strength": 1.2, "noise": 0.09}
        }
        
        # Create physics-based distortion model for this version
        if version and version in self.VERSION_PARAMS:
            params = self.VERSION_PARAMS[version]
            self.distortion_mask = self._create_physics_distortion_mask(
                img_size, img_size, params["outer_r"], params["inner_r"], 
                params["strength"]
            )
            self.noise_level = params["noise"]
            logger.info(f"Created physics-based distortion model for {version}")
            logger.info(f"  - Outer radius: {params['outer_r']}, Inner radius: {params['inner_r']}")
            logger.info(f"  - Distortion strength: {params['strength']}, Noise level: {params['noise']}")
        else:
            # Default combined distortion model
            self.distortion_mask = self._create_combined_distortion_mask(img_size, img_size)
            self.noise_level = 0.07
            logger.info("Created combined physics-based distortion model")
            
        self.mask_tensor = torch.from_numpy(self.distortion_mask).float().to(device)
        
        # Store noise pattern for z_est = Average(CY_ON - CN_ON)
        if noise_pattern is not None:
            self.noise_pattern = torch.from_numpy(noise_pattern).float().to(device)
        else:
            self.noise_pattern = torch.zeros(img_size, img_size).to(device)
            
        # Pre-compute SVD components for efficient DDRM sampling
        self._compute_svd_components()
        
    def _create_physics_distortion_mask(self, height, width, outer_radius, inner_radius, strength):
        """
        Create physics-based distortion model for blind zone
        Instead of binary masking, models physical ultrasound distortion
        """
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create smooth distortion profile (not binary)
        distortion_mask = np.zeros((height, width), dtype=np.float32)
        
        # Physics-based distortion model: gradual falloff
        in_blind_zone = (distance >= inner_radius) & (distance <= outer_radius)
        
        if np.any(in_blind_zone):
            # Normalized distance within blind zone [0, 1]
            zone_distance = (distance - inner_radius) / (outer_radius - inner_radius)
            zone_distance = np.clip(zone_distance, 0, 1)
            
            # Physics-based distortion profile (bell curve)
            # Maximum distortion in middle of blind zone
            distortion_profile = np.exp(-((zone_distance - 0.5) * 4) ** 2) * strength
            distortion_mask[in_blind_zone] = distortion_profile[in_blind_zone]
            
        return distortion_mask
    
    def _create_combined_distortion_mask(self, height, width):
        """Create combined physics-based distortion model for all versions"""
        combined_mask = np.zeros((height, width), dtype=np.float32)
        
        for version, params in self.VERSION_PARAMS.items():
            version_mask = self._create_physics_distortion_mask(
                height, width, params["outer_r"], params["inner_r"], params["strength"]
            )
            # Combine using maximum distortion
            combined_mask = np.maximum(combined_mask, version_mask)
            
        return combined_mask
    
    def _compute_svd_components(self):
        """
        Compute SVD components for efficient DDRM sampling
        Enhanced for physics-based distortion model but simplified for compatibility
        """
        # Simplified approach for DDRM compatibility
        # Create uniform singular values for the full image
        total_pixels = self.img_size * self.img_size
        
        # Create singular values based on physical distortion model
        # All pixels get singular values, but distorted regions get higher values
        singular_values = np.ones(total_pixels, dtype=np.float32) * 0.1  # Base value
        
        # Enhance singular values in distorted regions
        mask_flat = self.distortion_mask.flatten()
        singular_values = singular_values + mask_flat * 0.9  # Add distortion strength
        
        # Normalize and add regularization
        singular_values = singular_values / np.max(singular_values)
        singular_values = singular_values + 1e-6  # Regularization
        
        self._singulars = torch.from_numpy(singular_values).float().to(self.device)
        
        logger.info(f"Computed {len(self._singulars)} SVD components for full image")
        
    def singulars(self):
        """Returns singular values of the degradation operator"""
        return self._singulars
        
    def add_zeros(self, vec):
        """Add zeros to match dimensions"""
        # For blind zone, this relates to expanding from masked region to full image
        active_pixels = torch.sum(self.mask_tensor > 0.5).item()
        total_pixels = self.img_size * self.img_size
        
        if vec.shape[-1] == active_pixels:
            # Expand from active region to full image
            batch_size = vec.shape[0]
            expanded = torch.zeros(batch_size, total_pixels).to(self.device)
            
            mask_indices = torch.where(self.mask_tensor.flatten() > 0.5)[0]
            expanded[:, mask_indices] = vec
            
            return expanded
        else:
            return vec
    
    def V(self, vec):
        """
        Multiply by V matrix (reconstruction from SVD space)
        For ultrasound physics, V is approximately identity
        """
        # Handle input shapes and ensure correct dimensions
        if len(vec.shape) == 1:
            vec = vec.view(1, -1)
        elif len(vec.shape) == 4:  # Already in image format
            return vec
        
        batch_size = vec.shape[0]
        
        # Simply reshape to image format - V is identity for ultrasound
        if vec.shape[1] == self.img_size * self.img_size:
            return vec.view(batch_size, self.channels, self.img_size, self.img_size)
        else:
            # Handle partial vectors by zero-padding
            full_vec = torch.zeros(batch_size, self.img_size * self.img_size, device=vec.device)
            copy_size = min(vec.shape[1], full_vec.shape[1])
            full_vec[:, :copy_size] = vec[:, :copy_size]
            return full_vec.view(batch_size, self.channels, self.img_size, self.img_size)
        
    def Vt(self, vec):
        """
        Multiply by V transpose (projection to SVD space)
        For ultrasound physics, Vt is approximately identity (flatten)
        """
        # Simple flattening - Vt is identity for ultrasound
        if len(vec.shape) == 4:  # (B, C, H, W)
            return vec.view(vec.shape[0], -1)
        elif len(vec.shape) == 3:  # (B, H, W)  
            return vec.view(vec.shape[0], -1)
        elif len(vec.shape) == 1:  # (features,)
            return vec.view(1, -1)
        else:  # Already 2D
            return vec
        
    def U(self, vec):
        """
        Multiply by U matrix (measurement operator)
        For ultrasound, this applies the physics-based distortion
        """
        if len(vec.shape) == 4:  # (B, C, H, W)
            # Apply physics-based measurement and flatten
            measured = self.H(vec)
            return self.Vt(measured)
        else:
            # Reconstruct, apply measurement, project back
            reconstructed = self.V(vec)
            measured = self.H(reconstructed)
            return self.Vt(measured)
        
    def Ut(self, vec):
        """
        Multiply by U transpose
        For ultrasound physics, this is the adjoint measurement operator
        """
        # Ensure proper shape
        if len(vec.shape) == 1:
            vec = vec.view(1, -1)
        elif len(vec.shape) > 2:
            vec = vec.view(vec.shape[0], -1)
            
        # Reconstruct from measurement space
        reconstructed = self.V(vec)
        
        # Apply adjoint of measurement (approximate inverse)
        adjoint_result = self.H_pinv(reconstructed)
        
        return self.Vt(adjoint_result)
        
    def H(self, vec):
        """
        Apply physics-based degradation H*x + z
        Combines distortion operator and structural noise (controlled strength)
        """
        if len(vec.shape) == 4:  # (B, C, H, W)
            # Apply controlled physics-based distortion (configurable strength)
            mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0)
            distorted = vec * (1.0 + mask_expanded * self.distortion_factor)
            
            # Add controlled structural noise (configurable strength)
            noise_expanded = self.noise_pattern.unsqueeze(0).unsqueeze(0).expand_as(distorted)
            
            # Add noise only where distortion occurs
            distorted = distorted + noise_expanded * mask_expanded * self.noise_factor
            
            # Ensure output stays in reasonable range
            distorted = torch.clamp(distorted, 0.0, 1.0)
            return distorted
        else:  # Flattened
            vec_2d = vec.view(vec.shape[0], self.channels, self.img_size, self.img_size)
            result_2d = self.H(vec_2d)
            return result_2d.view(vec.shape[0], -1)
            
    def H_pinv(self, vec):
        """
        Apply physics-based pseudo-inverse H^+ 
        Solves argmin_x ||H*x + z - vec||² with regularization
        """
        if len(vec.shape) == 4:  # (B, C, H, W)
            # Remove structural noise first
            noise_expanded = self.noise_pattern.unsqueeze(0).unsqueeze(0).expand_as(vec)
            mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0).expand_as(vec)
            
            # Subtract estimated noise in distorted regions
            denoised = vec - noise_expanded * mask_expanded
            
            # Apply physics-based inverse with regularization
            # For distortion model (1 + mask), inverse is approximately (1 - mask/(1+mask))
            regularized_mask = self.mask_tensor / (1.0 + self.mask_tensor + 1e-6)
            inverse_expansion = (1.0 - regularized_mask).unsqueeze(0).unsqueeze(0)
            
            result = denoised * inverse_expansion
            return result
        else:  # Flattened
            vec_2d = vec.view(vec.shape[0], self.channels, self.img_size, self.img_size)
            result_2d = self.H_pinv(vec_2d)
            return result_2d.view(vec.shape[0], -1)


# Second UltrasoundBlindZoneWithNoise class removed to avoid conflicts
# Using the main physics-based implementation from line 13


def create_ultrasound_h_funcs(config, version=None, noise_pattern=None, distortion_factor=0.05, noise_factor=0.02):
    """Factory function to create appropriate H_functions for ultrasound"""
    
    channels = getattr(config.data, 'channels', 1)
    img_size = getattr(config.data, 'image_size', 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Creating UltrasoundBlindZone for version {version}")
    logger.info(f"  - Distortion factor: {distortion_factor}")
    logger.info(f"  - Noise factor: {noise_factor}")
    logger.info(f"  - Noise pattern: {'provided' if noise_pattern is not None else 'default zeros'}")
    return UltrasoundBlindZone(channels, img_size, device, version, noise_pattern, distortion_factor, noise_factor)


def estimate_version_artifacts(cn_on_path, cy_on_path, version):
    """
    Enhanced structural noise estimation: z_est = Average(CY_ON - CN_ON)
    Implements version-specific (V3-V7) blind zone artifact estimation
    """
    logger.info(f"Estimating structural noise artifacts for version {version}")
    
    # Load version-specific images with better pattern matching
    cn_files = []
    cy_files = []
    
    # Try multiple pattern variations for robustness
    patterns = [
        f"*CN*{version}*.bmp", f"CN*{version}*.bmp", f"*{version}*CN*.bmp",
        f"*CN_ON*{version}*.bmp", f"CN_ON*{version}*.bmp"
    ]
    
    for pattern in patterns:
        cn_matches = sorted(list(Path(cn_on_path).glob(pattern)))
        if cn_matches:
            cn_files = cn_matches
            break
    
    for pattern in patterns:
        pattern_cy = pattern.replace('CN', 'CY')
        cy_matches = sorted(list(Path(cy_on_path).glob(pattern_cy)))
        if cy_matches:
            cy_files = cy_matches
            break
    
    if not cn_files or not cy_files:
        logger.warning(f"No {version} files found in {cn_on_path} or {cy_on_path}")
        return None, None
        
    logger.info(f"Found {len(cn_files)} CN_ON and {len(cy_files)} CY_ON files for {version}")
    
    # Enhanced noise estimation with physics-based processing
    noise_patterns = []
    distortion_maps = []
    
    # Version-specific processing parameters
    version_params = {
        "V3": {"outer_r": 220, "inner_r": 85, "strength": 0.8},
        "V4": {"outer_r": 130, "inner_r": 50, "strength": 0.9}, 
        "V5": {"outer_r": 90, "inner_r": 30, "strength": 1.0},
        "V6": {"outer_r": 60, "inner_r": 20, "strength": 1.1},
        "V7": {"outer_r": 45, "inner_r": 15, "strength": 1.2}
    }
    
    params = version_params.get(version, {"outer_r": 100, "inner_r": 40, "strength": 1.0})
    
    for cn_file, cy_file in zip(cn_files[:15], cy_files[:15]):  # Use more samples for robustness
        cn_img = np.array(Image.open(cn_file).convert('L').resize((512, 512))) / 255.0
        cy_img = np.array(Image.open(cy_file).convert('L').resize((512, 512))) / 255.0
        
        # Compute structural noise: z = CY_ON - CN_ON
        structural_noise = cy_img - cn_img
        
        # Create version-specific region mask for focused estimation
        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        region_mask = ((distance >= params["inner_r"]) & (distance <= params["outer_r"])).astype(np.float32)
        
        # Apply region mask to focus on blind zone
        focused_noise = structural_noise * region_mask
        noise_patterns.append(focused_noise)
        
        # Estimate distortion strength map
        distortion_strength = np.abs(structural_noise) * region_mask
        distortion_maps.append(distortion_strength)
    
    # Compute average structural noise pattern z_est
    if noise_patterns:
        z_est = np.mean(noise_patterns, axis=0)
        distortion_est = np.mean(distortion_maps, axis=0)
        
        # Log version-specific statistics
        active_region = z_est[z_est != 0]
        if len(active_region) > 0:
            logger.info(f"{version} structural noise z_est stats:")
            logger.info(f"  - Mean: {np.mean(active_region):.4f}, Std: {np.std(active_region):.4f}")
            logger.info(f"  - Min: {np.min(active_region):.4f}, Max: {np.max(active_region):.4f}")
            logger.info(f"  - Coverage: {len(active_region) / (512*512) * 100:.2f}%")
        
        return z_est, distortion_est
    else:
        logger.warning(f"No valid noise patterns computed for {version}")
        return None, None


def estimate_degradation_operator(cn_oy_path, cy_oy_path, z_est, version):
    """
    Enhanced degradation operator estimation:
    H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²
    """
    logger.info(f"Estimating degradation operator H_est for version {version}")
    
    if z_est is None:
        logger.error("Structural noise z_est not provided")
        return None
    
    # Load version-specific OY images
    cn_oy_files = []
    cy_oy_files = []
    
    patterns = [
        f"*CN*{version}*.bmp", f"CN*{version}*.bmp", f"*{version}*CN*.bmp",
        f"*CN_OY*{version}*.bmp", f"CN_OY*{version}*.bmp"
    ]
    
    for pattern in patterns:
        cn_matches = sorted(list(Path(cn_oy_path).glob(pattern)))
        if cn_matches:
            cn_oy_files = cn_matches
            break
    
    for pattern in patterns:
        pattern_cy = pattern.replace('CN', 'CY')
        cy_matches = sorted(list(Path(cy_oy_path).glob(pattern_cy)))
        if cy_matches:
            cy_oy_files = cy_matches
            break
    
    if not cn_oy_files or not cy_oy_files:
        logger.warning(f"No {version} OY files found")
        return None
    
    logger.info(f"Found {len(cn_oy_files)} CN_OY and {len(cy_oy_files)} CY_OY files for {version}")
    
    # Solve for H: minimize ||H·(CN_OY) - (CY_OY - z_est)||²
    h_estimates = []
    
    for cn_file, cy_file in zip(cn_oy_files[:10], cy_oy_files[:10]):
        cn_oy = np.array(Image.open(cn_file).convert('L').resize((512, 512))) / 255.0
        cy_oy = np.array(Image.open(cy_file).convert('L').resize((512, 512))) / 255.0
        
        # Remove structural noise from measurement
        cy_corrected = cy_oy - z_est
        
        # Solve H·cn_oy ≈ cy_corrected
        # For pixel-wise operation: H[i,j] = cy_corrected[i,j] / (cn_oy[i,j] + eps)
        eps = 1e-6
        h_estimate = np.divide(cy_corrected, cn_oy + eps, out=np.zeros_like(cy_corrected), where=(cn_oy + eps) != 0)
        
        # Regularize extreme values
        h_estimate = np.clip(h_estimate, 0.1, 3.0)
        h_estimates.append(h_estimate)
    
    if h_estimates:
        H_est = np.mean(h_estimates, axis=0)
        logger.info(f"{version} degradation operator H_est computed")
        logger.info(f"  - Mean: {np.mean(H_est):.4f}, Std: {np.std(H_est):.4f}")
        return H_est
    else:
        return None