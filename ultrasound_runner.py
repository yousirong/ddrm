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

# Import cv2 for image processing
try:
    import cv2
except ImportError:
    print("Warning: cv2 not available for runner. Please install opencv-python.")
    cv2 = None

# Configure matplotlib for English fonts
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

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

        # Initialize registration cache
        self.registration_cache = {}
        self.cache_file = Path('registration_cache.json')
        self._load_registration_cache()

        # Extract custom thresholds from args
        self.custom_thresholds = {}
        if hasattr(args, 'threshold_v3'):
            self.custom_thresholds['V3'] = args.threshold_v3
        if hasattr(args, 'threshold_v4'):
            self.custom_thresholds['V4'] = args.threshold_v4
        if hasattr(args, 'threshold_v5'):
            self.custom_thresholds['V5'] = args.threshold_v5
        if hasattr(args, 'threshold_v6'):
            self.custom_thresholds['V6'] = args.threshold_v6
        if hasattr(args, 'threshold_v7'):
            self.custom_thresholds['V7'] = args.threshold_v7

        logger.info("Initialized UltrasoundDDRMRunner with base DDRM framework")
        if self.custom_thresholds:
            logger.info(f"Custom thresholds: {self.custom_thresholds}")

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
            custom_threshold = self.custom_thresholds.get(version, None)
            z_est, distortion_map = estimate_version_artifacts(cn_on_path, cy_on_path, version, custom_threshold, current_filename="estimation_mode")

            if z_est is not None:
                self.version_artifacts[version] = {
                    'noise_pattern': z_est,
                    'distortion_map': distortion_map
                }
                logger.info(f"{version} structural noise z_est estimated")

                # Step 2: Degradation operator estimation H_est (if OY data available)
                if cn_oy_path and cy_oy_path:
                    H_est = estimate_degradation_operator(cn_oy_path, cy_oy_path, z_est, version, current_filename="estimation_mode")
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

    def create_h_functions(self, version=None, filename=None):
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
        distortion_factor = getattr(self.args, 'distortion_factor', 0.025)
        noise_factor = getattr(self.args, 'noise_factor', 1.0)

        # Version-specific thresholds are now handled directly in the dataset building methods
        # No need to pass percentile parameters - they are managed internally
        version_thresholds = None  # Not used with dataset building approach

        # Get mask cleaning parameters
        tissue_min_size = getattr(self.args, 'tissue_min_size', 200)
        blind_zone_min_size = getattr(self.args, 'blind_zone_min_size', 100)

        # Get other enhanced detection parameters
        enhanced_tissue_detection = getattr(self.args, 'enhanced_tissue_detection', True)
        tissue_detection_mode = getattr(self.args, 'tissue_detection_mode', 'multi')
        clahe_clip_limit = getattr(self.args, 'clahe_clip_limit', 3.0)
        min_tissue_size_factor = getattr(self.args, 'min_tissue_size_factor', 1.0)
        complete_blind_zone_removal = getattr(self.args, 'complete_blind_zone_removal', True)
        preserve_background = getattr(self.args, 'preserve_background', True)

        return create_ultrasound_h_funcs(
            self.config, version, noise_pattern, distortion_factor, noise_factor,
            enhanced_tissue_detection, tissue_detection_mode, clahe_clip_limit,
            min_tissue_size_factor, complete_blind_zone_removal, preserve_background,
            version_thresholds, tissue_min_size, blind_zone_min_size, filename, self
        )

    def sample_ultrasound_sequence(self, test_images_path, output_dir, sigma_0=0.05, save_steps=None, optuna_mode=False, no_save_images=False, reference_path=None):
        """
        Enhanced sampling sequence for ultrasound images
        Uses version-specific processing and physics-based modeling
        """
        logger.info("Starting ultrasound DDRM sampling...")
        if optuna_mode:
            logger.info("Optuna mode enabled - using memory-only evaluation")
        if no_save_images:
            logger.info("Image saving disabled")
        if save_steps and not no_save_images:
            logger.info(f"Will save intermediate steps: {save_steps}")

        # Load model (use base DDRM model loading)
        model = self._load_ddrm_model()

        # Create output directory (only if saving images)
        output_dir = Path(output_dir)
        if not no_save_images:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for Optuna mode
        temp_dir = None
        if optuna_mode:
            temp_dir = Path(output_dir).parent / "temp_optuna_results"
            temp_dir.mkdir(exist_ok=True)

        # Load test images (Optuna 모드에서는 10개만)
        test_images = self._load_test_images(test_images_path, optuna_mode, reference_path)
        logger.info(f"Loaded {len(test_images)} test images{' (Optuna mode - 10 images only)' if optuna_mode else ''}")

        # Process each image with version-specific handling
        results = []

        for i, (image_path, image) in enumerate(test_images):
            logger.info(f"Processing image {i+1}/{len(test_images)}: {image_path.name}")

            # Detect version from filename
            version = self.detect_version_from_path(image_path)
            logger.info(f"Detected version: {version}")

            # Create version-specific H_functions
            H_funcs = self.create_h_functions(version, image_path.name)

            # Save individual artifacts for this image
            artifacts_dir = output_dir / "artifacts"
            self._save_individual_artifacts(image_path.name, H_funcs, artifacts_dir)

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

            # 조직 분석 시각화 저장 (Optuna 모드에서는 생략)
            if not no_save_images:
                self._save_tissue_analysis(x_orig, y_0, version, image_path, output_dir, H_funcs)

            # Save degraded image (Optuna 모드에서는 생략)
            degraded_path = None
            if not no_save_images:
                degraded_path = output_dir / f"{image_path.stem}_degraded.png"
                tvu.save_image(y_0.squeeze(), degraded_path)

            # Save original (Optuna 모드에서는 생략)
            orig_path = None
            if not no_save_images:
                orig_path = output_dir / f"{image_path.stem}_original.png"
                tvu.save_image(x_orig.squeeze(), orig_path)

            # DDRM restoration using enhanced sampling
            x_T = torch.randn_like(x_orig)

            # Create callback for saving intermediate steps (only if not in no_save_images mode)
            def save_step_callback(xt, step, x0_pred):
                if not no_save_images:
                    logger.info(f"Saving intermediate step {step} for {image_path.stem}")
                    # Save the denoised intermediate step
                    step_path = output_dir / f"{image_path.stem}_step_{step:03d}.png"
                    # Robust normalization for intermediate steps
                    if torch.abs(xt).max() > 5.0:
                        # Apply robust normalization
                        p1, p99 = torch.quantile(xt, torch.tensor([0.01, 0.99]))
                        xt_norm = (xt - p1) / (p99 - p1 + 1e-8)
                        xt_norm = xt_norm.clamp(0, 1)
                    else:
                        xt_norm = xt.clamp(0, 1)
                    tvu.save_image(xt_norm.squeeze(), step_path)
                    logger.info(f"Saved intermediate step {step}: {step_path.name}")

                    # Also save x0 prediction (current estimate of clean image)
                    if x0_pred is not None:
                        x0_path = output_dir / f"{image_path.stem}_step_{step:03d}_x0.png"
                        # Apply same normalization to x0 prediction
                        if torch.abs(x0_pred).max() > 5.0:
                            p1, p99 = torch.quantile(x0_pred, torch.tensor([0.01, 0.99]))
                            x0_norm = (x0_pred - p1) / (p99 - p1 + 1e-8)
                            x0_norm = x0_norm.clamp(0, 1)
                        else:
                            x0_norm = x0_pred.clamp(0, 1)
                        tvu.save_image(x0_norm.squeeze(), x0_path)

            # Use base DDRM sampling with ultrasound H_functions
            with torch.no_grad():
                restored_sequence = self.sample_image_ultrasound(
                    x_T, model, H_funcs, y_0, sigma_0, version=version,
                    save_steps=save_steps if not no_save_images else None,
                    save_callback=save_step_callback if save_steps and not no_save_images else None
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
                    target_mean = min(0.9, x_orig.mean().cpu().item() * 1.5)  # Target slightly brighter than input
                    current_mean = restored_tensor.mean()
                    restored_tensor = restored_tensor * (target_mean / current_mean)
                    restored_tensor = restored_tensor.clamp(0, 1)

                logger.info(f"Final image range: Min: {restored_tensor.min():.3f}, Max: {restored_tensor.max():.3f}, Mean: {restored_tensor.mean():.3f}")

                # Apply original color restoration for non-blind-zone areas
                if H_funcs.tissue_mask is not None:
                    restored_tensor = self._restore_original_colors(restored_tensor, H_funcs.tissue_mask, H_funcs.blind_zone_processing_mask, x_orig.squeeze().cpu(), version)

                # Save restored image to disk or temporary numpy file
                restored_path = None
                if not no_save_images:
                    restored_path = output_dir / f"{image_path.stem}_restored_{version}.png"
                    tvu.save_image(restored_tensor, restored_path)
                    logger.info(f"Saved restored image: {restored_path}")
                elif optuna_mode and temp_dir:
                    # Save as numpy array for Optuna evaluation
                    temp_name = f"{image_path.stem}_temp_{version}.npy"
                    temp_path = temp_dir / temp_name
                    np.save(temp_path, restored_tensor.numpy())
                    logger.info(f"Saved temporary result for Optuna: {temp_path}")

                results.append({
                    'original_path': image_path,
                    'version': version,
                    'restored_path': restored_path,
                    'degraded_path': degraded_path,
                    'restored_tensor': restored_tensor if optuna_mode else None  # Keep tensor in memory for Optuna
                })

        logger.info(f"Ultrasound DDRM sampling completed. Results saved to {output_dir}")
        return results

    def sample_image_ultrasound(self, x, model, H_funcs, y_0, sigma_0, last=True, version=None, save_steps=None, save_callback=None):
        """
        Enhanced DDRM sampling for ultrasound with physics-based corrections
        Uses base efficient_generalized_steps with ultrasound-specific H_functions
        """
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        logger.info(f"Starting DDRM sampling with {len(seq)} timesteps for {version}")

        # Basic DDRM sampling without tissue protection

        # Basic DDRM sampling
        x_sequence = efficient_generalized_steps(
            x, seq, model, self.betas, H_funcs, y_0, sigma_0,
            etaB=self.args.etaB,
            etaA=self.args.eta,
            etaC=self.args.eta,
            cls_fn=None,
            classes=None,
            save_steps=save_steps,
            save_callback=save_callback
        )

        if last:
            return x_sequence[0][-1]
        return x_sequence[0]

    def _load_ddrm_model(self):
        """Load DDRM model from user-specified diffusers path"""
        model_path = "/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100/best_model/unet"
        model = UNet2DModel.from_pretrained(model_path, local_files_only=True)
        model.to(self.device)
        model.eval()  # Set to evaluation mode

        # Don't use DataParallel if there's only one GPU or if it causes issues
        if torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        else:
            logger.info("Using single GPU, skipping DataParallel")

        logger.info(f"Loaded UNet2DModel from: {model_path}")
        return model

    def _enhance_tissue_pixels(self, restored_image, tissue_mask, original_image, version):
        """
        V3~V7 버전별 차등 질감 보존: 조직의 percentile 이상 부분만 원본 픽셀값 그대로 적용
        V3(85th): 상위 15%, V4(80th): 상위 20%, V5(75th): 상위 25%, V6(65th): 상위 35%, V7(55th): 상위 45%
        나머지 모든 영역은 복원 결과 완전 보존
        """
        if tissue_mask is None:
            logger.info("조직 마스크가 없어 후처리를 건너뜁니다")
            return restored_image

        # 텐서를 numpy로 변환
        if isinstance(restored_image, torch.Tensor):
            restored_np = restored_image.cpu().numpy()
        else:
            restored_np = restored_image

        if isinstance(original_image, torch.Tensor):
            original_np = original_image.cpu().numpy()
        else:
            original_np = original_image

        # 텐서 형태 맞추기
        if len(restored_np.shape) == 3:
            restored_np = restored_np[0]  # (1, H, W) -> (H, W)
        if len(original_np.shape) == 3:
            original_np = original_np[0]  # (1, H, W) -> (H, W)

        # 조직 마스크 형태 확인
        if isinstance(tissue_mask, torch.Tensor):
            tissue_mask = tissue_mask.cpu().numpy()

        # 복원된 이미지를 기본으로 사용 (모든 영역 보존)
        enhanced_np = restored_np.copy()
        tissue_region = tissue_mask > 0.1

        if not np.any(tissue_region):
            logger.warning(f"{version} 조직 영역이 없어 후처리를 건너뜁니다")
            return restored_image

        # 버전별 차등 percentile 임계값 (V3: 조직 많음 → 높은 임계값, V7: 조직 적음 → 낮은 임계값)
        version_percentiles = {
            'V3': 70,  # 대형 블라인드존: 상위 15%만 질감 보존
            'V4': 65,  # 중대형 블라인드존: 상위 20%만 질감 보존
            'V5': 60,  # 중형 블라인드존: 상위 25%만 질감 보존
            'V6': 55,  # 소형 블라인드존: 상위 35%만 질감 보존
            'V7': 50   # 최소형 블라인드존: 상위 45%만 질감 보존
        }

        # 버전별 percentile 임계값 사용 (기본값: V5)
        percentile_threshold = version_percentiles.get(version, 75)

        tissue_pixels = original_np[tissue_region]
        strong_tissue_threshold = np.percentile(tissue_pixels, percentile_threshold)
        strong_tissue_region = tissue_region & (original_np >= strong_tissue_threshold)

        if np.any(strong_tissue_region):
            # 버전별 percentile 이상 부분에 원본 픽셀값 분포 그대로 적용
            enhanced_np[strong_tissue_region] = original_np[strong_tissue_region]

            strong_count = np.sum(strong_tissue_region)
            total_tissue_count = np.sum(tissue_region)
            preservation_percentage = 100 - percentile_threshold  # 보존되는 조직 비율

            logger.info(f"{version} {percentile_threshold}th percentile 이상 원본 픽셀값 직접 적용: {strong_count}/{total_tissue_count}개 픽셀 (임계값: {strong_tissue_threshold:.3f})")
            logger.info(f"{version} 조직 질감 보존 비율: 상위 {preservation_percentage}% ({strong_count}개 픽셀)")
            logger.info(f"{version} 나머지 모든 영역 완전 보존: {total_tissue_count - strong_count}개 조직 픽셀 + 비조직 영역 모두 복원 결과 유지")
        else:
            logger.info(f"{version} {percentile_threshold}th percentile 이상 강한 조직 없음 - 모든 영역을 복원 결과 그대로 유지")

        logger.info(f"{version} 버전별 차등 처리: {percentile_threshold}th percentile 이상만 원본 픽셀값 직접 적용, 나머지 전부 복원 결과 유지")

        # 값 범위 클램핑
        enhanced_np = np.clip(enhanced_np, 0, 1)

        # torch.Tensor로 변환하여 반환
        return torch.from_numpy(enhanced_np).float()

    def _restore_original_colors(self, restored_image, tissue_mask, blind_zone_mask, original_image, version):
        """
        Restore original pixel values in non-blind-zone areas using smooth alpha blending
        to prevent hard edges.
        """
        if blind_zone_mask is None:
            logger.info("블라인드존 마스크가 없어 원본 색상 복원을 건너뜁니다")
            return restored_image

        # Ensure tensors are on the same device
        if not isinstance(restored_image, torch.Tensor):
            restored_image = torch.from_numpy(restored_image)
        restored_image = restored_image.to(self.device)

        if not isinstance(original_image, torch.Tensor):
            original_image = torch.from_numpy(original_image)
        original_image = original_image.to(self.device)

        # Ensure mask is a numpy array for cv2 processing
        if isinstance(blind_zone_mask, torch.Tensor):
            blind_zone_np = blind_zone_mask.cpu().numpy()
        else:
            blind_zone_np = blind_zone_mask

        # Ensure correct shape for processing (handle batches)
        if len(restored_image.shape) == 4: # B, C, H, W
             restored_image = restored_image.squeeze(0) # C, H, W
        if len(original_image.shape) == 4:
             original_image = original_image.squeeze(0)
        if len(blind_zone_np.shape) == 3:
            blind_zone_np = blind_zone_np[0]

        # Blur the blind zone mask to create a smooth alpha mask for blending.
        if cv2 is not None:
            # A larger kernel/sigma creates a more gradual transition.
            alpha_mask_np = cv2.GaussianBlur(blind_zone_np.astype(np.float32), (21, 21), 7)
        else:
            logger.warning("cv2 not available, blending will have a hard edge.")
            alpha_mask_np = blind_zone_np.astype(np.float32)

        # Convert the smoothed numpy mask back to a tensor on the correct device
        alpha_mask = torch.from_numpy(alpha_mask_np).to(self.device)

        # Ensure alpha_mask has the same dimensions as the images for broadcasting (C, H, W)
        if len(alpha_mask.shape) < len(restored_image.shape):
             alpha_mask = alpha_mask.unsqueeze(0)

        # Alpha blend:
        # Where alpha_mask is 1 (center of blind zone), use restored_image.
        # Where alpha_mask is 0 (far from blind zone), use original_image.
        blended_image = restored_image * alpha_mask + original_image * (1.0 - alpha_mask)

        logger.info(f"{version} 원본 색상 부드럽게 복원 완료 (Alpha Blending).")

        # Clamp final values and return as tensor
        return blended_image.clamp(0, 1)
    def _load_test_images(self, test_path, optuna_mode=False, reference_path=None):
        """Load test images with target-reference subtraction preprocessing"""
        test_path = Path(test_path)
        images = []

        if optuna_mode:
            # Optuna 모드: GT 패턴에 해당하는 10개 이미지만 로드
            target_patterns = [
                'CY_OY_PC_D000_V3_001.bmp', 'CY_OY_PC_D000_V3_201.bmp',
                'CY_OY_PC_D000_V4_001.bmp', 'CY_OY_PC_D000_V4_201.bmp',
                'CY_OY_PC_D000_V5_001.bmp', 'CY_OY_PC_D000_V5_201.bmp',
                'CY_OY_PC_D000_V6_001.bmp', 'CY_OY_PC_D000_V6_201.bmp',
                'CY_OY_PC_D000_V7_001.bmp', 'CY_OY_PC_D000_V7_201.bmp'
            ]

            for pattern in target_patterns:
                img_path = test_path / pattern
                if img_path.exists():
                    img = self._load_and_preprocess_image(img_path, reference_path)
                    if img is not None:
                        images.append((img_path, img))
                else:
                    logger.warning(f"Optuna target image not found: {img_path}")

            logger.info(f"Optuna mode: Loaded {len(images)}/10 target images")

        elif test_path.is_file():
            # Single image
            img = self._load_and_preprocess_image(test_path, reference_path)
            if img is not None:
                images.append((test_path, img))
        else:
            # Directory - 모든 이미지 로드
            for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg']:
                for img_path in test_path.glob(ext):
                    img = self._load_and_preprocess_image(img_path, reference_path)
                    if img is not None:
                        images.append((img_path, img))

        return sorted(images, key=lambda x: x[0].name)

    def _load_and_preprocess_image(self, target_path, reference_path=None):
        """
        Load and preprocess image with donut-based registration and Otsu blind zone detection

        Process:
        1. 도넛형태 블라인드존 영역 기준으로 target과 reference 정합
        2. 정합 알고리즘 (각도 회전, 이동)으로 블라인드존 형태 매칭
        3. Diff map 계산 후 도넛형태 안에서 픽셀값 반전
        4. Otsu 알고리즘으로 블라인드존만 정확히 검출
        5. Test 이미지에서 해당 부분만 제거
        """
        try:
            # Load target image
            target_img = Image.open(target_path).convert('L').resize((512, 512))
            target_array = np.array(target_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            # If no reference path provided, return original
            if reference_path is None:
                logger.info(f"No reference path - using original image: {target_path.name}")
                return target_img

            # Find corresponding reference image
            reference_path = Path(reference_path)
            target_filename = target_path.name

            # Convert target filename pattern to reference pattern
            # Target: CY_ON_PL_D000_V3_001.bmp -> Reference: CY_ON_PC_DC_V3_001.bmp
            # Extract version and ID from target filename
            import re

            # Parse target filename: CY_ON_PL_D000_V3_001.bmp
            target_match = re.search(r'CY_ON_PL_D\d+_V(\d+)_(\d+)\.bmp', target_filename)
            if target_match:
                version = target_match.group(1)  # V3, V4, etc.
                image_id = target_match.group(2)  # 001, 201, etc.
                ref_filename = f"CY_ON_PC_DC_V{version}_{image_id}.bmp"
            else:
                # Fallback to original logic if pattern doesn't match
                ref_filename = target_filename.replace('_PL_', '_')

            # Look for reference image
            ref_img_path = reference_path / ref_filename

            if not ref_img_path.exists():
                # Try alternative patterns for backward compatibility
                alternative_patterns = [
                    target_filename.replace('CY_ON_PL_', 'CY_ON_PC_DC_'),  # Direct replacement
                    target_filename.replace('_PL_D000_', '_PC_DC_'),        # Remove angle info
                    target_filename.replace('_PL_D045_', '_PC_DC_'),
                    target_filename.replace('_PL_D270_', '_PC_DC_'),
                    target_filename.replace('_PL_D315_', '_PC_DC_'),
                ]

                found = False
                for alt_pattern in alternative_patterns:
                    alt_path = reference_path / alt_pattern
                    if alt_path.exists():
                        ref_img_path = alt_path
                        found = True
                        break

                if not found:
                    logger.warning(f"Reference image not found for {target_filename}, using original")
                    return target_img

            # Load reference image
            ref_img = Image.open(ref_img_path).convert('L').resize((512, 512))
            ref_array = np.array(ref_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            # Get donut region parameters based on version
            version_str = f"V{version}"
            donut_params = self._get_donut_parameters(version_str)

            # Step 1: Create donut mask for registration
            donut_mask = self._create_donut_mask(512, 512, donut_params)

            # Step 2: Registration - check cache first, then compute if needed
            cached_result = self._get_cached_registration(target_filename, ref_img_path.name)

            if cached_result is not None:
                best_rotation, best_translation = cached_result
            else:
                logger.info("Computing new registration...")
                best_rotation, best_translation = self._register_donut_regions(
                    target_array, ref_array, donut_mask
                )
                # Cache the result
                self._cache_registration_result(target_filename, ref_img_path.name, best_rotation, best_translation)

            # Step 3: Apply transformation to reference
            aligned_ref = self._apply_transformation(ref_array, best_rotation, best_translation)

            # Step 4: Calculate diff map in donut region
            diff_map = target_array - aligned_ref

            # Step 5: Pixel inversion in donut region for better contrast
            donut_diff = diff_map * donut_mask
            inverted_donut_diff = (1.0 - donut_diff) * donut_mask

            # Step 6: Apply Otsu thresholding to detect blind zones
            blind_zone_mask = self._otsu_blind_zone_detection(inverted_donut_diff, donut_mask)

            # Step 7: Keep test image unchanged, save blind zone mask for H_funcs
            # Store the blind zone mask for later use in distortion operators
            self._store_blind_zone_mask(target_path.name, blind_zone_mask)

            # Return original target image unchanged
            preprocessed_img = target_img

            # Log detailed statistics
            blind_zone_ratio = np.sum(blind_zone_mask > 0) / blind_zone_mask.size * 100
            donut_coverage = np.sum(donut_mask > 0) / donut_mask.size * 100

            logger.info(f"=== DONUT-BASED REGISTRATION & OTSU BLIND ZONE DETECTION ===")
            logger.info(f"Target: {target_path.name}")
            logger.info(f"Reference: {ref_img_path.name}")
            logger.info(f"Version: {version_str}, Donut coverage: {donut_coverage:.1f}%")
            logger.info(f"Registration result: rotation={best_rotation:.1f}°, translation=({best_translation[0]:.1f}, {best_translation[1]:.1f})")
            logger.info(f"Blind zone detected: {blind_zone_ratio:.1f}% of image")
            logger.info(f"Original image preserved for DDRM processing")
            logger.info(f"Target image range: [{target_array.min():.3f}, {target_array.max():.3f}]")
            logger.info(f"=================================================================")

            return preprocessed_img

        except Exception as e:
            logger.error(f"Failed to preprocess {target_path}: {e}")
            # Fallback to original image
            try:
                return Image.open(target_path).convert('L').resize((512, 512))
            except:
                return None

    def _get_donut_parameters(self, version):
        """Get donut parameters based on version"""
        donut_params = {
            'V3': {'inner': 42, 'outer': 230},
            'V4': {'inner': 25, 'outer': 133},
            'V5': {'inner': 17, 'outer': 90},
            'V6': {'inner': 11, 'outer': 63},
            'V7': {'inner': 9, 'outer': 48}
        }
        return donut_params.get(version, {'inner': 25, 'outer': 133})  # Default to V4

    def _create_donut_mask(self, height, width, params):
        """Create donut-shaped mask"""
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        inner_radius = params['inner']
        outer_radius = params['outer']

        donut_mask = ((distance >= inner_radius) & (distance <= outer_radius)).astype(np.float32)
        return donut_mask

    def _register_donut_regions(self, target, reference, donut_mask):
        """Find best alignment between target and reference in donut region"""
        from scipy import ndimage
        from scipy.optimize import minimize_scalar

        # Extract donut regions
        target_donut = target * donut_mask
        ref_donut = reference * donut_mask

        best_rotation = 0
        best_translation = (0, 0)
        best_score = float('inf')

        # Search rotation angles
        rotation_angles = np.arange(-15, 16, 3)  # -15 to 15 degrees in 3-degree steps

        for angle in rotation_angles:
            # Rotate reference
            rotated_ref = ndimage.rotate(ref_donut, angle, reshape=False, order=1)

            # Search translations
            for dx in range(-10, 11, 2):  # -10 to 10 pixels in 2-pixel steps
                for dy in range(-10, 11, 2):
                    # Translate rotated reference
                    translated_ref = ndimage.shift(rotated_ref, (dy, dx), order=1)
                    translated_ref = translated_ref * donut_mask  # Re-apply mask

                    # Calculate similarity score (MSE)
                    diff = target_donut - translated_ref
                    score = np.mean(diff[donut_mask > 0]**2)

                    if score < best_score:
                        best_score = score
                        best_rotation = angle
                        best_translation = (dx, dy)

        return best_rotation, best_translation

    def _apply_transformation(self, image, rotation, translation):
        """Apply rotation and translation to image"""
        from scipy import ndimage

        # Apply rotation
        if rotation != 0:
            transformed = ndimage.rotate(image, rotation, reshape=False, order=1)
        else:
            transformed = image.copy()

        # Apply translation
        if translation != (0, 0):
            transformed = ndimage.shift(transformed, (translation[1], translation[0]), order=1)

        return transformed

    def _otsu_blind_zone_detection(self, inverted_diff, donut_mask):
        """Apply Otsu thresholding to detect blind zones with brightness normalization"""
        from skimage import filters, exposure

        # Get pixels within donut region
        donut_pixels = inverted_diff[donut_mask > 0]

        if len(donut_pixels) == 0:
            return np.zeros_like(inverted_diff)

        # Normalize brightness distribution within donut region to improve Otsu performance
        # Use histogram equalization to reduce brightness differences
        donut_region = inverted_diff * donut_mask

        # Apply adaptive histogram equalization within donut region
        if np.max(donut_pixels) > np.min(donut_pixels):  # Check if there's variation
            # Normalize to 0-1 range first
            normalized_pixels = (donut_pixels - donut_pixels.min()) / (donut_pixels.max() - donut_pixels.min())

            # Apply histogram equalization
            equalized_pixels = exposure.equalize_hist(normalized_pixels)

            # Create equalized image
            equalized_diff = np.zeros_like(inverted_diff)
            equalized_diff[donut_mask > 0] = equalized_pixels

            # Apply Otsu on equalized data
            threshold = filters.threshold_otsu(equalized_pixels)

            # --- START MODIFICATION ---
            # Adjust threshold to detect more blind zone area
            adjusted_threshold = threshold * 0.85
            logger.info(f"Otsu threshold: original={threshold:.3f}, adjusted={adjusted_threshold:.3f}")
            binary_mask = (equalized_diff > adjusted_threshold).astype(np.float32)

            # Apply only within donut region
            blind_zone_mask = binary_mask * donut_mask

            # Filter out small, noisy components from the blind zone mask
            if cv2 is not None:
                H, W = blind_zone_mask.shape[:2]
                min_area_ratio = 0.001 # Ratio to remove small components
                min_area = max(1, int(H * W * max(0.0, min_area_ratio)))

                u8 = (blind_zone_mask > 0.5).astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)

                if num > 1:
                    clean_mask = np.zeros_like(u8)
                    for i in range(1, num):
                        if stats[i, cv2.CC_STAT_AREA] >= min_area:
                            clean_mask[labels == i] = 1
                    blind_zone_mask = clean_mask.astype(np.float32)
                    logger.info(f"Filtered blind zone mask, keeping components with area >= {min_area}")
            # --- END MODIFICATION ---

        else:
            # Fallback to original method if no variation
            threshold = filters.threshold_otsu(donut_pixels)
            binary_mask = (inverted_diff > threshold).astype(np.float32)
            logger.info(f"Otsu original method: threshold={threshold:.3f}")
            # Apply only within donut region
            blind_zone_mask = binary_mask * donut_mask

        return blind_zone_mask

    def _store_blind_zone_mask(self, filename, mask):
        """Store blind zone mask for later use in H_funcs"""
        if not hasattr(self, 'blind_zone_masks'):
            self.blind_zone_masks = {}
        self.blind_zone_masks[filename] = mask
        logger.info(f"Stored blind zone mask for {filename}: {np.sum(mask > 0) / mask.size * 100:.1f}% coverage")

    def get_blind_zone_mask(self, filename):
        """Get stored blind zone mask for H_funcs"""
        if hasattr(self, 'blind_zone_masks') and filename in self.blind_zone_masks:
            return self.blind_zone_masks[filename]
        return None

    def _load_registration_cache(self):
        """Load registration cache from file"""
        try:
            import json
            import shutil
            if self.cache_file.exists():
                cache_size = self.cache_file.stat().st_size
                if cache_size == 0:
                    logger.warning(f"Cache file is empty: {self.cache_file}")
                    self.registration_cache = {}
                    return

                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Validate cache data structure
                if not isinstance(cache_data, dict):
                    logger.warning(f"Invalid cache file format: expected dict, got {type(cache_data)}")
                    self.registration_cache = {}
                    return

                # Validate each cache entry
                valid_entries = {}
                for key, value in cache_data.items():
                    if (
                        isinstance(value, dict) and
                        'rotation' in value and 'translation' in value and
                        isinstance(value['rotation'], (int, float)) and
                        isinstance(value['translation'], (list, tuple))):
                        valid_entries[key] = value
                    else:
                        logger.warning(f"Invalid cache entry format for {key}: {value}")

                self.registration_cache = valid_entries
                logger.info(f"Loaded registration cache: {len(self.registration_cache)} entries")

                if len(valid_entries) != len(cache_data):
                    logger.info(f"Cleaned {len(cache_data) - len(valid_entries)} invalid cache entries")
                    # Save the cleaned cache
                    self._save_registration_cache()
            else:
                logger.info("No existing registration cache found, starting fresh")
                self.registration_cache = {}
        except json.JSONDecodeError as je:
            logger.warning(f"Failed to parse registration cache JSON: {je}")
            logger.info("Backing up corrupted cache and starting fresh")
            try:
                backup_path = self.cache_file.with_suffix('.corrupted.bak')
                shutil.move(str(self.cache_file), str(backup_path))
                logger.info(f"Corrupted cache backed up to: {backup_path}")
            except Exception as backup_error:
                logger.warning(f"Failed to backup corrupted cache: {backup_error}")
            self.registration_cache = {}
        except Exception as e:
            logger.warning(f"Failed to load registration cache: {e}")
            self.registration_cache = {}

    def _save_registration_cache(self):
        """Save registration cache to file"""
        try:
            import json
            import tempfile
            import shutil
            import os

            # Validate cache data before saving
            for key, value in self.registration_cache.items():
                if not isinstance(value, dict):
                    logger.warning(f"Invalid cache entry type for {key}: {type(value)}")
                    continue
                if 'rotation' not in value or 'translation' not in value:
                    logger.warning(f"Missing required fields in cache entry for {key}")
                    continue
                if not isinstance(value['rotation'], (int, float)):
                    logger.warning(f"Invalid rotation type for {key}: {type(value['rotation'])}")
                    continue
                if not isinstance(value['translation'], (list, tuple)):
                    logger.warning(f"Invalid translation type for {key}: {type(value['translation'])}")
                    continue

            # Write to temporary file first, then move to avoid corruption
            temp_file = self.cache_file.with_suffix('.tmp')

            # Ensure directory exists
            temp_file.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_file, 'w') as f:
                json.dump(self.registration_cache, f, indent=2)

            # Verify temp file was written correctly
            if not temp_file.exists():
                raise FileNotFoundError(f"Temporary cache file was not created: {temp_file}")

            temp_size = temp_file.stat().st_size
            if temp_size == 0:
                raise ValueError(f"Temporary cache file is empty: {temp_file}")

            # Verify JSON can be read back
            try:
                with open(temp_file, 'r') as f:
                    test_load = json.load(f)
                logger.info(f"Cache validation successful: {len(test_load)} entries")
            except json.JSONDecodeError as je:
                raise ValueError(f"Invalid JSON in temporary cache file: {je}")

            # Atomically replace the cache file
            shutil.move(str(temp_file), str(self.cache_file))

            # Verify final file exists
            if not self.cache_file.exists():
                raise FileNotFoundError(f"Cache file was not created after move: {self.cache_file}")

            logger.info(f"Saved registration cache: {len(self.registration_cache)} entries to {self.cache_file}")

        except Exception as e:
            logger.error(f"Failed to save registration cache: {e}")
            # Clean up temp file if it exists
            temp_file = self.cache_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary cache file: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")

    def _get_cache_key(self, target_filename, ref_filename):
        """Generate cache key for target-reference pair"""
        return f"{target_filename}_{ref_filename}"

    def _get_cached_registration(self, target_filename, ref_filename):
        """Get cached registration result"""
        cache_key = self._get_cache_key(target_filename, ref_filename)
        if cache_key in self.registration_cache:
            result = self.registration_cache[cache_key]
            logger.info(f"Using cached registration for {target_filename}: rotation={result['rotation']}°, translation={result['translation']}")
            return result['rotation'], tuple(result['translation'])
        return None

    def _cache_registration_result(self, target_filename, ref_filename, rotation, translation):
        """Cache registration result"""
        cache_key = self._get_cache_key(target_filename, ref_filename)

        # Ensure proper data types for JSON serialization
        if hasattr(rotation, 'item'):  # numpy scalar
            rotation = rotation.item()
        elif hasattr(rotation, 'dtype'):  # numpy array
            rotation = float(rotation)
        else:
            rotation = float(rotation)

        # Convert translation to list of floats
        if hasattr(translation, 'tolist'):  # numpy array
            translation = translation.tolist()
        else:
            translation = [float(x) for x in translation]

        self.registration_cache[cache_key] = {
            'rotation': rotation,
            'translation': translation
        }
        logger.info(f"Cached registration for {target_filename}: rotation={rotation}°, translation={translation}")
        self._save_registration_cache()

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

    def _save_individual_artifacts(self, filename, H_funcs, output_dir):
        """Save individual image's z_est, H_est, and detected blind zone mask"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract base filename without extension
        base_name = Path(filename).stem

        # Save z_est (noise pattern) if available
        if hasattr(H_funcs, 'noise_pattern') and H_funcs.noise_pattern is not None:
            z_est_path = output_dir / f"z_est_{base_name}.npy"
            if isinstance(H_funcs.noise_pattern, torch.Tensor):
                np.save(z_est_path, H_funcs.noise_pattern.cpu().numpy())
            else:
                np.save(z_est_path, H_funcs.noise_pattern)
            logger.info(f"Saved z_est for {filename}")

        # Save H_est (distortion map) if available
        if hasattr(H_funcs, 'mask_tensor') and H_funcs.mask_tensor is not None:
            h_est_path = output_dir / f"H_est_{base_name}.npy"
            if isinstance(H_funcs.mask_tensor, torch.Tensor):
                np.save(h_est_path, H_funcs.mask_tensor.cpu().numpy())
            else:
                np.save(h_est_path, H_funcs.mask_tensor)
            logger.info(f"Saved H_est for {filename}")

        # Save detected blind zone mask if available
        if hasattr(H_funcs, 'detected_blind_zone_mask') and H_funcs.detected_blind_zone_mask is not None:
            mask_path = output_dir / f"blind_zone_mask_{base_name}.npy"
            np.save(mask_path, H_funcs.detected_blind_zone_mask)
            logger.info(f"Saved blind zone mask for {filename}")

            # Also save as PNG for visualization
            mask_png_path = output_dir / f"blind_zone_mask_{base_name}.png"
            mask_img = (H_funcs.detected_blind_zone_mask * 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(mask_img, mode='L').save(mask_png_path)

        # Save additional distortion information if available
        if hasattr(H_funcs, 'distortion_mask') and H_funcs.distortion_mask is not None:
            distortion_path = output_dir / f"distortion_mask_{base_name}.npy"
            np.save(distortion_path, H_funcs.distortion_mask)
            logger.info(f"Saved distortion mask for {filename}")

    def _save_tissue_analysis(self, x_orig, y_corrupted, version, image_path, output_dir, H_funcs):
        """
        조직 보호 분석 시각화 저장
        - 원본 이미지
        - 조직 검출 마스크
        - 블라인드존 마스크
        - 보호 영역 표시
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            # 이미지를 numpy로 변환
            orig_np = x_orig.squeeze().cpu().numpy()

            # 검출된 블라인드존 마스크 사용 (도넛 기반 + Otsu)
            if hasattr(H_funcs, 'detected_blind_zone_mask') and H_funcs.detected_blind_zone_mask is not None:
                logger.info("Using detected blind zone mask for tissue analysis")
                detected_blind_zone = H_funcs.detected_blind_zone_mask

                # 기존 조직/블라인드존 마스크와 검출된 마스크 비교
                if H_funcs.tissue_mask is not None:
                    tissue_mask = H_funcs.tissue_mask
                    blind_zone_mask = H_funcs.blind_zone_processing_mask

                    # 전체 블라인드존 마스크 (원본 도넛 영역)
                    full_blind_zone = (H_funcs.distortion_mask > 0.1).astype(np.float32)

                    # 보호된 조직 영역 (조직 ∩ 검출되지 않은 영역)
                    protected_tissue = tissue_mask * (1.0 - detected_blind_zone) * full_blind_zone
                else:
                    # 조직 마스크가 없으면 간단한 분석
                    tissue_mask = np.zeros_like(detected_blind_zone)
                    blind_zone_mask = detected_blind_zone
                    full_blind_zone = detected_blind_zone
                    protected_tissue = np.zeros_like(detected_blind_zone)

                logger.info(f"Detected blind zone coverage: {np.sum(detected_blind_zone > 0) / detected_blind_zone.size * 100:.1f}%")
            else:
                # Fallback to original method
                if H_funcs.tissue_mask is None or H_funcs.blind_zone_processing_mask is None:
                    logger.warning("No tissue mask or detected blind zone mask available")
                    return

                tissue_mask = H_funcs.tissue_mask
                blind_zone_mask = H_funcs.blind_zone_processing_mask
                full_blind_zone = (H_funcs.distortion_mask > 0.1).astype(np.float32)
                protected_tissue = tissue_mask * full_blind_zone
                detected_blind_zone = blind_zone_mask

            # 3x4 subplot 생성
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            fig.suptitle(f'Tissue Protection Analysis - {version} - {image_path.stem}', fontsize=16)

            # Row 1: 기본 이미지들
            axes[0,0].imshow(orig_np, cmap='gray')
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')

            axes[0,1].imshow(tissue_mask, cmap='Reds', alpha=0.8)
            axes[0,1].imshow(orig_np, cmap='gray', alpha=0.5)
            axes[0,1].set_title('Tissue Detection (Bright Regions)')
            axes[0,1].axis('off')

            axes[0,2].imshow(full_blind_zone, cmap='Blues', alpha=0.8)
            axes[0,2].imshow(orig_np, cmap='gray', alpha=0.5)
            axes[0,2].set_title(f'{version} Full Blind Zone')
            axes[0,2].axis('off')

            axes[0,3].imshow(protected_tissue, cmap='Greens', alpha=0.8)
            axes[0,3].imshow(orig_np, cmap='gray', alpha=0.5)
            axes[0,3].set_title('Protected Tissue Area')
            axes[0,3].axis('off')

            # Row 2: 처리 영역들
            axes[1,0].imshow(detected_blind_zone, cmap='Reds', alpha=0.8)
            axes[1,0].imshow(orig_np, cmap='gray', alpha=0.5)
            axes[1,0].set_title(f'Detected Blind Zone ({np.sum(detected_blind_zone > 0) / detected_blind_zone.size * 100:.1f}%)')
            axes[1,0].axis('off')

            # 배경 영역
            background_mask = 1.0 - tissue_mask - blind_zone_mask
            background_mask = np.clip(background_mask, 0, 1)
            axes[1,1].imshow(background_mask, cmap='Greys', alpha=0.6)
            axes[1,1].imshow(orig_np, cmap='gray', alpha=0.5)
            axes[1,1].set_title('Background Region (Minimal Processing)')
            axes[1,1].axis('off')

            # 처리 강도 맵
            processing_strength = np.zeros_like(orig_np)
            processing_strength += tissue_mask * 0.3  # 조직: 30% 강도
            processing_strength += blind_zone_mask * 2.0  # 순수 블라인드존: 200% 강도
            processing_strength += background_mask * 0.1  # 배경: 10% 강도

            im = axes[1,2].imshow(processing_strength, cmap='plasma', alpha=0.8)
            axes[1,2].imshow(orig_np, cmap='gray', alpha=0.4)
            axes[1,2].set_title('Processing Strength Map')
            axes[1,2].axis('off')
            plt.colorbar(im, ax=axes[1,2], shrink=0.6)

            # 가시 영역 shift 정보 가져오기
            # 중심점 고정 (angle offset 제거)

            # 기본 버전 정보
            version_info = {
                'V3': 'Large Blind Zone\nStrong Distortion',
                'V4': 'Med-Large Blind Zone\nMedium Distortion',
                'V5': 'Medium Blind Zone\nStandard Distortion',
                'V6': 'Small Blind Zone\nWeak Distortion',
                'V7': 'Minimal Blind Zone\nFine Distortion'
            }

            version_text = f"""
{version} Characteristics:

{version_info.get(version, 'Unknown Version')}

Processing Method:
✓ Tissue protection activated
✓ Differential correction strength
✓ Progressive processing
✓ Center fixed at image center"""

            axes[1,3].text(0.05, 0.95, version_text, transform=axes[1,3].transAxes,
                          verticalalignment='top', fontsize=11,
                          bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))
            axes[1,3].set_title(f'{version} Processing Info')
            axes[1,3].axis('off')

            # 보호 전략 오버레이
            combined_overlay = np.zeros((*orig_np.shape, 3))
            combined_overlay[:,:,0] = tissue_mask * 0.7  # 빨간색: 조직 (보호)
            combined_overlay[:,:,1] = blind_zone_mask * 0.7  # 초록색: 블라인드존 (제거)
            combined_overlay[:,:,2] = protected_tissue * 0.9  # 파란색: 보호된 조직

            axes[2,0].imshow(orig_np, cmap='gray')
            axes[2,0].imshow(combined_overlay, alpha=0.6)
            axes[2,0].set_title('Strategy Overlay\nRed:Tissue Protection, Green:Blind Zone Removal')
            axes[2,0].axis('off')

            # Row 3: 통계 및 정보
            tissue_coverage = np.sum(tissue_mask) / tissue_mask.size * 100
            blind_zone_coverage = np.sum(full_blind_zone) / full_blind_zone.size * 100
            protected_coverage = np.sum(protected_tissue) / protected_tissue.size * 100
            pure_blind_coverage = np.sum(blind_zone_mask) / blind_zone_mask.size * 100

            stats_text = f"""
Tissue Protection Statistics:

• Detected Tissue: {tissue_coverage:.1f}%
• Full Blind Zone: {blind_zone_coverage:.1f}%
• Protected Tissue: {protected_coverage:.1f}%
• Pure Blind Zone: {pure_blind_coverage:.1f}%

Spatial Configuration:
• Center fixed at image center (256, 256)
• Detection center fixed

Processing Strategy:
→ Tissue Area: 20-60% gentle correction
→ Pure Blind Zone: 120-200% strong correction
→ Background Area: 10% minimal correction

Protection Effect:
→ Tissue Loss Prevention: {protected_coverage:.1f}%
→ Blind Zone Removal: {pure_blind_coverage:.1f}%"""

            axes[2,1].text(0.05, 0.95, stats_text, transform=axes[2,1].transAxes,
                          verticalalignment='top', fontsize=11,
                          bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
            axes[2,1].set_title('Protection Statistics')
            axes[2,1].axis('off')

            # 처리 강도 히스토그램
            axes[2,2].hist(processing_strength.flatten(), bins=50, alpha=0.7, color='purple')
            axes[2,2].set_title('Processing Strength Distribution')
            axes[2,2].set_xlabel('Processing Strength')
            axes[2,2].set_ylabel('Pixel Count')

            # 가시 영역 시각화 (수정된 shift 적용)
            height, width = orig_np.shape
            center_y, center_x = height // 2, width // 2

            # 가시 영역 원 생성 (중심 고정)
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            version_visible_radius = {
                'V3': 240, 'V4': 150, 'V5': 110, 'V6': 80, 'V7': 65
            }
            visible_radius = version_visible_radius.get(version, 110)
            visible_region = (distance <= visible_radius).astype(np.float32)

            # 가시 영역과 도넛 영역을 함께 표시
            axes[2,3].imshow(orig_np, cmap='gray', alpha=0.6)
            axes[2,3].contour(visible_region, levels=[0.5], colors=['cyan'], linewidths=2, linestyles='-')
            axes[2,3].contour(full_blind_zone, levels=[0.1], colors=['blue'], linewidths=1, linestyles='--')

            # 중심점 표시 (고정된 중심점만)
            axes[2,3].plot(center_x, center_y, 'ro', markersize=8, label='Fixed Center')

            axes[2,3].set_title(f'Fixed Center Processing\nCyan: Visible Circle, Blue: Blind Zone')
            axes[2,3].legend(loc='upper right', fontsize=8)
            axes[2,3].axis('off')

            # 저장
            analysis_path = output_dir / f"{image_path.stem}_tissue_analysis_{version}.png"
            plt.tight_layout()
            plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"조직 분석 시각화 저장: {analysis_path}")
            logger.info(f"  - 조직 보호: {protected_coverage:.1f}%, 블라인드존 제거: {pure_blind_coverage:.1f}%")

        except Exception as e:
            logger.error(f"조직 분석 시각화 실패: {e}")
            # 기본 마스크 저장으로 폴백
            if hasattr(H_funcs, 'tissue_mask') and H_funcs.tissue_mask is not None:
                tissue_path = output_dir / f"{image_path.stem}_tissue_mask_{version}.png"
                tvu.save_image(torch.from_numpy(H_funcs.tissue_mask), tissue_path)
                logger.info(f"기본 조직 마스크 저장: {tissue_path}")


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
        # Parse save_steps if provided
        save_steps = None
        if hasattr(args, 'save_steps') and args.save_steps:
            try:
                save_steps = [int(x.strip()) for x in args.save_steps.split(',')]
                logger.info(f"Will save intermediate steps: {save_steps}")
            except ValueError:
                logger.warning(f"Invalid save_steps format: {args.save_steps}. Expected comma-separated integers.")

        results = runner.sample_ultrasound_sequence(
            args.test_images_path,
            args.image_folder,
            sigma_0=getattr(args, 'sigma_0', 0.05),
            save_steps=save_steps,
            optuna_mode=getattr(args, 'optuna_mode', False),
            no_save_images=getattr(args, 'no_save_images', False),
            reference_path=getattr(args, 'cy_on_path', None)
        )

        logger.info(f"Processed {len(results)} images successfully")
        return results
    else:
        logger.warning("No test images path provided")
        return []

if __name__ == "__main__":
    # This would be called from main script
    pass
