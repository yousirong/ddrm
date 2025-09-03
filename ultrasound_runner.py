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
            z_est, distortion_map = estimate_version_artifacts(cn_on_path, cy_on_path, version, custom_threshold)

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
        distortion_factor = getattr(self.args, 'distortion_factor', 0.025)
        noise_factor = getattr(self.args, 'noise_factor', 1.0)

        # Get V3~V7 도넛 기반 파라미터들
        version_thresholds = {}
        if hasattr(self.args, 'v3_tissue_percentile'):
            version_thresholds['V3'] = {
                'tissue_percentile': self.args.v3_tissue_percentile,
                'blind_zone_percentile': getattr(self.args, 'v3_blind_zone_percentile', 35)
            }
        if hasattr(self.args, 'v4_tissue_percentile'):
            version_thresholds['V4'] = {
                'tissue_percentile': self.args.v4_tissue_percentile,
                'blind_zone_percentile': getattr(self.args, 'v4_blind_zone_percentile', 40)
            }
        if hasattr(self.args, 'v5_tissue_percentile'):
            version_thresholds['V5'] = {
                'tissue_percentile': self.args.v5_tissue_percentile,
                'blind_zone_percentile': getattr(self.args, 'v5_blind_zone_percentile', 45)
            }
        if hasattr(self.args, 'v6_tissue_percentile'):
            version_thresholds['V6'] = {
                'tissue_percentile': self.args.v6_tissue_percentile,
                'blind_zone_percentile': getattr(self.args, 'v6_blind_zone_percentile', 50)
            }
        if hasattr(self.args, 'v7_tissue_percentile'):
            version_thresholds['V7'] = {
                'tissue_percentile': self.args.v7_tissue_percentile,
                'blind_zone_percentile': getattr(self.args, 'v7_blind_zone_percentile', 55)
            }

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
            version_thresholds, tissue_min_size, blind_zone_min_size
        )

    def sample_ultrasound_sequence(self, test_images_path, output_dir, sigma_0=0.05, save_steps=None, optuna_mode=False, no_save_images=False):
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
        test_images = self._load_test_images(test_images_path, optuna_mode)
        logger.info(f"Loaded {len(test_images)} test images{' (Optuna mode - 10 images only)' if optuna_mode else ''}")

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

                # Apply tissue enhancement post-processing - 조직을 원본처럼 밝게 복원
                #if H_funcs.tissue_mask is not None:
                #    restored_tensor = self._enhance_tissue_pixels(restored_tensor, H_funcs.tissue_mask, x_orig.squeeze().cpu(), version)

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
        model_path = "/home/juneyonglee/Desktop/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100/best_model/unet"
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
    def _load_test_images(self, test_path, optuna_mode=False):
        """Load test images with version detection - Optuna 모드에서는 10개만 로드"""
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
                    img = Image.open(img_path).convert('L').resize((512, 512))
                    images.append((img_path, img))
                else:
                    logger.warning(f"Optuna target image not found: {img_path}")

            logger.info(f"Optuna mode: Loaded {len(images)}/10 target images")

        elif test_path.is_file():
            # Single image
            img = Image.open(test_path).convert('L').resize((512, 512))
            images.append((test_path, img))
        else:
            # Directory - 모든 이미지 로드
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

            # 조직 마스크, 블라인드존 마스크 가져오기
            if H_funcs.tissue_mask is None or H_funcs.blind_zone_processing_mask is None:
                logger.warning("조직 마스크가 없어서 시각화를 건너뜀")
                return

            tissue_mask = H_funcs.tissue_mask
            blind_zone_mask = H_funcs.blind_zone_processing_mask

            # 전체 블라인드존 마스크 (원본)
            full_blind_zone = (H_funcs.distortion_mask > 0.1).astype(np.float32)

            # 보호된 조직 영역 (조직 ∩ 전체블라인드존)
            protected_tissue = tissue_mask * full_blind_zone

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
            axes[1,0].imshow(blind_zone_mask, cmap='Oranges', alpha=0.8)
            axes[1,0].imshow(orig_np, cmap='gray', alpha=0.5)
            axes[1,0].set_title('Pure Blind Zone (Strong Processing)')
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

            # 기본 버전 정보
            version_info = {
                'V3': 'Large Blind Zone\nStrong Distortion',
                'V4': 'Med-Large Blind Zone\nMedium Distortion',
                'V5': 'Medium Blind Zone\nStandard Distortion',
                'V6': 'Small Blind Zone\nWeak Distortion',
                'V7': 'Minimal Blind Zone\nFine Distortion'
            }

            version_text = f"""{version} Characteristics:

{version_info.get(version, 'Unknown Version')}

Processing Method:
✓ Tissue protection activated
✓ Differential correction strength
✓ Progressive processing"""

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

            stats_text = f"""Tissue Protection Statistics:

• Detected Tissue: {tissue_coverage:.1f}%
• Full Blind Zone: {blind_zone_coverage:.1f}%
• Protected Tissue: {protected_coverage:.1f}%
• Pure Blind Zone: {pure_blind_coverage:.1f}%

Processing Strategy:
→ Tissue Area: 20-60% gentle correction
→ Pure Blind Zone: 120-200% strong correction
→ Background Area: 10% minimal correction

Protection Effect:
→ Tissue Loss Prevention: {protected_coverage:.1f}%
→ Blind Zone Removal: {pure_blind_coverage:.1f}%
"""

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

            # 마스크 겹침 분석
            overlap_data = {
                'Tissue Only': np.sum((tissue_mask > 0.5) & (full_blind_zone == 0)),
                'Blind Zone Only': np.sum((tissue_mask == 0) & (full_blind_zone > 0.5)),
                'Tissue+Blind Zone': np.sum((tissue_mask > 0.5) & (full_blind_zone > 0.5)),
                'Background': np.sum((tissue_mask == 0) & (full_blind_zone == 0))
            }

            axes[2,3].pie(overlap_data.values(), labels=overlap_data.keys(), autopct='%1.1f%%', startangle=90)
            axes[2,3].set_title('Region Distribution')

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
            no_save_images=getattr(args, 'no_save_images', False)
        )

        logger.info(f"Processed {len(results)} images successfully")
        return results
    else:
        logger.warning("No test images path provided")
        return []

if __name__ == "__main__":
    # This would be called from main script
    pass