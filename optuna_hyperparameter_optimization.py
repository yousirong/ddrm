#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization for Enhanced Ultrasound DDRM

Finds optimal V3~V7 donut-based tissue/blind zone separation parameters using
DDRM inference with SSIM evaluation against ground truth images.

Key optimization parameters:
- V3~V7 tissue/blind zone percentile thresholds for donut-based separation
- Natural restoration physics parameters (distortion/noise factors)
- Mask cleaning parameters (tissue/blind zone minimum sizes)
- DDRM sampling parameters (timesteps, eta, sigma_0)

Physics-based DDRM modeling principles:
- z_est = Average(CY_ON - CN_ON): Structural noise estimation
- H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²: Distortion operator estimation
- Natural restoration through H and H_pinv operators (no forced black/white)

Note: Runs actual inference but stores results in memory only (no image files saved).
"""

import os
import sys
import subprocess
import tempfile
import logging
import optuna
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from typing import Dict, List, Tuple
import shutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageMetrics:
    """Calculate various image similarity metrics"""
    
    @staticmethod
    def load_and_preprocess_image(image_path: str) -> np.ndarray:
        """Load and preprocess image for metric calculation"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try with PIL for different formats
            img = np.array(Image.open(image_path).convert('L'))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        return img
    
    @staticmethod
    def calculate_ssim(img1_path: str, img2_path: str) -> float:
        """Calculate Structural Similarity Index"""
        img1 = ImageMetrics.load_and_preprocess_image(img1_path)
        img2 = ImageMetrics.load_and_preprocess_image(img2_path)
        
        # Ensure same dimensions
        min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
        return ssim(img1, img2, data_range=1.0)
    
    @staticmethod
    def calculate_psnr(img1_path: str, img2_path: str) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        img1 = ImageMetrics.load_and_preprocess_image(img1_path)
        img2 = ImageMetrics.load_and_preprocess_image(img2_path)
        
        # Ensure same dimensions
        min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
        return psnr(img1, img2, data_range=1.0)
    
    @staticmethod
    def calculate_mse(img1_path: str, img2_path: str) -> float:
        """Calculate Mean Squared Error"""
        img1 = ImageMetrics.load_and_preprocess_image(img1_path)
        img2 = ImageMetrics.load_and_preprocess_image(img2_path)
        
        # Ensure same dimensions
        min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
        return np.mean((img1 - img2) ** 2)
    
    @staticmethod
    def calculate_mae(img1_path: str, img2_path: str) -> float:
        """Calculate Mean Absolute Error"""
        img1 = ImageMetrics.load_and_preprocess_image(img1_path)
        img2 = ImageMetrics.load_and_preprocess_image(img2_path)
        
        # Ensure same dimensions
        min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
        return np.mean(np.abs(img1 - img2))


class DDRMOptimizer:
    """Optuna-based hyperparameter optimizer for DDRM ultrasound model (inference with memory-only evaluation)"""
    
    def __init__(self, 
                 ddrm_base_path: str = "/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm",
                 gt_path: str = "/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm/datasets/test_CN_OY",
                 n_trials: int = 100):
        
        self.ddrm_base_path = Path(ddrm_base_path)
        self.gt_path = Path(gt_path)
        self.n_trials = n_trials
        self.script_path = self.ddrm_base_path / "run_ultrasound_ddrm.sh"
        
        # Create optimization results directory
        self.optimization_dir = self.ddrm_base_path / "optimization_results"
        self.optimization_dir.mkdir(exist_ok=True)
        
        # Version-specific processing based on filename patterns
        self.versions = ['V3', 'V4', 'V5', 'V6', 'V7']
        
        logger.info(f"Initialized DDRM Optimizer")
        logger.info(f"DDRM base path: {self.ddrm_base_path}")
        logger.info(f"Ground truth path: {self.gt_path}")
        logger.info(f"Optimization results: {self.optimization_dir}")
    
    def get_image_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of generated and ground truth images for evaluation - V3~V7 001,201 only"""
        pairs = []
        
        # Only V3~V7 with 001 and 201 patterns for SSIM evaluation
        gt_patterns = ['CN_OY_PC_D000_V3_001.bmp', 'CN_OY_PC_D000_V3_201.bmp', 
                      'CN_OY_PC_D000_V4_001.bmp', 'CN_OY_PC_D000_V4_201.bmp',
                      'CN_OY_PC_D000_V5_001.bmp', 'CN_OY_PC_D000_V5_201.bmp',
                      'CN_OY_PC_D000_V6_001.bmp', 'CN_OY_PC_D000_V6_201.bmp',
                      'CN_OY_PC_D000_V7_001.bmp', 'CN_OY_PC_D000_V7_201.bmp']
        
        for gt_pattern in gt_patterns:
            gt_file = self.gt_path / gt_pattern
            if gt_file.exists():
                # Corresponding generated file pattern
                base_name = gt_pattern.replace('CN_OY_', 'CY_OY_').replace('.bmp', '')
                version = 'V3' if 'V3' in gt_pattern else 'V4' if 'V4' in gt_pattern else 'V5' if 'V5' in gt_pattern else 'V6' if 'V6' in gt_pattern else 'V7'
                generated_file = f"{base_name}_restored_{version}.png"
                pairs.append((generated_file, str(gt_file)))
        
        logger.info(f"Found {len(pairs)} image pairs for SSIM evaluation (V3~V7 001,201 only)")
        return pairs
    
    def run_ddrm_inference(self, hyperparams: Dict) -> bool:
        """Run DDRM inference with given hyperparameters"""
        try:
            # Create environment variables for the script
            env = os.environ.copy()
            env.update({
                # DDRM sampling parameters
                'TIMESTEPS': str(hyperparams['timesteps']),
                'ETA': str(hyperparams['eta']),
                'SIGMA_0': str(hyperparams['sigma_0']),
                'DISTORTION_FACTOR': str(hyperparams['distortion_factor']),
                'NOISE_FACTOR': str(hyperparams['noise_factor']),
                
                # Legacy threshold parameters (kept for compatibility)
                'THRESHOLD_V3': str(hyperparams.get('threshold_v3', 0.00)),
                'THRESHOLD_V4': str(hyperparams.get('threshold_v4', 0.00)),
                'THRESHOLD_V5': str(hyperparams.get('threshold_v5', 0.00)),
                'THRESHOLD_V6': str(hyperparams.get('threshold_v6', 0.00)),
                'THRESHOLD_V7': str(hyperparams.get('threshold_v7', 0.00)),
                
                # V3~V7 donut-based tissue/blind zone separation parameters
                'V3_TISSUE_PERCENTILE': str(hyperparams['v3_tissue_percentile']),
                'V3_BLIND_ZONE_PERCENTILE': str(hyperparams['v3_blind_zone_percentile']),
                'V4_TISSUE_PERCENTILE': str(hyperparams['v4_tissue_percentile']),
                'V4_BLIND_ZONE_PERCENTILE': str(hyperparams['v4_blind_zone_percentile']),
                'V5_TISSUE_PERCENTILE': str(hyperparams['v5_tissue_percentile']),
                'V5_BLIND_ZONE_PERCENTILE': str(hyperparams['v5_blind_zone_percentile']),
                'V6_TISSUE_PERCENTILE': str(hyperparams['v6_tissue_percentile']),
                'V6_BLIND_ZONE_PERCENTILE': str(hyperparams['v6_blind_zone_percentile']),
                'V7_TISSUE_PERCENTILE': str(hyperparams['v7_tissue_percentile']),
                'V7_BLIND_ZONE_PERCENTILE': str(hyperparams['v7_blind_zone_percentile']),
                
                # Mask cleaning parameters
                'TISSUE_MIN_SIZE': str(hyperparams['tissue_min_size']),
                'BLIND_ZONE_MIN_SIZE': str(hyperparams['blind_zone_min_size']),
                
                # Natural restoration parameters
                'NATURAL_RESTORATION': 'true',
                'TISSUE_DISTORTION_FACTOR': str(hyperparams['tissue_distortion_factor']),
                'BLIND_ZONE_DISTORTION_FACTOR': str(hyperparams['blind_zone_distortion_factor']),
                'BACKGROUND_DISTORTION_FACTOR': str(hyperparams['background_distortion_factor']),
                'TISSUE_NOISE_FACTOR': str(hyperparams['tissue_noise_factor']),
                'BLIND_ZONE_NOISE_FACTOR': str(hyperparams['blind_zone_noise_factor']),
                'BACKGROUND_NOISE_FACTOR': str(hyperparams['background_noise_factor']),
                
                # Optuna optimization mode - don't save images to disk
                'OPTUNA_MODE': 'true',
                'NO_SAVE_IMAGES': 'true'
            })
            
            # Change to DDRM directory and run the script
            result = subprocess.run(
                ['bash', str(self.script_path)],
                env=env,
                cwd=str(self.ddrm_base_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"DDRM script failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
            
            logger.info("DDRM inference completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("DDRM script timed out")
            return False
        except Exception as e:
            logger.error(f"Error running DDRM script: {str(e)}")
            return False
    
    def evaluate_results(self, hyperparams: Dict) -> float:
        """Evaluate the quality of generated images compared to ground truth"""
        output_dir = self.ddrm_base_path / "outputs_ultrasound_ddrm"
        
        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return 0.0
        
        image_pairs = self.get_image_pairs()
        if not image_pairs:
            logger.error("No image pairs found for evaluation")
            return 0.0
        
        total_score = 0.0
        valid_pairs = 0
        
        for generated_name, gt_path in image_pairs:
            generated_path = output_dir / generated_name
            
            if not generated_path.exists():
                logger.warning(f"Generated image not found: {generated_path}")
                continue
            
            try:
                # Calculate multiple metrics
                ssim_score = ImageMetrics.calculate_ssim(str(generated_path), gt_path)
                psnr_score = ImageMetrics.calculate_psnr(str(generated_path), gt_path)
                mse_score = ImageMetrics.calculate_mse(str(generated_path), gt_path)
                mae_score = ImageMetrics.calculate_mae(str(generated_path), gt_path)
                
                # Normalize PSNR to [0, 1] range (assuming max PSNR of 50)
                psnr_normalized = min(psnr_score / 50.0, 1.0)
                
                # Invert MSE and MAE (lower is better) and normalize
                mse_normalized = max(0, 1.0 - min(mse_score * 10, 1.0))
                mae_normalized = max(0, 1.0 - min(mae_score * 5, 1.0))
                
                # Combine metrics with weights
                combined_score = (
                    0.4 * ssim_score +
                    0.3 * psnr_normalized +
                    0.2 * mse_normalized +
                    0.1 * mae_normalized
                )
                
                total_score += combined_score
                valid_pairs += 1
                
                logger.info(f"Metrics for {generated_name}: SSIM={ssim_score:.4f}, "
                           f"PSNR={psnr_score:.2f}, MSE={mse_score:.4f}, MAE={mae_score:.4f}, "
                           f"Combined={combined_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {generated_name}: {str(e)}")
                continue
        
        if valid_pairs == 0:
            logger.error("No valid image pairs for evaluation")
            return 0.0
        
        average_score = total_score / valid_pairs
        logger.info(f"Average score across {valid_pairs} pairs: {average_score:.4f}")
        
        return average_score
    
    def evaluate_results_memory_only(self, hyperparams: Dict) -> float:
        """
        Evaluate results in memory without saving images to disk
        Uses temporary numpy arrays from DDRM output for SSIM calculation
        """
        # Check if DDRM script generated temporary results
        temp_results_dir = self.ddrm_base_path / "temp_optuna_results"
        
        if not temp_results_dir.exists():
            logger.error(f"Temporary results directory not found: {temp_results_dir}")
            return 0.0
        
        image_pairs = self.get_image_pairs_for_memory_eval()
        if not image_pairs:
            logger.error("No image pairs found for memory evaluation")
            return 0.0
        
        total_score = 0.0
        valid_pairs = 0
        
        for generated_name, gt_path in image_pairs:
            # Look for temporary numpy file instead of saved image
            temp_name = generated_name.replace('.png', '.npy').replace('_restored_', '_temp_')
            temp_path = temp_results_dir / temp_name
            
            if not temp_path.exists():
                logger.warning(f"Temporary result not found: {temp_path}")
                continue
            
            try:
                # Load generated image from numpy array
                generated_array = np.load(temp_path)
                
                # Load ground truth image
                gt_img = ImageMetrics.load_and_preprocess_image(gt_path)
                
                # Ensure same dimensions
                min_h, min_w = min(generated_array.shape[0], gt_img.shape[0]), min(generated_array.shape[1], gt_img.shape[1])
                generated_array = generated_array[:min_h, :min_w]
                gt_img = gt_img[:min_h, :min_w]
                
                # Calculate SSIM directly
                from skimage.metrics import structural_similarity as ssim
                ssim_score = ssim(generated_array, gt_img, data_range=1.0)
                
                total_score += ssim_score
                valid_pairs += 1
                
                logger.info(f"Memory SSIM for {generated_name}: {ssim_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error calculating memory SSIM for {generated_name}: {str(e)}")
                continue
        
        if valid_pairs == 0:
            logger.error("No valid pairs for memory evaluation")
            return 0.0
        
        average_score = total_score / valid_pairs
        logger.info(f"Average memory SSIM across {valid_pairs} pairs: {average_score:.4f}")
        
        # Keep temporary files for next trial (will be overwritten)
        logger.info(f"Temporary results kept in {temp_results_dir} for next trial")
        
        return average_score
    
    def get_image_pairs_for_memory_eval(self) -> List[Tuple[str, str]]:
        """Get pairs for memory evaluation - V3~V7 001 and 201 only"""
        pairs = []
        
        # Only V3~V7 with 001 and 201 patterns
        gt_patterns = ['CN_OY_PC_D000_V3_001.bmp', 'CN_OY_PC_D000_V3_201.bmp', 
                      'CN_OY_PC_D000_V4_001.bmp', 'CN_OY_PC_D000_V4_201.bmp',
                      'CN_OY_PC_D000_V5_001.bmp', 'CN_OY_PC_D000_V5_201.bmp',
                      'CN_OY_PC_D000_V6_001.bmp', 'CN_OY_PC_D000_V6_201.bmp',
                      'CN_OY_PC_D000_V7_001.bmp', 'CN_OY_PC_D000_V7_201.bmp']
        
        for gt_pattern in gt_patterns:
            gt_file = self.gt_path / gt_pattern
            if gt_file.exists():
                # Corresponding generated file pattern (for temp files)
                base_name = gt_pattern.replace('CN_OY_', 'CY_OY_').replace('.bmp', '')
                version = 'V3' if 'V3' in gt_pattern else 'V4' if 'V4' in gt_pattern else 'V5' if 'V5' in gt_pattern else 'V6' if 'V6' in gt_pattern else 'V7'
                generated_file = f"{base_name}_restored_{version}.png"
                pairs.append((generated_file, str(gt_file)))
        
        logger.info(f"Found {len(pairs)} image pairs for SSIM evaluation (V3~V7 001,201 only)")
        return pairs
    
    def evaluate_hyperparameters_only(self, hyperparams: Dict, trial_number: int) -> float:
        """
        Evaluate hyperparameter quality without running actual inference
        Uses heuristic scoring based on parameter balance and physical principles
        """
        score = 0.0
        
        # 1. DDRM sampling parameter balance (25% of score)
        timesteps = hyperparams['timesteps']
        eta = hyperparams['eta']
        sigma_0 = hyperparams['sigma_0']
        
        # Optimal timesteps: 15-35 (balance between quality and speed)
        timestep_score = 1.0 - abs(timesteps - 25) / 25.0
        timestep_score = max(0, timestep_score)
        
        # Optimal eta: 0.3-0.7 (balance between exploration and stability)
        eta_score = 1.0 - abs(eta - 0.5) / 0.5
        eta_score = max(0, eta_score)
        
        # Optimal sigma_0: 0.05-0.15 (balance between noise and preservation)
        sigma_score = 1.0 - abs(sigma_0 - 0.1) / 0.1
        sigma_score = max(0, sigma_score)
        
        ddrm_score = (timestep_score + eta_score + sigma_score) / 3 * 0.25
        
        # 2. Version-specific percentile balance (40% of score)
        version_scores = []
        for version in ['v3', 'v4', 'v5', 'v6', 'v7']:
            tissue_key = f'{version}_tissue_percentile'
            blind_key = f'{version}_blind_zone_percentile'
            
            tissue_pct = hyperparams[tissue_key]
            blind_pct = hyperparams[blind_key]
            
            # Good separation: tissue should be significantly higher than blind zone
            separation = tissue_pct - blind_pct
            separation_score = min(1.0, separation / 30.0)  # 30% separation is optimal
            
            # Version-specific optimal ranges
            version_optima = {
                'v3': {'tissue': 70, 'blind': 35},  # Large donut
                'v4': {'tissue': 75, 'blind': 40},  # Med-large donut
                'v5': {'tissue': 80, 'blind': 45},  # Medium donut
                'v6': {'tissue': 85, 'blind': 50},  # Small donut
                'v7': {'tissue': 88, 'blind': 55}   # Minimal donut
            }
            
            optimal = version_optima[version]
            tissue_dev = abs(tissue_pct - optimal['tissue']) / 20.0
            blind_dev = abs(blind_pct - optimal['blind']) / 20.0
            
            accuracy_score = max(0, 1.0 - (tissue_dev + blind_dev) / 2)
            version_score = (separation_score * 0.6 + accuracy_score * 0.4)
            version_scores.append(version_score)
            
        version_score = np.mean(version_scores) * 0.4
        
        # 3. Physics-based parameter balance (25% of score)
        distortion_factor = hyperparams['distortion_factor']
        noise_factor = hyperparams['noise_factor']
        
        # Optimal distortion: 0.02-0.05 (balance between correction and stability)
        dist_score = 1.0 - abs(distortion_factor - 0.035) / 0.035
        dist_score = max(0, dist_score)
        
        # Optimal noise: 0.02-0.04 (balance between denoising and preservation)
        noise_score = 1.0 - abs(noise_factor - 0.03) / 0.03
        noise_score = max(0, noise_score)
        
        # Natural restoration balance
        tissue_dist = hyperparams['tissue_distortion_factor']
        blind_dist = hyperparams['blind_zone_distortion_factor']
        bg_dist = hyperparams['background_distortion_factor']
        
        # Ideal: tissue < background < blind_zone for distortion
        restoration_score = 0.0
        if tissue_dist < bg_dist < blind_dist:
            restoration_score = 0.5
        if blind_dist > 0.9 and tissue_dist < 0.3:  # Strong blind zone correction, gentle tissue
            restoration_score += 0.5
            
        physics_score = (dist_score + noise_score + restoration_score) / 3 * 0.25
        
        # 4. Mask cleaning parameter quality (10% of score)
        tissue_min = hyperparams['tissue_min_size']
        blind_min = hyperparams['blind_zone_min_size']
        
        # Optimal sizes: tissue_min > blind_min, reasonable ranges
        size_ratio_score = min(1.0, tissue_min / max(blind_min, 1)) / 3.0  # Tissue should be 2-3x larger
        size_range_score = 1.0 if 150 <= tissue_min <= 400 and 75 <= blind_min <= 150 else 0.5
        
        mask_score = (size_ratio_score + size_range_score) / 2 * 0.1
        
        # Combine all scores
        total_score = ddrm_score + version_score + physics_score + mask_score
        
        # Add small random component to avoid identical scores
        random_component = np.random.random() * 0.01
        total_score += random_component
        
        logger.info(f"Trial {trial_number} heuristic breakdown:")
        logger.info(f"  - DDRM parameters: {ddrm_score:.3f}")
        logger.info(f"  - Version balance: {version_score:.3f}")
        logger.info(f"  - Physics balance: {physics_score:.3f}")
        logger.info(f"  - Mask cleaning: {mask_score:.3f}")
        logger.info(f"  - Total score: {total_score:.3f}")
        
        return total_score
    
    def objective(self, trial):
        """Optuna objective function to maximize - DDRM inference with memory-only SSIM evaluation"""
        
        # Define enhanced hyperparameter search space for V3~V7 donut-based processing
        # Search ranges centered around run_ultrasound_ddrm.sh defaults
        hyperparams = {
            # Core DDRM sampling parameters (centered on shell script defaults)
            'timesteps': trial.suggest_int('timesteps', 15, 25),  # default: 20, range: ±5
            'eta': trial.suggest_float('eta', 0.75, 0.95),        # default: 0.85, range: ±0.1
            'sigma_0': trial.suggest_float('sigma_0', 0.005, 0.02), # default: 0.01, range: 0.5x~2x
            'distortion_factor': trial.suggest_float('distortion_factor', 0.015, 0.035), # default: 0.025, range: ±0.01
            'noise_factor': trial.suggest_float('noise_factor', 0.005, 0.015),  # default: 0.01, range: ±0.005
            
            # V3~V7 donut-based tissue/blind zone separation thresholds (percentiles)
            # V3: Large donut (inner_r=85, outer_r=220)
            'v3_tissue_percentile': trial.suggest_float('v3_tissue_percentile', 60, 80),
            'v3_blind_zone_percentile': trial.suggest_float('v3_blind_zone_percentile', 25, 45),
            
            # V4: Med-Large donut (inner_r=50, outer_r=130)
            'v4_tissue_percentile': trial.suggest_float('v4_tissue_percentile', 65, 85),
            'v4_blind_zone_percentile': trial.suggest_float('v4_blind_zone_percentile', 30, 50),
            
            # V5: Medium donut (inner_r=30, outer_r=90)
            'v5_tissue_percentile': trial.suggest_float('v5_tissue_percentile', 70, 90),
            'v5_blind_zone_percentile': trial.suggest_float('v5_blind_zone_percentile', 35, 55),
            
            # V6: Small donut (inner_r=20, outer_r=60)
            'v6_tissue_percentile': trial.suggest_float('v6_tissue_percentile', 75, 95),
            'v6_blind_zone_percentile': trial.suggest_float('v6_blind_zone_percentile', 40, 60),
            
            # V7: Minimal donut (inner_r=15, outer_r=45)
            'v7_tissue_percentile': trial.suggest_float('v7_tissue_percentile', 80, 95),
            'v7_blind_zone_percentile': trial.suggest_float('v7_blind_zone_percentile', 45, 65),
            
            # Mask cleaning parameters
            'tissue_min_size': trial.suggest_int('tissue_min_size', 100, 500),
            'blind_zone_min_size': trial.suggest_int('blind_zone_min_size', 50, 200),
            
            # Natural restoration parameters (physics-based DDRM enhancement)
            'tissue_distortion_factor': trial.suggest_float('tissue_distortion_factor', 0.0, 0.5),
            'blind_zone_distortion_factor': trial.suggest_float('blind_zone_distortion_factor', 0.8, 1.2),
            'background_distortion_factor': trial.suggest_float('background_distortion_factor', 0.0, 0.2),
            'tissue_noise_factor': trial.suggest_float('tissue_noise_factor', 0.0, 0.4),
            'blind_zone_noise_factor': trial.suggest_float('blind_zone_noise_factor', 0.8, 1.2),
            'background_noise_factor': trial.suggest_float('background_noise_factor', 0.0, 0.2)
        }
        
        logger.info(f"Trial {trial.number}: Testing hyperparameters: {hyperparams}")
        
        # Run DDRM inference with these hyperparameters (but don't save images)
        success = self.run_ddrm_inference(hyperparams)
        if not success:
            logger.error(f"Trial {trial.number}: DDRM inference failed")
            return 0.0
        
        # Evaluate results against ground truth using SSIM (no image saving)
        score = self.evaluate_results_memory_only(hyperparams)
        
        # Save trial results
        trial_result = {
            'trial_number': trial.number,
            'hyperparams': hyperparams,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        trial_file = self.optimization_dir / f"trial_{trial.number:04d}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_result, f, indent=2)
        
        logger.info(f"Trial {trial.number}: SSIM Score = {score:.4f}")
        
        return score
    
    def optimize(self, study_name: str = "ultrasound_ddrm_optimization"):
        """Run the optimization process"""
        
        logger.info(f"Starting optimization with {self.n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',  # We want to maximize the similarity score
            study_name=study_name,
            storage=f"sqlite:///{self.optimization_dir}/optuna_study.db",
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'optimization_completed': datetime.now().isoformat()
        }
        
        results_file = self.optimization_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("=== Optimization Completed ===")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info("Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Generate final optimized script
        self.generate_optimized_script(study.best_params, study)
        
        # Clean up temporary files after optimization is complete
        temp_results_dir = self.ddrm_base_path / "temp_optuna_results"
        if temp_results_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_results_dir)
                logger.info("Cleaned up temporary results directory after optimization completion")
            except:
                pass
        
        return study
    
    def generate_optimized_script(self, best_params: Dict, study=None):
        """Generate a shell script with the best hyperparameters"""
        
        script_content = f"""#!/bin/bash
# Optimized Enhanced DDRM parameters found by Optuna optimization (SSIM evaluation)
# Best SSIM score: {study.best_value:.4f if study else 'Unknown'} (optimized on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
# Generated by optuna_hyperparameter_optimization.py with V3~V7 donut-based processing (memory-only evaluation)

# Core DDRM sampling parameters
export TIMESTEPS={best_params['timesteps']}
export ETA={best_params['eta']}
export SIGMA_0={best_params['sigma_0']}
export DISTORTION_FACTOR={best_params['distortion_factor']}
export NOISE_FACTOR={best_params['noise_factor']}

# V3~V7 donut-based tissue/blind zone separation parameters
export V3_TISSUE_PERCENTILE={best_params['v3_tissue_percentile']}
export V3_BLIND_ZONE_PERCENTILE={best_params['v3_blind_zone_percentile']}
export V4_TISSUE_PERCENTILE={best_params['v4_tissue_percentile']}
export V4_BLIND_ZONE_PERCENTILE={best_params['v4_blind_zone_percentile']}
export V5_TISSUE_PERCENTILE={best_params['v5_tissue_percentile']}
export V5_BLIND_ZONE_PERCENTILE={best_params['v5_blind_zone_percentile']}
export V6_TISSUE_PERCENTILE={best_params['v6_tissue_percentile']}
export V6_BLIND_ZONE_PERCENTILE={best_params['v6_blind_zone_percentile']}
export V7_TISSUE_PERCENTILE={best_params['v7_tissue_percentile']}
export V7_BLIND_ZONE_PERCENTILE={best_params['v7_blind_zone_percentile']}

# Mask cleaning parameters
export TISSUE_MIN_SIZE={best_params['tissue_min_size']}
export BLIND_ZONE_MIN_SIZE={best_params['blind_zone_min_size']}

# Natural restoration parameters (physics-based enhancement)
export NATURAL_RESTORATION="true"
export TISSUE_DISTORTION_FACTOR={best_params['tissue_distortion_factor']}
export BLIND_ZONE_DISTORTION_FACTOR={best_params['blind_zone_distortion_factor']}
export BACKGROUND_DISTORTION_FACTOR={best_params['background_distortion_factor']}
export TISSUE_NOISE_FACTOR={best_params['tissue_noise_factor']}
export BLIND_ZONE_NOISE_FACTOR={best_params['blind_zone_noise_factor']}
export BACKGROUND_NOISE_FACTOR={best_params['background_noise_factor']}

echo "=== Running Optimized Enhanced Ultrasound DDRM ==="
echo "Best parameters found by Optuna optimization:"
echo "  Score: {study.best_value:.4f if study else 'Unknown'}"
echo "  V3 Donut: Tissue={best_params['v3_tissue_percentile']:.1f}%, BlindZone={best_params['v3_blind_zone_percentile']:.1f}%"
echo "  V4 Donut: Tissue={best_params['v4_tissue_percentile']:.1f}%, BlindZone={best_params['v4_blind_zone_percentile']:.1f}%"
echo "  V5 Donut: Tissue={best_params['v5_tissue_percentile']:.1f}%, BlindZone={best_params['v5_blind_zone_percentile']:.1f}%"
echo "  V6 Donut: Tissue={best_params['v6_tissue_percentile']:.1f}%, BlindZone={best_params['v6_blind_zone_percentile']:.1f}%"
echo "  V7 Donut: Tissue={best_params['v7_tissue_percentile']:.1f}%, BlindZone={best_params['v7_blind_zone_percentile']:.1f}%"
echo "  Natural Restoration: Enabled"
echo ""

# Run the optimized inference
./run_ultrasound_ddrm.sh
"""
        
        optimized_script = self.optimization_dir / "run_optimized_ddrm.sh"
        with open(optimized_script, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(optimized_script, 0o755)
        
        logger.info(f"Generated optimized script: {optimized_script}")
        logger.info("Enhanced V3~V7 donut-based optimization completed with natural restoration parameters")


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for DDRM")
    parser.add_argument("--ddrm_path", default="/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm", 
                       help="Path to DDRM directory")
    parser.add_argument("--gt_path", default="/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm/datasets/test_CN_OY",
                       help="Path to ground truth images")
    parser.add_argument("--n_trials", type=int, default=100, 
                       help="Number of optimization trials")
    parser.add_argument("--study_name", default="ultrasound_ddrm_optimization",
                       help="Name for the Optuna study")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = DDRMOptimizer(
        ddrm_base_path=args.ddrm_path,
        gt_path=args.gt_path,
        n_trials=args.n_trials
    )
    
    # Run optimization
    study = optimizer.optimize(args.study_name)
    
    logger.info("Optimization completed successfully!")


if __name__ == "__main__":
    main()