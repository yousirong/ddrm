#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization for Ultrasound DDRM
Optimizes hyperparameters to maximize similarity between generated images 
from CY_OY_PC_D000_V3_000_restored_V3 and ground truth CN_OY images.
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
    """Optuna-based hyperparameter optimizer for DDRM ultrasound model"""
    
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
        """Get pairs of generated and ground truth images for evaluation"""
        pairs = []
        
        # Look for ground truth images in CN_OY directory
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
        
        logger.info(f"Found {len(pairs)} image pairs for evaluation")
        return pairs
    
    def run_ddrm_inference(self, hyperparams: Dict) -> bool:
        """Run DDRM inference with given hyperparameters"""
        try:
            # Create environment variables for the script
            env = os.environ.copy()
            env.update({
                'TIMESTEPS': str(hyperparams['timesteps']),
                'ETA': str(hyperparams['eta']),
                'SIGMA_0': str(hyperparams['sigma_0']),
                'DISTORTION_FACTOR': str(hyperparams['distortion_factor']),
                'NOISE_FACTOR': str(hyperparams['noise_factor']),
                'THRESHOLD_V3': str(hyperparams['threshold_v3']),
                'THRESHOLD_V4': str(hyperparams['threshold_v4']),
                'THRESHOLD_V5': str(hyperparams['threshold_v5']),
                'THRESHOLD_V6': str(hyperparams['threshold_v6']),
                'THRESHOLD_V7': str(hyperparams['threshold_v7'])
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
    
    def objective(self, trial):
        """Optuna objective function to maximize"""
        
        # Define hyperparameter search space
        hyperparams = {
            'timesteps': trial.suggest_int('timesteps', 10, 50),
            'eta': trial.suggest_float('eta', 0.1, 1.0),
            'sigma_0': trial.suggest_float('sigma_0', 0.01, 0.2),
            'distortion_factor': trial.suggest_float('distortion_factor', 0.005, 0.1),
            'noise_factor': trial.suggest_float('noise_factor', 0.005, 0.1),
            'threshold_v3': trial.suggest_float('threshold_v3', 0.02, 0.15),
            'threshold_v4': trial.suggest_float('threshold_v4', 0.04, 0.18),
            'threshold_v5': trial.suggest_float('threshold_v5', 0.06, 0.20),
            'threshold_v6': trial.suggest_float('threshold_v6', 0.08, 0.25),
            'threshold_v7': trial.suggest_float('threshold_v7', 0.10, 0.30)
        }
        
        logger.info(f"Trial {trial.number}: Testing hyperparameters: {hyperparams}")
        
        # Run DDRM inference with these hyperparameters
        success = self.run_ddrm_inference(hyperparams)
        if not success:
            logger.error(f"Trial {trial.number}: DDRM inference failed")
            return 0.0
        
        # Evaluate the results
        score = self.evaluate_results(hyperparams)
        
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
        
        logger.info(f"Trial {trial.number}: Score = {score:.4f}")
        
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
        self.generate_optimized_script(study.best_params)
        
        return study
    
    def generate_optimized_script(self, best_params: Dict):
        """Generate a shell script with the best hyperparameters"""
        
        script_content = f"""#!/bin/bash
# Optimized DDRM parameters found by Optuna optimization
# Best score: Generated by optuna_hyperparameter_optimization.py
# Generated on: {datetime.now().isoformat()}

export TIMESTEPS={best_params['timesteps']}
export ETA={best_params['eta']}
export SIGMA_0={best_params['sigma_0']}
export DISTORTION_FACTOR={best_params['distortion_factor']}
export NOISE_FACTOR={best_params['noise_factor']}
export THRESHOLD_V3={best_params['threshold_v3']}
export THRESHOLD_V4={best_params['threshold_v4']}
export THRESHOLD_V5={best_params['threshold_v5']}
export THRESHOLD_V6={best_params['threshold_v6']}
export THRESHOLD_V7={best_params['threshold_v7']}

# Run the optimized inference
./run_ultrasound_ddrm.sh
"""
        
        optimized_script = self.optimization_dir / "run_optimized_ddrm.sh"
        with open(optimized_script, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(optimized_script, 0o755)
        
        logger.info(f"Generated optimized script: {optimized_script}")


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