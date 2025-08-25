#!/usr/bin/env python3
"""
Version-Specific Optuna Hyperparameter Optimization for Ultrasound DDRM
Optimizes hyperparameters separately for each version (V3-V7) since blind zone sizes differ.
Based on CLAUDE.md note: "CY_ON - CN_ON은 파일명에서 V3~V7에 따라 블라인드 존의 크기가 차이나서 각각 다르게 처리해야해"
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
import re
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VersionSpecificImageMetrics:
    """Calculate image similarity metrics with version-specific handling"""
    
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
    def get_version_from_filename(filename: str) -> str:
        """Extract version (V3, V4, V5, V6, V7) from filename"""
        match = re.search(r'_V([3-7])_', filename)
        return f"V{match.group(1)}" if match else "Unknown"
    
    @staticmethod
    def calculate_version_specific_metrics(img1_path: str, img2_path: str, version: str) -> Dict[str, float]:
        """Calculate metrics with version-specific weighting"""
        img1 = VersionSpecificImageMetrics.load_and_preprocess_image(img1_path)
        img2 = VersionSpecificImageMetrics.load_and_preprocess_image(img2_path)
        
        # Ensure same dimensions
        min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
        # Calculate base metrics
        ssim_score = ssim(img1, img2, data_range=1.0)
        psnr_score = psnr(img1, img2, data_range=1.0)
        mse_score = np.mean((img1 - img2) ** 2)
        mae_score = np.mean(np.abs(img1 - img2))
        
        # Version-specific weighting based on blind zone characteristics
        # V3: Smaller blind zones - prioritize detail preservation
        # V7: Larger blind zones - prioritize overall structure
        version_weights = {
            'V3': {'ssim': 0.5, 'psnr': 0.3, 'mse': 0.15, 'mae': 0.05},  # Detail-focused
            'V4': {'ssim': 0.45, 'psnr': 0.3, 'mse': 0.15, 'mae': 0.1},
            'V5': {'ssim': 0.4, 'psnr': 0.3, 'mse': 0.2, 'mae': 0.1},
            'V6': {'ssim': 0.35, 'psnr': 0.3, 'mse': 0.25, 'mae': 0.1},
            'V7': {'ssim': 0.3, 'psnr': 0.3, 'mse': 0.3, 'mae': 0.1}     # Structure-focused
        }
        
        weights = version_weights.get(version, version_weights['V5'])
        
        # Normalize scores
        psnr_normalized = min(psnr_score / 50.0, 1.0)
        mse_normalized = max(0, 1.0 - min(mse_score * 10, 1.0))
        mae_normalized = max(0, 1.0 - min(mae_score * 5, 1.0))
        
        # Calculate weighted score
        weighted_score = (
            weights['ssim'] * ssim_score +
            weights['psnr'] * psnr_normalized +
            weights['mse'] * mse_normalized +
            weights['mae'] * mae_normalized
        )
        
        return {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'mse': mse_score,
            'mae': mae_score,
            'weighted_score': weighted_score,
            'version': version
        }


class VersionSpecificDDRMOptimizer:
    """Version-specific Optuna hyperparameter optimizer for DDRM ultrasound model"""
    
    def __init__(self, 
                 ddrm_base_path: str = "/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm",
                 gt_path: str = "/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm/datasets/test_CN_OY",
                 n_trials: int = 50,
                 target_version: str = "V3"):
        
        self.ddrm_base_path = Path(ddrm_base_path)
        self.gt_path = Path(gt_path)
        self.n_trials = n_trials
        self.target_version = target_version
        self.script_path = self.ddrm_base_path / "run_ultrasound_ddrm.sh"
        
        # Create optimization results directory
        self.optimization_dir = self.ddrm_base_path / f"optimization_results_{target_version}"
        self.optimization_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Version-Specific DDRM Optimizer for {target_version}")
        logger.info(f"DDRM base path: {self.ddrm_base_path}")
        logger.info(f"Ground truth path: {self.gt_path}")
        logger.info(f"Optimization results: {self.optimization_dir}")
    
    def get_version_specific_image_pairs(self) -> List[Tuple[str, str]]:
        """Get pairs of generated and ground truth images for specific version"""
        pairs = []
        
        # Look for ground truth images matching the target version
        gt_files = list(self.gt_path.glob(f"*_{self.target_version}_*.bmp"))
        
        for gt_file in gt_files:
            # Convert CN_OY pattern to CY_OY pattern for generated files
            gt_name = gt_file.name
            generated_name = gt_name.replace('CN_OY_', 'CY_OY_').replace('.bmp', f'_restored_{self.target_version}.png')
            pairs.append((generated_name, str(gt_file)))
        
        logger.info(f"Found {len(pairs)} image pairs for {self.target_version} evaluation")
        return pairs
    
    def get_version_specific_hyperparameter_ranges(self) -> Dict:
        """Get hyperparameter ranges specific to each version"""
        # Based on blind zone size differences, different versions need different parameter ranges
        version_ranges = {
            'V3': {  # Smaller blind zones - more precise parameters
                'timesteps': (15, 40),
                'eta': (0.7, 0.95),
                'sigma_0': (0.02, 0.08),
                'distortion_factor': (0.01, 0.05),
                'noise_factor': (0.005, 0.03),
                'threshold': (0.04, 0.12)
            },
            'V4': {
                'timesteps': (15, 45),
                'eta': (0.65, 0.95),
                'sigma_0': (0.03, 0.1),
                'distortion_factor': (0.015, 0.06),
                'noise_factor': (0.008, 0.04),
                'threshold': (0.06, 0.15)
            },
            'V5': {
                'timesteps': (20, 50),
                'eta': (0.6, 0.9),
                'sigma_0': (0.04, 0.12),
                'distortion_factor': (0.02, 0.07),
                'noise_factor': (0.01, 0.05),
                'threshold': (0.08, 0.18)
            },
            'V6': {
                'timesteps': (20, 55),
                'eta': (0.55, 0.85),
                'sigma_0': (0.05, 0.15),
                'distortion_factor': (0.025, 0.08),
                'noise_factor': (0.015, 0.06),
                'threshold': (0.1, 0.22)
            },
            'V7': {  # Larger blind zones - broader parameters
                'timesteps': (25, 60),
                'eta': (0.5, 0.8),
                'sigma_0': (0.06, 0.2),
                'distortion_factor': (0.03, 0.1),
                'noise_factor': (0.02, 0.08),
                'threshold': (0.12, 0.26)
            }
        }
        
        return version_ranges.get(self.target_version, version_ranges['V5'])
    
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
                # Version-specific threshold
                f'THRESHOLD_{self.target_version}': str(hyperparams[f'threshold_{self.target_version.lower()}'])
            })
            
            # Also set other thresholds to reasonable defaults
            default_thresholds = {
                'V3': 0.08, 'V4': 0.10, 'V5': 0.12, 'V6': 0.15, 'V7': 0.18
            }
            
            for version, default_val in default_thresholds.items():
                if version != self.target_version:
                    env[f'THRESHOLD_{version}'] = str(default_val)
            
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
            
            logger.info(f"DDRM inference completed for {self.target_version}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("DDRM script timed out")
            return False
        except Exception as e:
            logger.error(f"Error running DDRM script: {str(e)}")
            return False
    
    def evaluate_results(self, hyperparams: Dict) -> float:
        """Evaluate the quality of generated images for specific version"""
        output_dir = self.ddrm_base_path / "outputs_ultrasound_ddrm"
        
        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return 0.0
        
        image_pairs = self.get_version_specific_image_pairs()
        if not image_pairs:
            logger.error(f"No image pairs found for {self.target_version} evaluation")
            return 0.0
        
        total_score = 0.0
        valid_pairs = 0
        
        for generated_name, gt_path in image_pairs:
            generated_path = output_dir / generated_name
            
            if not generated_path.exists():
                logger.warning(f"Generated image not found: {generated_path}")
                continue
            
            try:
                version = VersionSpecificImageMetrics.get_version_from_filename(generated_name)
                metrics = VersionSpecificImageMetrics.calculate_version_specific_metrics(
                    str(generated_path), gt_path, version
                )
                
                total_score += metrics['weighted_score']
                valid_pairs += 1
                
                logger.info(f"Metrics for {generated_name}: SSIM={metrics['ssim']:.4f}, "
                           f"PSNR={metrics['psnr']:.2f}, Score={metrics['weighted_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {generated_name}: {str(e)}")
                continue
        
        if valid_pairs == 0:
            logger.error(f"No valid image pairs for {self.target_version} evaluation")
            return 0.0
        
        average_score = total_score / valid_pairs
        logger.info(f"{self.target_version} average score across {valid_pairs} pairs: {average_score:.4f}")
        
        return average_score
    
    def objective(self, trial):
        """Optuna objective function for version-specific optimization"""
        
        # Get version-specific hyperparameter ranges
        param_ranges = self.get_version_specific_hyperparameter_ranges()
        
        # Define hyperparameters with version-specific ranges
        hyperparams = {
            'timesteps': trial.suggest_int('timesteps', *param_ranges['timesteps']),
            'eta': trial.suggest_float('eta', *param_ranges['eta']),
            'sigma_0': trial.suggest_float('sigma_0', *param_ranges['sigma_0']),
            'distortion_factor': trial.suggest_float('distortion_factor', *param_ranges['distortion_factor']),
            'noise_factor': trial.suggest_float('noise_factor', *param_ranges['noise_factor']),
            f'threshold_{self.target_version.lower()}': trial.suggest_float(f'threshold_{self.target_version.lower()}', *param_ranges['threshold'])
        }
        
        logger.info(f"Trial {trial.number} ({self.target_version}): Testing hyperparameters: {hyperparams}")
        
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
            'version': self.target_version,
            'hyperparams': hyperparams,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        trial_file = self.optimization_dir / f"trial_{trial.number:04d}_{self.target_version}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_result, f, indent=2)
        
        logger.info(f"Trial {trial.number} ({self.target_version}): Score = {score:.4f}")
        
        return score
    
    def optimize(self, study_name: str = None):
        """Run the version-specific optimization process"""
        
        if study_name is None:
            study_name = f"ultrasound_ddrm_optimization_{self.target_version}"
        
        logger.info(f"Starting {self.target_version} optimization with {self.n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f"sqlite:///{self.optimization_dir}/optuna_study_{self.target_version}.db",
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save results
        results = {
            'version': self.target_version,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'optimization_completed': datetime.now().isoformat()
        }
        
        results_file = self.optimization_dir / f"optimization_results_{self.target_version}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"=== {self.target_version} Optimization Completed ===")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info("Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Generate version-specific optimized script
        self.generate_optimized_script(study.best_params)
        
        return study
    
    def generate_optimized_script(self, best_params: Dict):
        """Generate a shell script with the best hyperparameters for this version"""
        
        script_content = f"""#!/bin/bash
# Optimized DDRM parameters for {self.target_version} found by Optuna optimization
# Best score: Generated by optuna_version_specific_optimization.py
# Generated on: {datetime.now().isoformat()}

export TIMESTEPS={best_params['timesteps']}
export ETA={best_params['eta']}
export SIGMA_0={best_params['sigma_0']}
export DISTORTION_FACTOR={best_params['distortion_factor']}
export NOISE_FACTOR={best_params['noise_factor']}
export THRESHOLD_{self.target_version}={best_params[f'threshold_{self.target_version.lower()}']}

# Set default values for other versions
export THRESHOLD_V3=0.08
export THRESHOLD_V4=0.10
export THRESHOLD_V5=0.12
export THRESHOLD_V6=0.15
export THRESHOLD_V7=0.18

# Override with optimized value for target version
export THRESHOLD_{self.target_version}={best_params[f'threshold_{self.target_version.lower()}']}

echo "Running optimized DDRM inference for {self.target_version}"
echo "Optimized parameters:"
echo "  TIMESTEPS: $TIMESTEPS"
echo "  ETA: $ETA"
echo "  SIGMA_0: $SIGMA_0"
echo "  DISTORTION_FACTOR: $DISTORTION_FACTOR"
echo "  NOISE_FACTOR: $NOISE_FACTOR"
echo "  THRESHOLD_{self.target_version}: $THRESHOLD_{self.target_version}"

# Run the optimized inference
./run_ultrasound_ddrm.sh
"""
        
        optimized_script = self.optimization_dir / f"run_optimized_ddrm_{self.target_version}.sh"
        with open(optimized_script, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(optimized_script, 0o755)
        
        logger.info(f"Generated optimized script for {self.target_version}: {optimized_script}")


def optimize_all_versions(ddrm_path: str, gt_path: str, n_trials_per_version: int = 30):
    """Optimize hyperparameters for all versions (V3-V7)"""
    
    versions = ['V3', 'V4', 'V5', 'V6', 'V7']
    results = {}
    
    logger.info("Starting optimization for all versions (V3-V7)")
    
    for version in versions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting optimization for {version}")
        logger.info(f"{'='*50}")
        
        optimizer = VersionSpecificDDRMOptimizer(
            ddrm_base_path=ddrm_path,
            gt_path=gt_path,
            n_trials=n_trials_per_version,
            target_version=version
        )
        
        study = optimizer.optimize()
        results[version] = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
    
    # Save combined results
    combined_results = {
        'all_versions_optimization': results,
        'optimization_completed': datetime.now().isoformat(),
        'total_trials': n_trials_per_version * len(versions)
    }
    
    results_path = Path(ddrm_path) / "optimization_results_all_versions.json"
    with open(results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("ALL VERSIONS OPTIMIZATION COMPLETED")
    logger.info(f"{'='*60}")
    
    for version, result in results.items():
        logger.info(f"{version}: Best score = {result['best_value']:.4f}")
    
    logger.info(f"Combined results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Version-specific Optuna optimization for DDRM")
    parser.add_argument("--ddrm_path", default="/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm", 
                       help="Path to DDRM directory")
    parser.add_argument("--gt_path", default="/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm/datasets/test_CN_OY",
                       help="Path to ground truth images")
    parser.add_argument("--version", choices=['V3', 'V4', 'V5', 'V6', 'V7', 'ALL'], default='V3',
                       help="Target version to optimize (or ALL for all versions)")
    parser.add_argument("--n_trials", type=int, default=30,
                       help="Number of optimization trials per version")
    
    args = parser.parse_args()
    
    if args.version == 'ALL':
        optimize_all_versions(args.ddrm_path, args.gt_path, args.n_trials)
    else:
        # Single version optimization
        optimizer = VersionSpecificDDRMOptimizer(
            ddrm_base_path=args.ddrm_path,
            gt_path=args.gt_path,
            n_trials=args.n_trials,
            target_version=args.version
        )
        
        study = optimizer.optimize()
        logger.info(f"Optimization for {args.version} completed successfully!")


if __name__ == "__main__":
    main()