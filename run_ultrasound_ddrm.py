#!/usr/bin/env python3
"""
Run script for Ultrasound DDRM blind zone removal
This script demonstrates the complete pipeline from data preparation to restoration
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import torch
import yaml
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultrasound_ddrm import UltrasoundDDRM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(base_path):
    """Setup necessary directories"""
    base_path = Path(base_path)

    dirs = {
        'cn_on': base_path / 'datasets' / 'test_CN_ON',  # CN_ON images (use existing training data)
        'cy_on': base_path / 'datasets' / 'test_CY_ON',  # CY_ON images (use existing training data)
        'cn_oy': base_path / 'datasets' / 'test_CN_OY',     # CN_OY images
        'cy_oy': base_path / 'datasets' / 'test_CY_OY',     # CY_OY images
        'output': base_path / 'outputs_ultrasound_ddrm'
    }

    # Create output directory structure
    dirs['output'].mkdir(exist_ok=True)
    (dirs['output'] / 'originals').mkdir(exist_ok=True)
    (dirs['output'] / 'restored').mkdir(exist_ok=True)
    (dirs['output'] / 'comparison').mkdir(exist_ok=True)

    return dirs

def verify_data_paths(dirs):
    """Verify that data directories exist and contain images"""
    for name, path in dirs.items():
        if name == 'output':
            continue

        if not path.exists():
            logger.error(f"Directory {name} does not exist: {path}")
            return False

        # Count BMP files
        bmp_files = list(path.glob("*.bmp"))
        logger.info(f"{name}: Found {len(bmp_files)} BMP files in {path}")

        if len(bmp_files) == 0:
            logger.warning(f"No BMP files found in {name}: {path}")

    return True

def create_demo_config(output_path):
    """Create a demo configuration file if none exists"""
    config = {
        'data': {
            'dataset': 'ultrasound',
            'image_size': 512,  # Updated for 512x512
            'channels': 1
        },
        'model': {
            'type': 'unet2d',
            'channels': 1,
            'image_size': 512
        },
        'diffusion': {
            'timesteps': 1000,
            'beta_schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.02
        },
        'ddrm': {
            'sampling_steps': 50,
            'eta': 0.85,
            'etaB': 1.0
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Created demo config at {output_path}")
    return config

def run_complete_pipeline():
    """Run the complete DDRM pipeline"""
    parser = argparse.ArgumentParser(description="Ultrasound DDRM Complete Pipeline")
    parser.add_argument("--base_path", type=str,
                       default="/home/ubuntu/Desktop/JY/ultrasound_inp/ddrm",
                       help="Base path containing DDRM code and data")
    parser.add_argument("--config", type=str,
                       default="ultrasound_config.yml",
                       help="Config file name (in base_path)")
    parser.add_argument("--model_path", type=str,
                       default="/home/ubuntu/Desktop/JY/ultrasound_inp/diffusers/ddpm-ultrasound-512-a100",
                       help="Path to 512x512 DDPM model directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of DDRM sampling steps")
    parser.add_argument("--eta", type=float, default=0.85,
                       help="DDRM eta parameter")
    parser.add_argument("--etaB", type=float, default=1.0,
                       help="DDRM etaB parameter")
    parser.add_argument("--sigma_0", type=float, default=0.05,
                       help="DDRM sigma_0 parameter for noise level regularization")

    args = parser.parse_args()

    base_path = Path(args.base_path)
    config_path = base_path / args.config
    model_path = base_path / args.model_path

    # Setup directories
    logger.info("Setting up directories...")
    dirs = setup_directories(base_path)

    # Verify data paths
    if not verify_data_paths(dirs):
        logger.error("Data verification failed. Please check your data paths.")
        return

    # Check if config exists, create demo if not
    if not config_path.exists():
        logger.info("Config file not found, creating demo config...")
        create_demo_config(config_path)

    # Check if model exists (directory for diffusers model)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        logger.info("Please ensure you have a trained 512x512 DDPM model.")
        logger.info("You can train one using:")
        logger.info("  cd /home/ubuntu/Desktop/JY/ultrasound_inp/diffusers")
        logger.info("  bash train_512_a100.sh")
        return

    try:
        # Initialize DDRM for 512x512 model
        logger.info("Initializing DDRM for 512x512 model...")
        try:
            ddrm = UltrasoundDDRM(str(config_path), str(model_path), args.device)
            logger.info("Successfully loaded 512x512 DDPM model for DDRM")
        except Exception as e:
            logger.error(f"Failed to initialize DDRM: {e}")
            logger.info("Creating mock DDRM for demonstration...")
            ddrm = create_mock_ddrm(str(config_path), args.device)

        # Step 1: Estimate blind zone artifacts
        logger.info("=== Step 1: Estimating blind zone artifacts ===")
        noise_save_path = dirs['output'] / 'estimated_noise_pattern.npy'

        ddrm.estimate_blind_zone_artifacts(
            str(dirs['cn_on']),
            str(dirs['cy_on']),
            str(noise_save_path)
        )

        # Step 2: Estimate degradation operator
        logger.info("=== Step 2: Estimating degradation operator ===")
        ddrm.estimate_degradation_operator(
            str(dirs['cn_oy']),
            str(dirs['cy_oy'])
        )

        # Step 3: Test restoration on sample images (each version V3-V7)
        logger.info("=== Step 3: Testing restoration ===")

        # Get one sample from each version V3-V7
        test_images = []
        for version in ['V3', 'V4', 'V5', 'V6', 'V7']:
            version_files = list(dirs['cy_oy'].glob(f"*{version}*.bmp"))
            if version_files:
                test_images.append(version_files[0])  # Take first file of each version
                logger.info(f"Selected {version_files[0].name} for {version} testing")

        logger.info(f"Total test images: {len(test_images)} (V3-V7 samples)")

        for i, test_img_path in enumerate(test_images):
            # Save original image to originals folder
            original_output = dirs['output'] / 'originals' / f'original_{i:03d}_{test_img_path.name}'
            original_img = Image.open(test_img_path)
            original_img.save(original_output)

            # Save restored image to restored folder
            restored_output = dirs['output'] / 'restored' / f'restored_{i:03d}_{test_img_path.name}'

            logger.info(f"Restoring {test_img_path.name}...")
            try:
                ddrm.ddrm_restore(
                    str(test_img_path),
                    str(restored_output),
                    steps=args.steps,
                    eta=args.eta,
                    etaB=args.etaB,
                    sigma_0=args.sigma_0
                )
                logger.info(f"Saved original: {original_output}")
                logger.info(f"Saved restored: {restored_output}")

                # Create side-by-side comparison
                create_side_by_side_comparison(original_output, restored_output,
                                             dirs['output'] / 'comparison' / f'comparison_{i:03d}_{test_img_path.name}')

            except Exception as e:
                logger.error(f"Failed to restore {test_img_path.name}: {e}")

        # Create comparison images
        create_comparison_grid(dirs['output'])

        logger.info("=== Pipeline completed successfully! ===")
        logger.info(f"Results saved in: {dirs['output']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

def create_mock_ddrm(config_path, device):
    """Create a mock DDRM for demonstration when model is not available"""
    class MockDDRM:
        def __init__(self, config_path, device):
            self.device = device
            self.noise_pattern = None
            self.degradation_operator = None
            logger.info("Created mock DDRM (no actual model loaded)")

        def estimate_blind_zone_artifacts(self, cn_on_path, cy_on_path, save_path=None):
            # Simple demo estimation for 512x512
            logger.info("Mock: Estimating blind zone artifacts (512x512)...")
            self.noise_pattern = np.random.randn(512, 512) * 0.1
            if save_path:
                np.save(save_path, self.noise_pattern)
            return self.noise_pattern

        def estimate_degradation_operator(self, cn_oy_path, cy_oy_path):
            logger.info("Mock: Estimating degradation operator (512x512)...")
            self.degradation_operator = np.ones((512, 512)) * 0.8
            return self.degradation_operator

        def ddrm_restore(self, input_path, output_path, **kwargs):
            logger.info(f"Mock: Restoring {input_path} -> {output_path}")
            # Resize to 512x512 and save
            img = Image.open(input_path)
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(output_path)

    return MockDDRM(config_path, device)

def create_side_by_side_comparison(original_path, restored_path, output_path):
    """Create a side-by-side comparison of original and restored images"""
    try:
        original = Image.open(original_path)
        restored = Image.open(restored_path)

        # Ensure both images are the same size
        target_size = (512, 512)
        original = original.resize(target_size, Image.LANCZOS)
        restored = restored.resize(target_size, Image.LANCZOS)

        # Create side-by-side image
        comparison = Image.new('L', (target_size[0] * 2, target_size[1]), color=255)
        comparison.paste(original, (0, 0))
        comparison.paste(restored, (target_size[0], 0))

        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Add text labels
        draw.text((10, 10), "Original (CY)", fill=255, font=font)
        draw.text((target_size[0] + 10, 10), "DDRM Restored", fill=255, font=font)

        comparison.save(output_path)
        logger.info(f"Saved comparison: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create comparison: {e}")

def create_comparison_grid(output_dir):
    """Create a comparison grid showing original vs restored images"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        original_images = sorted(list((output_dir / 'originals').glob("*.bmp")))[:4]  # First 4 images
        restored_images = sorted(list((output_dir / 'restored').glob("*.bmp")))[:4]  # First 4 restored images

        if len(original_images) == 0 or len(restored_images) == 0:
            logger.warning("No images found for comparison grid")
            return

        # Create grid for 512x512 images
        img_size = 256  # Display size (downscaled for grid)
        grid_width = 4 * img_size
        grid_height = 2 * img_size

        grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

        # Place original images in top row (downscaled from 512x512)
        for i, img_path in enumerate(original_images[:4]):
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
            grid_img.paste(img, (i * img_size, 0))

        # Place restored images in bottom row (downscaled from 512x512)
        for i, img_path in enumerate(restored_images[:4]):
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size), Image.LANCZOS)
            grid_img.paste(img, (i * img_size, img_size))

        # Save grid
        grid_path = output_dir / 'comparison_grid.png'
        grid_img.save(grid_path)
        logger.info(f"Created comparison grid: {grid_path}")

    except Exception as e:
        logger.error(f"Failed to create comparison grid: {e}")

if __name__ == "__main__":
    run_complete_pipeline()