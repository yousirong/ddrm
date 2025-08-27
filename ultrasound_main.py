import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from pathlib import Path

from ultrasound_runner import run_ultrasound_ddrm

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--use_pretrained", action="store_true"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=20, help="number of steps involved"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )
    parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
    parser.add_argument(
        "--sigma_0", type=float, default=0.05, help="Sigma_0 for noise level"
    )
    
    # Physics model strength parameters
    parser.add_argument(
        "--distortion_factor", type=float, default=0.025, 
        help="Distortion model strength factor (default: 0.05, original: 0.3)"
    )
    parser.add_argument(
        "--noise_factor", type=float, default=1.0,
        help="Noise factor for physics modeling (default: 0.02, original: 0.1)"
    )
    
    # Step-wise image saving
    parser.add_argument(
        "--save_steps", type=str, default="",
        help="Comma-separated list of steps to save intermediate images (e.g., '10,20,50,90')"
    )
    
    # Version-specific threshold parameters
    parser.add_argument(
        "--threshold_v3", type=float, default=0.08,
        help="Threshold for V3 blind zone detection (default: 0.08)"
    )
    parser.add_argument(
        "--threshold_v4", type=float, default=0.10,
        help="Threshold for V4 blind zone detection (default: 0.10)"
    )
    parser.add_argument(
        "--threshold_v5", type=float, default=0.12,
        help="Threshold for V5 blind zone detection (default: 0.12)"
    )
    parser.add_argument(
        "--threshold_v6", type=float, default=0.15,
        help="Threshold for V6 blind zone detection (default: 0.15)"
    )
    parser.add_argument(
        "--threshold_v7", type=float, default=0.18,
        help="Threshold for V7 blind zone detection (default: 0.18)"
    )
    
    # Enhanced ultrasound-specific arguments for DDRM
    parser.add_argument(
        "--deg", type=str, default="ultrasound_blind",
        help="Degradation type (ultrasound blind zone)"
    )
    
    # Data paths for artifact estimation  
    parser.add_argument(
        "--cn_on_path", type=str, 
        help="Path to CN_ON training images for z_est = Average(CY_ON - CN_ON)"
    )
    parser.add_argument(
        "--cy_on_path", type=str,
        help="Path to CY_ON training images for z_est = Average(CY_ON - CN_ON)" 
    )
    parser.add_argument(
        "--cn_oy_path", type=str,
        help="Path to CN_OY images for H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²"
    )
    parser.add_argument(
        "--cy_oy_path", type=str,
        help="Path to CY_OY images for H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²"
    )
    
    # Test images
    parser.add_argument(
        "--test_images_path", type=str, required=True,
        help="Path to test images for restoration"
    )
    
    # Output paths
    parser.add_argument(
        "--artifact_save_dir", type=str,
        help="Directory to save estimated artifacts (z_est, H_est)"
    )
    
    # Tissue protection features
    parser.add_argument(
        "--tissue_protection", action="store_true",
        help="Enable tissue protection during denoising process"
    )
    parser.add_argument(
        "--verbose_tissue", action="store_true",
        help="Enable verbose tissue protection logging"
    )
    
    # Enhanced tissue detection parameters
    parser.add_argument(
        "--enhanced_tissue_detection", action="store_true",
        help="Use enhanced multi-method tissue detection"
    )
    parser.add_argument(
        "--tissue_detection_mode", type=str, default="multi",
        choices=["multi", "adaptive", "edge", "simple"],
        help="Tissue detection method: multi (default), adaptive, edge, or simple"
    )
    parser.add_argument(
        "--clahe_clip_limit", type=float, default=3.0,
        help="CLAHE clip limit for contrast enhancement (default: 3.0)"
    )
    parser.add_argument(
        "--min_tissue_size_factor", type=float, default=1.0,
        help="Minimum tissue size threshold multiplier (default: 1.0)"
    )
    
    # Blind zone and background processing control
    parser.add_argument(
        "--complete_blind_zone_removal", action="store_true",
        help="Enable complete (black) removal of blind zones"
    )
    parser.add_argument(
        "--preserve_background", action="store_true",
        help="Preserve background areas unchanged"
    )
    
    # V3~V7 Donut-based tissue/blind zone separation parameters
    parser.add_argument(
        "--v3_tissue_percentile", type=float, default=65,
        help="V3 tissue separation threshold (percentile, default: 65)"
    )
    parser.add_argument(
        "--v3_blind_zone_percentile", type=float, default=35,
        help="V3 blind zone separation threshold (percentile, default: 35)"
    )
    parser.add_argument(
        "--v4_tissue_percentile", type=float, default=70,
        help="V4 tissue separation threshold (percentile, default: 70)"
    )
    parser.add_argument(
        "--v4_blind_zone_percentile", type=float, default=40,
        help="V4 blind zone separation threshold (percentile, default: 40)"
    )
    parser.add_argument(
        "--v5_tissue_percentile", type=float, default=75,
        help="V5 tissue separation threshold (percentile, default: 75)"
    )
    parser.add_argument(
        "--v5_blind_zone_percentile", type=float, default=45,
        help="V5 blind zone separation threshold (percentile, default: 45)"
    )
    parser.add_argument(
        "--v6_tissue_percentile", type=float, default=80,
        help="V6 tissue separation threshold (percentile, default: 80)"
    )
    parser.add_argument(
        "--v6_blind_zone_percentile", type=float, default=50,
        help="V6 blind zone separation threshold (percentile, default: 50)"
    )
    parser.add_argument(
        "--v7_tissue_percentile", type=float, default=85,
        help="V7 tissue separation threshold (percentile, default: 85)"
    )
    parser.add_argument(
        "--v7_blind_zone_percentile", type=float, default=55,
        help="V7 blind zone separation threshold (percentile, default: 55)"
    )
    
    # Mask cleaning parameters
    parser.add_argument(
        "--tissue_min_size", type=int, default=200,
        help="Minimum tissue region size in pixels (default: 200)"
    )
    parser.add_argument(
        "--blind_zone_min_size", type=int, default=100,
        help="Minimum blind zone region size in pixels (default: 100)"
    )
    
    # Optuna optimization mode parameters
    parser.add_argument(
        "--optuna_mode", action="store_true",
        help="Enable Optuna optimization mode (memory-only evaluation)"
    )
    parser.add_argument(
        "--no_save_images", action="store_true", 
        help="Disable saving images to disk (for Optuna optimization)"
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample and not args.fid and not args.interpolation:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb_logger = None

    # Create output directory
    output_path = Path(args.image_folder)
    output_path.mkdir(exist_ok=True)
    args.image_folder = str(output_path)

    # setup logger
    os.makedirs(args.log_path, exist_ok=True)
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("=== Enhanced Ultrasound DDRM with Physics-based Blind Zone Modeling ===")
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    
    # Log key parameters
    logging.info("Key Parameters:")
    logging.info(f"  - Test images: {args.test_images_path}")
    logging.info(f"  - CN_ON path: {args.cn_on_path}")
    logging.info(f"  - CY_ON path: {args.cy_on_path}")
    logging.info(f"  - CN_OY path: {args.cn_oy_path}")
    logging.info(f"  - CY_OY path: {args.cy_oy_path}")
    logging.info(f"  - Output folder: {args.image_folder}")
    logging.info(f"  - Timesteps: {args.timesteps}")
    logging.info(f"  - Eta: {args.eta}")
    logging.info(f"  - Sigma_0: {args.sigma_0}")
    logging.info(f"  - Distortion factor: {args.distortion_factor}")
    logging.info(f"  - Noise factor: {args.noise_factor}")
    logging.info(f"  - Threshold V3: {args.threshold_v3}")
    logging.info(f"  - Threshold V4: {args.threshold_v4}")
    logging.info(f"  - Threshold V5: {args.threshold_v5}")
    logging.info(f"  - Threshold V6: {args.threshold_v6}")
    logging.info(f"  - Threshold V7: {args.threshold_v7}")
    if args.save_steps:
        logging.info(f"  - Save intermediate steps: {args.save_steps}")

    try:
        # Run enhanced ultrasound DDRM restoration
        logging.info("Starting Enhanced Ultrasound DDRM Restoration...")
        logging.info("Methodology:")
        logging.info("  1. z_est = Average(CY_ON - CN_ON): Structural noise estimation")
        logging.info("  2. H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²: Distortion operator")
        logging.info("  3. Physics-based modeling: Blind zone as physical distortion")
        logging.info("  4. Version-specific processing (V3-V7)")
        
        results = run_ultrasound_ddrm(args, config)
        
        # Print summary
        if results:
            logging.info(f"=== Restoration Completed Successfully ===")
            logging.info(f"Total processed: {len(results)} images")
            
            # Group by version
            version_counts = {}
            for result in results:
                version = result.get('version', 'Unknown')
                version_counts[version] = version_counts.get(version, 0) + 1
            
            for version, count in version_counts.items():
                logging.info(f"  - {version}: {count} images")
            
            logging.info(f"Results saved to: {args.image_folder}")
        else:
            logging.warning("No results returned")
        
    except Exception as e:
        logging.error("=== Error during restoration ===")
        logging.error(f"Error: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())