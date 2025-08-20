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
        "--timesteps", type=int, default=1000, help="number of steps involved"
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
        "--distortion_factor", type=float, default=0.05, 
        help="Distortion model strength factor (default: 0.05, original: 0.3)"
    )
    parser.add_argument(
        "--noise_factor", type=float, default=0.02,
        help="Noise factor for physics modeling (default: 0.02, original: 0.1)"
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