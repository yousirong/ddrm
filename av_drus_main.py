"""
AV-DRUS Main Execution Script
Adaptive Variance Diffusion Restoration for Ultrasound
"""

import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

# Import AV-DRUS specific modules
from runners.av_drus_diffusion import AVDRUSDiffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="AV-DRUS: Adaptive Variance Diffusion Restoration for Ultrasound")

    parser.add_argument(
        "--config", type=str, default="av_drus_busi.yml", help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="av_drus_exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default="av_drus_busi",
        help="A string for documentation purpose. Will be the name of the log folder.",
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
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--train",
        action="store_true", 
        help="Whether to train the model"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="av_drus_images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50, help="number of steps involved"
    )
    
    # AV-DRUS specific arguments
    parser.add_argument(
        "--busi_path", 
        type=str, 
        default="/home/ubuntu/Desktop/JY/PAADI/Dataset_BUSI_with_GT",
        help="Path to BUSI dataset"
    )
    parser.add_argument(
        "--use_busi", 
        action="store_true", 
        default=True,
        help="Use BUSI dataset"
    )
    parser.add_argument(
        "--sigma_0", type=float, default=0.1, help="Noise level"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta for sampling"
    )
    parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    
    # Model checkpoint
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint"
    )

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    config_path = os.path.join("configs", args.config) if not args.config.startswith('/') else args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
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
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    
    logging.info("🔬 AV-DRUS: Adaptive Variance Diffusion Restoration for Ultrasound")
    logging.info("📁 BUSI Dataset Path: {}".format(args.busi_path))
    logging.info("⚙️  Using {} timesteps for sampling".format(args.timesteps))

    try:
        runner = AVDRUSDiffusion(args, config)
        
        if args.train:
            logging.info("🏋️  Starting AV-DRUS training...")
            runner.train()
        else:
            logging.info("🎯 Starting AV-DRUS sampling/reconstruction...")
            runner.sample()
            
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())