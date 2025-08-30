#!/bin/bash

# AV-DRUS: Adaptive Variance Diffusion Restoration for Ultrasound
# Run script for BUSI dataset experiments

echo "🔬 Starting AV-DRUS experiments on BUSI dataset"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p av_drus_exp/logs/av_drus_busi
mkdir -p av_drus_exp/image_samples
mkdir -p av_drus_exp/tensorboard

# Configuration
CONFIG="av_drus_busi.yml"
DOC="av_drus_busi_$(date +%Y%m%d_%H%M%S)"
BUSI_PATH="/home/ubuntu/Desktop/JY/PAADI/Dataset_BUSI_with_GT"
TIMESTEPS=50
SIGMA=0.1

echo "📁 BUSI Dataset Path: $BUSI_PATH"
echo "⚙️  Configuration: $CONFIG"
echo "📝 Document ID: $DOC"
echo "🔢 Sampling timesteps: $TIMESTEPS"

# Check if BUSI dataset exists
if [ ! -d "$BUSI_PATH" ]; then
    echo "❌ Error: BUSI dataset not found at $BUSI_PATH"
    echo "Please download the BUSI dataset and update the path"
    exit 1
fi

# Run AV-DRUS sampling/reconstruction
echo "🎯 Running AV-DRUS reconstruction..."
python av_drus_main.py \
    --config $CONFIG \
    --doc $DOC \
    --sample \
    --busi_path $BUSI_PATH \
    --use_busi \
    --timesteps $TIMESTEPS \
    --sigma_0 $SIGMA \
    --subset_start 0 \
    --subset_end 20 \
    --ni

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "✅ AV-DRUS reconstruction completed successfully!"
    echo "📂 Results saved in: av_drus_exp/image_samples/"
    echo "📊 Logs saved in: av_drus_exp/logs/$DOC/"
else
    echo "❌ AV-DRUS reconstruction failed!"
    exit 1
fi

echo "🎉 Experiment completed!"