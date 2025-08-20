#!/bin/bash

# Change to the script's directory so that relative paths work correctly
cd "$(dirname "$0")"

# Enhanced Ultrasound DDRM with Physics-based Blind Zone Modeling
# Script to run ultrasound blind zone removal using DDRM methodology
#
# Usage:
#   ./run_ultrasound_ddrm.sh                    # Use default parameters
#   DISTORTION_FACTOR=0.1 ./run_ultrasound_ddrm.sh  # Override distortion strength
#   NOISE_FACTOR=0.05 ./run_ultrasound_ddrm.sh      # Override noise strength
#
# To customize parameters, edit the values below or set environment variables:
#   DISTORTION_FACTOR: Physics model distortion strength (default: 0.05, original: 0.3)
#   NOISE_FACTOR: Physics model noise strength (default: 0.02, original: 0.1)

echo "=== Enhanced Ultrasound DDRM Runner ==="
echo "Physics-based blind zone modeling with V3-V7 version handling"
echo ""

# Default configuration
CONFIG="ultrasound_config.yml"
DOC="ultrasound_ddrm_$(date +%Y%m%d_%H%M%S)"
TIMESTEPS=${TIMESTEPS:-20}
ETA=${ETA:-0.85}
SIGMA_0=${SIGMA_0:-0.05}
DISTORTION_FACTOR=${DISTORTION_FACTOR:-0.025}  # Physics model distortion strength (original: 0.3)
NOISE_FACTOR=${NOISE_FACTOR:-0.01}            # Physics model noise strength (original: 0.1)

# Data paths - Based on actual dataset structure
CN_ON_PATH="datasets/test_CN_ON"  # Path with CN_ON images for z_est estimation
CY_ON_PATH="datasets/test_CY_ON"  # Path with CY_ON images for z_est estimation
CN_OY_PATH="datasets/test_CN_OY"      # Path with CN_OY images for H_est estimation
CY_OY_PATH="datasets/test_CY_OY"      # Path with CY_OY images for H_est estimation
TEST_PATH="datasets/test_CY_OY"           # Path to test images for restoration (using some training images as demo)

OUTPUT_DIR="outputs_ultrasound_ddrm"

# Create config if it doesn't exist
if [ ! -f "$CONFIG" ]; then
    echo "Creating default ultrasound config..."
    cat > $CONFIG << EOF
model:
  type: simple
  in_channels: 1
  out_ch: 1
  ch: 128
  ch_mult: [1, 1, 2, 2, 4, 4]
  num_res_blocks: 2
  attn_resolutions: [16]
  dropout: 0.0
  var_type: fixedlarge
  resamp_with_conv: true
  ema_rate: 0.999
  ema: true

diffusion:
  beta_schedule: linear
  beta_start: 0.0001
  beta_end: 0.02
  num_diffusion_timesteps: 1000

data:
  dataset: ULTRASOUND
  image_size: 512
  channels: 1
  logit_transform: false
  uniform_dequantization: false
  gaussian_dequantization: false
  random_flip: false
  rescaled: true
  num_workers: 4

sampling:
  method: ddpm
  batch_size: 1
  last_only: true
  sample_step: 1
EOF
    echo "Created: $CONFIG"
fi

# Check if data paths exist
echo "Checking data paths..."
for path in "$CN_ON_PATH" "$CY_ON_PATH" "$TEST_PATH"; do
    if [ ! -d "$path" ]; then
        echo "Warning: Path not found: $path"
        echo "Please update the data paths in this script to match your dataset structure"
    else
        echo "Found: $path"
    fi
done

echo ""
echo "Running Enhanced Ultrasound DDRM..."
echo "Configuration:"
echo "  - Config: $CONFIG"
echo "  - Document: $DOC"
echo "  - CN_ON path: $CN_ON_PATH"
echo "  - CY_ON path: $CY_ON_PATH"
echo "  - CN_OY path: $CN_OY_PATH"
echo "  - CY_OY path: $CY_OY_PATH"
echo "  - Test path: $TEST_PATH"
echo "  - Output: $OUTPUT_DIR"
echo "  - Timesteps: $TIMESTEPS"
echo "  - Eta: $ETA"
echo "  - Sigma_0: $SIGMA_0"
echo "  - Distortion factor: $DISTORTION_FACTOR"
echo "  - Noise factor: $NOISE_FACTOR"
echo ""

# Run the enhanced ultrasound DDRM
python ultrasound_main.py \
    --config $CONFIG \
    --doc $DOC \
    --timesteps $TIMESTEPS \
    --eta $ETA \
    --sigma_0 $SIGMA_0 \
    --distortion_factor $DISTORTION_FACTOR \
    --noise_factor $NOISE_FACTOR \
    --cn_on_path $CN_ON_PATH \
    --cy_on_path $CY_ON_PATH \
    --cn_oy_path $CN_OY_PATH \
    --cy_oy_path $CY_OY_PATH \
    --test_images_path $TEST_PATH \
    --artifact_save_dir "${OUTPUT_DIR}/artifacts" \
    --image_folder $OUTPUT_DIR \
    --sample \
    --verbose info

echo ""
echo "=== Enhanced Ultrasound DDRM Completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key Features Implemented:"
echo "1. z_est = Average(CY_ON - CN_ON): Structural noise estimation"
echo "2. H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²: Distortion operator estimation"
echo "3. Physics-based modeling: Blind zone as physical distortion, not masking"
echo "4. Version-specific processing (V3-V7) with different blind zone characteristics"
echo "5. Integration with base DDRM efficient_generalized_steps sampling"
echo ""

# Show summary of results
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output files:"
    ls -la $OUTPUT_DIR/

    if [ -d "${OUTPUT_DIR}/artifacts" ]; then
        echo ""
        echo "Estimated artifacts:"
        ls -la ${OUTPUT_DIR}/artifacts/
    fi
fi