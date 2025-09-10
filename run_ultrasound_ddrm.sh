#!/bin/bash

# Change to the script's directory so that relative paths work correctly
cd "$(dirname "$0")"

# Enhanced Ultrasound DDRM with V3-V7 Donut-based Tissue/Blind Zone Separation
# Script to run ultrasound blind zone removal using DDRM methodology with version-specific donut regions
#
# Usage:
#   ./run_ultrasound_ddrm.sh                    # Use default parameters
#   DISTORTION_FACTOR=0.1 ./run_ultrasound_ddrm.sh  # Override distortion strength
#   NOISE_FACTOR=0.05 ./run_ultrasound_ddrm.sh      # Override noise strength
#   SAVE_STEPS="5,10,15" ./run_ultrasound_ddrm.sh   # Save intermediate steps 5, 10, 15
#
# Key Features:
#   - V3~V7 hardcoded donut regions for precise tissue/blind zone separation
#   - Tissue (bright areas) protection during denoising
#   - Blind zone (dark areas) complete removal
#   - Version-specific percentile thresholds for optimal separation
#
# To customize parameters, edit the values below or set environment variables:
#   DISTORTION_FACTOR: Physics model distortion strength (default: 0.025, original: 0.3)
#   NOISE_FACTOR: Physics model noise strength (default: 0.01, original: 0.1)
#   SAVE_STEPS: Comma-separated steps to save intermediate images (e.g., "5,10,15")

echo "=== Enhanced Ultrasound DDRM Runner ==="
echo "V3-V7 Donut-based tissue/blind zone separation with version-specific processing"
echo ""

# Default configuration
CONFIG="ultrasound_config.yml"
DOC="ultrasound_ddrm_$(date +%Y%m%d_%H%M%S)"
TIMESTEPS=${TIMESTEPS:-20}
ETA=${ETA:-0.85}
SIGMA_0=${SIGMA_0:-0.005}
DISTORTION_FACTOR=${DISTORTION_FACTOR:-0.025}  # Physics model distortion strength (original: 0.3)
NOISE_FACTOR=${NOISE_FACTOR:-0.01}            # Physics model noise strength (original: 0.1)
SAVE_STEPS=${SAVE_STEPS:-""}                   # Comma-separated steps to save intermediate images (e.g., "5,10,15")

# Enhanced features from recent updates - TISSUE POST-PROCESSING COMMENTED OUT
# TISSUE_PROTECTION=${TISSUE_PROTECTION:-"false"}  # Disable tissue protection for full DDRM processing
# VERBOSE_TISSUE=${VERBOSE_TISSUE:-"false"}       # Enable detailed tissue protection logging

# Enhanced tissue detection parameters - TISSUE POST-PROCESSING COMMENTED OUT
# ENHANCED_TISSUE_DETECTION=${ENHANCED_TISSUE_DETECTION:-"false"}  # Disable tissue detection for full DDRM processing
# TISSUE_DETECTION_MODE=${TISSUE_DETECTION_MODE:-"multi"}        # multi/adaptive/edge/simple
# CLAHE_CLIP_LIMIT=${CLAHE_CLIP_LIMIT:-3.0}                     # CLAHE enhancement strength
# MIN_TISSUE_SIZE_FACTOR=${MIN_TISSUE_SIZE_FACTOR:-1.0}          # Tissue size threshold multiplier

# Set tissue processing to disabled when commented out
TISSUE_PROTECTION="false"
VERBOSE_TISSUE="false"
ENHANCED_TISSUE_DETECTION="false"
TISSUE_DETECTION_MODE="multi"
CLAHE_CLIP_LIMIT=3.0
MIN_TISSUE_SIZE_FACTOR=1.0

# Optuna optimization mode - prevent image saving
OPTUNA_MODE=${OPTUNA_MODE:-"false"}                            # Enable memory-only evaluation for Optuna
NO_SAVE_IMAGES=${NO_SAVE_IMAGES:-"false"}                      # Disable saving images to disk

# Blind zone and background processing control
COMPLETE_BLIND_ZONE_REMOVAL=${COMPLETE_BLIND_ZONE_REMOVAL:-"false"}  # Disable complete blind zone removal for full DDRM processing
PRESERVE_BACKGROUND=${PRESERVE_BACKGROUND:-"false"}                  # Process background for full DDRM processing

# V3~V7 parameters are now handled internally by the dataset building approach
# Parameters moved to ultrasound_h_funcs.py radius_map and threshold processing
# V3: (42, 82, 230), V4: (25, 48, 133), V5: (17, 32, 90), V6: (11, 22, 63), V7: (9, 17, 48)

# Mask cleaning parameters - Enhanced for angle-aware detection
TISSUE_MIN_SIZE=${TISSUE_MIN_SIZE:-200}                 # Minimum tissue region size (pixels)
BLIND_ZONE_MIN_SIZE=${BLIND_ZONE_MIN_SIZE:-100}         # Minimum blind zone region size (pixels)

# Natural restoration parameters (physics-based DDRM enhancement)
NATURAL_RESTORATION=${NATURAL_RESTORATION:-"false"}      # Disable selective restoration for uniform DDRM processing
TISSUE_DISTORTION_FACTOR=${TISSUE_DISTORTION_FACTOR:-0.1}  # Tissue distortion strength multiplier (0.3 = 30% of base)
TISSUE_NOISE_FACTOR=${TISSUE_NOISE_FACTOR:-0.05}        # Tissue noise strength multiplier (0.2 = 20% of base)
BLIND_ZONE_DISTORTION_FACTOR=${BLIND_ZONE_DISTORTION_FACTOR:-1.0}  # Blind zone distortion strength multiplier
BLIND_ZONE_NOISE_FACTOR=${BLIND_ZONE_NOISE_FACTOR:-1.0}  # Blind zone noise strength multiplier
BACKGROUND_NOISE_FACTOR=${BACKGROUND_NOISE_FACTOR:-0.1}  # Background noise strength multiplier
BACKGROUND_DISTORTION_FACTOR=${BACKGROUND_DISTORTION_FACTOR:-0.1}  # Background distortion strength multiplier

# Version-specific blind zone detection thresholds
THRESHOLD_V3=${THRESHOLD_V3:-0.0}    # V3: Use full donut region (no threshold)
THRESHOLD_V4=${THRESHOLD_V4:-0.0}    # V4: Use full donut region (no threshold)
THRESHOLD_V5=${THRESHOLD_V5:-0.0}    # V5: Use full donut region (no threshold)
THRESHOLD_V6=${THRESHOLD_V6:-0.0}    # V6: Use full donut region (no threshold)
THRESHOLD_V7=${THRESHOLD_V7:-0.0}    # V7: Use full donut region (no threshold)



# Data paths - Based on actual dataset structure
CN_ON_PATH="datasets/test_CN_ON"  # Path with CN_ON images for z_est estimation
CY_ON_PATH="datasets/test_CY_ON"  # Path with CY_ON images for z_est estimation

CN_OY_PATH="datasets/test_CN_OY"      # Path with CN_OY images for H_est estimation
CY_OY_PATH="datasets/test_CY_OY"      # Path with CY_OY images for H_est estimation
TEST_PATH="datasets/test_CY_ON_PL"           # Path to test images for restoration (using some training images as demo)

OUTPUT_DIR="outputs_ultrasound_ddrm_upgrade"

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
echo "  - Tissue protection: $TISSUE_PROTECTION"
echo "  - Verbose tissue logs: $VERBOSE_TISSUE"
echo "  - Enhanced tissue detection: $ENHANCED_TISSUE_DETECTION"
echo "  - Tissue detection mode: $TISSUE_DETECTION_MODE"
echo "  - CLAHE clip limit: $CLAHE_CLIP_LIMIT"
echo "  - Complete blind zone removal: $COMPLETE_BLIND_ZONE_REMOVAL"
echo "  - Preserve background: $PRESERVE_BACKGROUND"
echo "  - V3-V7 Thresholds: Now handled by dataset building approach with radius_map"
echo "  - Mask cleaning: Tissue min size=${TISSUE_MIN_SIZE}, Blind zone min size=${BLIND_ZONE_MIN_SIZE}"
echo "  - Natural restoration: ${NATURAL_RESTORATION}"
echo "  - Distortion factors: Tissue=${TISSUE_DISTORTION_FACTOR}, BlindZone=${BLIND_ZONE_DISTORTION_FACTOR}, Background=${BACKGROUND_DISTORTION_FACTOR}"
echo "  - Noise factors: Tissue=${TISSUE_NOISE_FACTOR}, BlindZone=${BLIND_ZONE_NOISE_FACTOR}, Background=${BACKGROUND_NOISE_FACTOR}"
if [ "$OPTUNA_MODE" = "true" ]; then
    echo "  - Optuna optimization mode: ENABLED (memory-only evaluation)"
fi
if [ "$NO_SAVE_IMAGES" = "true" ]; then
    echo "  - Image saving: DISABLED"
fi
if [ -n "$SAVE_STEPS" ]; then
    echo "  - Save intermediate steps: $SAVE_STEPS"
fi
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
    --threshold_v3 $THRESHOLD_V3 \
    --threshold_v4 $THRESHOLD_V4 \
    --threshold_v5 $THRESHOLD_V5 \
    --threshold_v6 $THRESHOLD_V6 \
    --threshold_v7 $THRESHOLD_V7 \
    $([ -n "$SAVE_STEPS" ] && echo "--save_steps $SAVE_STEPS") \
    $([ "$TISSUE_PROTECTION" = "true" ] && echo "--tissue_protection") \
    $([ "$VERBOSE_TISSUE" = "true" ] && echo "--verbose_tissue") \
    $([ "$ENHANCED_TISSUE_DETECTION" = "true" ] && echo "--enhanced_tissue_detection") \
    --tissue_detection_mode $TISSUE_DETECTION_MODE \
    --clahe_clip_limit $CLAHE_CLIP_LIMIT \
    --min_tissue_size_factor $MIN_TISSUE_SIZE_FACTOR \
    $([ "$COMPLETE_BLIND_ZONE_REMOVAL" = "true" ] && echo "--complete_blind_zone_removal") \
    $([ "$PRESERVE_BACKGROUND" = "true" ] && echo "--preserve_background") \
    --tissue_min_size $TISSUE_MIN_SIZE \
    --blind_zone_min_size $BLIND_ZONE_MIN_SIZE \
    --cn_on_path $CN_ON_PATH \
    --cy_on_path $CY_ON_PATH \
    --cn_oy_path $CN_OY_PATH \
    --cy_oy_path $CY_OY_PATH \
    --test_images_path $TEST_PATH \
    --artifact_save_dir "${OUTPUT_DIR}/artifacts" \
    --image_folder $OUTPUT_DIR \
    $([ "$OPTUNA_MODE" = "true" ] && echo "--optuna_mode") \
    $([ "$NO_SAVE_IMAGES" = "true" ] && echo "--no_save_images") \
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
echo "4. V3-V7 Donut-based tissue/blind zone separation:"
echo "   - V3: (42, 82, 230), V4: (25, 48, 133), V5: (17, 32, 90), V6: (11, 22, 63), V7: (9, 17, 48)"
echo "   - Thresholds managed by dataset building method with Otsu and percentile processing"
echo "5. Natural physics-based restoration:"
echo "   - Tissue: Protected with ${TISSUE_DISTORTION_FACTOR}x distortion and ${TISSUE_NOISE_FACTOR}x noise"
echo "   - Blind zone: Standard restoration with ${BLIND_ZONE_DISTORTION_FACTOR}x distortion and ${BLIND_ZONE_NOISE_FACTOR}x noise"
echo "   - Background: Minimal processing with ${BACKGROUND_DISTORTION_FACTOR}x distortion and ${BACKGROUND_NOISE_FACTOR}x noise"
echo "6. Natural DDRM inference through H and H_pinv operators (no forced black/white)"
echo "7. Version-specific percentile-based brightness separation"
echo "8. Enhanced multi-method tissue detection for strong blind zones"
echo "9. Mask cleaning with size filtering (tissue: ${TISSUE_MIN_SIZE}px, blind zone: ${BLIND_ZONE_MIN_SIZE}px)"
echo "10. Integration with base DDRM efficient_generalized_steps sampling"
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