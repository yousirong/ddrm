#!/bin/bash

for i in $(seq 0.0 0.1 0.9); do
    # 소수 첫째 자리까지만 남기기 (bash에서 부동소수점 처리)
    sigma=$(printf "%.1f" "$i")
    input_dir="imagenet_inp$sigma"
    log_file="log_sigma_${sigma}.txt"

    echo "▶ Running with sigma_0 = $sigma, input = $input_dir"

    python main.py --ni \
        --config imagenet_256.yml \
        --doc imagenet_inpainting_stronger \
        --timesteps 20 \
        --eta 0.85 \
        --etaB 1 \
        --deg inp \
        --sigma_0 $sigma \
        -i $input_dir \
        > "$log_file" 2>&1
done
