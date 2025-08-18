#!/usr/bin/env python3
"""
간단한 CY→CN 번역 테스트 스크립트
"""

import torch
import sys
import os
sys.path.append('/home/juneyonglee/Desktop/ddrm')

from ultra_main import (
    InpaintConfig, CYtoCNTranslationDataset, mae_translation, 
    load_grayscale, to_tensor, data_transform
)
from pathlib import Path

def test_cy_to_cn_dataset():
    """CY→CN 데이터셋 테스트"""
    print("=== CY→CN 데이터셋 테스트 ===")
    
    cfg = InpaintConfig(
        translation_mode=True,
        images_dir="/home/juneyonglee/Desktop/ddrm/datasets/test_CY",
        cn_targets_dir="/home/juneyonglee/Desktop/ddrm/datasets/test_CN",
        masks_dir="/home/juneyonglee/Desktop/ddrm/inp_masks/cy_mask/thresh_48_alpha2_0",
        mask_suffix="_mask_alpha2.0.png",
        image_size=512
    )
    
    dataset = CYtoCNTranslationDataset(
        cy_images_dir=cfg.images_dir,
        cn_targets_dir=cfg.cn_targets_dir,
        masks_dir=cfg.masks_dir,
        image_size=cfg.image_size,
        data_channels=cfg.data_channels,
        model_channels=cfg.model_in_channels,
        mask_suffix=cfg.mask_suffix,
        mask_bin_thresh=cfg.mask_bin_thresh,
        known_is_white=cfg.known_is_white,
        invert_mask=cfg.invert_mask,
        skip_missing=True
    )
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 첫 번째 샘플 테스트
    cy_x, cn_x, mask, cy_path, cn_path, mask_path = dataset[0]
    print(f"CY 이미지: {Path(cy_path).name}")
    print(f"CN 이미지: {Path(cn_path).name}")
    print(f"마스크: {Path(mask_path).name}")
    print(f"CY 텐서 크기: {cy_x.shape}")
    print(f"CN 텐서 크기: {cn_x.shape}")
    print(f"마스크 크기: {mask.shape}")
    
    # MAE 계산 테스트
    cy_x_batch = cy_x.unsqueeze(0)  # [1,C,H,W]
    cn_x_batch = cn_x.unsqueeze(0)  # [1,C,H,W]
    
    # 동일한 이미지로 MAE (0에 가까워야 함)
    mae_same = mae_translation(cn_x_batch, cn_x_batch)
    print(f"동일 이미지 MAE: {mae_same:.6f}")
    
    # CY와 CN 간 MAE
    mae_cy_cn = mae_translation(cy_x_batch, cn_x_batch)
    print(f"CY→CN MAE: {mae_cy_cn:.6f}")
    
    return dataset

def test_mae_calculation():
    """MAE 계산 함수 테스트"""
    print("\n=== MAE 계산 테스트 ===")
    
    # 간단한 테스트 텐서 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [-1,1] 범위의 가짜 이미지들
    pred = torch.randn(1, 3, 64, 64, device=device) * 0.5  # 예측
    target1 = pred.clone()  # 동일한 타겟
    target2 = torch.randn(1, 3, 64, 64, device=device) * 0.5  # 다른 타겟
    
    mae_same = mae_translation(pred, target1)
    mae_diff = mae_translation(pred, target2)
    
    print(f"동일 이미지 MAE: {mae_same:.6f} (0에 가까워야 함)")
    print(f"다른 이미지 MAE: {mae_diff:.6f} (0보다 커야 함)")
    
    assert mae_same < 1e-5, f"동일 이미지 MAE가 너무 큼: {mae_same}"
    assert mae_diff > mae_same, f"다른 이미지 MAE가 동일 이미지보다 작음"
    
    print("✓ MAE 계산 함수가 올바르게 작동합니다")

if __name__ == "__main__":
    test_mae_calculation()
    dataset = test_cy_to_cn_dataset()
    print(f"\n✓ CY→CN 번역 설정이 성공적으로 구현되었습니다!")
    print(f"✓ {len(dataset)}개의 CY-CN 이미지 쌍이 준비되었습니다.")
    print("\n사용법:")
    print("python ultra_main.py --translation-mode \\")
    print("  --images-dir /home/juneyonglee/Desktop/ddrm/datasets/test_CY \\")
    print("  --cn-targets-dir /home/juneyonglee/Desktop/ddrm/datasets/test_CN \\")
    print("  --masks-dir /home/juneyonglee/Desktop/ddrm/inp_masks/cy_mask/thresh_48_alpha2_0 \\")
    print("  --mask-suffix '_mask_alpha2.0.png' \\")
    print("  --output-dir /home/juneyonglee/Desktop/ddrm/outputs_cy_to_cn \\")
    print("  --ckpt-path /home/juneyonglee/Desktop/ddrm/final_mixed_training_model.pt \\")
    print("  --optuna --optuna-trials 50 --optuna-samples 10")