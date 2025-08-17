import os
import cv2
import numpy as np
from pathlib import Path
import glob
from scipy.ndimage import gaussian_filter

def find_matching_mask(image_path, mask_dir):
    """
    이미지 파일에 대응하는 마스크 파일을 찾는 함수
    """
    image_name = os.path.basename(image_path)
    
    # CY_OY_PC_D000_V3_001.bmp -> V3_001
    parts = image_name.split('_')
    # parts = ['CY', 'OY', 'PC', 'D000', 'V3', '001.bmp']
    v_number = parts[4]  # V3
    last_number = parts[5].split('.')[0]  # 001
    
    mask_pattern = f"{v_number}_{last_number}_mask_alpha2.0.png"
    mask_path = os.path.join(mask_dir, mask_pattern)
    
    if os.path.exists(mask_path):
        return mask_path
    return None

def apply_gaussian_denoising_to_masked_area(image_path, mask_path, sigma=1.0):
    """
    마스크 영역에만 가우시안 denoising을 적용하는 함수
    """
    # 이미지 읽기 (흑백으로)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    # 마스크 읽기
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"마스크를 읽을 수 없습니다: {mask_path}")
        return None
    
    # 마스크 크기를 이미지와 맞춤
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # 마스크를 0과 1로 정규화
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 가우시안 필터 적용
    denoised_image = gaussian_filter(image.astype(np.float32), sigma=sigma)
    
    # 마스크 영역에만 denoising 적용, 나머지는 원본 유지
    result = image.astype(np.float32) * (1 - mask_normalized) + denoised_image * mask_normalized
    
    return result.astype(np.uint8)

def process_images_ending_with_1():
    """
    1로 끝나는 데이터셋 이미지들을 처리하는 메인 함수
    """
    dataset_dir = "/home/juneyonglee/Desktop/ddrm/datasets/test_CY"
    mask_dir = "/home/juneyonglee/Desktop/ddrm/inp_masks/cy_mask/thresh_48_alpha2_0"
    output_dir = "/home/juneyonglee/Desktop/ddrm/outputs_masked_denoising"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1로 끝나는 이미지 파일들 찾기
    image_files = glob.glob(os.path.join(dataset_dir, "*1.bmp"))
    
    processed_count = 0
    total_count = len(image_files)
    
    print(f"총 {total_count}개의 이미지를 처리합니다...")
    
    for image_path in image_files:
        print(f"처리 중: {os.path.basename(image_path)}")
        
        # 대응하는 마스크 찾기
        mask_path = find_matching_mask(image_path, mask_dir)
        
        if mask_path is None:
            print(f"  마스크를 찾을 수 없습니다: {os.path.basename(image_path)}")
            continue
        
        print(f"  마스크 발견: {os.path.basename(mask_path)}")
        
        # 마스크 영역에 가우시안 denoising 적용
        result = apply_gaussian_denoising_to_masked_area(image_path, mask_path, sigma=1.5)
        
        if result is not None:
            # 결과 저장
            output_filename = f"denoised_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result)
            print(f"  저장 완료: {output_filename}")
            processed_count += 1
        else:
            print(f"  처리 실패: {os.path.basename(image_path)}")
    
    print(f"\n처리 완료: {processed_count}/{total_count}개 이미지")
    print(f"결과는 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    process_images_ending_with_1()