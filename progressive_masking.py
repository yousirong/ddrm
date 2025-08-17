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
    v_number = parts[4]  # V3
    last_number = parts[5].split('.')[0]  # 001
    
    mask_pattern = f"{v_number}_{last_number}_mask_alpha2.0.png"
    mask_path = os.path.join(mask_dir, mask_pattern)
    
    if os.path.exists(mask_path):
        return mask_path
    return None

def create_progressive_mask_steps(mask, num_steps=5):
    """
    마스크를 여러 단계로 나누어 점진적으로 적용되는 마스크들을 생성
    """
    # 마스크를 0과 1로 정규화
    mask_normalized = mask.astype(np.float32) / 255.0
    
    progressive_masks = []
    
    for step in range(num_steps + 1):  # 0부터 num_steps까지
        # 각 스텝에서의 마스크 강도 (0.0에서 1.0까지)
        intensity = step / num_steps
        
        # 거리 변환을 사용하여 중심부터 바깥쪽으로 점진적으로 마스킹
        # 마스크의 거리 변환 계산
        mask_binary = (mask_normalized > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        
        # 거리를 정규화
        if dist_transform.max() > 0:
            dist_normalized = dist_transform / dist_transform.max()
        else:
            dist_normalized = dist_transform
        
        # 현재 스텝에서의 threshold 계산
        threshold = 1.0 - intensity  # intensity가 높을수록 더 많은 영역이 마스킹됨
        
        # 점진적 마스크 생성
        step_mask = (dist_normalized >= threshold).astype(np.float32)
        progressive_masks.append(step_mask * mask_normalized)
    
    return progressive_masks

def apply_progressive_denoising(image_path, mask_path, num_steps=5, sigma=1.5):
    """
    이미지에 step별로 점진적으로 denoising을 적용
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
    
    # 가우시안 필터 적용한 denoised 이미지 생성
    denoised_image = gaussian_filter(image.astype(np.float32), sigma=sigma)
    
    # 점진적 마스크들 생성
    progressive_masks = create_progressive_mask_steps(mask, num_steps)
    
    results = []
    
    for step, step_mask in enumerate(progressive_masks):
        # 마스크 영역에만 denoising 적용, 나머지는 원본 유지
        result = image.astype(np.float32) * (1 - step_mask) + denoised_image * step_mask
        results.append(result.astype(np.uint8))
    
    return results

def create_progressive_visualization():
    """
    점진적 마스킹 효과를 보여주는 시각화 생성
    """
    dataset_dir = "/home/juneyonglee/Desktop/ddrm/datasets/test_CY"
    mask_dir = "/home/juneyonglee/Desktop/ddrm/inp_masks/cy_mask/thresh_48_alpha2_0"
    output_dir = "/home/juneyonglee/Desktop/ddrm/outputs_progressive_steps"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1로 끝나는 이미지 파일들 중 몇 개만 선택 (데모용)
    image_files = glob.glob(os.path.join(dataset_dir, "*001.bmp"))
    demo_files = image_files[:3]  # 처음 3개만 사용
    
    num_steps = 6  # 0부터 6까지 7단계
    
    print(f"Step별 점진적 마스킹 효과 생성 중...")
    
    for image_path in demo_files:
        image_name = os.path.basename(image_path).split('.')[0]
        print(f"처리 중: {image_name}")
        
        # 대응하는 마스크 찾기
        mask_path = find_matching_mask(image_path, mask_dir)
        
        if mask_path is None:
            print(f"  마스크를 찾을 수 없습니다: {image_name}")
            continue
        
        # 점진적 denoising 적용
        step_results = apply_progressive_denoising(image_path, mask_path, num_steps, sigma=2.0)
        
        if step_results is None:
            print(f"  처리 실패: {image_name}")
            continue
        
        # 각 스텝별로 저장
        for step, result in enumerate(step_results):
            output_filename = f"{image_name}_step_{step:02d}.bmp"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result)
        
        print(f"  저장 완료: {len(step_results)}개 스텝 이미지")
        
        # 하나의 이미지에 대해 모든 스텝을 합친 이미지도 생성
        create_combined_steps_image(step_results, image_name, output_dir)
    
    print(f"\n결과는 {output_dir}에 저장되었습니다.")

def create_combined_steps_image(step_results, image_name, output_dir):
    """
    모든 스텝을 하나의 이미지에 나란히 배치
    """
    if not step_results:
        return
    
    h, w = step_results[0].shape
    num_steps = len(step_results)
    
    # 가로로 나란히 배치 (3x3 또는 적절한 grid로)
    cols = min(4, num_steps)  # 최대 4열
    rows = (num_steps + cols - 1) // cols  # 필요한 행 수
    
    combined_h = h * rows
    combined_w = w * cols
    combined_image = np.zeros((combined_h, combined_w), dtype=np.uint8)
    
    for i, step_result in enumerate(step_results):
        row = i // cols
        col = i % cols
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        combined_image[y_start:y_end, x_start:x_end] = step_result
        
        # 스텝 번호 텍스트 추가
        cv2.putText(combined_image, f"Step {i}", 
                   (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    
    # 합쳐진 이미지 저장
    combined_filename = f"{image_name}_all_steps.bmp"
    combined_path = os.path.join(output_dir, combined_filename)
    cv2.imwrite(combined_path, combined_image)

if __name__ == "__main__":
    create_progressive_visualization()