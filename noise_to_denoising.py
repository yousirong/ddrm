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

def add_gaussian_noise_to_masked_area(image, mask, noise_std=30):
    """
    마스크 영역에만 가우시안 노이즈를 추가
    """
    # 마스크를 0과 1로 정규화
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 가우시안 노이즈 생성
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    
    # 마스크 영역에만 노이즈 추가
    noisy_image = image.astype(np.float32) + noise * mask_normalized
    
    # 픽셀 값을 0-255 범위로 클리핑
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)

def create_progressive_denoising_steps(noisy_image, original_image, mask, num_steps=6):
    """
    노이즈가 있는 이미지를 점진적으로 denoising하는 단계별 이미지들 생성
    """
    # 마스크를 0과 1로 정규화
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 거리 변환을 사용하여 마스크 내에서 중심부터 바깥쪽으로 점진적 복원
    mask_binary = (mask_normalized > 0.5).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    
    # 거리를 정규화
    if dist_transform.max() > 0:
        dist_normalized = dist_transform / dist_transform.max()
    else:
        dist_normalized = dist_transform
    
    # 다양한 강도의 가우시안 필터 적용
    sigmas = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]  # 강한 필터부터 약한 필터까지
    
    results = []
    results.append(noisy_image)  # Step 0: 노이즈가 있는 원본
    
    for step in range(1, num_steps + 1):
        # 현재 스텝에서의 복원 진행도 (0.0에서 1.0까지)
        progress = step / num_steps
        
        # 거리 기반 threshold (중심부터 바깥쪽으로 점진적 복원)
        threshold = 1.0 - progress
        restoration_mask = (dist_normalized >= threshold).astype(np.float32) * mask_normalized
        
        # 현재 스텝에서 사용할 sigma 값
        sigma_idx = min(step - 1, len(sigmas) - 1)
        current_sigma = sigmas[sigma_idx]
        
        # 가우시안 필터로 denoising
        denoised = gaussian_filter(noisy_image.astype(np.float32), sigma=current_sigma)
        
        # 복원된 영역과 아직 노이즈가 있는 영역 결합
        result = noisy_image.astype(np.float32) * (1 - restoration_mask) + denoised * restoration_mask
        
        # 마스크 외부는 항상 원본 유지
        final_result = original_image.astype(np.float32) * (1 - mask_normalized) + result * mask_normalized
        
        results.append(final_result.astype(np.uint8))
    
    return results

def create_noise_to_denoising_visualization():
    """
    노이즈 추가 -> 점진적 denoising 과정을 보여주는 시각화 생성
    """
    dataset_dir = "/home/juneyonglee/Desktop/ddrm/datasets/test_CY"
    mask_dir = "/home/juneyonglee/Desktop/ddrm/inp_masks/cy_mask/thresh_48_alpha2_0"
    output_dir = "/home/juneyonglee/Desktop/ddrm/outputs_noise_to_denoising"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1로 끝나는 이미지 파일들 중 몇 개만 선택 (데모용)
    image_files = glob.glob(os.path.join(dataset_dir, "*001.bmp"))
    demo_files = image_files[:3]  # 처음 3개만 사용
    
    num_steps = 6  # 노이즈 -> 6단계 denoising
    
    print(f"노이즈 -> Denoising 과정 시각화 생성 중...")
    
    # 재현 가능한 결과를 위해 시드 고정
    np.random.seed(42)
    
    for image_path in demo_files:
        image_name = os.path.basename(image_path).split('.')[0]
        print(f"처리 중: {image_name}")
        
        # 이미지와 마스크 로드
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            print(f"  이미지를 읽을 수 없습니다: {image_path}")
            continue
        
        mask_path = find_matching_mask(image_path, mask_dir)
        if mask_path is None:
            print(f"  마스크를 찾을 수 없습니다: {image_name}")
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  마스크를 읽을 수 없습니다: {mask_path}")
            continue
        
        # 마스크 크기를 이미지와 맞춤
        if mask.shape != original_image.shape:
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        
        # 마스크 영역에 노이즈 추가
        noisy_image = add_gaussian_noise_to_masked_area(original_image, mask, noise_std=40)
        
        # 점진적 denoising 단계들 생성
        step_results = create_progressive_denoising_steps(noisy_image, original_image, mask, num_steps)
        
        # 각 스텝별로 저장
        for step, result in enumerate(step_results):
            if step == 0:
                output_filename = f"{image_name}_noisy.bmp"
            else:
                output_filename = f"{image_name}_denoise_step_{step:02d}.bmp"
            
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, result)
        
        # 원본 이미지도 저장 (비교용)
        original_filename = f"{image_name}_original.bmp"
        original_path = os.path.join(output_dir, original_filename)
        cv2.imwrite(original_path, original_image)
        
        print(f"  저장 완료: 원본 + 노이즈 + {num_steps}단계 denoising")
        
        # 모든 단계를 하나의 이미지로 결합
        create_combined_process_image(original_image, step_results, image_name, output_dir)
    
    print(f"\n결과는 {output_dir}에 저장되었습니다.")

def create_combined_process_image(original_image, step_results, image_name, output_dir):
    """
    원본 -> 노이즈 -> denoising 과정을 하나의 이미지에 나란히 배치
    """
    if not step_results:
        return
    
    h, w = original_image.shape
    # 원본 + 노이즈 + denoising 단계들
    total_images = 1 + len(step_results)  # 원본 + step_results
    
    # 적절한 grid 크기 계산
    cols = min(4, total_images)
    rows = (total_images + cols - 1) // cols
    
    combined_h = h * rows
    combined_w = w * cols
    combined_image = np.zeros((combined_h, combined_w), dtype=np.uint8)
    
    # 이미지들 배치
    all_images = [original_image] + step_results
    labels = ["Original"] + ["Noisy"] + [f"Denoise {i}" for i in range(1, len(step_results))]
    
    for i, (img, label) in enumerate(zip(all_images, labels)):
        row = i // cols
        col = i % cols
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        combined_image[y_start:y_end, x_start:x_end] = img
        
        # 라벨 텍스트 추가
        cv2.putText(combined_image, label, 
                   (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    
    # 합쳐진 이미지 저장
    combined_filename = f"{image_name}_noise_to_denoise_process.bmp"
    combined_path = os.path.join(output_dir, combined_filename)
    cv2.imwrite(combined_path, combined_image)

if __name__ == "__main__":
    create_noise_to_denoising_visualization()