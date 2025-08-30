#!/usr/bin/env python3
"""
ultrasound_runner.py 파일의 _enhance_tissue_pixels 함수를 새 버전으로 교체하는 스크립트
"""

import re

# 새 함수 내용 (들여쓰기 포함)
new_function_content = '''    def _enhance_tissue_pixels(self, restored_image, tissue_mask, original_image, version):
        """
        V3~V7 버전별 차등 질감 보존: 조직의 percentile 이상 부분만 원본 픽셀값 그대로 적용
        V3(85th): 상위 15%, V4(80th): 상위 20%, V5(75th): 상위 25%, V6(65th): 상위 35%, V7(55th): 상위 45%
        나머지 모든 영역은 복원 결과 완전 보존
        """
        if tissue_mask is None:
            logger.info("조직 마스크가 없어 후처리를 건너뜁니다")
            return restored_image
        
        # 텐서를 numpy로 변환
        if isinstance(restored_image, torch.Tensor):
            restored_np = restored_image.cpu().numpy()
        else:
            restored_np = restored_image
            
        if isinstance(original_image, torch.Tensor):
            original_np = original_image.cpu().numpy()
        else:
            original_np = original_image
            
        # 텐서 형태 맞추기
        if len(restored_np.shape) == 3:
            restored_np = restored_np[0]  # (1, H, W) -> (H, W)
        if len(original_np.shape) == 3:
            original_np = original_np[0]  # (1, H, W) -> (H, W)
            
        # 조직 마스크 형태 확인
        if isinstance(tissue_mask, torch.Tensor):
            tissue_mask = tissue_mask.cpu().numpy()
        
        # 복원된 이미지를 기본으로 사용 (모든 영역 보존)
        enhanced_np = restored_np.copy()
        tissue_region = tissue_mask > 0.1
        
        if not np.any(tissue_region):
            logger.warning(f"{version} 조직 영역이 없어 후처리를 건너뜁니다")
            return restored_image
        
        # 버전별 차등 percentile 임계값 (V3: 조직 많음 → 높은 임계값, V7: 조직 적음 → 낮은 임계값)
        version_percentiles = {
            'V3': 85,  # 대형 블라인드존: 상위 15%만 질감 보존
            'V4': 80,  # 중대형 블라인드존: 상위 20%만 질감 보존
            'V5': 75,  # 중형 블라인드존: 상위 25%만 질감 보존
            'V6': 65,  # 소형 블라인드존: 상위 35%만 질감 보존
            'V7': 55   # 최소형 블라인드존: 상위 45%만 질감 보존
        }
        
        # 버전별 percentile 임계값 사용 (기본값: V5)
        percentile_threshold = version_percentiles.get(version, 75)
        
        tissue_pixels = original_np[tissue_region]
        strong_tissue_threshold = np.percentile(tissue_pixels, percentile_threshold)
        strong_tissue_region = tissue_region & (original_np >= strong_tissue_threshold)
        
        if np.any(strong_tissue_region):
            # 버전별 percentile 이상 부분에 원본 픽셀값 분포 그대로 적용
            enhanced_np[strong_tissue_region] = original_np[strong_tissue_region]
            
            strong_count = np.sum(strong_tissue_region)
            total_tissue_count = np.sum(tissue_region)
            preservation_percentage = 100 - percentile_threshold  # 보존되는 조직 비율
            
            logger.info(f"{version} {percentile_threshold}th percentile 이상 원본 픽셀값 직접 적용: {strong_count}/{total_tissue_count}개 픽셀 (임계값: {strong_tissue_threshold:.3f})")
            logger.info(f"{version} 조직 질감 보존 비율: 상위 {preservation_percentage}% ({strong_count}개 픽셀)")
            logger.info(f"{version} 나머지 모든 영역 완전 보존: {total_tissue_count - strong_count}개 조직 픽셀 + 비조직 영역 모두 복원 결과 유지")
        else:
            logger.info(f"{version} {percentile_threshold}th percentile 이상 강한 조직 없음 - 모든 영역을 복원 결과 그대로 유지")
        
        logger.info(f"{version} 버전별 차등 처리: {percentile_threshold}th percentile 이상만 원본 픽셀값 직접 적용, 나머지 전부 복원 결과 유지")
        
        # 값 범위 클램핑
        enhanced_np = np.clip(enhanced_np, 0, 1)
        
        # torch.Tensor로 변환하여 반환
        return torch.from_numpy(enhanced_np).float()
'''

# 파일 읽기
with open('ultrasound_runner.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 함수 패턴을 찾아 교체
# 정규표현식으로 def _enhance_tissue_pixels부터 다음 def까지 찾기
pattern = r'(    def _enhance_tissue_pixels\(self, restored_image, tissue_mask, original_image, version\):.*?)(?=    def \w+|\Z)'
match = re.search(pattern, content, re.DOTALL)

if match:
    print("함수를 찾았습니다. 교체 중...")
    # 기존 함수를 새 함수로 교체
    new_content = content.replace(match.group(1), new_function_content)
    
    # 파일에 쓰기
    with open('ultrasound_runner.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("함수 교체 완료!")
else:
    print("함수를 찾을 수 없습니다.")