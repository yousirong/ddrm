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