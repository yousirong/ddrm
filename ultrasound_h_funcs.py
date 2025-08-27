import torch
import numpy as np
import os
from functions.svd_replacement import H_functions
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class UltrasoundBlindZone(H_functions):
    """
    Enhanced H_functions implementation for ultrasound blind zone removal
    Handles version-specific (V3-V7) donut-shaped blind zones with physical modeling
    
    Based on DDRM methodology with ultrasound-specific modifications:
    - z_est = Average(CY_ON - CN_ON): structural noise estimation 
    - H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²: distortion operator
    - Physics-based modeling: blind zone as physical distortion, not masking
    """
    
    def __init__(self, channels, img_size, device, version=None, noise_pattern=None, distortion_factor=0.025, noise_factor=1.0, 
                 enhanced_tissue_detection=True, tissue_detection_mode='multi', clahe_clip_limit=3.0, 
                 min_tissue_size_factor=1.0, complete_blind_zone_removal=True, preserve_background=True,
                 version_thresholds=None, tissue_min_size=200, blind_zone_min_size=100):
        self.channels = channels
        self.img_size = img_size
        self.device = device
        self.distortion_factor = distortion_factor
        self.noise_factor = noise_factor
        
        # Enhanced detection parameters
        self.enhanced_tissue_detection = enhanced_tissue_detection
        self.tissue_detection_mode = tissue_detection_mode
        self.clahe_clip_limit = clahe_clip_limit
        self.min_tissue_size_factor = min_tissue_size_factor
        self.complete_blind_zone_removal = complete_blind_zone_removal
        self.preserve_background = preserve_background
        
        # V3~V7 도넛 기반 파라미터들
        self.version_thresholds = version_thresholds or {}
        self.tissue_min_size = tissue_min_size
        self.blind_zone_min_size = blind_zone_min_size
        
        logger.info(f"*** UltrasoundBlindZone.__init__: distortion_factor={self.distortion_factor}, noise_factor={self.noise_factor} ***")
        logger.info(f"*** Enhanced detection: {self.enhanced_tissue_detection}, mode: {self.tissue_detection_mode}, CLAHE: {self.clahe_clip_limit} ***")
        logger.info(f"*** Complete blind zone removal: {self.complete_blind_zone_removal}, Preserve background: {self.preserve_background} ***")
        self.version = version
        
        # 조직 보호 관련 변수들 (H/H_pinv는 변경하지 않고 디노이징에서만 사용)
        self.tissue_mask = None
        self.blind_zone_processing_mask = None
        
        # Enhanced version-specific parameters based on blind zone physics
        # (outer_radius, inner_radius, distortion_strength, noise_level)
        self.VERSION_PARAMS = {
            "V3": {"outer_r": 220, "inner_r": 85, "strength": 0.8, "noise": 0.05},
            "V4": {"outer_r": 130, "inner_r": 50, "strength": 0.9, "noise": 0.06}, 
            "V5": {"outer_r": 90, "inner_r": 30, "strength": 1.0, "noise": 0.07},
            "V6": {"outer_r": 60, "inner_r": 20, "strength": 1.1, "noise": 0.08},
            "V7": {"outer_r": 45, "inner_r": 15, "strength": 1.2, "noise": 0.09}
        }
        
        # Create physics-based distortion model for this version
        if version and version in self.VERSION_PARAMS:
            params = self.VERSION_PARAMS[version]
            self.distortion_mask = self._create_physics_distortion_mask(
                img_size, img_size, params["outer_r"], params["inner_r"], 
                params["strength"]
            )
            self.noise_level = params["noise"]
            logger.info(f"Created physics-based distortion model for {version}")
            logger.info(f"  - Outer radius: {params['outer_r']}, Inner radius: {params['inner_r']}")
            logger.info(f"  - Distortion strength: {params['strength']}, Noise level: {params['noise']}")
        else:
            # Default combined distortion model
            self.distortion_mask = self._create_combined_distortion_mask(img_size, img_size)
            self.noise_level = 0.07
            logger.info("Created combined physics-based distortion model")
            
        self.mask_tensor = torch.from_numpy(self.distortion_mask).float().to(device)
        
        # Store noise pattern for z_est = Average(CY_ON - CN_ON)
        if noise_pattern is not None:
            self.noise_pattern = torch.from_numpy(noise_pattern).float().to(device)
        else:
            self.noise_pattern = torch.zeros(img_size, img_size).to(device)
            
        # Pre-compute SVD components for efficient DDRM sampling
        self._compute_svd_components()
        
    def _create_physics_distortion_mask(self, height, width, outer_radius, inner_radius, strength):
        """
        Create physics-based distortion model for blind zone
        Instead of binary masking, models physical ultrasound distortion
        """
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create smooth distortion profile (not binary)
        distortion_mask = np.zeros((height, width), dtype=np.float32)
        
        # Physics-based distortion model: gradual falloff
        in_blind_zone = (distance >= inner_radius) & (distance <= outer_radius)
        
        if np.any(in_blind_zone):
            # Normalized distance within blind zone [0, 1]
            zone_distance = (distance - inner_radius) / (outer_radius - inner_radius)
            zone_distance = np.clip(zone_distance, 0, 1)
            
            # Physics-based distortion profile (bell curve)
            # Maximum distortion in middle of blind zone
            distortion_profile = np.exp(-((zone_distance - 0.5) * 4) ** 2) * strength
            distortion_mask[in_blind_zone] = distortion_profile[in_blind_zone]
            
        return distortion_mask
    
    def _create_combined_distortion_mask(self, height, width):
        """Create combined physics-based distortion model for all versions"""
        combined_mask = np.zeros((height, width), dtype=np.float32)
        
        for version, params in self.VERSION_PARAMS.items():
            version_mask = self._create_physics_distortion_mask(
                height, width, params["outer_r"], params["inner_r"], params["strength"]
            )
            # Combine using maximum distortion
            combined_mask = np.maximum(combined_mask, version_mask)
            
        return combined_mask
    
    def detect_and_protect_tissue(self, input_image):
        """
        V3~V7 도넛 형태 내부에서 조직과 블라인드존 구분 검출:
        1. V3~V7 버전에 따른 도넛 형태 영역 결정
        2. 도넛 영역 내에서 밝은 조직과 어두운 블라인드존 구분
        3. 블라인드존만 제거하고 조직은 보호
        """
        if isinstance(input_image, torch.Tensor):
            if len(input_image.shape) == 4:  # (B, C, H, W)
                image_np = input_image[0, 0].cpu().numpy()
            elif len(input_image.shape) == 3:
                image_np = input_image[0].cpu().numpy() if input_image.shape[0] == 1 else input_image.cpu().numpy()
            else:
                image_np = input_image.cpu().numpy()
        else:
            image_np = input_image
        
        # === 1단계: 버전별 도넛 영역 결정 ===
        donut_region = self._create_version_donut_region(image_np.shape, self.version)
        
        # === 2단계: 도넛 내에서 조직과 블라인드존 구분 ===
        self.tissue_mask, blind_zone_mask = self._separate_tissue_and_blind_zone_in_donut(
            image_np, donut_region, self.version
        )
        
        # === 3단계: 블라인드존만 제거를 위한 마스크 ===
        self.blind_zone_processing_mask = blind_zone_mask
        
        # 로그 출력
        donut_coverage = np.sum(donut_region) / donut_region.size * 100
        tissue_coverage = np.sum(self.tissue_mask) / self.tissue_mask.size * 100
        blind_zone_coverage = np.sum(blind_zone_mask) / blind_zone_mask.size * 100
        
        logger.info(f"{self.version} 도넛 영역에서 조직/블라인드존 구분 완료:")
        logger.info(f"  - 도넛 영역: {donut_coverage:.1f}%")
        logger.info(f"  - 조직 영역 (보호): {tissue_coverage:.1f}%")
        logger.info(f"  - 블라인드존 영역 (제거): {blind_zone_coverage:.1f}%")
        
        return self.tissue_mask, self.blind_zone_processing_mask
    
    
    
    
    
    
    
    def _create_version_donut_region(self, shape, version):
        """
        V3~V7 버전별 도넛 형태 영역 생성 (하드코딩)
        조직과 블라인드존이 모두 있는 영역
        """
        height, width = shape
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 버전별 도넛 매개변수 (하드코딩)
        version_params = {
            'V3': {'inner_r': 85, 'outer_r': 220},   # 대형 도넛
            'V4': {'inner_r': 50, 'outer_r': 130},   # 중대형 도넛
            'V5': {'inner_r': 30, 'outer_r': 90},    # 중형 도넛
            'V6': {'inner_r': 20, 'outer_r': 60},    # 소형 도넛
            'V7': {'inner_r': 15, 'outer_r': 45}     # 최소형 도넛
        }
        
        # 기본값은 V5 사용
        params = version_params.get(version, version_params['V5'])
        inner_r, outer_r = params['inner_r'], params['outer_r']
        
        # 도넛 영역 마스크 생성
        donut_mask = ((distance >= inner_r) & (distance <= outer_r)).astype(np.float32)
        
        logger.info(f"{version} 도넛 영역 생성: inner_r={inner_r}, outer_r={outer_r}")
        logger.info(f"  - 도넛 영역 비율: {np.sum(donut_mask) / donut_mask.size * 100:.1f}%")
        
        return donut_mask
    
    def _separate_tissue_and_blind_zone_in_donut(self, image, donut_region, version):
        """
        도넛 영역 내에서 조직(밝은 영역)과 블라인드존(어두운 영역) 구분
        조직은 블라인드존보다 무조건 밝음
        """
        # 도넛 영역 내의 픽셀값만 추출
        donut_pixels = image * donut_region
        donut_values = donut_pixels[donut_region > 0.5]  # 도넛 영역의 픽셀값들
        
        if len(donut_values) == 0:
            logger.warning(f"{version} 도넛 영역에 픽셀이 없음")
            return np.zeros_like(donut_region), np.zeros_like(donut_region)
        
        # 버전별 조직/블라인드존 구분 임계값 (명령줄 인수로 전달된 값 사용)
        default_thresholds = {
            'V3': {'tissue_percentile': 65, 'blind_zone_percentile': 35},  # V3: 밝기 차이가 큰 경우
            'V4': {'tissue_percentile': 70, 'blind_zone_percentile': 40},  # V4: 중간 정도 차이
            'V5': {'tissue_percentile': 75, 'blind_zone_percentile': 45},  # V5: 표준 설정
            'V6': {'tissue_percentile': 80, 'blind_zone_percentile': 50},  # V6: 차이가 작은 경우
            'V7': {'tissue_percentile': 85, 'blind_zone_percentile': 55}   # V7: 가장 미세한 차이
        }
        
        # 전달된 값이 있으면 사용, 없으면 기본값 사용
        if self.version_thresholds and version in self.version_thresholds:
            thresholds = self.version_thresholds[version]
            logger.info(f"{version} 커스텀 임계값 사용: {thresholds}")
        else:
            thresholds = default_thresholds.get(version, default_thresholds['V5'])
            logger.info(f"{version} 기본 임계값 사용: {thresholds}")
        
        # 도넛 내 픽셀값 분석
        tissue_threshold = np.percentile(donut_values, thresholds['tissue_percentile'])
        blind_zone_threshold = np.percentile(donut_values, thresholds['blind_zone_percentile'])
        
        logger.info(f"{version} 도넛 내 픽셀 분석:")
        logger.info(f"  - 도넛 픽셀값 범위: {np.min(donut_values):.3f} ~ {np.max(donut_values):.3f}")
        logger.info(f"  - 조직 임계값 ({thresholds['tissue_percentile']}%): {tissue_threshold:.3f}")
        logger.info(f"  - 블라인드존 임계값 ({thresholds['blind_zone_percentile']}%): {blind_zone_threshold:.3f}")
        
        # 조직 마스크: 도넛 내에서 밝은 영역
        tissue_mask = np.zeros_like(image)
        tissue_condition = (donut_region > 0.5) & (image >= tissue_threshold)
        tissue_mask[tissue_condition] = 1.0
        
        # 블라인드존 마스크: 도넛 내에서 어두운 영역 (조직 제외)
        blind_zone_mask = np.zeros_like(image)
        blind_zone_condition = (donut_region > 0.5) & (image <= blind_zone_threshold) & (tissue_mask == 0)
        blind_zone_mask[blind_zone_condition] = 1.0
        
        # 마스크 후처리 (노이즈 제거) - 커스텀 최소 크기 사용
        tissue_mask = self._clean_mask(tissue_mask, min_size=self.tissue_min_size)
        blind_zone_mask = self._clean_mask(blind_zone_mask, min_size=self.blind_zone_min_size)
        
        # 결과 로깅
        tissue_coverage = np.sum(tissue_mask) / tissue_mask.size * 100
        blind_zone_coverage = np.sum(blind_zone_mask) / blind_zone_mask.size * 100
        total_donut_coverage = np.sum(donut_region) / donut_region.size * 100
        
        logger.info(f"{version} 조직/블라인드존 구분 결과:")
        logger.info(f"  - 전체 도넛 영역: {total_donut_coverage:.1f}%")
        logger.info(f"  - 조직 영역 (보호): {tissue_coverage:.1f}%")
        logger.info(f"  - 블라인드존 영역 (제거): {blind_zone_coverage:.1f}%")
        
        return tissue_mask, blind_zone_mask
    
    def _clean_mask(self, mask, min_size=100):
        """
        마스크 후처리: 작은 노이즈 영역 제거
        """
        try:
            import cv2
            from scipy import ndimage
            
            # 이진 마스크로 변환
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # 형태학적 연산으로 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # 연결된 컯포너트 분석
            labeled, num_features = ndimage.label(cleaned)
            
            if num_features == 0:
                return np.zeros_like(mask)
            
            # 크기가 충분한 영역만 유지
            final_mask = np.zeros_like(mask, dtype=np.float32)
            for i in range(1, num_features + 1):
                region = (labeled == i)
                if np.sum(region) >= min_size:
                    final_mask[region] = 1.0
            
            # 가우시안 블러로 부드럽게
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 1.0)
            
            return final_mask
            
        except ImportError:
            # OpenCV가 없으면 기본 처리
            return mask
        except:
            return mask
    
    def _detect_tissue_regions(self, image):
        """
        개선된 조직 영역 검출: 블라인드존이 강한 경우에도 조직을 잘 탐지
        - 적응형 다중 검출 방법
        - CLAHE 전처리
        - 지역적 대비 분석
        """
        try:
            import cv2
            from scipy import ndimage
            
            # scikit-image는 선택적으로 사용
            try:
                from skimage import filters
                has_skimage = True
            except ImportError:
                has_skimage = False
                logger.warning("scikit-image not available, using fallback methods")
            
            # 이미지 정규화
            img_norm = (image * 255).astype(np.uint8)
            logger.info(f"원본 이미지 통계: mean={np.mean(img_norm):.1f}, std={np.std(img_norm):.1f}")
            
            # === 1단계: 블라인드존 대응 전처리 ===
            enhanced_img = self._enhance_image_for_tissue_detection(img_norm)
            
            # === 2단계: 검출 방법 선택 ===
            masks = []
            
            if self.tissue_detection_mode == 'simple':
                # 단순 모드: 기본 임계값만
                simple_mask = self._simple_threshold_detection(image)
                masks.append(simple_mask)
                logger.info("단순 검출 모드 사용")
                
            elif self.tissue_detection_mode == 'adaptive':
                # 적응형 모드: 적응형 임계값만
                adaptive_mask = self._adaptive_threshold_detection(enhanced_img)
                masks.append(adaptive_mask)
                logger.info("적응형 검출 모드 사용")
                
            elif self.tissue_detection_mode == 'edge':
                # 엣지 모드: 엣지 검출만
                edge_mask = self._edge_enhanced_detection(enhanced_img)
                masks.append(edge_mask)
                logger.info("엣지 검출 모드 사용")
                
            else:  # multi 모드 (기본)
                # 방법 1: 적응형 임계값
                adaptive_mask = self._adaptive_threshold_detection(enhanced_img)
                masks.append(adaptive_mask)
                
                # 방법 2: 지역 대비 기반 검출
                contrast_mask = self._local_contrast_detection(enhanced_img)
                masks.append(contrast_mask)
                
                # 방법 3: 엣지 강화 검출 (블라인드존이 강한 경우)
                if self._is_strong_blind_zone(img_norm):
                    edge_mask = self._edge_enhanced_detection(enhanced_img)
                    masks.append(edge_mask)
                    logger.info("강한 블라인드존 감지 - 엣지 강화 검출 추가")
                
                logger.info("다중 검출 모드 사용")
            
            # === 3단계: 다중 마스크 융합 ===
            combined_mask = self._combine_detection_masks(masks, enhanced_img)
            
            # === 4단계: 향상된 후처리 ===
            final_mask = self._advanced_post_processing(combined_mask, img_norm)
            
            # 결과 통계
            tissue_coverage = np.sum(final_mask > 0.5) / final_mask.size * 100
            logger.info(f"개선된 조직 검출 완료: {tissue_coverage:.1f}% 영역")
            
            if tissue_coverage > 0:
                logger.info(f"검출 방법 수: {len(masks)}, 최종 마스크 강도: {np.max(final_mask):.2f}")
                return final_mask
            else:
                # 검출 실패 시 폴백 방법
                logger.warning("다중 검출 실패 - 폴백 방법 사용")
                return self._fallback_detection(img_norm)
                
        except ImportError:
            logger.warning("OpenCV 없이 단순 임계값 방법 사용")
            return self._simple_threshold_detection(image)
        except Exception as e:
            logger.warning(f"조직 검출 실패: {e}, 단순 방법 사용")
            return self._simple_threshold_detection(image)
    
    def _enhance_image_for_tissue_detection(self, img_norm):
        """블라인드존 대응 이미지 전처리"""
        try:
            import cv2
            
            # CLAHE로 대비 개선 (설정 가능한 클립 리미트 사용)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8,8))
            enhanced = clahe.apply(img_norm)
            
            # 가우시안 블러로 노이즈 감소
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 언샤프 마스킹으로 디테일 강화
            unsharp = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
            
            logger.info(f"전처리 후 통계: mean={np.mean(unsharp):.1f}, std={np.std(unsharp):.1f}")
            return unsharp
        except:
            return img_norm
    
    def _is_strong_blind_zone(self, img_norm):
        """강한 블라인드존 여부 판단"""
        # 이미지의 다양성과 대비를 분석
        std_val = np.std(img_norm)
        mean_val = np.mean(img_norm)
        
        # 낮은 표준편차와 낮은 평균값은 강한 블라인드존을 의미
        is_strong = std_val < 25 or mean_val < 30
        logger.info(f"블라인드존 강도 분석: std={std_val:.1f}, mean={mean_val:.1f}, strong={is_strong}")
        return is_strong
    
    def _adaptive_threshold_detection(self, enhanced_img):
        """적응형 임계값 기반 조직 검출"""
        try:
            import cv2
            
            # Otsu의 방법으로 자동 임계값 계산
            try:
                from skimage import filters
                otsu_thresh = filters.threshold_otsu(enhanced_img)
            except ImportError:
                # scikit-image 없을 때 OpenCV Otsu 사용
                _, otsu_thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 적응형 임계값 (지역적)
            adaptive_thresh = cv2.adaptiveThreshold(
                enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 2
            )
            
            # 두 방법을 결합
            otsu_mask = (enhanced_img > otsu_thresh * 0.8).astype(np.float32)
            adaptive_mask = (adaptive_thresh / 255.0).astype(np.float32)
            
            combined = np.maximum(otsu_mask * 0.7, adaptive_mask * 0.3)
            logger.info(f"적응형 검출: Otsu={otsu_thresh:.1f}, 검출률={np.sum(combined > 0.3) / combined.size * 100:.1f}%")
            
            return combined
        except:
            # 폴백: 단순 임계값
            thresh = np.percentile(enhanced_img, 75)
            return (enhanced_img > thresh).astype(np.float32)
    
    def _local_contrast_detection(self, enhanced_img):
        """지역 대비 기반 조직 검출"""
        try:
            import cv2
            from scipy import ndimage
            
            # 지역 평균 계산
            kernel = np.ones((15, 15), np.float32) / 225
            local_mean = cv2.filter2D(enhanced_img.astype(np.float32), -1, kernel)
            
            # 지역 표준편차 계산
            local_sq_mean = cv2.filter2D((enhanced_img.astype(np.float32))**2, -1, kernel)
            local_std = np.sqrt(local_sq_mean - local_mean**2)
            
            # 대비 맵 계산
            contrast_map = local_std / (local_mean + 1e-6)
            
            # 높은 대비 영역을 조직으로 간주
            contrast_thresh = np.percentile(contrast_map, 70)
            contrast_mask = (contrast_map > contrast_thresh).astype(np.float32)
            
            # 밝기도 고려
            brightness_thresh = np.percentile(enhanced_img, 60)
            brightness_mask = (enhanced_img > brightness_thresh).astype(np.float32)
            
            # 대비와 밝기 결합
            combined = contrast_mask * 0.6 + brightness_mask * 0.4
            combined = np.clip(combined, 0, 1)
            
            logger.info(f"대비 검출: 대비임계값={contrast_thresh:.3f}, 검출률={np.sum(combined > 0.4) / combined.size * 100:.1f}%")
            return combined
        except:
            # 폴백
            return (enhanced_img > np.mean(enhanced_img)).astype(np.float32)
    
    def _edge_enhanced_detection(self, enhanced_img):
        """엣지 강화 기반 조직 검출 (블라인드존이 강한 경우)"""
        try:
            import cv2
            
            # 소벨 엣지 검출
            sobelx = cv2.Sobel(enhanced_img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(enhanced_img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            
            # 캐니 엣지 검출
            canny = cv2.Canny(enhanced_img, 30, 100)
            
            # 엣지 밀도 계산 (지역적)
            kernel = np.ones((11, 11), np.float32) / 121
            edge_density = cv2.filter2D(canny.astype(np.float32), -1, kernel)
            
            # 높은 엣지 밀도 영역을 조직 경계로 간주
            edge_thresh = np.percentile(edge_density, 80)
            edge_mask = (edge_density > edge_thresh).astype(np.float32)
            
            # 형태학적 연산으로 내부 채우기
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            filled_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
            
            logger.info(f"엣지 검출: 엣지밀도임계값={edge_thresh:.3f}, 검출률={np.sum(filled_mask > 0.5) / filled_mask.size * 100:.1f}%")
            return filled_mask
        except:
            # 폴백
            return (enhanced_img > np.percentile(enhanced_img, 85)).astype(np.float32)
    
    def _combine_detection_masks(self, masks, enhanced_img):
        """다중 검출 마스크 융합"""
        if not masks:
            return np.zeros_like(enhanced_img, dtype=np.float32)
        
        # 가중 평균 융합
        weights = [0.4, 0.4, 0.2] if len(masks) == 3 else [0.5, 0.5]
        weights = weights[:len(masks)]
        
        combined = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            combined += mask * weight
        
        # 투표 방식 추가 (2개 이상의 방법에서 검출된 경우 강화)
        if len(masks) >= 2:
            vote_count = sum([(mask > 0.3).astype(np.float32) for mask in masks])
            consensus = (vote_count >= 2).astype(np.float32)
            combined = np.maximum(combined, consensus * 0.8)
        
        combined = np.clip(combined, 0, 1)
        logger.info(f"마스크 융합: {len(masks)}개 방법, 최대값={np.max(combined):.2f}")
        
        return combined
    
    def _advanced_post_processing(self, combined_mask, img_norm):
        """향상된 후처리"""
        try:
            import cv2
            from scipy import ndimage
            
            # 임계값 적용
            binary_mask = (combined_mask > 0.4).astype(np.uint8)
            
            # 형태학적 연산으로 정리
            # Opening으로 작은 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Closing으로 구멍 채우기
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            # 연결된 컴포넌트 분석
            labeled, num_features = ndimage.label(closed)
            
            if num_features == 0:
                return np.zeros_like(combined_mask)
            
            # 동적 최소 크기 계산 (이미지 품질에 따라)
            img_quality = np.std(img_norm) / np.mean(img_norm) if np.mean(img_norm) > 0 else 0
            base_min_size = 500
            quality_factor = max(0.5, min(2.0, img_quality * 10))
            min_tissue_size = int(base_min_size * quality_factor * self.min_tissue_size_factor)
            
            logger.info(f"후처리: 품질지수={quality_factor:.2f}, 최소크기={min_tissue_size}")
            
            # 크기와 모양 기반 필터링
            final_mask = np.zeros_like(closed, dtype=np.float32)
            valid_regions = 0
            
            for i in range(1, num_features + 1):
                region = (labeled == i)
                region_size = np.sum(region)
                
                if region_size >= min_tissue_size:
                    # 모양 분석 (원형도, 종횡비 등)
                    if self._is_valid_tissue_shape(region):
                        final_mask[region] = 1.0
                        valid_regions += 1
            
            # 가우시안 블러로 경계 부드럽게
            if valid_regions > 0:
                final_mask = cv2.GaussianBlur(final_mask, (7, 7), 1.5)
                final_mask = np.clip(final_mask, 0, 1)
            
            logger.info(f"후처리 완료: {valid_regions}개 유효 영역")
            return final_mask
            
        except:
            # 간단한 후처리
            binary = (combined_mask > 0.5).astype(np.float32)
            return cv2.GaussianBlur(binary, (5, 5), 1.0) if 'cv2' in locals() else binary
    
    def _is_valid_tissue_shape(self, region):
        """조직 모양 유효성 검사"""
        try:
            import cv2
            
            # 컨투어 찾기
            region_uint8 = region.astype(np.uint8)
            contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False
            
            # 가장 큰 컨투어 선택
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            if area < 100:
                return False
            
            # 원형도 계산
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return False
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 경계 상자 종횡비
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 조직의 일반적인 모양 특성
            # - 너무 길쭉하지 않음 (종횡비 0.3~3.0)
            # - 너무 복잡하지 않음 (원형도 > 0.1)
            valid_aspect = 0.3 <= aspect_ratio <= 3.0
            valid_circularity = circularity > 0.1
            
            return valid_aspect and valid_circularity
            
        except:
            return True  # 분석 실패 시 허용
    
    def _fallback_detection(self, img_norm):
        """폴백 검출 방법"""
        # 여러 임계값 시도
        thresholds = [0.15, 0.12, 0.10, 0.08]
        
        for thresh in thresholds:
            mask = (img_norm > thresh * 255).astype(np.float32)
            coverage = np.sum(mask) / mask.size * 100
            
            if 1.0 <= coverage <= 50.0:  # 적절한 범위의 검출률
                logger.info(f"폴백 검출 성공: 임계값={thresh}, 검출률={coverage:.1f}%")
                return mask
        
        # 최후 수단: 매우 낮은 임계값
        return (img_norm > 0.05 * 255).astype(np.float32)
    
    def _simple_threshold_detection(self, image):
        """단순 임계값 검출 (OpenCV 없는 경우)"""
        version_thresholds = {
            'V3': 0.08, 'V4': 0.10, 'V5': 0.12, 'V6': 0.15, 'V7': 0.18  # 임계값 낮춤
        }
        threshold = version_thresholds.get(self.version, 0.12)
        return (image > threshold).astype(np.float32)
    
    def _compute_svd_components(self):
        """
        Compute SVD components for efficient DDRM sampling
        Enhanced for physics-based distortion model but simplified for compatibility
        """
        # Simplified approach for DDRM compatibility
        # Create uniform singular values for the full image
        total_pixels = self.img_size * self.img_size
        
        # Create singular values based on physical distortion model
        # All pixels get singular values, but distorted regions get higher values
        singular_values = np.ones(total_pixels, dtype=np.float32) * 0.1  # Base value
        
        # Enhance singular values in distorted regions
        mask_flat = self.distortion_mask.flatten()
        singular_values = singular_values + mask_flat * 0.9  # Add distortion strength
        
        # Normalize and add regularization
        singular_values = singular_values / np.max(singular_values)
        singular_values = singular_values + 1e-6  # Regularization
        
        self._singulars = torch.from_numpy(singular_values).float().to(self.device)
        
        logger.info(f"Computed {len(self._singulars)} SVD components for full image")
        
    def singulars(self):
        """Returns singular values of the degradation operator"""
        return self._singulars
        
    def add_zeros(self, vec):
        """Add zeros to match dimensions"""
        # For blind zone, this relates to expanding from masked region to full image
        active_pixels = torch.sum(self.mask_tensor > 0.5).item()
        total_pixels = self.img_size * self.img_size
        
        if vec.shape[-1] == active_pixels:
            # Expand from active region to full image
            batch_size = vec.shape[0]
            expanded = torch.zeros(batch_size, total_pixels).to(self.device)
            
            mask_indices = torch.where(self.mask_tensor.flatten() > 0.5)[0]
            expanded[:, mask_indices] = vec
            
            return expanded
        else:
            return vec
    
    def V(self, vec):
        """
        Multiply by V matrix (reconstruction from SVD space)
        For ultrasound physics, V is approximately identity
        """
        # Handle input shapes and ensure correct dimensions
        if len(vec.shape) == 1:
            vec = vec.view(1, -1)
        elif len(vec.shape) == 4:  # Already in image format
            return vec
        
        batch_size = vec.shape[0]
        
        # Simply reshape to image format - V is identity for ultrasound
        if vec.shape[1] == self.img_size * self.img_size:
            return vec.view(batch_size, self.channels, self.img_size, self.img_size)
        else:
            # Handle partial vectors by zero-padding
            full_vec = torch.zeros(batch_size, self.img_size * self.img_size, device=vec.device)
            copy_size = min(vec.shape[1], full_vec.shape[1])
            full_vec[:, :copy_size] = vec[:, :copy_size]
            return full_vec.view(batch_size, self.channels, self.img_size, self.img_size)
        
    def Vt(self, vec):
        """
        Multiply by V transpose (projection to SVD space)
        For ultrasound physics, Vt is approximately identity (flatten)
        """
        # Simple flattening - Vt is identity for ultrasound
        if len(vec.shape) == 4:  # (B, C, H, W)
            return vec.view(vec.shape[0], -1)
        elif len(vec.shape) == 3:  # (B, H, W)  
            return vec.view(vec.shape[0], -1)
        elif len(vec.shape) == 1:  # (features,)
            return vec.view(1, -1)
        else:  # Already 2D
            return vec
        
    def U(self, vec):
        """
        Multiply by U matrix (measurement operator)
        For ultrasound, this applies the physics-based distortion
        """
        if len(vec.shape) == 4:  # (B, C, H, W)
            # Apply physics-based measurement and flatten
            measured = self.H(vec)
            return self.Vt(measured)
        else:
            # Reconstruct, apply measurement, project back
            reconstructed = self.V(vec)
            measured = self.H(reconstructed)
            return self.Vt(measured)
        
    def Ut(self, vec):
        """
        Multiply by U transpose
        For ultrasound physics, this is the adjoint measurement operator
        """
        # Ensure proper shape
        if len(vec.shape) == 1:
            vec = vec.view(1, -1)
        elif len(vec.shape) > 2:
            vec = vec.view(vec.shape[0], -1)
            
        # Reconstruct from measurement space
        reconstructed = self.V(vec)
        
        # Apply adjoint of measurement (approximate inverse)
        adjoint_result = self.H_pinv(reconstructed)
        
        return self.Vt(adjoint_result)
        
    def H(self, vec):
        """
        Physics-based degradation H*x + z with tissue/blind zone differentiation
        """
        if len(vec.shape) == 4:  # (B, C, H, W)
            # Initialize tissue detection if needed
            if self.tissue_mask is None:
                self.detect_and_protect_tissue(vec)
            
            # Apply differentiated distortion based on tissue/blind zone masks
            mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0)
            
            if hasattr(self, 'tissue_mask') and self.tissue_mask is not None:
                tissue_tensor = torch.from_numpy(self.tissue_mask).float().to(vec.device)
                blind_zone_tensor = torch.from_numpy(self.blind_zone_processing_mask).float().to(vec.device)
                
                tissue_expanded = tissue_tensor.unsqueeze(0).unsqueeze(0)
                blind_zone_expanded = blind_zone_tensor.unsqueeze(0).unsqueeze(0)
                
                # Tissue region: reduced distortion (protection)
                tissue_distortion_factor = self.distortion_factor * float(os.getenv('TISSUE_DISTORTION_FACTOR', '0.3'))
                tissue_distortion = vec * (1.0 + mask_expanded * tissue_distortion_factor) * tissue_expanded
                
                # Blind zone region: configurable distortion
                blind_zone_distortion_factor = self.distortion_factor * float(os.getenv('BLIND_ZONE_DISTORTION_FACTOR', '1.0'))
                blind_zone_distortion = vec * (1.0 + mask_expanded * blind_zone_distortion_factor) * blind_zone_expanded
                
                # Background region: minimal distortion
                background_mask = 1.0 - tissue_expanded - blind_zone_expanded
                background_mask = torch.clamp(background_mask, 0.0, 1.0)
                background_distortion_factor = self.distortion_factor * float(os.getenv('BACKGROUND_DISTORTION_FACTOR', '0.1'))
                background_distortion = vec * (1.0 + mask_expanded * background_distortion_factor) * background_mask
                
                # Combine all distortions
                distorted = tissue_distortion + blind_zone_distortion + background_distortion
                
            else:
                # Fallback: uniform distortion
                distorted = vec * (1.0 + mask_expanded * self.distortion_factor)
            
            # Add structural noise with differentiation
            noise_expanded = self.noise_pattern.unsqueeze(0).unsqueeze(0).expand_as(distorted)
            
            if hasattr(self, 'tissue_mask') and self.tissue_mask is not None:
                # Differentiated noise application
                tissue_noise = noise_expanded * mask_expanded * (self.noise_factor * float(os.getenv('TISSUE_NOISE_FACTOR', '0.2'))) * tissue_expanded
                blind_zone_noise = noise_expanded * mask_expanded * (self.noise_factor * float(os.getenv('BLIND_ZONE_NOISE_FACTOR', '1.0'))) * blind_zone_expanded
                background_noise = noise_expanded * mask_expanded * (self.noise_factor * float(os.getenv('BACKGROUND_NOISE_FACTOR', '0.1'))) * background_mask
                
                total_noise = tissue_noise + blind_zone_noise + background_noise
            else:
                # Uniform noise
                total_noise = noise_expanded * mask_expanded * self.noise_factor
            
            distorted = distorted + total_noise
            
            # Range clamp
            distorted = torch.clamp(distorted, 0.0, 1.0)
            
            return distorted
        else:  # Flattened
            vec_2d = vec.view(vec.shape[0], self.channels, self.img_size, self.img_size)
            result_2d = self.H(vec_2d)
            return result_2d.view(vec.shape[0], -1)
            
    def H_pinv(self, vec):
        """
        Physics-based pseudo-inverse H^+ with tissue/blind zone differentiation
        """
        if len(vec.shape) == 4:  # (B, C, H, W)
            # Remove structural noise
            noise_expanded = self.noise_pattern.unsqueeze(0).unsqueeze(0).expand_as(vec)
            mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0).expand_as(vec)
            denoised = vec - noise_expanded * mask_expanded * self.noise_factor
            
            # Apply differentiated inverse operation
            if hasattr(self, 'tissue_mask') and self.tissue_mask is not None:
                tissue_tensor = torch.from_numpy(self.tissue_mask).float().to(vec.device)
                blind_zone_tensor = torch.from_numpy(self.blind_zone_processing_mask).float().to(vec.device)
                
                tissue_expanded = tissue_tensor.unsqueeze(0).unsqueeze(0)
                blind_zone_expanded = blind_zone_tensor.unsqueeze(0).unsqueeze(0)
                
                # Tissue region: gentle inverse (protection)
                tissue_regularization = 0.95
                tissue_inverse = tissue_expanded * tissue_regularization
                
                # Blind zone region: strong inverse (restoration)
                blind_zone_regularized_mask = self.mask_tensor / (1.0 + self.mask_tensor * 0.5 + 1e-6)
                blind_zone_inverse = (1.0 - blind_zone_regularized_mask).unsqueeze(0).unsqueeze(0)
                blind_zone_correction = blind_zone_expanded * blind_zone_inverse
                
                # Background region: standard inverse
                background_mask = 1.0 - tissue_expanded - blind_zone_expanded
                background_mask = torch.clamp(background_mask, 0.0, 1.0)
                background_regularized_mask = self.mask_tensor / (1.0 + self.mask_tensor + 1e-6)
                background_inverse = (1.0 - background_regularized_mask).unsqueeze(0).unsqueeze(0)
                background_correction = background_mask * background_inverse
                
                # Combine all inverse operations
                adaptive_inverse = tissue_inverse + blind_zone_correction + background_correction
                result = denoised * adaptive_inverse
                
            else:
                # Fallback: basic inverse
                regularized_mask = self.mask_tensor / (1.0 + self.mask_tensor + 1e-6)
                inverse_expansion = (1.0 - regularized_mask).unsqueeze(0).unsqueeze(0)
                result = denoised * inverse_expansion
            
            # Stabilize result
            result = torch.clamp(result, -2.0, 2.0)
            
            return result
        else:  # Flattened
            vec_2d = vec.view(vec.shape[0], self.channels, self.img_size, self.img_size)
            result_2d = self.H_pinv(vec_2d)
            return result_2d.view(vec.shape[0], -1)
    
    def _get_donut_mask(self, shape, version):
        """버전별 도넛 마스크 생성"""
        if len(shape) == 4:
            height, width = shape[2], shape[3]
        else:
            height, width = self.img_size, self.img_size
            
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 버전별 도넛 매개변수
        version_params = {
            'V3': {'inner_r': 85, 'outer_r': 220},   # 대형 도넛
            'V4': {'inner_r': 50, 'outer_r': 130},   # 중대형 도넛
            'V5': {'inner_r': 30, 'outer_r': 90},    # 중형 도넛
            'V6': {'inner_r': 20, 'outer_r': 60},    # 소형 도넛
            'V7': {'inner_r': 15, 'outer_r': 45}     # 최소형 도넛
        }
        
        params = version_params.get(version, version_params['V5'])
        inner_r, outer_r = params['inner_r'], params['outer_r']
        
        # 도넛 영역 마스크 생성 (inner_r < distance <= outer_r)
        donut_mask = ((distance >= inner_r) & (distance <= outer_r)).astype(np.float32)
        
        return torch.from_numpy(donut_mask).float()


# Second UltrasoundBlindZoneWithNoise class removed to avoid conflicts
# Using the main physics-based implementation from line 13


def create_ultrasound_h_funcs(config, version=None, noise_pattern=None, distortion_factor=0.025, noise_factor=1.0,
                             enhanced_tissue_detection=True, tissue_detection_mode='multi', clahe_clip_limit=3.0,
                             min_tissue_size_factor=1.0, complete_blind_zone_removal=True, preserve_background=True,
                             version_thresholds=None, tissue_min_size=200, blind_zone_min_size=100):
    """Factory function to create appropriate H_functions for ultrasound"""
    
    channels = getattr(config.data, 'channels', 1)
    img_size = getattr(config.data, 'image_size', 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Creating UltrasoundBlindZone for version {version}")
    logger.info(f"  - Distortion factor: {distortion_factor}")
    logger.info(f"  - Noise factor: {noise_factor}")
    logger.info(f"  - Enhanced tissue detection: {enhanced_tissue_detection}")
    logger.info(f"  - Tissue detection mode: {tissue_detection_mode}")
    logger.info(f"  - CLAHE clip limit: {clahe_clip_limit}")
    logger.info(f"  - Complete blind zone removal: {complete_blind_zone_removal}")
    logger.info(f"  - Preserve background: {preserve_background}")
    logger.info(f"  - Noise pattern: {'provided' if noise_pattern is not None else 'default zeros'}")
    logger.info(f"*** CREATING UltrasoundBlindZone with enhanced features ***")
    
    return UltrasoundBlindZone(
        channels, img_size, device, version, noise_pattern, distortion_factor, noise_factor,
        enhanced_tissue_detection, tissue_detection_mode, clahe_clip_limit,
        min_tissue_size_factor, complete_blind_zone_removal, preserve_background,
        version_thresholds, tissue_min_size, blind_zone_min_size
    )


def estimate_version_artifacts(cn_on_path, cy_on_path, version, custom_threshold=None):
    """
    Enhanced structural noise estimation: z_est = Average(CY_ON - CN_ON)
    Implements version-specific (V3-V7) blind zone artifact estimation
    """
    logger.info(f"Estimating structural noise artifacts for version {version}")
    
    # Load version-specific images with better pattern matching
    cn_files = []
    cy_files = []
    
    # Try multiple pattern variations for robustness
    patterns = [
        f"*CN*{version}*.bmp", f"CN*{version}*.bmp", f"*{version}*CN*.bmp",
        f"*CN_ON*{version}*.bmp", f"CN_ON*{version}*.bmp"
    ]
    
    for pattern in patterns:
        cn_matches = sorted(list(Path(cn_on_path).glob(pattern)))
        if cn_matches:
            cn_files = cn_matches
            break
    
    for pattern in patterns:
        pattern_cy = pattern.replace('CN', 'CY')
        cy_matches = sorted(list(Path(cy_on_path).glob(pattern_cy)))
        if cy_matches:
            cy_files = cy_matches
            break
    
    if not cn_files or not cy_files:
        logger.warning(f"No {version} files found in {cn_on_path} or {cy_on_path}")
        return None, None
        
    logger.info(f"Found {len(cn_files)} CN_ON and {len(cy_files)} CY_ON files for {version}")
    
    # Enhanced noise estimation with physics-based processing
    noise_patterns = []
    distortion_maps = []
    
    # Version-specific processing parameters
    version_params = {
        "V3": {"outer_r": 220, "inner_r": 85, "strength": 0.8},
        "V4": {"outer_r": 130, "inner_r": 50, "strength": 0.9}, 
        "V5": {"outer_r": 90, "inner_r": 30, "strength": 1.0},
        "V6": {"outer_r": 60, "inner_r": 20, "strength": 1.1},
        "V7": {"outer_r": 45, "inner_r": 15, "strength": 1.2}
    }
    
    params = version_params.get(version, {"outer_r": 100, "inner_r": 40, "strength": 1.0})
    
    for cn_file, cy_file in zip(cn_files[:15], cy_files[:15]):  # Use more samples for robustness
        cn_img = np.array(Image.open(cn_file).convert('L').resize((512, 512))) / 255.0
        cy_img = np.array(Image.open(cy_file).convert('L').resize((512, 512))) / 255.0
        
        # Compute structural noise: z = CY_ON - CN_ON
        structural_noise = cy_img - cn_img
        
        # Create version-specific region mask for focused estimation
        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        region_mask = ((distance >= params["inner_r"]) & (distance <= params["outer_r"])).astype(np.float32)
        
        # Apply region mask to focus on blind zone
        focused_noise = structural_noise * region_mask
        noise_patterns.append(focused_noise)
        
        # Estimate distortion strength map
        distortion_strength = np.abs(structural_noise) * region_mask
        distortion_maps.append(distortion_strength)
    
    # Compute average structural noise pattern z_est
    if noise_patterns:
        z_est = np.mean(noise_patterns, axis=0)
        distortion_est = np.mean(distortion_maps, axis=0)
        
        # Apply custom threshold if provided (more permissive)
        if custom_threshold is not None and custom_threshold > 0.0:
            logger.info(f"Applying custom threshold {custom_threshold} for {version}")
            # Only consider pixels where noise intensity exceeds threshold
            noise_intensity = np.abs(z_est)
            threshold_mask = noise_intensity > custom_threshold
            z_est = z_est * threshold_mask.astype(np.float32)
            distortion_est = distortion_est * threshold_mask.astype(np.float32)
            logger.info(f"After threshold: {np.sum(threshold_mask)} pixels remain active")
        else:
            logger.info(f"No threshold applied for {version} - using full donut region")
        
        # Log version-specific statistics
        active_region = z_est[z_est != 0]
        if len(active_region) > 0:
            logger.info(f"{version} structural noise z_est stats:")
            logger.info(f"  - Mean: {np.mean(active_region):.4f}, Std: {np.std(active_region):.4f}")
            logger.info(f"  - Min: {np.min(active_region):.4f}, Max: {np.max(active_region):.4f}")
            logger.info(f"  - Coverage: {len(active_region) / (512*512) * 100:.2f}%")
        
        return z_est, distortion_est
    else:
        logger.warning(f"No valid noise patterns computed for {version}")
        return None, None


def estimate_degradation_operator(cn_oy_path, cy_oy_path, z_est, version):
    """
    Enhanced degradation operator estimation:
    H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||²
    """
    logger.info(f"Estimating degradation operator H_est for version {version}")
    
    if z_est is None:
        logger.error("Structural noise z_est not provided")
        return None
    
    # Load version-specific OY images
    cn_oy_files = []
    cy_oy_files = []
    
    patterns = [
        f"*CN*{version}*.bmp", f"CN*{version}*.bmp", f"*{version}*CN*.bmp",
        f"*CN_OY*{version}*.bmp", f"CN_OY*{version}*.bmp"
    ]
    
    for pattern in patterns:
        cn_matches = sorted(list(Path(cn_oy_path).glob(pattern)))
        if cn_matches:
            cn_oy_files = cn_matches
            break
    
    for pattern in patterns:
        pattern_cy = pattern.replace('CN', 'CY')
        cy_matches = sorted(list(Path(cy_oy_path).glob(pattern_cy)))
        if cy_matches:
            cy_oy_files = cy_matches
            break
    
    if not cn_oy_files or not cy_oy_files:
        logger.warning(f"No {version} OY files found")
        return None
    
    logger.info(f"Found {len(cn_oy_files)} CN_OY and {len(cy_oy_files)} CY_OY files for {version}")
    
    # Solve for H: minimize ||H·(CN_OY) - (CY_OY - z_est)||²
    h_estimates = []
    
    for cn_file, cy_file in zip(cn_oy_files[:10], cy_oy_files[:10]):
        cn_oy = np.array(Image.open(cn_file).convert('L').resize((512, 512))) / 255.0
        cy_oy = np.array(Image.open(cy_file).convert('L').resize((512, 512))) / 255.0
        
        # Remove structural noise from measurement
        cy_corrected = cy_oy - z_est
        
        # Solve H·cn_oy ≈ cy_corrected
        # For pixel-wise operation: H[i,j] = cy_corrected[i,j] / (cn_oy[i,j] + eps)
        eps = 1e-6
        h_estimate = np.divide(cy_corrected, cn_oy + eps, out=np.zeros_like(cy_corrected), where=(cn_oy + eps) != 0)
        
        # Regularize extreme values
        h_estimate = np.clip(h_estimate, 0.1, 3.0)
        h_estimates.append(h_estimate)
    
    if h_estimates:
        H_est = np.mean(h_estimates, axis=0)
        logger.info(f"{version} degradation operator H_est computed")
        logger.info(f"  - Mean: {np.mean(H_est):.4f}, Std: {np.std(H_est):.4f}")
        return H_est
    else:
        return None