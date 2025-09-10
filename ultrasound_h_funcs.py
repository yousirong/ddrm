import torch
import numpy as np
import os
from functions.svd_replacement import H_functions
from pathlib import Path
from PIL import Image
import logging

# Import cv2 and scipy for dataset building methods
try:
    import cv2
except ImportError:
    print("Warning: cv2 not available. Please install opencv-python: pip install opencv-python")
    cv2 = None

try:
    from scipy import ndimage
except ImportError:
    print("Warning: scipy not available. Please install scipy: pip install scipy")
    ndimage = None

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
                 version_thresholds=None, tissue_min_size=200, blind_zone_min_size=100, filename=None, runner=None):
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

        # V3~V7 도넛 기반 파라미터들 (build_*_dataset.py에서 추출)
        self.version_thresholds = version_thresholds or {}
        self.tissue_min_size = tissue_min_size
        self.blind_zone_min_size = blind_zone_min_size

        # Dataset building에서 사용된 마스크 생성 파라미터 추가
        self.radius_map = {
            'V3': (42, 82, 230),
            'V4': (25, 48, 133),
            'V5': (17, 32, 90),
            'V6': (11, 22, 63),
            'V7': (9, 17, 48),
        }

        # 임계값 및 필터링 파라미터
        self.thresh_mode = 'hist'  # 'hist' 또는 'fixed'
        self.fixed_thr_u8 = 50
        self.keep_side = 'high'  # 'high': 밝은쪽 검출, 'low': 어두운쪽 검출
        self.min_area_ratio = 0.0005
        self.hist_bins = 64
        self.otsu_delta_u8 = [-10, 10]
        self.otsu_scale = [0.70, 1.30]
        self.percentile_q = None

        logger.info(f"*** UltrasoundBlindZone.__init__: distortion_factor={self.distortion_factor}, noise_factor={self.noise_factor} ***")
        logger.info(f"*** Enhanced detection: {self.enhanced_tissue_detection}, mode: {self.tissue_detection_mode}, CLAHE: {self.clahe_clip_limit} ***")
        logger.info(f"*** Complete blind zone removal: {self.complete_blind_zone_removal}, Preserve background: {self.preserve_background} ***")
        self.version = version
        self.current_filename = filename
        logger.info(f"*** 파일명 설정됨: '{filename}' (type: {type(filename)}) ***")
        if filename:
            logger.info(f"*** 파일명 내용 확인: '{filename}' ***")
        else:
            logger.warning(f"*** 경고: 파일명이 None 또는 빈 값으로 전달됨! ***")

        # Runner 저장 (블라인드존 마스크 접근용)
        self.runner = runner
        self.detected_blind_zone_mask = None
        if runner and filename:
            self.detected_blind_zone_mask = runner.get_blind_zone_mask(filename)
            if self.detected_blind_zone_mask is not None:
                coverage = np.sum(self.detected_blind_zone_mask > 0) / self.detected_blind_zone_mask.size * 100
                logger.info(f"*** 검출된 블라인드존 마스크 로드: {coverage:.1f}% 커버리지 ***")

        # 조직 보호 관련 변수들 (H/H_pinv는 변경하지 않고 디노이징에서만 사용)
        self.tissue_mask = None
        self.blind_zone_processing_mask = None
        # self.current_filename은 이미 위에서 설정됨 - 다시 설정하지 않음

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
        Create physics-based distortion model for blind zone within visible region
        블라인드존은 중앙 고정, 보이는 영역만 각도별 이동
        """
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # 블라인드존은 항상 중앙 고정 (angle offset 적용 안함)
        blind_zone_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # 보이는 영역은 이미지 중심 사용
        visible_center_y = center_y
        visible_center_x = center_x
        visible_distance = np.sqrt((x - visible_center_x)**2 + (y - visible_center_y)**2)

        # Create visible region circle (shifted)
        version_visible_radius = {
            'V3': 175, 'V4': 150, 'V5': 110, 'V6': 105, 'V7': 95
        }
        visible_radius = version_visible_radius.get(self.version, 110)
        visible_region = (visible_distance <= visible_radius).astype(np.float32)

        # Create smooth distortion profile (not binary) - 블라인드존은 중앙 고정
        distortion_mask = np.zeros((height, width), dtype=np.float32)

        # Physics-based distortion model: gradual falloff (블라인드존 고정)
        in_blind_zone = (blind_zone_distance >= inner_radius) & (blind_zone_distance <= outer_radius)

        if np.any(in_blind_zone):
            # Normalized distance within blind zone [0, 1]
            zone_distance = (blind_zone_distance - inner_radius) / (outer_radius - inner_radius)
            zone_distance = np.clip(zone_distance, 0, 1)

            # Physics-based distortion profile (bell curve)
            # Maximum distortion in middle of blind zone
            distortion_profile = np.exp(-((zone_distance - 0.5) * 4) ** 2) * strength
            distortion_mask[in_blind_zone] = distortion_profile[in_blind_zone]

        # Constrain distortion to visible region (보이는 영역만 제한)
        constrained_distortion_mask = distortion_mask * visible_region

        return constrained_distortion_mask

    def _create_combined_distortion_mask(self, height, width):
        """Create combined physics-based distortion model for all versions within visible region
        블라인드존은 중앙 고정, 보이는 영역만 각도별 이동"""
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2

        # 보이는 영역은 이미지 중심 사용
        visible_center_y = center_y
        visible_center_x = center_x
        visible_distance = np.sqrt((x - visible_center_x)**2 + (y - visible_center_y)**2)

        # Create visible region circle with the shifted center
        version_visible_radius = {
            'V3': 175, 'V4': 150, 'V5': 110, 'V6': 105, 'V7': 95
        }
        visible_radius = version_visible_radius.get(self.version, 110)
        visible_region = (visible_distance <= visible_radius).astype(np.float32)

        combined_mask = np.zeros((height, width), dtype=np.float32)

        for version, params in self.VERSION_PARAMS.items():
            version_mask = self._create_physics_distortion_mask(
                height, width, params["outer_r"], params["inner_r"], params["strength"]
            )
            # Combine using maximum distortion
            combined_mask = np.maximum(combined_mask, version_mask)

        # Ensure the combined mask is constrained to visible region
        constrained_combined_mask = combined_mask * visible_region

        return constrained_combined_mask

    def get_combined_mask_with_otsu(self, image_np, version, visible_region):
        """
        Uses Otsu thresholding to create a clear separation between tissue and blind zone,
        and returns a final mask for processing.
        The protected tissue area is defined as the entire visible region minus the blind zone.
        """
        if cv2 is None:
            logger.warning("cv2 not available for Otsu-based mask generation.")
            return self._simple_threshold_detection(image_np), np.zeros_like(image_np)

        # 1. Define the region of interest (donut) and constrain it
        donut_region = self._create_version_donut_region(image_np.shape, version)
        constrained_donut_region = donut_region * visible_region

        # 2. Apply Otsu thresholding on the donut region to find the blind zone
        donut_pixels = image_np[constrained_donut_region > 0.5]
        if donut_pixels.size == 0:
            # No donut, so no blind zone from it. Tissue is the whole visible region.
            return visible_region, np.zeros_like(image_np)

        donut_pixels_u8 = (np.clip(donut_pixels, 0, 1) * 255).astype(np.uint8)
        otsu_thresh_val, _ = cv2.threshold(donut_pixels_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        otsu_adjustment_factor = 1.0
        adjusted_thresh_val = min(otsu_thresh_val * otsu_adjustment_factor, 255)
        logger.info(f"Otsu threshold: original={otsu_thresh_val}, adjusted={adjusted_thresh_val}")

        image_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)

        # Blind zone is dark areas within the constrained donut
        blind_zone_mask = ((image_u8 < adjusted_thresh_val) & (constrained_donut_region > 0.5)).astype(np.float32)

        # Post-process blind zone mask
        min_area = 0.0005
        blind_zone_mask = self.filter_small_components(blind_zone_mask, min_area_ratio=min_area)

        # Tissue is defined as the visible region minus the detected blind zone
        tissue_mask = visible_region - blind_zone_mask
        tissue_mask = np.clip(tissue_mask, 0, 1)

        logger.info(f"Otsu-based separation (visible region as tissue): tissue_coverage={np.sum(tissue_mask > 0.5) / tissue_mask.size * 100:.1f}%, "
                    f"blind_zone_coverage={np.sum(blind_zone_mask > 0.5) / blind_zone_mask.size * 100:.1f}%")

        return tissue_mask, blind_zone_mask

    def detect_and_protect_tissue(self, input_image):
        """
        가시 영역 원 내부에서 V3~V7 도넛 형태의 조직과 블라인드존 구분 검출:
        1. 각도별 중심점 이동을 적용한 가시 영역 원 생성
        2. 가시 영역 내에서 V3~V7 버전에 따른 도넛 형태 영역 결정
        3. 도넛 영역 내에서 Otsu 방법으로 밝은 조직과 어두운 블라인드존 구분
        4. 블라인드존만 제거하고 조직은 보호
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

        # === 1단계: 가시 영역 원 생성 ===
        height, width = image_np.shape
        center_y, center_x = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        version_visible_radius = {
            'V3': 175, 'V4': 150, 'V5': 110, 'V6': 105, 'V7': 95
        }
        visible_radius = version_visible_radius.get(self.version, 110)
        visible_region = (distance <= visible_radius).astype(np.float32)

        # === 2단계 & 3단계: Otsu 기반으로 조직과 블라인드존 구분 ===
        self.tissue_mask, self.blind_zone_processing_mask = self.get_combined_mask_with_otsu(image_np, self.version, visible_region)

        # === 4단계: 원본 이미지 저장 및 강한 조직 식별 ===
        self.original_image = image_np.copy()
        self.strong_tissue_mask = self._identify_strong_tissue(image_np, self.tissue_mask)

        # === 5단계: 블라인드존만 제거를 위한 마스크 ===
        self.blind_zone_processing_mask = blind_zone_mask

        # 로그 출력
        visible_coverage = np.sum(visible_region) / visible_region.size * 100
        tissue_coverage = np.sum(self.tissue_mask) / self.tissue_mask.size * 100
        blind_zone_coverage = np.sum(self.blind_zone_processing_mask) / self.blind_zone_processing_mask.size * 100

        logger.info(f"{self.version} Otsu 기반 조직/블라인드존 구분 완료:")
        logger.info(f"  - 가시 영역: {visible_coverage:.1f}% (중심점 고정)")
        logger.info(f"  - 조직 영역 (보호): {tissue_coverage:.1f}%")
        logger.info(f"  - 블라인드존 영역 (제거): {blind_zone_coverage:.1f}%")

        return self.tissue_mask, self.blind_zone_processing_mask







    def _get_angle_center_offset(self):
        """
        파일 이름에서 각도(예: D045)를 파싱하여 중심점 이동 오프셋 계산
        """
        if not self.current_filename or not isinstance(self.current_filename, str):
            return {'x': 0, 'y': 0}

        try:
            # 파일명에서 'D' 다음에 오는 세 자리 숫자(각도)를 찾습니다.
            import re
            match = re.search(r'_D(\d{3})_', self.current_filename)
            if not match:
                # D가 없는 이전 파일 형식 지원 (예: CY_ON_PL_V3_001.png)
                return {'x': 0, 'y': 0}

            angle_deg = float(match.group(1))
            angle_rad = np.deg2rad(angle_deg)

            # 오프셋 계산 (이동 반경은 예시 값, 필요시 조정)
            # 이 값은 가시 영역의 중심을 얼마나 이동시킬지를 결정합니다.
            offset_radius = 20  # 예: 20 픽셀
            offset_x = int(offset_radius * np.cos(angle_rad - np.pi / 2))
            offset_y = int(offset_radius * np.sin(angle_rad - np.pi / 2))

            return {'x': offset_x, 'y': offset_y}
        except Exception as e:
            logger.warning(f"Angle parsing failed for '{self.current_filename}': {e}")
            return {'x': 0, 'y': 0}

    def _create_version_donut_region(self, shape, version):
        """
        V3~V7 버전별 도넛 형태 영역 생성 (build_dataset.py 방식 사용)
        블라인드존/조직은 중앙 고정, 보이는 영역만 각도별 이동
        """
        height, width = shape

        # Dataset building 방식의 make_roi_masks_from_V 함수 사용
        donut_mask, r_in, r_out, block_mask = self.make_roi_masks_from_V(height, width, version)

        # 각도 정보는 로그용으로만 사용
        logger.info(f"{version} 도넛 영역 생성 (dataset building 방식): r_in={r_in}, r_out={r_out} (중앙 고정)")
        logger.info(f"  - 중심점 고정")
        logger.info(f"  - 도넛 영역 비율: {np.sum(donut_mask) / donut_mask.size * 100:.1f}%")
        logger.info(f"  - 차폐 영역 비율: {np.sum(block_mask) / block_mask.size * 100:.1f}%")

        return donut_mask


    def _create_visible_region_circle(self, shape, version):
        """
        각도별 중심점 이동을 적용한 가시 영역 원 생성
        블라인드존과 조직 탐지가 모두 이 원 안에서 이루어짐
        """
        height, width = shape
        center_y, center_x = height // 2, width // 2

        # 각도별 중심점 이동 적용
        angle_offset = self._get_angle_center_offset()
        center_y += angle_offset['y']
        center_x += angle_offset['x']

        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # 버전별 가시 영역 반지름 (도넛의 outer_radius보다 약간 크게 설정)
        version_visible_radius = {
            'V3': 175,  # V3 outer_r(220) + 여유분
            'V4': 150,  # V4 outer_r(130) + 여유분
            'V5': 110,  # V5 outer_r(90) + 여유분
            'V6': 105,   # V6 outer_r(60) + 여유분
            'V7': 85    # V7 outer_r(45) + 여유분
        }

        visible_radius = version_visible_radius.get(version, 110)
        visible_circle = (distance <= visible_radius).astype(np.float32)

        logger.info(f"{version} 가시 영역 원 생성:")
        logger.info(f"  - 반지름: {visible_radius}")
        logger.info(f"  - 중심점 이동: ({angle_offset['x']:+.0f}, {angle_offset['y']:+.0f})")
        logger.info(f"  - 가시 영역 비율: {np.sum(visible_circle) / visible_circle.size * 100:.1f}%")

        return visible_circle
    def set_current_filename(self, filename):
        """현재 처리 중인 파일명 설정"""
        self.current_filename = filename
        logger.info(f"파일명 설정: {filename}")

    def _separate_tissue_and_blind_zone_in_donut(self, image, donut_region, version):
        """
        도넛 영역 내에서 조직(밝은 영역)과 블라인드존(어두운 영역) 구분
        build_dataset.py 방식의 임계값 처리 사용 (가시 영역 제한 적용)
        """
        height, width = image.shape

        # `donut_region`은 이미 가시 영역이 적용된 상태.
        # `block_band`는 `make_roi_masks_from_V`에서 가져와야 함.
        _, _, _, block_band = self.make_roi_masks_from_V(height, width, version)

        # === 임계값 샘플 생성 & 처리 (process_image_with_dataset_method 로직 통합) ===
        tissue_masks = []
        blind_zone_masks = []

        # 임계값 생성 시 `donut_region`을 마스크로 사용
        for thr01, meta in self.gen_threshold_variants(image, donut_region):
            # (1) 검출 마스크 (ROI 내에서만)
            det_mask = self.build_binary_mask(image, thr01, keep=self.keep_side).astype(np.float32)
            det_mask = (det_mask * donut_region).astype(np.float32)
            det_mask = self.filter_small_components(det_mask, min_area_ratio=self.min_area_ratio)

            # (2) 임계값에 따라 조직/블라인드존 분류
            if self.keep_side == 'high':
                # 밝은 영역 검출 -> 조직
                tissue_mask = det_mask
                blind_zone_mask = (donut_region * (1.0 - det_mask) * (1.0 - block_band)).astype(np.float32)
            else:
                # 어두운 영역 검출 -> 블라인드존
                blind_zone_mask = det_mask
                tissue_mask = (donut_region * (1.0 - det_mask) * (1.0 - block_band)).astype(np.float32)

            tissue_masks.append(tissue_mask)
            blind_zone_masks.append(blind_zone_mask)

        # 여러 임계값 결과의 평균/통합
        if tissue_masks:
            final_tissue_mask = np.mean(tissue_masks, axis=0)
            final_blind_zone_mask = np.mean(blind_zone_masks, axis=0)

            # 이진화
            final_tissue_mask = (final_tissue_mask > 0.5).astype(np.float32)
            final_blind_zone_mask = (final_blind_zone_mask > 0.5).astype(np.float32)

            # 후처리
            final_tissue_mask = self.filter_small_components(final_tissue_mask, min_area_ratio=self.min_area_ratio)
            final_blind_zone_mask = self.filter_small_components(final_blind_zone_mask, min_area_ratio=self.min_area_ratio)

            tissue_mask, blind_zone_mask = final_tissue_mask, final_blind_zone_mask
        else:
            tissue_mask, blind_zone_mask = np.zeros_like(image), np.zeros_like(image)

        # 결과 로깅
        tissue_coverage = np.sum(tissue_mask) / tissue_mask.size * 100
        blind_zone_coverage = np.sum(blind_zone_mask) / blind_zone_mask.size * 100
        total_donut_coverage = np.sum(donut_region) / donut_region.size * 100

        angle_offset = self._get_angle_center_offset()

        logger.info(f"{version} 조직/블라인드존 구분 결과 (dataset building 방식, 가시영역 제한):")
        logger.info(f"  - 제한된 도넛 영역: {total_donut_coverage:.1f}% (shift: x={angle_offset['x']:+d}, y={angle_offset['y']:+d})")
        logger.info(f"  - 조직 영역 (보호): {tissue_coverage:.1f}%")
        logger.info(f"  - 블라인드존 영역 (제거): {blind_zone_coverage:.1f}%")

        return tissue_mask, blind_zone_mask

    def _identify_strong_tissue(self, image, tissue_mask):
        """
        조직 영역 내에서 70% 이상의 강한 조직 픽셀 식별
        """
        if tissue_mask is None or np.sum(tissue_mask) == 0:
            return np.zeros_like(image)

        # 조직 영역의 픽셀값들
        tissue_pixels = image[tissue_mask > 0.5]

        if len(tissue_pixels) == 0:
            return np.zeros_like(image)

        # 70th percentile 이상을 강한 조직으로 정의 (더 많은 밝은 영역 포함)
        strong_tissue_threshold = np.percentile(tissue_pixels, 70)

        # 강한 조직 마스크 생성
        strong_tissue_mask = np.zeros_like(image)
        strong_tissue_condition = (tissue_mask > 0.5) & (image >= strong_tissue_threshold)
        strong_tissue_mask[strong_tissue_condition] = 1.0

        strong_count = np.sum(strong_tissue_mask > 0.5)
        total_tissue_count = np.sum(tissue_mask > 0.5)

        logger.info(f"{self.version} 강한 조직 식별: {strong_count}/{total_tissue_count}개 픽셀 (임계값: {strong_tissue_threshold:.3f}, 가시 영역 고정)")

        return strong_tissue_mask

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
            logger.info(f"개선된 조직 검출 완료: {tissue_coverage:.1f}% 영역 (가시 영역 고정)")

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
        is_strong = std_val < 20 or mean_val < 30
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
            otsu_mask = (enhanced_img > otsu_thresh * 0.9).astype(np.float32)
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
            # Use detected blind zone mask if available
            if self.detected_blind_zone_mask is not None:
                logger.info("*** H(): 검출된 블라인드존 마스크 사용 ***")
                # Convert detected mask to torch tensor
                detected_mask_tensor = torch.from_numpy(self.detected_blind_zone_mask).float().to(vec.device)
                detected_mask_expanded = detected_mask_tensor.unsqueeze(0).unsqueeze(0)

                # Apply distortion only to detected blind zone regions
                distorted = vec.clone()
                # z_est 노이즈 추가 (검출된 영역에만)
                if self.noise_pattern is not None:
                    # Handle both numpy array and tensor types
                    if isinstance(self.noise_pattern, torch.Tensor):
                        noise_tensor = self.noise_pattern.float().to(vec.device)
                    else:
                        noise_tensor = torch.from_numpy(self.noise_pattern).float().to(vec.device)
                    noise_tensor = noise_tensor.unsqueeze(0).unsqueeze(0)
                    distorted = distorted + noise_tensor * detected_mask_expanded

                # H_est 왜곡 적용 (검출된 영역에만)
                mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0)
                distortion_strength = self.distortion_factor
                distorted = distorted * (1.0 + mask_expanded * distortion_strength * detected_mask_expanded)

                logger.info(f"*** H(): 검출된 블라인드존에만 왜곡 적용 완료 ***")
                return distorted

            # Initialize tissue detection if needed (fallback)
            if self.tissue_mask is None:
                self.detect_and_protect_tissue(vec)

            # Apply differentiated distortion based on tissue/blind zone masks
            mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0)

            if hasattr(self, 'tissue_mask') and self.tissue_mask is not None:
                tissue_tensor = torch.from_numpy(self.tissue_mask).float().to(vec.device)
                blind_zone_tensor = torch.from_numpy(self.blind_zone_processing_mask).float().to(vec.device)

                tissue_expanded = tissue_tensor.unsqueeze(0).unsqueeze(0)
                blind_zone_expanded = blind_zone_tensor.unsqueeze(0).unsqueeze(0)

                # Tissue region: NO distortion (완전 보호 - 원본 유지)
                tissue_distortion = vec * tissue_expanded

                # Blind zone region: FULL distortion (블라인드존만 왜곡 적용)
                blind_zone_distortion_factor = self.distortion_factor * float(os.getenv('BLIND_ZONE_DISTORTION_FACTOR', '1.0'))
                blind_zone_distortion = vec * (1.0 + mask_expanded * blind_zone_distortion_factor) * blind_zone_expanded

                # Background region: NO distortion (완전 보호 - 원본 유지)
                background_mask = 1.0 - tissue_expanded - blind_zone_expanded
                background_mask = torch.clamp(background_mask, 0.0, 1.0)
                background_distortion = vec * background_mask

                # Combine: 블라인드존만 왜곡, 나머지는 원본 유지
                distorted = tissue_distortion + blind_zone_distortion + background_distortion

            else:
                # Fallback: uniform distortion
                distorted = vec * (1.0 + mask_expanded * self.distortion_factor)

            # Add structural noise with differentiation
            noise_expanded = self.noise_pattern.unsqueeze(0).unsqueeze(0).expand_as(distorted)

            if hasattr(self, 'tissue_mask') and self.tissue_mask is not None:
                # Differentiated noise application - 블라인드존에만 노이즈 적용
                tissue_noise = torch.zeros_like(noise_expanded)  # 조직: 노이즈 없음 (원본 유지)
                blind_zone_noise = noise_expanded * mask_expanded * (self.noise_factor * float(os.getenv('BLIND_ZONE_NOISE_FACTOR', '1.0'))) * blind_zone_expanded
                background_noise = torch.zeros_like(noise_expanded)  # 배경: 노이즈 없음 (원본 유지)

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
        Physics-based pseudo-inverse H^+.
        Restores the blind zone and preserves other areas using a hard mask based on Otsu thresholding.
        """
        if len(vec.shape) != 4:  # Ensure (B, C, H, W)
            vec_2d = vec.view(vec.shape[0], self.channels, self.img_size, self.img_size)
            result_2d = self.H_pinv(vec_2d)
            return result_2d.view(vec.shape[0], -1)

        # Ensure masks are generated using the new Otsu-based method
        if self.tissue_mask is None or self.blind_zone_processing_mask is None:
            logger.info("H_pinv: Masks not found, generating them now.")
            self.detect_and_protect_tissue(vec)

        # --- 1. Invert the degradation process on the whole image ---
        # This will contain the restored version of the blind zone.

        # Remove noise (simplified, applied everywhere for inversion)
        noise_expanded = self.noise_pattern.unsqueeze(0).unsqueeze(0).expand_as(vec)
        mask_expanded = self.mask_tensor.unsqueeze(0).unsqueeze(0).expand_as(vec)
        denoised = vec - noise_expanded * mask_expanded * self.noise_factor

        # Apply inverse operation (simplified, applied everywhere for inversion)
        regularized_mask = self.mask_tensor / (1.0 + self.mask_tensor + 1e-6)
        inverse_expansion = (1.0 - regularized_mask).unsqueeze(0).unsqueeze(0)
        restored_result = denoised * inverse_expansion
        restored_result = torch.clamp(restored_result, 0.0, 1.0)  # Clamp to valid range

        # --- 2. Combine restored image with original using the hard mask ---
        if hasattr(self, 'original_image') and self.original_image is not None:
            original_tensor = torch.from_numpy(self.original_image).float().to(vec.device)
            original_expanded = original_tensor.unsqueeze(0).unsqueeze(0)

            blind_zone_tensor = torch.from_numpy(self.blind_zone_processing_mask).float().to(vec.device)
            blind_zone_expanded = blind_zone_tensor.unsqueeze(0).unsqueeze(0)

            # Use restored result only for the blind zone, otherwise use original
            final_result = torch.where(blind_zone_expanded > 0.5, restored_result, original_expanded)

            logger.info("H_pinv: Combined restored blind zone with original image using Otsu-based mask.")

            return torch.clamp(final_result, -2.0, 2.0)
        else:
            logger.warning("H_pinv: original_image not found. Returning fully restored image.")
            return torch.clamp(restored_result, -2.0, 2.0)

    def _get_donut_mask(self, shape, version):
        """가시 영역 원 내에서 버전별 도넛 마스크 생성
        블라인드존/도넛은 중앙 고정, 보이는 영역만 각도별 이동"""
        if len(shape) == 4:
            height, width = shape[2], shape[3]
        else:
            height, width = self.img_size, self.img_size

        center_y, center_x = height // 2, width // 2

        y, x = np.ogrid[:height, :width]

        # 블라인드존/도넛 영역은 중앙 고정
        blind_zone_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # 보이는 영역은 이미지 중심 사용
        visible_center_y = center_y
        visible_center_x = center_x
        visible_distance = np.sqrt((x - visible_center_x)**2 + (y - visible_center_y)**2)

        # 가시 영역 원 생성 (shifted center 사용)
        version_visible_radius = {
            'V3': 175, 'V4': 150, 'V5': 110, 'V6': 105, 'V7': 95
        }
        visible_radius = version_visible_radius.get(version, 110)
        visible_region = (visible_distance <= visible_radius).astype(np.float32)

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

        # 도넛 영역 마스크 생성 (중앙 고정 거리 사용)
        donut_mask = ((blind_zone_distance >= inner_r) & (blind_zone_distance <= outer_r)).astype(np.float32)

        # 가시 영역으로 제한
        constrained_donut_mask = donut_mask * visible_region

        return torch.from_numpy(constrained_donut_mask).float()

    def make_roi_masks_from_V(self, h, w, v_token: str):
        """
        build_dataset.py의 make_roi_masks_from_V 함수 구현
        radius_map이 3튜플일 때: (block_lo, r_in, r_out)
          - block_lo ~ r_in     구간은 항상 0(검정, 무효)
          - r_in ~ r_out        구간만 ROI(도넛)으로 사용
        반환: donut_mask, r_in, r_out, block_mask
        """
        if v_token not in self.radius_map:
            raise ValueError(f"radius_map에 '{v_token}' 항목이 없습니다.")
        vals = self.radius_map[v_token]
        if len(vals) == 2:
            r_in, r_out = vals
            r_block_lo = None
        elif len(vals) == 3:
            r_block_lo, r_in, r_out = vals
        else:
            raise ValueError("radius_map 값은 (r_in,r_out) 또는 (block_lo,r_in,r_out) 이어야 합니다.")

        max_r = min(h, w) / 2.0 - 1.0
        r_in   = float(np.clip(r_in, 0.0, max_r))
        r_out  = float(np.clip(r_out, r_in + 1e-6, max_r))
        if len(vals) == 3:
            r_block_lo = float(np.clip(r_block_lo, 0.0, r_in))
        cx = (w - 1) / 2.0; cy = (h - 1) / 2.0
        yy, xx = np.indices((h, w), dtype=np.float32)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        donut_mask = ((rr >= r_in) & (rr <= r_out)).astype(np.float32)
        if len(vals) == 3:
            # 차폐 구간: [r_block_lo, r_in)  → 항상 0으로 만들기 위함
            block_mask = ((rr >= r_block_lo) & (rr < r_in)).astype(np.float32)
        else:
            block_mask = np.zeros((h, w), dtype=np.float32)

        return donut_mask, r_in, r_out, block_mask

    def compute_threshold(self, img01: np.ndarray, mode='hist', fixed_thr_u8=50, mask: np.ndarray=None,
                         otsu_delta_u8: float = 0.0, otsu_scale: float = 1.0, percentile_q: float = None):
        """
        build_dataset.py의 compute_threshold 함수 구현
        """
        if mode == 'fixed':
            return float(np.clip(fixed_thr_u8, 0, 255)) / 255.0
        vec = img01[mask > 0.5].flatten() if mask is not None else img01.flatten()
        vec = np.clip(vec, 0, 1)
        if vec.size == 0: return 0.5
        if percentile_q is not None:
            q = float(np.clip(percentile_q, 0.0, 1.0))
            thr01 = float(np.percentile(vec, q * 100.0))
            return float(np.clip(thr01, 0.0, 1.0))

        if cv2 is None:
            # Fallback to simple thresholding if cv2 is not available
            logger.warning("cv2 not available, using mean-based threshold")
            thr01 = float(np.mean(vec))
            return float(np.clip(thr01, 0.0, 1.0))

        u8 = (vec * 255).astype(np.uint8).reshape(-1, 1)
        thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_u8 = int(np.clip(thr_u8 * float(otsu_scale) + float(otsu_delta_u8), 0, 255))
        return float(thr_u8) / 255.0

    def build_binary_mask(self, src01: np.ndarray, thr01: float, keep='high'):
        """
        build_dataset.py의 build_binary_mask 함수 구현
        """
        return (src01 >= thr01).astype(np.float32) if keep == 'high' else (src01 <= thr01).astype(np.float32)

    def filter_small_components(self, bin_mask01: np.ndarray, min_area_ratio=0.0005):
        """
        build_dataset.py의 filter_small_components 함수 구현
        """
        H, W = bin_mask01.shape[:2]
        min_area = max(1, int(H * W * max(0.0, min_area_ratio)))

        if cv2 is None:
            # Fallback: return original mask if cv2 is not available
            logger.warning("cv2 not available, skipping small component filtering")
            return bin_mask01

        u8 = (bin_mask01 > 0.5).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        if num <= 1: return bin_mask01
        out = np.zeros_like(u8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 1
        return out.astype(np.float32)

    def gen_threshold_variants(self, img, donut):
        """
        build_dataset.py의 gen_threshold_variants 함수 구현
        임계값 후보 생성
        """
        import random
        np.random.seed(42)

        def _is_range(x): return isinstance(x, (list, tuple)) and len(x) == 2
        def _sample_from(x):
            if _is_range(x):
                lo, hi = float(x[0]), float(x[1])
                return float(np.random.uniform(lo, hi))
            return float(x)

        if self.percentile_q is not None:
            any_range = _is_range(self.percentile_q)
            ns = 2 if any_range else 1
            for _ in range(ns):
                q = float(_sample_from(self.percentile_q))
                thr01 = self.compute_threshold(
                    img, mode='hist', fixed_thr_u8=self.fixed_thr_u8, mask=donut,
                    percentile_q=q
                )
                meta = {'mode': 'perc', 'q': q}
                yield thr01, meta
        else:
            # --- Otsu 모드 ---
            any_range = _is_range(self.otsu_scale) or _is_range(self.otsu_delta_u8)
            target_n = 3 if any_range else 1

            # 1) baseline 먼저 (scale=1.0, delta=0)
            baseline_s = 1.0
            baseline_d = 0.0
            thr01 = self.compute_threshold(
                img, mode='hist', fixed_thr_u8=self.fixed_thr_u8, mask=donut,
                otsu_scale=baseline_s, otsu_delta_u8=baseline_d, percentile_q=None
            )
            yield thr01, {'mode': 'otsu', 'scale': baseline_s, 'delta': baseline_d}

            if target_n == 1:
                return

            # 2) 나머지 랜덤
            combos = set()
            combos.add((round(baseline_s, 3), int(round(baseline_d))))

            while len(combos) < target_n:
                s = float(_sample_from(self.otsu_scale))
                d = float(_sample_from(self.otsu_delta_u8))
                key = (round(s, 3), int(round(d)))
                if key in combos:
                    continue
                combos.add(key)
                thr01 = self.compute_threshold(
                    img, mode='hist', fixed_thr_u8=self.fixed_thr_u8, mask=donut,
                    otsu_scale=key[0], otsu_delta_u8=key[1], percentile_q=None
                )
                yield thr01, {'mode': 'otsu', 'scale': key[0], 'delta': key[1]}

    def process_image_with_dataset_method(self, image, version):
        """
        build_dataset.py 방식을 사용한 이미지 처리
        조직과 블라인드존 구분을 위한 임계값 기반 처리
        """
        H, W = image.shape

        # === 도넛 ROI + 차폐 밴드 ===
        donut, r_in, r_out, block_band = self.make_roi_masks_from_V(H, W, version)

        # === 임계값 샘플 생성 & 처리 ===
        tissue_masks = []
        blind_zone_masks = []

        for thr01, meta in self.gen_threshold_variants(image, donut):
            # (1) 검출 마스크 (ROI 내에서만)
            det_mask = self.build_binary_mask(image, thr01, keep=self.keep_side).astype(np.float32)
            det_mask = (det_mask * donut).astype(np.float32)
            det_mask = self.filter_small_components(det_mask, min_area_ratio=self.min_area_ratio)

            # (2) 최종 keep 마스크 = (1 - donut) + det, 단 block_band는 무조건 0으로 차폐
            keep_mask = ((1.0 - donut) + det_mask).astype(np.float32)
            keep_mask = (keep_mask * (1.0 - block_band)).astype(np.float32)

            # 임계값에 따라 조직/블라인드존 분류
            if self.keep_side == 'high':
                # 밝은 영역을 검출하는 경우 - 검출된 영역은 조직
                tissue_mask = det_mask
                blind_zone_mask = (donut * (1.0 - det_mask) * (1.0 - block_band)).astype(np.float32)
            else:
                # 어두운 영역을 검출하는 경우 - 검출된 영역은 블라인드존
                blind_zone_mask = det_mask
                tissue_mask = (donut * (1.0 - det_mask) * (1.0 - block_band)).astype(np.float32)

            tissue_masks.append(tissue_mask)
            blind_zone_masks.append(blind_zone_mask)

        # 여러 임계값 결과의 평균
        if tissue_masks:
            final_tissue_mask = np.mean(tissue_masks, axis=0)
            final_blind_zone_mask = np.mean(blind_zone_masks, axis=0)

            # 이진화
            final_tissue_mask = (final_tissue_mask > 0.5).astype(np.float32)
            final_blind_zone_mask = (final_blind_zone_mask > 0.5).astype(np.float32)

            # 후처리
            final_tissue_mask = self.filter_small_components(final_tissue_mask, min_area_ratio=self.min_area_ratio)
            final_blind_zone_mask = self.filter_small_components(final_blind_zone_mask, min_area_ratio=self.min_area_ratio)

            return final_tissue_mask, final_blind_zone_mask
        else:
            return np.zeros_like(image), np.zeros_like(image)


# Second UltrasoundBlindZoneWithNoise class removed to avoid conflicts
# Using the main physics-based implementation from line 13


def create_ultrasound_h_funcs(config, version=None, noise_pattern=None, distortion_factor=0.025, noise_factor=1.0,
                             enhanced_tissue_detection=True, tissue_detection_mode='multi', clahe_clip_limit=3.0,
                             min_tissue_size_factor=1.0, complete_blind_zone_removal=True, preserve_background=True,
                             version_thresholds=None, tissue_min_size=200, blind_zone_min_size=100, filename=None, runner=None):
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
        version_thresholds, tissue_min_size, blind_zone_min_size, filename, runner
    )


def estimate_version_artifacts(cn_on_path, cy_on_path, version, custom_threshold=None, current_filename=None):
    """
    Enhanced structural noise estimation: z_est = Average(CY_ON - CN_ON)
    Implements version-specific (V3-V7) blind zone artifact estimation, excluding tissue regions.
    """
    logger.info(f"Estimating structural noise artifacts for version {version}")
    if current_filename:
        logger.info(f"Processing filename: {current_filename}")

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
        "V3": {"outer_r": 220, "inner_r": 85, "strength": 0.8, "tissue_percentile": 75},
        "V4": {"outer_r": 130, "inner_r": 50, "strength": 0.9, "tissue_percentile": 75},
        "V5": {"outer_r": 90, "inner_r": 30, "strength": 1.0, "tissue_percentile": 75},
        "V6": {"outer_r": 60, "inner_r": 20, "strength": 1.1, "tissue_percentile": 75},
        "V7": {"outer_r": 45, "inner_r": 15, "strength": 1.2, "tissue_percentile": 75}
    }

    params = version_params.get(version, {"outer_r": 100, "inner_r": 40, "strength": 1.0, "tissue_percentile": 75})

    for cn_file, cy_file in zip(cn_files[:15], cy_files[:15]):  # Use more samples for robustness
        cn_img = np.array(Image.open(cn_file).convert('L').resize((512, 512))) / 255.0
        cy_img = np.array(Image.open(cy_file).convert('L').resize((512, 512))) / 255.0

        # 엡실론(나눗셈 오류 방지) 및 안정화 계수
        epsilon = 1e-6
        stabilization_factor = 0.01  # 안정화 계수
        # CY_ON = M * CN_ON + z_est 모델 기반 z_est 추정
        # 1. 곱셈 왜곡 M 추정: M = CY_ON / CN_ON
        multiplicative_map = np.divide(cy_img, cn_img + epsilon)
        multiplicative_map = np.clip(multiplicative_map, 0.1, 5.0)  # 극단값 제한

        # 2. z_est 추정: z_est = CY_ON - M * CN_ON
        structural_noise = (cy_img - multiplicative_map * cn_img)
        # Create version-specific region mask for focused estimation within visible region

        # Create temporary UltrasoundBlindZone instance for visible region calculation
        temp_h_funcs = UltrasoundBlindZone(1, 512, torch.device('cpu'), version, None, 0.025, 1.0, filename=current_filename)
        visible_region = temp_h_funcs._create_visible_region_circle((512, 512), version)

        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256

        # 중심점 고정 (중심점 이동 제거)

        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        donut_region = ((distance >= params["inner_r"]) & (distance <= params["outer_r"])).astype(np.float32)

        # Constrain region mask to visible region
        region_mask = donut_region * visible_region

        # Detect and exclude tissue from z_est
        donut_pixels = cy_img[region_mask > 0.5]
        if len(donut_pixels) > 0:
            tissue_threshold = np.percentile(donut_pixels, params['tissue_percentile'])
            tissue_mask = ((cy_img >= tissue_threshold) & (region_mask > 0.5)).astype(np.float32)
        else:
            tissue_mask = np.zeros_like(cy_img)

        # Apply region mask and tissue exclusion to focus on blind zone noise
        focused_noise = structural_noise * region_mask * (1.0 - tissue_mask)
        noise_patterns.append(focused_noise)

        # Estimate distortion strength map
        distortion_strength = np.abs(structural_noise) * region_mask * (1.0 - tissue_mask)
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
            logger.info(f"{version} structural noise z_est stats (tissue excluded):")
            logger.info(f"  - Mean: {np.mean(active_region):.4f}, Std: {np.std(active_region):.4f}")
            logger.info(f"  - Min: {np.min(active_region):.4f}, Max: {np.max(active_region):.4f}")
            logger.info(f"  - Coverage: {len(active_region) / (512*512) * 100:.2f}%")

        return z_est, distortion_est
    else:
        logger.warning(f"No valid noise patterns computed for {version}")
        return None, None


def estimate_degradation_operator(cn_oy_path, cy_oy_path, z_est, version, current_filename=None):
    """
    Enhanced degradation operator estimation:
    H_est = argmin_H ||H·(CN_OY) - (CY_OY - z_est)||², excluding tissue regions.
    """
    logger.info(f"Estimating degradation operator H_est for version {version}")
    if current_filename:
        logger.info(f"Processing filename: {current_filename}")

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

    version_params = {
        "V3": {"outer_r": 220, "inner_r": 85, "tissue_percentile": 75},
        "V4": {"outer_r": 130, "inner_r": 50, "tissue_percentile": 75},
        "V5": {"outer_r": 90, "inner_r": 30, "tissue_percentile": 75},
        "V6": {"outer_r": 60, "inner_r": 20, "tissue_percentile": 75},
        "V7": {"outer_r": 45, "inner_r": 15, "tissue_percentile": 75}
    }
    params = version_params.get(version, {"outer_r": 100, "inner_r": 40, "tissue_percentile": 75})

    for cn_file, cy_file in zip(cn_oy_files[:10], cy_oy_files[:10]):
        cn_oy = np.array(Image.open(cn_file).convert('L').resize((512, 512))) / 255.0
        cy_oy = np.array(Image.open(cy_file).convert('L').resize((512, 512))) / 255.0

        # Remove structural noise from measurement
        cy_corrected = cy_oy - z_est

        # Solve H·cn_oy ≈ cy_corrected
        eps = 1e-6
        h_estimate = np.divide(cy_corrected, cn_oy + eps, out=np.zeros_like(cy_corrected), where=(cn_oy + eps) != 0)

        # Detect and exclude tissue from H_est within visible region

        # Create temporary UltrasoundBlindZone instance for visible region calculation
        temp_h_funcs = UltrasoundBlindZone(1, 512, torch.device('cpu'), version, None, 0.025, 1.0, filename=current_filename)
        visible_region = temp_h_funcs._create_visible_region_circle((512, 512), version)

        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256

        # 중심점 고정 (중심점 이동 제거)

        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        donut_region = ((distance >= params["inner_r"]) & (distance <= params["outer_r"])).astype(np.float32)

        # Constrain region mask to visible region
        region_mask = donut_region * visible_region

        donut_pixels = cy_oy[region_mask > 0.5]
        if len(donut_pixels) > 0:
            tissue_threshold = np.percentile(donut_pixels, params['tissue_percentile'])
            tissue_mask = ((cy_oy >= tissue_threshold) & (region_mask > 0.5)).astype(np.float32)
            h_estimate[tissue_mask > 0.5] = 1.0 # Set H to identity for tissue

        # Regularize extreme values
        h_estimate = np.clip(h_estimate, 0.1, 3.0)
        h_estimates.append(h_estimate)

    if h_estimates:
        H_est = np.mean(h_estimates, axis=0)
        logger.info(f"{version} degradation operator H_est computed (tissue excluded)")
        logger.info(f"  - Mean: {np.mean(H_est):.4f}, Std: {np.std(H_est):.4f}")
        return H_est
    else:
        return None