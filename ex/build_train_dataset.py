import os
import cv2
import numpy as np
from pathlib import Path
import hashlib  # 고유 ID 생성용

# ========= 입력 데이터 =========
root_dirs = [
    "./dataset/datasets_v0.04/P0",
    "./dataset/datasets_v0.04/P2",
]

# 예시: 필터 조건
# conditions = {
#     'C': 'CN',
#     'O': 'OY',
#     'P': 'PC',
#     'D': 'DC~359',
#     'V': 'V3~7',
#     'N': '000~999',
# }

# 예시: 필터 조건
# conditions = {
#     'C': 'CN',
#     'O': 'ON',
#     'P': 'PL',
#     'D': 'DC~359',
#     'V': 'V3~7',
#     'N': '000~999',
# }

# 예시: 필터 조건
conditions = {
    'C': 'CN',
    'O': 'ON',
    'P': 'PC',
    'D': 'DC~359',
    'V': 'V3~7',
    'N': '000~999',
}

# ========= 마스크 생성 옵션 =========
OUT_ROOT = Path("./masks_out")
OUTPUT_EXT = ".jpg"
THRESH_MODE  = 'hist'   # 'hist' (Otsu/Percentile) | 'fixed'
FIXED_THR_U8 = 50
KEEP_SIDE    = 'high'   # 'high': 밝은쪽 검출, 'low': 어두운쪽 검출
MIN_AREA_RATIO = 0.0005
HIST_BINS = 64                   # 히스토그램 구간 개수
SAVE_HIST    = True
USE_DONUT_ROI = True
USE_GRAYSCALE = True
SAVE_DET_MASK_DEBUG = True # 검출 마스크도 디버그로 저장

# --- Otsu/Percentile 미세조정 (스칼라 또는 [min, max]) ---
OTSU_DELTA_U8 = [-10, 10]          # 예: 0 또는 [-10, 10]
OTSU_SCALE    = [0.70, 1.30]       # 예: 1.0 또는 [0.95, 1.05]
PERCENTILE_Q  = None               # 예: None 또는 0.85 또는 [0.85, 0.90]
THR_SAMPLES   = 10                 # 임계값 샘플 수(예: 10 또는 1)

# radius_map이 3튜플일 때: (block_lo, r_in, r_out)
#   - block_lo ~ r_in     구간은 항상 0(검정, 무효)
#   - r_in ~ r_out        구간만 ROI(도넛)으로 사용
radius_map = {
    'V3': (42, 82, 230),
    'V4': (25, 48, 133),
    'V5': (17, 32, 90),
    'V6': (11, 22, 63),
    'V7': (9, 17, 48),
}

# ========= 증강 옵션 (연속값 샘플링) =========
AUG_ENABLE = True
AUG_ROOT = Path("./aug_out")
AUG_OUTPUT_EXT = ".jpg"
RANDOM_SEED = 42

# 공간 변환(이미지/마스크 동일 적용)
FLIPS = ["none", "h", "v", "hv"]  # 이건 이산(좌우/상하/둘다)
ROT_ANGLE_RANGE = (-359.0, 359.0) # deg, 연속
SHIFT_RANGE_X = (-2.0, 2.0)       # pixels, 연속
SHIFT_RANGE_Y = (-2.0, 2.0)       # pixels, 연속
SAMPLES_PER_FLIP = 3              # 각 flip 조합 당 랜덤 샘플 개수

BORDER_VALUE_IMG = 0   # 회전/쉬프트 바깥 영역: 이미지(검정)
BORDER_VALUE_MASK = 0  # 회전/쉬프트 바깥 영역: 마스크(배경=0)

# 포토메트릭(이미지에만)
CONTRAST_ALPHA_RANGE = (0.90, 1.10)  # 연속 곱
BRIGHTNESS_BETA_RANGE_U8 = (-25, 25) # 연속 더하기(u8)
SAT_SCALE_RANGE = (0.90, 1.10)       # 연속 배율(3채널에서만)

# ========= ENDOLFIN 내보내기 옵션 =========
EXPORT_ENDOLFIN = True
ENDOLFIN_TRAIN_ROOT = Path("./dataset/ENDOLFIN_new/Train")  
FRAME_INDEX_SPEC = "0:5:70"   # "시작:간격:끝(포함)" 혹은 (start, step, end)
EXPORT_NAME_MODE = "full"     # "full": 파일 스템 전체 사용 / "base": "__aug__" 앞부분만 사용
ENDOLFIN_JPEG_QUALITY = 100    # JPEG 저장 품질(마스크도 .jpg 요구사항이면 동일 품질 사용)

# ========= 저장 채널 옵션(핵심) =========
# 'gray' → 1채널 그레이스케일로 저장
# 'rgb'  → 3채널 컬러로 저장(OpenCV는 BGR로 기록되지만 3채널 보장)
SAVE_COLOR_MODE = "rgb"   # 'gray' 또는 'rgb'

# ========= 유틸 =========
def parse_filename(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() != '.bmp':
        return None
    parts = name.split('_')
    if len(parts) != 6:
        return None
    C, O, P, D, V, N = parts
    return {'C': C, 'O': O, 'P': P, 'D': D, 'V': V, 'N': N, 'name': name, 'ext': ext}

def _expand_P_token(tok: str):
    """
    예시:
      'PC'       -> {'PC'}
      'PC,PL'    -> {'PC','PL'}
      'PC,L'     -> {'PC','PL'}     # 축약 표기
      'P C , L'  -> {'PC','PL'}     # 공백 허용
      'P,C,L'    -> {'PC','PL'}     # 'P' 단독 + 나머지 상속
    """
    if tok is None:
        return set()
    t = tok.strip().upper()
    for sep in [' ', ';', '|', '/']:
        t = t.replace(sep, ',')
    parts = [p for p in t.split(',') if p]

    if not parts:
        return set()

    out = set()
    inherit_prefix = 'P' if parts[0].startswith('P') else ''

    if len(parts) >= 1 and parts[0] == 'P':
        parts = parts[1:]
        inherit_prefix = 'P'

    for p in parts:
        if p.startswith('P'):
            out.add(p)
        else:
            out.add((inherit_prefix + p) if inherit_prefix else p)
    return out

def _expand_V_token(tok):
    t = tok.upper()
    if '~' in t:
        left, right = t.split('~')
        if not left.startswith('V'): return set()
        v0 = int(left[1:]); v1 = int(right)
        return {f"V{v}" for v in range(v0, v1 + 1)}
    return {t}

def _expand_N_token(tok):
    t = tok
    if '~' in t:
        left, right = t.split('~')
        width = len(left)
        n0 = int(left); n1 = int(right)
        return {str(n).zfill(width) for n in range(n0, n1 + 1)}
    return {t}

def _expand_D_token(tok):
    t = tok.upper().replace(' ', '')
    if t == 'DC': return {'DC'}
    if '~' in t:
        left, right = t.split('~')
        def parse_side(x):
            if x == 'DC': return ('DC', 0, 3)
            if x.startswith('D'):
                digits = x[1:]
                if digits == '': return ('D', 0, 3)
                if digits.isdigit(): return ('D', int(digits), len(digits))
                return None
            if x.isdigit(): return ('', int(x), len(x))
            return None
        L = parse_side(left); R = parse_side(right)
        if L is None or R is None: return set()
        width = max(3, L[2] if L[0] != 'DC' else 3, R[2])
        vals = set()
        include_DC = (L[0] == 'DC')
        n0 = L[1] if L[0] in ('D', '') else 0
        if L[0] == 'DC': n0 = 0
        n1 = R[1]
        if n0 > n1: n0, n1 = n1, n0
        for i in range(n0, n1 + 1):
            vals.add(f"D{str(i).zfill(width)}")
        if include_DC: vals.add('DC')
        return vals
    if t.startswith('D') and (len(t) == 1 or t[1:].isdigit()):
        return {t}
    return set()

def _expand_simple_list(tok):
    return set(p.strip().upper() for p in tok.split(',') if p.strip())

def parse_conditions(conds: dict):
    out = {'C': None, 'O': None, 'P': None, 'D': None, 'V': None, 'N': None}
    for k, v in conds.items():
        if k not in out or not v: continue
        if k == 'V': out[k] = _expand_V_token(v)
        elif k == 'N': out[k] = _expand_N_token(v)
        elif k == 'D': out[k] = _expand_D_token(v)
        elif k == 'P': out[k] = _expand_P_token(v)
        else: out[k] = _expand_simple_list(v)
    return out

def record_matches(rec, cons_sets):
    for k, allowed in cons_sets.items():
        if allowed is None: continue
        if rec[k] not in allowed: return False
    return True

def load_image(path, grayscale=True):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32) / 255.0

def compute_threshold(img01: np.ndarray,
                      mode='hist',
                      fixed_thr_u8=50,
                      mask: np.ndarray=None,
                      otsu_delta_u8: int = 0,
                      otsu_scale: float = 1.0,
                      percentile_q: float = None):
    if mode == 'fixed':
        return float(np.clip(fixed_thr_u8, 0, 255)) / 255.0
    vec = img01[mask > 0.5].flatten() if mask is not None else img01.flatten()
    vec = np.clip(vec, 0, 1)
    if vec.size == 0: return 0.5
    if percentile_q is not None:
        q = float(np.clip(percentile_q, 0.0, 1.0))
        thr01 = float(np.percentile(vec, q * 100.0))
        return float(np.clip(thr01, 0.0, 1.0))
    u8 = (vec * 255).astype(np.uint8).reshape(-1, 1)
    thr_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_u8 = int(np.clip(thr_u8 * otsu_scale + otsu_delta_u8, 0, 255))
    return float(thr_u8) / 255.0

def build_binary_mask(src01: np.ndarray, thr01: float, keep='high'):
    return (src01 >= thr01).astype(np.float32) if keep == 'high' else (src01 <= thr01).astype(np.float32)

def filter_small_components(bin_mask01: np.ndarray, min_area_ratio=0.0005):
    H, W = bin_mask01.shape[:2]
    min_area = max(1, int(H * W * max(0.0, min_area_ratio)))
    u8 = (bin_mask01 > 0.5).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if num <= 1: return bin_mask01
    out = np.zeros_like(u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out.astype(np.float32)

def make_histogram(img01: np.ndarray, bins=HIST_BINS, mask: np.ndarray=None):
    vec = img01[mask > 0.5].flatten() if mask is not None else img01.flatten()
    counts, bin_edges = np.histogram(vec, bins=bins, range=(0.0, 1.0))
    return counts.astype(np.int64), bin_edges

def to_uint8(img01):
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)

# ========= 공용 저장 함수(핵심) =========
def imwrite(path, img, color_mode=None, jpeg_quality=None):
    """
    path: str 또는 Path
    img: 0~1 float 또는 uint8, 1/3/4채널 허용
    color_mode: 'gray' 또는 'rgb' (None이면 SAVE_COLOR_MODE 사용)
    """
    if color_mode is None:
        color_mode = SAVE_COLOR_MODE

    arr = img
    # float(0~1) → uint8
    if not isinstance(arr, np.ndarray):
        raise ValueError("imwrite: img must be a numpy array.")
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)

    # 채널 정규화
    if color_mode == 'rgb':
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        # 3채널이면 그대로(BGR로 저장)
    else:  # 'gray'
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        # 1채널이면 그대로

    # JPEG 품질 옵션
    path_str = str(path)
    if (jpeg_quality is not None) and path_str.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(path_str, arr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        cv2.imwrite(path_str, arr)

def save_histogram_image(counts: np.ndarray, bin_edges: np.ndarray, out_path: Path, thr01: float = None, size=(640, 320)):
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 30
    plot_w = W - pad_l - pad_r; plot_h = H - pad_t - pad_b
    cv2.rectangle(img, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (0, 0, 0), 1)
    if counts.max() > 0:
        cmax = counts.max()
        for i, c in enumerate(counts):
            x0 = pad_l + int(i * (plot_w / len(counts)))
            x1 = pad_l + int((i + 1) * (plot_w / len(counts)))
            h = int(c / cmax * (plot_h - 1))
            cv2.rectangle(img, (x0, pad_t + plot_h - h), (x1 - 1, pad_t + plot_h - 1), (50, 50, 50), -1)
    if thr01 is not None:
        x_thr = pad_l + int(thr01 * plot_w)
        cv2.line(img, (x_thr, pad_t), (x_thr, pad_t + plot_h), (0, 0, 255), 2)
        cv2.putText(img, f"thr={thr01:.3f}", (min(x_thr + 5, W - 120), pad_t + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    # 공용 저장 함수 사용(옵션 일괄 적용)
    imwrite(out_path, img)

# ========= 임계값 샘플링 유틸 =========
rng = np.random.default_rng(RANDOM_SEED)

def _is_range(x):
    return isinstance(x, (list, tuple)) and len(x) == 2

def _sample_from(x):
    if _is_range(x):
        lo, hi = float(x[0]), float(x[1])
        return float(rng.uniform(lo, hi))
    return x

def thr_code_token(meta):
    if meta['mode'] == 'perc':
        q = meta['q']
        return f"thrP_q{q:.3f}".replace('.', '_')
    else:
        s = meta['scale']; d = meta['delta']
        s_tok = f"s{s:.3f}".replace('.', '_')
        d_tok = f"d{d:+.0f}".replace('+','p').replace('-','m')
        return f"thrO_{s_tok}_{d_tok}"

def gen_threshold_variants(img, donut):
    """
    임계값 후보 생성:
      - Percentile 모드: 범위면 THR_SAMPLES개 랜덤, 스칼라면 1개.
      - Otsu 모드: 항상 baseline (scale=1.00, delta=0) 1개 포함 +
                   나머지 (THR_SAMPLES-1)개를 범위에서 랜덤 샘플링.
                   범위가 아니면 baseline 1개만.
    """
    if PERCENTILE_Q is not None:
        any_range = _is_range(PERCENTILE_Q)
        ns = THR_SAMPLES if any_range else 1
        for _ in range(ns):
            q = float(_sample_from(PERCENTILE_Q))
            thr01 = compute_threshold(
                img, mode='hist', fixed_thr_u8=FIXED_THR_U8, mask=donut,
                percentile_q=q
            )
            meta = {'mode': 'perc', 'q': q}
            yield thr01, meta
    else:
        # --- Otsu 모드 ---
        any_range = _is_range(OTSU_SCALE) or _is_range(OTSU_DELTA_U8)
        target_n = THR_SAMPLES if any_range else 1

        # 1) baseline 먼저 (scale=1.0, delta=0)
        baseline_s = 1.0
        baseline_d = 0.0
        thr01 = compute_threshold(
            img, mode='hist', fixed_thr_u8=FIXED_THR_U8, mask=donut,
            otsu_scale=baseline_s, otsu_delta_u8=baseline_d, percentile_q=None
        )
        yield thr01, {'mode': 'otsu', 'scale': baseline_s, 'delta': baseline_d}

        if target_n == 1:
            return

        # 2) 나머지 랜덤 (중복 토큰 방지: 3자리 반올림 + delta 정수화 기준)
        combos = set()
        combos.add( (round(baseline_s, 3), int(round(baseline_d))) )

        while len(combos) < target_n:
            s = float(_sample_from(OTSU_SCALE))
            d = float(_sample_from(OTSU_DELTA_U8))
            key = (round(s, 3), int(round(d)))
            if key in combos:
                continue
            combos.add(key)
            thr01 = compute_threshold(
                img, mode='hist', fixed_thr_u8=FIXED_THR_U8, mask=donut,
                otsu_scale=key[0], otsu_delta_u8=key[1], percentile_q=None
            )
            yield thr01, {'mode': 'otsu', 'scale': key[0], 'delta': key[1]}

# ========= ROI/차폐 마스크 생성 =========
def make_roi_masks_from_V(h, w, v_token: str):
    """
    radius_map:
      - (r_in, r_out) 또는
      - (block_lo, r_in, r_out)
    반환: donut_mask(ROI), r_in, r_out, block_band(차폐밴드; 없으면 0)
    """
    if v_token not in radius_map:
        raise ValueError(f"radius_map에 '{v_token}' 항목이 없습니다.")
    vals = radius_map[v_token]
    if len(vals) == 2:
        r_in, r_out = vals
        block_lo = None
    elif len(vals) == 3:
        block_lo, r_in, r_out = vals
    else:
        raise ValueError("radius_map 값은 (r_in,r_out) 또는 (block_lo,r_in,r_out) 이어야 합니다.")

    max_r = min(h, w) / 2.0 - 1.0
    r_in  = float(np.clip(r_in,  0.0, max_r))
    r_out = float(np.clip(r_out, r_in + 1e-6, max_r))
    if len(vals) == 3:
        block_lo = float(np.clip(block_lo, 0.0, r_in))

    cx = (w - 1) / 2.0; cy = (h - 1) / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    donut = ((rr >= r_in) & (rr <= r_out)).astype(np.float32)
    if len(vals) == 3:
        block_band = ((rr >= block_lo) & (rr < r_in)).astype(np.float32)
    else:
        block_band = np.zeros((h, w), dtype=np.float32)

    return donut, r_in, r_out, block_band

# ========= 증강 유틸 =========
def sample_uniform(lo, hi):
    return float(rng.uniform(lo, hi))

def apply_flip(img, mask, mode):
    if mode == "none": return img, mask
    flip_code = {'h': 1, 'v': 0, 'hv': -1}.get(mode, None)
    if flip_code is None: return img, mask
    return cv2.flip(img, flip_code), cv2.flip(mask, flip_code)

def warp_rotate_shift(img, angle_deg, dx, dy, is_mask=False):
    H, W = img.shape[:2]
    center = (W/2.0, H/2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[0,2] += dx
    M[1,2] += dy
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    # 마스크일 경우 흰색(255)으로 여백 채우기
    if is_mask:
        border_val = 255
    else:
        border_val = 0
    warped = cv2.warpAffine(
        img, M, (W, H),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val
    )
    if is_mask:
        if warped.ndim == 3:
            warped = cv2.cvtColor(to_uint8(warped), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warped = (warped >= 0.5).astype(np.float32)
    return warped

def adjust_brightness_contrast_saturation(img01, alpha=1.0, beta_u8=0, sat_scale=1.0):
    out = img01 * float(alpha) + float(beta_u8)/255.0
    out = np.clip(out, 0.0, 1.0)
    if out.ndim == 3 and out.shape[2] == 3 and (sat_scale is not None) and (abs(sat_scale-1.0) > 1e-6):
        u8 = to_uint8(out)
        hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
        S = hsv[...,1].astype(np.float32) * float(sat_scale)
        hsv[...,1] = np.clip(S, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    return out

def flip_token(mode): return {"none":"F0", "h":"Fh", "v":"Fv", "hv":"Fhv"}.get(mode, "F0")
def angle_token(a):   return f"R{('p' if a>=0 else 'm')}{abs(a):06.2f}".replace('.','_')
def shift_token(dx,dy):
    sx = f"{dx:+.1f}".replace('+','p').replace('-','m').replace('.','_')
    sy = f"{dy:+.1f}".replace('+','p').replace('-','m').replace('.','_')
    return f"Sx{sx}_y{sy}"
def color_token(alpha, beta, sat):
    a = f"A{alpha:.3f}".replace('.','_')
    b = f"B{beta:+.0f}".replace('+','p').replace('-','m')
    s = f"S{sat:.3f}".replace('.','_')
    return f"{a}_{b}_{s}"

def augment_and_save(ds_name, base_name, img, mask, thr_token_str):
    """img/mask: 0~1 float, mask는 0/1. thr_token별 하위폴더에 저장"""
    out_dir = AUG_ROOT / ds_name / base_name / thr_token_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) 기본(증강X) 1쌍 항상 저장
    base_code = "F0__Rp0_00__Sx0_0_y0_0__A1_000_Bp0_S1_000"
    imwrite(out_dir / f"{base_name}__aug__{base_code}_img{AUG_OUTPUT_EXT}", img, color_mode='rgb')   # ✅ RGB
    imwrite(out_dir / f"{base_name}__aug__{base_code}_mask{AUG_OUTPUT_EXT}", mask, color_mode='gray')  # ✅ GRAY


    if not AUG_ENABLE:
        return 1  # 저장 쌍 개수

    saved = 1
    for fmode in FLIPS:
        img_f, mask_f = apply_flip(img, mask, fmode)
        for _ in range(SAMPLES_PER_FLIP):
            ang = sample_uniform(*ROT_ANGLE_RANGE)
            dx  = sample_uniform(*SHIFT_RANGE_X)
            dy  = sample_uniform(*SHIFT_RANGE_Y)

            img_r = warp_rotate_shift(img_f, ang, dx, dy, is_mask=False)
            mask_r = warp_rotate_shift(mask_f, ang, dx, dy, is_mask=True)

            alpha = sample_uniform(*CONTRAST_ALPHA_RANGE)
            beta  = sample_uniform(*BRIGHTNESS_BETA_RANGE_U8)
            sat   = sample_uniform(*SAT_SCALE_RANGE)

            img_final = adjust_brightness_contrast_saturation(img_r, alpha, beta, sat)
            mask_final = mask_r

            code = f"{flip_token(fmode)}__{angle_token(ang)}__{shift_token(dx,dy)}__{color_token(alpha,beta,sat)}"
            img_out = out_dir / f"{base_name}__aug__{code}_img{AUG_OUTPUT_EXT}"
            msk_out = out_dir / f"{base_name}__aug__{code}_mask{AUG_OUTPUT_EXT}"
            imwrite(img_out, img_final,   color_mode='rgb')   # ✅ RGB
            imwrite(msk_out, mask_final,  color_mode='gray')  # ✅ GRAY
            saved += 1
    return saved

# ========= ENDOLFIN 내보내기 유틸 =========
def _parse_frame_spec(spec):
    """'0:5:70' -> [0,5,10,...,70], 또는 (start,step,end) 튜플도 허용."""
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        start, step, end = map(int, spec)
    elif isinstance(spec, str):
        parts = spec.split(':')
        if len(parts) != 3:
            raise ValueError("FRAME_INDEX_SPEC 형식은 'start:step:end'이어야 합니다. 예: '0:5:70'")
        start, step, end = map(int, parts)
    else:
        raise ValueError("FRAME_INDEX_SPEC는 'start:step:end' 문자열 또는 (start, step, end) 튜플이어야 합니다.")
    if step <= 0:
        raise ValueError("step은 양수여야 합니다.")
    if end < start:
        start, end = end, start
    return list(range(start, end + 1, step))

def _derive_folder_name_from_stem(stem: str):
    """EXPORT_NAME_MODE 에 따라 폴더명 결정."""
    if EXPORT_NAME_MODE == "base":
        return stem.split("__aug__")[0]
    return stem  # "full"

def _to_base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0: return "0"
    s = []
    while n > 0:
        n, r = divmod(n, 36)
        s.append(digits[r])
    return "".join(reversed(s))

def make_unique_id(parts, length=10) -> str:
    key = "|".join(map(str, parts))
    h = hashlib.sha1(key.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    b36 = _to_base36(n)
    return (b36 * ((length // len(b36)) + 1))[:length]

def export_endolfin_from_aug():
    """
    AUG_ROOT 하위 *_img.<ext>, *_mask.<ext> 쌍을 찾아
    ENDOLFIN 포맷으로 복제.
    폴더 이름: <folder_name_base>__<uid>
      - 앞 숫자 접두사 제거
      - uid는 (img_path, mask_path, folder_name_base, aug_token)에서 결정적 생성
    """
    if not EXPORT_ENDOLFIN:
        return 0

    jpeg_dir = ENDOLFIN_TRAIN_ROOT / "JPEGImages"
    mask_dir = ENDOLFIN_TRAIN_ROOT / "object_masks"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ext = AUG_OUTPUT_EXT.lower()
    img_glob = f"*_*img{ext}"

    frames = _parse_frame_spec(FRAME_INDEX_SPEC)

    mapping_rows = []
    saved_pairs = 0

    img_paths = sorted(AUG_ROOT.rglob(img_glob))

    for img_path in img_paths:
        if not img_path.is_file():
            continue

        stem = img_path.stem  # 예: base__aug__..._img
        if not stem.endswith("_img"):
            continue

        mask_path = img_path.with_name(stem[:-4] + "_mask" + img_path.suffix)
        if not mask_path.exists():
            continue

        stem_no_img = stem[:-4]  # "_img" 제거
        parts = stem_no_img.split("__aug__")
        if EXPORT_NAME_MODE == "base":
            folder_name_base = parts[0]                    # "__aug__" 앞부분만
        else:
            folder_name_base = stem_no_img                 # 전체 사용("full")
        aug_token = parts[1] if len(parts) > 1 else "noaug"

        # 뒤에 uid 추가 (접두 숫자 제거)
        uid = make_unique_id([img_path, mask_path, folder_name_base, aug_token], length=10)
        unique_folder = f"{folder_name_base}__{uid}"

        out_img_folder = jpeg_dir / unique_folder
        out_msk_folder = mask_dir / unique_folder
        out_img_folder.mkdir(parents=True, exist_ok=True)
        out_msk_folder.mkdir(parents=True, exist_ok=True)

        src_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        src_msk = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        for f in frames:
            fname = f"{folder_name_base}__aug__{aug_token}_{f:05d}{ext}"
            # ✅ 내보내기 시에도 채널 고정
            imwrite(out_img_folder / fname, src_img, color_mode='rgb',  jpeg_quality=ENDOLFIN_JPEG_QUALITY)  # RGB
            imwrite(out_msk_folder / fname, src_msk, color_mode='gray', jpeg_quality=ENDOLFIN_JPEG_QUALITY)  # GRAY

        mapping_rows.append([unique_folder, str(img_path), str(mask_path), folder_name_base, aug_token, uid])
        saved_pairs += 1

    # 매핑 CSV 저장
    map_csv = ENDOLFIN_TRAIN_ROOT / "_folder_mapping.csv"
    with open(map_csv, "w", encoding="utf-8") as f:
        f.write("unique_folder,img_path,mask_path,folder_name_base,aug_token,uid\n")
        for row in mapping_rows:
            f.write(",".join(row) + "\n")

    return saved_pairs

# ========= 메인 처리 =========
def main():
    cons_sets = parse_conditions(conditions)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if AUG_ENABLE: AUG_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    total_aug_pairs = 0

    for root in root_dirs:
        root_p = Path(root)
        if not root_p.is_dir():
            print(f"[경고] 폴더 없음: {root_p}")
            continue

        ds_name = root_p.name  # P0 / P2
        out_dir_root = OUT_ROOT / ds_name
        out_dir_root.mkdir(parents=True, exist_ok=True)

        for fname in sorted(os.listdir(root_p)):
            if not fname.lower().endswith('.bmp'):
                continue

            total += 1
            rec = parse_filename(fname)

            if rec is None:
                continue
            if not record_matches(rec, cons_sets):
                continue
            matched += 1

            src_path = root_p / fname
            try:
                img = load_image(src_path, grayscale=USE_GRAYSCALE)
            except FileNotFoundError:
                print(f"[경고] 읽기 실패: {src_path}")
                continue

            H, W = img.shape[:2]
            name = rec['name']                     # 폴더명 스템
            out_dir = out_dir_root / name
            out_dir.mkdir(parents=True, exist_ok=True)

            # === 도넛 ROI + 차폐 밴드 ===
            if USE_DONUT_ROI:
                if rec['V'] in radius_map:
                    donut, r_in, r_out, block_band = make_roi_masks_from_V(H, W, rec['V'])
                else:
                    print(f"[경고] radius_map에 '{rec['V']}' 없음. 전체 ROI 사용.")
                    donut = np.ones((H, W), dtype=np.float32)
                    block_band = np.zeros((H, W), dtype=np.float32)
            else:
                donut = np.ones((H, W), dtype=np.float32)
                block_band = np.zeros((H, W), dtype=np.float32)

            # === 임계값 샘플 생성 & 저장 (thr 토큰을 '하위 폴더명'으로 사용) ===
            aug_saved_local = 0
            thr_count = 0
            for thr01, meta in gen_threshold_variants(img, donut):
                thr_count += 1
                # (1) 검출 마스크 (ROI 내에서만)
                det_mask = build_binary_mask(img, thr01, keep=KEEP_SIDE).astype(np.float32)
                det_mask = (det_mask * donut).astype(np.float32)
                det_mask = filter_small_components(det_mask, min_area_ratio=MIN_AREA_RATIO)

                # (2) 최종 keep 마스크 = (1 - donut) + det, 단 block_band는 무조건 0으로 차폐
                keep_mask = ((1.0 - donut) + det_mask).astype(np.float32)
                keep_mask = (keep_mask * (1.0 - block_band)).astype(np.float32)

                # ---- 저장 폴더: .../<이름>/<thr토큰>/ ----
                thr_tok = thr_code_token(meta)
                thr_dir = out_dir / thr_tok
                thr_dir.mkdir(parents=True, exist_ok=True)

                # 원본/도넛도 thr 폴더에 저장
                imwrite(thr_dir / f"{name}_img{OUTPUT_EXT}", img, color_mode='rgb')   # ✅ 이미지: RGB(3채널)
                imwrite(thr_dir / f"{name}_donut_mask{OUTPUT_EXT}", donut, color_mode='gray')  # ✅ 마스크: GRAY(1채널)

                # (검출/최종 마스크) 저장
                if SAVE_DET_MASK_DEBUG:
                    imwrite(thr_dir / f"{name}_det_mask{OUTPUT_EXT}", det_mask, color_mode='gray')  # ✅ GRAY
                imwrite(thr_dir / f"{name}_mask{OUTPUT_EXT}", keep_mask, color_mode='gray')  # ✅ GRAY

                # 히스토그램 (thr 폴더에 저장)
                if SAVE_HIST:
                    counts, bin_edges = make_histogram(img, bins=64, mask=donut)
                    save_histogram_image(counts, bin_edges, thr_dir / f"{name}_hist{OUTPUT_EXT}", thr01=thr01, size=(640, 320))
                    with open(thr_dir / f"{name}_hist.csv", "w") as hf:
                        hf.write("bin_left,bin_right,count\n")
                        for i in range(len(counts)):
                            hf.write(f"{bin_edges[i]:.6f},{bin_edges[i+1]:.6f},{int(counts[i])}\n")

                # === 증강: 임계값 샘플별로 동일한 구조 사용
                aug_saved_local += augment_and_save(ds_name, name, img, keep_mask, thr_tok)

            total_aug_pairs += aug_saved_local
            print(f"[OK] {ds_name}/{fname} -> {out_dir} | thr_variants={thr_count} | aug_pairs_saved={aug_saved_local}")

    print(f"\n완료. 총 파일: {total}, 매칭: {matched}")
    if AUG_ENABLE:
        print(f"증강 저장 쌍(기본 포함): {total_aug_pairs} → 출력: {AUG_ROOT.resolve()}")
    print(f"마스크 출력 폴더: {OUT_ROOT.resolve()}")

    # ====== ENDOLFIN export ======
    if EXPORT_ENDOLFIN:
        pairs = export_endolfin_from_aug()
        print(f"ENDOLFIN 내보내기 완료: {pairs}개 폴더 복제됨 → {ENDOLFIN_TRAIN_ROOT.resolve()}")

if __name__ == "__main__":
    main()
