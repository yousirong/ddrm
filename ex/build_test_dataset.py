import os
import re
import cv2
import numpy as np
import optuna
from math import sqrt
from pathlib import Path
import hashlib  # 고유 ID 생성을 위해

# ========= 사용자 설정 =========
root_dirs = [
    "./dataset/datasets_v0.04/P0",
    "./dataset/datasets_v0.04/P2",
]

selection_ref = ['CY', 'ON', 'PC', 'DC', 'V3~7', '000~999']          # 기준(A)
# selection_tgt = ['CY', 'OY', 'PC', 'D000~D359', 'V3~7', '000~999']   # 타깃(B)
selection_tgt = ['CY', 'ON', 'PL', 'D000~D359', 'V3~7', '000~999']

# diff_keys = ['O']
diff_keys = ['P']
EXACTLY_ONE_DIFF = False
ignore_keys = ['D']

# ---- A(참조)만 워핑(회전+이동) ----
ANGLE_MIN, ANGLE_MAX = -7.0, 7.0
SHIFT_RANGE_X = (-2.0, 2.0)
SHIFT_RANGE_Y = (-2.0, 2.0)

USE_GRAYSCALE = True
INTERP = cv2.INTER_LINEAR
BORDER_MODE = cv2.BORDER_CONSTANT
BORDER_FILL01 = 1.0   # 여백을 흰색(1.0)으로 채움

VALUE_SOURCE = 'diff'           # 'diff' | 'A' | 'B'
THRESH_MODE  = 'hist'           # 'hist' | 'fixed'
FIXED_THR_U8 = 50
DIFF_KEEP    = 'high'
MIN_AREA_RATIO = 0.0005
HIST_BINS = 64

# --- Otsu/Percentile 미세조정 (스칼라 또는 [min, max]) ---
OTSU_DELTA_U8 = [-10, 10]
OTSU_SCALE    = [0.80, 1.20]
PERCENTILE_Q  = None
THR_SAMPLES   = 2

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

JPEG_QUALITY = 100   # 모든 저장 .jpg 품질

# ========= 저장 채널 옵션(핵심) =========
# 'gray' → 1채널 그레이스케일로 저장
# 'rgb'  → 3채널 컬러로 저장(OpenCV는 BGR로 기록되지만 3채널 보장)
SAVE_COLOR_MODE = "rgb"   # 'gray' 또는 'rgb'

# ========= 유틸 =========
def parse_filename(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() != '.bmp': return None
    parts = name.split('_')
    if len(parts) != 6: return None
    C, O, P, D, V, N = parts
    return {'C': C, 'O': O, 'P': P, 'D': D, 'V': V, 'N': N, 'name': name, 'ext': ext}

def _expand_V_token(tok):
    if '~' in tok:
        left, right = tok.split('~')
        if not left.startswith('V'): return []
        v0 = int(left[1:]); v1 = int(right)
        return [f"V{v}" for v in range(v0, v1 + 1)]
    return [tok]

def _expand_N_token(tok):
    if '~' in tok:
        left, right = tok.split('~'); width = len(left)
        n0 = int(left); n1 = int(right)
        return [str(n).zfill(width) for n in range(n0, n1 + 1)]
    return [tok]

def _expand_D_token(tok):
    t = tok.upper()
    if t == "DC": return ["DC"]
    if '~' in t:
        left, right = t.split('~')
        if not (left.startswith('D') and right.startswith('D')): return []
        d0 = int(left[1:]); d1 = int(right[1:])
        width = len(left) - 1
        return [f"D{str(i).zfill(width)}" for i in range(d0, d1 + 1)]
    if t.startswith('D') and (len(t) == 1 or t[1:].isdigit()): return [t]
    return []

def parse_selection_tokens(tokens):
    constraints = {'C': None, 'O': None, 'P': None, 'D': None, 'V': None, 'N': None}
    for t in tokens:
        if t.startswith('V'):
            vals = _expand_V_token(t)
            if vals: constraints['V'] = set(vals) if constraints['V'] is None else constraints['V'].union(vals); continue
        if t.startswith('D'):
            vals = _expand_D_token(t)
            if vals: constraints['D'] = set(vals) if constraints['D'] is None else constraints['D'].union(vals); continue
        if t.replace('~', '').isdigit():
            vals = _expand_N_token(t)
            if vals: constraints['N'] = set(vals) if constraints['N'] is None else constraints['N'].union(vals); continue
        if t.isalpha() and t.isupper():
            if t in {'ON','OY'}:
                constraints['O'] = {t} if constraints['O'] is None else constraints['O'].union({t})
            elif len(t) >= 2:
                if constraints['C'] is None: constraints['C'] = {t}
                else: constraints['P'] = {t} if constraints['P'] is None else constraints['P'].union({t})
    return constraints

def match_with_constraints(rec, cons):
    for k, allowed in cons.items():
        if allowed is None: 
            continue
        if rec[k] not in allowed:
            return False
    return True

def load_image(path, grayscale=True):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32) / 255.0

def _resolve_border_value_for_image(img: np.ndarray, fill01: float = 1.0):
    ch = img.shape[2] if img.ndim == 3 else 1
    if img.dtype == np.uint8:
        v = int(round(np.clip(fill01, 0.0, 1.0) * 255))
    else:
        v = float(np.clip(fill01, 0.0, 1.0))
    if ch == 1:
        return v
    return tuple([v] * ch)

def rotate_shift_image_and_mask(img, angle_deg, dx, dy,
                                interp=INTERP, border_mode=BORDER_MODE):
    """
    A(참조) 영상에 회전+평행이동 적용.
    - 영상 여백은 흰색(BORDER_FILL01)으로 채움
    - 유효마스크(mask_warp)는 화면 밖 0 (평가용)
    """
    h, w = img.shape[:2]
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[0, 2] += float(dx)
    M[1, 2] += float(dy)

    img_border_val = _resolve_border_value_for_image(img, BORDER_FILL01)
    warped = cv2.warpAffine(img, M, (w, h),
                            flags=interp,
                            borderMode=border_mode,
                            borderValue=img_border_val)

    mask = np.ones((h, w), dtype=np.float32)
    mask_warp = cv2.warpAffine(mask, M, (w, h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=border_mode,
                               borderValue=0.0)
    mask_warp = (mask_warp > 0.5).astype(np.float32)
    return warped, mask_warp

def make_same_size(a, b):
    ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
    h = min(ha, hb); w = min(wa, wb)
    def center_crop(img, h, w):
        H, W = img.shape[:2]; y0 = (H-h)//2; x0 = (W-w)//2
        return img[y0:y0+h, x0:x0+w, ...] if img.ndim==3 else img[y0:y0+h, x0:x0+w]
    if (ha, wa) != (h, w): a = center_crop(a, h, w)
    if (hb, wb) != (h, w): b = center_crop(b, h, w)
    return a, b

def mse_rmse_r2(a, b, valid_mask=None):
    if valid_mask is not None:
        if a.ndim == 3: valid_mask_exp = np.repeat(valid_mask[:, :, None], a.shape[2], axis=2)
        else: valid_mask_exp = valid_mask
        a = a * valid_mask_exp; b = b * valid_mask_exp
    diff = a - b
    mse = np.mean(diff**2); rmse = sqrt(mse)
    y_true = a.flatten(); y_pred = b.flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return mse, rmse, r2

# ---- ROI/차폐 마스크 생성 ----
def make_roi_masks_from_V(h, w, v_token: str):
    """
    radius_map이 2튜플이면 (r_in, r_out) 도넛만 생성, block_mask는 0
    radius_map이 3튜플이면 (block_lo, r_in, r_out)
      - block_lo ~ r_in : block(차폐) 마스크 = 1
      - r_in ~ r_out    : donut(ROI) 마스크 = 1
    반환: donut_mask, r_in, r_out, block_mask
    """
    if v_token not in radius_map:
        raise ValueError(f"radius_map에 '{v_token}' 항목이 없습니다.")
    vals = radius_map[v_token]
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

def compute_threshold(img01: np.ndarray, mode='hist', fixed_thr_u8=50, mask: np.ndarray=None,
                      otsu_delta_u8: float = 0.0, otsu_scale: float = 1.0, percentile_q: float = None):
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
    thr_u8 = int(np.clip(thr_u8 * float(otsu_scale) + float(otsu_delta_u8), 0, 255))
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
    if color_mode is None:
        color_mode = SAVE_COLOR_MODE

    arr = img
    if not isinstance(arr, np.ndarray):
        raise ValueError("imwrite: img must be a numpy array.")
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)

    if color_mode == 'rgb':
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    else:  # 'gray'
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)

    path_str = str(path)
    q = JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality)
    if path_str.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(path_str, arr, [cv2.IMWRITE_JPEG_QUALITY, q])
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
        txt = f"thr={thr01:.3f}"
        cv2.putText(img, txt, (min(x_thr + 5, W - 120), pad_t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    # ✅ 히스토그램은 시각화 이미지이므로 RGB로 저장
    imwrite(out_path, img, color_mode='rgb', jpeg_quality=JPEG_QUALITY)

def apply_valid_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    if a.ndim == 3: mask_exp = np.repeat(mask[:, :, None], a.shape[2], axis=2)
    else: mask_exp = mask
    return a * mask_exp, b * mask_exp

# ========= 임계값 샘플링 도우미 =========
rng = np.random.default_rng(42)
def _is_range(x): return isinstance(x, (list, tuple)) and len(x) == 2
def _sample_from(x):
    if _is_range(x):
        lo, hi = float(x[0]), float(x[1]); return float(rng.uniform(lo, hi))
    return float(x)

def thr_code_token(meta):
    if meta['mode'] == 'perc':
        q = meta['q']; return f"thrP_q{q:.3f}".replace('.', '_')
    else:
        s = meta['scale']; d = meta['delta']
        s_tok = f"s{s:.3f}".replace('.', '_')
        d_tok = f"d{d:+.0f}".replace('+','p').replace('-','m')
        return f"thrO_{s_tok}_{d_tok}"

def gen_thr_param_variants():
    if PERCENTILE_Q is not None:
        if _is_range(PERCENTILE_Q):
            for _ in range(THR_SAMPLES):
                yield {'mode': 'perc', 'q': _sample_from(PERCENTILE_Q)}
        else:
            yield {'mode': 'perc', 'q': float(PERCENTILE_Q)}
    else:
        yield {'mode': 'otsu', 'scale': 1.0, 'delta': 0.0}
        if _is_range(OTSU_SCALE) or _is_range(OTSU_DELTA_U8):
            need = max(THR_SAMPLES - 1, 0)
            combos = set([(round(1.0, 3), int(round(0.0)))])
            while len(combos) - 1 < need:
                s = _sample_from(OTSU_SCALE if _is_range(OTSU_SCALE) else OTSU_SCALE)
                d = _sample_from(OTSU_DELTA_U8 if _is_range(OTSU_DELTA_U8) else OTSU_DELTA_U8)
                combos.add((round(s, 3), int(round(d))))
            for (s, d) in combos:
                if s == 1.0 and d == 0: continue
                yield {'mode': 'otsu', 'scale': float(s), 'delta': float(d)}

# ======= 토큰 유틸 =======
def angle_token(angle_deg: float) -> str:
    sign = 'p' if angle_deg >= 0 else 'm'
    val = abs(angle_deg)
    s = f"{val:06.2f}".replace('.', '_')
    return f"rot_{sign}{s}deg"

def shift_token(dx: float, dy: float) -> str:
    def fmt(v):
        return ('p' if v >= 0 else 'm') + f"{abs(v):05.2f}".replace('.', '_')
    return f"Sx{fmt(dx)}_Sy{fmt(dy)}"

# ======= 목적함수: (항상) A만 회전 + 시프트, ROI 내 MSE 최소화 =======
def objective_pose_ref_warp_only(trial, imgA, imgB, v_token):
    angle = trial.suggest_float("angle_deg", ANGLE_MIN, ANGLE_MAX)
    dx    = trial.suggest_float("shift_x",  SHIFT_RANGE_X[0], SHIFT_RANGE_X[1])
    dy    = trial.suggest_float("shift_y",  SHIFT_RANGE_Y[0], SHIFT_RANGE_Y[1])

    warpedA, mask_warp = rotate_shift_image_and_mask(imgA, angle, dx, dy)
    A2, B2 = make_same_size(warpedA, imgB)
    mask_warp = make_same_size(mask_warp, mask_warp)[0]

    H, W = A2.shape[:2]
    donut, _, _, _ = make_roi_masks_from_V(H, W, v_token)
    valid_mask = (mask_warp * donut).astype(np.float32)

    mse, _, _ = mse_rmse_r2(A2, B2, valid_mask=valid_mask)
    return mse

# ======= 고유 ID 생성(sha1 → base36) =======
def _to_base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0: return "0"
    s = []
    while n > 0:
        n, r = divmod(n, 36)
        s.append(digits[r])
    return "".join(reversed(s))

def make_unique_id(parts, length=10) -> str:
    """
    parts(list[str])를 sha1 해싱해 base36로 변환 후 length 글자로 자름(결정적).
    """
    if isinstance(parts, (list, tuple)):
        key = "|".join(map(str, parts))
    else:
        key = str(parts)
    h = hashlib.sha1(key.encode("utf-8")).digest()  # 20 bytes
    n = int.from_bytes(h[:8], "big")  # 상위 8바이트만 사용
    b36 = _to_base36(n)
    if len(b36) < length:
        b36 = (b36 * ((length // len(b36)) + 1))[:length]
    else:
        b36 = b36[:length]
    return b36

# ========= Test 세트 복제 유틸 =========
def parse_seq_spec(spec):
    if isinstance(spec, str):
        a,b,c = map(int, spec.split(':'))
        start, step, end = a,b,c
    else:
        start, step, end = spec
    if step <= 0: raise ValueError("step은 1 이상이어야 합니다.")
    return list(range(start, end + 1, step))

def export_endolfin_test(
    src_root = Path("./rotation_optuna_results"),
    dst_root = Path("./dataset/ENDOLFIN_new/Test"),
    seq_spec = "0:5:70",
    zpad = 5,
):
    """
    항상 selection_tgt(B, 타깃) 이미지를 Test에 복제.
    - JPEGImages : 타깃(B) 이미지 (원본) 복제
    - object_masks : 타깃(B) 좌표계의 valid_mask 복제
    폴더명에는 thrTok/rotTok 유지 + 맨 끝에 결정적 고유 ID를 10자 붙임.
      예) JPEGImages/CY_OY_PC_D000_V3_000__thrO_s1_000_dp0__rot_m000_39deg__abc123xyz0
          object_masks/CY_OY_PC_D000_V3_000__thrO_s1_000_dp0__rot_m000_39deg__abc123xyz0
    """
    seq = parse_seq_spec(seq_spec)

    pat_mask = re.compile(r"^(.+?)(?:__|_)valid_mask\.jpg$", re.IGNORECASE)     # <b_stem>__valid_mask.jpg
    pat_rot_token = re.compile(r"(rot_[pm]\d{3}_\d+deg)", re.IGNORECASE)        # 폴더명에서 rotTok 추출

    jpeg_root = Path(dst_root) / "JPEGImages"
    mask_root = Path(dst_root) / "object_masks"
    copied = 0

    # thr 디렉토리 순회
    for thr_dir in sorted(Path(src_root).glob("*/thr*")):
        thr_tok = thr_dir.name
        # 각 회전 폴더(angdir) 순회
        for angdir in sorted(thr_dir.iterdir()):
            if not angdir.is_dir(): continue

            # rot 토큰은 폴더명에서 추출(참조 A가 얼마나 회전됐는지 기록용)
            m_rot = pat_rot_token.search(angdir.name)
            if not m_rot:
                continue
            rot_tok = m_rot.group(1)

            # 타깃 마스크(베이스네임=타깃 stem)
            mask_path = None
            tgt_stem  = None
            for f in angdir.glob("*valid_mask.jpg"):
                mm = pat_mask.match(f.name)
                if mm:
                    mask_path = f
                    tgt_stem  = mm.group(1)  # 항상 selection_tgt의 stem
                    break
            if mask_path is None or tgt_stem is None:
                continue

            # 타깃(B) 이미지(원본) 경로
            img_path = angdir / f"{tgt_stem}.jpg"
            if not img_path.exists():
                print(f"[경고] 타깃 이미지 없음: {img_path}")
                continue

            # 고유 ID 생성(결정적): tgt_stem + thr_tok + rot_tok
            uid = make_unique_id([tgt_stem, thr_tok, rot_tok], length=10)

            subname = f"{tgt_stem}__{thr_tok}__{rot_tok}__{uid}"
            out_img_dir = jpeg_root / subname
            out_msk_dir = mask_root / subname
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_msk_dir.mkdir(parents=True, exist_ok=True)

            # 로드
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                print(f"[경고] 읽기 실패: {img_path} / {mask_path}")
                continue

            for n in seq:
                fname = f"{n:0{zpad}d}.jpg"
                # ✅ 이미지는 RGB 3채널로 저장
                imwrite(out_img_dir / fname, img, color_mode='rgb',  jpeg_quality=JPEG_QUALITY)
                # ✅ 마스크는 GRAY 1채널로 저장
                imwrite(out_msk_dir / fname, msk, color_mode='gray', jpeg_quality=JPEG_QUALITY)
                copied += 2

    print(f"[OK] Test 세트 복제 완료(타깃 기준): 파일 {copied}개 저장 → {Path(dst_root).resolve()}")

# ========= 메인 =========
def main():
    # 스캔
    entries = []
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"[경고] 폴더 없음: {root_dir}")
            continue
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith('.bmp'): continue
            rec = parse_filename(fname)
            if rec is None: continue
            rec['path'] = os.path.join(root_dir, fname)
            entries.append(rec)
    if not entries: raise RuntimeError(f"No BMP files in roots: {root_dirs}")

    cons_ref = parse_selection_tokens(selection_ref)
    cons_tgt = parse_selection_tokens(selection_tgt)
    A_candidates = [e for e in entries if match_with_constraints(e, cons_ref)]
    B_candidates = [e for e in entries if match_with_constraints(e, cons_tgt)]
    if not A_candidates: raise RuntimeError("No reference(A) files matched selection_ref.")
    if not B_candidates: raise RuntimeError("No target(B) files matched selection_tgt.")

    all_fields = ('C','O','P','D','V','N')
    use_diff = [k for k in diff_keys if k in all_fields]
    use_ignore = [k for k in ignore_keys if k in all_fields]
    if not use_diff: raise ValueError("diff_keys must contain at least one of: C,O,P,D,V,N")
    base_fields = tuple(f for f in all_fields if f not in use_diff and f not in use_ignore)
    def make_key(e): return tuple(e[k] for k in base_fields)

    idxA, idxB = {}, {}
    for e in A_candidates: idxA.setdefault(make_key(e), []).append(e)
    for e in B_candidates: idxB.setdefault(make_key(e), []).append(e)
    common_keys = sorted(set(idxA.keys()) & set(idxB.keys()))
    if not common_keys:
        sampleA = list({make_key(e) for e in A_candidates})[:10]
        sampleB = list({make_key(e) for e in B_candidates})[:10]
        raise RuntimeError(f"No pairs share identical {base_fields} (diff:{use_diff}, ignore:{use_ignore}).\nA keys sample: {sampleA}\nB keys sample: {sampleB}")

    out_root = Path("./rotation_optuna_results"); out_root.mkdir(parents=True, exist_ok=True)

    print(f"페어 키 개수: {len(common_keys)}")
    for k in common_keys:
        A_list = idxA[k]; B_list = idxB[k]
        key_desc = "_".join(f"{f}={val}" for f, val in zip(base_fields, k))

        for A_rec in A_list:
            for B_rec in B_list:
                diffs = [f for f in use_diff if A_rec[f] != B_rec[f]]
                if EXACTLY_ONE_DIFF:
                    if len(diffs) != 1: continue
                else:
                    if len(diffs) == 0: continue

                A_path, B_path = A_rec['path'], B_rec['path']
                print(f"\n=== Pair [{key_desc}] | " + " / ".join(f"{f}:{A_rec[f]} -> {B_rec[f]}" for f in use_diff) + " ===")
                print(f"A: {os.path.basename(A_path)}"); print(f"B: {os.path.basename(B_path)}")

                imgA = load_image(A_path, grayscale=USE_GRAYSCALE)
                imgB = load_image(B_path, grayscale=USE_GRAYSCALE)
                imgA, imgB = make_same_size(imgA, imgB)
                v_token = A_rec['V']

                # ---- 1) (항상) A만 회전+시프트 최적화 (TPE) ----
                study = optuna.create_study(direction="minimize")
                study.optimize(
                    lambda t: objective_pose_ref_warp_only(t, imgA, imgB, v_token),
                    n_trials=100,
                    show_progress_bar=True
                )
                best_angle = study.best_params["angle_deg"]
                best_dx    = study.best_params["shift_x"]
                best_dy    = study.best_params["shift_y"]
                print(f"[pose-opt] angle={best_angle:.4f} deg, dx={best_dx:.3f}, dy={best_dy:.3f} | MSE: {study.best_value:.6f}")

                # 최적 파라미터 적용 (A만 워핑, B는 고정)
                warpedA_opt, warp_maskA = rotate_shift_image_and_mask(imgA, best_angle, best_dx, best_dy)
                A_opt, B_opt = make_same_size(warpedA_opt, imgB)
                warp_maskA   = make_same_size(warp_maskA, warp_maskA)[0]
                H, W = A_opt.shape[:2]
                donut, r_in, r_out, block_band = make_roi_masks_from_V(H, W, v_token)

                # 히스토그램/마스크 산출용 소스
                if VALUE_SOURCE == 'diff':
                    #src_for_hist = np.abs(A_opt - B_opt); src_tag = 'diff'
                    src_for_hist = np.maximum(B_opt - A_opt, 0.0); src_tag = 'diff'
                elif VALUE_SOURCE == 'A':
                    src_for_hist = A_opt; src_tag = 'A'
                else:
                    src_for_hist = B_opt; src_tag = 'B'

                # ---- 2) 고정 파라미터에서 임계값 여러 개 적용 ----
                ang_tok = angle_token(best_angle)
                shf_tok = shift_token(best_dx, best_dy)

                # 폴더 이름 구성
                a_name = os.path.basename(A_path); a_stem, _ = os.path.splitext(a_name)  # ref stem
                b_name = os.path.basename(B_path); b_stem, _ = os.path.splitext(b_name)  # tgt stem

                diff_desc = "__".join([f"{f}_{A_rec[f]}-to-{B_rec[f]}" for f in use_diff if A_rec[f] != B_rec[f]])
                key_part  = "__".join([f"{f}_{A_rec[f]}" for f in base_fields])
                ign_part  = "__".join([f"{f}_{A_rec[f]}" for f in use_ignore]) if use_ignore else ""
                pair_folder_name = f"{key_part}__{diff_desc}" + (f"__IGN_{ign_part}" if ign_part else "")
                pair_root = out_root / pair_folder_name
                pair_root.mkdir(parents=True, exist_ok=True)

                for thr_meta in gen_thr_param_variants():
                    thr_tok = thr_code_token(thr_meta)
                    sub_root = pair_root / thr_tok
                    sub_root.mkdir(parents=True, exist_ok=True)

                    # angdir은 'ref가 어떻게 워핑되었는지' 기록(타깃은 고정)
                    angdir_name = f"{a_stem}__{ang_tok}__{shf_tok}"
                    pair_out = sub_root / angdir_name
                    pair_out.mkdir(parents=True, exist_ok=True)

                    # 임계값 결정(도넛 ROI 내부)
                    if thr_meta['mode'] == 'perc':
                        thr_opt = compute_threshold(src_for_hist, mode=THRESH_MODE, fixed_thr_u8=FIXED_THR_U8,
                                                    mask=donut, percentile_q=thr_meta['q'])
                    else:
                        thr_opt = compute_threshold(src_for_hist, mode=THRESH_MODE, fixed_thr_u8=FIXED_THR_U8,
                                                    mask=donut, otsu_scale=thr_meta['scale'], otsu_delta_u8=thr_meta['delta'])

                    # 검출 마스크(타깃 좌표계, ROI=donut). 최종 valid는 block_band로 차폐.
                    det_mask = build_binary_mask(src_for_hist, thr_opt, keep=DIFF_KEEP)
                    det_mask = (det_mask * donut).astype(np.float32)
                    det_mask = filter_small_components(det_mask, min_area_ratio=MIN_AREA_RATIO)

                    # 저장용 유효마스크(항상 타깃 좌표계)
                    #   - 기본: (1 - donut) + det_mask  (ROI 안은 검출부만, 바깥은 1)
                    #   - 추가: block_band(예: 42~72)는 무조건 0으로 차폐
                    valid_mask_tgt = ((1.0 - donut) + det_mask).astype(np.float32)
                    valid_mask_tgt = (valid_mask_tgt * (1.0 - block_band)).astype(np.float32)

                    # 메트릭(리포트용)
                    mse_warp_only, rmse_warp_only, r2_warp_only = mse_rmse_r2(A_opt, B_opt, valid_mask=warp_maskA)
                    A_v, B_v = apply_valid_mask(A_opt, B_opt, valid_mask_tgt)
                    mse_v, rmse_v, r2_v = mse_rmse_r2(A_v, B_v, valid_mask=None)

                    # 파일명 (항상 타깃 stem으로 결과 저장)
                    a_raw_name      = f"{a_stem}.jpg"
                    b_tgt_name      = f"{b_stem}.jpg"                   # 타깃 원본
                    a_warp_name     = f"{a_stem}__{ang_tok}.jpg"        # 워핑된 A(참고)
                    diff_jpg_name   = f"{a_stem}__vs__{b_stem}__{ang_tok}_diff.jpg"
                    donut_jpg_name  = f"{b_stem}__donut_roi.jpg"        # 도넛(72~230 등)
                    det_jpg_name    = f"{b_stem}__{src_tag}_det_mask_raw.jpg"
                    valid_jpg_name  = f"{b_stem}__valid_mask.jpg"       # 최종 유효마스크(차폐 포함)
                    hist_jpg_name   = f"{b_stem}__{src_tag}_hist.jpg"

                    # 저장(.jpg 통일)
                    # ✅ 이미지는 RGB 3채널로 저장
                    imwrite(pair_out / a_raw_name,  imgA,  color_mode='rgb',  jpeg_quality=JPEG_QUALITY)
                    imwrite(pair_out / b_tgt_name,  imgB,  color_mode='rgb',  jpeg_quality=JPEG_QUALITY)
                    imwrite(pair_out / a_warp_name, A_opt, color_mode='rgb',  jpeg_quality=JPEG_QUALITY)

                    #diff_vis = np.abs(A_opt - B_opt)
                    diff_vis = np.maximum(B_opt - A_opt, 0.0)
                    imwrite(pair_out / diff_jpg_name, diff_vis, color_mode='rgb', jpeg_quality=JPEG_QUALITY)

                    # ✅ 마스크/ROI 류는 GRAY 1채널로 저장
                    imwrite(pair_out / donut_jpg_name, donut,             color_mode='gray', jpeg_quality=JPEG_QUALITY)
                    imwrite(pair_out / det_jpg_name,   det_mask,          color_mode='gray', jpeg_quality=JPEG_QUALITY)
                    imwrite(pair_out / valid_jpg_name, valid_mask_tgt,    color_mode='gray', jpeg_quality=JPEG_QUALITY)

                    counts, bin_edges = make_histogram(src_for_hist, bins=HIST_BINS, mask=donut)
                    save_histogram_image(counts, bin_edges, pair_out / hist_jpg_name, thr01=thr_opt, size=(640, 320))

                    # 메트릭 텍스트
                    with open(pair_out / f"{a_stem}__vs__{b_stem}__{ang_tok}_metrics.txt", "w") as f:
                        f.write(f"BASE (equal fields): {base_fields}\n")
                        for fkey in base_fields: f.write(f"  {fkey}: {A_rec[fkey]}\n")
                        f.write(f"DIFF fields: {use_diff}\n")
                        for dkey in use_diff:  f.write(f"  {dkey}: {A_rec[dkey]} -> {B_rec[dkey]}\n")
                        if use_ignore:
                            f.write(f"IGNORE fields: {use_ignore}\n")
                            for ikey in use_ignore:
                                f.write(f"  A.{ikey}: {A_rec[ikey]} | B.{ikey}: {B_rec[ikey]}\n")
                        f.write(f"A(raw): {a_raw_name}\n")
                        f.write(f"B(target raw): {b_tgt_name}\n")
                        f.write(f"A(warped): {a_warp_name}\n")
                        f.write(f"[thr_meta] {thr_meta}\n")
                        f.write(f"Best angle (deg): {best_angle:.4f}\n")
                        f.write(f"Best shift (dx, dy px): ({best_dx:.4f}, {best_dy:.4f})\n")
                        f.write(f"MSE(with warp-valid only): {mse_warp_only:.8f}\n")
                        f.write(f"RMSE(with warp-valid only): {rmse_warp_only:.8f}\n")
                        f.write(f"R2(with warp-valid only): {r2_warp_only:.8f}\n")
                        f.write("\n[Mask/ROI]\n")
                        f.write(f"VALUE_SOURCE: {VALUE_SOURCE}\n")
                        f.write(f"THRESH_MODE: {THRESH_MODE}\n")
                        if THRESH_MODE == 'fixed':
                            f.write(f"FIXED_THR_U8: {FIXED_THR_U8}\n")
                        f.write(f"Resolved thr01 (donut, V={A_rec['V']}): {thr_opt:.6f}\n")
                        f.write(f"MIN_AREA_RATIO: {MIN_AREA_RATIO:.8f}\n")
                        f.write(f"DONUT radii: r_in={r_in:.1f}px, r_out={r_out:.1f}px\n")
                        f.write(f"(target-valid) MSE: {mse_v:.8f}\n")
                        f.write(f"(target-valid) RMSE: {rmse_v:.8f}\n")
                        f.write(f"(target-valid) R2: {r2_v:.8f}\n")

    print(f"\n완료. 결과 폴더: {out_root.resolve()}")

if __name__ == "__main__":
    # 1) (A만) 회전+시프트 최적화 + 산출(.jpg)
    main()

    # 2) Test 세트로 복제 (항상 selection_tgt 이미지/마스크, 폴더명에 고유 ID 부여)
    DO_EXPORT_TEST = True
    if DO_EXPORT_TEST:
        export_endolfin_test(
            src_root = Path("./rotation_optuna_results"),
            dst_root = Path("./dataset/ENDOLFIN_new/Test"),
            seq_spec = "0:5:70",
            zpad = 5
        )
