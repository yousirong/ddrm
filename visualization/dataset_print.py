import os
import itertools
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    cylinder = 'yes' if parts[0] == 'CY' else 'no'
    object_present = 'yes' if parts[1] == 'OY' else 'no'
    position = 'center' if parts[2] == 'PC' else 'tilted'
    angle_code = parts[3]
    if angle_code == 'DXXX':
        angle = 'none'
    elif angle_code.startswith('D'):
        angle = angle_code[1:] + ' deg'
    else:
        angle = angle_code
    view_depth = int(parts[4][1:])
    probe_serial = parts[5]
    probe = probe_serial[0]
    serial = probe_serial[1:]
    serial_tail = serial  # <- XX 부분만 유지 (N은 무시)

    return {
        'filename': filename,
        'cylinder': cylinder,
        'object': object_present,
        'position': position,
        'angle': angle,
        'view_depth': view_depth,
        'probe': probe,
        'serial': serial,
        'serial_tail': serial_tail  # ✅ 추가
    }


def compare_attributes(a1, a2):
    diffs = {}
    for key in ['cylinder', 'object', 'position', 'angle', 'probe', 'serial']:
        if a1[key] != a2[key]:
            diffs[key] = (a1[key], a2[key])
    return diffs


def create_comparison_image(img_left, img_right, left_name, right_name, diffs):
    w, h = img_left.width, img_left.height
    header = f"Comparison: {left_name} (CY) vs {right_name} (CN)"
    lines = [header] + [f"{k}: {v1} vs {v2}" for k, (v1, v2) in diffs.items()]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), "Hg", font=font)
    text_height = bbox[3] - bbox[1]
    line_height = text_height + 4
    total_text_height = line_height * len(lines) + 10
    canvas = Image.new("RGB", (w * 2, h + total_text_height), (255, 255, 255))
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (w, 0))
    draw = ImageDraw.Draw(canvas)
    y = h + 5
    x = 10
    for line in lines:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_height
    return canvas


def process_directory(base_dir):
    out_dir = os.path.join(base_dir, 'comparisons')
    os.makedirs(out_dir, exist_ok=True)

    bmp_files = [f for f in os.listdir(base_dir) if f.lower().endswith('.bmp')]
    attrs = {f: parse_filename(f) for f in bmp_files}

    view_groups = defaultdict(list)
    for f, a in attrs.items():
        view_groups[a['view_depth']].append(f)

    comparison_log = []

    for depth, files in view_groups.items():
        if len(files) < 2:
            continue
        for f1, f2 in itertools.combinations(files, 2):
            a1, a2 = attrs[f1], attrs[f2]

            # ✅ 조건들
            if a1['angle'] != a2['angle']:
                continue
            if a1['serial_tail'] != a2['serial_tail']:  # <- serial_tail 비교
                continue
            if a1['object'] != a2['object']:
                continue
            if a1['cylinder'] == a2['cylinder']:
                continue

            # CY 왼쪽, CN 오른쪽
            if a1['cylinder'] == 'yes':
                lf, rf, la, ra = f1, f2, a1, a2
            else:
                lf, rf, la, ra = f2, f1, a2, a1

            diffs = compare_attributes(la, ra)
            if not diffs:
                continue

            img_l = Image.open(os.path.join(base_dir, lf)).convert('RGB')
            img_r = Image.open(os.path.join(base_dir, rf)).convert('RGB')
            result_img = create_comparison_image(img_l, img_r, lf, rf, diffs)

            out_name = f"{os.path.splitext(lf)[0]}_vs_{os.path.splitext(rf)[0]}.jpg"
            result_img.save(os.path.join(out_dir, out_name), 'JPEG')
            print(f"Saved comparison image: {out_name}")

            # ✅ CSV 로그 기록
            comparison_log.append({
                'left_image': lf,
                'right_image': rf,
                'angle': la['angle'],
                'serial_tail': la['serial_tail'],
                'view_depth': la['view_depth'],
                'object': la['object'],
                'diff_keys': ', '.join(diffs.keys())
            })

    # ✅ CSV 저장
    csv_path = os.path.join(out_dir, 'comparison_pairs.csv')
    pd.DataFrame(comparison_log).to_csv(csv_path, index=False)
    print(f"✅ Saved CSV log: {csv_path}")


if __name__ == '__main__':
    # 두 디렉터리 각각 처리
    for p_dir in ['P0', 'P2']:
        base_path = f'/home/ubuntu/Desktop/JY/ddrm/ultrasound/datasets_v0.03/{p_dir}'
        print(f"\n📁 Processing: {base_path}")
        process_directory(base_path)
