import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import matplotlib
matplotlib.use('Agg')

# 설정
versions = ['V3', 'V4', 'V5', 'V6', 'V7']
img_folder = 'ultrasound/datasets_v0.03/P0'
output_path = 'CY_circle_crop_noise_comparison.png'

# donut crop 반지름 (outer, inner)
VERSION_RADIUS = {
    'V3': [(220, 85)],
    'V4': [(130, 50)],
    'V5': [(90, 30)],
    'V6': [(60, 20)],
    'V7': [(45, 15)]
}

# Gaussian noise 표준편차 설정
weak_sigma = 10     # 약한 노이즈
mid_sigma = 20      # 중간 노이즈 (약한보다 조금 더 강함)
strong_sigma = 50   # 강한 노이즈

# donut crop & mask 생성 함수
def donut_crop_with_mask(img_array, center, inner_r, outer_r):
    h, w = img_array.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = ((dist > inner_r) & (dist <= outer_r)).astype(np.uint8)
    cropped = np.zeros_like(img_array)
    cropped[mask == 1] = img_array[mask == 1]
    return cropped, mask

# 플롯 준비: 버전 수 × 6열
fig, axes = plt.subplots(len(versions), 6, figsize=(18, len(versions) * 3))

for i, version in enumerate(versions):
    # filename = f'CN_ON_PC_DC_{version}_001.bmp'
    filename = f'CY_ON_PC_DC_{version}_001.bmp'
    img_path = os.path.join(img_folder, filename)
    arr = np.array(Image.open(img_path).convert('L'))
    h, w = arr.shape
    center = (w // 2, h // 2)

    # donut 반지름 (outer, inner)
    outer_r, inner_r = VERSION_RADIUS.get(version, [(int(min(h, w) * 0.2), int(min(h, w) * 0.1))])[0]

    # donut crop 및 mask
    cropped, mask = donut_crop_with_mask(arr, center, inner_r, outer_r)

    # 약한 노이즈
    weak_noise = np.random.normal(0, weak_sigma, size=arr.shape)
    weak_img = cropped.copy()
    weak_img[mask == 1] = weak_img[mask == 1] + weak_noise[mask == 1]
    weak_img = np.clip(weak_img, 0, 255).astype(np.uint8)

    # 중간 노이즈
    mid_noise = np.random.normal(0, mid_sigma, size=arr.shape)
    mid_img = cropped.copy()
    mid_img[mask == 1] = mid_img[mask == 1] + mid_noise[mask == 1]
    mid_img = np.clip(mid_img, 0, 255).astype(np.uint8)

    # 강한 노이즈
    strong_noise = np.random.normal(0, strong_sigma, size=arr.shape)
    strong_img = np.zeros_like(arr)
    strong_img[mask == 1] = strong_noise[mask == 1]
    strong_img = np.clip(strong_img, 0, 255).astype(np.uint8)

    # 1열: 원본
    ax = axes[i, 0]
    ax.imshow(arr, cmap='gray')
    ax.set_title(f'{version} Original')
    ax.axis('off')

    # 2열: Donut Crop
    ax = axes[i, 1]
    ax.imshow(cropped, cmap='gray')
    ax.set_title('Donut Cropped')
    ax.axis('off')

    # 3열: Mask
    ax = axes[i, 2]
    ax.imshow(mask, cmap='gray')
    ax.set_title('Mask')
    ax.axis('off')

    # 4열: 약한 가우시안 노이즈
    ax = axes[i, 3]
    ax.imshow(weak_img, cmap='gray')
    ax.set_title('Weak Noise')
    ax.axis('off')

    # 5열: 중간 가우시안 노이즈
    ax = axes[i, 4]
    ax.imshow(mid_img, cmap='gray')
    ax.set_title('Medium Noise')
    ax.axis('off')

    # 6열: 강한 가우시안 노이즈
    ax = axes[i, 5]
    ax.imshow(strong_img, cmap='gray')
    ax.set_title('Strong Noise')
    ax.axis('off')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'Saved comparison image to {output_path}')
