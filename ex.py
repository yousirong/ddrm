import os
from PIL import Image
from glob import glob

# 입력 베이스 디렉토리
base_dir = "/home/ubuntu/Desktop/JY/ddrm/exp/image_samples"
output_dir = "/home/ubuntu/Desktop/JY/ddrm/exp/combine"
os.makedirs(output_dir, exist_ok=True)

# 폴더 이름들: imagenet_inp0.0 ~ imagenet_inp0.9
folder_names = [f"imagenet_inp{round(i / 10, 1):.1f}" for i in range(10)]

# 기준 폴더에서 인덱스 추출
ref_folder = os.path.join(base_dir, folder_names[0])
orig_files = sorted(glob(os.path.join(ref_folder, "orig_*.png")))
indices = [os.path.basename(f).split("_")[1].split(".")[0] for f in orig_files]

for idx in indices:
    orig_row = []
    y0_row = []
    out_row = []

    for folder in folder_names:
        folder_path = os.path.join(base_dir, folder)
        orig_path = os.path.join(folder_path, f"orig_{idx}.png")
        y0_path = os.path.join(folder_path, f"y0_{idx}.png")
        out_path = os.path.join(folder_path, f"{idx}_-1.png")

        if not (os.path.exists(orig_path) and os.path.exists(y0_path) and os.path.exists(out_path)):
            print(f"Missing file for index {idx} in folder {folder}")
            continue

        orig = Image.open(orig_path).convert("RGB")
        y0 = Image.open(y0_path).convert("RGB")
        out = Image.open(out_path).convert("RGB")

        # 크기 맞춤
        w, h = orig.size
        y0 = y0.resize((w, h))
        out = out.resize((w, h))

        orig_row.append(orig)
        y0_row.append(y0)
        out_row.append(out)

    if len(orig_row) != len(folder_names):
        print(f"Skipped index {idx} due to incomplete data.")
        continue

    # 가로로 이어붙이기 (각 행 별)
    def concat_images_horizontally(images):
        total_w = sum([img.width for img in images])
        h = images[0].height
        row_img = Image.new("RGB", (total_w, h))
        x_offset = 0
        for img in images:
            row_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return row_img

    row1 = concat_images_horizontally(orig_row)
    row2 = concat_images_horizontally(y0_row)
    row3 = concat_images_horizontally(out_row)

    # 세로로 이어붙이기 (행 기준)
    final_h = row1.height + row2.height + row3.height
    final_w = row1.width
    combined_img = Image.new("RGB", (final_w, final_h))
    combined_img.paste(row1, (0, 0))
    combined_img.paste(row2, (0, row1.height))
    combined_img.paste(row3, (0, row1.height + row2.height))

    combined_img.save(os.path.join(output_dir, f"combined_{idx}.png"))
    print(f"Saved: combined_{idx}.png")
