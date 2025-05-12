"""
segment_and_crop.py

功能：
1. 读取输入目录中所有以 “*_alpha.png” 命名的带透明背景的脸谱图像（尺寸不一致）。
2. 按照给定的百分比，将每张图裁剪成 5 个部位：
   - forehead：H[  0.0%–35.0%], W[  0.0%–100.0%]
   - eyes    ：H[ 35.0%–52.8%], W[  0.0%– 40.0%] & W[ 60.0%–100.0%]（合并到一张图）
   - nose    ：H[ 35.0%–71.7%], W[ 35.0%– 65.0%]
   - cheeks  ：H[ 52.8%–71.7%], W[  0.0%– 40.0%] & W[ 60.0%–100.0%]（合并到一张图）
   - chin    ：H[ 71.7%–100.0%], W[ 25.0%– 75.0%]
3. 将裁剪出的 5 张图分别保存到输出目录，命名为 “<原文件名去掉 _alpha>_<部位>_alpha.png”，保留透明通道。

依赖库：
    pip install pillow tqdm

用法：
    修改 INPUT_DIR、OUTPUT_DIR，然后执行：
        python segment_and_crop.py
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ————————————— 用户配置 —————————————
INPUT_DIR  = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_alpha")  # 输入目录
OUTPUT_DIR = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_segment_alpha")  # 输出目录
# —————————————————————————————————————————————


def ensure_output_dir():
    """确保输出目录存在。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_parts(img: Image.Image):
    """
    根据给定百分比裁剪图像的五个部位，返回 dict{name: Image}：
      - forehead, eyes, nose, cheeks, chin
    """
    W, H = img.size

    # 1) 前额 (forehead)
    fh_y0 = int(0.00 * H)
    fh_y1 = int(0.35 * H)
    fh_x0, fh_x1 = 0, W

    # 2) 眼睛 (eyes)，两个水平片段
    ey_y0 = int(0.35 * H)
    ey_y1 = int(0.528 * H)
    # 左眼水平范围
    ey_lx0 = 0
    ey_lx1 = int(0.40 * W)
    # 右眼水平范围
    ey_rx0 = int(0.60 * W)
    ey_rx1 = W

    # 3) 鼻子 (nose)
    no_y0 = int(0.35 * H)
    no_y1 = int(0.717 * H)
    no_x0 = int(0.35 * W)
    no_x1 = int(0.65 * W)

    # 4) 脸颊 (cheeks)，两个水平片段
    ch_y0 = int(0.528 * H)
    ch_y1 = int(0.717 * H)
    ch_lx0 = 0
    ch_lx1 = int(0.40 * W)
    ch_rx0 = int(0.60 * W)
    ch_rx1 = W

    # 5) 下巴 (chin)
    cb_y0 = int(0.717 * H)
    cb_y1 = H
    cb_x0 = int(0.25 * W)
    cb_x1 = int(0.75 * W)

    parts = {}

    # 裁剪前额
    parts["forehead"] = img.crop((fh_x0, fh_y0, fh_x1, fh_y1))

    # 裁剪眼睛：合并左右两块到一张透明图
    ey_h = ey_y1 - ey_y0
    eyes_img = Image.new("RGBA", (W, ey_h), (0,0,0,0))
    # 左眼
    left_eye = img.crop((ey_lx0, ey_y0, ey_lx1, ey_y1))
    eyes_img.paste(left_eye, (ey_lx0, 0), left_eye)
    # 右眼
    right_eye = img.crop((ey_rx0, ey_y0, ey_rx1, ey_y1))
    eyes_img.paste(right_eye, (ey_rx0, 0), right_eye)
    parts["eyes"] = eyes_img

    # 裁剪鼻子
    parts["nose"] = img.crop((no_x0, no_y0, no_x1, no_y1))

    # 裁剪脸颊：左右合并
    cheeks_img = Image.new("RGBA", (W, ch_y1 - ch_y0), (0,0,0,0))
    left_cheek  = img.crop((ch_lx0, ch_y0, ch_lx1, ch_y1))
    cheeks_img.paste(left_cheek, (ch_lx0, 0), left_cheek)
    right_cheek = img.crop((ch_rx0, ch_y0, ch_rx1, ch_y1))
    cheeks_img.paste(right_cheek, (ch_rx0, 0), right_cheek)
    parts["cheeks"] = cheeks_img

    # 裁剪下巴
    parts["chin"] = img.crop((cb_x0, cb_y0, cb_x1, cb_y1))

    return parts


def main():
    ensure_output_dir()
    # 遍历所有 "*_alpha.png"
    for img_path in tqdm(sorted(INPUT_DIR.glob("*_alpha.png")), desc="Processing", unit="img"):
        # 载入并转换为 RGBA
        img = Image.open(img_path).convert("RGBA")
        # 获取裁剪结果
        parts = crop_parts(img)

        # 构造文件名前缀：去掉 "_alpha"
        stem = img_path.stem
        if stem.endswith("_alpha"):
            stem = stem[:-6]

        # 存储各部位图像
        for name, part_img in parts.items():
            out_name = f"{stem}_{name}_alpha.png"
            part_img.save(OUTPUT_DIR / out_name, format="PNG")

    print(f"✅ 所有部位裁剪完成，保存在：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
