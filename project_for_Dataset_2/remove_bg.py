#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
remove_bg_yellow.py

功能：
1. 批量读取目录中 `001.png` 至 `640.png`（任意三位数字 + .png），带浅黄色背景的脸谱图像
2. 自动采样图片四角的背景色（浅黄色），构建色差阈值内的“近似背景”掩码
3. 使用 flood-fill，从四角填充连通背景，避免抠到主体内部同色区域
4. 合成新的 Alpha 通道（背景透明），保存为 `<原名>_alpha.png`

依赖：
    pip install opencv-python numpy tqdm Pillow

用法：
    修改 INPUT_DIR 为你图片所在目录，运行：
        python remove_bg_yellow.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ————— 用户配置区 —————
# 输入 / 输出目录
INPUT_DIR  = Path("/path/to/your/new_dataset")  # ← 修改为实际路径
OUTPUT_DIR = INPUT_DIR / "alpha_png"            # 输出目录
OUTPUT_DIR.mkdir(exist_ok=True)

# 色差阈值：针对浅黄色背景，建议略大一些 (0–255)
COLOR_THRESH = 40

# 左上角采样面积占比，越小采样越集中（采样四角）
SAMPLE_FRAC = 0.002  # 0.2% 面积

# 匹配三位数字命名的 PNG
FILE_PATTERN = "[0-9][0-9][0-9].png"
# ————————————————————————

def sample_bg_color(img: np.ndarray, frac: float) -> np.ndarray:
    """
    从图像四个角落各采样小区域，取所有像素的均值作为背景色。
    相比只取左上，四角更能兼容渐变或不均匀的浅色背景。
    """
    h, w = img.shape[:2]
    side = max(1, int(min(h, w) * np.sqrt(frac)))  # 采样正方形边长
    # 4 个角的 ROI
    rois = [
        img[0:side, 0:side],           # 左上
        img[0:side, w-side:w],         # 右上
        img[h-side:h, 0:side],         # 左下
        img[h-side:h, w-side:w],       # 右下
    ]
    # 拼接所有 ROI 的像素，计算均值
    pixels = np.vstack([roi.reshape(-1,3) for roi in rois])
    bg_color = pixels.mean(axis=0)
    return bg_color

def create_bg_mask(img: np.ndarray, bg_color: np.ndarray, thresh: float) -> np.ndarray:
    """
    1) 计算每像素与 bg_color 的色差，阈值化为“可能背景”二值图
    2) 对这张二值图做 flood-fill，从 4 个角开始，保留外部连通部分
    3) 返回背景掩码：255=背景，0=前景
    """
    h, w = img.shape[:2]
    # 计算每像素与背景色的 Euclidean 距离
    diff = np.linalg.norm(img.astype(np.float32) - bg_color, axis=2)
    similar = (diff <= thresh).astype(np.uint8) * 255  # 255 = 可能背景

    # flood-fill 需要额外边框
    mask = similar.copy()
    ff_mask = np.zeros((h+2, w+2), np.uint8)

    # 从四个角开始填充：把连通的 255 区域标记为 1
    for y, x in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
        if mask[y, x] == 255:
            cv2.floodFill(mask, ff_mask, (x,y), 1)

    # mask==1 的区域即外部背景
    bg_mask = (mask == 1).astype(np.uint8) * 255
    return bg_mask

def add_alpha_and_save(src_path: Path, bg_mask: np.ndarray):
    """
    从 src_path 读取原图（含 RGB + 原 Alpha），
    丢弃原有 Alpha，合成新 Alpha（背景透明），保存 PNG。
    """
    # 读取 RGBA
    pil = Image.open(src_path).convert("RGBA")
    arr = np.array(pil)
    # 丢掉旧 alpha，只用 RGB
    rgb = arr[..., :3]

    # 新 alpha：背景区域 alpha=0，前景 alpha=255
    alpha = cv2.bitwise_not(bg_mask)  # 255->0, 0->255
    # 合并 BGRA
    b, g, r = cv2.split(rgb[..., ::-1])  # PIL 是 RGB，OpenCV 是 BGR
    bgra = cv2.merge([b, g, r, alpha])

    # 保存
    out_path = OUTPUT_DIR / f"{src_path.stem}_alpha.png"
    cv2.imwrite(str(out_path), bgra)

def main():
    files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    if not files:
        print(f"❌ 未找到匹配 {FILE_PATTERN} 的文件，请检查 INPUT_DIR")
        return

    for path in tqdm(files, desc="Removing background"):
        # 1) 读取 BGR
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ 无法读取 {path.name}，跳过")
            continue

        # 2) 采样背景色
        bg_col = sample_bg_color(img, SAMPLE_FRAC)

        # 3) 生成背景掩码
        bg_mask = create_bg_mask(img, bg_col, COLOR_THRESH)

        # 4) 合成透明背景 PNG
        add_alpha_and_save(path, bg_mask)

    print(f"\n✅ 全部处理完毕，透明图保存在：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
