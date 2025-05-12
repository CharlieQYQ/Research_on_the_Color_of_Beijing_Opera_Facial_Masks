#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
remove_bg_with_device_check.py

功能：
1. 批量读取目录中 `001.png` 至 `640.png`（任意三位数字 + .png），带浅黄色背景的脸谱图像
2. 使用 rembg 库智能去除背景，并检测是否使用 GPU 加速
3. 保存为 `<原名>_alpha.png`

依赖：
    pip install rembg[gpu] Pillow tqdm torch torchvision

用法：
    修改 INPUT_DIR 为你图片所在目录，运行：
        python remove_bg_with_device_check.py
"""

import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rembg import remove  # 导入 rembg 的核心函数
import torch


# ————— 用户配置区 —————
# 输入 / 输出目录
INPUT_DIR = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/project_for_Dataset_2/test_img")  # ← 修改为实际路径
OUTPUT_DIR = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/project_for_Dataset_2/test_img_alpha")  # 输出目录
OUTPUT_DIR.mkdir(exist_ok=True)

# 匹配三位数字命名的 PNG
FILE_PATTERN = "[0-9][0-9][0-9].png"
# ————————————————————————


def check_device():
    """
    检查当前 PyTorch 使用的设备（CPU 或 GPU）。
    """
    if torch.backends.mps.is_available():  # 检查 M1 芯片的 MPS 支持
        device = "MPS (GPU)"
    elif torch.cuda.is_available():  # 检查 CUDA 支持
        device = f"CUDA (GPU) - {torch.cuda.get_device_name(0)}"
    else:
        device = "CPU"
    return device


def process_image_with_rembg(src_path: Path, out_path: Path):
    """
    使用 rembg 去除图片背景，并保存结果。
    :param src_path: 输入图片路径
    :param out_path: 输出图片路径
    """
    try:
        # 读取图片
        input_image = Image.open(src_path)
        # 使用 rembg 去除背景
        output_image = remove(input_image)
        # 保存结果
        output_image.save(out_path)
    except Exception as e:
        print(f"⚠️ 处理 {src_path.name} 时出错：{e}")


def main():
    # 检测当前设备
    device = check_device()
    print(f"🚀 当前计算设备：{device}")

    # 获取所有匹配的文件
    files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    if not files:
        print(f"❌ 未找到匹配 {FILE_PATTERN} 的文件，请检查 INPUT_DIR")
        return

    for path in tqdm(files, desc="Removing background with rembg"):
        # 构造输出路径
        out_path = OUTPUT_DIR / f"{path.stem}_alpha.png"
        # 处理单张图片
        process_image_with_rembg(path, out_path)

    print(f"\n✅ 全部处理完毕，透明图保存在：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()