#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_color_data.py

功能：
1. 分别读取两个 JSON 文件：
   - 全脸色彩分析结果（键如 "1_alpha.png": [...]）
   - 分部位色彩分析结果（键如 "1": {...}）
2. 将两个 JSON 中对应索引的数据合并：
   - 把全脸结果重命名为 "full" 并插入到对应分部位字典里
3. 将合并后的结果写入新的 JSON 文件

依赖库：
    标准库，无需额外安装
"""

import json
from pathlib import Path
from tqdm import tqdm

# ----------------- 配置区 -----------------
# 全脸分析结果 JSON 路径（键为 "<index>_alpha.png"）
FULL_JSON_PATH = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_full_color_analysis.json")
# 分部位分析结果 JSON 路径（键为 "<index>"）
SEGMENT_JSON_PATH = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_segment_color_analysis.json")
# 合并后输出的 JSON 路径
OUTPUT_JSON_PATH = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_merged_color_analysis.json")
# -------------------------------------------


def load_json(path: Path) -> dict:
    """
    从指定路径加载 JSON 文件并返回 Python 字典。
    如果文件不存在或无法解析，会抛出异常。
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: Path) -> None:
    """
    将 Python 字典保存为 JSON 文件，保证中文不被转义并美化缩进。
    """
    # 排序
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
    sorted_dict = {k: v for k, v in sorted_items}

    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, ensure_ascii=False, indent=2)


def merge_full_into_segment(full_data: dict, seg_data: dict) -> dict:
    """
    将全脸(full_data)和分部位(seg_data)数据合并：
    - full_data 的键形如 "1_alpha.png"，值为列表
    - seg_data 的键形如 "1"，值为 dict
    合并策略：
      对每个 seg_data 的索引 key:
        1. 在 full_data 中找 key + "_alpha.png"
        2. 如果存在，将它的列表重命名为 "full" 并插入到 seg_data[key] 中
    返回合并后的新字典。
    """
    merged = {}
    for idx, parts_dict in tqdm(seg_data.items(), desc="Merge Operation in progress", unit="dict"):
        # 先复制已有的分部位结果
        merged[idx] = parts_dict.copy()
        # 构造对应的全脸文件名
        full_key = f"{idx}_alpha.png"
        if full_key in full_data:
            # 插入新的 "Full" 键
            merged[idx]["full"] = full_data[full_key]
        else:
            # 如果没有对应全脸数据，可选：插入空列表或打印警告
            merged[idx]["full"] = []
            print(f"⚠️ 警告：未在全脸数据中找到键 {full_key}，已插入空列表。")
        # check
        if check_keys(data=merged[idx]):
            pass
        else:
            print(f"Error! Target {idx} has wrong keys: {merged[idx].keys()} !")
    return merged


def check_keys(data: dict) -> bool:
    """
    检查字典是否包含五个部位和全脸的数据，一共六个键：['eyes', 'nose', 'chin', 'cheeks', 'forehead', 'full']
    :param data: 合并后的字典
    :return: True/false
    """
    target_keys = ['eyes', 'nose', 'chin', 'cheeks', 'forehead', 'full']
    data_keys = data.keys()

    return set(target_keys) == set(data_keys)


def main():
    # 1. 加载两个 JSON 文件
    print(f"加载全脸 JSON：{FULL_JSON_PATH}")
    full_data = load_json(FULL_JSON_PATH)
    print(f"加载分部位 JSON：{SEGMENT_JSON_PATH}")
    seg_data = load_json(SEGMENT_JSON_PATH)

    # 2. 合并
    print("正在合并数据…")
    merged_data = merge_full_into_segment(full_data, seg_data)

    # 3. 写入新的 JSON
    print(f"保存合并结果到：{OUTPUT_JSON_PATH}")
    save_json(merged_data, OUTPUT_JSON_PATH)
    print("✅ 合并完成！")


if __name__ == "__main__":
    main()
