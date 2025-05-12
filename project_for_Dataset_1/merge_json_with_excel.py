#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_excel_json.py

功能：
1. 读取一个包含表头的 Excel 文件和一个 JSON 文件：
   - Excel 中必须有一列名为 “index”，其值为数字（或数字字符串）
   - JSON 文件顶层键同样为这些 “index” 值（字符串形式）
2. 将二者按 “index” 对应合并：
   - Excel 的每一行：将表头列名→该单元格的值（空单元格用 Python None）
   - JSON 中对应 “index” 的字典，作为新键 “color” 的值
   - 最终形成形如 { "index": { <excel 字段>: <value>, ..., "color": { … } } }
3. 将所有合并后的记录写入指定的输出 JSON 文件，格式美观（indent=2）。

依赖库：
    pip install pandas openpyxl

用法：
    修改下列配置：EXCEL_PATH、INPUT_JSON_PATH、OUTPUT_JSON_PATH
    然后执行：
        python merge_excel_json.py
"""

import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ----------------- 配置区 -----------------
# Excel 文件路径
EXCEL_PATH       = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_info_selected.xlsx")
# 原始 JSON 文件路径（已合并部位+全脸色彩分析结果）
INPUT_JSON_PATH  = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_merged_color_analysis.json")
# 合并后输出 JSON 路径
OUTPUT_JSON_PATH = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_all_in_one.json")
# -------------------------------------------


def load_excel(path: Path) -> pd.DataFrame:
    """
    读取 Excel 文件为 DataFrame。
    - 使用 openpyxl 引擎
    - 保证 index 列以字符串形式读取，方便后续与 JSON 键匹配
    """
    df = pd.read_excel(path, engine="openpyxl", dtype={"index": str})
    return df


def normalize_excel_row(row: pd.Series) -> dict:
    """
    将 DataFrame 的一行转换为普通 dict：
    - 将所有列名映射到它们对应的值
    - 对于 pandas 的 NaN 或空字符串，统一转为 Python None
    """
    result = {}
    for col, val in row.items():
        # 如果是空值（NaN）或空字符串，赋值 None
        if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
            result[col] = None
        else:
            # 保留原始类型，或强制转换为 str
            result[col] = str(val) if not isinstance(val, (int, float, bool)) else val
    return result


def load_json(path: Path) -> dict:
    """从指定路径加载 JSON 数据为 Python 字典。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """
    将字典以漂亮的格式写入 JSON 文件：
    - ensure_ascii=False 保留中文
    - indent=2 缩进
    """
    # 排序
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
    sorted_dict = {k: v for k, v in sorted_items}

    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, ensure_ascii=False, indent=2)


def merge_data(df: pd.DataFrame, color_data: dict) -> dict:
    """
    按 index 合并 Excel 行和 JSON 中的 color_data：
    返回合并后的字典，结构：
    {
      "<index>": {
         <excel columns>: <value>,
         ...,
         "color": { … }        # 对应 color_data[index]
      },
      ...
    }
    """
    merged = {}
    # 遍历每一行
    for _, row in tqdm(df.iterrows(), desc="Processing", unit="item"):
        idx = row["index"]  # 必须存在这一列
        # 将行转换为字段字典
        info_dict = normalize_excel_row(row)
        # 从 JSON 中取出对应的 color 子字典，若不存在则赋空 dict
        color_dict = color_data.get(idx, {})
        # 插入到 info_dict
        info_dict["color"] = color_dict
        # 存入结果
        merged[idx] = info_dict
    return merged


def main():
    # 1. 读取 Excel
    print(f"读取 Excel：{EXCEL_PATH}")
    df = load_excel(EXCEL_PATH)

    # 2. 读取 JSON（色彩分析结果）
    print(f"读取 JSON：{INPUT_JSON_PATH}")
    color_data = load_json(INPUT_JSON_PATH)

    # 3. 合并
    print("合并数据中…")
    merged = merge_data(df, color_data)

    # 4. 保存到新 JSON
    print(f"保存合并结果到：{OUTPUT_JSON_PATH}")
    save_json(merged, OUTPUT_JSON_PATH)

    print("✅ 完成！")


if __name__ == "__main__":
    main()
