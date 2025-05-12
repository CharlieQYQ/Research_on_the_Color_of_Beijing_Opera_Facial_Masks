#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mixed_clustering_kprototypes.py

功能：
- 读取合并后的 JSON，每个样本包含 metadata + 多个部位的颜色列表（full, forehead, eyes, nose, cheeks, chin）
- 支持两种聚类模式，通过修改 main() 函数中的 MODE 和 PART_NAME 变量：
    * MODE = "full" ：对整脸聚类，使用 JSON 中的 "full" 键
    * MODE = "part" ：对单个部位聚类，使用 PART_NAME 指定的键（如 "eyes"）
- 提取数值特征：前 4 大颜色的 RGB + 百分比，共 4×4 = 16 维
- 提取类别特征：可在 CATEGORICAL_FIELDS 中列任意多或少的字段
- 使用 K-Prototypes 对混合数据聚类
- 输出带 cluster 标签的 JSON，并绘制聚类分布柱状图

依赖：
    pip install numpy pandas matplotlib kmodes scikit-learn tqdm

用法：
    直接修改 main() 中的 MODE 和 PART_NAME，然后运行：
        python mixed_clustering_kprototypes.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------ 配置区域 ------------------
INPUT_JSON  = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_all_in_one.json")  # 输入 JSON 路径
OUTPUT_JSON = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/test-run_cluster_res.json")     # 输出 JSON 路径
N_CLUSTERS  = 5                                  # 聚类簇数
TOP_COLORS  = 6                                  # 用前几个颜色作为特征

# 如果研究某个标签字段，只在此列表中写一个即可
CATEGORICAL_FIELDS = ["role_type", "role_category", "makeup_pattern"]
# CATEGORICAL_FIELDS = ["role_type", "role_category", "source_repertoire",
#                       "repertoire_reference", "makeup_pattern", "painting_technique"]
# 可选的聚类模式： "full" 或 "part"
MODE = "full"
# 如果 MODE="part"，在下面指定部位名称：forehead, eyes, nose, cheeks, chin
PART_NAME = "eyes"
# ----------------------------------------------


def load_data(json_path):
    """
    加载 JSON 并转换为 DataFrame：
    - JSON 顶级键（索引）成为 DataFrame 的一列 'json_key'
    - 所有列名统一转为小写，方便后续一律用小写查找
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # 从字典创建 DataFrame，orient='index' 把 key 放进 index
    df = pd.DataFrame.from_dict(raw, orient='index')
    df.index.name = "json_key"
    df = df.reset_index()

    # 把所有列名转为小写
    df.columns = [col.lower() for col in df.columns]
    return df


def extract_color_features(color_list, top_k=TOP_COLORS):
    """
    将颜色列表转为定长向量：
    [r1,g1,b1,p1, r2,g2,b2,p2, …] 共 top_k*4 维
    """
    feat = []
    for i in range(top_k):
        if i < len(color_list):
            c = color_list[i]
            feat.extend(c['rgb'])
            feat.append(c['percentage'])
        else:
            feat.extend([0,0,0,0.0])
    return feat


def prepare_features(df: pd.DataFrame):
    """
    构造混合特征矩阵 X 及类别列索引 cat_idx。
    核心改动：full/part 都是 df["color"] 里的子键，不再当顶级列取。
    """
    # 先决定取哪个子键
    key = MODE if MODE == "full" else PART_NAME.lower()
    # 检查每行 color dict 里有没有这个 key
    missing = df["color"].apply(lambda c: key not in c)
    if missing.any():
        raise ValueError(f"部分记录的 color 中缺少 '{key}'，请检查 JSON。")

    # 因为full并不是一个顶级列，而是color列中保存的dict的一个key，所以不能直接从列中找full等颜色信息
    # 用 apply 从 color dict 中提取列表，再转向量
    num_feats = df["color"].apply(lambda c: extract_color_features(c[key]))
    X_num = np.vstack(num_feats.values).astype(float)

    # 类别特征
    for fld in CATEGORICAL_FIELDS:
        if fld.lower() not in df.columns:
            raise ValueError(f"类别字段 '{fld}' 不在 DataFrame 列中。")
    X_cat = df[[f.lower() for f in CATEGORICAL_FIELDS]].astype(str)

    # 合并数值 + 类别
    X_all = pd.concat([pd.DataFrame(X_num, index=df.index), X_cat], axis=1)
    cat_idx = [X_all.columns.get_loc(col.lower()) for col in CATEGORICAL_FIELDS]
    return X_all.values, cat_idx


def run_kprototypes(X, cat_idx):
    """
    对混合特征 X 运行 K-Prototypes 聚类并返回标签。
    """
    kp = KPrototypes(n_clusters=N_CLUSTERS, init='Cao', random_state=0)
    labels = kp.fit_predict(X, categorical=cat_idx)
    return labels


def append_and_save(df, labels):
    """
    将聚类标签写回 DataFrame 并保存至 JSON：
    - 以 json_key 作为最外层字典的 key
    - 保留原始所有字段 + 新增 'cluster'
    """
    df['cluster'] = labels
    out = {}
    for _, row in df.iterrows():
        key = row['json_key']
        entry = row.to_dict()
        entry.pop('json_key')
        out[key] = entry
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ 聚类结果已写入 {OUTPUT_JSON}")


def visualize_clusters(df):
    """
    绘制每个簇的样本数量柱状图。
    """
    counts = df['cluster'].value_counts().sort_index()
    plt.figure(figsize=(6,4))
    counts.plot(kind='bar')
    plt.title(f"Cluster distribution (mode={MODE}, part={PART_NAME})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    # matplotlib后端会有方法报错
    # plt.show()

    # 保存到文件而不是调用 plt.show()
    VIS_DIR = Path("visualizations")
    VIS_DIR.mkdir(exist_ok=True)
    out_path = VIS_DIR / f"cluster_dist_{MODE}{'' if MODE == 'full' else '_' + PART_NAME}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Cluster distribution 已保存到 {out_path}")


def main():
    # 1. 加载数据
    df = load_data(INPUT_JSON)

    # 2. 准备特征
    X, cat_idx = prepare_features(df)

    # 3. 聚类
    labels = run_kprototypes(X, cat_idx)

    # 4. 保存结果
    append_and_save(df, labels)

    # 5. 可视化
    visualize_clusters(df)


if __name__ == "__main__":
    main()
