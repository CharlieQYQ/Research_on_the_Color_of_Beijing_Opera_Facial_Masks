#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mixed_clustering_kprototypes_with_analysis.py

功能：
- 加载 all-in-one JSON，每行包含 metadata + color 子字典（full, forehead, …）
- 支持 MODE="full"/"part" 切换整脸或单部位聚类
- 提取数值特征：TOP_COLORS 大颜色的 RGB + 百分比
- 提取类别特征：CATEGORICAL_FIELDS 列表
- 使用 K-Prototypes 聚类
- 输出带 cluster 标签的 JSON
- 新增聚类质量与簇解释分析：
    1) 样本量 & Silhouette Score
    2) 簇中心颜色条形图
    3) 类别属性分布
    4) 合并展示示例提示
依赖：
    pip install numpy pandas matplotlib kmodes scikit-learn tqdm
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# ------------------ 全局配置 ------------------
INPUT_JSON  = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_all_in_one.json")
OUTPUT_JSON = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/test-run_cluster_res.json")
N_CLUSTERS  = 3
TOP_COLORS  = 6

# 想研究的类别标签列表（可只写一个）
CATEGORICAL_FIELDS = ["role_type"]
# CATEGORICAL_FIELDS = ["role_type", "role_category", "source_repertoire",
#                       "repertoire_reference", "makeup_pattern", "painting_technique"]
MODE      = "full"    # "full" 或 "part"
PART_NAME = "eyes"    # 当 MODE="part" 时使用
# ---------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    """加载 JSON 并展开为 DataFrame，每行一条记录。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index.name = "json_key"
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def extract_color_features(color_list, top_k=TOP_COLORS):
    """
    将 color_list 转成定长向量 [r,g,b,p,...] 共 top_k*4 维。
    """
    feat = []
    for i in range(top_k):
        if i < len(color_list):
            c = color_list[i]
            feat.extend(c["rgb"])
            feat.append(c["percentage"])
        else:
            feat.extend([0,0,0,0.0])
    return feat


def prepare_features(df: pd.DataFrame):
    """
    从 df["color"] 中取 full 或 part 列，构造混合特征：
    - X_num: 数值特征矩阵
    - cat_idx: 类别列索引
    """
    key = MODE if MODE=="full" else PART_NAME.lower()
    # 检查每行 color dict 包含该 key
    if df["color"].apply(lambda d: key not in d).any():
        raise ValueError(f"部分记录的 color 中缺少 '{key}' 键")

    # ===== 新增 =====: 提取数值特征
    num_feats = df["color"].apply(lambda d: extract_color_features(d[key]))
    X_num = np.vstack(num_feats.values).astype(float)

    # 类别特征检查
    for fld in CATEGORICAL_FIELDS:
        if fld.lower() not in df.columns:
            raise ValueError(f"类别字段 '{fld}' 不存在")
    X_cat = df[[f.lower() for f in CATEGORICAL_FIELDS]].astype(str)

    # 合并
    X_all = pd.concat([pd.DataFrame(X_num, index=df.index), X_cat], axis=1)
    cat_idx = [X_all.columns.get_loc(col.lower()) for col in CATEGORICAL_FIELDS]
    return X_all.values, X_num, cat_idx  # 返回 X_num 以便后续 Silhouette


def run_kprototypes(X, cat_idx):
    kp = KPrototypes(n_clusters=N_CLUSTERS, init="Cao", random_state=0)
    return kp.fit_predict(X, categorical=cat_idx)


def save_results(df: pd.DataFrame, labels):
    df["cluster"] = labels
    out = {}
    for _, row in df.iterrows():
        key = row["json_key"]
        rec = row.to_dict()
        rec.pop("json_key")
        out[key] = rec
    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 聚类结果已写入 {OUTPUT_JSON}")


def visualize_clusters(df: pd.DataFrame):
    """绘制每个簇样本数柱状图并保存。"""
    counts = df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(6,4))
    counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Cluster Distribution ({MODE}{'' if MODE=='full' else ':'+PART_NAME})")
    plt.xlabel("Cluster"); plt.ylabel("Count")
    plt.tight_layout()
    path = Path("cluster_distribution.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")

# ===== 新增分析函数 =====


def analyze_cluster_quality(X_num, labels):
    """
    1) 打印每个簇样本量
    2) 计算并打印 Silhouette Score
    """
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Cluster Sample Counts ===")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples")
    if len(unique) > 1:
        score = silhouette_score(X_num, labels)
        print(f"Silhouette Score: {score:.4f}")
    else:
        print("Silhouette Score: N/A (only one cluster)")


def visualize_cluster_centroids(X_num, labels):
    """
    根据数值特征 X_num 计算每簇中心，并画出颜色条形图，展示主色与比例。
    """
    top_k = TOP_COLORS
    clusters = np.unique(labels)
    centroids = [X_num[labels==u].mean(axis=0) for u in clusters]

    plt.figure(figsize=(8, len(clusters)*0.8))
    for i, (u, vec) in enumerate(zip(clusters, centroids)):
        offset = 0
        for j in range(top_k):
            r,g,b,p = vec[j*4:(j+1)*4]
            plt.barh(i, p, left=offset, color=(r/255,g/255,b/255), edgecolor="black")
            offset += p
        plt.text(1.02, i, f"Cluster {u}", va="center")
    plt.xlabel("Proportion"); plt.yticks([])
    plt.title("Cluster Centroid Colors")
    plt.tight_layout()
    path = Path("cluster_centroids.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def print_category_distributions(df: pd.DataFrame):
    """
    打印每个簇在各类别字段上的占比分布。
    """
    print("\n=== Category Distributions per Cluster ===")
    for u in sorted(df["cluster"].unique()):
        sub = df[df["cluster"]==u]
        print(f"\n-- Cluster {u} --")
        for cat in CATEGORICAL_FIELDS:
            counts = sub[cat].value_counts(normalize=True)
            items = ", ".join([f"{idx}:{val:.0%}" for idx,val in counts.items()])
            print(f"  {cat}: {items}")

# =========================================


def main():
    # 1. 加载
    df = load_data(INPUT_JSON)
    # 2. 特征
    X_all, X_num, cat_idx = prepare_features(df)
    # 3. 聚类
    labels = run_kprototypes(X_all, cat_idx)
    # 4. 保存
    save_results(df, labels)
    # 5. 分析与可视化
    analyze_cluster_quality(X_num, labels)       # (1) 样本量 & silhouette
    visualize_cluster_centroids(X_num, labels)   # (2) 簇中心颜色
    print_category_distributions(df)             # (3) 类别分布
    visualize_clusters(df)                       # (4) 分布柱状图提示


if __name__ == "__main__":
    main()
