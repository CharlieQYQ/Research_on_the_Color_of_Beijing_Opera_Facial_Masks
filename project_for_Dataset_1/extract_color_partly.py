"""
grouped_color_analysis.py

功能：
1. 读取指定目录中所有以 “<index>_<part>_alpha.png” 命名的图像，
   其中 part ∈ {forehead, eyes, nose, cheeks, chin}，图像带有透明通道。
2. 对每张“部位”图像：
   a. 提取非透明像素的 RGB，转换到 CIELAB 空间；
   b. 使用 K-Means 聚类合并相似颜色（CIE2000 色差阈值合并）；
   c. 计算各颜色簇像素占比，取前 4 种颜色（不足则取实际数量）。
3. 将所有索引的结果保存为 JSON，格式如下：
   {
     "1": {
       "forehead": [ {"rgb":[r,g,b], "percentage":0.523}, … ],
       "eyes":     [ … ],
       …
     },
     "2": { … },
     …
   }
4. 可视化：对每个索引，在一张大图中：
   - 上排五张部位图（原始 RGBA 合成到白底背景）；
   - 下排五个水平条形图，展示对应部位的颜色与占比。
5. 详细注释。

依赖：
    pip install pillow numpy opencv-python scikit-learn colormath tqdm matplotlib
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from tqdm import tqdm
import matplotlib.pyplot as plt


# 如果 numpy 没有 asscalar，补丁
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda x: x.item()

# -------------------- 配置区 --------------------
INPUT_DIR = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_segment_alpha")  # 图片输入目录
OUTPUT_JSON = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_segment_color_analysis.json") # JSON 输出路径
VIS_DIR = Path("/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/test_result")                # 可视化存图目录
VIS_DIR.mkdir(exist_ok=True)

# 部位名称列表，确保读取时顺序一致
PARTS = ["forehead", "eyes", "nose", "cheeks", "chin"]

# KMeans 聚类数量上限（后面会截取前 4 个）
BASE_CLUSTERS = 8

# 合并相似颜色的 CIE2000 阈值
MERGE_THRESHOLD = 10.0

# ------------------------------------------------


def load_lab_pixels(img_path):
    """
    加载 RGBA 图像，提取非透明像素，转换到 CIELAB 颜色空间。
    返回：lab_pixels (N,3) ndarray。
    """
    # 用 PIL 读入，保证 RGBA
    pil = Image.open(img_path).convert("RGBA")
    arr = np.array(pil)  # (H,W,4)
    # alpha 层 >0 的像素才算有效
    alpha = arr[..., 3]
    rgb_pixels = arr[..., :3][alpha > 0]  # (M,3)

    # 转换每个 RGB 到 LAB
    lab_list = []
    for rgb in rgb_pixels:
        # cv2 需要 uint8 和 BGR 顺序
        bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0,0]
        lab_list.append(lab.astype(float))
    if not lab_list:
        return np.empty((0,3), dtype=float)
    return np.stack(lab_list, axis=0)


def merge_similar_colors(centers, percentages, threshold=MERGE_THRESHOLD):
    """
    使用 CIE2000 将相似的聚类中心合并：
    - centers: (K,3) LAB 簇心
    - percentages: (K,) 各簇所占比例
    返回：list of dict {'rgb':[r,g,b], 'percentage':p}
    """
    merged = []
    # 逐一尝试将每个 center 合并到已有分组
    for lab, pct in zip(centers, percentages):
        lab_color = LabColor(*lab)
        found = False
        for m in merged:
            other = LabColor(*m['lab'])
            # 计算色差
            de = delta_e_cie2000(lab_color, other)
            if de < threshold:
                m['weight'] += pct
                found = True
                break
        if not found:
            merged.append({'lab': lab.tolist(), 'weight': pct})

    # 将 LAB 转回 RGB 并整理输出
    out = []
    for m in merged:
        lab = np.uint8([[[ *m['lab'] ]]])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0,0]
        rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
        out.append({'rgb': rgb, 'percentage': round(m['weight'], 4)})
    # 按比例降序
    out.sort(key=lambda x: x['percentage'], reverse=True)
    return out


def extract_top_colors(image_path, top_n=4):
    """
    对一张部位图进行 K-Means 聚类和合并，返回前 top_n 颜色及比例。
    """
    lab_pixels = load_lab_pixels(image_path)
    if lab_pixels.shape[0] == 0:
        return []

    # 确定聚类数：不超过 BASE_CLUSTERS 且不超过像素数
    k = min(BASE_CLUSTERS, lab_pixels.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(lab_pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 统计各簇频次
    counts = np.bincount(labels, minlength=k)
    percentages = counts / counts.sum()

    # 合并相似
    merged = merge_similar_colors(centers, percentages)
    # 取前 top_n
    return merged[:min(top_n, len(merged))]


def group_images_by_index(directory):
    """
    在目录下查找所有 "*_<part>_alpha.png" 文件，
    按索引分组，返回 dict { index_str: { part: Path } }。
    """
    groups = defaultdict(dict)
    for path in directory.glob("*_alpha.png"):
        name = path.stem  # e.g. "1_forehead_alpha"
        parts = name.split("_")
        if len(parts) >= 3:
            idx, part = parts[0], parts[1]
            if part in PARTS:
                groups[idx][part] = path
    return groups


def visualize_group(idx, part_paths, color_stats):
    """
    将一个索引的五个部位在一张大图中可视化：
    - 上排：5 张部位小图（合成白底 + RGBA）
    - 下排：5 个水平条形图，展示对应部位的颜色与比例
    """
    fig, axes = plt.subplots(2, len(PARTS), figsize=(4*len(PARTS), 8))
    fig.suptitle(f"Index {idx}", fontsize=16)

    for i, part in enumerate(PARTS):
        # 上：合成白底
        ax_img = axes[0, i]
        img = Image.open(part_paths[part]).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255,255,255,255))
        comp = Image.alpha_composite(bg, img).convert("RGB")
        ax_img.imshow(comp)
        ax_img.set_title(part)
        ax_img.axis("off")

        # 下：水平条形图
        ax_bar = axes[1, i]
        stats = color_stats.get(part, [])
        if stats:
            # 准备数据
            labels = [f"{c['percentage']*100:.1f}%" for c in stats]
            values = [c['percentage'] for c in stats]
            colors = [np.array(c['rgb'])/255 for c in stats]
            # 画水平 bar
            y_pos = np.arange(len(values))
            ax_bar.barh(y_pos, values, color=colors, edgecolor="black")
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(labels)
            ax_bar.set_xlim(0, max(values)*1.1)
        ax_bar.invert_yaxis()  # 最大值在上
        ax_bar.set_title(f"{part} colors")
        ax_bar.set_xlabel("Proportion")
    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = VIS_DIR / f"{idx}_overview.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # 分组
    groups = group_images_by_index(INPUT_DIR)

    # 存放最终结果
    all_results = {}

    # 为每个索引处理
    for idx, part_dict in tqdm(groups.items(), desc="Groups"):
        stats = {}
        # 对每个部位计算前 4 色
        for part, path in part_dict.items():
            stats[part] = extract_top_colors(path, top_n=4)
        all_results[idx] = stats
        # 可视化
        # visualize_group(idx, part_dict, stats)

    # 写入 JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 完成！JSON: {OUTPUT_JSON}  可视化: {VIS_DIR}/")


if __name__ == "__main__":
    main()
