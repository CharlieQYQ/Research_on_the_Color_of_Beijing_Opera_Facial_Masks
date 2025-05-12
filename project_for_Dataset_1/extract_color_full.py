import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# 添加猴子补丁修复colormath兼容性问题
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda x: x.item()


def load_image(image_path):
    """加载图像并转换为LAB颜色空间"""
    img = Image.open(image_path).convert('RGBA')
    img_array = np.array(img)

    # 提取有效像素（排除透明区域）
    alpha = img_array[:, :, 3]
    valid_pixels = img_array[alpha > 0][:, :3]

    # 转换为LAB颜色空间
    lab_pixels = []
    for pixel in valid_pixels:
        lab = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_RGB2LAB)[0][0]
        lab_pixels.append(lab)

    return np.array(lab_pixels), img_array.shape


def merge_similar_colors(colors, percentages, threshold=10):
    """使用CIE2000色差算法合并相似颜色"""
    merged = []
    for color, percent in zip(colors, percentages):
        found = False
        for m in merged:
            color1 = LabColor(*color)
            color2 = LabColor(*m['lab'])

            # 此处调用numpy中的asscalar()方法已经通过猴子补丁修复
            delta_e = delta_e_cie2000(color1, color2)

            if delta_e < threshold:
                m['count'] += percent
                found = True
                break
        if not found:
            merged.append({
                'lab': color.tolist(),
                'count': percent
            })

    # 转换回RGB并格式化
    result = []
    for m in merged:
        rgb = cv2.cvtColor(np.uint8([[m['lab']]]), cv2.COLOR_LAB2RGB)[0][0]
        result.append({
            'rgb': rgb.tolist(),
            'percentage': round(m['count'], 4)
        })

    return sorted(result, key=lambda x: -x['percentage'])


def visualize_colors(image_path, color_data, save_path=None):
    """
    可视化颜色分布
    :param image_path: 原始图片路径
    :param color_data: 分析结果[{'rgb': [r,g,b], 'percentage': float}]
    :param save_path: 可视化结果保存路径
    """
    plt.figure(figsize=(12, 6))

    # 显示原始图片（处理透明背景）
    ax1 = plt.subplot(1, 2, 1)
    img = Image.open(image_path).convert('RGBA')
    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(background, img).convert('RGB')
    plt.imshow(composite)
    plt.axis('off')
    plt.title('Original Image')

    # 显示颜色分布
    ax2 = plt.subplot(1, 2, 2)
    current_y = 0.9
    max_percent = max(c['percentage'] for c in color_data)

    for color_info in color_data:
        rgb = color_info['rgb']
        percent = color_info['percentage']

        # 颜色块
        ax2.add_patch(plt.Rectangle(
            (0.1, current_y - 0.05), 0.3, 0.1,
            facecolor=[x / 255 for x in rgb],
            edgecolor='black',
            linewidth=0.5
        ))

        # 百分比条
        bar_length = 0.5 * (percent / max_percent)
        ax2.add_patch(plt.Rectangle(
            (0.45, current_y - 0.025), bar_length, 0.05,
            facecolor=[x / 255 for x in rgb]
        ))

        # 文本标签
        text = f"RGB{rgb} - {percent:.2%}"
        plt.text(0.48, current_y - 0.03, text, fontsize=9)

        current_y -= 0.15

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    plt.title('Color Distribution')

    # 保存或显示结果
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()


def analyze_image(image_path, color_tolerance=10, visualize=False):
    """分析单张图片颜色"""
    lab_pixels, img_shape = load_image(image_path)
    if len(lab_pixels) == 0:
        return []

    # 自动确定聚类数量
    max_clusters = min(20, int(np.sqrt(len(lab_pixels) / 100)))
    kmeans = KMeans(n_clusters=max_clusters, random_state=0)
    labels = kmeans.fit_predict(lab_pixels)

    # 计算各聚类中心占比
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / counts.sum()

    # 合并相似颜色
    merged_colors = merge_similar_colors(
        kmeans.cluster_centers_,
        percentages,
        threshold=color_tolerance
    )

    # 可视化处理
    if visualize:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(
            os.path.dirname(image_path),
            f"{base_name}_colors.png"
        )
        visualize_colors(image_path, merged_colors[:6], vis_path)

    return merged_colors[:6]


def process_folder(input_dir, output_json, color_tolerance=10, visualize=False):
    """批量处理图片"""
    results = {}

    # 获取排序后的文件列表
    files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    for filename in tqdm(files, desc="Processing", unit="img"):
        file_path = os.path.join(input_dir, filename)
        colors = analyze_image(file_path, color_tolerance, visualize=visualize)
        results[filename] = colors

    # 保存JSON结果
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = "/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_alpha"
    OUTPUT_JSON = "Dataset_1_full_color_analysis.json"

    # 运行处理流程
    process_folder(INPUT_DIR, OUTPUT_JSON, color_tolerance=13, visualize=True)
