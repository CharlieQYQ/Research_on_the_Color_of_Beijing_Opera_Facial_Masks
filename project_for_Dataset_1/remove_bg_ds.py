import os
import re
import cv2
import numpy as np
from tqdm import tqdm


def process_image_advanced(img_path, output_path, threshold=240, margin=3):
    """
    改进版背景去除核心函数
    :param img_path: 输入图片路径
    :param output_path: 输出图片路径
    :param threshold: 背景判断阈值（0-255）
    :param margin: 安全边距（像素）
    """
    # 读取图像并确保包含alpha通道
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    # 处理不同通道格式的图片
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 转换为灰度图并进行阈值处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作（去除噪点+填充小孔洞）
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 寻找最大轮廓（假设主体是最大轮廓）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # 创建空白蒙版并绘制最大轮廓
    mask = np.zeros_like(gray)
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    # 添加安全边距（防止残留边缘）
    mask = cv2.erode(mask, kernel, iterations=margin)

    # 应用透明效果：保留蒙版区域，透明化其他区域
    img[:, :, 3] = cv2.bitwise_and(img[:, :, 3], mask)

    # 保存结果
    cv2.imwrite(output_path, img)
    return True


def batch_process(input_dir, output_dir, threshold=245, margin=3):
    """
    批量处理函数（带进度条）
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param threshold: 背景判断阈值
    :param margin: 安全边距（像素）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取符合条件的文件列表
    file_list = sorted([
        f for f in os.listdir(input_dir)
        if re.match(r'^\d+\.(jpg|jpeg|png)$', f, re.IGNORECASE)
    ], key=lambda x: int(re.search(r'\d+', x).group()))

    # 使用tqdm显示进度
    for filename in tqdm(file_list, desc="Processing images", unit="image"):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_alpha.png"
        output_path = os.path.join(output_dir, output_filename)

        # 处理单张图片
        success = process_image_advanced(input_path, output_path, threshold, margin)

        if not success:
            tqdm.write(f"处理失败: {filename}")


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    INPUT_DIR = "/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1"  # 替换为实际输入路径
    OUTPUT_DIR = "/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1_alpha"  # 替换为实际输出路径

    # 启动处理流程
    batch_process(INPUT_DIR, OUTPUT_DIR, threshold=245, margin=3)
