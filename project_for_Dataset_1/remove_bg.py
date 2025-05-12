"""
remove_bg.py
------------
批量去除图片四周近似白色背景（图案内部同色不受影响），
支持同时处理 .jpg / .jpeg / .png，并输出 *_alpha.png。

运行:
    python remove_bg.py
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ——————————————— 用户可配置参数 ———————————————
INPUT_DIR = Path(r"/Users/charlieqyq/Documents/学习/研究生/研一下/色彩研究/Code/Dataset_1")  # ← 修改为实际路径
OUTPUT_DIR = INPUT_DIR / "alpha_png"               # 输出目录
COLOR_THRESH = 30                                  # 允许的色差阈值 (0–255)
SAMPLE_FRAC = 0.005                               # 左上角采样面积比例
# ————————————————————————————————————————————————


def collect_images(folder: Path) -> list[Path]:
    """
    收集数字命名且扩展名为 jpg/jpeg/png 的文件，按数字顺序返回 Path 列表。
    """
    patterns = ["*.jp*g", "*.png"]                 # *.jpg + *.jpeg + *.png
    files = []
    for pat in patterns:
        files.extend(folder.glob(pat))
    # 仅保留纯数字文件（stem.isdigit()），并按数字升序
    digits = [p for p in files if p.stem.isdigit()]
    return sorted(digits, key=lambda p: int(p.stem))


def sample_bg_color(img: np.ndarray, frac: float) -> np.ndarray:
    """采样左上角 frac 面积的均值颜色 (BGR)。"""
    h, w = img.shape[:2]
    side = max(1, int(np.sqrt(frac) * min(h, w)))
    roi = img[0:side, 0:side]
    return roi.reshape(-1, 3).mean(axis=0)


def create_bg_mask(img: np.ndarray, bg_col: np.ndarray,
                   thresh: int) -> np.ndarray:
    """
    颜色阈值 + flood-fill 生成仅含“外部背景”的掩码 (255=背景)。
    """
    h, w = img.shape[:2]
    diff = np.linalg.norm(img.astype(np.int16) - bg_col, axis=2)
    similar = (diff <= thresh).astype(np.uint8) * 255  # 255=候选背景
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    ff_mask[1:-1, 1:-1] = similar  # 原图区域填充候选掩码

    # 关键修改：传入image参数 + 调整种子坐标
    corners = [(0, 0), (0, w+1), (h+1, 0), (h+1, w+1)]  # 掩码四角坐标
    for seed in corners:
        cv2.floodFill(
            image=img.copy(),  # 占位参数，实际不修改
            mask=ff_mask,
            seedPoint=seed,
            newVal=0,         # 填充值为0（后续转换为背景）
            loDiff=(thresh, thresh, thresh),
            upDiff=(thresh, thresh, thresh),
            flags=cv2.FLOODFILL_MASK_ONLY | 4  # 仅操作掩码 + 4连通
        )

    # 外部背景在掩码中为0 → 转换为255
    return (ff_mask[1:-1, 1:-1] == 0).astype(np.uint8) * 255


def add_alpha_and_save(img_bgr: np.ndarray, bg_mask: np.ndarray,
                       out_path: Path):
    """以 bg_mask 反相作为 Alpha，合并 BGRA 并保存 PNG。"""
    alpha = cv2.bitwise_not(bg_mask)              # 背景透明 (0)
    b, g, r = cv2.split(img_bgr)
    bgra = cv2.merge((b, g, r, alpha))
    cv2.imwrite(str(out_path), bgra)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    img_paths = collect_images(INPUT_DIR)

    if not img_paths:
        print(f"❗ 未在 {INPUT_DIR} 找到符合条件的图片")
        return
    else:
        print("Load images ready.")

    for img_path in tqdm(img_paths, desc="Processing"):
        # 以 IMREAD_UNCHANGED 读取，便于识别 4 通道 PNG
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ 无法读取 {img_path.name}，已跳过")
            continue

        # 若已有 Alpha 通道 → 仅保留 BGR 部分
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # 1) 背景颜色采样
        bg_color = sample_bg_color(img, SAMPLE_FRAC)

        # 2) 背景掩码
        bg_mask = create_bg_mask(img, bg_color, COLOR_THRESH)

        # 3) 合成新 Alpha & 保存
        out_name = f"{img_path.stem}_alpha.png"
        add_alpha_and_save(img, bg_mask, OUTPUT_DIR / out_name)

    print(f"\n✅ 完成！全部结果已输出到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
