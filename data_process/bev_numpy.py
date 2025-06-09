import argparse
import os
import time

import cv2
import numpy as np
import math

from PIL import ImageOps, Image

from matplotlib import pyplot as plt

from dataset.VIGOR_correct import VIGOR

def crop_image_with_padding(image, x, y, r=128):
    """
    使用 OpenCV 实现裁剪并填充。
    image: 输入的 NumPy 图像，形状为 (H, W, 3)。
    x, y: 裁剪区域的中心坐标。
    r: 正方形区域的半径。
    """
    # 计算需要的填充量
    top_padding = max(0, r - y)
    bottom_padding = max(0, (y + r) - image.shape[0])
    left_padding = max(0, r - x)
    right_padding = max(0, (x + r) - image.shape[1])

    # 使用 cv2.copyMakeBorder 进行填充
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 计算新的中心位置
    new_x = x + left_padding
    new_y = y + top_padding

    # 进行裁剪
    cropped_image = padded_image[new_y - r:new_y + r, new_x - r:new_x + r]

    return cropped_image


def get_bev(panorama_image,patch_size=512,fov=80):
    # 获取全景图像的尺寸
    Hp, Wp, _ = panorama_image.shape

    # 定义遥感视角的图像尺寸
    Hb, Wb = patch_size,patch_size  # 可以根据需要设定输出图像大小

    # 定义FoV参数
    f = 0.5 * Wb / math.tan(math.radians(fov))  # 焦距计算

    # 创建输出图像（初始化为黑色）
    bird_eye_view = np.zeros((Hb, Wb, 3), dtype=np.uint8)

    # 对于每个遥感视角图像中的像素点进行映射
    for ub in range(Wb):
        for vb in range(Hb):
            # 根据公式 (1) 计算对应的全景图像中的坐标 (up, vp)
            x1 = (Wb / 2 - ub)
            y1 = (Hb / 2 - vb)
            denominator = np.sqrt(x1**2 + y1**2)

            if denominator != 0:
                theta = np.arctan2(x1, y1)
                r = np.sqrt(x1**2 + y1**2)
                up = (1 - np.arctan2(Wb / 2 - ub, Hb / 2 - vb) / np.pi) * Wp / 2
                vp = (0.5 - np.arctan2(-f, denominator) / np.pi) * Hp

                # 确保坐标在图像范围内
                up = int(np.clip(up, 0, Wp - 1))
                vp = int(np.clip(vp, 0, Hp - 1))

                # 将全景图像的像素值映射到鸟瞰图上
                bird_eye_view[vb, ub] = panorama_image[vp, up]

    bev = cv2.resize(bird_eye_view, (256, 256))
    return bev

# 示例用法
if __name__ == "__main__":
    # 参数配置

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)

    # 数据集参数
    parser.add_argument('--cross_area', default=False, action='store_true', help='Cross_area or same_area')
    parser.add_argument("--dataset", type=str, default="vigor", help="")
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--ori_noise', type=float, default=0.0, help='orientation noise for VIGOR')

    # 保存路径
    parser.add_argument("--output_dir", "-o", type=str, default="../outputs/data",
                        help="output directory")
    parser.add_argument("--output_dir_segment", "-o1", type=str, default="../outputs/outputs_vigor_segment",
                        help="output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_segment, exist_ok=True)
    semi_index = 0
    val_dataset = VIGOR(args=args, mode='val', same_area=False, semi_index=semi_index, length=150, gt="HC")

    time_num = 0
    index_list = [37]
    ratio_threshold = 1
    position_threshold = 30
    for index in index_list:
        item = val_dataset[index]
        image_numpy_r, image_numpy_g, angle, sat_delta = item
        real_y, real_x = sat_delta

        image_numpy_list = [image_numpy_r, image_numpy_g]
        # 生成全景裁剪图

        bev = get_bev(image_numpy_g,512,80)
        plt.imsave(os.path.join(args.output_dir,str(index)+'bev.png'), bev)
        crop = crop_image_with_padding(image_numpy_r, 436, 355, 128)
        # crop = crop_image_with_padding(image_numpy_r,real_x,real_y,128)
        plt.imsave(os.path.join(args.output_dir,str(index)+'crop(436, 355).png'), crop)
        # 使用 Matplotlib 显示图像和差异图像
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Image A")
        plt.imshow(bev)


        plt.subplot(1, 2, 2)
        plt.title("Image B")
        plt.imshow(crop)

        plt.show()