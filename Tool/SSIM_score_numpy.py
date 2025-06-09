import argparse
import os
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from data_prepare.bev_numpy import get_bev, crop_image_with_padding
from data_prepare.crop_fov import crop_and_resize_image
from data_prepare.proj_fov import transform_image
from dataset.VIGOR_correct import VIGOR


def calculate_ssim_score(imageA, imageB, win_size=31, visualize=False):
    if imageA is None or imageB is None:
        raise ValueError("One or both of the image  are invalid.")

    # 确保图像尺寸一致
    imageA = cv2.resize(imageA, (imageB.shape[1], imageB.shape[0]))
    # 计算多通道（彩色）图像的 SSIM
    ssim_score, diff = ssim(imageA, imageB, win_size=win_size, multichannel=True, full=True, channel_axis=2)

    # 可视化图像及差异图像
    if visualize:
        # 将差异图像归一化到 [0, 255] 范围
        diff = (diff * 255).astype("uint8")

        # 使用 Matplotlib 显示图像和差异图像
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.title("Image A")
        plt.imshow(imageA)

        plt.subplot(1, 3, 2)
        plt.title("Image B")
        plt.imshow(imageB)

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.axis('off')
        plt.imshow(diff, cmap='viridis')

        # plt.show()
        plt.savefig(os.path.join(args.output_dir, 'ssim_g.png'),bbox_inches="tight", dpi=300, pad_inches=0.0)
    return ssim_score


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
    parser.add_argument("--output_dir", "-o", type=str,default="../outputs/data",
                        help="output directory")
    parser.add_argument("--output_dir_segment", "-o1", type=str, default="../outputs/outputs_vigor_segment",
                        help="output directory")

    args = parser.parse_args()
    start_time_part0 = time.time()


    end_time_part0 = time.time()
    execution_time_part0 = end_time_part0 - start_time_part0
    print(f"*******************加载模型用时：{execution_time_part0} 秒")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_segment, exist_ok=True)
    semi_index = 0
    val_dataset = VIGOR(args=args, mode='val', same_area=False, semi_index=semi_index, length=1000, gt="HC")


    time_num = 0
    index = 27
    ratio_threshold = 1
    position_threshold = 30
    item = val_dataset[index]
    image_numpy_r, image_numpy_g, angle, sat_delta = item
    real_y, real_x = sat_delta

    image_numpy_list = [image_numpy_r, image_numpy_g]

    g_crop = crop_and_resize_image(image_numpy_g,80)
    r_proj = transform_image(image_numpy_r, real_x, real_y, 256,80)

    bev = get_bev(image_numpy_g,512,80)
    r_crop = crop_image_with_padding(image_numpy_r, real_x, real_y, 128)

    cx = real_x
    cy = real_y
    start_time_part1 = time.time()
    # 计算sift得分
    # 计算ssim得分
    ssim_score = calculate_ssim_score(g_crop, r_proj, win_size=31,visualize=True)
    # ssim_score = calculate_ssim_score(bev, r_crop, win_size=31, visualize=True)
    score_time = time.time() - start_time_part1
    print("index:",index,"计算得分用时：", score_time)
    time_num +=score_time
    print("总用时：",time_num)

