import argparse
import os
import matplotlib.pyplot as plt

import numpy as np
import cv2
from PIL import Image

from dataset.VIGOR_correct import VIGOR


def crop_and_resize_image(input_numpy,fov = 80):
    """
    Crop the upper half of the image and resize it to 256x256 using NumPy and OpenCV.

    Args:
        input_numpy (np.ndarray): Input image with shape (H, W, C) and values in range [0, 255].

    Returns:
        np.ndarray: Output image with shape (256, 256, C) and values in range [0, 255].
    """
    # Get the height and width of the input image
    height, width, _ = input_numpy.shape

    e = int(height // 2 * np.tan(np.radians(90 - fov)))
    # Crop the lower half of the image
    cropped_numpy = input_numpy[height // 2 + e:, :]

    # Resize the cropped image to 256x256
    resized_numpy = cv2.resize(cropped_numpy, (256, 256), interpolation=cv2.INTER_LINEAR)

    return resized_numpy


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
    parser.add_argument("--output_dir", "-o", type=str,default="../outputs/data1",
                        help="output directory")
    parser.add_argument("--output_dir_segment", "-o1", type=str, default="../outputs/outputs_vigor_segment",
                        help="output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_segment, exist_ok=True)
    semi_index = 0
    val_dataset = VIGOR(args=args, mode='val', same_area=False, semi_index=semi_index, length=1000, gt="HC")


    time_num = 0
    index = 37
    ratio_threshold = 1
    position_threshold = 30
    item = val_dataset[index]
    image_numpy_r, image_numpy_g, angle, sat_delta = item
    real_y, real_x = sat_delta
    cx = real_x
    cy = real_y
    image_numpy_list = [image_numpy_r, image_numpy_g]
    # 生成全景裁剪图
    image_numpy_crop = crop_and_resize_image(image_numpy_g,80)
    plt.imsave(os.path.join(args.output_dir, str(index)+"_(" +str(cx)+","+str(cy)+")" +'fov_crop_g.png'),image_numpy_crop)
    plt.imshow(image_numpy_crop)
    plt.title('Color Image')
    plt.axis('off')  # 隐藏坐标轴
    plt.show()