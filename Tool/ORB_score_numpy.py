import argparse
import os
import time

from data_prepare.bev_numpy import get_bev, crop_image_with_padding
from data_prepare.crop_fov import crop_and_resize_image
from data_prepare.proj_fov import transform_image
from dataset.VIGOR_correct import VIGOR

import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_orb_score(imageA, imageB, position_threshold=30, alpha=0.1, visualize=False):
    """
    用 ORB 计算两个图像的相似度分数
    Args:
        imageA (np.array): 图像 A 的数组。
        imageB (np.array): 图像 B 的数组。
        position_threshold (int): 位置约束的阈值，默认是 30。
        alpha (float): 控制距离衰减的参数，默认是 0.1。
        visualize (bool): 是否生成特征点匹配的可视化图像，默认是 False。

    Returns:
        float: 计算出的相似度分数，考虑了匹配的特征点的距离加权。
    """

    if imageA is None or imageB is None:
        raise ValueError("One or both of the images are invalid.")

    # 确保图像尺寸一致
    imageA = cv2.resize(imageA, (imageB.shape[1], imageB.shape[0]))

    # 将图像转换为灰度图
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 初始化 ORB 特征点检测器
    # nfeatures  控制返回的最大特征点数量。如果你想要更多特征点，可以增加这个值。(默认500)
    # ORB内部也有基于Harris响应值的筛选，但这不通过阈值直接控制
    # orb = cv2.ORB_create(nfeatures=500)
    orb = cv2.ORB_create()

    # 检测和计算特征点和描述符
    kpA, desA = orb.detectAndCompute(grayA, None)
    kpB, desB = orb.detectAndCompute(grayB, None)
    # 检查描述符是否为空
    if desA is None or desB is None:
        return 0.0, 0
    # 初始化 Brute-Force 匹配器
    # Hamming 距离 用来计算两个二进制向量的不同位数的数量（即两个描述符有多少位不同）。
    #  crossCheck（交叉检查）策略，要求两个特征点在彼此的最佳匹配中都相互匹配，减少错误匹配。

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 计算特征点匹配
    matches = bf.match(desA, desB)

    # 计算匹配分数
    similarity_score = 0
    good_matches = []
    for m in matches:
        ptA = kpA[m.queryIdx].pt
        ptB = kpB[m.trainIdx].pt
        pixel_distance = np.linalg.norm(np.array(ptA) - np.array(ptB))
        if pixel_distance < position_threshold:
            good_matches.append(m)
            similarity_score += np.exp(-alpha * pixel_distance)

    # 可视化特征点匹配
    if visualize:
        result_image = cv2.drawMatches(
            imageA, kpA, imageB, kpB, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image)
        plt.title("Feature Point Matching with ORB")
        plt.axis('off')
        plt.show()
        plt.imsave(os.path.join(args.output_dir, '37orb_r.png'), result_image)
    return similarity_score, len(good_matches)


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
    index = 37
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

    start_time_part1 = time.time()
    # 计算sift得分
    # orb_score, orb_num = calculate_orb_score(g_crop, r_proj,position_threshold=position_threshold,visualize=True)
    orb_score, orb_num = calculate_orb_score(bev, r_crop, position_threshold=position_threshold,visualize=True)
    score_time = time.time() - start_time_part1
    print("index:",index,"计算得分用时：", score_time)
    print("得分:",orb_score)
