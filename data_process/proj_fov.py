import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates
import time
from PIL import Image

from dataset.VIGOR_correct import VIGOR

import numpy as np
from scipy.ndimage import map_coordinates


def get_polar_original_vectorized(u, v, a, b, S, fov):
    a = int(a)
    b = int(b)
    z = 20
    H = S
    W = 2 * S


    # 使用FOV限制tan函数的范围
    tan_v = np.tan(np.pi * v / H)
    # 计算允许的最大tan值
    # max_tan_fov = np.tan(np.radians(fov))
    # tan_v = np.clip(tan_v, -max_tan_fov, max_tan_fov)

    x = a + z * tan_v * np.cos(-2 * np.pi * u / W + np.pi / 2)
    y = b - z * tan_v * np.sin(-2 * np.pi * u / W + np.pi / 2)
    return x, y

def transform_image(input_numpy, a, b, output_size=256, fov=90):
    original_image = input_numpy
    original_height, original_width, _ = original_image.shape

    S = min(original_width, original_height)
    new_width = output_size
    new_height = output_size
    e = int(output_size * np.tan(np.radians(90-fov)))

    # 创建网格用于向量化计算
    u = np.linspace(0, new_width - 1, new_width)
    v = np.linspace(new_height+e, new_height*2 - 1, new_height)
    uu, vv = np.meshgrid(u, v)

    # 缩放因子
    scale_u = S * 2 / new_width
    scale_v = S / (new_height * 2)

    # 计算原始坐标
    original_x, original_y = get_polar_original_vectorized(uu * scale_u, vv * scale_v, a, b, S,fov)

    # 使用 map_coordinates 进行双线性插值
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(3):  # 对于每个颜色通道
        new_image[:, :, i] = map_coordinates(original_image[:, :, i], [original_y, original_x], order=1, mode='nearest')


    return new_image


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

    # 生成 转换图
    image_numpy_projective = transform_image(image_numpy_r, 436, 355, 256,80)
    # image_numpy_projective = transform_image(image_numpy_r, real_x,real_y, 256)
    plt.imsave(os.path.join(args.output_dir, str(index) + 'fov_proj(436, 355).png'), image_numpy_projective)
    plt.imshow(image_numpy_projective)
    plt.title('Color Image')
    plt.axis('off')  # 隐藏坐标轴
    plt.show()
