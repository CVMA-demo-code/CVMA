import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torch.utils.data as data
import cv2
from Tool.Mercator import *
# 返回的delta 为像素坐标值（不是偏移值）,由两种图片名中的经纬度值计算得出
class VIGOR(Dataset):
    # Rewrite based on HC-NET
    def __init__(self, args, mode='train', root=r"D:\Project\DataSet\VIGOR", same_area=True, semi_index=0, length=-1, gt="slice"):
        self.semi_index = semi_index
        # same_area = not args.cross_area
        label_root = 'splits__corrected'  # 'splits' splits__corrected
        self.cur_index = 0
        if same_area:
            # self.train_city_list  = ['NewYork']
            self.train_city_list  =['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']

            # self.test_city_list = ['NewYork']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            # self.train_city_list = ['NewYork']
            # self.test_city_list = ['SanFrancisco']
            self.train_city_list = ['NewYork', 'Seattle']
            # self.test_city_list = ['SanFrancisco']
            self.test_city_list = ['SanFrancisco', 'Chicago']
            # self.test_city_list = ['Seattle']

        # 全景 [图片地址] 列表
        pano_list = []
        # 四张遥感图片 的 图片地址
        pano_label = []
        # 四张遥感图片的相机位置  [[(x1,y1),(x2,y2),(x3,y3),(x4,y4)],[...],...]
        sat_delta = []
        # 根据训练测试的数据集划分 选择标签文件
        if mode == 'train':
            for city in self.train_city_list:
                label_fname = os.path.join(root, label_root, city,
                                           'same_area_balanced_train__corrected.txt' if same_area else 'pano_label_balanced__corrected.txt')
                with open(label_fname, 'r') as file:
                    for line in file.readlines():
                        data = np.array(line.split(' '))
                        label = []
                        for i in [1, 4, 7, 10]:
                            label.append(os.path.join(root, city, 'satellite', data[i]))
                        pona_path = os.path.join(root, city, 'panorama', data[0])
                        pano_list.append(pona_path)

                        pano_label.append(label)
                        if gt == "slice":
                            # 用Slice 改进数据集自带偏移量 计算相机坐标
                            offset = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                            delta = np.array([(320 + y, 320 - x) for y, x in offset])
                        else:
                            # HC_NET:利用经纬度计算相机gt坐标
                            pano_gps = np.array(pona_path[:-5].split(',')[-2:]).astype(float)
                            pano_gps = torch.from_numpy(pano_gps)
                            delta = []
                            for sat_path in label:
                                sat_gps = np.array(sat_path[:-4].split('_')[-2:]).astype(float)
                                sat_gps = torch.from_numpy(sat_gps)
                                zoom = 20
                                y = get_pixel_tensor(sat_gps[0], sat_gps[1], pano_gps[0], pano_gps[1], zoom)
                                # (y,x)
                                coordinate = torch.round(y[1]).int(), torch.round(y[0]).int()
                                delta.append(coordinate)

                        sat_delta.append(delta)

                        self.cur_index += 1
                        if self.cur_index == length:
                            break
                if self.cur_index == length:
                    break
        else:
            for city in self.test_city_list:
                label_fname = os.path.join(root, label_root, city, 'same_area_balanced_test__corrected.txt'
                if same_area else 'pano_label_balanced__corrected.txt')
                with open(label_fname, 'r') as file:
                    for line in file.readlines():
                        data = np.array(line.split(' '))
                        label = []
                        for i in [1, 4, 7, 10]:
                            label.append(os.path.join(root, city, 'satellite', data[i]))
                        pona_path = os.path.join(root, city, 'panorama', data[0])
                        pano_list.append(pona_path)

                        pano_label.append(label)
                        if gt == "slice":
                            # 用Slice 改进数据集自带偏移量 计算相机坐标
                            offset = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                            delta = np.array([(320 + y, 320 - x) for y, x in offset])
                        else:
                            # HC_NET:利用经纬度计算相机gt坐标
                            pano_gps = np.array(pona_path[:-5].split(',')[-2:]).astype(float)
                            pano_gps = torch.from_numpy(pano_gps)
                            delta = []
                            for sat_path in label:
                                sat_gps = np.array(sat_path[:-4].split('_')[-2:]).astype(float)
                                sat_gps = torch.from_numpy(sat_gps)
                                zoom = 20
                                y = get_pixel_tensor(sat_gps[0], sat_gps[1], pano_gps[0], pano_gps[1], zoom)
                                # (y,x)
                                # coordinate = torch.round(y[1]).int(), torch.round(y[0]).int()
                                coordinate = (int(np.round(y[1].item())), int(np.round(y[0].item())))

                                delta.append(coordinate)

                        # 上面得到坐标的方式二选一
                        sat_delta.append(delta)
                        self.cur_index += 1
                        if self.cur_index == length:
                            break
                if self.cur_index == length:
                    break

        self.pano_list = pano_list
        self.pano_label = pano_label
        self.sat_delta = sat_delta

        self.mode = mode
        self.ori_noise = args.ori_noise
        # self.out = None
    def get_pano_list(self):
        return self.pano_list

    def get_sat_list(self):
        return self.pano_label

    def get_delta_list(self):
        return self.sat_delta

    def __len__(self):
        return len(self.pano_list)

    def __getitem__(self, idx):
        pano_path = self.pano_list[idx]
        select_ = self.semi_index
        sat_path = self.pano_label[idx][select_]
        # =================== read satellite map ===================================
        sat = cv2.imread(sat_path, 1)[:, :, ::-1].copy()
        sat_tensor = torch.from_numpy(sat)
        # =================== read ground map ===================================
        pano = cv2.imread(pano_path, 1)[:, :, ::-1].copy()
        rotation_range = self.ori_noise
        random_ori = np.random.uniform(-1, 1) * rotation_range / 360
        ori_angle = random_ori * 360
        pano = np.roll(pano, int(random_ori * pano.shape[1]), axis=1)
        pano_tensor = torch.from_numpy(pano)
        sat_delta = self.sat_delta[idx][select_]
        # return sat_tensor, pano_tensor, ori_angle, sat_delta
        return sat, pano, ori_angle, sat_delta
