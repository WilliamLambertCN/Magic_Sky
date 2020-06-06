# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('.')
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
from matplotlib import pyplot as plt
from tools.common_tools import set_seed

set_seed()

class SkyDataset(Dataset):

    def __init__(self, data_dir, transform=None, in_size=224):
        super(SkyDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.label_path_list = list()
        self.in_size = in_size

        # 获取img, mask的path
        self._get_img_path()

    def __getitem__(self, index):

        path_label = self.label_path_list[index]
        path_img = self.img_path_list[index]

        img_pil = Image.open(path_img).convert('RGB')
        img_pil = img_pil.resize((self.in_size, self.in_size), Image.BILINEAR)

        label_pil = Image.open(path_label).convert('L')
        label_pil = label_pil.resize((self.in_size, self.in_size), Image.NEAREST)

        if self.transform is not None:
            img_pil = self.transform(img_pil)
            label_pil = self.transform(label_pil)
        # plt.subplot(121).imshow(img_pil)
        # plt.subplot(122).imshow(label_pil)
        # plt.show()

        img_hwc = np.array(img_pil)
        img_chw = img_hwc.transpose((2, 0, 1))
        label_hw = np.array(label_pil)
        label_hw[label_hw != 0] = 1
        label_chw = label_hw[np.newaxis, :, :]

        img_chw_tensor = torch.from_numpy(img_chw).float()
        label_chw_tensor = torch.from_numpy(label_chw).float()

        return img_chw_tensor, label_chw_tensor

    def __len__(self):
        return len(self.label_path_list)

    def _get_img_path(self):

        img_list = glob(self.data_dir + '/image/*')
        label_list = glob(self.data_dir + '/mask/*')
        if len(label_list) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        if len(label_list) != len(img_list):
            raise Exception("\nImages %d and labels %d are inconsistent! Please checkout your dataset!".format(
                    len(label_list), len(img_list), self.data_dir))
        self.label_path_list = label_list
        self.img_path_list = img_list
