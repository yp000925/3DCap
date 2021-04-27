from __future__ import print_function, division
import sys
import os
import torch,torchvision
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
import pandas as pd
from PIL import Image


class holoData(Dataset):
    '''
    Dataset loader for 3D particles detection
    Return holograms and labels
    '''
    def __init__(self, root_dir, file_name = 'train_data.csv', transform = None):
        '''
        :param holo_dir: directory for holograms
        :param depthmap_dir: directory for depthmap
        :param xycentre_dir: directory for xycentre
        :param file_name: file_name
        :param transform:
        '''
        self.root_dir = root_dir
        self.file_name = file_name
        self.transform = transform
        self.file = pd.read_csv(os.path.join(root_dir,file_name))


    def __getitem__(self, idx):
        img_name = self.file.iloc[idx][0]
        img_path = os.path.join(self.root_dir, 'img', img_name)
        size_projection_path = os.path.join(self.root_dir, 'size_projection', img_name)
        xycentre_path = os.path.join(self.root_dir, 'xycentre', img_name)
        xy_mask_path = os.path.join(self.root_dir, 'xy_mask', img_name)
        img = self.read_img(img_path)
        size_projection = self.read_img(size_projection_path)
        xycentre = self.read_img(xycentre_path)
        xy_mask = self.read_img(xy_mask_path)
        return img,size_projection,xycentre,xy_mask

    def read_img(self,img_name):
        img = Image.open(img_name)
        # img = np.array(img).astype(np.float32)
        img = torchvision.transforms.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.file)

if __name__ == "__main__":
    root_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/data_holo'
    file_name = 'train.csv'
    dataset = holoData(root_dir,file_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=1)
    idx = 0
    for data in dataloader:
        img, size_projection, xycentre, xy_mask = data
        break
