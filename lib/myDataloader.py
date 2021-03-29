from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
import pandas as pd
from PIL import Image

class myDataloader(Dataset):
    """
    Dataset for 3D particle detection using capsule net
    """
    def __init__(self, root_dir, file_name = 'train_data.csv', transform = None, size=1024):
        '''

        :param holo_dir: directory for holograms
        :param depthmap_dir: directory for depthmap
        :param xycentre_dir: directory for xycentre
        :param file_name: file_name
        :param transform:
        '''
        # self.holo_dir = 'holo_dir'
        # self.depthmap_dir = 'depthmap_dir'
        # self.xycentre_dir = xycentre_dir
        self.root_dir = root_dir
        self.file_name = file_name
        self.transform = transform
        self.file = pd.read_csv(os.path.join(root_dir,file_name))
        self.N =size

    def __getitem__(self, idx):
        data = self.file[idx]
        holo_path = os.path.join(self.root_dir, 'hologram', data['hologram'])
        param_path = os.path.join(self.root_dir, 'param', data['param'])
        img = self.read_img(holo_path)
        depthmap = self.get_depthmap(param_path)
        xycentre = self.get_xycentre(param_path)


    def get_maps(self,param_path):
        param = pd.read_csv(param_path)
        param = self.param_transfer(param)
        xy_projection, xy_mask = self.get_xy_projection(param)
        xycentre = self.get_xycentre(param)
        return (xy_projection,xycentre)

    def get_xy_projection(self,param):
        arr = np.zeros((256,self.N,self.N))

        for _,particle in param.iterrows():
            px,py,pz,size = particle.x,particle.y,particle.z,particle.size
            Y, X = np.mgrid[:self.N, :self.N]
            Y = Y - py
            X = X - px
            dist_sq = Y ** 2 + X ** 2
            z_slice = np.zeros((self.N,self.N))
            z_slice[dist_sq <= size ** 2] = pz
            arr[pz,:,:] += z_slice # 可能某个depth上面有多个particles

        map = arr.sum(axis=0)
        # check whether there are overlapping
        mask = np.zeros(arr.shape)
        mask[arr.nonzero()]=1
        mask = mask.sum(axis=0)
        mask = ~(mask > 1)#在后面计算loss的时候，只计算没有overlap的pixel，即mask里面为false的情况忽略
        return map,

    def get_xycentre(self,param):
        arr = np.zeros((self.N, self.N))
        idx_x = np.array(param['x'])
        idx_y = np.array(param['y'])
        arr[[idx_y,idx_x]] = 1.0
        return arr

    def param_transfer(self,param):
        x = param['x']
        y = param['y']
        z = param['z']
        size = param['size']
        frame = 10 * 1e-3
        N=1024
        xyres = frame/N
        px = int(x / frame * N + N / 2)
        py = int(N / 2 + y / frame * N)
        pz = int((z - 1 * 1e-2) * 256)
        psize = int(size/xyres)
        param_pixel = pd.DataFrame()
        param_pixel['x'] = px
        param_pixel['y'] = py
        param_pixel['z'] = pz
        param_pixel['size'] = psize
        return param_pixel


    def read_img(self,img_name):
        img = Image.open(img_name)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img).astype(np.float32)
        return img/255.0

