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
        data = self.file.iloc[idx]
        holo_path = os.path.join(self.root_dir, 'hologram', data['hologram'])
        param_path = os.path.join(self.root_dir, 'param', data['param'])
        img = self.read_img(holo_path)
        param = self.load_param(param_path)
        size_projection,xycentre,xy_mask = self.get_maps(param)
        return img,size_projection,xycentre,xy_mask

    def __len__(self):
        return len(self.file)

    def get_maps(self,param):
        size_projection, xy_mask = self.get_xy_projection(param)
        xycentre = self.get_xycentre(param)
        return (size_projection,xycentre,xy_mask)

    def get_xy_projection(self,param):
        """
        :param param: px,py,pz,psize stored in dataframe
        :return: map: the xy_projection map, the pixel value is the corresponding depth, range from 0-1
                 mask: the indication map for overlapping 0: the overlap exists -> ignored when calculate the loss
        """
        arr = np.zeros((256,self.N,self.N))
        particle_field = np.zeros(arr.shape) # one stands for the exist of particle

        for _,particle in param.iterrows():
            px,py,pz,psize = particle.x,particle.y,particle.z,particle.size
            Y, X = np.mgrid[:self.N, :self.N]
            Y = Y - py
            X = X - px
            dist_sq = Y ** 2 + X ** 2
            z_slice = np.zeros((self.N,self.N))
            particle_field_slice = np.zeros((self.N,self.N))
            z_slice[dist_sq <= psize ** 2] = pz
            particle_field_slice[dist_sq <= psize ** 2] = 1
            arr[pz,:,:] += z_slice # 可能某个depth上面有多个particles
            particle_field[pz,:,:] += particle_field_slice

        map = arr.sum(axis=0)/255.0
        # check whether there are overlapping
        particle_field_proj = particle_field.sum(axis=0)
        mask_map = np.ones((self.N,self.N))
        mask_map[particle_field_proj>1] = 0 #在后面计算loss的时候，只计算没有overlap的pixel，即mask里面为0的情况忽略
        return map, mask_map

    def get_xycentre(self,param):
        arr = np.zeros((self.N, self.N))
        idx_x = np.array(param['x'].values)
        idx_y = np.array(param['y'].values)
        arr[(idx_y,idx_x)] = 1.0
        return arr

    def load_param(self,param_path):
        param = pd.read_csv(param_path)
        x = param['x'].values
        y = param['y'].values
        z = param['z'].values
        size = param['size']
        frame = 10 * 1e-3
        N=1024
        xyres = frame/N
        px = (x / frame * N + N / 2).astype(np.int)
        py = (N / 2 + y / frame * N).astype(np.int)
        pz = ((z - 1 * 1e-2)/ (3 * 1e-2 - 1 * 1e-2)*255).astype(np.int)
        psize = (size/xyres).astype(np.int)
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


if __name__ == "__main__":
    root_dir ='/Users/zhangyunping/PycharmProjects/Holo_synthetic/data_holo'
    file_path = 'check.csv'
    dataset = myDataloader(root_dir,file_path)
    # img = Image.open("/Users/zhangyunping/PycharmProjects/Holo_synthetic/data_holo/hologram/0.jpg")
    # param = dataloader.load_param(root_dir + '/param/0.csv')
    # img = dataloader.read_img(root_dir + '/hologram/0.jpg')
    # size_projection, xycentre, xy_mask = dataloader.get_maps(param)
    # size_p = Image.fromarray(size_projection*255.0)
    # size_p.show()
    # xyc = Image.fromarray(xycentre*255.0)
    # xyc.show()
    # xy_m = Image.fromarray(xy_mask*255.0)
    # xy_m.show()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=1)
    for data in dataloader:
        img, size_projection, xycentre, xy_mask = data

        break


