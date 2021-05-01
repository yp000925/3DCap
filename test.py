
import torch
from models import Cap3D
import torch.nn.functional as F
from lib.holoData import holoData
import torch.optim as optim
from utils import mse_TV_regularization_with_mask, depthmap_loss_with_mask, format_time, log_creater
import time
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import lbtoolbox as lb
from signal import SIGINT, SIGTERM

model = Cap3D()
print("Resuming ===========>")
ckp_path = ''
checkpoint = torch.load(ckp_path)
state_dict = checkpoint['net']
model.load_state_dict(state_dict)
print('Loaded')

if torch.cuda.is_available():
    model = model.cuda()

root_dir = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/data_holo/data_holo'
dataset = holoData(root_dir, 'val_small.csv')
model.eval()
with torch.no_grad():
    img, size_projection, xycentre, xy_mask = dataset[0]
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        pred_xycentroid, pred_depthmap = model(img.cuda().float())
    else:
        pred_xycentroid, pred_depthmap = model(img.float())


