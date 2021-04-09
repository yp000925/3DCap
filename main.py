import torch
from models import Cap3D
import torch.nn.functional as F

from utils import mse_TV_regularization, depthmap_loss
import time




def train_epoch(model, datasetloader, optimizer, epoch):
    for iter_num, data in enumerate(datasetloader):
        t1 = time.time()
        optimizer.zero_grad()
        img, size_projection, xycentre, xy_mask = data

        if torch.cuda.is_available():
            pred_xycentroid, pred_depthmap = model(img.cuda().float())
        else:
            pred_xycentroid,pred_depthmap = model(img.float())

        tv_loss = mse_TV_regularization(pred_xycentroid,xycentre)
        dp_loss = depthmap_loss(pred_depthmap,size_projection)