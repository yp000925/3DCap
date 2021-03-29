import torch
from models import Cap3D
import torch.nn.functional as F

from utils import mse_TV_regularization,depthmap_loss
import time




def train_epoch(model, datasetloader, optimizer, epoch):
    for iter_num, data in enumerate(datasetloader):
        t1 = time.time()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            pred_xycentroid, pred_depthmap = model(data['img'].cuda().float())
        else:
            pred_xycentroid,pred_depthmap = model(data['img'].float())