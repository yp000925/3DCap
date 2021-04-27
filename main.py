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


parser = argparse.ArgumentParser(description='PyTorch 3D capture net Training')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', default=1, type=int, help='the training batch size')
parser.add_argument('--n_epochs', default=15, type=int, help='training epoches')
parser.add_argument('--exp_name', default='experiment0', type=str, help='experiment name')
parser.add_argument('--root_dir', default='/Users/zhangyunping/PycharmProjects/Holo_synthetic/data_holo',type=str,help='data root location')
parser.add_argument('--train_data',default='train_small.csv', type=str, help='train_dataset')
parser.add_argument('--valid_data',default='val_small.csv', type=str, help='validation_dataset')
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# TRAIN_LOSS = AverageMeter()
# VAL_LOSS= AverageMeter()
# ITER_number = AverageMeter()
# iter_time = AverageMeter()
# XYCENTER_LOSS = AverageMeter()
# DEPTHMAP_LOSS = AverageMeter()


def print_log(total_iter, is_training = True):
    global TRAIN_LOSS
    global VAL_LOSS
    global ITER_number
    global iter_time
    global XYCENTER_LOSS
    global DEPTHMAP_LOSS
    global logger

    if is_training:
        logger.info('\rIteration: {}/{} | XYcenter loss: {:1.5f} | '
          'Depthmap loss: {:1.5f} | Running loss average: {:1.5f}   Time : {:s} / {:s} in epoch'.format(
        ITER_number.val, total_iter, XYCENTER_LOSS.val, DEPTHMAP_LOSS.val, TRAIN_LOSS.avg,
        format_time(iter_time.sum), format_time(iter_time.val*total_iter)))
    else:
        logger.info('validation loss at this epoch is {:.4f}'.format(VAL_LOSS.val))

def train_epoch(model, datasetloader, optimizer):
    global TRAIN_LOSS
    global val_loss
    global iter_time
    global InterrutedFlag

    model.train()
    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:

        for data in datasetloader:
            t1 = time.time()
            optimizer.zero_grad()
            img, size_projection, xycentre, xy_mask = data

            if torch.cuda.is_available():
                pred_xycentroid, pred_depthmap = model(img.cuda().float())  # return (centroid_map,rendering_3D_map)
            else:
                pred_xycentroid,pred_depthmap = model(img.float())

            xycentre_loss = mse_TV_regularization_with_mask(pred_xycentroid,xycentre,xy_mask, alpha = 1e-4)
            dp_loss = depthmap_loss_with_mask(pred_depthmap,size_projection,xy_mask)

            loss = xycentre_loss+dp_loss

            if bool(loss == 0):
                continue

            loss.backward()

            # torch.nn.utils.clip_grad_norm(model.parameters(),0.1)

            optimizer.step()

            t2 = time.time()
            time_elp = t2 - t1
            TRAIN_LOSS.update(float(loss))
            iter_time.update(float(time_elp))
            XYCENTER_LOSS.update(float(xycentre_loss))
            DEPTHMAP_LOSS.update(float(dp_loss))
            writer.add_scalar("iteration_loss", TRAIN_LOSS.val,TRAIN_LOSS.count)
            writer.add_scalar("xycenter_loss", XYCENTER_LOSS.val,XYCENTER_LOSS.count)
            writer.add_scalar("depthmap_loss",DEPTHMAP_LOSS.val,DEPTHMAP_LOSS.count)

            print_log(is_training=True,total_iter=len(datasetloader))

            if u.interrupted:
                logger.info("Interrupted on request!")
                InterrutedFlag = True
                break


def validation(model,dataloader):
    global VAL_LOSS
    global InterrutedFlag


    VAL_LOSS.reset()
    model.eval()
    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
        for data in dataloader:
            img, size_projection, xycentre, xy_mask = data
            if torch.cuda.is_available():
                pred_xycentroid, pred_depthmap = model(img.cuda().float())
            else:
                pred_xycentroid, pred_depthmap = model(img.float())

            xycentre_loss = mse_TV_regularization_with_mask(pred_xycentroid, xycentre, xy_mask)
            dp_loss = depthmap_loss_with_mask(pred_depthmap, size_projection, xy_mask)

            _loss = xycentre_loss + dp_loss

            VAL_LOSS.update(float(_loss))

            print_log(len(dataloader), is_training=False)

            if u.interrupted:
                print("Interrupted on request!")
                InterrutedFlag = True
                break

def save_log(model, n_epoch, interrupted=False):
    global TRAIN_LOSS
    global logger
    logger.info('Saving..')
    state = {
        'net': model.state_dict(),
        'avg_loss': TRAIN_LOSS.avg,
        'epoch': n_epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not interrupted:
        torch.save(state, './checkpoint/ckpt_%d.pt'%(n_epoch))
    else:
        logger.info("Interrupted on request!")
        torch.save(state, './checkpoint/ckpt_interrupted.pt')
    logger.info('\nSaved')

if __name__ == "__main__":
    TRAIN_LOSS = AverageMeter()
    VAL_LOSS = AverageMeter()
    ITER_number = AverageMeter()
    iter_time = AverageMeter()
    XYCENTER_LOSS = AverageMeter()
    DEPTHMAP_LOSS = AverageMeter()
    InterruptedFlag = False
    args = parser.parse_args()

    logger = log_creater(args.exp_name)
    logger.info(args)

    if not os.path.isdir('summary'):
        os.mkdir('summary')
    writer = SummaryWriter('./summary/log')

    logger.info('==> Preparing data..')

    root_dir = args.root_dir
    train_data = holoData(root_dir, args.train_data)
    valid_data = holoData(root_dir, args.valid_data)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    model = Cap3D()
    logger.info(model)
    param_cnt = sum([param.numel() for param in model.parameters()  if param.requires_grad])
    logger.info('total trainable params: {:d} '.format(param_cnt))
    #
    # if args.resume:
    #     print('==> Resuming from checkpoint..')
    #     checkpoint = torch.load('./checkpoint/net_epoch0.pt')
    #     state_dict = checkpoint['net']
    #     new_state_dict = OrderedDict([(key.split('module.')[-1], state_dict[key]) for key in state_dict])
    #     model.load_state_dict(new_state_dict)
    #     start_epoch = checkpoint['epoch']
    #     print('loaded')
    # # use_gpu = True

    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(model).cuda()
    else:
        net = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    for epoch in range(args.n_epochs):
        logger.info('\nTraining Epoch:%d\n==================================================================' % epoch)
        train_epoch(model, trainloader, optimizer)
        writer.add_scalar('train_loss_epoch', TRAIN_LOSS.avg, epoch)

        if InterruptedFlag:
            save_log(model, epoch, interrupted=True)
            break

        save_log(model, epoch)
        logger.info('\nValidation\n==================================================================')
        validation(model, validloader)
        writer.add_scalar('validation_loss_epoch', VAL_LOSS.avg, epoch)

        if InterruptedFlag:
            save_log(model, epoch, interrupted=True)
            break





