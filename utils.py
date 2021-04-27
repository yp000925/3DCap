import torch
import os
import time
import logging
def _hober_loss(input, target, sigma=0.001, reduction='sum'):
    diff = torch.abs(input - target)
    cond = diff < sigma
    loss = torch.where(cond, 0.5 * diff ** 2, diff*sigma - 0.5 * sigma**2)
    if reduction == 'sum':
        return torch.sum(loss)
    if reduction == 'avg':
        return torch.mean(loss)
    return torch.sum(loss, dim=1)

def depthmap_loss(output, target):
    hober_loss = _hober_loss(output,target, sigma =0.001, reduction='sum')
    return hober_loss

def depthmap_loss_with_mask(output,target,mask):
    hober_loss = _hober_loss_with_mask(output, target, mask, sigma=0.001, reduction='sum')
    return hober_loss

def _hober_loss_with_mask(input,target,mask,sigma=0.001, reduction='sum'):
    """
    :param input:
    :param target:
    :param mask: float tensor 1: count the loss 0: ignore the loss
    :param sigma:
    :param reduction:
    :return:
    """
    diff = torch.abs(input - target)
    cond = diff < sigma
    loss = torch.where(cond, 0.5 * diff ** 2, diff*sigma - 0.5 * sigma**2)
    loss = loss*mask
    if reduction == 'sum':
        return torch.sum(loss)
    if reduction == 'avg':
        return torch.mean(loss)
    return torch.sum(loss, dim=1)

def mse_TV_regularization(output, target, alpha = 1e-4):
    mse_loss = torch.sum(torch.pow((output-target),2))
    tv_regularizor = _tv(output)
    return (1-alpha)*mse_loss+alpha*tv_regularizor

def mse_TV_regularization_with_mask(output, target, mask, alpha = 1e-4):
    """
    :param output:
    :param target:
    :param mask: float tensor 1: count the loss 0: ignore the loss
    :param alpha:
    :return:
    """
    if torch.cuda.is_available():
        mse_loss = torch.sum(torch.pow((output.cuda()-target.cuda()),2)*mask.cuda())
        tv_regularizor = _tv(output.cuda())
    else:
        mse_loss = torch.sum(torch.pow((output-target),2)*mask)
        tv_regularizor = _tv(output)
    return (1-alpha)*mse_loss+alpha*tv_regularizor

def _tv(img):
    """
    Compute total variation.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.

    Returns:
    - total_variance: PyTorch Variable holding a scalar giving the total variation
      for img.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))

    total_variance = torch.sqrt(h_variance + w_variance)
    return total_variance

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.INFO)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s]  [%(levelname)s]%(message)s',datefmt="%a %b %d %H:%M:%S %Y")

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log

if __name__ == "__main__":
    # a = torch.rand((2,3,4,4))
    a = torch.zeros((2,1,512,512))
    b = torch.rand((2,1,512,512))
    print(mse_TV_regularization_with_mask(a,b,b))
    # print(mse_TV_regularization(a,b),depthmap_loss(a,b))
