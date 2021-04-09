import torch

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



if __name__ == "__main__":
    # a = torch.rand((2,3,4,4))
    a = torch.zeros((1,3,2,2))
    b = torch.rand((1,3,2,2))
    print(mse_TV_regularization(a,b),depthmap_loss(a,b))
