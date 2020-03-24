import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class period_L1(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        periodic Squared Error
        '''
        super().__init__()
        self.reduction = reduction

    def forward(self, theta_pred, theta_gt):
        # assert theta_pred.shape == theta_gt.shape
        dt = theta_pred - theta_gt

        # periodic SE
        dt = torch.abs(torch.remainder(dt-np.pi/2,np.pi) - np.pi/2)
        
        assert (dt >= 0).all()
        if self.reduction == 'sum':
            loss = dt.sum()
        elif self.reduction == 'mean':
            loss = dt.mean()
        elif self.reduction == 'none':
            loss = dt
        return loss


class period_L2(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        periodic Squared Error
        '''
        super().__init__()
        if reduction == 'sum':
            self.reduction = reduction_sum
        elif reduction == 'mean':
            self.reduction = reduction_mean
        elif reduction == 'none':
            self.reduction = reduction_none
        else:
            raise Exception('unknown reduction')

    def forward(self, theta_pred, theta_gt):
        # assert theta_pred.shape == theta_gt.shape
        dt = theta_pred - theta_gt
        # periodic SE
        loss = (torch.remainder(dt-np.pi/2,np.pi) - np.pi/2) ** 2
        
        assert (loss >= 0).all()
        loss = self.reduction(loss)
        return loss


class bcel2bce(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        periodic Squared Error + BCE
        '''
        super().__init__()
        self.reduction = reduction
        self.constant = torch.log(torch.Tensor([4]).float())

    def forward(self, theta_pred, theta_gt):
        # assert theta_pred.shape == theta_gt.shape
        dt = theta_pred - theta_gt
        left_mask = dt < -np.pi
        right_mask = dt > np.pi
        middle_mask = ~(left_mask | right_mask)

        # BCE
        dt[left_mask] = - torch.log((dt[left_mask]+1.5*np.pi) / np.pi) \
                        - torch.log(1 - (dt[left_mask]+1.5*np.pi)/np.pi) \
                        - self.constant
        # periodic SE
        dt[middle_mask] = (torch.remainder(dt[middle_mask]-np.pi/2,np.pi) - np.pi/2) ** 2
        # BCE
        dt[right_mask] = - torch.log((dt[right_mask]-0.5*np.pi) / np.pi) \
                        - torch.log(1 - (dt[right_mask]-0.5*np.pi)/np.pi) \
                        - self.constant
        
        assert (dt >= 0).all()
        if self.reduction == 'sum':
            loss = dt.sum()
        elif self.reduction == 'mean':
            loss = dt.mean()
        elif self.reduction == 'none':
            loss = dt
        return loss


class FocalBCE(nn.Module):

    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super().__init__()
        self.focusing_param = gamma
        self.balance_param = alpha
        self.bce = nn.BCELoss(reduction='none')

        if reduction == 'sum':
            self.reduction = reduction_sum
        elif reduction == 'mean':
            self.reduction = reduction_mean
        elif reduction == 'none':
            self.reduction = reduction_none
        else:
            raise Exception('unknown reduction')

    def forward(self, output, target):
        logpt = - F.binary_cross_entropy(output, target, reduction='none')
        p_top = torch.exp(logpt)
        focal_loss = - (1 - p_top).pow(self.focusing_param) * logpt
        factor = torch.ones_like(focal_loss)
        factor[target > 0.5] = self.balance_param
        factor[target < 0.5] = 1 / self.balance_param
        focal_loss = focal_loss * factor
        
        focal_loss = self.reduction(focal_loss)
        return focal_loss


def smooth_L1_loss(pred, target, beta=1, weight=None, reduction='none'):
    '''
    Smooth L1 Loss. Original: https://github.com/facebookresearch/fvcore
    '''
    assert beta > 1e-5, 'If beta is smaller than 1e-5, use L1 instead'
    err = torch.abs(pred - target)
    loss = torch.where(err <= beta, 0.5*err*err/beta, err - 0.5*beta)
    if weight is not None:
        loss = weight * loss

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'none':
        pass
    else:
        raise Exception('Unknown reduction')
    return loss


def iou_loss(pred, target, iou_type='giou', reduction='mean'):
    '''
    Original: https://github.com/tianzhi0549/FCOS

    Args:
        iou_type: str, default: 'giou'
    '''
    raise Exception('Deprecated')
    assert pred.shape == target.shape
    assert pred.requires_grad and (not target.requires_grad)
    p_left = pred[..., 0] # left
    p_top = pred[..., 1] # top
    p_right = pred[..., 2] # right
    p_bottom = pred[..., 3] # bottom

    gt_left = target[..., 0]
    gt_top = target[..., 1]
    gt_right = target[..., 2]
    gt_bottom = target[..., 3]

    pred_area = (p_left + p_right) * (p_top + p_bottom)
    target_area = (gt_left + gt_right) * (gt_top + gt_bottom)

    w_intersect = torch.min(p_left, gt_left) + torch.min(p_right, gt_right)
    h_intersect = torch.min(p_bottom, gt_bottom) + torch.min(p_top, gt_top)
    intersect = w_intersect * h_intersect
    union = target_area + pred_area - intersect
    ious = (intersect + 1) / (union + 1)
    if iou_type == 'iou':
        losses = -torch.log(ious)
    if iou_type == 'linear_iou':
        losses = 1 - ious
    elif iou_type == 'giou':
        G_w = torch.max(p_left, gt_left) + torch.max(p_right, gt_right)
        G_h = torch.max(p_bottom, gt_bottom) + torch.max(p_top, gt_top)
        G_union = G_w * G_h + 1e-8
        # GIoU loss is between -1 to 1. Add 1 to make it visually appealing
        losses = 1 + (G_union - union) / G_union - ious

    if reduction == 'sum':
        return losses.sum()
    elif reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise Exception('unknown reduction')


def reduction_sum(loss):
    return loss.sum()

def reduction_mean(loss):
    return loss.mean()

def reduction_none(loss):
    return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # lossfunc = FocalBCE(reduction='none')
    # lossfunc = bcel2bce(reduction='none')
    lossfunc = smooth_L1_loss

    x = torch.linspace(-10, 10, steps=10001, requires_grad=True)
    # out = torch.sigmoid(x)
    out = x
    
    loss = lossfunc(out,torch.zeros_like(x))
    loss.sum().backward()
    dx = x.grad.detach().numpy()

    # x2 = torch.linspace(-10, 10, steps=10001, requires_grad=True)
    # out2 = torch.sigmoid(x2)
    # # x = torch.linspace(0, 1, steps=1001, requires_grad=True)
    # loss_bce = F.binary_cross_entropy(out2,torch.zeros_like(x), reduction='none')
    # loss_bce.sum().backward()
    # dx_bce = x2.grad.detach().numpy()

    # dx[dx >= (np.pi/2)**2] = np.inf
    x = x.detach().numpy()
    plt.figure()
    # plt.plot(x, loss_bce.detach().numpy(), label='BCE')
    plt.plot(x, loss.detach().numpy(), label='Loss')
    plt.plot(x, dx, linestyle='--', label='Derivative')
    # plt.plot(x, dx_bce, linestyle='--', label='derivative of BCE')
    # plt.vlines(0,-10,10)
    plt.legend()
    plt.ylim((-8,8))
    # plt.xticks(np.arange(-2*np.pi, 3*np.pi, 0.5*np.pi), rotation=0)
    plt.xticks(np.arange(-10, 10, 1), rotation=0)
    # plt.xticks(np.arange(0,1.1,0.2))
    plt.grid(linestyle=':')
    # plt.xlabel('Predicted probability when ground truth = 1')
    plt.ylabel('Loss')
    plt.title('Loss functions')

    plt.show()
