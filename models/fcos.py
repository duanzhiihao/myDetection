import torch


def _ltrb_to_xywh(ltrb, nG, stride):
    '''
    transform (top,left,bottom,right) to (cx,cy,w,h)
    '''
    # training, (nB, nA, nG, nG, 4)
    assert ltrb.dim() == 5 and ltrb.shape[2] == ltrb.shape[3] == nG
    # if torch.rand(1) > 0.9: assert (ltrb[..., 0:4] <= nG).all()
    device = ltrb.device
    x_ = torch.arange(nG, dtype=torch.float, device=device) * stride
    centers_y, centers_x = torch.meshgrid(x_, x_)
    centers_x = centers_x.view(1,1,nG,nG) + stride / 2
    centers_y = centers_y.view(1,1,nG,nG) + stride / 2

    xywh = torch.empty_like(ltrb)
    xywh[..., 0] = centers_x - (ltrb[...,0] - ltrb[...,2])/2 # cx
    xywh[..., 1] = centers_y - (ltrb[...,1] - ltrb[...,3])/2 # cy
    xywh[..., 2] = ltrb[...,0] + ltrb[...,2] # w
    xywh[..., 3] = ltrb[...,1] + ltrb[...,3] # h
    return xywh


def _xywh_to_ltrb(xywh, nG, stride):
    '''
    transform (cx,cy,w,h) to (top,left,bottom,right).
    xywh should be unnormalized.
    '''
    assert (xywh > 0).all() and (xywh <= nG*stride).all()
    xywh = xywh.clone() / stride # now in 0-nG range
    centers_x = xywh[..., 0].floor() + 0.5
    centers_y = xywh[..., 1].floor() + 0.5

    ltrb = torch.empty_like(xywh)
    ltrb[..., 0] = centers_x - (xywh[..., 0] - xywh[..., 2]/2) # left
    ltrb[..., 1] = centers_y - (xywh[..., 1] - xywh[..., 3]/2) # top
    ltrb[..., 2] = xywh[..., 0] + xywh[..., 2]/2 - centers_x # right
    ltrb[..., 3] = xywh[..., 1] + xywh[..., 3]/2 - centers_y # bottom
    return ltrb * stride
