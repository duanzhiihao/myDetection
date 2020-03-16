import torch


def _ltrb_to_xywh(ltrb, nG):
    '''
    transform (top,left,bottom,right) to (cx,cy,w,h).
    ltrb should be normalized in the grid range
    '''
    # if torch.rand(1) > 0.9: assert (ltrb[..., 0:4] <= nG).all()
    device = ltrb.device
    mesh_x = torch.arange(nG, dtype=torch.float, device=device).repeat(nG,1)
    centers_x = mesh_x.view(1,1,nG,nG) + 0.5
    centers_y = mesh_x.t().view(1,1,nG,nG) + 0.5

    xywh = torch.empty_like(ltrb)
    xywh[..., 0] = centers_x - ltrb[...,0] + ltrb[...,2] # cx
    xywh[..., 1] = centers_y - ltrb[...,1] + ltrb[...,3] # cy
    xywh[..., 2] = ltrb[...,0] + ltrb[...,2] # w
    xywh[..., 3] = ltrb[...,1] + ltrb[...,3] # h
    return xywh


def _xywh_to_ltrb(xywh, nG):
    '''
    transform (cx,cy,w,h) to (top,left,bottom,right).
    xywh should be normalized in the grid range
    '''
    if torch.rand(1) > 0.9: assert (xywh[..., 0:2] <= nG).all()
    device = xywh.device
    centers_x = xywh[..., 0].floor() + 0.5
    centers_y = xywh[..., 1].floor() + 0.5

    ltrb = torch.empty_like(xywh)
    ltrb[..., 0] = centers_x - (xywh[..., 0] - xywh[..., 2]/2) # left
    ltrb[..., 1] = centers_y - (xywh[..., 1] - xywh[..., 3]/2) # top
    ltrb[..., 2] = xywh[..., 0] + xywh[..., 2]/2 - centers_x # right
    ltrb[..., 3] = xywh[..., 1] + xywh[..., 3]/2 - centers_y # bottom
    return ltrb
