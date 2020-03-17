import torch


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    """
    assert bboxes_a.dim() == bboxes_b.dim() == 2
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError()

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def xywh2mask(xywh, mask_size, resolution=1):
    '''
    Args:
        xywh: torch.tensor, rows of (cx,cy,w,h)
        mask_size: int for tuple, (h,w)
        resolution: the range of xywh. resolution=1 means xywh is normalized
    '''
    assert xywh.dim() == 2 and xywh.shape[-1] >= 4 and resolution == 1
    if torch.rand(1) > 0.99:
        if (xywh <= 0).any() or (xywh >= resolution).any():
            print('Warning: some xywh are out of range')
    device = xywh.device
    mh,mw = (mask_size,mask_size) if isinstance(mask_size,int) else mask_size

    # boundaries
    shape = xywh.shape[:-1]
    left = (xywh[..., 0] - xywh[..., 2] / 2).view(*shape,1,1)
    top = (xywh[..., 1] - xywh[..., 3] / 2).view(*shape,1,1)
    right = (xywh[..., 0] + xywh[..., 2] / 2).view(*shape,1,1)
    bottom = (xywh[..., 1] + xywh[..., 3] / 2).view(*shape,1,1)

    # create meshgrid
    x_ = torch.linspace(0,1,steps=mw+1, device=device)[:-1]
    y_ = torch.linspace(0,1,steps=mh+1, device=device)[:-1]
    gy, gx = torch.meshgrid(x_, y_)
    gx = gx.unsqueeze_(0) + resolution / (2*mw)
    gy = gy.unsqueeze_(0) + resolution / (2*mh)
    
    # build mask
    masks = (gx > left) & (gx < right) & (gy > top) & (gy < bottom)

    return masks
