import torch
from math import pi
from pycocotools import mask as maskUtils


def bboxes_iou(bboxes_a, bboxes_b, xyxy=False):
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


def iou_rle(boxes1, boxes2, bb_format='cxcywhd', **kwargs):
    '''
    use mask method to calculate IOU between boxes1 and boxes2

    Args:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, degree)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, degree)
        bb_format: str,

    Return:
        iou_matrix: tensor, shape(N,M), float32, 
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert bb_format == 'cxcywhd'

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array. converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    assert boxes1.device == boxes2.device
    device = boxes1.device
    boxes1, boxes2 = boxes1.cpu().clone().detach(), boxes2.cpu().clone().detach()
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 5
    
    size = kwargs.get('img_hw', 2048)
    imh, imw = (size, size) if isinstance(size, int) else size
    if kwargs.get('normalized', False):
        # the [x,y,w,h] are between 0~1
        # assert (boxes1[:,:4] <= 1).all() and (boxes2[:,:4] <= 1).all()
        boxes1[:,0] *= imw
        boxes1[:,1] *= imh
        boxes1[:,2] *= imw
        boxes1[:,3] *= imh
        boxes2[:,0] *= imw
        boxes2[:,1] *= imh
        boxes2[:,2] *= imw
        boxes2[:,3] *= imh
    if bb_format == 'cxcywhd':
        # convert to radian
        boxes1[:,4] = boxes1[:,4] * pi / 180
        boxes2[:,4] = boxes2[:,4] * pi / 180

    b1 = xywha2vertex(boxes1, is_degree=False, stack=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False, stack=False).tolist()
    debug = 1
    
    b1 = maskUtils.frPyObjects(b1, imh, imw)
    b2 = maskUtils.frPyObjects(b2, imh, imw)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return torch.from_numpy(ious).to(device=device)


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


def xywha2vertex(box, is_degree, stack=True):
    '''
    Args:
        box: tensor, shape(batch,5), 5=(x,y,w,h,a), xy is center,
             angle is radian

    Return:
        tensor, shape(batch,4,2): topleft, topright, br, bl
    '''
    assert is_degree == False and box.dim() == 2 and box.shape[1] >= 5
    batch = box.shape[0]
    device = box.device

    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate two vector
    verti = torch.empty((batch,2), dtype=torch.float32, device=device)
    verti[:,0] = (h/2) * torch.sin(rad)
    verti[:,1] = - (h/2) * torch.cos(rad)

    hori = torch.empty(batch,2, dtype=torch.float32, device=device)
    hori[:,0] = (w/2) * torch.cos(rad)
    hori[:,1] = (w/2) * torch.sin(rad)


    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    if not stack:
        return torch.cat([tl,tr,br,bl], dim=1)
    return torch.stack((tl,tr,br,bl), dim=1)


def nms_rotbb(boxes, scores, nms_thres=0.45, bb_format='cxcywhd', img_size=2048,
              majority=None):
    '''
    Apply single class non-maximum suppression for rotated bounding boxes.
    
    Args:
        boxes: rows of (x,y,w,h,angle)
        scores:
        nms_thres:
        bb_format: True if input angle is in degree
        img_size: int or tuple-like
        majority (optional): int, a BB is suppresssed if the number of votes \
        less than majority. default: None
    
    Returns:
        keep: torch.int64
    '''
    # Sanity check
    if bb_format != 'cxcywhd':
        raise NotImplementedError()
    assert (boxes.dim() == 2) and (boxes.shape[1] == 5)
    device = boxes.device
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)
    
    # sort by confidence
    idx = torch.argsort(scores, descending=True)
    boxes = boxes[idx,:]

    valid = torch.zeros(boxes.shape[0], dtype=torch.bool, device=device)
    # the first one is always valid
    valid[0] = True
    # only one candidate at the beginning. Its votes number is 1 (it self)
    votes = [1]
    for i in range(1, boxes.shape[0]):
        # compute IoU with valid boxes
        # ious = iou_mask(boxes[i], boxes[valid,:], True, 32, is_degree=is_degree)
        ious = iou_rle(boxes[i], boxes[valid,:], bb_format=bb_format,
                       img_size=img_size)
        # the i'th BB is invalid if it is similar to any valid BB
        if (ious >= nms_thres).any():
            if majority is not None:
                # take down the votes for majority voting
                vote_idx = torch.argmax(ious).item()
                votes[vote_idx] += 1
            continue
        # else, this box is valid
        valid[i] = True
        # the votes number of the new candidate BB is 1 (it self)
        votes.append(1)

    keep = idx[valid]
    if majority is None:
        # standard NMS
        return keep
    votes_valid = (torch.Tensor(votes) >= majority)
    return keep[votes_valid]


def cxcywh_to_x1y1x2y2(cxcywh):
    assert cxcywh.shape[-1] >= 4
    x1y1x2y2 = cxcywh.clone()
    x1y1x2y2[...,0] = (cxcywh[..., 0] - cxcywh[..., 2] / 2)
    x1y1x2y2[...,1] = (cxcywh[..., 1] - cxcywh[..., 3] / 2)
    x1y1x2y2[...,2] = (cxcywh[..., 0] + cxcywh[..., 2] / 2)
    x1y1x2y2[...,3] = (cxcywh[..., 1] + cxcywh[..., 3] / 2)
    return x1y1x2y2
