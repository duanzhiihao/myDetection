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
    '''IoU between rotated bounding boxes

    Calculate IOU between boxes1 and boxes2 using binary masks and \
        Run Length Encoding (RLE)

    Args:
        boxes1, boxes2: torch.tensor or numpy, must have the same type
        bb_format: str, must be one of the following:
                   'cxcywhd': [center x, center y, width, height, degree(clockwise)]
        img_hw (optional but recommended): image height and width

    Return:
        iou_matrix: tensor, shape(N,M), float32, 
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert type(boxes1) == type(boxes2)
    assert bb_format == 'cxcywhd'

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
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
    if bb_format == 'cxcywhd':
        # convert to radian
        boxes1[:,4] = boxes1[:,4] * pi / 180
        boxes2[:,4] = boxes2[:,4] * pi / 180

    b1 = xywha2vertex(boxes1, is_degree=False, stack=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False, stack=False).tolist()
    
    b1 = maskUtils.frPyObjects(b1, imh, imw)
    b2 = maskUtils.frPyObjects(b2, imh, imw)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    if kwargs.get('return_numpy', False):
        return ious
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


def vertex2masks(vertices, mask_size=128):
    '''
    Convert vertices to binary masks

    Args:
        vertices: tensor, shape(batch,4,2)
                  4 means 4 corners of the box, in 0~1 normalized range
                    top left [x,y],
                    top right [x,y],
                    bottom right [x,y],
                    bottom left [x,y]
        mask_size: int or (h,w), size of the output tensor

    Returns:
        tensor, shape(batch,size,size), 0/1 mask of the bounding box
    '''
    # assert (vertices >= 0).all() and (vertices <= 1).all()
    assert vertices.dim() == 3 and vertices.shape[1:3] == (4,2)
    device = vertices.device
    batch = vertices.shape[0]
    mh,mw = (mask_size,mask_size) if isinstance(mask_size,int) else mask_size

    # create meshgrid
    gx = torch.linspace(0, 1, steps=mw, device=device).view(1,1,-1)
    gy = torch.linspace(0, 1, steps=mh, device=device).view(1,-1,1)
    gx = torch.arange(0, mw, dtype=torch.float, device=device).view(1,1,-1)
    gy = torch.arange(0, mh, dtype=torch.float, device=device).view(1,-1,1)

    # for example batch=9, all the following shape(9,1,1)
    tl_x = vertices[:,0,0].view(-1,1,1)
    tl_y = vertices[:,0,1].view(-1,1,1)
    tr_x = vertices[:,1,0].view(-1,1,1)
    tr_y = vertices[:,1,1].view(-1,1,1)
    br_x = vertices[:,2,0].view(-1,1,1)
    br_y = vertices[:,2,1].view(-1,1,1)
    bl_x = vertices[:,3,0].view(-1,1,1)
    bl_y = vertices[:,3,1].view(-1,1,1)

    # x1y1=tl, x2y2=tr
    masks = (tr_y-tl_y)*gx + (tl_x-tr_x)*gy + tl_y*tr_x - tr_y*tl_x < 0
    # x1y1=tr, x2y2=br
    masks *= (br_y-tr_y)*gx + (tr_x-br_x)*gy + tr_y*br_x - br_y*tr_x < 0
    # x1y1=br, x2y2=bl
    masks *= (bl_y-br_y)*gx + (br_x-bl_x)*gy + br_y*bl_x - bl_y*br_x < 0
    # x1y1=bl, x2y2=tl
    masks *= (tl_y-bl_y)*gx + (bl_x-tl_x)*gy + bl_y*tl_x - tl_y*bl_x < 0

    assert masks.shape == (batch, mh, mw)
    return masks


def bbox_to_mask(bboxes: torch.FloatTensor, bb_format='cxcywhd',
                 mask_size=2048) -> torch.BoolTensor:
    '''
    Convert bounding boxes to binary masks

    Args:
        bboxes: bounding boxes, shape [N, bb_param]
    
    Return:
        masks: shape [N, mask_size, mask_size]
    '''
    assert isinstance(bboxes, torch.FloatTensor) and bboxes.dim() == 2
    if bb_format == 'cxcywhd':
        assert bboxes.shape[1] == 5
        bboxes = bboxes.clone()
        bboxes[:,4] = bboxes[:,4] / 180 * pi
        vertices = xywha2vertex(bboxes, is_degree=False)
        masks = vertex2masks(vertices, mask_size=mask_size)
    else:
        raise NotImplementedError()

    return masks


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


def cxcywh_to_x1y1x2y2(cxcywh: torch.tensor) -> torch.tensor:
    assert cxcywh.shape[-1] >= 4
    x1y1x2y2 = cxcywh.clone()
    x1y1x2y2[...,0] = (cxcywh[..., 0] - cxcywh[..., 2] / 2)
    x1y1x2y2[...,1] = (cxcywh[..., 1] - cxcywh[..., 3] / 2)
    x1y1x2y2[...,2] = (cxcywh[..., 0] + cxcywh[..., 2] / 2)
    x1y1x2y2[...,3] = (cxcywh[..., 1] + cxcywh[..., 3] / 2)
    return x1y1x2y2
