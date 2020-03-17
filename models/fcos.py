import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

from .backbones import get_backbone
from .fpns import get_fpn
from utils.structures import ImageObjects


class FCOS(nn.Module):
    def __init__(self, backbone, fpn, class_num=80, **kwargs):
        super().__init__()
        self.backbone, chs, strides = get_backbone(name=backbone)
        self.fpnrpn = get_fpn(fpn, in_channels=chs, class_num=class_num)

        self.head = nn.ModuleList()
        for s in strides:
            self.head.append(FCOSLayer(stride_all=strides, stride=s,
                                       class_num=class_num))

        self.input_normalization = kwargs.get('img_norm', False)

    def forward(self, x, labels=None):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        assert x.dim() == 4 and x.shape[2] == x.shape[3]
        self.img_size = x.shape[2]
        # normalization
        if self.input_normalization:
            for i in range(x.shape[0]):
                x[i] = tvf.normalize(x[i], [0.485,0.456,0.406], [0.229,0.224,0.225],
                                    inplace=True)
                # debug = (x.mean(), x.std())

        # go through the backbone and the feature payamid network
        features = self.backbone(x)
        pyramid_dts = self.fpnrpn(features)

        dts_all = []
        losses_all = []
        for i, raw_dts in enumerate(pyramid_dts):
            bboxes, loss = self.head[i].forward(raw_dts, self.img_size, labels)
            dts_all.append(bboxes)
            losses_all.append(loss)
        
        if labels is None:
            bboxes = torch.cat(dts_all, dim=1)
            bbs, confs, clss = bboxes[...,:4], bboxes[...,4], bboxes[...,5:]
            cls_score, cls_idx = clss.max(dim=2, keepdim=False)
            # debug = boxes[boxes[...,5]>0.5]
            bb = ImageObjects(bboxes=bbs, cats=cls_idx, scores=confs*cls_score)
            return bb
        else:
            # check all the gt objects are assigned
            assert isinstance(labels, list)
            # gt_num = sum([t.shape[0] for t in labels])
            # assigned = self.yolo_L.gt_num + self.yolo_M.gt_num + self.yolo_S.gt_num
            # assert assigned == gt_num
            self.loss_str = ''
            for m in self.head:
                self.loss_str += m.loss_str + '\n'
            loss = sum(losses_all)
            return loss


class FCOSLayer():
    def __init__(self, **kwargs):
        self.strides_all = torch.Tensor(kwargs['stride_all'])
        self.stride = kwargs['stride']
        self.n_cls = kwargs['class_num']

        self.ltrb_setting = 'exp_l2'
    
    def forward(self, raw, img_size, labels=None):
        assert raw.shape[1] == raw.shape[2]
        # raw shape(BatchSize, 5+cls_num, FeatureSize, FeatureSize)
        device = raw.device
        nB = raw.shape[0] # batch size
        nG = raw.shape[1] # grid size, i.e., prediction resolution
        nCH = 5 + self.n_cls # number of channels for each object
        assert nG * self.stride == img_size

        raw = raw.view(nB, nCH, nG, nG)
        raw = raw.permute(0, 2, 3, 1).contiguous()
        # (l,t,r,b,conf,categories)
        # if self.ltrb_setting.startswith('relu'):
        #     # ReLU activation
        #     tnf.relu(raw[..., 0:4], inplace=True)

        # ----------------------- logits to prediction -----------------------
        preds = raw.detach().clone()
        # left, top, right, bottom
        if self.ltrb_setting.startswith('exp'):
            preds[...,0:4] = torch.exp(preds[...,0:4]) * self.stride
        # unnormalized
        preds[..., 0:4].clamp_(min=0, max=img_size)
        preds[..., 0:4] = _ltrb_to_xywh(preds[..., 0:4], nG, self.stride) # xywh
        # confidence
        preds[..., 4] = torch.sigmoid(preds[..., 4])
        # categories
        if self.n_cls > 0:
            preds[..., 5:] = torch.sigmoid(preds[..., 5:])
        preds = preds.view(nB, nG*nG, nCH).cpu()

        if labels is None:
            return preds, None

        assert isinstance(labels, list)
        # traverse all images in a batch
        valid_gt_num = 0
        gt_mask = torch.zeros(nB, nG, nG, dtype=torch.bool)
        # weighted = torch.zeros(nB, nA, nG, nG)
        target = torch.zeros(nB, nG, nG, nCH)
        for b in range(nB):
            num_gt = labels[b].shape[0]
            if num_gt == 0:
                # no ground truth
                continue
            assert labels[b].shape[1] == 5
            gt_xywh = labels[b][:,1:5]

            # determine this layer is responsible for which gt BBs
            # gt_areas = 




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
