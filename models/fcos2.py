import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf
# from fvcore.nn import smooth_l1_loss

from .backbones import get_backbone_fpn
from .rpns import get_rpn
import models.losses as lossLib
from utils.structures import ImageObjects


class FCOS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_class = cfg['num_class']

        self.backbone, self.fpn, fpn_info = get_backbone_fpn(cfg['backbone_fpn'])
        self.rpn = get_rpn(cfg['rpn'], fpn_info['feature_channels'], num_class)

        anchors = [0, 64, 128, 256, 512, 1e8]
        strides = fpn_info['feature_strides']
        self.bb_layer = nn.ModuleList()
        # anchors = [0, 64, 128, 256, 512, 1e8]
        for branch_i, s in enumerate(strides):
            amin = anchors[branch_i]
            amax = anchors[branch_i+1] if branch_i != len(strides)-1 else 1e8
            self.bb_layer.append(FCOSLayer(stride_all=strides, stride=s,
                                num_class=num_class, anchor_range=[amin,amax],
                                ltrb_setting=cfg['ltrb_setting']))

        self.input_format = 'BGR_255_norm'

    def forward(self, x, labels=None):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        assert x.dim() == 4
        self.img_size = x.shape[2:4]

        # go through the backbone and the feature payamid network
        features = self.backbone(x)
        features = self.fpn(features)
        all_branch_preds = self.rpn(features)

        dts_all = []
        losses_all = []
        for i, raw_preds in enumerate(all_branch_preds):
            dts, loss = self.bb_layer[i](raw_preds, self.img_size, labels)
            dts_all.append(dts)
            losses_all.append(loss)
        
        if labels is None:
            batch_bbs = torch.cat([d['bbox'] for d in dts_all], dim=1)
            batch_cls_idx = torch.cat([d['class_idx'] for d in dts_all], dim=1)
            batch_confs = torch.cat([d['conf'] for d in dts_all], dim=1)
            # bboxes = torch.cat(dts_all, dim=1)
            # bbs, confs, clss = bboxes[...,:4], bboxes[...,4], bboxes[...,5:]
            # cls_score, cls_idx = clss.max(dim=2, keepdim=False)
            # debug = boxes[boxes[...,5]>0.5]

            p_objects = []
            for bbs, cls_idx, confs in zip(batch_bbs, batch_cls_idx, batch_confs):
                p_objects.append(ImageObjects(bboxes=bbs, cats=cls_idx, scores=confs))
            return p_objects
        else:
            # check all the gt objects are assigned
            # assert isinstance(labels, list)
            # total_gt_num = sum([t.shape[0] for t in labels])
            # assigned_gt_num = sum(branch._assigned_num for branch in self.bb_layer)
            # assert assigned_gt_num == total_gt_num
            self.loss_str = ''
            for m in self.bb_layer:
                self.loss_str += m.loss_str + '\n'
            loss = sum(losses_all)
            return loss


class FCOSLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.anch_min = kwargs['anchor_range'][0]
        self.anch_max = kwargs['anchor_range'][1]
        self.strides_all = torch.Tensor(kwargs['stride_all'])
        self.stride = kwargs['stride']
        self.n_cls = kwargs['num_class']
        self.center_region = 0.5 # positive sample center region

        self.ltrb_setting = kwargs.get('ltrb_setting', 'exp_sl1')
        # self.ltrb_setting = 'exp_l2'
    
    def forward(self, raw, img_size, labels=None):
        assert isinstance(raw, dict)

        p_ltrb = raw['bbox']
        device = p_ltrb.device
        nB = p_ltrb.shape[0] # batch size
        nH, nW = p_ltrb.shape[1:3] # prediction grid size

        p_center_logits = raw['center']
        p_cls_logits = raw['class']
        
        if self.ltrb_setting.startswith('relu'):
            p_ltrb = tnf.relu(p_ltrb, inplace=True)

        # When testing, calculate and return the final predictions
        if labels is None:
            # ---------------------------- testing ----------------------------
            # activation function for left, top, right, bottom
            if self.ltrb_setting.startswith('exp'):
                p_ltrb = torch.exp(p_ltrb) * self.stride
            elif self.ltrb_setting.startswith('relu'):
                p_ltrb = p_ltrb * self.stride
            else:
                raise Exception('Unknown ltrb_setting')
            # Translate from ltrb to cxcywh
            p_ltrb = _ltrb_to_xywh(p_ltrb, nH, nW, self.stride)
            # Logistic activation for 'centerness'
            p_center_conf = torch.sigmoid(p_center_logits)
            # Logistic activation for categories
            p_cls = torch.sigmoid(p_cls_logits)
            cls_score, cls_idx = torch.max(p_cls, dim=3, keepdim=True)
            confs = torch.sqrt(p_center_conf * cls_score)
            preds = {
                'bbox': p_ltrb.view(nB, -1, 4),
                'class_idx': cls_idx.view(nB, -1),
                'conf': confs.view(nB, -1),
            }
            return preds, None

        # ------------------------------ training ------------------------------
        assert isinstance(labels, list)
        # Initialize the prediction target of the whole batch
        ValidGTnum = 0
        PenaltyMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        TargetLTRB = torch.zeros(nB, nH, nW, 4)
        TargetConf = torch.zeros(nB, nH, nW, 1)
        TargetCls = torch.zeros(nB, nH, nW, self.n_cls) if self.n_cls > 0 else None
        # traverse all images in a batch
        for b in range(nB):
            if labels[b].shape[0] == 0:
                # no ground truth
                continue
            assert labels[b].shape[1] == 5

            # Determine this layer is responsible for which gt BBs
            gt_ltrb_all = _xywh_to_ltrb(labels[b][:,1:5], self.stride)
            max_reg, _ = torch.max(gt_ltrb_all, dim=1)
            valid_mask = (max_reg > self.anch_min) & (max_reg <= self.anch_max)

            # Only use the gt BBs that this layer is responsible for
            gt_num = valid_mask.sum()
            ValidGTnum += gt_num
            valid_gts = labels[b][valid_mask,:]
            # Sort the gt BBs by their area in descending order
            areas = valid_gts[:,3] * valid_gts[:,4]
            aidx = torch.argsort(areas, descending=True)
            # Below are the labels used in this branch/feature level
            gt_cls = valid_gts[aidx, 0].long()
            gt_xywh = valid_gts[aidx, 1:5]

            # Determine which grid cell is responsible for which gt BB
            gt_masks, gt_ltrb, gt_conf = xywh2target(gt_xywh, (nH,nW), img_size,
                                                     self.center_region)
            # for i, mask in enumerate(gt_masks):
            #     print(f'{mask.sum()} cells is responsible for the current BB.')
            #     print(gt_xywh[i,:], img_size)
            #     import matplotlib.pyplot as plt
            #     plt.figure()
            #     plt.imshow(mask.numpy().astype('float32'), cmap='gray')
            #     plt.figure()
            #     plt.imshow(gt_conf[i].numpy().astype('float32'), cmap='gray')
            #     plt.show()
            
            # Since the gt labels are already sorted by area (descending), \
            # small objects are set later so they get higher priority
            for i, duty_mask in enumerate(gt_masks):
            # for duty_mask, ltrb_matrix, cls_ in zip(gt_masks, gt_ltrb, gt_cls):
                # Get non-zeros indices
                indices = duty_mask.nonzero()
                di1, di2 = indices[:,0], indices[:,1]
                # Calculate target ltrb for each location
                
                # Set regression target
                ltrb_ = gt_ltrb[i][duty_mask]
                if self.ltrb_setting.startswith('exp'):
                    TargetLTRB[b,di1,di2,:] = torch.log(ltrb_/self.stride + 1e-8)
                else:
                    raise NotImplementedError()
                TargetConf[b, di1, di2] = gt_conf[i][duty_mask].unsqueeze_(-1)
                if self.n_cls > 0:
                    TargetCls[b, di1, di2, gt_cls[i]] = 1
            # update the responsibility mask of the whole batch
            PenaltyMask[b] = (torch.sum(gt_masks, dim=0) > 0.5)

        # Transfer targets to GPU
        TargetLTRB = TargetLTRB.to(device=device)
        TargetConf = TargetConf.to(device=device)
        TargetCls = TargetCls.to(device=device) if self.n_cls > 0 else None

        # Compute loss
        pLTRB, gtLTRB = p_ltrb[PenaltyMask], TargetLTRB[PenaltyMask]
        if self.ltrb_setting.endswith('sl1'):
            # smooth L1 loss for l,t,r,b
            loss_bbox = 0.5 * lossLib.smooth_L1_loss(pLTRB, gtLTRB, beta=1,
                                                     reduction='sum')
            # loss_bbox = 0.5 * tnf.smooth_l1_loss(pLTRB, gtLTRB, reduction='sum')
            # loss_bbox = 0.5 * smooth_l1_loss(pLTRB, gtLTRB, 1, reduction='sum')
        elif self.ltrb_setting.endswith('l2'):
            # smooth L1 loss for l,t,r,b
            loss_bbox = 0.5 * tnf.mse_loss(pLTRB, gtLTRB, reduction='sum')
        # Binary cross entropy for confidence and classes
        bce_logits = tnf.binary_cross_entropy_with_logits
        loss_conf = bce_logits(p_center_logits, TargetConf, reduction='sum')
        if self.n_cls > 0:
            loss_cls = bce_logits(p_cls_logits[PenaltyMask],
                                  TargetCls[PenaltyMask], reduction='sum')
        else:
            loss_cls = 0
        loss = loss_bbox + loss_conf + loss_cls
        
        # logging
        ngt = ValidGTnum + 1e-16
        self.loss_str = f'level_{nH}x{nW} total {int(ngt)} objects: ' \
                        f'bbox/gt {loss_bbox/ngt:.3f}, conf {loss_conf:.3f}, ' \
                        f'class/gt {loss_cls/ngt:.3f}'
        self._assigned_num = ValidGTnum
        return None, loss


def xywh2target(xywh, mask_size, resolution=(1,1), center_region=0.5):
    '''
    Args:
        xywh: torch.tensor, rows of (cx,cy,w,h)
        mask_size: tuple, (h,w)
        resolution: the range of xywh. resolution=(1,1) means xywh is normalized
    '''
    assert xywh.dim() == 2 and xywh.shape[-1] >= 4
    if torch.rand(1) > 0.99:
        if (xywh <= 0).any() or (xywh >= max(resolution)).any():
            print('Warning: some xywh are out of range')
    device = xywh.device
    mh, mw = mask_size
    imgh, imgw = resolution

    def _xywh2xyxy(_xywh, cr):
        # boundaries
        shape = _xywh.shape[:-1]
        x1 = (_xywh[..., 0] - _xywh[..., 2] * cr / 2).view(*shape,1,1)
        y1 = (_xywh[..., 1] - _xywh[..., 3] * cr / 2).view(*shape,1,1)
        x2 = (_xywh[..., 0] + _xywh[..., 2] * cr / 2).view(*shape,1,1)
        y2 = (_xywh[..., 1] + _xywh[..., 3] * cr / 2).view(*shape,1,1)
        # create meshgrid
        return x1, y1, x2, y2
    x1, y1, x2, y2 = _xywh2xyxy(xywh, cr=center_region)
    x_ = torch.linspace(0,imgw,steps=mw+1, device=device)[:-1]
    y_ = torch.linspace(0,imgh,steps=mh+1, device=device)[:-1]
    gy, gx = torch.meshgrid(x_, y_)
    gx = gx.unsqueeze_(0) + imgw / (2*mw) # x meshgrid of size (1,mh,mw)
    gy = gy.unsqueeze_(0) + imgh / (2*mh) # y meshgrid of size (1,mh,mw)
    # build mask
    masks = (gx > x1) & (gx < x2) & (gy > y1) & (gy < y2)

    # calculate regression targets at each location
    x1, y1, x2, y2 = _xywh2xyxy(xywh, cr=1)
    t_l, t_t, t_r, t_b = gx-x1, gy-y1, x2-gx, y2-gy
    ltrb_target = torch.stack([t_l, t_t, t_r, t_b], dim=-1)
    ltrb_target *= masks.unsqueeze(-1)
    conf_target = torch.min(t_l, t_r) / torch.max(t_l, t_r) * \
                  torch.min(t_t, t_b) / torch.max(t_t, t_b)
    conf_target.mul_(masks).sqrt_()
    return masks, ltrb_target, conf_target


def _ltrb_to_xywh(ltrb, nH, nW, stride):
    '''
    transform (top,left,bottom,right) to (cx,cy,w,h)
    '''
    # training, (..., nH, nW, >= 4)
    assert ltrb.dim() >= 3 and ltrb.shape[-3] == nH and ltrb.shape[-2] == nW
    # if torch.rand(1) > 0.9: assert (ltrb[..., 0:4] <= nG).all()
    device = ltrb.device
    y_ = torch.arange(nH, dtype=torch.float, device=device) * stride
    x_ = torch.arange(nW, dtype=torch.float, device=device) * stride
    centers_y, centers_x = torch.meshgrid(y_, x_)
    for _ in range(ltrb.dim() - 3):
        centers_x.unsqueeze_(0)
        centers_y.unsqueeze_(0)
    centers_x = centers_x + stride / 2
    centers_y = centers_y + stride / 2

    xywh = torch.empty_like(ltrb)
    xywh[..., 0] = centers_x - (ltrb[...,0] - ltrb[...,2])/2 # cx
    xywh[..., 1] = centers_y - (ltrb[...,1] - ltrb[...,3])/2 # cy
    xywh[..., 2] = ltrb[...,0] + ltrb[...,2] # w
    xywh[..., 3] = ltrb[...,1] + ltrb[...,3] # h
    return xywh


def _xywh_to_ltrb(xywh, stride):
    '''
    transform (cx,cy,w,h) to (top,left,bottom,right).
    xywh should be unnormalized.
    '''
    assert (xywh > 0).all()
    xywh = xywh.clone() / stride # now in 0-nHW range
    centers_x = xywh[..., 0].floor() + 0.5
    centers_y = xywh[..., 1].floor() + 0.5

    ltrb = torch.empty_like(xywh)
    ltrb[..., 0] = centers_x - (xywh[..., 0] - xywh[..., 2]/2) # left
    ltrb[..., 1] = centers_y - (xywh[..., 1] - xywh[..., 3]/2) # top
    ltrb[..., 2] = xywh[..., 0] + xywh[..., 2]/2 - centers_x # right
    ltrb[..., 3] = xywh[..., 1] + xywh[..., 3]/2 - centers_y # bottom
    return ltrb * stride
