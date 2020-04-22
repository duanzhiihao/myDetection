import torch
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf
# from fvcore.nn import smooth_l1_loss

import models.losses as lossLib
from utils.structures import ImageObjects


class FCOSLayer(torch.nn.Module):
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        self.anch_min = cfg['model.fcos.anchors'][level_i]
        self.anch_max = cfg['model.fcos.anchors'][level_i+1]
        self.stride = cfg['model.fpn.out_strides'][level_i]
        self.n_cls = cfg['general.num_class']

        self.center_region = 0.5 # positive sample center region
        self.ltrb_setting = 'exp_sl1' # TODO
    
    def forward(self, raw, img_size, labels=None):
        stride = self.stride
        img_h, img_w = img_size
        nH, nW = int(img_h / stride), int(img_w / stride)
        nCls = self.n_cls
        assert isinstance(raw, dict)

        t_ltrb = raw['bbox']
        center_logits = raw['center']
        cls_logits = raw['class']
        nB = t_ltrb.shape[0] # batch size
        assert t_ltrb.shape == (nB, nH, nW, 4)
        assert center_logits.shape == (nB, nH, nW, 1)
        assert cls_logits.shape == (nB, nH, nW, nCls)
        device = t_ltrb.device
        
        if self.ltrb_setting.startswith('relu'):
            t_ltrb = tnf.relu(t_ltrb, inplace=True)

        # When testing, calculate and return the final predictions
        if labels is None:
            # ---------------------------- testing ----------------------------
            # activation function for left, top, right, bottom
            if self.ltrb_setting.startswith('exp'):
                p_ltrb = torch.exp(t_ltrb) * stride
            elif self.ltrb_setting.startswith('relu'):
                p_ltrb = t_ltrb * stride
            else:
                raise Exception('Unknown ltrb_setting')
            # Translate from ltrb to cxcywh
            p_ltrb = _ltrb_to_xywh(p_ltrb, nH, nW, stride)
            # Logistic activation for 'centerness'
            p_center = torch.sigmoid(center_logits)
            # Logistic activation for categories
            p_cls = torch.sigmoid(cls_logits)
            cls_score, cls_idx = torch.max(p_cls, dim=3, keepdim=True)
            confs = torch.sqrt(p_center * cls_score)
            preds = {
                'bbox': p_ltrb.view(nB, nH*nW, 4),
                'class_idx': cls_idx.view(nB, nH*nW),
                'score': confs.view(nB, nH*nW),
            }
            return preds, None

        # ------------------------------ training ------------------------------
        assert isinstance(labels, list)
        # Build x,y meshgrid with size (1,nH,nW)
        x_ = torch.linspace(0, img_w, steps=nW+1)[:-1] + 0.5 * stride
        y_ = torch.linspace(0, img_h, steps=nH+1)[:-1] + 0.5 * stride
        gy, gx = torch.meshgrid(y_, x_)
        # Initialize the prediction target of the batch
        # PenaltyMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        PositiveMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        # TODO IgnoredMask
        TargetLTRB = torch.zeros(nB, nH, nW, 4)
        TargetCtr = torch.zeros(nB, nH, nW, 1)
        TargetCls = torch.zeros(nB, nH, nW, self.n_cls) if self.n_cls > 0 else None
        assert self.n_cls > 0
        # traverse all images in a batch
        for b in range(nB):
            im_labels = labels[b]
            assert isinstance(im_labels, ImageObjects)
            if len(im_labels) == 0:
                continue
            im_labels.sanity_check()

            gt_xywh = im_labels.bboxes
            areas = gt_xywh[:,2] * gt_xywh[:,3]
            lg2sml_idx = torch.argsort(areas, descending=True)
            gt_xywh = gt_xywh[lg2sml_idx, :]
            gt_cls_idx = im_labels.cats[lg2sml_idx]

            # Since the gt labels are sorted by area (descending), \
            # small object targets are set later so they get higher priority
            for bb, cidx in zip(gt_xywh, gt_cls_idx):
                # Convert cxcywh to x1y1x2y2
                Tx1, Ty1, Tx2, Ty2 = _xywh_to_xyxy(bb, cr=1)
                # regression target at each location
                tgt_l, tgt_t, tgt_r, tgt_b = gx-Tx1, gy-Ty1, Tx2-gx, Ty2-gy
                # stacking them together, we get target for xywh
                tgt_ltrb = torch.stack([tgt_l, tgt_t, tgt_r, tgt_b], dim=-1)
                assert tgt_ltrb.shape == (nH, nW, 4)
                # full bounding box mask
                bbox_mask = torch.prod((tgt_ltrb > 0), dim=-1).bool()
                # Find positive samples for this bounding box
                # 1. sample the center part of the bounding box
                Cx1, Cy1, Cx2, Cy2 = _xywh_to_xyxy(bb, cr=self.center_region)
                center_mask = (gx > Cx1) & (gx < Cx2) & (gy > Cy1) & (gy < Cy2)
                anch_mask = (self.anch_min < tgt_ltrb) & (tgt_ltrb < self.anch_max)
                anch_mask = torch.prod(anch_mask, dim=-1).bool()
                pos_mask = center_mask & anch_mask
                if not pos_mask.any():
                    continue
                # import matplotlib.pyplot as plt
                # plt.imshow(mask.numpy(), cmap='gray'); plt.show()
                TargetLTRB[b, pos_mask, :] = tgt_ltrb[pos_mask, :]
                # compute target for center score
                tgt_center = torch.min(tgt_l, tgt_r) / torch.max(tgt_l, tgt_r) * \
                            torch.min(tgt_t, tgt_b) / torch.max(tgt_t, tgt_b)
                tgt_center.mul_(bbox_mask).sqrt_()
                # import matplotlib.pyplot as plt
                # plt.imshow(tgt_center.numpy(), cmap='gray'); plt.show()
                TargetCtr[b, bbox_mask] = tgt_center[bbox_mask]
                # set target for category classification
                TargetCls[b, pos_mask, cidx] = 1
                # Update the batch positive sample mask
                PositiveMask[b] = PositiveMask[b] | pos_mask


            # Determine this layer is responsible for which gt BBs
            # gt_ltrb_all = _xywh_to_ltrb(labels[b][:,1:5], stride)
            # max_reg, _ = torch.max(gt_ltrb_all, dim=1)
            # valid_mask = (max_reg > self.anch_min) & (max_reg <= self.anch_max)

            # Only use the gt BBs that this layer is responsible for
            # gt_num = valid_mask.sum()
            # ValidGTnum += gt_num
            # valid_gts = labels[b][valid_mask,:]
            # Sort the gt BBs by their area in descending order
            # areas = valid_gts[:,3] * valid_gts[:,4]
            # aidx = torch.argsort(areas, descending=True)
            # Below are the labels used in this branch/feature level
            # gt_cls = valid_gts[aidx, 0].long()
            # gt_xywh = valid_gts[aidx, 1:5]

            # Determine which grid cell is responsible for which gt BB
            # gt_xywh = im_labels.bboxes
            # gt_masks, gt_ltrb, gt_conf = xywh2target(gt_xywh, (nH,nW), img_size,
            #                                          self.center_region)
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
            # for i, duty_mask in enumerate(gt_masks):
            # # for duty_mask, ltrb_matrix, cls_ in zip(gt_masks, gt_ltrb, gt_cls):
            #     # Get non-zeros indices
            #     indices = duty_mask.nonzero()
            #     di1, di2 = indices[:,0], indices[:,1]
            #     # Calculate target ltrb for each location
                
            #     # Set regression target
            #     ltrb_ = gt_ltrb[i][duty_mask]
            #     if self.ltrb_setting.startswith('exp'):
            #         TargetLTRB[b,di1,di2,:] = torch.log(ltrb_/stride + 1e-8)
            #     else:
            #         raise NotImplementedError()
            #     TargetCtr[b, di1, di2] = gt_conf[i][duty_mask].unsqueeze_(-1)
            #     if self.n_cls > 0:
            #         TargetCls[b, di1, di2, gt_cls[i]] = 1
            # # update the responsibility mask of the whole batch
            # PenaltyMask[b] = (torch.sum(gt_masks, dim=0) > 0.5)

        # Transfer targets to GPU
        PositiveMask = PositiveMask.to(device=device)
        TargetLTRB = TargetLTRB.to(device=device)
        TargetCtr = TargetCtr.to(device=device)
        TargetCls = TargetCls.to(device=device) if self.n_cls > 0 else None

        # Compute loss
        pLTRB, tgtLTRB = t_ltrb[PositiveMask], TargetLTRB[PositiveMask]
        assert (tgtLTRB > 0).all() # Sanity check
        if self.ltrb_setting.startswith('exp'):
            tgtLTRB = torch.log(tgtLTRB/stride + 1e-8)
        if self.ltrb_setting.endswith('sl1'):
            # smooth L1 loss for l,t,r,b
            loss_bbox = lossLib.smooth_L1_loss(pLTRB, tgtLTRB, beta=1,
                            weight=TargetCtr[PositiveMask], reduction='sum')
            # loss_bbox = tnf.smooth_l1_loss(pLTRB, tgtLTRB, reduction='mean')
            # loss_bbox = 0.5 * smooth_l1_loss(pLTRB, gtLTRB, 1, reduction='sum')
        elif self.ltrb_setting.endswith('l2'):
            loss_bbox = tnf.mse_loss(pLTRB, tgtLTRB, reduction='mean')
        else: raise NotImplementedError()
        # Binary cross entropy for center score and classes
        bce_logits = tnf.binary_cross_entropy_with_logits
        loss_center = bce_logits(center_logits, TargetCtr, reduction='mean')
        loss_cls = bce_logits(cls_logits, TargetCls, reduction='sum')
        loss = loss_bbox + loss_center + loss_cls / (PositiveMask.sum() + 1)
        
        # logging
        pos_num = PositiveMask.sum().cpu().item()
        total_sample_num = nB * nH * nW
        self.loss_str = f'level_{nH}x{nW} pos {pos_num}/{total_sample_num}: ' \
                        f'bbox/gt {loss_bbox:.3f}, center {loss_center:.3f}, ' \
                        f'class/gt {loss_cls:.3f}'
        return None, loss


def xywh2target(xywh, mask_size, resolution, center_region=0.5):
    '''
    Args:
        xywh: torch.tensor, rows of (cx,cy,w,h)
        mask_size: tuple, (h,w)
        resolution: the range of xywh. resolution=(1,1) means xywh is normalized
    '''
    assert xywh.dim() == 2 and xywh.shape[-1] == 4
    if torch.rand(1) > 0.99:
        if (xywh <= 0).any() or (xywh >= max(resolution)).any():
            print('Warning: some xywh are out of range')
    device = xywh.device
    mh, mw = mask_size
    imgh, imgw = resolution

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
    center_target = torch.min(t_l, t_r) / torch.max(t_l, t_r) * \
                    torch.min(t_t, t_b) / torch.max(t_t, t_b)
    center_target.mul_(masks).sqrt_()
    return masks, ltrb_target, center_target


def _xywh_to_xyxy(_xywh, cr: float):
    cx, cy, w, h = _xywh
    x1 = cx - w * cr / 2
    y1 = cy - h * cr / 2
    x2 = cx + w * cr / 2
    y2 = cy + h * cr / 2
    return (x1, y1, x2, y2)


def _xywh2xyxy(_xywh: torch.tensor, cr: float):
    '''
    cr: center region
    '''
    raise Exception()
    # boundaries
    shape = _xywh.shape[:-1]
    x1 = (_xywh[..., 0] - _xywh[..., 2] * cr / 2).view(*shape,1,1)
    y1 = (_xywh[..., 1] - _xywh[..., 3] * cr / 2).view(*shape,1,1)
    x2 = (_xywh[..., 0] + _xywh[..., 2] * cr / 2).view(*shape,1,1)
    y2 = (_xywh[..., 1] + _xywh[..., 3] * cr / 2).view(*shape,1,1)
    # create meshgrid
    return x1, y1, x2, y2


def _ltrb_to_xywh(ltrb, nH, nW, stride):
    '''
    transform (top,left,bottom,right) to (cx,cy,w,h)
    '''
    # training, (..., nH, nW, 4)
    assert ltrb.shape[-3] == nH and ltrb.shape[-2] == nW and ltrb.shape[-1] == 4
    # if torch.rand(1) > 0.9: assert (ltrb[..., 0:4] <= nG).all()
    device = ltrb.device
    y_ = torch.arange(nH, dtype=torch.float, device=device).view(nH, 1)
    x_ = torch.arange(nW, dtype=torch.float, device=device).view(1, nW)
    for _ in range(ltrb.dim() - 3):
        y_.unsqueeze_(0)
        x_.unsqueeze_(0)
    y_ = y_ * stride + stride / 2
    x_ = x_ * stride + stride / 2

    xywh = torch.empty_like(ltrb)
    xywh[..., 0] = x_ + (ltrb[..., 2] - ltrb[..., 0]) / 2 # cx
    xywh[..., 1] = y_ + (ltrb[..., 3] - ltrb[..., 1]) / 2 # cy
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
