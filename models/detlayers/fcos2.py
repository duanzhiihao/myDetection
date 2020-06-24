import torch
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf
# from fvcore.nn import smooth_l1_loss

import models.losses as lossLib
from utils.structures import ImageObjects
from utils.bbox_ops import bboxes_iou


class FCOSLayer(torch.nn.Module):
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        self.anch_min = cfg['model.fcos.anchors'][level_i]
        self.anch_max = cfg['model.fcos.anchors'][level_i+1]
        self.stride   = cfg['model.fpn.out_strides'][level_i]
        self.n_cls    = cfg['general.num_class']

        self.center_region = 0.5 # positive sample center region
        self.ltrb_setting  = 'exp_sl1'
        self.ignore_thre   = cfg['model.fcos2.ignored_threshold']
        self.bb_format     = cfg['general.pred_bbox_format']

    def forward(self, raw, img_size, labels=None):
        stride = self.stride
        img_h, img_w = img_size
        nH, nW = int(img_h / stride), int(img_w / stride)
        nCls = self.n_cls
        assert isinstance(raw, dict)

        t_ltrb = raw['bbox']
        conf_logits = raw['conf']
        cls_logits = raw['class']
        nB = t_ltrb.shape[0] # batch size
        assert t_ltrb.shape == (nB, nH, nW, 4)
        assert conf_logits.shape == (nB, nH, nW, 1)
        assert cls_logits.shape == (nB, nH, nW, nCls)
        device = t_ltrb.device
        
        # activation function for left, top, right, bottom
        if self.ltrb_setting.startswith('exp'):
            p_ltrb = torch.exp(t_ltrb.detach()) * stride
        elif self.ltrb_setting.startswith('relu'):
            p_ltrb = tnf.relu(t_ltrb.detach()) * stride
        else:
            raise Exception('Unknown ltrb_setting')

        # ---------------------------- testing ----------------------------
        # Force the prediction to be in the image
        p_xyxy = _ltrb_to(p_ltrb, nH, nW, stride, 'x1y1x2y2')
        p_xyxy[..., 0].clamp_(min=0, max=img_w)
        p_xyxy[..., 1].clamp_(min=0, max=img_h)
        p_xyxy[..., 2].clamp_(min=0, max=img_w)
        p_xyxy[..., 3].clamp_(min=0, max=img_h)
        p_xywh = _xyxy_to_xywh(p_xyxy)
        # Logistic activation for 'centerness'
        p_conf = torch.sigmoid(conf_logits.detach())
        # Logistic activation for categories
        p_cls = torch.sigmoid(cls_logits.detach())
        cls_score, cls_idx = torch.max(p_cls, dim=3, keepdim=True)
        confs = torch.sqrt(p_conf * cls_score)
        preds = {
            'bbox': p_xywh.view(nB, nH*nW, 4),
            'class_idx': cls_idx.view(nB, nH*nW),
            'score': confs.view(nB, nH*nW),
        }
        # Return the final predictions when testing
        if labels is None:
            return preds, None

        p_xywh = _ltrb_to(p_ltrb, nH, nW, stride, 'cxcywh').cpu()
        # ------------------------------ training ------------------------------
        assert isinstance(labels, list)
        # Build x,y meshgrid with size (1,nH,nW)
        x_ = torch.linspace(0, img_w, steps=nW+1)[:-1] + 0.5 * stride
        y_ = torch.linspace(0, img_h, steps=nH+1)[:-1] + 0.5 * stride
        gy, gx = torch.meshgrid(y_, x_)
        # Initialize the prediction target of the batch
        # positive: at the center region and max(tgt_ltrb) in (min, max)
        # ignored: predicted bbox IoU with GT > 0.6
        # Conf: positive or (not ignored)
        # LTRB, Ctr, CLs: positive
        PositiveMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        IgnoredMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        TargetConf = torch.zeros(nB, nH, nW, 1)
        TargetLTRB = torch.zeros(nB, nH, nW, 4)
        # TargetCtr = torch.zeros(nB, nH, nW, 1)
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

            # ignore the conf of a pred BB if it matches a gt more than self.thres
            ious = bboxes_iou(p_xywh[b].view(-1,4), gt_xywh, xyxy=False)
            iou_with_gt, _ = torch.max(ious, dim=1)
            IgnoredMask[b] = (iou_with_gt > self.ignore_thre).view(nH, nW)

            # Since the gt labels are sorted by area (descending), \
            # small object targets are set later so they get higher priority
            for bb, cidx in zip(gt_xywh, gt_cls_idx):
                # Convert cxcywh to x1y1x2y2
                Tx1, Ty1, Tx2, Ty2 = _xywh_to_xyxy(bb, cr=1)
                # regression target at each location
                tgt_l, tgt_t, tgt_r, tgt_b = gx-Tx1, gy-Ty1, Tx2-gx, Ty2-gy
                # stacking them together, we get target for ltrb
                tgt_ltrb = torch.stack([tgt_l, tgt_t, tgt_r, tgt_b], dim=-1)
                assert tgt_ltrb.shape == (nH, nW, 4)
                # full bounding box mask
                bbox_mask = torch.prod((tgt_ltrb > 0), dim=-1).bool()
                # Find positive samples for this bounding box
                # 1. the center part of the bounding box
                Cx1, Cy1, Cx2, Cy2 = _xywh_to_xyxy(bb, cr=self.center_region)
                center_mask = (gx > Cx1) & (gx < Cx2) & (gy > Cy1) & (gy < Cy2)
                # 2. max predicted ltrb within the range
                max_tgt_ltrb, _ = torch.max(tgt_ltrb, dim=-1)
                anch_mask = (self.anch_min < max_tgt_ltrb) & (max_tgt_ltrb < self.anch_max)
                # 3. positive samples must satisfy both 1 and 2
                pos_mask = center_mask & anch_mask
                if not pos_mask.any():
                    continue
                # set target for ltrb
                TargetLTRB[b, pos_mask, :] = tgt_ltrb[pos_mask, :]
                # compute target for center score
                # tgt_center = torch.min(tgt_l, tgt_r) / torch.max(tgt_l, tgt_r) * \
                #             torch.min(tgt_t, tgt_b) / torch.max(tgt_t, tgt_b)
                # tgt_center.mul_(bbox_mask).sqrt_()
                # import matplotlib.pyplot as plt
                # plt.imshow(tgt_center.numpy(), cmap='gray'); plt.show()
                # assert TargetCtr.shape[-1] == 1
                # TargetCtr[b, bbox_mask] = tgt_center[bbox_mask].unsqueeze(-1)
                # the target for confidence socre is 1
                TargetConf[b, pos_mask] = 1
                # set target for category classification
                _Hidx, _Widx = pos_mask.nonzero(as_tuple=True)
                TargetCls[b, _Hidx, _Widx, cidx] = 1
                # Update the batch positive sample mask
                PositiveMask[b] = PositiveMask[b] | pos_mask

        # Transfer targets to GPU
        PositiveMask = PositiveMask.to(device=device)
        IgnoredMask = IgnoredMask.to(device=device)
        TargetConf = TargetConf.to(device=device)
        TargetLTRB = TargetLTRB.to(device=device)
        # TargetCtr = TargetCtr.to(device=device)
        TargetCls = TargetCls.to(device=device) if self.n_cls > 0 else None

        # Compute loss
        pLTRB, tgtLTRB = t_ltrb[PositiveMask], TargetLTRB[PositiveMask]
        assert (tgtLTRB > 0).all() # Sanity check
        if self.ltrb_setting.startswith('exp'):
            tgtLTRB = torch.log(tgtLTRB / stride)
        else: raise NotImplementedError()
        if self.ltrb_setting.endswith('sl1'):
            # smooth L1 loss for l,t,r,b
            loss_bbox = lossLib.smooth_L1_loss(pLTRB, tgtLTRB, beta=0.2,
                                               reduction='sum')
        elif self.ltrb_setting.endswith('l2'):
            loss_bbox = tnf.mse_loss(pLTRB, tgtLTRB, reduction='sum')
        else: raise NotImplementedError()
        bce_logits = tnf.binary_cross_entropy_with_logits
        # Binary cross entropy for confidence score
        _penalty = PositiveMask | (~IgnoredMask)
        _pConf, _tgtConf = conf_logits[_penalty], TargetConf[_penalty]
        loss_conf = bce_logits(_pConf, _tgtConf, reduction='sum')
        # Binary cross entropy for category classification
        _pCls, _tgtCls = cls_logits[PositiveMask], TargetCls[PositiveMask]
        loss_cls = bce_logits(_pCls, _tgtCls, reduction='sum')
        loss = loss_bbox + loss_conf + loss_cls
        loss = loss / nB

        # logging
        pos_num = PositiveMask.sum().cpu().item()
        total_sample_num = nB * nH * nW
        ignored_num = (IgnoredMask & (~PositiveMask)).sum().cpu().item()
        self.loss_str = f'level_{nH}x{nW}, pos {pos_num}/{total_sample_num}, ' \
                        f'ignored {ignored_num}/{total_sample_num}: ' \
                        f'bbox/gt {loss_bbox:.3f}, conf {loss_conf:.3f}, ' \
                        f'class/gt {loss_cls:.3f}'
        return preds, loss


class FCOS_ATSS_Layer(torch.nn.Module):
    def __init__(self, level_i: int, cfg: dict):
        super().__init__()
        self.strides_all = cfg['model.fpn.out_strides']
        self.stride = cfg['model.fpn.out_strides'][level_i]
        self.n_cls = cfg['general.num_class']

        self.anchors_all = cfg['model.atss.anchors']
        self.anchor = self.anchors_all[level_i]
        self.topk = cfg['model.atss.topk_per_level']
        self.ltrb_setting = 'exp_sl1'
        self.ignore_thre = cfg['model.fcos2.ignored_threshold']
    
    def forward(self, raw, img_size, labels=None):
        stride = self.stride
        img_h, img_w = img_size
        nH, nW = int(img_h / stride), int(img_w / stride)
        nCls = self.n_cls
        assert isinstance(raw, dict)

        t_ltrb = raw['bbox']
        conf_logits = raw['conf']
        cls_logits = raw['class']
        nB = t_ltrb.shape[0] # batch size
        assert t_ltrb.shape == (nB, nH, nW, 4)
        assert conf_logits.shape == (nB, nH, nW, 1)
        assert cls_logits.shape == (nB, nH, nW, nCls)
        device = t_ltrb.device
        
        # activation function for left, top, right, bottom
        if self.ltrb_setting.startswith('exp'):
            p_ltrb = torch.exp(t_ltrb.detach()) * stride
        elif self.ltrb_setting.startswith('relu'):
            p_ltrb = tnf.relu(t_ltrb.detach()) * stride
        else:
            raise Exception('Unknown ltrb_setting')

        # ---------------------------- testing ----------------------------
        # Force the prediction to be in the image
        p_xyxy = _ltrb_to(p_ltrb, nH, nW, stride, 'x1y1x2y2')
        p_xyxy[..., 0].clamp_(min=0, max=img_w)
        p_xyxy[..., 1].clamp_(min=0, max=img_h)
        p_xyxy[..., 2].clamp_(min=0, max=img_w)
        p_xyxy[..., 3].clamp_(min=0, max=img_h)
        p_xywh = _xyxy_to_xywh(p_xyxy)
        # Logistic activation for 'centerness'
        p_conf = torch.sigmoid(conf_logits.detach())
        # Logistic activation for categories
        p_cls = torch.sigmoid(cls_logits.detach())
        cls_score, cls_idx = torch.max(p_cls, dim=3, keepdim=True)
        confs = torch.sqrt(p_conf * cls_score)
        preds = {
            'bbox': p_xywh.view(nB, nH*nW, 4),
            'class_idx': cls_idx.view(nB, nH*nW),
            'score': confs.view(nB, nH*nW),
        }
        # Return the final predictions when testing
        if labels is None:
            return preds, None

        p_xywh = _ltrb_to(p_ltrb, nH, nW, stride, 'cxcywh').cpu()
        # ------------------------------ training ------------------------------
        assert isinstance(labels, list)
        # Build x,y meshgrid with size (1,nH,nW)
        x_ = torch.linspace(0, img_w, steps=nW+1)[:-1] + 0.5 * stride
        y_ = torch.linspace(0, img_h, steps=nH+1)[:-1] + 0.5 * stride
        gy, gx = torch.meshgrid(y_, x_)
        gy, gx = gy.contiguous(), gx.contiguous()
        # Build x,y meshgrid for all levels
        # Calculating this at each level is not very efficient
        # Ideally this should be done only once
        # But to achieve that, code structure must be changed.
        all_anchor_bbs = []
        for li, s in enumerate(self.strides_all):
            assert img_w % s == 0 and img_h % s == 0
            _sdH, _sdW = img_h // s, img_w // s
            _x = torch.linspace(0, img_w, steps=_sdW+1)[:-1] + 0.5 * s
            _y = torch.linspace(0, img_h, steps=_sdH+1)[:-1] + 0.5 * s
            _gy, _gx = torch.meshgrid(_y, _x)
            # if s == stride:
            #     assert (_gy == gy).all() and (_gx == gx).all()
            assert _gy.shape == _gx.shape == (_sdH, _sdW)
            anch_wh = torch.ones(_sdH*_sdW, 2) * self.anchors_all[li]
            anch_bbs = torch.cat(
                [_gx.reshape(-1,1), _gy.reshape(-1,1), anch_wh], dim=1)
            all_anchor_bbs.append(anch_bbs)
        # Initialize the prediction target of the batch
        # positive: at the center region and max(tgt_ltrb) in (min, max)
        # ignored: predicted bbox IoU with GT > 0.6
        # Conf: positive or (not ignored)
        # LTRB, Ctr, CLs: positive
        PositiveMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        IgnoredMask = torch.zeros(nB, nH, nW, dtype=torch.bool)
        TargetConf = torch.zeros(nB, nH, nW, 1)
        TargetLTRB = torch.zeros(nB, nH, nW, 4)
        # TargetCtr = torch.zeros(nB, nH, nW, 1)
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

            # ignore the conf of a pred BB if it matches a gt more than 0.7
            ious = bboxes_iou(p_xywh[b].view(-1,4), gt_xywh, xyxy=False)
            iou_with_gt, _ = torch.max(ious, dim=1)
            IgnoredMask[b] = (iou_with_gt > self.ignore_thre).view(nH, nW)

            # Since the gt labels are sorted by area (descending), \
            # small object targets are set later so they get higher priority
            for bb, cidx in zip(gt_xywh, gt_cls_idx):
                # Convert cxcywh to x1y1x2y2
                Tx1, Ty1, Tx2, Ty2 = _xywh_to_xyxy(bb, cr=1)
                # regression target at each location
                tgt_l, tgt_t, tgt_r, tgt_b = gx-Tx1, gy-Ty1, Tx2-gx, Ty2-gy
                # stacking them together, we get target for ltrb
                tgt_ltrb = torch.stack([tgt_l, tgt_t, tgt_r, tgt_b], dim=-1)
                assert tgt_ltrb.shape == (nH, nW, 4)
                # full bounding box mask
                bbox_mask = torch.prod((tgt_ltrb > 0), dim=-1).bool()
                # Find positive samples for this bounding box
                thres = _get_atss_threshold(bb, all_anchor_bbs, self.topk)
                anch_bbs = torch.stack([gx, gy], dim=-1)
                anch_bbs = torch.cat(
                    [anch_bbs, torch.ones(nH,nW,2)*self.anchor], dim=-1
                ).view(nH*nW, 4)
                ious = bboxes_iou(anch_bbs, bb.view(1,4), xyxy=False).squeeze()
                pos_mask = (ious > thres).view(nH, nW)
                pos_mask = pos_mask & bbox_mask
                if not pos_mask.any():
                    continue
                # set target for ltrb
                TargetLTRB[b, pos_mask, :] = tgt_ltrb[pos_mask, :]
                # the target for confidence socre is 1
                TargetConf[b, pos_mask] = 1
                # set target for category classification
                _Hidx, _Widx = pos_mask.nonzero(as_tuple=True)
                TargetCls[b, _Hidx, _Widx, cidx] = 1
                # Update the batch positive sample mask
                PositiveMask[b] = PositiveMask[b] | pos_mask

        # Transfer targets to GPU
        PositiveMask = PositiveMask.to(device=device)
        IgnoredMask = IgnoredMask.to(device=device)
        TargetConf = TargetConf.to(device=device)
        TargetLTRB = TargetLTRB.to(device=device)
        # TargetCtr = TargetCtr.to(device=device)
        TargetCls = TargetCls.to(device=device) if self.n_cls > 0 else None

        # Compute loss
        pLTRB, tgtLTRB = t_ltrb[PositiveMask], TargetLTRB[PositiveMask]
        assert (tgtLTRB > 0).all() # Sanity check
        if self.ltrb_setting.startswith('exp'):
            tgtLTRB = torch.log(tgtLTRB / stride)
        else: raise NotImplementedError()
        if self.ltrb_setting.endswith('sl1'):
            # smooth L1 loss for l,t,r,b
            loss_bbox = lossLib.smooth_L1_loss(pLTRB, tgtLTRB, beta=0.2,
                                               reduction='sum')
        elif self.ltrb_setting.endswith('l2'):
            loss_bbox = tnf.mse_loss(pLTRB, tgtLTRB, reduction='sum')
        else: raise NotImplementedError()
        bce_logits = tnf.binary_cross_entropy_with_logits
        # Binary cross entropy for confidence score
        _penalty = PositiveMask | (~IgnoredMask)
        _pConf, _tgtConf = conf_logits[_penalty], TargetConf[_penalty]
        loss_conf = bce_logits(_pConf, _tgtConf, reduction='sum')
        # Binary cross entropy for category classification
        _pCls, _tgtCls = cls_logits[PositiveMask], TargetCls[PositiveMask]
        loss_cls = bce_logits(_pCls, _tgtCls, reduction='sum')
        loss = loss_bbox + loss_conf + loss_cls
        
        # logging
        pos_num = PositiveMask.sum().cpu().item()
        total_sample_num = nB * nH * nW
        ignored_num = (IgnoredMask & (~PositiveMask)).sum().cpu().item()
        self.loss_str = f'level_{nH}x{nW}, pos {pos_num}/{total_sample_num}, ' \
                        f'ignored {ignored_num}/{total_sample_num}: ' \
                        f'bbox/gt {loss_bbox:.3f}, conf {loss_conf:.3f}, ' \
                        f'class/gt {loss_cls:.3f}'
        return preds, loss


def _get_atss_threshold(gtbb, all_anchors, k):
    '''
    1. Choose the nearest k anchors from each level based on the L2 distance
    2. Put them together, calculate the IoU with ground truth
    3. Adaptive threshold = mean + standard deviation
    '''
    assert isinstance(all_anchors, list) and isinstance(k, int)
    cx, cy, w, h = gtbb
    all_candidates = []
    for anchors in all_anchors:
        l2_2 = (cx - anchors[:,0]).pow(2) + (cy - anchors[:,1]).pow(2)
        _, idxs = torch.topk(l2_2, k, largest=False, sorted=False)
        all_candidates.append(anchors[idxs,:])
    all_candidates = torch.cat(all_candidates, dim=0)
    assert all_candidates.shape == (len(all_anchors)*k, 4)
    ious = bboxes_iou(gtbb.view(1,4), all_candidates, xyxy=False).squeeze()
    assert ious.dim() == 1
    mean, std = ious.mean(), ious.std()
    atss_thres = mean + std
    # debug = ious > atss_thres
    return atss_thres


def _xywh_to_xyxy(_xywh, cr: float):
    cx, cy, w, h = _xywh
    x1 = cx - w * cr / 2
    y1 = cy - h * cr / 2
    x2 = cx + w * cr / 2
    y2 = cy + h * cr / 2
    return (x1, y1, x2, y2)


def _xyxy_to_xywh(_xyxy):
    x1, y1, x2, y2 = _xyxy[...,0], _xyxy[...,1], _xyxy[...,2], _xyxy[...,3]
    _xywh = torch.empty_like(_xyxy)
    _xywh[..., 0] = (x1 + x2) / 2 # cx
    _xywh[..., 1] = (y1 + y2) / 2 # cy
    _xywh[..., 2] = x2 - x1 # w
    _xywh[..., 3] = y2 - y1 # h
    return _xywh


def _ltrb_to(ltrb, nH, nW, stride, out_format):
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

    if out_format == 'cxcywh':
        xywh = torch.empty_like(ltrb)
        xywh[..., 0] = x_ + (ltrb[..., 2] - ltrb[..., 0]) / 2 # cx
        xywh[..., 1] = y_ + (ltrb[..., 3] - ltrb[..., 1]) / 2 # cy
        xywh[..., 2] = ltrb[...,0] + ltrb[...,2] # w
        xywh[..., 3] = ltrb[...,1] + ltrb[...,3] # h
        return xywh
    elif out_format == 'x1y1x2y2':
        xyxy = torch.empty_like(ltrb)
        xyxy[..., 0] = x_ - ltrb[..., 0]
        xyxy[..., 1] = y_ - ltrb[..., 1]
        xyxy[..., 2] = x_ + ltrb[..., 2]
        xyxy[..., 3] = y_ + ltrb[..., 3]
        return xyxy
    else:
        raise Exception('Umknown bounding box format')


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
